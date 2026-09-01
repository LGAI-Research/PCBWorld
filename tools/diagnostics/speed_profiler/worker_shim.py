"""Forkserver-preload shim — inject worker-side timing with ZERO base edits.

Mechanism (G-2 spike, 2026-07-09): ``mp.set_forkserver_preload(["...worker_shim"])``
BEFORE the first forkserver worker spawns makes the forkserver *server* import this
module; on import it rebinds ``pcb_world.vec.backends.subproc._decoder_worker`` to a
timed wrapper. Every worker forked from the server inherits the patched target
(the pool passes ``target=_decoder_worker`` by reference, resolved to this attr in
the server). Confirmed: ``patched_marker=True`` in the child.

The wrapper stamps the worker's own compute duration ``recv->ready`` into
``info["_prof"]["worker_compute_s"]`` for ``step`` commands (metadata only —
observation/tuple arity unchanged, so numerics/RNG are untouched). This separates
true worker compute from the main-serial unpickle and the straggler ordering in
the H1 barrier decomposition, WITHOUT cross-process clock subtraction.

Keep this module light — do NOT import torch (the server would then carry a CUDA
context into every worker). It only touches the env infra (subproc / traceback).

Gated on ``CADAGENT_PROFILE_WORKER``; a no-op import when unset (so the module is
harmless even if accidentally preloaded).
"""
from __future__ import annotations

import os
import time

_PATCHED = False


def _timed_decoder_worker(remote, parent_remote, env_fn_wrapper, board_factory_wrapper=None,
                          role="env"):
    """Timed mirror of ``pcb_world.vec.backends.subproc._decoder_worker``
    (base @ 2026-08-28 ipc(P2): die on ``EngineServerCrashed`` so the parent's
    dead-worker respawn fires; prior 66743b05f ``env = None`` reload fix; incl.
    pcb_world.diag crash instrumentation). Identical control flow; the ONLY
    change is stamping ``info["_prof"]["worker_compute_s"]`` on the ``step``
    reply. Re-sync if the base worker loop changes.
    """
    import gc
    import traceback

    from pcb_world import diag
    from pcb_world.engine.router_client import EngineServerCrashed

    PC = time.perf_counter
    parent_remote.close()
    wd = diag.WorkerDiag(role)

    env = env_fn_wrapper.var()
    board_factory = board_factory_wrapper.var if board_factory_wrapper is not None else None

    while True:
        cmd = data = None
        try:
            cmd, data = remote.recv()
            t_cmd = PC()
            if cmd == "reload_board":
                if board_factory is None:
                    raise RuntimeError("reload_board called but worker has no board_factory")
                board_path, reload_seq = data
                # ``env = None`` (not ``del``) — base fix 66743b05f: on a
                # board_factory raise the except-handler below still reads
                # ``env``; with ``del`` that was an UnboundLocalError masking
                # the real exception.
                env.close(); env = None; gc.collect()
                env = board_factory(board_path, reload_seq)
                remote.send(True)
                continue
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                wd.tick()
                # worker compute = recv -> ready (route+DRC+obs build), before pickle/transfer.
                if isinstance(info, dict):
                    info = {**info, "_prof": {"worker_compute_s": PC() - t_cmd}}
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                # data = seed | None; None keeps the historical no-arg call.
                obs, info = env.reset(seed=data) if data is not None else env.reset()
                remote.send((obs, info))
            elif cmd == "action_masks":
                remote.send(env.action_masks())
            elif cmd == "start_route_pointer_indices":
                remote.send(env.start_route_pointer_indices())
            elif cmd == "mode_mask":
                remote.send(env.mode_mask())
            elif cmd == "net_valid_mask":
                remote.send(env.net_valid_mask())
            elif cmd == "env_method":
                method_name, method_args, method_kwargs = data
                remote.send(getattr(env, method_name)(*method_args, **method_kwargs))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "close":
                if env is not None:  # None after a failed reload_board
                    env.close()
                remote.close(); wd.close_clean(); break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented")
        except EOFError:
            wd.close_clean()
            break
        except KeyboardInterrupt:
            wd.close_clean()
            break
        except EngineServerCrashed:
            # Mirror of the base's die-on-server-crash: re-raise so this
            # worker dies (pipe EOF) and the parent's existing dead-worker
            # respawn fires — full rationale in the base worker.
            wd.dump_error(
                traceback.format_exc(),
                cmd=cmd,
                data=data,
                board_path=getattr(env, "board_path", None),
            )
            raise
        except Exception:
            tb = wd.dump_error(
                traceback.format_exc(),
                cmd=cmd,
                data=data,
                board_path=getattr(env, "board_path", None),
                last_obs=getattr(env, "_last_obs", None),
            )
            remote.send(("__error__", tb))


def install() -> bool:
    """Rebind subproc._decoder_worker to the timed wrapper. Idempotent.
    Returns True if patched. Called on import (in the forkserver server) and
    callable directly for tests."""
    global _PATCHED
    if _PATCHED:
        return True
    import pcb_world.vec.backends.subproc as _subproc
    _subproc._decoder_worker = _timed_decoder_worker
    _PATCHED = True
    print(f"[speed_profiler.worker_shim] patched _decoder_worker in pid={os.getpid()}",
          flush=True)
    return True


# Auto-install on import when profiling is enabled (the forkserver-preload path).
if os.environ.get("CADAGENT_PROFILE_WORKER"):
    install()
