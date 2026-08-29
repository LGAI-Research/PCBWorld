"""Subprocess-parallel vectorized env for Decoder-only policy training.

Extends SB3's SubprocVecEnv pattern to support:
  - **Dict observations** returned as ``list[dict]`` (no np.stack).
  - **No auto-reset** — the main process controls reset timing, which is
    essential for PPO truncation bootstrapping (need ``action_masks()`` etc.
    on the truncated state *before* reset).
  - **Selective stepping** — only step a subset of envs (GRPO excludes
    done envs from subsequent steps).
  - **Batch mask queries** — ``get_action_masks()``, ``get_pointer_masks()``,
    ``get_mode_masks()`` call the corresponding env methods in parallel.

Uses a local ``CloudpickleWrapper`` for env factory serialization and the same
``multiprocessing.Pipe`` / ``Process`` infrastructure as SB3's SubprocVecEnv.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import cloudpickle
import numpy as np

from pcb_world import diag
from pcb_world.engine.router_client import EngineServerCrashed
from pcb_world.vec.backends.base import VecBackend


class CloudpickleWrapper:
    """Wrap a variable so ``multiprocessing`` serializes it with cloudpickle.

    Env factories often close over lambdas / local functions that the stdlib
    ``pickle`` (used by ``multiprocessing``) cannot handle; cloudpickle can.
    Inlined here (mirrors Stable-Baselines3's helper) so this backend needs
    no SB3 import — importing SB3 transitively pulls in OpenCV, a heavy
    dependency this backend has no use for.
    """

    def __init__(self, var: Any) -> None:
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _decoder_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    board_factory_wrapper: CloudpickleWrapper | None = None,
    role: str = "env",
) -> None:
    """Event loop running inside each subprocess.

    Unlike SB3's ``_worker``, this worker does **not** auto-reset the env
    on done.  The main process sends an explicit ``"reset"`` command when
    ready.  This lets the main process query the terminal-state masks
    (needed for PPO truncation bootstrap) before resetting.

    If ``board_factory_wrapper`` is given, the worker also accepts a
    ``"reload_board"`` command that rebuilds the env from a new board
    path without tearing down the subprocess (avoids the KiCad singleton
    RLRouter re-init cost).

    Crash diagnostics (pcb_world.diag): fatal-signal log + stderr redirect
    are installed before the env exists, so even construction crashes are
    captured. Cleanup runs on the loop's close/EOF paths — multiprocessing
    children skip atexit (os._exit), so it cannot live there. Note stderr
    (KiCad wx asserts etc.) goes to ``var/crashlogs/<stem>_stderr.log``,
    not the parent's tty/log.
    """
    parent_remote.close()
    wd = diag.WorkerDiag(role)

    env = env_fn_wrapper.var()
    board_factory = (
        board_factory_wrapper.var if board_factory_wrapper is not None else None
    )

    while True:
        cmd = data = None
        try:
            cmd, data = remote.recv()
            if cmd == "reload_board":
                if board_factory is None:
                    raise RuntimeError(
                        "reload_board called but worker has no board_factory"
                    )
                board_path, reload_seq = data
                env.close()
                # ``env = None`` (not ``del env``): if board_factory raises, the
                # except-handler below still references ``env`` — with ``del``
                # that was an UnboundLocalError that masked the real exception
                # (the parent then saw only EOF and silently respawned).
                env = None
                gc.collect()
                env = board_factory(board_path, reload_seq)
                remote.send(True)
                continue
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                wd.tick()
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                # data = seed | None. None keeps the historical call shape
                # (``env.reset()``) so non-gymnasium envs — the LLM branch, test
                # doubles — stay callable; a seed is only ever passed when the
                # caller asked for one.
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
                method = getattr(env, method_name)
                remote.send(method(*method_args, **method_kwargs))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "close":
                if env is not None:  # None after a failed reload_board
                    env.close()
                remote.close()
                wd.close_clean()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented")
        except EOFError:
            wd.close_clean()
            break
        except KeyboardInterrupt:
            wd.close_clean()
            break
        except EngineServerCrashed:
            # Engine-IPC crash semantics = in-process semantics, by dying:
            # a fatal C++ signal kills the ENGINE SERVER child, not this
            # worker, and surfaces here as EngineServerCrashed. Re-raise so
            # this worker dies too (pipe EOF) — the parent's existing
            # dead-worker recovery then fires unchanged on every path (step
            # respawn + synthetic terminated step, reload_board respawn,
            # fatal EOFError on reset/mask paths), with the same postmortem
            # accounting an in-process segfault produced. Restarting the
            # server in-worker instead would need respawn-equivalent
            # machinery rebuilt on each of those paths. The native backtrace
            # is in the server's own var/crashlogs artifact; the context
            # dump below records the env side.
            wd.dump_error(
                traceback.format_exc(),
                cmd=cmd,
                data=data,
                board_path=getattr(env, "board_path", None),
            )
            raise
        except Exception:
            # Catch-all context dump: any exception — guard or not — leaves its
            # context where it fired, since only the tb string crosses the pipe.
            tb = wd.dump_error(
                traceback.format_exc(),
                cmd=cmd,
                data=data,
                board_path=getattr(env, "board_path", None),
                last_obs=getattr(env, "_last_obs", None),
            )
            remote.send(("__error__", tb))


# ---------------------------------------------------------------------------
# Main-process pool
# ---------------------------------------------------------------------------

class SubprocDecoderVecEnv(VecBackend):
    """Pool of ``KiCadRLWrapper`` environments in separate subprocesses.

    Designed as a **drop-in replacement** for ``list[KiCadRLWrapper]``
    in the decoder training collectors.  Supports both indexed access
    (``pool[i].action_masks()``) and batch operations
    (``pool.step_send() / pool.step_recv()``).

    Parameters
    ----------
    env_fns : list of callables
        Each callable ``env_fns[i]()`` creates one ``KiCadRLWrapper``.
    start_method : str, optional
        multiprocessing start method.  ``"forkserver"`` (default) is
        recommended because KiCad's C++ state is not fork-safe.
    advance_rng_on_reload : bool
        Advance each worker's per-env random stream on every env rebuild
        (board hot-swap or crash respawn) by passing an incrementing
        ``reload_seq`` to its board factory.  **Training pools want this on**:
        the wrapper seeds its RNG once at construction, so rebuilding on the
        same seed rewinds augmentation / slot_perm / auto-net-select to their
        first draws — with a per-iteration board reload the same transforms
        replay every iteration.  **Eval pools want it off** (the default):
        rewinding is what makes a board's rollout independent of which boards
        the worker processed before it.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], Any]],
        start_method: str | None = None,
        board_factories: list[Callable[..., Any]] | None = None,
        advance_rng_on_reload: bool = False,
    ) -> None:
        self._n_envs = len(env_fns)
        self._closed = False
        self._supports_reload = board_factories is not None
        if board_factories is not None and len(board_factories) != self._n_envs:
            raise ValueError(
                "board_factories length must match env_fns length"
            )

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"

        self._ctx = mp.get_context(start_method)
        self.remotes: list[mp.connection.Connection] = []
        self.work_remotes: list[mp.connection.Connection] = []
        self.processes: list[mp.Process] = []
        # Keep the env_fns + factories around so we can respawn a worker
        # whose subprocess died (e.g. C++ SIGSEGV from the PnS router).
        self._env_fns: list[Callable[[], Any]] = list(env_fns)
        self._board_factories: list[Callable[..., Any]] | None = (
            list(board_factories) if board_factories is not None else None
        )
        self._last_boards: list[str | None] = [None] * self._n_envs
        # Cached last-obs per env — returned as the "reset-like" observation
        # when a worker crashes mid-step, so the collector has something to
        # feed the policy on the next step.
        self._last_obs: list[dict | None] = [None] * self._n_envs
        self._respawn_count: list[int] = [0] * self._n_envs
        # Per-env env-rebuild counter mixed into the worker's RNG seed (see
        # ``advance_rng_on_reload``). Owned by the parent so a crashed worker's
        # replacement resumes the sequence instead of replaying it.
        self._advance_rng_on_reload = bool(advance_rng_on_reload)
        self._rng_reload_seq: list[int] = [0] * self._n_envs

        for i, env_fn in enumerate(env_fns):
            self._spawn_worker(i)

        # State for async step
        self._pending_indices: list[int] | None = None
        # Per-index flag: True when _safe_send already detected a crash
        # and respawned, so _safe_recv_step should skip recv() and just
        # synthesize a terminated response.
        self._send_crashed: list[bool] = [False] * self._n_envs

    # ------------------------------------------------------------------
    # Worker (re)spawn
    # ------------------------------------------------------------------
    def _spawn_worker(self, idx: int) -> None:
        """Start (or restart) the subprocess for env index ``idx``."""
        ctx = self._ctx
        parent_conn, child_conn = ctx.Pipe()
        factory_wrapper = (
            CloudpickleWrapper(self._board_factories[idx])
            if self._board_factories is not None else None
        )
        # If respawning after reload_board(), construct the env on the
        # most recently requested board so the new worker is in the
        # right state before we send a reset.
        env_fn = self._env_fns[idx]
        last_board = self._last_boards[idx]
        if last_board is not None and self._board_factories is not None:
            factory = self._board_factories[idx]
            seq = self._rng_reload_seq[idx]
            env_fn = lambda f=factory, b=last_board, q=seq: f(b, q)
        process = ctx.Process(
            target=_decoder_worker,
            args=(
                child_conn, parent_conn,
                CloudpickleWrapper(env_fn), factory_wrapper,
                f"env{idx}",
            ),
            daemon=True,
        )
        process.start()
        child_conn.close()
        # Replace slots (or append for initial spawn).
        if idx < len(self.remotes):
            try:
                self.remotes[idx].close()
            except Exception:
                pass
            self.remotes[idx] = parent_conn
            self.work_remotes[idx] = child_conn
            self.processes[idx] = process
        else:
            self.remotes.append(parent_conn)
            self.work_remotes.append(child_conn)
            self.processes.append(process)

    @property
    def respawn_total(self) -> int:
        """Cumulative worker respawns, all causes (step + mask/reset paths).

        The trainer logs this as ``diag/worker_respawn_total`` — the curve's
        slope is the crash rate.
        """
        return sum(self._respawn_count)

    def _recover_worker(self, idx: int, why: str) -> dict:
        """Respawn worker ``idx`` after a crash and return a reset obs.

        Sends "reset" to the fresh worker so the env is ready for the
        next step. Respawns are unlimited by design (intermittent engine
        segfaults are tolerated); the postmortem json records the cause.
        """
        import sys
        self._respawn_count[idx] += 1
        if self._advance_rng_on_reload:
            self._rng_reload_seq[idx] += 1
        # Best-effort teardown of the dead process.
        proc = self.processes[idx]
        try:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=2)
        except Exception:
            pass
        print(
            f"[SubprocDecoderVecEnv] env {idx} crashed ({why}); "
            f"exitcode={proc.exitcode}, "
            f"respawning (total respawns: {self._respawn_count[idx]})",
            file=sys.stderr, flush=True,
        )
        diag.write_postmortem(
            f"env{idx}", proc, why,
            respawn_count=self._respawn_count[idx],
            board=self._last_boards[idx],
        )
        self._spawn_worker(idx)
        # Reset the fresh env and cache its obs.
        self.remotes[idx].send(("reset", None))
        obs, _info = self._check(self.remotes[idx].recv())
        self._last_obs[idx] = obs
        return obs

    # ------------------------------------------------------------------
    # Container protocol (len, getitem) — list-like access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_envs

    def __getitem__(self, idx: int) -> _EnvProxy:
        return _EnvProxy(self, idx)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self) -> list[dict]:
        """Reset all envs in parallel. Returns list[obs_dict]."""
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        return [self._check(r)[0] for r in results]

    def reset(self, idx: int) -> tuple[dict, dict]:
        """Reset a single env. Returns (obs_dict, info)."""
        self.remotes[idx].send(("reset", None))
        result = self.remotes[idx].recv()
        return self._check(result)

    def reset_batch(
        self, indices: Sequence[int], seeds: Sequence[int] | None = None,
    ) -> list[tuple[dict, dict]]:
        """Reset a subset of envs in parallel.

        ``seeds`` (parallel to ``indices``) seeds each env's gymnasium
        ``np_random``; see :meth:`VecBackend.reset_batch`.
        """
        idxs = list(indices)
        if seeds is None:
            payloads: list[int | None] = [None] * len(idxs)
        else:
            payloads = [int(s) for s in seeds]
            if len(payloads) != len(idxs):
                raise ValueError(
                    f"reset_batch: {len(payloads)} seeds for {len(idxs)} indices"
                )
        for i, seed in zip(idxs, payloads):
            self.remotes[i].send(("reset", seed))
        results = [self.remotes[i].recv() for i in idxs]
        return [self._check(r) for r in results]

    # ------------------------------------------------------------------
    # Step — batch (all envs)
    # ------------------------------------------------------------------

    def step_async(self, actions: np.ndarray | Sequence) -> None:
        """Send step commands to ALL envs (non-blocking)."""
        for i, remote in enumerate(self.remotes):
            self._safe_send(i, ("step", actions[i]))
        self._pending_indices = list(range(self._n_envs))

    def step_wait(self) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Collect results from ``step_async``.

        Returns
        -------
        obs_list : list[dict]
        rewards : (N,) float
        terminated : (N,) bool
        truncated : (N,) bool
        infos : list[dict]
        """
        assert self._pending_indices is not None, "call step_async first"
        results_by_idx: dict[int, Any] = {}
        for i in self._pending_indices:
            results_by_idx[i] = self._safe_recv_step(i)
        self._pending_indices = None
        obs_l, rew_l, term_l, trunc_l, info_l = [], [], [], [], []
        for i in range(self._n_envs):
            o, r, t, tr, info = results_by_idx[i]
            obs_l.append(o); rew_l.append(r); term_l.append(t); trunc_l.append(tr); info_l.append(info)
        return (
            obs_l,
            np.array(rew_l, dtype=np.float64),
            np.array(term_l, dtype=bool),
            np.array(trunc_l, dtype=bool),
            info_l,
        )

    # ------------------------------------------------------------------
    # Step — selective (subset of envs, for GRPO)
    # ------------------------------------------------------------------

    def step_async_selective(
        self,
        indices: Sequence[int],
        actions: np.ndarray | Sequence,
    ) -> None:
        """Send step commands to a SUBSET of envs."""
        for k, i in enumerate(indices):
            self._safe_send(i, ("step", actions[k]))
        self._pending_indices = list(indices)

    def step_wait_selective(
        self,
    ) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Collect results from ``step_async_selective``."""
        assert self._pending_indices is not None
        results = [self._safe_recv_step(i) for i in self._pending_indices]
        self._pending_indices = None
        obs, rews, terms, truncs, infos = zip(*results)
        return (
            list(obs),
            np.array(rews, dtype=np.float64),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
            list(infos),
        )

    # ------------------------------------------------------------------
    # Batch mask queries
    # ------------------------------------------------------------------

    def get_action_masks(self, indices: Sequence[int] | None = None) -> np.ndarray:
        """Parallel ``action_masks()`` → (len(indices), NUM_ACTIONS) bool."""
        idxs = indices if indices is not None else range(self._n_envs)
        for i in idxs:
            self.remotes[i].send(("action_masks", None))
        masks = [self._check(self.remotes[i].recv()) for i in idxs]
        return np.stack(masks)

    def get_pointer_masks(self, indices: Sequence[int] | None = None) -> np.ndarray:
        """Parallel ``start_route_pointer_indices()`` → ``(len(indices), K_max)``
        int64 right-padded with ``-1``.

        Per-env match counts vary (0–N depending on how many cands share
        the start_route ``(x, y)``), so the batch is right-padded so
        downstream policy code can rely on a rectangular ndarray. ``-1``
        is the sentinel for "no entry"; the policy treats every non-
        negative column as a cand index to mask to ``-inf``.
        """
        idxs = list(indices) if indices is not None else list(range(self._n_envs))
        for i in idxs:
            self.remotes[i].send(("start_route_pointer_indices", None))
        per_env = [self._check(self.remotes[i].recv()) for i in idxs]
        if not per_env:
            return np.zeros((0, 0), dtype=np.int64)
        K_max = max((p.shape[0] for p in per_env), default=0)
        if K_max == 0:
            return np.full((len(per_env), 0), -1, dtype=np.int64)
        out = np.full((len(per_env), K_max), -1, dtype=np.int64)
        for i, p in enumerate(per_env):
            if p.shape[0]:
                out[i, : p.shape[0]] = p
        return out

    def get_mode_masks(self, indices: Sequence[int] | None = None) -> np.ndarray:
        """Parallel ``mode_mask()`` → (len(indices), 3) bool."""
        idxs = indices if indices is not None else range(self._n_envs)
        for i in idxs:
            self.remotes[i].send(("mode_mask", None))
        modes = [self._check(self.remotes[i].recv()) for i in idxs]
        return np.stack(modes)

    def get_net_valid_masks(
        self, indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Parallel ``net_valid_mask()`` → (len(indices), M_max) bool.

        Per-env nets may differ in count; the batch is right-padded with
        False so downstream indexers can rely on a rectangular ndarray.
        Padded positions must be masked to -inf in the policy's net
        pointer logits (mirrors the net_indices=-1 handling).
        """
        idxs = list(indices) if indices is not None else list(range(self._n_envs))
        for i in idxs:
            self.remotes[i].send(("net_valid_mask", None))
        per_env = [self._check(self.remotes[i].recv()) for i in idxs]
        if not per_env:
            return np.zeros((0, 0), dtype=bool)
        M_max = max(m.shape[0] for m in per_env)
        out = np.zeros((len(per_env), M_max), dtype=bool)
        for i, m in enumerate(per_env):
            out[i, : m.shape[0]] = m
        return out

    # ------------------------------------------------------------------
    # Multi-board support: swap the underlying board in every worker
    # ------------------------------------------------------------------

    def _reload_indices(self, pairs: Sequence[tuple[int, str]], caller: str) -> None:
        """Dispatch ``reload_board`` to each ``(idx, path)`` in parallel, join,
        and respawn any worker that died at the pipe (crash recovery parity
        with the step path — otherwise a dead worker would make reload raise
        EOFError and kill the whole trainer).

        ``_last_boards`` is updated before the send, so ``_spawn_worker``
        inside ``_recover_worker`` rebuilds the dead worker directly on its
        NEW board — the recovery reset doubles as this reload's ack. Only
        connection-level failures are recovered; an error *reported* by a live
        worker (e.g. board parse failure) still raises, since a respawn on the
        same board would fail identically.
        """
        dead: list[tuple[int, str]] = []
        for i, path in pairs:
            self._last_boards[i] = path
            if self._advance_rng_on_reload:
                self._rng_reload_seq[i] += 1
            try:
                self.remotes[i].send(
                    ("reload_board", (path, self._rng_reload_seq[i])),
                )
            except (BrokenPipeError, OSError) as e:
                dead.append((i, f"{caller} send: {type(e).__name__}"))
        dead_idx = {i for i, _ in dead}
        for i, _ in pairs:
            if i in dead_idx:
                continue
            try:
                self._check(self.remotes[i].recv())
            except (EOFError, OSError) as e:
                dead.append((i, f"{caller} recv: {type(e).__name__}"))
        for i, why in dead:
            self._recover_worker(i, why)

    def reload_board(self, board_path: str) -> None:
        """Tell every worker to close its env and rebuild it on a new board.

        Requires the pool to have been constructed with ``board_factories``.
        The subprocess itself is **not** restarted — only the inner
        ``KiCadRLWrapper`` / ``PCBWorld`` is recreated — so KiCad's
        singleton RLRouter is initialized once per process lifetime.
        """
        if not self._supports_reload:
            raise RuntimeError(
                "reload_board requires board_factories at pool construction"
            )
        self._reload_indices(
            [(i, board_path) for i in range(self._n_envs)], "reload_board",
        )

    def reload_board_slot(self, idx: int, board_path: str) -> None:
        """Reload only one worker's env to a new board.

        Used by the async eval scheduler, which swaps boards per-env as
        individual slots finish their rollout quota on their current
        board. Blocks until the single worker acknowledges.
        """
        if not self._supports_reload:
            raise RuntimeError(
                "reload_board_slot requires board_factories at pool construction"
            )
        self._reload_indices([(idx, board_path)], "reload_board_slot")

    def reload_boards(self, board_paths: Sequence[str]) -> None:
        """Per-env variant of :meth:`reload_board`.

        Each worker ``i`` reloads to ``board_paths[i]``.  All reloads are
        dispatched in parallel, then joined.  Safe because KiCad's
        RLRouter singleton is per-process — each subprocess owns its own.
        """
        if not self._supports_reload:
            raise RuntimeError(
                "reload_boards requires board_factories at pool construction"
            )
        if len(board_paths) != self._n_envs:
            raise ValueError(
                f"board_paths length {len(board_paths)} != n_envs {self._n_envs}"
            )
        self._reload_indices(list(enumerate(board_paths)), "reload_boards")

    # ------------------------------------------------------------------
    # Crash recovery (VecBackend contract)
    # ------------------------------------------------------------------

    def rebuild(self, indices: Sequence[int] | None = None) -> None:
        """Respawn workers in place — recovery from a dead subprocess.

        Best-effort terminates the old process for each index, then restarts
        it (rebuilding the env on the worker's most recently requested board
        via :meth:`_spawn_worker`). The caller is expected to ``reset_*`` the
        rebuilt workers afterwards. (Note: ``step`` already auto-recovers a
        crashed worker inline via ``_safe_send`` / ``_safe_recv_step``; this is
        the explicit, caller-driven counterpart that the ``VecBackend``
        contract requires.)
        """
        target = range(self._n_envs) if indices is None else indices
        for idx in target:
            try:
                if self.processes[idx].is_alive():
                    self.processes[idx].terminate()
                self.processes[idx].join(timeout=2)
            except Exception:  # noqa: BLE001 — already-dead process, ignore
                pass
            if self._advance_rng_on_reload:
                self._rng_reload_seq[idx] += 1
            self._spawn_worker(idx)
            self._send_crashed[idx] = False
            self._last_obs[idx] = None

    # ------------------------------------------------------------------
    # Generic env_method (matches SB3 API)
    # ------------------------------------------------------------------

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Sequence[int] | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        idxs = indices if indices is not None else range(self._n_envs)
        for i in idxs:
            self.remotes[i].send(
                ("env_method", (method_name, method_args, method_kwargs))
            )
        return [self._check(self.remotes[i].recv()) for i in idxs]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        if self._pending_indices is not None:
            for i in self._pending_indices:
                self.remotes[i].recv()
            self._pending_indices = None
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self._closed = True

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> SubprocDecoderVecEnv:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _check(result: Any) -> Any:
        """Re-raise worker exceptions in the main process."""
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "__error__":
            raise RuntimeError(f"Worker exception:\n{result[1]}")
        return result

    # ------------------------------------------------------------------
    # Crash recovery helpers
    # ------------------------------------------------------------------
    _DEAD_WORKER_ERRORS = (
        EOFError, BrokenPipeError, ConnectionResetError, OSError,
    )

    def _safe_send(self, idx: int, payload: Any) -> None:
        """send() that respawns the worker if the pipe is already dead.

        On crash, does NOT re-send the payload; ``_safe_recv_step`` will
        synthesize a terminated response instead (see ``_send_crashed``).
        """
        try:
            self.remotes[idx].send(payload)
        except self._DEAD_WORKER_ERRORS as e:
            self._recover_worker(idx, f"send: {type(e).__name__}")
            self._send_crashed[idx] = True

    def _safe_recv_step(
        self, idx: int,
    ) -> tuple[dict, float, bool, bool, dict]:
        """recv() for a step response; respawns on crash and returns
        a synthetic terminated step (reset_obs, -1.0, True, False, info).

        ``info["engine_crash"]=True`` is set so downstream can log it.
        """
        # Crashed before send completed — worker has been respawned and
        # reset; return a synthetic terminated tuple so the collector's
        # done-handling path kicks in.
        if self._send_crashed[idx]:
            self._send_crashed[idx] = False
            obs = self._last_obs[idx]
            # Invariant: _send_crashed is only ever set right after
            # _recover_worker(), which caches the fresh reset obs — fail loudly
            # (broken invariant = a bug) instead of leaking an empty obs.
            assert obs is not None, (
                f"env {idx}: _send_crashed set without a cached recovery obs "
                "(_recover_worker must fill _last_obs before the flag)"
            )
            return (obs, -1.0, True, False, {"engine_crash": True,
                                               "action_success": False})
        try:
            res = self.remotes[idx].recv()
        except self._DEAD_WORKER_ERRORS as e:
            obs = self._recover_worker(idx, f"recv: {type(e).__name__}")
            return (obs, -1.0, True, False, {"engine_crash": True,
                                               "action_success": False})
        # Worker raised a Python exception during step() — _check will
        # raise. Treat that the same as a death: log and synthesize.
        if (isinstance(res, tuple) and len(res) == 2
                and res[0] == "__error__"):
            obs = self._recover_worker(idx, f"worker exception:\n{res[1]}")
            return (obs, -1.0, True, False, {"engine_crash": True,
                                               "action_success": False,
                                               "traceback": res[1]})
        obs, r, term, trunc, info = res
        self._last_obs[idx] = obs
        return obs, r, term, trunc, info


# ---------------------------------------------------------------------------
# EnvProxy — thin wrapper for pool[i].method() syntax
# ---------------------------------------------------------------------------

class _EnvProxy:
    """Allows ``pool[i].step(action)`` / ``pool[i].action_masks()`` etc."""

    __slots__ = ("_pool", "_idx")

    def __init__(self, pool: SubprocDecoderVecEnv, idx: int) -> None:
        self._pool = pool
        self._idx = idx

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._pool.remotes[self._idx].send(("step", action))
        result = self._pool._check(self._pool.remotes[self._idx].recv())
        return result

    def reset(self) -> tuple[dict, dict]:
        return self._pool.reset(self._idx)

    def action_masks(self) -> np.ndarray:
        self._pool.remotes[self._idx].send(("action_masks", None))
        return self._pool._check(self._pool.remotes[self._idx].recv())

    def start_route_pointer_indices(self) -> np.ndarray:
        self._pool.remotes[self._idx].send(("start_route_pointer_indices", None))
        return self._pool._check(self._pool.remotes[self._idx].recv())

    def mode_mask(self) -> np.ndarray:
        self._pool.remotes[self._idx].send(("mode_mask", None))
        return self._pool._check(self._pool.remotes[self._idx].recv())

    def net_valid_mask(self) -> np.ndarray:
        self._pool.remotes[self._idx].send(("net_valid_mask", None))
        return self._pool._check(self._pool.remotes[self._idx].recv())
