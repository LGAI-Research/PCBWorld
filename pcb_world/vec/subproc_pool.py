"""Subprocess-parallel pool for post-hoc KiCad evaluation.

Slim sibling of :class:`pcb_world.vec.backends.subproc.SubprocDecoderVecEnv`:
same forkserver/spawn subprocess pattern, same ``mp.Pipe`` IPC, same crash
respawn discipline — but each worker only knows how to call
:func:`eval.metrics.evaluate_one` on a (routed_pcb, pro_path) pair.

Rationale: KiCad's ``RLRouter`` is a per-process singleton and its C++ state
is not fork-safe, so any parallel KiCad workload must use ``forkserver`` (or
``spawn``) subprocesses. Stage-2 post-hoc DRC does not need the gym
step/reset/observation machinery that ``SubprocDecoderVecEnv`` carries, so we
keep this pool minimal — one command (``eval``), one response (metric dict).
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import traceback
from collections.abc import Callable, Sequence
from typing import Any

from configs.loader.schema import DEFAULTS

from pcb_world import diag
from pcb_world.engine.router_client import EngineServerCrashed


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _eval_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    reward_config: str,
    check_angle: int,
    role: str = "eval",
) -> None:
    """Event loop running inside each subprocess.

    Imports ``evaluate_one`` lazily so the heavy KiCad C++ binding is loaded
    in the worker process (not the parent). Per-eval exceptions are caught
    and returned via the ``("__error__", traceback)`` sentinel so a single
    bad board does not kill the worker.

    Crash diagnostics (pcb_world.diag) are installed before the lazy import,
    so even binding-load crashes are captured; cleanup runs on the loop's
    close/EOF paths (children skip atexit). Worker stderr goes to
    ``var/crashlogs/<stem>_stderr.log``, not the parent's tty/log.
    """
    parent_remote.close()
    wd = diag.WorkerDiag(role)

    # Lazy import: keep the parent process free of kicad_rl_router state
    # so it can stay light (CSV writing, scheduling only).
    from eval.metrics import evaluate_one

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            wd.close_clean()
            break
        except KeyboardInterrupt:
            wd.close_clean()
            break

        if cmd == "close":
            try:
                remote.close()
            except Exception:
                pass
            wd.close_clean()
            break

        if cmd != "eval":
            remote.send(("__error__", f"unknown command: {cmd}"))
            continue

        routed_pcb, pro_path = data
        try:
            result = evaluate_one(
                routed_pcb,
                pro_path,
                reward_config_name=reward_config,
                check_angle=check_angle,
            )
            wd.tick()
            remote.send(result)
        except EngineServerCrashed:
            # Die so the parent's dead-worker respawn fires — same rationale
            # as the decoder worker (pcb_world/vec/backends/subproc.py): the
            # C++ crash killed the engine-server child, and in-process mode
            # would have killed THIS worker.
            wd.dump_error(
                traceback.format_exc(),
                routed_pcb=routed_pcb,
                pro_path=pro_path,
            )
            raise
        except Exception:
            tb = wd.dump_error(
                traceback.format_exc(),
                routed_pcb=routed_pcb,
                pro_path=pro_path,
            )
            remote.send(("__error__", tb))
        finally:
            # Help GC release the KiCad RLRouter singleton between boards.
            gc.collect()


# ---------------------------------------------------------------------------
# Main-process pool
# ---------------------------------------------------------------------------


class SubprocEvalPool:
    """Pool of ``evaluate_one`` workers in separate subprocesses.

    Parameters
    ----------
    n_workers : int
        Number of worker subprocesses.
    reward_config : str
        Reward YAML name passed to ``evaluate_one`` (used for the
        ``final_potential`` channel only).
    check_angle : int
        ``45`` or ``90``; routing-angle DRV mode.
    start_method : str, optional
        multiprocessing start method. ``"forkserver"`` (default when
        available) matches the existing rollout pool; ``fork`` is unsafe
        for KiCad C++ state.
    """

    _DEAD_WORKER_ERRORS = (
        EOFError,
        BrokenPipeError,
        ConnectionResetError,
        OSError,
    )

    def __init__(
        self,
        n_workers: int,
        *,
        reward_config: str = DEFAULTS.reward_config,
        check_angle: int = DEFAULTS.check_angle,
        start_method: str | None = None,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        if start_method is None:
            available = mp.get_all_start_methods()
            start_method = "forkserver" if "forkserver" in available else "spawn"
        if start_method == "fork":
            raise ValueError(
                "fork start method is unsafe for KiCad C++ state; "
                "use forkserver or spawn"
            )

        self._n_workers = n_workers
        self._reward_config = reward_config
        self._check_angle = check_angle
        self._ctx = mp.get_context(start_method)
        self._closed = False

        self.remotes: list[mp.connection.Connection] = []
        self.work_remotes: list[mp.connection.Connection] = []
        self.processes: list[mp.Process] = []
        self._respawn_count: list[int] = [0] * n_workers
        for i in range(n_workers):
            self._spawn_worker(i)

    # ------------------------------------------------------------------
    # Worker (re)spawn
    # ------------------------------------------------------------------

    def _spawn_worker(self, idx: int) -> None:
        ctx = self._ctx
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=_eval_worker,
            args=(
                child_conn,
                parent_conn,
                self._reward_config,
                self._check_angle,
                f"eval{idx}",
            ),
            daemon=True,
        )
        process.start()
        child_conn.close()
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

    def _recover_worker(self, idx: int, why: str, board: str | None = None) -> None:
        """Respawn after a crash — unlimited by design; postmortem records the cause."""
        import sys

        self._respawn_count[idx] += 1
        proc = self.processes[idx]
        try:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=2)
        except Exception:
            pass
        print(
            f"[SubprocEvalPool] worker {idx} crashed ({why}); "
            f"exitcode={proc.exitcode}, "
            f"respawning (total respawns: {self._respawn_count[idx]})",
            file=sys.stderr,
            flush=True,
        )
        diag.write_postmortem(
            f"eval{idx}", proc, why,
            respawn_count=self._respawn_count[idx],
            board=board,
        )
        self._spawn_worker(idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def evaluate_many(
        self,
        jobs: Sequence[tuple[str, str | None]],
        *,
        on_result: Callable[[int, dict], None] | None = None,
    ) -> list[dict]:
        """Run ``evaluate_one`` on each ``(routed_pcb, pro_path)`` job.

        Distributes jobs across the worker pool. Returns results in input
        order. If a worker crashes mid-job, the worker is respawned and the
        affected job's result becomes an error dict with ``error`` and
        ``traceback`` keys.

        If ``on_result`` is given, it is called as
        ``on_result(job_idx, result_dict)`` each time a result is collected,
        for incremental CSV flushing / progress logging.
        """
        results: list[dict | None] = [None] * len(jobs)
        if not jobs:
            return []  # type: ignore[return-value]

        # job_idx currently in flight at each worker slot, or None if idle.
        in_flight: list[int | None] = [None] * self._n_workers
        next_job = 0

        def _dispatch(slot: int) -> None:
            nonlocal next_job
            if next_job >= len(jobs):
                return
            routed_pcb, pro_path = jobs[next_job]
            try:
                self.remotes[slot].send(("eval", (routed_pcb, pro_path)))
                in_flight[slot] = next_job
                next_job += 1
            except self._DEAD_WORKER_ERRORS as e:
                # Pipe already dead — respawn and synthesize an error result
                # for this job.
                err_result = {
                    "board": str(routed_pcb),
                    "pro": str(pro_path) if pro_path else None,
                    "error": f"send_failed: {type(e).__name__}",
                    "traceback": "",
                }
                results[next_job] = err_result
                if on_result is not None:
                    on_result(next_job, err_result)
                self._recover_worker(slot, f"send: {type(e).__name__}", board=str(routed_pcb))
                in_flight[slot] = None
                next_job += 1

        # Prime: one job per worker, up to n_workers or len(jobs).
        for slot in range(min(self._n_workers, len(jobs))):
            _dispatch(slot)

        # Drain: each completed worker gets the next job until done.
        while any(s is not None for s in in_flight):
            ready = mp.connection.wait(
                [self.remotes[s] for s in range(self._n_workers)
                 if in_flight[s] is not None]
            )
            # Map ready connections back to slot indices.
            ready_slots: list[int] = []
            for conn in ready:
                for slot in range(self._n_workers):
                    if in_flight[slot] is not None and self.remotes[slot] is conn:
                        ready_slots.append(slot)
                        break

            for slot in ready_slots:
                job_idx = in_flight[slot]
                assert job_idx is not None
                routed_pcb, pro_path = jobs[job_idx]
                try:
                    res = self.remotes[slot].recv()
                except self._DEAD_WORKER_ERRORS as e:
                    res = {
                        "board": str(routed_pcb),
                        "pro": str(pro_path) if pro_path else None,
                        "error": f"recv_failed: {type(e).__name__}",
                        "traceback": "",
                    }
                    self._recover_worker(slot, f"recv: {type(e).__name__}", board=str(routed_pcb))
                else:
                    if (
                        isinstance(res, tuple)
                        and len(res) == 2
                        and res[0] == "__error__"
                    ):
                        res = {
                            "board": str(routed_pcb),
                            "pro": str(pro_path) if pro_path else None,
                            "error": "worker_exception",
                            "traceback": res[1],
                        }
                results[job_idx] = res  # type: ignore[assignment]
                if on_result is not None:
                    on_result(job_idx, res)
                in_flight[slot] = None
                _dispatch(slot)

        # All slots idle and next_job == len(jobs) ⇒ done.
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for process in self.processes:
            try:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for remote in self.remotes:
            try:
                remote.close()
            except Exception:
                pass
        self._closed = True

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> SubprocEvalPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
