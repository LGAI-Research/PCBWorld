"""Per-worker diag bundle — the one composition every pipe-worker entry installs.

One object owns the worker-side writers (fatal-signal log + stderr redirect)
and the lifecycle moments shared by all pipe workers (decoder env pool, eval
pool, profiler shim): ``tick()`` on the hot path, ``dump_error()`` in the
catch-all so the context-dump path rides the traceback across the pipe, and
``close_clean()`` on the clean exit paths ONLY (close/EOF/KeyboardInterrupt)
— a worker dying any other way must leave its artifacts behind.
"""
from __future__ import annotations

import os
from typing import Any

from pcb_world.diag import ENV_STEM, artifact_stem
from pcb_world.diag.crash_handler import install_crash_handler, remove_log_if_empty
from pcb_world.diag.dump import dump_context
from pcb_world.diag.stderr_tail import StderrTail


class WorkerDiag:
    def __init__(self, role: str) -> None:
        # Handler first: it exports the process stem the tail then reuses.
        self._handler = install_crash_handler(role, register_atexit=False)
        self._tail = StderrTail(os.environ.get(ENV_STEM) or artifact_stem(role))
        self._tail.install()

    def tick(self) -> None:
        self._tail.tick()

    def dump_error(self, tb: str, **payload: Any) -> str:
        """Dump exception context; return ``tb`` with the dump path appended."""
        dump = dump_context("worker_exception", traceback=tb, **payload)
        if dump is not None:
            tb += f"\n[diag] context dump: {dump}"
        return tb

    def close_clean(self) -> None:
        self._tail.close_clean()
        if self._handler is not None:
            remove_log_if_empty(*self._handler)
