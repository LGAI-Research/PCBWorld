"""Redirect fd 2 to a per-process file so the engine's last stderr survives a crash.

A plain file (dup2), not a pipe/ring: SIGSEGV kills any pump thread with the
process, losing whatever a pipe still buffered — with dup2 every write is
already in the page cache. Size is bounded by a cheap watchdog instead:
``tick()`` counts calls and every ``check_every`` does one fstat; past
``max_bytes`` it snapshots the last lines and truncates in place.
"""
from __future__ import annotations

import os
import sys

from pcb_world.diag import default_log_dir, diag_enabled

_TAIL_LINES = 200


class StderrTail:
    def __init__(
        self,
        stem: str,
        log_dir: str | None = None,
        max_bytes: int = 50 * 2**20,
        check_every: int = 512,
    ) -> None:
        self._path = os.path.join(log_dir or default_log_dir(), f"{stem}_stderr.log")
        self._max_bytes = max_bytes
        self._check_every = check_every
        self._count = 0
        self._snapshots = 0
        self._installed = False

    def install(self) -> None:
        if not diag_enabled():
            return
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
            sys.stderr.flush()
            os.dup2(fd, 2)  # sys.stderr wraps fd 2, so it keeps working
            os.close(fd)
            self._installed = True
        except OSError:
            pass

    def tick(self) -> None:
        if not self._installed:
            return
        self._count += 1
        if self._count % self._check_every:
            return
        try:
            if os.fstat(2).st_size > self._max_bytes:
                self._truncate()
        except OSError:
            pass

    def _truncate(self) -> None:
        self._snapshots += 1
        snap = self._path.replace("_stderr.log", f"_stderr_tailsnap{self._snapshots}.log")
        try:
            with open(self._path, "rb") as f:
                f.seek(-min(self._max_bytes, 2**20), os.SEEK_END)
                tail = f.read().splitlines()[-_TAIL_LINES:]
            with open(snap, "wb") as f:
                f.write(b"\n".join(tail) + b"\n")
        except OSError:
            pass
        sys.stderr.flush()
        os.ftruncate(2, 0)
        os.lseek(2, 0, os.SEEK_SET)

    def close_clean(self) -> None:
        """Clean exit: the stderr evidence is not needed — remove it."""
        if not self._installed:
            return
        try:
            sys.stderr.flush()
            os.remove(self._path)
        except OSError:
            pass
