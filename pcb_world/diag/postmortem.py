"""Parent-side death record for a crashed worker.

SIGKILL (the OOM killer) cannot be observed from inside the dying process, so
the parent writes the verdict at respawn time: exit code, decoded signal, and
a glob that pairs the dead pid with the in-process artifacts it left behind
(fatal-signal log, steps.jsonl, stderr tail — none of which reach their clean
-exit deletion when the worker dies).
"""
from __future__ import annotations

import json
import os
import signal
import socket
import time
from typing import Any

from pcb_world.diag import default_log_dir, diag_enabled

_SIGNAL_HINTS = {
    signal.SIGKILL: "SIGKILL — OOM killer likely (check rss_mb trail in the steps.jsonl)",
    signal.SIGSEGV: "SIGSEGV — see the native backtrace in the dead pid's .log",
    signal.SIGABRT: "SIGABRT — assert/abort; see .log and stderr tail",
    signal.SIGBUS: "SIGBUS",
}


def write_postmortem(
    role: str,
    proc,
    why: str,
    *,
    respawn_count: int,
    board: Any = None,
) -> str | None:
    """Write ``<ts>_<role>_postmortem.json`` for a dead ``mp.Process``. Never raises."""
    if not diag_enabled():
        return None
    try:
        exitcode = proc.exitcode
        sig = None
        if exitcode is not None and exitcode < 0:
            try:
                s = signal.Signals(-exitcode)
                sig = _SIGNAL_HINTS.get(s, s.name)
            except ValueError:
                sig = f"unknown signal {-exitcode}"
        record = {
            "role": role,
            "host": socket.gethostname(),
            "pid": proc.pid,
            "exitcode": exitcode,
            "signal": sig,
            "why": str(why)[:2000],
            "respawn_count": respawn_count,
            "board": board,
            "artifacts_glob": f"*_{proc.pid}*",
        }
        path = os.path.join(
            default_log_dir(),
            f"{time.strftime('%y%m%d_%H%M%S')}_{role}_postmortem.json",
        )
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        return path
    except Exception:
        return None
