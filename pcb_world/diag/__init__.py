"""Crash/diagnostics toolkit — stdlib-only at import time (no torch, no engine).

Design invariants (do not weaken):
- Never fsync: flush() to the kernel page cache is enough — it survives
  SIGSEGV/SIGKILL of the writing process; only a kernel panic loses it.
- No work at crash time: state is streamed *before* risky calls; context dumps
  happen only on exception paths. Signal handlers dump stacks, nothing else.
- Diagnostics never break the workload: every public entry is fail-soft.
- Killswitch: KICAD_CRASH_DIAG=0 disables every writer here (guards in
  methods/rl_agent still raise — only their dump becomes a no-op).

Artifact pairing contract — one stem per process lifetime,
``<ts>_<role>_<host>_<pid>`` (pytest keeps its legacy host-less stem):
  <stem>.log            fatal-signal stacks (native C++ + Python)  [crash_handler]
  <stem>_steps.jsonl    per-step action/state/RSS stream           [step_stream]
  <stem>_stderr.log     engine stderr (wx asserts etc.)            [stderr_tail]
  <ts>_<tag>_pid<pid>.pt/.json   guard context dumps               [dump]
  <ts>_<role>_postmortem.json    death cause, written by the PARENT [vec backends]
A clean exit deletes its own files: anything left in the dir means an incident.
"""
from __future__ import annotations

import os
import socket
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: environ keys — the worker entry exports the stem it chose so that code deep
#: inside the process (CrashLogger) lands on the same stem without plumbing.
ENV_STEM = "KICAD_DIAG_STEM"
ENV_ROLE = "KICAD_DIAG_ROLE"


def diag_enabled() -> bool:
    return os.environ.get("KICAD_CRASH_DIAG", "1") != "0"


def default_log_dir() -> str:
    d = os.environ.get("KICAD_CRASH_LOG_DIR") or str(PROJECT_ROOT / "var" / "crashlogs")
    os.makedirs(d, exist_ok=True)
    return d


def artifact_stem(role: str, *, with_host: bool = True) -> str:
    ts = time.strftime("%y%m%d_%H%M%S")
    host = f"_{socket.gethostname().split('.')[0]}" if with_host else ""
    return f"{ts}_{role}{host}_{os.getpid()}"


def read_rss_mb() -> float | None:
    """Resident set size in MB via /proc (Linux; ~µs). None elsewhere."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE_MB
    except OSError:
        return None


_PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 2**20 if hasattr(os, "sysconf") else 0.0

from pcb_world.diag.crash_handler import install_crash_handler, remove_log_if_empty  # noqa: E402
from pcb_world.diag.dump import dump_context, guard_fail  # noqa: E402
from pcb_world.diag.postmortem import write_postmortem  # noqa: E402
from pcb_world.diag.stderr_tail import StderrTail  # noqa: E402
from pcb_world.diag.step_stream import CrashLogger  # noqa: E402
from pcb_world.diag.worker import WorkerDiag  # noqa: E402

__all__ = [
    "PROJECT_ROOT",
    "ENV_STEM",
    "ENV_ROLE",
    "diag_enabled",
    "default_log_dir",
    "artifact_stem",
    "read_rss_mb",
    "install_crash_handler",
    "remove_log_if_empty",
    "dump_context",
    "guard_fail",
    "write_postmortem",
    "StderrTail",
    "CrashLogger",
    "WorkerDiag",
]
