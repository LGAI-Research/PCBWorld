"""Per-step action/state stream — the ``<stem>_e<env_id>_steps.jsonl`` artifact.

Streamed BEFORE each risky C++ call so the record survives a SIGSEGV/SIGKILL
(flush to the kernel page cache only — never fsync); board path + the
episode's action sequence make a crash deterministically replayable.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from pcb_world.diag import ENV_STEM, artifact_stem, default_log_dir, diag_enabled, read_rss_mb


class CrashLogger:
    """Streams each step to disk so the record survives a SIGSEGV/SIGKILL.

    Every entry is flushed BEFORE the C++ dispatch — flush() reaches the kernel
    page cache, which outlives the process (never fsync: that would cost ms on
    NFS for nothing). The file shares its stem with the process's fatal-signal
    log (crash_handler) so postmortems pair by pid. Truncated on env.reset()
    (current episode only); a clean close() deletes the file — anything left in
    the crashlog dir means an incident. KICAD_CRASH_DIAG=0 disables all writes
    (every method no-ops).

    Slow-step preservation: an episode whose engine dispatch exceeded
    ``KICAD_SLOW_STEP_KEEP_S`` seconds (default 3.0 — above the ~1.1s cost of a
    normal 250-iter-capped shove failure, so the harvest is pathology, not the
    cap floor; 0 disables) is copied to
    ``<stem>_e<id>_steps_slow<dur>s_<k>.jsonl`` at the episode boundary instead
    of being truncated/deleted — a deterministic repro harvest for slow-shove
    engine work. At most ``KICAD_SLOW_STEP_KEEP_MAX`` (default 20) copies per
    env instance so a pathological run cannot flood the disk.
    """

    def __init__(self, env_id: int = 0, log_dir: str | None = None) -> None:
        self._file = None
        self._step = 0
        self._preserve_dur = 0.0
        self._kept = 0
        try:
            self._keep_s = float(os.environ.get("KICAD_SLOW_STEP_KEEP_S", "3.0"))
            self._keep_max = int(os.environ.get("KICAD_SLOW_STEP_KEEP_MAX", "20"))
        except ValueError:
            self._keep_s, self._keep_max = 3.0, 20
        if not diag_enabled():
            self._path = ""
            return
        # <process stem>_e<env instance id>: the process stem pairs this file
        # with the process's fatal-signal/stderr logs; the env-id suffix keeps
        # several envs in one process (list backend, tests) from clobbering
        # one file.
        stem = os.environ.get(ENV_STEM) or artifact_stem("env")
        self._path = os.path.join(
            log_dir or default_log_dir(), f"{stem}_e{env_id}_steps.jsonl"
        )
        self._file = open(self._path, "w")

    def on_reset(self, board_path: str) -> None:
        """Clear log at episode start (preserving a flagged slow episode first)."""
        if self._file is None:
            return
        self._preserve_if_flagged()
        self._file.seek(0)
        self._file.truncate()
        self._step = 0
        self._write({
            "event": "reset",
            "board": board_path,
        })

    def on_pre_step(self, action: dict[str, Any], router_state: dict) -> None:
        """Log action + state BEFORE dispatching to C++ engine."""
        if self._file is None:
            return
        self._step += 1
        self._write({
            "event": "pre_step",
            "step": self._step,
            "action": _safe_serialize(action),
            "router_head": _safe_serialize(router_state),
            "rss_mb": read_rss_mb(),
        })

    def on_post_step(self, success: bool, info: dict[str, Any]) -> None:
        """Log result AFTER C++ engine returns (confirms no crash)."""
        if self._file is None:
            return
        self._write({
            "event": "post_step",
            "step": self._step,
            "success": success,
            "unrouted": info.get("unrouted_count"),
            "drc": info.get("drc_violations"),
        })

    def on_step_time(self, dur_s: float) -> None:
        """Engine latency of the step just logged; flag a slow episode for keep."""
        if self._file is None or self._keep_s <= 0 or dur_s < self._keep_s:
            return
        self._write({"event": "slow_step", "step": self._step, "dur_s": round(dur_s, 3)})
        self._preserve_dur = max(self._preserve_dur, dur_s)

    def _preserve_if_flagged(self) -> None:
        """Copy the current episode's stream aside if a slow step flagged it."""
        if self._preserve_dur <= 0.0:
            return
        if self._kept < self._keep_max:
            try:
                self._file.flush()
                dst = (f"{self._path[:-len('.jsonl')]}"
                       f"_slow{self._preserve_dur:.0f}s_{self._kept}.jsonl")
                shutil.copyfile(self._path, dst)
                self._kept += 1
            except OSError:
                pass  # diagnostics never break the workload
        self._preserve_dur = 0.0

    def close(self) -> None:
        """Clean close: the episode ended without a crash — remove the evidence."""
        if self._file is None or self._file.closed:
            return
        self._preserve_if_flagged()
        self._file.close()
        try:
            os.remove(self._path)
        except OSError:
            pass

    def _write(self, data: dict) -> None:
        self._file.write(json.dumps(data, default=str) + "\n")
        self._file.flush()


def _safe_serialize(obj: Any) -> Any:
    """Convert non-serializable values for JSON logging."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy array/scalar — duck-typed so this
        return obj.tolist()     # module stays stdlib-only at import time
    return obj
