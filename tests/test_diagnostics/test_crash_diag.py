"""pcb_world.diag — unified crash diagnostics.

Covers the four artifact writers and their lifecycle contract ("stream ahead,
delete on clean exit"):
- install_crash_handler: fatal-signal log (subprocess smoke + real SIGSEGV)
- dump_context: guard payload roundtrip (.pt + .json, detach, killswitch)
- StderrTail: fd-2 redirect + size watchdog truncation (subprocess)
- write_postmortem + SubprocDecoderVecEnv._recover_worker: parent-side death
  record on SIGKILL (the OOM-killer signature) with unlimited respawn

The real end-to-end segfault sim inside a *decoder worker* stays manual (a
segfaulting pytest worker would trip xdist itself).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _run_child(code: str, tmp_path, expect_signal: int | None = None) -> subprocess.CompletedProcess:
    """Run a python child with the repo on sys.path and diag dir at tmp_path."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["KICAD_CRASH_LOG_DIR"] = str(tmp_path)
    env.pop("KICAD_DIAG_STEM", None)   # pytest process's stem must not leak in
    env.pop("KICAD_DIAG_ROLE", None)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    if expect_signal is not None:
        assert proc.returncode == -expect_signal, (proc.returncode, proc.stderr)
    return proc


# ---------------------------------------------------------------------------
# install_crash_handler
# ---------------------------------------------------------------------------

def test_crash_handler_clean_exit_removes_log(tmp_path):
    _run_child(
        """
        from pcb_world.diag import install_crash_handler
        path, log = install_crash_handler("smoke")
        import os
        assert os.path.exists(path)
        """,
        tmp_path,
    )
    # atexit ran in the (normal) child → empty log removed; only crashtrace.so may remain.
    assert list(tmp_path.glob("*.log")) == []


def test_crash_handler_captures_sigsegv(tmp_path):
    _run_child(
        """
        from pcb_world.diag import install_crash_handler
        install_crash_handler("segv")
        import ctypes
        ctypes.string_at(0)   # null deref → SIGSEGV
        """,
        tmp_path,
        expect_signal=signal.SIGSEGV,
    )
    logs = list(tmp_path.glob("*_segv_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(errors="replace")
    # Either a native backtrace (crashtrace.c, needs gcc) or a faulthandler
    # stack — at least one of the two is always left behind (no gcc ->
    # falls back to faulthandler-only).
    assert "fatal signal" in text or "Segmentation" in text


def test_crash_handler_killswitch(tmp_path):
    proc = _run_child(
        """
        import os
        os.environ["KICAD_CRASH_DIAG"] = "0"
        from pcb_world.diag import install_crash_handler
        assert install_crash_handler("off") is None
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.glob("*.log")) == []


# ---------------------------------------------------------------------------
# dump_context
# ---------------------------------------------------------------------------

def test_dump_context_roundtrip(tmp_path, monkeypatch):
    import numpy as np
    import torch

    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    from pcb_world.diag import dump_context

    grad_t = torch.ones(3, requires_grad=True) * 2.0   # gradient-path tensor
    path = dump_context(
        "unit_test",
        tensor=grad_t,
        array=np.arange(4),
        nested={"t": torch.zeros(2), "s": "x"},
        none_val=None,
    )
    assert path is not None and path.endswith(".pt")
    payload = torch.load(path, weights_only=False)
    assert not payload["tensor"].requires_grad          # confirm detached
    assert payload["tensor"].tolist() == [2.0, 2.0, 2.0]
    assert payload["array"].tolist() == [0, 1, 2, 3]
    assert payload["nested"]["s"] == "x"
    assert payload["none_val"] is None

    summary = json.loads(Path(path[: -len(".pt")] + ".json").read_text())
    assert summary["tag"] == "unit_test"
    assert "Tensor" in summary["keys"]["tensor"]
    assert summary["dropped_keys"] == []


def test_dump_context_killswitch(tmp_path, monkeypatch):
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KICAD_CRASH_DIAG", "0")
    from pcb_world.diag import dump_context

    assert dump_context("off", x=1) is None
    assert os.listdir(tmp_path) == []


def test_guard_fail_dumps_and_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    from pcb_world.diag import guard_fail

    with pytest.raises(RuntimeError) as ei:
        guard_fail("guard_unit", "impossible state — k=1", payload_key=[1, 2, 3])
    msg = str(ei.value)
    assert msg.startswith("impossible state — k=1 dump=")
    assert os.path.exists(msg.split("dump=", 1)[1])


def test_guard_fail_killswitch(tmp_path, monkeypatch):
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KICAD_CRASH_DIAG", "0")
    from pcb_world.diag import guard_fail

    with pytest.raises(RuntimeError) as ei:
        guard_fail("guard_off", "still raises", x=1)
    assert str(ei.value).endswith("dump=None")
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# StderrTail
# ---------------------------------------------------------------------------

def test_stderr_tail_redirect_and_truncate(tmp_path):
    proc = _run_child(
        """
        import os, sys
        from pcb_world.diag import StderrTail, artifact_stem

        stem = artifact_stem("tailtest")
        tail = StderrTail(stem, max_bytes=4096, check_every=1)
        tail.install()
        os.write(2, b"needle-before-truncate\\n")
        for i in range(400):                     # ~10KB > max_bytes
            os.write(2, f"spamline {i}\\n".encode())
            tail.tick()
        os.write(2, b"needle-after-truncate\\n")
        print("PATH=" + tail._path)              # stdout is not redirected
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    tail_path = Path(proc.stdout.strip().split("PATH=")[1])
    # The main file is truncated to stay under max_bytes; the last write survives
    assert tail_path.exists()
    assert tail_path.stat().st_size < 8192
    assert "needle-after-truncate" in tail_path.read_text()
    # The pre-truncation tail snapshot is preserved
    snaps = list(tmp_path.glob("*tailtest*_stderr_tailsnap*.log"))
    assert snaps and "spamline" in snaps[0].read_text()


def test_stderr_tail_close_clean_removes_file(tmp_path):
    proc = _run_child(
        """
        import os
        from pcb_world.diag import StderrTail, artifact_stem

        tail = StderrTail(artifact_stem("tailclean"))
        tail.install()
        os.write(2, b"transient noise\\n")
        tail.close_clean()
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.glob("*tailclean*_stderr.log")) == []


# ---------------------------------------------------------------------------
# write_postmortem (unit) + SIGKILL → _recover_worker (integration)
# ---------------------------------------------------------------------------

def test_write_postmortem_decodes_signals(tmp_path, monkeypatch):
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    from pcb_world.diag import write_postmortem

    fake = SimpleNamespace(exitcode=-9, pid=4242)
    path = write_postmortem("env3", fake, "recv: EOFError",
                            respawn_count=2, board="b.kicad_pcb")
    rec = json.loads(Path(path).read_text())
    assert rec["exitcode"] == -9
    assert "SIGKILL" in rec["signal"] and "OOM" in rec["signal"]
    assert rec["board"] == "b.kicad_pcb"
    assert rec["artifacts_glob"] == "*_4242*"


class _DummyEnv:
    """Minimal env for the decoder-worker loop (no KiCad)."""

    def reset(self):
        return {"ok": 1}, {}

    def step(self, action):
        return {"ok": 1}, 0.0, False, False, {}

    def close(self):
        pass


def _make_dummy_env():
    return _DummyEnv()


def test_sigkill_worker_postmortem_and_respawn(tmp_path, monkeypatch):
    """On the OOM-killer signature (SIGKILL), the parent must leave an
    exitcode -9 postmortem and the fresh worker must work correctly per
    the unlimited-respawn contract."""
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    from pcb_world.vec.backends.subproc import SubprocDecoderVecEnv

    pool = SubprocDecoderVecEnv([_make_dummy_env], start_method="spawn")
    try:
        pid = pool.processes[0].pid
        os.kill(pid, signal.SIGKILL)
        pool.processes[0].join(10)

        obs = pool._recover_worker(0, "test: synthetic SIGKILL")
        assert obs == {"ok": 1}                       # reset works after respawn
        assert pool.respawn_total == 1                # source of diag/worker_respawn_total

        pms = list(tmp_path.glob("*_env0_postmortem.json"))
        assert len(pms) == 1
        rec = json.loads(pms[0].read_text())
        assert rec["exitcode"] == -9
        assert "SIGKILL" in rec["signal"]
        assert rec["pid"] == pid
        assert rec["respawn_count"] == 1
    finally:
        pool.close()
