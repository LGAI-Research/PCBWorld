"""Engine-server crash semantics under engine IPC.

Under IPC a fatal C++ signal kills the dedicated ENGINE SERVER child, not
the client process. These tests force that death (SIGKILL / SIGSEGV) and
pin the contract:

- the next engine call raises ``EngineServerCrashed`` naming the server pid
  (never a hang, a naked OSError, or a silent in-process fallback), and so
  does every call after it (the conn stays crashed);
- the server installs the same ``pcb_world.diag`` fatal-signal hook as any
  worker, so the native-crash artifact lands in the crashlog dir FROM THE
  SERVER PROCESS — with the ``KICAD_CRASH_DIAG=0`` killswitch honored;
- the crashed conn's ``/tmp/krl_ipc_*`` dir is removed by the client.

N/A with ``KICAD_ENGINE_IPC=0`` (no server exists) — skipped there.
"""

from __future__ import annotations

import os
import signal

import pytest

from pcb_world.engine import engine_available, router_client
from pcb_world.engine.kicad_engine import KiCadEngine
from pcb_world.engine.router_client import EngineServerCrashed, ipc_enabled

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"

pytestmark = [
    pytest.mark.skipif(not engine_available(), reason="C++ router build not present"),
    pytest.mark.skipif(
        not ipc_enabled(),
        reason="KICAD_ENGINE_IPC=0 — no engine server, crash semantics N/A",
    ),
]


@pytest.fixture(autouse=True)
def _fresh_server():
    """Drain parked servers so each test's engine spawns a FRESH server
    child that inherits the test's monkeypatched env (crashlog dir,
    killswitch); a parked reuse would carry the env of its original spawn."""
    while router_client._IDLE_SERVERS:
        router_client._IDLE_SERVERS.pop().kill()
    yield


def _kill_server(eng, sig: int) -> int:
    conn = eng._r._conn
    os.kill(conn.pid, sig)
    conn.proc.wait(timeout=30)
    return conn.pid


def test_sigkill_raises_engine_server_crashed_and_cleans_tmpdir():
    eng = KiCadEngine(BOARD)
    try:
        tmpdir = eng._r._conn.tmpdir
        pid = _kill_server(eng, signal.SIGKILL)
        with pytest.raises(EngineServerCrashed) as ei:
            eng.build_connectivity()          # non-getter → guaranteed RPC
        assert str(pid) in str(ei.value)
        # The crashed conn's socket tmpdir is removed on detection.
        assert not os.path.isdir(tmpdir)
        # The conn stays crashed: later ops on the closed socket must ALSO
        # raise EngineServerCrashed (not a naked EBADF OSError), so a broad
        # guard swallowing the first raise cannot turn into a silent limp.
        with pytest.raises(EngineServerCrashed):
            eng.build_connectivity()
    finally:
        eng.close()                           # teardown after crash is clean


def test_sigsegv_server_writes_crashlog(tmp_path, monkeypatch):
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    eng = KiCadEngine(BOARD)                  # fresh server inherits the env
    try:
        pid = _kill_server(eng, signal.SIGSEGV)
        with pytest.raises(EngineServerCrashed):
            eng.build_connectivity()
        logs = [
            p for p in tmp_path.glob("*.log")
            if "engine_server" in p.name and p.name.endswith(f"_{pid}.log")
        ]
        assert len(logs) == 1, list(tmp_path.iterdir())
        text = logs[0].read_text(errors="replace")
        # Native backtrace (crashtrace.c, needs gcc) or faulthandler stack —
        # one of the two is always written (same contract as test_crash_diag).
        assert "fatal signal" in text or "Segmentation" in text
    finally:
        eng.close()


def test_sigsegv_crashlog_killswitch(tmp_path, monkeypatch):
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KICAD_CRASH_DIAG", "0")
    eng = KiCadEngine(BOARD)
    try:
        _kill_server(eng, signal.SIGSEGV)
        with pytest.raises(EngineServerCrashed):   # failure policy unaffected
            eng.build_connectivity()
        assert list(tmp_path.glob("*.log")) == []  # no diag artifact
    finally:
        eng.close()
