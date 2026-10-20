"""Engine-server startup under engine IPC: a server that accepts the client but then dies
or freezes before the handshake must surface as ``EngineServerCrashed`` within the
client's startup deadline, with the client's conn dir removed.

Split from ``test_ipc_startup.py`` so the pytest-xdist loadfile scheduler runs the
groups on different workers.
"""

from __future__ import annotations

import os
import textwrap
import time

import pytest

from pcb_world.engine import engine_available, router_client
from pcb_world.engine.router_client import _SERVER_SCRIPT, ipc_enabled

pytestmark = [
    pytest.mark.skipif(not engine_available(), reason="C++ router build not present"),
    pytest.mark.skipif(
        not ipc_enabled(),
        reason="KICAD_ENGINE_IPC=0 — no engine server, startup handshake N/A",
    ),
]


def _track_conn_dirs(monkeypatch) -> list[str]:
    """Record every conn dir this test's client creates, so the "dir removed"
    assertion looks at the client's own directory instead of a global
    ``/tmp/krl_ipc_*`` snapshot — any concurrent engine spawn (another xdist
    worker, a training job sharing the node) would otherwise fail the test."""
    made: list[str] = []
    real = router_client.tempfile.mkdtemp

    def recording(*a, **k):
        made.append(real(*a, **k))
        return made[-1]

    monkeypatch.setattr(router_client.tempfile, "mkdtemp", recording)
    return made


def test_server_death_after_accept_is_reported_as_crash(tmp_path, monkeypatch):
    """Dies between accept() and the handshake: the client must raise
    EngineServerCrashed naming the op (not an AttributeError from a conn
    whose pid/socket fields were not yet populated)."""
    wrapper = tmp_path / "die_before_handshake_server.py"
    wrapper.write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {os.path.dirname(_SERVER_SCRIPT)!r})
        import rl_engine_server as srv
        srv._send = lambda *a: os._exit(3)     # first send == the handshake
        srv.main(sys.argv[1], int(sys.argv[2]))
    """))
    monkeypatch.setattr(router_client, "_SERVER_SCRIPT", str(wrapper))
    made = _track_conn_dirs(monkeypatch)
    with pytest.raises(router_client.EngineServerCrashed) as ei:
        router_client._ServerConn()
    assert "'<handshake>'" in str(ei.value) and "exit 3" in str(ei.value), str(ei.value)
    assert made and not os.path.exists(made[0]), "conn dir not killed"


def test_server_frozen_after_accept_hits_the_startup_deadline(tmp_path, monkeypatch):
    """Accepts but never sends the handshake (frozen process): the client
    must give up within its startup deadline, not block in recv forever."""
    wrapper = tmp_path / "frozen_after_accept_server.py"
    wrapper.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {os.path.dirname(_SERVER_SCRIPT)!r})
        import rl_engine_server as srv
        srv._send = lambda *a: time.sleep(1000)   # first send == the handshake
        srv.main(sys.argv[1], int(sys.argv[2]))
    """))
    monkeypatch.setattr(router_client, "_SERVER_SCRIPT", str(wrapper))
    monkeypatch.setattr(router_client, "_STARTUP_DEADLINE_S", 3.0)
    made = _track_conn_dirs(monkeypatch)
    t0 = time.monotonic()
    with pytest.raises(router_client.EngineServerCrashed, match="no handshake"):
        router_client._ServerConn()
    assert time.monotonic() - t0 < 30
    assert made and not os.path.exists(made[0]), "conn dir not killed"
