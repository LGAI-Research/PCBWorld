"""Engine-server startup under engine IPC: a client that stalls past the server's accept
deadline (``ACCEPT_TIMEOUT_S``) must see the server's stated reason as
``EngineServerCrashed`` — whether it stalls before or after the socket-path check.

Split from ``test_ipc_startup.py`` (each test waits out a real deadline, ~4 s) so the
pytest-xdist loadfile scheduler runs the groups on different workers.
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


def test_stalled_client_sees_the_accept_deadline_reason(tmp_path, monkeypatch):
    """A client that stalls past the server's accept deadline must get the
    server's stated reason, not a bare 'died during startup'."""
    wrapper = tmp_path / "short_accept_server.py"
    wrapper.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.path.dirname(_SERVER_SCRIPT)!r})
        import rl_engine_server as srv
        srv.ACCEPT_TIMEOUT_S = 1.5
        srv.main(sys.argv[1], int(sys.argv[2]))
    """))
    monkeypatch.setattr(router_client, "_SERVER_SCRIPT", str(wrapper))
    real_exists = os.path.exists

    def stalled_exists(p):               # the client's first poll stalls 4 s
        monkeypatch.setattr(os.path, "exists", real_exists)
        time.sleep(4.0)
        return real_exists(p)
    monkeypatch.setattr(os.path, "exists", stalled_exists)
    with pytest.raises(router_client.EngineServerCrashed) as ei:
        router_client._ServerConn()
    assert "before any client connected" in str(ei.value), str(ei.value)


def test_socket_removed_between_exists_and_connect_is_a_server_exit(tmp_path, monkeypatch):
    """Stall after the path check instead: the server's deadline removes the
    socket, connect() hits ENOENT — that must surface as EngineServerCrashed
    (with the reason), not as a raw FileNotFoundError that skips kill()."""
    wrapper = tmp_path / "short_accept_server.py"
    wrapper.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.path.dirname(_SERVER_SCRIPT)!r})
        import rl_engine_server as srv
        srv.ACCEPT_TIMEOUT_S = 1.5
        srv.main(sys.argv[1], int(sys.argv[2]))
    """))
    monkeypatch.setattr(router_client, "_SERVER_SCRIPT", str(wrapper))
    real_exists = os.path.exists

    def stall_after_first_true(p):
        seen = real_exists(p)
        if seen:
            monkeypatch.setattr(os.path, "exists", real_exists)
            time.sleep(4.0)                  # socket vanishes meanwhile
        return seen
    monkeypatch.setattr(os.path, "exists", stall_after_first_true)
    with pytest.raises(router_client.EngineServerCrashed) as ei:
        router_client._ServerConn()
    assert "before any client connected" in str(ei.value), str(ei.value)
