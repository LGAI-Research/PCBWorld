"""Engine-server startup handshake under engine IPC.

The client (``router_client._ServerConn``) polls for the socket path and
connects the instant it appears. The server must therefore publish the path
only once it is listening: a path that exists but is not yet listened on
makes connect() fail with ECONNREFUSED. These tests reproduce the race
deterministically by delaying ``listen()`` inside a real server process
and connecting with no retry the moment the path shows up.

A server whose client never connects (the spawner died or raised first)
must not idle in accept() forever: it exits when its parent changes or when
nobody connects within ``ACCEPT_TIMEOUT_S``, removing its socket dir.

The client, for its part, retries a refused connect within its startup
deadline, so a server that still publishes before listening (an older
engine bundle) cannot take a worker down.

N/A with ``KICAD_ENGINE_IPC=0`` (no server exists) — skipped there.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest

from pcb_world.engine import engine_available, router_client
from pcb_world.engine.router_client import (
    _REPO_ROOT, _SERVER_SCRIPT, ipc_enabled,
)

pytestmark = [
    pytest.mark.skipif(not engine_available(), reason="C++ router build not present"),
    pytest.mark.skipif(
        not ipc_enabled(),
        reason="KICAD_ENGINE_IPC=0 — no engine server, startup handshake N/A",
    ),
]

# Runs the real server module with listen() delayed, so the bind→listen
# window is wide enough for the test client to hit deterministically.
_SLOW_LISTEN_SERVER = textwrap.dedent("""
    import socket, sys, time
    sys.path.insert(0, {server_dir!r})
    import rl_engine_server as srv
    _listen = socket.socket.listen
    def _slow_listen(self, *a):
        time.sleep({delay})
        return _listen(self, *a)
    socket.socket.listen = _slow_listen
    srv.main({sock_path!r})
""")


def _spawn_slow_listen_server(sock_path: str, delay_s: float, crashlog_dir: str):
    env = dict(os.environ)
    env.setdefault("PCBWORLD_KICAD_RL_BUILD_DIR", os.path.join(_REPO_ROOT, "build_rl"))
    env["KICAD_CRASH_LOG_DIR"] = crashlog_dir
    code = _SLOW_LISTEN_SERVER.format(
        server_dir=os.path.dirname(_SERVER_SCRIPT), delay=delay_s, sock_path=sock_path)
    return subprocess.Popen([sys.executable, "-c", code], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _connect_the_instant_path_appears(sock_path: str, proc, timeout_s: float = 120.0):
    """The client's contract, minus any retry: exists → connect, once."""
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(sock_path):
        assert proc.poll() is None, proc.stderr.read().decode()[-2000:]
        assert time.monotonic() < deadline, "server never published its socket"
        time.sleep(0.005)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)          # ECONNREFUSED here == the startup race
    return s


def test_published_socket_is_always_listening(tmp_path):
    tmpdir = tempfile.mkdtemp(prefix="krl_ipc_")     # the client's dir shape
    sock_path = os.path.join(tmpdir, "s.sock")
    proc = _spawn_slow_listen_server(sock_path, delay_s=0.5, crashlog_dir=str(tmp_path))
    try:
        s = _connect_the_instant_path_appears(sock_path, proc)
        s.settimeout(30)
        assert s.recv(4), "server accepted but sent no handshake"
        assert os.listdir(tmpdir) == ["s.sock"], "pending name leaked"
        s.close()                                     # EOF → server exits + cleans dir
        assert proc.wait(timeout=30) == 0
        assert not os.path.isdir(tmpdir), f"krl_ipc litter: {tmpdir}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)


_SHORT_ACCEPT_SERVER = textwrap.dedent("""
    import sys
    sys.path.insert(0, {server_dir!r})
    import rl_engine_server as srv
    srv.ACCEPT_TIMEOUT_S = {timeout}
    srv.main({sock_path!r})
""")

# A client that spawns the real server script and exits before connecting —
# the shape left behind when the client raises inside _ServerConn.__init__.
_DYING_SPAWNER = textwrap.dedent("""
    import os, subprocess, sys
    p = subprocess.Popen([sys.executable, {server!r}, {sock_path!r}, str(os.getpid())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(p.pid, flush=True)
""")


def _server_env(crashlog_dir: str) -> dict:
    env = dict(os.environ)
    env.setdefault("PCBWORLD_KICAD_RL_BUILD_DIR", os.path.join(_REPO_ROOT, "build_rl"))
    env["KICAD_CRASH_LOG_DIR"] = crashlog_dir
    return env


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_pid_gone(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


def test_server_exits_when_spawner_dies_before_connecting(tmp_path):
    tmpdir = tempfile.mkdtemp(prefix="krl_ipc_")
    sock_path = os.path.join(tmpdir, "s.sock")
    code = _DYING_SPAWNER.format(server=_SERVER_SCRIPT, sock_path=sock_path)
    out = subprocess.run([sys.executable, "-c", code], env=_server_env(str(tmp_path)),
                         capture_output=True, text=True, timeout=60, check=True).stdout
    srv_pid = int(out.split()[0])              # spawner already gone: server re-parented
    try:
        # Expected at the first 1 s accept timeout after the import (the
        # parent check fails at once); the bound only covers a cold import.
        assert _wait_pid_gone(srv_pid, timeout_s=180), "orphan server kept running"
        assert not os.path.isdir(tmpdir), f"krl_ipc litter: {tmpdir}"
    finally:
        if _pid_alive(srv_pid):
            os.kill(srv_pid, 9)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_server_exits_when_no_client_connects(tmp_path):
    tmpdir = tempfile.mkdtemp(prefix="krl_ipc_")
    sock_path = os.path.join(tmpdir, "s.sock")
    code = _SHORT_ACCEPT_SERVER.format(
        server_dir=os.path.dirname(_SERVER_SCRIPT), timeout=2.0, sock_path=sock_path)
    proc = subprocess.Popen([sys.executable, "-c", code], env=_server_env(str(tmp_path)),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 180                # bounds a cold import
        while not os.path.exists(sock_path):
            assert proc.poll() is None, proc.stderr.read().decode()[-2000:]
            assert time.monotonic() < deadline
            time.sleep(0.005)
        t0 = time.monotonic()                            # deadline starts after publish
        # We (the parent) stay alive and never connect: only the deadline fires.
        assert proc.wait(timeout=30) == 0
        assert time.monotonic() - t0 >= 2.0
        assert b"none within 2 s" in proc.stderr.read()
        assert not os.path.isdir(tmpdir), f"krl_ipc litter: {tmpdir}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)


# A stand-in server with the OLD publish order (bind, then listen later) and
# a valid handshake — no C++ needed, so it isolates the client's behavior.
_BIND_THEN_LISTEN_SERVER = textwrap.dedent("""
    import os, pickle, socket, struct, sys, time
    sys.path.insert(0, {repo_root!r})
    from pcb_world.engine.containers import KRL_FIELDS
    from pcb_world.engine.wire import KRL_CONSTANT_NAMES
    from pcb_world.engine.router_client import PROTOCOL_VERSION
    sock_path = sys.argv[1]
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)            # the client sees the path now ...
    time.sleep(0.5)                # ... and connect() is refused until here
    srv.listen(1)
    conn, _ = srv.accept()
    data = pickle.dumps({{"protocol": PROTOCOL_VERSION, "schema": KRL_FIELDS,
                          "constants": {{c: 0 for c in KRL_CONSTANT_NAMES}},
                          "pid": os.getpid()}})
    conn.sendall(struct.pack(">Q", len(data)) + data)
    conn.recv(1)                   # EOF when the client kills us
""")


def test_client_retries_refused_connect_until_server_listens(tmp_path, monkeypatch):
    fake = tmp_path / "bind_then_listen_server.py"
    fake.write_text(_BIND_THEN_LISTEN_SERVER.format(repo_root=_REPO_ROOT))
    monkeypatch.setattr(router_client, "_SERVER_SCRIPT", str(fake))
    conn = router_client._ServerConn()       # ConnectionRefusedError == no retry
    try:
        assert conn.pid == conn.proc.pid     # handshake came from that server
    finally:
        conn.kill()
    assert not os.path.isdir(conn.tmpdir)


def test_sigterm_before_publish_leaves_no_pending_file(tmp_path):
    tmpdir = tempfile.mkdtemp(prefix="krl_ipc_")
    sock_path = os.path.join(tmpdir, "s.sock")
    proc = _spawn_slow_listen_server(sock_path, delay_s=3.0, crashlog_dir=str(tmp_path))
    try:
        deadline = time.monotonic() + 120
        while not os.listdir(tmpdir):                 # wait for the bind (pending file)
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.005)
        proc.terminate()                              # SIGTERM inside the bind→rename window
        assert proc.wait(timeout=30) == 0
        assert not os.path.isdir(tmpdir), f"krl_ipc litter: {os.listdir(tmpdir)}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)


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
