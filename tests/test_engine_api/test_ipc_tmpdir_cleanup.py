"""``/tmp/krl_ipc_*`` socket-dir lifecycle: no litter after a client exits.

Each ``_ServerConn`` creates a ``krl_ipc_*`` tmpdir (socket + server stderr
log). Two cleanup owners cover every exit shape:

- client atexit (``router_client._kill_all_conns``): normal interpreter
  exit tears down live AND parked servers, removing their dirs;
- the server itself: a client that skips atexit (multiprocessing workers
  exit via ``os._exit``; kill -9) drops the socket — the server sees EOF
  and removes its own dir before exiting.

(The crashed-server case — client removes the dir on detection — is pinned
in test_ipc_server_crash.py.) N/A with ``KICAD_ENGINE_IPC=0``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from pcb_world.engine import engine_available
from pcb_world.engine.router_client import ipc_enabled

REPO_ROOT = str(Path(__file__).resolve().parents[2])

pytestmark = [
    pytest.mark.skipif(not engine_available(), reason="C++ router build not present"),
    pytest.mark.skipif(
        not ipc_enabled(),
        reason="KICAD_ENGINE_IPC=0 — no engine server, tmpdir lifecycle N/A",
    ),
]

_CHILD = textwrap.dedent("""
    import os, sys
    from pcb_world.engine.kicad_engine import KiCadEngine
    eng = KiCadEngine("tests/fixtures/simple_routing_board.kicad_pcb")
    conn = eng._r._conn
    print("TMPDIR=" + conn.tmpdir)
    print("SRV=%d" % conn.pid)
    sys.stdout.flush()
    eng.close()          # parks the server (router closed, process alive)
    {exit_line}
""")


def _run_client(exit_line: str) -> tuple[str, int]:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(exit_line=exit_line)],
        env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    fields = dict(
        line.split("=", 1) for line in proc.stdout.split() if "=" in line
    )
    return fields["TMPDIR"], int(fields["SRV"])


def _srv_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_gone(tmpdir: str, srv: int, deadline_s: float = 30.0) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline and (os.path.isdir(tmpdir) or _srv_alive(srv)):
        time.sleep(0.1)
    assert not os.path.isdir(tmpdir), f"krl_ipc litter: {tmpdir}"
    assert not _srv_alive(srv), f"leaked engine server pid {srv}"


def test_parked_server_cleaned_on_normal_exit():
    tmpdir, srv = _run_client("")
    _wait_gone(tmpdir, srv)


def test_server_cleans_dir_when_client_skips_atexit():
    # os._exit(0) skips atexit — exactly how multiprocessing workers exit.
    # The server must notice the EOF, remove its dir, and die.
    tmpdir, srv = _run_client("os._exit(0)")
    _wait_gone(tmpdir, srv)
