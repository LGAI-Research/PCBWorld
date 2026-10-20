"""Parked-server RSS budget (``KICAD_ENGINE_MAX_RSS_MB``).

The engine server process keeps growing per routed step (the measurement and
the why: ``router_client._parse_rss_budget_mb``), so a server parked and reused
across thousands of boards grows without bound. The client therefore reads the
server's resident set when an engine is released
and kills it instead of parking it once it exceeds the budget. Pinned here:

- over budget: the server is killed at release (never mid-episode), the next
  engine spawns a fresh process (new pid) and the recycle counter advances;
- within budget, or budget off (0): the server is parked and reused (same
  pid), and with the budget off /proc is not even read;
- the env var is parsed strictly — an unparsable value raises, never a
  silent fallback to the default.

N/A with ``KICAD_ENGINE_IPC=0`` (no server exists) — skipped there.
"""

from __future__ import annotations

import pytest

from pcb_world.engine import engine_available, router_client
from pcb_world.engine.kicad_engine import KiCadEngine
from pcb_world.engine.router_client import ipc_enabled
from tests.helpers.engine import open_engine

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"

pytestmark = [
    pytest.mark.skipif(not engine_available(), reason="C++ router build not present"),
    pytest.mark.skipif(
        not ipc_enabled(),
        reason="KICAD_ENGINE_IPC=0 — no engine server, parking N/A",
    ),
]


def _drain_pool() -> None:
    while router_client._IDLE_SERVERS:
        router_client._IDLE_SERVERS.pop().kill()


@pytest.fixture(autouse=True)
def _isolated_pool(monkeypatch):
    """Start from an empty pool with parking on, and put every knob back."""
    _drain_pool()
    monkeypatch.setattr(router_client, "_MAX_IDLE", 2)
    monkeypatch.setattr(router_client, "_MAX_RSS_MB", router_client._MAX_RSS_MB)
    monkeypatch.setattr(router_client, "_RSS_RECYCLES", 0)
    yield
    _drain_pool()


def _server(eng: KiCadEngine) -> router_client._ServerConn:
    return eng._r._conn


def test_server_over_budget_is_killed_not_parked(monkeypatch):
    monkeypatch.setattr(router_client, "_MAX_RSS_MB", 1)   # any live server is > 1 MB
    eng = open_engine(BOARD)
    conn = _server(eng)
    pid = conn.pid
    assert conn.rss_mb() > 1
    eng.close()
    assert conn.proc.poll() is not None                    # killed ...
    assert not router_client._IDLE_SERVERS                 # ... not parked
    assert router_client._RSS_RECYCLES == 1
    eng2 = open_engine(BOARD)
    try:
        assert _server(eng2).pid != pid                    # fresh process
    finally:
        eng2.close()


def test_server_within_budget_is_parked_and_reused(monkeypatch):
    monkeypatch.setattr(router_client, "_MAX_RSS_MB", 1 << 20)   # 1 TB: never exceeded
    eng = open_engine(BOARD)
    pid = _server(eng).pid
    eng.close()
    assert [c.pid for c in router_client._IDLE_SERVERS] == [pid]
    eng2 = open_engine(BOARD)
    try:
        assert _server(eng2).pid == pid
        assert router_client._RSS_RECYCLES == 0
    finally:
        eng2.close()


def test_budget_off_parks_without_reading_rss(monkeypatch):
    monkeypatch.setattr(router_client, "_MAX_RSS_MB", 0)
    monkeypatch.setattr(
        router_client._ServerConn, "rss_mb",
        lambda self: pytest.fail("rss_mb() read with the budget off"))
    eng = open_engine(BOARD)
    pid = _server(eng).pid
    eng.close()
    assert [c.pid for c in router_client._IDLE_SERVERS] == [pid]


def test_budget_setter_and_env_parsing():
    router_client.set_server_rss_budget(512)
    assert router_client._MAX_RSS_MB == 512
    router_client.set_server_rss_budget(-5)
    assert router_client._MAX_RSS_MB == 0
    parse = router_client._parse_rss_budget_mb
    assert parse(None) == 1024
    assert parse("") == 1024
    assert parse("0") == 0
    assert parse("2048") == 2048
    with pytest.raises(ValueError):
        parse("1GB")
    with pytest.raises(ValueError):
        parse("-1")
