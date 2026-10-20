"""make_via turns via placement off on EVERY exit, including a refused final FixRoute.

The leak this pins: ``make_via`` turns via placement on, then ``fix_route`` — and when
PNS refuses the final FixRoute (here: MarkObstacles mode, a target the head reaches
but cannot commit at) the session stays alive with the mode still on. The next
``make_line`` from that head would then try to end on a via. Reproduced on the real
engine: simple_routing_board, NET1 pad (10, 10), via target at the board corner (0, 0).
"""
from __future__ import annotations

import os

import pytest

from pcb_world.engine import engine_available
from tests.helpers.engine import open_engine

BOARD = os.path.join(os.path.dirname(__file__), "..", "fixtures", "simple_routing_board.kicad_pcb")
START, TARGET = (10.0, 10.0), (0.0, 0.0)    # a NET1 pad -> the board corner
MODE_MARK_OBSTACLES = 0


@pytest.fixture
def engine():
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")

    e = open_engine(BOARD)
    e.build_connectivity()
    yield e
    if e.is_routing():
        e.cancel_route()
    e.close()


def test_refused_make_via_with_live_session_leaves_via_mode_off(engine):
    """``action.make_via`` itself (the env dispatcher additionally cancels a failed
    action's session; direct callers — search, data generation — do not)."""
    from pcb_world.core.action import make_via

    assert engine.start_route(START[0], START[1], 1)
    vias_before = engine.get_via_count()

    ok, info = make_via(engine, TARGET[0], TARGET[1], MODE_MARK_OBSTACLES)

    # The case under test: the final FixRoute was refused and the session survived it.
    assert not ok and info["routed"] is False and "rejected" not in info
    assert engine.is_routing(), "precondition: the refused FixRoute keeps the session alive"
    assert engine.get_via_count() == vias_before
    # The action owns the mode: it is off again although the session lives on.
    assert engine.is_placing_via() is False


@pytest.mark.parametrize("restore", ["restore", "restore_incremental"])
def test_restore_refuses_a_session_whose_head_left_its_net(engine, restore):
    """After that refused make_via the head sits at the target, off NET1's copper: re-opening
    there would start a session on another net, so restore raises (after the board is restored)
    instead of silently routing the wrong net."""
    from pcb_world.core.action import make_via

    assert engine.start_route(START[0], START[1], 1)
    make_via(engine, TARGET[0], TARGET[1], MODE_MARK_OBSTACLES)
    assert engine.is_routing() and engine.get_current_net_code() == 1
    handle = engine.checkpoint()
    tracks = len(engine.get_tracks())
    engine.cancel_route()
    with pytest.raises(RuntimeError, match="could not re-open the checkpointed routing session"):
        getattr(engine, restore)(handle)
    assert not engine.is_routing(), "no session may be left open on another net"
    assert len(engine.get_tracks()) == tracks
