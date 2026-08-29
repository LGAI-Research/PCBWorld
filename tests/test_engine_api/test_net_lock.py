"""``KiCadEngine.lock_net`` — fix a net's copper as an immovable (solid) obstacle.

Locking sets the BOARD lock flag on a net's tracks/vias/arcs; ``SyncWorld``
marks them ``PNS::MK_LOCKED`` and the shove engine walks around them instead of
pushing them. Used to fix an already-routed net while routing others
(net-subset / staged routing).

Board: ``two_net_multiterm_board`` — NET1 pads on the y≈5-10 band, NET2 on
y≈20-25. NET1 is routed as a straight horizontal track at y=10; a Shove-mode
NET2 route down across it either shoves NET1 aside (unlocked) or walks around it
(locked).
"""

import os

import pytest

from pcb_world.engine.kicad_engine import KiCadEngine

BOARD = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "two_net_multiterm_board.kicad_pcb"
)
NET1 = 1
SHOVE = 1


def _net1_geom(eng):
    return sorted(
        (round(t.x1_mm, 2), round(t.y1_mm, 2), round(t.x2_mm, 2), round(t.y2_mm, 2))
        for t in eng.get_tracks() if t.net_code == NET1
    )


def _route_net1_straight(eng):
    """NET1 as a single straight track (10,10)→(40,10)."""
    eng.set_routing_mode(SHOVE)
    eng.start_route(10, 10, 1)
    eng.fix_route(40, 10)
    eng.build_connectivity()


def _route_net2_across(eng):
    """Shove-mode NET2 route from its (10,20) pad down to (30,5), crossing NET1's
    y=10 track. reject_if_stuck=False so the router commits the shove/walk result."""
    eng.set_routing_mode(SHOVE)
    eng.start_route(10, 20, 1)
    ok = eng.fix_route(30, 5, reject_if_stuck=False)
    eng.build_connectivity()
    return ok


def test_locked_net_is_not_shoved():
    """With NET1 locked, a colliding Shove-mode route leaves NET1's geometry
    untouched (walked around); unlocked, the same route shoves NET1 aside.
    The differential isolates the lock effect."""
    # Control: unlocked → NET1 is shoved (geometry changes).
    eng = KiCadEngine(BOARD)
    _route_net1_straight(eng)
    before = _net1_geom(eng)
    _route_net2_across(eng)
    after_unlocked = _net1_geom(eng)
    eng.close()
    assert after_unlocked != before, "precondition: unlocked NET1 should be shoved"

    # Locked → NET1 stays exactly the straight track.
    eng = KiCadEngine(BOARD)
    _route_net1_straight(eng)
    before = _net1_geom(eng)
    n = eng.lock_net(NET1)
    ok = _route_net2_across(eng)
    after_locked = _net1_geom(eng)
    eng.close()

    assert n == len(before)           # every NET1 segment got the lock flag
    assert after_locked == before     # NET1 not moved by the shove
    assert ok                          # NET2 still routed (walked around)


def test_lock_net_count_and_unlock():
    """lock_net returns the #items flagged; unlock returns the same; a net with
    no copper returns 0."""
    eng = KiCadEngine(BOARD)
    _route_net1_straight(eng)
    n_tracks = len(_net1_geom(eng))

    assert eng.lock_net(NET1) == n_tracks
    assert eng.lock_net(NET1, locked=False) == n_tracks
    # NET2 has no routed copper yet → nothing to lock.
    assert eng.lock_net(2) == 0
    eng.close()
