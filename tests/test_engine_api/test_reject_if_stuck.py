"""``fix_route(reject_if_stuck=True)`` aborts without a partial stub when stuck.

When the walkaround cannot reach the requested target — the head jams against
the board edge or existing copper — ``reject_if_stuck=True`` recovers the
engine's own reached/stuck verdict (``Placer()->CurrentEnd() != pos``) and
aborts without committing, so no dangling stub is drawn and ``fix_route``
returns False. Mirrors ``ROUTER::Finish``, which commits only on exact arrival.

Fixtures (a raw ``RLRouter`` keeps their pre-routed copper — unlike ``PCBWorld``,
which clears tracks on load):
  * ``donut_cutout_board`` — a rounded-square cutout (arc corners, 0046-like) in
    the middle; a straight top->bottom net jams against the tessellated arc edge
    and cannot detour around it. reject_if_stuck=False leaves a partial stub;
    reject_if_stuck=True commits nothing.
  * ``diagonal_cross_board`` — corner-hugging pads; once one diagonal is routed,
    the other can only be drawn by crossing it (no detour) and is rejected.
  * ``simple_routing_board`` — clear routes, used to prove reachable targets
    (full connections and mid-route waypoints) still commit under the flag.

Regression guard for the make_line stuck-invalidation.
"""

import pytest

from tests.test_engine_api.conftest import FIXTURES_DIR


def _router(board):
    krl = pytest.importorskip("kicad_rl_router")
    path = FIXTURES_DIR / f"{board}.kicad_pcb"
    if not path.exists():
        pytest.skip(f"Board not found: {path}")
    r = krl.RLRouter(str(path), "", 42)
    r.build_connectivity()
    r.set_routing_mode(2)  # walkaround
    return r


def _layer(r, x, y):
    by_xy = {(round(p.x_mm, 2), round(p.y_mm, 2)): p.layer for p in r.get_pads()}
    return by_xy.get((round(x, 2), round(y, 2)), 0)


DONUT_TOP, DONUT_BOT = (20.0, 5.0), (20.0, 35.0)   # jams on the rounded cutout
TL, BR = (2.5, 2.5), (27.5, 27.5)                  # diagonal_cross NET1
TR, BL = (27.5, 2.5), (2.5, 27.5)                  # diagonal_cross NET2


def test_reject_aborts_without_partial_stub():
    """Stuck route -> ok False, no track committed, session recovered."""
    r = _router("donut_cutout_board")
    r.start_route(*DONUT_TOP, _layer(r, *DONUT_TOP))
    tracks_before = r.get_track_count()

    ok = r.fix_route(*DONUT_BOT, True, True)  # force_finish, reject_if_stuck

    assert ok is False
    assert r.get_track_count() == tracks_before   # no partial stub
    assert r.is_routing() is False                # session cleaned up


def test_default_leaves_partial_stub():
    """reject_if_stuck=False (default): legacy commits a stub that connects nothing."""
    r = _router("donut_cutout_board")
    r.start_route(*DONUT_TOP, _layer(r, *DONUT_TOP))
    tracks_before = r.get_track_count()
    unrouted_before = r.get_unrouted_count()

    ok = r.fix_route(*DONUT_BOT, True)  # reject_if_stuck omitted -> False

    assert ok is True
    assert r.get_track_count() > tracks_before        # a dangling stub was drawn
    r.build_connectivity()
    assert r.get_unrouted_count() == unrouted_before  # ...but the net is NOT connected


def test_second_diagonal_cannot_be_drawn():
    """Once one diagonal is routed, the crossing diagonal is rejected (no detour)."""
    r = _router("diagonal_cross_board")
    r.start_route(*TL, _layer(r, *TL))
    assert r.fix_route(*BR, True, True) is True   # first diagonal routes fine
    r.build_connectivity()

    r.start_route(*TR, _layer(r, *TR))
    tracks_before = r.get_track_count()
    ok = r.fix_route(*BL, True, True)             # crossing diagonal -> stuck

    assert ok is False
    assert r.get_track_count() == tracks_before
    assert r.is_routing() is False


def test_reject_allows_reachable_route():
    """A route that actually reaches commits and connects under the flag."""
    r = _router("simple_routing_board")
    rats = r.get_ratsnest()[0]
    layer = _layer(r, rats.x1_mm, rats.y1_mm)
    unrouted_before = r.get_unrouted_count()

    r.start_route(rats.x1_mm, rats.y1_mm, layer)
    ok = r.fix_route(rats.x2_mm, rats.y2_mm, True, True)

    assert ok is True
    r.build_connectivity()
    assert r.get_unrouted_count() < unrouted_before   # connection made


def test_reject_allows_intermediate_waypoint():
    """A reachable free-space waypoint commits even though no net connects yet.

    reject_if_stuck keys on reaching the requested point, NOT on completing a
    net, so a mid-route waypoint (toward a bend, not a pad) is still committed.
    """
    r = _router("simple_routing_board")
    rats = r.get_ratsnest()[0]
    layer = _layer(r, rats.x1_mm, rats.y1_mm)
    mid_x = (rats.x1_mm + rats.x2_mm) / 2.0
    mid_y = (rats.y1_mm + rats.y2_mm) / 2.0

    r.start_route(rats.x1_mm, rats.y1_mm, layer)
    tracks_before = r.get_track_count()
    unrouted_before = r.get_unrouted_count()

    ok = r.fix_route(mid_x, mid_y, True, True)  # midpoint waypoint

    assert ok is True                             # reached the waypoint -> committed
    assert r.get_track_count() > tracks_before
    r.build_connectivity()
    assert r.get_unrouted_count() == unrouted_before  # net still open, stub kept
