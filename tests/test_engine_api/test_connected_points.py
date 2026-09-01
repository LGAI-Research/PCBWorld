"""``KiCadEngine.get_connected_points`` — the connectivity cluster at a point.

Answers "what copper am I already electrically joined to?" from KiCad's
``CONNECTIVITY_DATA`` (whole cluster, transitive), returned as
``(x_mm, y_mm, human_layer)`` anchors: pad / via centres and track endpoints,
one entry per copper layer the item spans. The RL candidate filter uses it to
drop redundant-loop targets — see pcb_world/vec/candidate_pool.py.

Board: ``two_net_multiterm_board`` — NET1 has three SMD pads on layer 1:
(10,10), (40,10) and (25,5).

Board: ``layer_sentinels_board`` — drilled copper: thru-hole pad "3" at
(105,105) (net 2, both faces; its SMD sibling "4" at (108,105) is unrouted) and
a lone via at (104,108) (net 3, F.Cu-B.Cu). PNS indexes a drilled item's HOLE
as a separate parentless ITEM that wins the deterministic ``itemAt()``
tie-break inside the drill radius, so these centres must resolve through
``BoardItem()`` (regression: ``Parent()`` is null on a HOLE → ``[]``).
"""

import os

from pcb_world.engine.kicad_engine import KiCadEngine

BOARD = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "two_net_multiterm_board.kicad_pcb"
)
SENTINELS = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "layer_sentinels_board.kicad_pcb"
)
NET1 = 1
SHOVE = 1

P_LEFT = (10.0, 10.0)
P_RIGHT = (40.0, 10.0)
P_LONE = (25.0, 5.0)


def _keys(eng, pt, layer=1):
    return {
        (round(x, 2), round(y, 2), lay)
        for x, y, lay in eng.get_connected_points(pt[0], pt[1], layer)
    }


def test_unrouted_pad_is_its_own_cluster():
    """Before any copper, each pad is alone — so a filter built on this drops
    only the pad itself, never a legitimate target."""
    eng = KiCadEngine(BOARD)
    try:
        eng.build_connectivity()
        for pad in (P_LEFT, P_RIGHT, P_LONE):
            assert _keys(eng, pad) == {(pad[0], pad[1], 1)}
    finally:
        eng.close()


def test_routing_merges_the_two_endpoints_only():
    """Routing (10,10)→(40,10) joins those two pads (plus the track's own
    endpoints); the untouched third pad stays in its own cluster."""
    eng = KiCadEngine(BOARD)
    try:
        eng.build_connectivity()
        eng.set_routing_mode(SHOVE)
        eng.start_route(P_LEFT[0], P_LEFT[1], 1)
        eng.fix_route(P_RIGHT[0], P_RIGHT[1])
        eng.build_connectivity()

        merged = {(P_LEFT[0], P_LEFT[1], 1), (P_RIGHT[0], P_RIGHT[1], 1)}
        assert _keys(eng, P_LEFT) == merged
        assert _keys(eng, P_RIGHT) == merged
        # The lone pad is exactly what the agent still has to reach.
        assert _keys(eng, P_LONE) == {(P_LONE[0], P_LONE[1], 1)}
    finally:
        eng.close()


def test_no_copper_at_point_returns_empty():
    """An empty position has nothing to be connected to — the filter reads this
    as "no restriction" rather than "everything is connected"."""
    eng = KiCadEngine(BOARD)
    try:
        eng.build_connectivity()
        assert eng.get_connected_points(99.0, 99.0, 1) == []
    finally:
        eng.close()


def test_result_is_layer_qualified():
    """Every anchor carries the human layer it lives on, so the candidate
    filter can distinguish stacked-but-unconnected copper instead of matching
    on (x, y) alone."""
    eng = KiCadEngine(BOARD)
    try:
        eng.build_connectivity()
        pts = eng.get_connected_points(P_LEFT[0], P_LEFT[1], 1)
        assert pts and all(isinstance(lay, int) and lay >= 1 for _x, _y, lay in pts)
    finally:
        eng.close()


def test_thru_hole_pad_centre_resolves_to_the_pad():
    """The drill centre of a thru-hole pad sits on its HOLE item, whose
    ``Parent()`` is null; the query must still report the pad (both copper
    faces) and nothing else — the unrouted SMD sibling stays separate."""
    eng = KiCadEngine(SENTINELS)
    try:
        eng.build_connectivity()
        both = {(105.0, 105.0, 1), (105.0, 105.0, 2)}
        assert _keys(eng, (105.0, 105.0), layer=1) == both
        assert _keys(eng, (105.0, 105.0), layer=2) == both
        assert _keys(eng, (108.0, 105.0), layer=1) == {(108.0, 105.0, 1)}
    finally:
        eng.close()


def test_via_centre_resolves_to_the_via():
    """Same HOLE trap for a via: its centre must report the via on every layer
    it spans (a lone via is its own cluster)."""
    eng = KiCadEngine(SENTINELS)
    try:
        eng.build_connectivity()
        both = {(104.0, 108.0, 1), (104.0, 108.0, 2)}
        assert _keys(eng, (104.0, 108.0), layer=1) == both
        assert _keys(eng, (104.0, 108.0), layer=2) == both
    finally:
        eng.close()
