"""Regression guard: 90-degree corner modes can still get around obstacles.

Stock KiCad (8.0 through 10.0.6, ``pns_walkaround.cpp`` / ``pns_node.cpp``) replaces the
octagonal obstacle hull with its axis-aligned bounding box in the 90-degree corner modes
but never marks that 4-point chain closed. ``SHAPE_LINE_CHAIN::PointInside()`` rejects
open chains, so ``LINE::Walkaround()`` cannot classify any vertex as inside the obstacle
and the walk fails: in MITERED_90 the router routes nothing whenever the direct path
crosses copper, in both walkaround and shove mode. The fork closes those hulls
(``engine/kicad-patches/kicad/pcbnew/router/``); the KiCad 9.0.8 engine fails this test.

Board: net A pads at (105,112) and (120,112) mm, a net B track at x=110 between them.
"""
from __future__ import annotations

import os

import pytest

from pcb_world.engine.kicad_engine import KiCadEngine

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "corner_mode_90_walkaround.kicad_pcb")
START, END = (105.0, 112.0), (120.0, 112.0)
LAYER = 1
OBSTACLE_TRACKS = 1
WALKAROUND, SHOVE = 2, 1
MITERED_45, MITERED_90 = 0, 2


def _route_across(routing_mode: int, corner_mode: int):
    eng = KiCadEngine(FIXTURE, allow_default_rules=True)
    try:
        eng.build_connectivity()
        eng.set_routing_mode(routing_mode)
        eng.set_corner_mode(corner_mode)
        assert eng.start_route(*START, LAYER)
        ok = eng.fix_route(*END, force_finish=True, reject_if_stuck=False)
        tracks = eng.get_tracks()
    finally:
        eng.close()
    return ok, tracks


@pytest.mark.parametrize("routing_mode", [WALKAROUND, SHOVE], ids=["walkaround", "shove"])
def test_90_degree_mode_routes_around_obstacle(routing_mode):
    ok, tracks = _route_across(routing_mode, MITERED_90)
    assert ok, "fix_route failed: the 90-degree hull is open again (walkaround cannot detour)"
    new = [t for t in tracks if not (t.x1_mm == t.x2_mm == 110.0)]
    assert len(new) >= 3, f"expected a detour around the obstacle, got {len(new)} track(s)"
    for t in new:
        assert t.x1_mm == t.x2_mm or t.y1_mm == t.y2_mm, f"diagonal segment in 90-degree mode: {t}"


def test_45_degree_mode_still_routes_around_obstacle():
    ok, tracks = _route_across(WALKAROUND, MITERED_45)
    assert ok
    assert len(tracks) - OBSTACLE_TRACKS >= 3
