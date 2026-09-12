"""Unit tests for the new C++ observation bindings.

Tests: get_pads(), get_ratsnest(), get_unrouted_count().
"""

from __future__ import annotations

import os

import pytest

import kicad_rl_router as krl

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "simple_obstacle_board.kicad_pcb"
)


@pytest.fixture()
def router() -> krl.RLRouter:
    return krl.RLRouter(BOARD_PATH)


# ------------------------------------------------------------------
# get_pads
# ------------------------------------------------------------------


class TestGetPads:
    def test_returns_list(self, router: krl.RLRouter) -> None:
        pads = router.get_pads()
        assert isinstance(pads, list)

    def test_pad_count(self, router: krl.RLRouter) -> None:
        pads = router.get_pads()
        assert len(pads) == 3  # P1, P2, OBS1

    def test_pad_fields(self, router: krl.RLRouter) -> None:
        pad = router.get_pads()[0]
        for attr in ("x_mm", "y_mm", "width_mm", "height_mm",
                      "layer", "net_code", "net_name", "pad_name",
                      "footprint_ref"):
            assert hasattr(pad, attr)

    def test_p1_location(self, router: krl.RLRouter) -> None:
        pads = {p.footprint_ref: p for p in router.get_pads()}
        p1 = pads["P1"]
        assert abs(p1.x_mm - 0.0) < 0.01
        assert abs(p1.y_mm - 0.0) < 0.01
        assert p1.net_name == "NET1"

    def test_p2_location(self, router: krl.RLRouter) -> None:
        pads = {p.footprint_ref: p for p in router.get_pads()}
        p2 = pads["P2"]
        assert abs(p2.x_mm - 3.0) < 0.01
        assert abs(p2.y_mm - 5.0) < 0.01
        assert p2.net_name == "NET1"

    def test_obstacle_pad(self, router: krl.RLRouter) -> None:
        pads = {p.footprint_ref: p for p in router.get_pads()}
        obs = pads["OBS1"]
        assert abs(obs.x_mm - 2.0) < 0.01
        assert abs(obs.y_mm - 0.0) < 0.01
        assert abs(obs.width_mm - 2.0) < 0.01
        assert abs(obs.height_mm - 2.0) < 0.01
        assert obs.net_name == "NET_OBSTACLE"


# ------------------------------------------------------------------
# get_ratsnest
# ------------------------------------------------------------------


class TestGetRatsnest:
    def test_returns_list(self, router: krl.RLRouter) -> None:
        edges = router.get_ratsnest()
        assert isinstance(edges, list)

    def test_unrouted_board_has_edges(self, router: krl.RLRouter) -> None:
        edges = router.get_ratsnest()
        assert len(edges) >= 1

    def test_edge_fields(self, router: krl.RLRouter) -> None:
        edge = router.get_ratsnest()[0]
        for attr in ("x1_mm", "y1_mm", "x2_mm", "y2_mm", "net_code"):
            assert hasattr(edge, attr)

    def test_edge_connects_p1_p2(self, router: krl.RLRouter) -> None:
        edge = router.get_ratsnest()[0]
        assert edge.net_code == 1
        assert abs(edge.x1_mm - 0.0) < 0.01
        assert abs(edge.y1_mm - 0.0) < 0.01
        assert abs(edge.x2_mm - 3.0) < 0.01
        assert abs(edge.y2_mm - 5.0) < 0.01

    def test_ratsnest_decreases_after_routing(self, router: krl.RLRouter) -> None:
        before = len(router.get_ratsnest())
        router.start_route(0.0, 0.0, 0)
        router.move(-0.5, 2.5)
        router.fix_route(3.0, 5.0, True)
        after = len(router.get_ratsnest())
        assert after < before


# ------------------------------------------------------------------
# get_unrouted_count
# ------------------------------------------------------------------


class TestGetUnroutedCount:
    def test_positive_before_routing(self, router: krl.RLRouter) -> None:
        assert router.get_unrouted_count() > 0

    def test_decreases_after_routing(self, router: krl.RLRouter) -> None:
        before = router.get_unrouted_count()
        router.start_route(0.0, 0.0, 0)
        router.move(-0.5, 2.5)
        router.fix_route(3.0, 5.0, True)
        after = router.get_unrouted_count()
        assert after < before

    def test_zero_when_all_routed(self, router: krl.RLRouter) -> None:
        router.start_route(0.0, 0.0, 0)
        router.move(-0.5, 2.5)
        router.fix_route(3.0, 5.0, True)
        assert router.get_unrouted_count() == 0
