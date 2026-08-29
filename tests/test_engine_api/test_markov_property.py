"""Markov-property C++ APIs and their integration with obs and reward.

Covers the route_head, current_net_code, routing_target and wip_segments
bindings, and how the reward system consumes them.

Run:
    PYTHONPATH=build_rl/pcbnew/python/rl:. pytest tests/test_engine_api/test_markov_property.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure build path is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_RL = str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl")
if BUILD_RL not in sys.path:
    sys.path.insert(0, BUILD_RL)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.engine import KiCadEngine  # noqa: E402
from pcb_world.core.reward import RewardFunction, RewardState  # noqa: E402

BOARD = str(PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def router() -> KiCadEngine:
    """Singleton router for all tests in this module."""
    return KiCadEngine(BOARD)


def _clean(router: KiCadEngine) -> None:
    """Cancel active sessions and delete all tracks."""
    if router.is_routing():
        router.cancel_route()
    if router.is_dragging():
        router.cancel_drag()
    while router.get_track_count() > 0:
        router.delete_track_by_index(0)
    router.build_connectivity()


# ===========================================================================
# Test Group 1: C++ API — getRouteHead / getCurrentNetCode / getRoutingTarget
# ===========================================================================


class TestRouteHeadAPI:
    """Test get_route_head() C++ binding."""

    def test_idle_returns_invalid(self, router: KiCadEngine) -> None:
        """When not routing, route head should be {0, 0, -1}."""
        _clean(router)
        head = router.get_route_head()
        assert len(head) == 3, f"Expected 3 elements, got {len(head)}"
        assert head[2] < 0, f"Expected layer < 0 (not routing), got {head[2]}"

    def test_routing_returns_valid(self, router: KiCadEngine) -> None:
        """During routing, route head should have valid coords and layer >= 0."""
        _clean(router)
        ok = router.start_route(0.0, 0.0, 1)
        assert ok, "start_route failed"

        head = router.get_route_head()
        assert len(head) == 3
        assert head[2] >= 0, f"Expected valid layer, got {head[2]}"
        # Head should be near start position (0, 0)
        assert abs(head[0]) < 1.0, f"Head X too far from start: {head[0]}"
        assert abs(head[1]) < 1.0, f"Head Y too far from start: {head[1]}"

        router.cancel_route()

    def test_head_moves_with_move(self, router: KiCadEngine) -> None:
        """After move(), route head should update to new position."""
        _clean(router)
        router.start_route(0.0, 0.0, 1)
        router.move(1.5, 2.5)

        head = router.get_route_head()
        assert head[2] >= 0
        # Head should be near (1.5, 2.5) after move
        assert abs(head[0] - 1.5) < 1.0, f"Head X={head[0]}, expected near 1.5"
        assert abs(head[1] - 2.5) < 1.0, f"Head Y={head[1]}, expected near 2.5"

        router.cancel_route()


class TestCurrentNetCodeAPI:
    """Test get_current_net_code() C++ binding."""

    def test_idle_returns_negative(self, router: KiCadEngine) -> None:
        """When not routing, net code should be -1."""
        _clean(router)
        nc = router.get_current_net_code()
        assert nc == -1, f"Expected -1, got {nc}"

    def test_routing_returns_positive(self, router: KiCadEngine) -> None:
        """During routing from a pad, net code should be positive."""
        _clean(router)
        ok = router.start_route(0.0, 0.0, 1)  # P1 is on NET1
        assert ok

        nc = router.get_current_net_code()
        assert nc > 0, f"Expected positive net code, got {nc}"

        router.cancel_route()


class TestRoutingTargetAPI:
    """Test get_routing_target() C++ binding."""

    def test_idle_returns_invalid(self, router: KiCadEngine) -> None:
        """When not routing, target should be {0, 0, -1}."""
        _clean(router)
        target = router.get_routing_target()
        assert len(target) == 3
        assert target[2] < 0

    def test_routing_returns_target_near_p2(self, router: KiCadEngine) -> None:
        """When routing from P1(0,0), target should be near P2(3,5)."""
        _clean(router)
        ok = router.start_route(0.0, 0.0, 1)
        assert ok

        target = router.get_routing_target()
        assert target[2] >= 0, f"Expected valid layer, got {target[2]}"
        # P2 is at (3, 5) — target should be near it
        dist = math.hypot(target[0] - 3.0, target[1] - 5.0)
        assert dist < 2.0, f"Target ({target[0]:.2f}, {target[1]:.2f}) too far from P2(3,5): dist={dist:.2f}"

        router.cancel_route()


class TestWipSegmentsAPI:
    """Test get_wip_segments() C++ binding."""

    def test_idle_returns_empty(self, router: KiCadEngine) -> None:
        """When not routing, WIP segments should be empty."""
        _clean(router)
        # raw binding surface — reaches the router over either transport
        wip = router._r.get_wip_segments()
        assert len(wip) == 0

    def test_routing_with_move_returns_list(self, router: KiCadEngine) -> None:
        """After start_route + move, get_wip_segments returns a list.

        Note: Placer()->Traces() only contains segments committed via
        intermediate fix_route(force_finish=False). Preview segments from
        move() are rendered through Placer()->Head(), which is not exposed.
        So after just move() the list may be empty — that is correct behavior.
        """
        _clean(router)
        router.start_route(0.0, 0.0, 1)
        router.move(1.5, 2.5)

        wip = router._r.get_wip_segments()
        assert isinstance(wip, list), "Should return a list"
        # If segments are present, validate their fields
        for seg in wip:
            assert hasattr(seg, "x1_mm")
            assert hasattr(seg, "y1_mm")
            assert hasattr(seg, "x2_mm")
            assert hasattr(seg, "y2_mm")
            assert seg.net_code > 0

        router.cancel_route()


# ===========================================================================
# Test Group 2: Reward distance shaping
# ===========================================================================


class TestDistanceShaping:
    """Test distance-based reward shaping."""

    def test_distance_helper_idle(self, router: KiCadEngine) -> None:
        """compute_head_target_distance returns None when not routing."""
        _clean(router)
        rstate = router.get_routing_session_state()
        d = RewardFunction.compute_head_target_distance(rstate)
        assert d is None

    def test_distance_helper_routing(self, router: KiCadEngine) -> None:
        """compute_head_target_distance returns positive float when routing."""
        _clean(router)

        router.start_route(0.0, 0.0, 1)
        rstate = router.get_routing_session_state()
        d = RewardFunction.compute_head_target_distance(rstate)
        assert d is not None
        assert d > 0.0, f"Expected positive distance, got {d}"

        router.cancel_route()

    def test_distance_decreases_toward_target(self, router: KiCadEngine) -> None:
        """Moving toward target should decrease distance."""
        _clean(router)

        router.start_route(0.0, 0.0, 1)
        rstate1 = router.get_routing_session_state()
        d1 = RewardFunction.compute_head_target_distance(rstate1)

        # Move toward P2(3,5)
        router.move(1.5, 2.5)
        rstate2 = router.get_routing_session_state()
        d2 = RewardFunction.compute_head_target_distance(rstate2)

        assert d1 is not None and d2 is not None
        assert d2 < d1, f"Distance should decrease: {d1:.3f} → {d2:.3f}"

        router.cancel_route()

    def test_reward_positive_when_closer(self) -> None:
        """Reward should include positive distance shaping when getting closer."""
        rf = RewardFunction(
            completion_weight=0.0,
            wirelength_weight=0.0,
            step_cost=0.0,
            completion_bonus=0.0,
            distance_shaping_weight=1.0,
        )
        before = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)
        after = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)

        r = rf.compute(before, after, prev_distance=5.0, curr_distance=3.0)
        assert r == pytest.approx(2.0), f"Expected +2.0 distance shaping, got {r}"

    def test_reward_negative_when_farther(self) -> None:
        """Reward should include negative distance shaping when getting farther."""
        rf = RewardFunction(
            completion_weight=0.0,
            wirelength_weight=0.0,
            step_cost=0.0,
            completion_bonus=0.0,
            distance_shaping_weight=1.0,
        )
        before = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)
        after = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)

        r = rf.compute(before, after, prev_distance=3.0, curr_distance=5.0)
        assert r == pytest.approx(-2.0), f"Expected -2.0 distance shaping, got {r}"

    def test_reward_no_shaping_when_idle(self) -> None:
        """Distance shaping should be zero when distances are None (IDLE)."""
        rf = RewardFunction(
            completion_weight=0.0,
            wirelength_weight=0.0,
            step_cost=0.0,
            completion_bonus=0.0,
            distance_shaping_weight=1.0,
        )
        before = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)
        after = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)

        r = rf.compute(before, after, prev_distance=None, curr_distance=None)
        assert r == pytest.approx(0.0), f"Expected 0.0 with None distances, got {r}"


