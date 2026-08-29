"""Before/After state tests for KiCad RL Router APIs.

For each API, captures router state before and after the call,
verifying that only expected state changes occurred.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build_rl"
RL_MODULE_DIR = BUILD_DIR / "pcbnew" / "python" / "rl"
BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"

sys.path.insert(0, str(RL_MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers.state_capture import (
    assert_idle,
    assert_routing_active,
    assert_state_unchanged,
    assert_tracks_changed,
    capture_state,
    compare_states,
)

START = (0.0, 0.0)
END = (3.0, 5.0)


def _import_krl():
    """Import kicad_rl_router module, skip if unavailable."""
    try:
        import kicad_rl_router as krl

        return krl
    except ImportError:
        pytest.skip(f"kicad_rl_router not found. Build path: {RL_MODULE_DIR}")


@pytest.fixture
def router():
    """Fresh router for each test."""
    if not BOARD_PATH.exists():
        pytest.skip(f"Test board not found: {BOARD_PATH}")
    krl = _import_krl()
    r = krl.RLRouter(str(BOARD_PATH))
    r.build_connectivity()
    return r


def _route_simple(r, krl) -> bool:
    """Route P1->P2. Returns success."""
    r.set_routing_mode(krl.MODE_WALKAROUND)
    r.start_route(START[0], START[1], 0)
    r.move(1.5, 2.5)
    r.fix_route(1.5, 2.5, force_finish=False)
    return r.fix_route(END[0], END[1])


# ──────────────────────────────────────────────
# Configuration APIs
# ──────────────────────────────────────────────


class TestSetRoutingMode:
    """set_routing_mode() before/after state tests."""

    def test_set_mode_walkaround(self, router) -> None:
        before = capture_state(router)
        router.set_routing_mode(2)  # WALKAROUND
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_set_mode_shove(self, router) -> None:
        before = capture_state(router)
        router.set_routing_mode(1)  # SHOVE
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_set_mode_mark_obstacles(self, router) -> None:
        before = capture_state(router)
        router.set_routing_mode(0)  # MARK_OBSTACLES
        after = capture_state(router)
        assert_state_unchanged(before, after)


class TestSetTrackWidth:
    """set_track_width() before/after state tests."""

    def test_set_width_custom(self, router) -> None:
        before = capture_state(router)
        router.set_track_width(0.25)
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_set_width_zero_uses_design_rules(self, router) -> None:
        before = capture_state(router)
        router.set_track_width(0)
        after = capture_state(router)
        assert_state_unchanged(before, after)


class TestSetViaDiameter:
    """set_via_diameter() before/after state tests."""

    def test_set_via_diameter(self, router) -> None:
        before = capture_state(router)
        router.set_via_diameter(0.8)
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_set_via_diameter_small(self, router) -> None:
        before = capture_state(router)
        router.set_via_diameter(0.4)
        after = capture_state(router)
        assert_state_unchanged(before, after)


class TestSetViaDrill:
    """set_via_drill() before/after state tests."""

    def test_set_via_drill(self, router) -> None:
        before = capture_state(router)
        router.set_via_drill(0.4)
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_set_via_drill_small(self, router) -> None:
        before = capture_state(router)
        router.set_via_drill(0.2)
        after = capture_state(router)
        assert_state_unchanged(before, after)


# ──────────────────────────────────────────────
# Routing APIs
# ──────────────────────────────────────────────


class TestStartRoute:
    """start_route() before/after state tests."""

    def test_start_route_changes_state_to_routing(self, router) -> None:
        before = capture_state(router)
        assert_idle(before)
        result = router.start_route(START[0], START[1], 0)
        after = capture_state(router)
        assert result is True
        assert_routing_active(after)
        assert before["track_count"] == after["track_count"]
        router.cancel_route()

    def test_start_route_idle_before_routing_after(self, router) -> None:
        before = capture_state(router)
        assert not before["is_routing"]
        router.start_route(START[0], START[1], 0)
        after = capture_state(router)
        assert after["is_routing"]
        router.cancel_route()

    def test_start_route_does_not_add_tracks(self, router) -> None:
        before = capture_state(router)
        router.start_route(START[0], START[1], 0)
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]
        router.cancel_route()

    def test_start_route_invalid_pos_no_crash(self, router) -> None:
        before = capture_state(router)
        result = router.start_route(100.0, 100.0, 0)
        after = capture_state(router)
        # No crash is the main requirement; routing may or may not start
        assert isinstance(result, bool)
        if result:
            router.cancel_route()


class TestMove:
    """move() before/after state tests."""

    def test_move_during_routing_stays_active(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        before = capture_state(router)
        assert_routing_active(before)
        result = router.move(1.0, 1.0)
        after = capture_state(router)
        assert result is True
        assert_routing_active(after)
        router.cancel_route()

    def test_move_does_not_commit_tracks(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        before = capture_state(router)
        router.move(1.5, 2.5)
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]
        router.cancel_route()

    def test_move_without_routing_no_state_change(self, router) -> None:
        before = capture_state(router)
        router.move(1.0, 2.0)
        after = capture_state(router)
        assert_state_unchanged(before, after)


class TestFixRoute:
    """fix_route() before/after state tests."""

    def test_fix_route_force_finish_adds_tracks(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.move(END[0], END[1])
        before = capture_state(router)
        result = router.fix_route(END[0], END[1], True)
        after = capture_state(router)
        if result:
            assert_idle(after)
            assert after["track_count"] > before["track_count"]

    def test_fix_route_default_finish_adds_tracks(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.move(1.5, 2.5)
        router.fix_route(1.5, 2.5, force_finish=False)
        before = capture_state(router)
        result = router.fix_route(END[0], END[1])
        after = capture_state(router)
        if result:
            assert after["track_count"] >= before["track_count"]

    def test_fix_route_waypoint_keeps_routing_active(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.move(1.0, 1.0)
        router.fix_route(1.0, 1.0, force_finish=False)
        after = capture_state(router)
        assert_routing_active(after)
        router.cancel_route()


class TestCancelRoute:
    """cancel_route() before/after state tests."""

    def test_cancel_restores_idle_state(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.move(1.0, 2.0)
        router.cancel_route()
        after = capture_state(router)
        assert_idle(after)

    def test_cancel_does_not_add_tracks(self, router) -> None:
        initial = capture_state(router)
        router.start_route(START[0], START[1], 0)
        router.move(1.0, 2.0)
        router.cancel_route()
        after = capture_state(router)
        assert initial["track_count"] == after["track_count"]

    def test_cancel_from_idle_no_change(self, router) -> None:
        before = capture_state(router)
        router.cancel_route()
        after = capture_state(router)
        assert_state_unchanged(before, after)


class TestFinish:
    """finish() before/after state tests."""

    def test_finish_autoroutes_and_returns_idle(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        before = capture_state(router)
        result = router.finish()
        after = capture_state(router)
        if result:
            assert after["track_count"] > before["track_count"]
            assert_idle(after)

    def test_finish_without_routing_no_change(self, router) -> None:
        before = capture_state(router)
        router.finish()
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]


# ──────────────────────────────────────────────
# Routing Control
# ──────────────────────────────────────────────


class TestUndoLastSegment:
    """undo_last_segment() before/after state tests."""

    def test_undo_during_routing_stays_active(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.move(1.0, 1.0)
        router.undo_last_segment()
        after = capture_state(router)
        assert_routing_active(after)
        router.cancel_route()

    def test_undo_does_not_commit_tracks(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.move(1.0, 1.0)
        before = capture_state(router)
        router.undo_last_segment()
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]
        router.cancel_route()


class TestFlipPosture:
    """flip_posture() before/after state tests."""

    def test_flip_posture_no_track_commit(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        before = capture_state(router)
        router.flip_posture()
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]
        assert_routing_active(after)
        router.cancel_route()

    def test_flip_posture_stays_routing(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.move(1.0, 1.0)
        before = capture_state(router)
        router.flip_posture()
        after = capture_state(router)
        assert_routing_active(after)
        assert before["track_count"] == after["track_count"]
        router.cancel_route()


class TestToggleVia:
    """toggle_via() before/after state tests."""

    def test_toggle_via_flips_placing_via(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        before_via = router.is_placing_via()
        router.toggle_via()
        after_via = router.is_placing_via()
        assert before_via != after_via
        router.cancel_route()

    def test_toggle_via_twice_restores(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        original = router.is_placing_via()
        router.toggle_via()
        router.toggle_via()
        assert router.is_placing_via() == original
        router.cancel_route()


class TestSwitchLayer:
    """switch_layer() before/after state tests."""

    def test_switch_layer_changes_current_layer(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        assert router.get_current_layer() == 0  # F.Cu
        result = router.switch_layer(31)  # B.Cu
        if result:
            assert router.get_current_layer() == 31
        router.cancel_route()

    def test_switch_layer_stays_routing(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        router.switch_layer(31)
        after = capture_state(router)
        assert_routing_active(after)
        router.cancel_route()


# ──────────────────────────────────────────────
# Track Management
# ──────────────────────────────────────────────


class TestDeleteTrackByIndex:
    """delete_track_by_index() before/after state tests."""

    def test_delete_track_decrements_count(self, router) -> None:
        krl = _import_krl()
        _route_simple(router, krl)
        router.build_connectivity()
        before = capture_state(router)
        assert before["track_count"] > 0
        result = router.delete_track_by_index(0)
        router.build_connectivity()
        after = capture_state(router)
        assert result is True
        assert after["track_count"] == before["track_count"] - 1

    def test_delete_invalid_index_no_change(self, router) -> None:
        before = capture_state(router)
        result = router.delete_track_by_index(999)
        after = capture_state(router)
        assert result is False
        assert_state_unchanged(before, after)


class TestDeleteTrackNear:
    """delete_track_near() before/after state tests."""

    def test_delete_near_existing_track_decrements_count(self, router) -> None:
        krl = _import_krl()
        _route_simple(router, krl)
        router.build_connectivity()
        tracks = router.get_tracks()
        assert len(tracks) > 0
        t = tracks[0]
        before = capture_state(router)
        result = router.delete_track_near(
            t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm, t.layer, t.net_code, 0.1)
        router.build_connectivity()
        after = capture_state(router)
        assert result is True
        assert after["track_count"] < before["track_count"]

    def test_delete_near_no_track_no_change(self, router) -> None:
        before = capture_state(router)
        result = router.delete_track_near(99.0, 99.0, 99.0, 99.0, 0, 1, 0.01)
        after = capture_state(router)
        assert result is False
        assert_state_unchanged(before, after)


# ──────────────────────────────────────────────
# DRC
# ──────────────────────────────────────────────


class TestRunDRC:
    """run_drc() before/after state tests."""

    def test_run_drc_returns_count(self, router) -> None:
        drc = router.run_drc()
        assert isinstance(drc, list)
        assert len(drc) >= 0

    def test_run_drc_does_not_change_tracks_or_routing(self, router) -> None:
        before = capture_state(router)
        router.run_drc()
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]
        assert before["is_routing"] == after["is_routing"]


class TestGetDRCViolations:
    """get_drc_violations() before/after state tests."""

    def test_violations_length_matches_count(self, router) -> None:
        router.run_drc()
        violations = router.get_drc_violations()
        count = router.get_drc_violation_count()
        assert len(violations) == count

    def test_get_violations_does_not_change_state(self, router) -> None:
        router.run_drc()
        before = capture_state(router)
        router.get_drc_violations()
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]
        assert before["is_routing"] == after["is_routing"]


# ──────────────────────────────────────────────
# Connectivity
# ──────────────────────────────────────────────


class TestBuildConnectivity:
    """build_connectivity() before/after state tests."""

    def test_build_connectivity_preserves_track_count(self, router) -> None:
        before = capture_state(router)
        router.build_connectivity()
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]

    def test_build_connectivity_ratsnest_non_negative(self, router) -> None:
        router.build_connectivity()
        after = capture_state(router)
        assert after["ratsnest_count"] >= 0


class TestRecalculateRatsnest:
    """recalculate_ratsnest() before/after state tests."""

    def test_recalculate_ratsnest_non_negative(self, router) -> None:
        router.recalculate_ratsnest()
        after = capture_state(router)
        assert after["ratsnest_count"] >= 0

    def test_recalculate_preserves_track_count(self, router) -> None:
        before = capture_state(router)
        router.recalculate_ratsnest()
        after = capture_state(router)
        assert before["track_count"] == after["track_count"]


# ──────────────────────────────────────────────
# Board Query (read-only)
# ──────────────────────────────────────────────


class TestBoardQueries:
    """Board query APIs should never change router state."""

    def test_get_tracks_readonly(self, router) -> None:
        before = capture_state(router)
        router.get_tracks()
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_get_pads_readonly(self, router) -> None:
        before = capture_state(router)
        pads = router.get_pads()
        after = capture_state(router)
        assert_state_unchanged(before, after)
        assert len(pads) > 0

    def test_get_ratsnest_readonly(self, router) -> None:
        before = capture_state(router)
        router.get_ratsnest()
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_get_board_bbox_positive_dimensions(self, router) -> None:
        bbox = router.get_board_bbox()
        assert bbox.width_mm > 0
        assert bbox.height_mm > 0

    def test_get_copper_layer_count_at_least_two(self, router) -> None:
        count = router.get_copper_layer_count()
        assert count >= 2

    def test_get_track_count_matches_get_tracks(self, router) -> None:
        assert router.get_track_count() == len(router.get_tracks())

    def test_get_unrouted_count_non_negative(self, router) -> None:
        assert router.get_unrouted_count() >= 0


# ──────────────────────────────────────────────
# Drag APIs
# ──────────────────────────────────────────────


class TestDrag:
    """start_drag() / cancel_drag() before/after state tests."""

    def test_start_drag_no_track_graceful(self, router) -> None:
        before = capture_state(router)
        result = router.start_drag(50.0, 50.0, 0, 0x17)
        after = capture_state(router)
        assert isinstance(result, bool)
        if not result:
            assert_idle(after)

    def test_drag_existing_track_toggles_dragging(self, router) -> None:
        krl = _import_krl()
        _route_simple(router, krl)
        router.build_connectivity()
        tracks = router.get_tracks()
        if len(tracks) == 0:
            pytest.skip("No tracks to drag")
        t = tracks[0]
        mid_x = (t.x1_mm + t.x2_mm) / 2
        mid_y = (t.y1_mm + t.y2_mm) / 2
        result = router.start_drag(mid_x, mid_y, 0, 0x17)
        if result:
            assert router.is_dragging()
            router.cancel_drag()
            assert not router.is_dragging()

    def test_cancel_drag_from_idle_no_crash(self, router) -> None:
        before = capture_state(router)
        router.cancel_drag()
        after = capture_state(router)
        assert_idle(after)
        assert before["track_count"] == after["track_count"]


# ──────────────────────────────────────────────
# State & I/O
# ──────────────────────────────────────────────


class TestGetRouterState:
    """get_router_state() before/after state tests."""

    def test_idle_state_is_zero(self, router) -> None:
        state = router.get_router_state()
        assert state == 0  # STATE_IDLE

    def test_routing_state_is_three(self, router) -> None:
        router.start_route(START[0], START[1], 0)
        state = router.get_router_state()
        assert state == 3  # STATE_ROUTE_TRACK
        router.cancel_route()

    def test_get_router_state_readonly(self, router) -> None:
        before = capture_state(router)
        router.get_router_state()
        after = capture_state(router)
        assert_state_unchanged(before, after)


class TestGetFailureReason:
    """get_failure_reason() before/after state tests."""

    def test_no_failure_returns_str(self, router) -> None:
        reason = router.get_failure_reason()
        assert isinstance(reason, str)

    def test_get_failure_reason_readonly(self, router) -> None:
        before = capture_state(router)
        router.get_failure_reason()
        after = capture_state(router)
        assert_state_unchanged(before, after)


class TestSave:
    """save() before/after state tests."""

    def test_save_creates_file(self, router, tmp_path) -> None:
        output = str(tmp_path / "test_output.kicad_pcb")
        router.save(output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_save_does_not_change_router_state(self, router, tmp_path) -> None:
        before = capture_state(router)
        output = str(tmp_path / "save_state_check.kicad_pcb")
        router.save(output)
        after = capture_state(router)
        assert_state_unchanged(before, after)

    def test_save_after_routing_preserves_tracks(self, router, tmp_path) -> None:
        krl = _import_krl()
        _route_simple(router, krl)
        track_count = router.get_track_count()
        output = str(tmp_path / "save_after_route.kicad_pcb")
        router.save(output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0
        assert router.get_track_count() == track_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
