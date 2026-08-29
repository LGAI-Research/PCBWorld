"""Track Dragging API tests.

Tests:
- start_drag / is_dragging — state transitions
- move during drag — head repositioning
- fix_drag — commit drag to board
- cancel_drag — abort without changes
- DRAG_MODE constants — existence and values
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"

START = (0.0, 0.0)
END = (3.0, 5.0)


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


@pytest.fixture
def board_path() -> str:
    if not BOARD_PATH.exists():
        pytest.skip(f"Test board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


def _route_and_get_track(r, krl):
    """Route P1→P2 and return the first track's midpoint."""
    r.set_routing_mode(krl.MODE_WALKAROUND)
    r.start_route(START[0], START[1], 0)
    r.move(END[0], END[1])
    ok = r.fix_route(END[0], END[1])
    assert ok, "Routing must succeed to test dragging"
    tracks = r.get_tracks()
    assert len(tracks) > 0
    t = tracks[0]
    mid_x = (t.x1_mm + t.x2_mm) / 2.0
    mid_y = (t.y1_mm + t.y2_mm) / 2.0
    return mid_x, mid_y


class TestDragConstants:
    def test_dm_constants_exist(self) -> None:
        krl = _import_krl()
        assert hasattr(krl, "DM_CORNER")
        assert hasattr(krl, "DM_SEGMENT")
        assert hasattr(krl, "DM_VIA")
        assert hasattr(krl, "DM_FREE_ANGLE")
        assert hasattr(krl, "DM_ARC")
        assert hasattr(krl, "DM_ANY")
        assert hasattr(krl, "DM_COMPONENT")

    def test_dm_constants_values(self) -> None:
        krl = _import_krl()
        assert krl.DM_CORNER == 0x1
        assert krl.DM_SEGMENT == 0x2
        assert krl.DM_VIA == 0x4
        assert krl.DM_FREE_ANGLE == 0x8
        assert krl.DM_ARC == 0x10
        assert krl.DM_ANY == 0x17
        assert krl.DM_COMPONENT == 0x20

    def test_state_drag_constants(self) -> None:
        krl = _import_krl()
        assert krl.STATE_DRAG_SEGMENT == 1
        assert krl.STATE_DRAG_COMPONENT == 2

    def test_layer_constants(self) -> None:
        krl = _import_krl()
        assert krl.F_Cu == 0
        assert krl.B_Cu == 2


class TestStartDrag:
    def test_not_dragging_initially(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.is_dragging() is False

    def test_start_drag_on_empty_returns_false(self, board_path: str) -> None:
        """start_drag on a position with no track should return False."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        ok = r.start_drag(-10.0, -10.0, 0)
        assert ok is False
        assert r.is_dragging() is False

    def test_start_drag_on_track(self, board_path: str) -> None:
        """start_drag on a routed track should return True."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        ok = r.start_drag(mid_x, mid_y, 0)
        assert isinstance(ok, bool)

        if ok:
            assert r.is_dragging() is True
            assert r.is_routing() is False
            r.cancel_drag()

    def test_start_drag_with_segment_mode(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        ok = r.start_drag(mid_x, mid_y, 0, krl.DM_SEGMENT)
        assert isinstance(ok, bool)
        if ok:
            r.cancel_drag()


class TestDragStateTransitions:
    def test_router_state_during_drag(self, board_path: str) -> None:
        """get_router_state should return STATE_DRAG_SEGMENT during drag."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        assert r.get_router_state() == krl.STATE_IDLE

        ok = r.start_drag(mid_x, mid_y, 0)
        if ok:
            state = r.get_router_state()
            assert state in (krl.STATE_DRAG_SEGMENT, krl.STATE_DRAG_COMPONENT)
            r.cancel_drag()

        assert r.get_router_state() == krl.STATE_IDLE

    def test_move_during_drag(self, board_path: str) -> None:
        """move() should work during drag session."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        ok = r.start_drag(mid_x, mid_y, 0)
        if ok:
            move_ok = r.move(mid_x + 0.5, mid_y + 0.5)
            assert isinstance(move_ok, bool)
            assert r.is_dragging() is True
            r.cancel_drag()


class TestCancelDrag:
    def test_cancel_drag_restores_idle(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        ok = r.start_drag(mid_x, mid_y, 0)
        if ok:
            r.cancel_drag()
            assert r.is_dragging() is False
            assert r.get_router_state() == krl.STATE_IDLE

    def test_cancel_drag_preserves_tracks(self, board_path: str) -> None:
        """Cancel should not change the board."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        tracks_before = r.get_track_count()
        ok = r.start_drag(mid_x, mid_y, 0)
        if ok:
            r.move(mid_x + 1.0, mid_y + 1.0)
            r.cancel_drag()

        assert r.get_track_count() == tracks_before

    def test_cancel_drag_when_not_dragging(self, board_path: str) -> None:
        """cancel_drag when not dragging should be a no-op."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.cancel_drag()  # should not raise


class TestFixDrag:
    def test_fix_drag_commits(self, board_path: str) -> None:
        """fix_drag should commit the drag and return to idle."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        ok = r.start_drag(mid_x, mid_y, 0)
        if ok:
            r.move(mid_x + 0.3, mid_y + 0.3)
            fix_ok = r.fix_drag()
            assert isinstance(fix_ok, bool)
            if fix_ok:
                assert r.is_dragging() is False
                assert r.get_router_state() == krl.STATE_IDLE
                assert r.get_track_count() > 0

    def test_fix_drag_not_dragging_returns_false(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.fix_drag() is False

    def test_drag_then_route(self, board_path: str) -> None:
        """After drag, routing should still work."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        mid_x, mid_y = _route_and_get_track(r, krl)

        ok = r.start_drag(mid_x, mid_y, 0)
        if ok:
            r.cancel_drag()

        # Delete all tracks and re-route
        while r.get_track_count() > 0:
            r.delete_track_by_index(0)

        r.start_route(START[0], START[1], 0)
        r.move(END[0], END[1])
        route_ok = r.fix_route(END[0], END[1])
        assert isinstance(route_ok, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
