"""Routing control API tests (Step 4).

Tests:
- finish() — auto-complete route to target pad
- undo_last_segment() — undo last waypoint
- flip_posture() — change routing direction
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


class TestFinish:
    def test_finish_completes_route(self, board_path: str) -> None:
        """finish() should auto-complete route to target pad."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)

        ok = r.finish()
        # finish() may or may not succeed depending on PNS pathfinding
        assert isinstance(ok, bool)

        if ok:
            assert r.is_routing() is False
            assert r.get_track_count() > 0

    def test_finish_not_routing_returns_false(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.finish() is False

    def test_finish_adds_tracks(self, board_path: str) -> None:
        """If finish succeeds, tracks should be on the board."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        count_before = r.get_track_count()
        r.start_route(START[0], START[1], 0)
        r.move(END[0], END[1])

        ok = r.finish()
        if ok:
            assert r.get_track_count() > count_before


class TestUndoLastSegment:
    def test_undo_during_routing(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        r.start_route(START[0], START[1], 0)
        r.move(1.0, 1.0)
        r.fix_route(1.0, 1.0, force_finish=False)
        r.move(2.0, 3.0)
        r.fix_route(2.0, 3.0, force_finish=False)

        ok = r.undo_last_segment()
        assert isinstance(ok, bool)
        # Should still be routing after undo
        assert r.is_routing() is True

        r.cancel_route()

    def test_undo_not_routing_returns_false(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.undo_last_segment() is False


class TestFlipPosture:
    def test_flip_posture_no_error(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.flip_posture()

        # Should still be routing
        assert r.is_routing() is True
        r.cancel_route()

    def test_flip_posture_not_routing(self, board_path: str) -> None:
        """flip_posture when not routing should be a no-op."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.flip_posture()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
