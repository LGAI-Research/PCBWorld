"""Tests for delete_via_by_index / delete_via_near / get_via_count.

Test boards:
- simple_obstacle_board.kicad_pcb: board with no vias
- sample_board.kicad_pcb: K nets, each net k has a via at coordinate (k, k)

Coverage:
  - get_via_count() return value matches len(get_vias())
  - count decreases after delete_via_by_index() removes a via
  - delete_via_near() removes a via by coordinate
  - out-of-range index/coordinate returns False
  - board is empty after all vias are deleted
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build_rl"
RL_MODULE_DIR = BUILD_DIR / "pcbnew" / "python" / "rl"

sys.path.insert(0, str(RL_MODULE_DIR))


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip(f"kicad_rl_router module not found: {RL_MODULE_DIR}")


# ──────────────────────────────────────────────
# 1. get_via_count basic checks
# ──────────────────────────────────────────────
class TestGetViaCount:
    """Basic behavior of get_via_count()."""

    def test_returns_int(self, sample_board_path: str) -> None:
        """get_via_count() returns an int."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        count = r.get_via_count()
        assert isinstance(count, int)

    def test_matches_get_vias_length(self, sample_board_path: str) -> None:
        """get_via_count() matches len(get_vias())."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        assert r.get_via_count() == len(r.get_vias())

    def test_zero_on_board_without_vias(self, board_path: str) -> None:
        """Returns 0 on a board with no vias."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.get_via_count() == 0


# ──────────────────────────────────────────────
# 2. delete_via_by_index
# ──────────────────────────────────────────────
class TestDeleteViaByIndex:
    """Behavior of delete_via_by_index()."""

    def test_delete_first_via(self, sample_board_path: str) -> None:
        """Deleting the via at index 0 decreases the count by 1."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        before = r.get_via_count()
        assert before > 0, "test board has no vias"
        result = r.delete_via_by_index(0)
        assert result is True
        assert r.get_via_count() == before - 1

    def test_delete_returns_false_for_invalid_index(self, sample_board_path: str) -> None:
        """Returns False for an out-of-range index."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        count = r.get_via_count()
        result = r.delete_via_by_index(count)  # out of range
        assert result is False

    def test_delete_negative_index_returns_false(self, sample_board_path: str) -> None:
        """Returns False for a negative index."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        result = r.delete_via_by_index(-1)
        assert result is False

    def test_delete_on_empty_board(self, board_path: str) -> None:
        """Returns False when deleting on a board with no vias."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        result = r.delete_via_by_index(0)
        assert result is False

    def test_delete_all_vias_by_index(self, sample_board_path: str) -> None:
        """Count reaches 0 after sequentially deleting all vias."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        total = r.get_via_count()
        assert total > 0
        for _ in range(total):
            assert r.delete_via_by_index(0) is True
        assert r.get_via_count() == 0
        assert len(r.get_vias()) == 0


# ──────────────────────────────────────────────
# 3. delete_via_near
# ──────────────────────────────────────────────
class TestDeleteViaNear:
    """Behavior of delete_via_near()."""

    def test_delete_via_at_known_position(self, sample_board_path: str) -> None:
        """Deletes the via at a known coordinate."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        vias = r.get_vias()
        assert len(vias) > 0
        target = vias[0]
        before = r.get_via_count()
        result = r.delete_via_near(target.x_mm, target.y_mm, target.net_code, 0.1)
        assert result is True
        assert r.get_via_count() == before - 1

    def test_delete_returns_false_for_faraway_coords(self, sample_board_path: str) -> None:
        """Returns False for a coordinate outside the board."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        result = r.delete_via_near(99999.0, 99999.0, 1, 0.1)
        assert result is False

    def test_delete_on_empty_board(self, board_path: str) -> None:
        """Returns False when deleting on a board with no vias."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        result = r.delete_via_near(0.0, 0.0, 1, 1.0)
        assert result is False


# ──────────────────────────────────────────────
# 4. Independence from tracks
# ──────────────────────────────────────────────
class TestViaTrackIndependence:
    """Verifies via deletion does not affect tracks."""

    def test_via_delete_does_not_affect_tracks(self, sample_board_path: str) -> None:
        """Track count is unchanged after deleting a via."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        track_count_before = r.get_track_count()
        via_count = r.get_via_count()
        if via_count == 0:
            pytest.skip("board has no vias")
        r.delete_via_by_index(0)
        assert r.get_track_count() == track_count_before
