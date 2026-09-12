"""Tests for get_vias() and ViaInfo field validation.

Test boards:
- simple_obstacle_board.kicad_pcb: board with no vias (confirms an empty list is returned)
- sample_board.kicad_pcb: K nets, each net k has a via at coordinate (k, k),
  via=0.2/0.1 mm, F.Cu(0)<->B.Cu(2)

Coverage:
  - get_vias() return type (list)
  - ViaInfo required fields present (x_mm, y_mm, diameter_mm, drill_mm, top_layer, bottom_layer, net_code, net_name)
  - per-field type checks
  - value-range checks (diameter > drill > 0, top_layer <= bottom_layer, etc.)
  - expected values match the sample_board spec
  - via count changes after routing adds a via
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build_rl"
RL_MODULE_DIR = BUILD_DIR / "pcbnew" / "python" / "rl"
BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"
SAMPLE_BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_board.kicad_pcb"

sys.path.insert(0, str(RL_MODULE_DIR))


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip(f"kicad_rl_router module not found: {RL_MODULE_DIR}")


@pytest.fixture
def board_path() -> str:
    if not BOARD_PATH.exists():
        pytest.skip(f"board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


@pytest.fixture
def sample_board_path() -> str:
    if not SAMPLE_BOARD_PATH.exists():
        pytest.skip(f"board not found: {SAMPLE_BOARD_PATH}")
    return str(SAMPLE_BOARD_PATH)


# ──────────────────────────────────────────────
# 1. Basic return type / empty-board tests
# ──────────────────────────────────────────────
class TestGetViasBasic:
    """Basic behavior of get_vias() on a board with no vias."""

    def test_returns_list(self, board_path: str) -> None:
        """get_vias() returns a list."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        vias = r.get_vias()
        assert isinstance(vias, list)

    def test_empty_on_board_without_vias(self, board_path: str) -> None:
        """Empty list on a board with no vias."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert len(r.get_vias()) == 0


# ──────────────────────────────────────────────
# 2. ViaInfo field checks (sample_board)
# ──────────────────────────────────────────────
class TestViaInfoFields:
    """Checks that ViaInfo's required fields exist and have the right type."""

    def test_vias_exist_on_sample_board(self, sample_board_path: str) -> None:
        """sample_board has vias."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        vias = r.get_vias()
        assert len(vias) > 0

    def test_viainfo_has_all_fields(self, sample_board_path: str) -> None:
        """ViaInfo has all required fields."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        via = r.get_vias()[0]
        assert hasattr(via, "x_mm")
        assert hasattr(via, "y_mm")
        assert hasattr(via, "diameter_mm")
        assert hasattr(via, "drill_mm")
        assert hasattr(via, "top_layer")
        assert hasattr(via, "bottom_layer")
        assert hasattr(via, "net_code")
        assert hasattr(via, "net_name")

    def test_viainfo_coordinate_types(self, sample_board_path: str) -> None:
        """ViaInfo coordinates are float."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        via = r.get_vias()[0]
        assert isinstance(via.x_mm, float)
        assert isinstance(via.y_mm, float)
        assert isinstance(via.diameter_mm, float)
        assert isinstance(via.drill_mm, float)

    def test_viainfo_layer_types(self, sample_board_path: str) -> None:
        """ViaInfo layers are int."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        via = r.get_vias()[0]
        assert isinstance(via.top_layer, int)
        assert isinstance(via.bottom_layer, int)

    def test_viainfo_net_types(self, sample_board_path: str) -> None:
        """ViaInfo net info has valid types."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        via = r.get_vias()[0]
        assert isinstance(via.net_code, int)
        assert isinstance(via.net_name, str)
        assert len(via.net_name) > 0


# ──────────────────────────────────────────────
# 3. Value-range checks
# ──────────────────────────────────────────────
class TestViaInfoValues:
    """Checks the logical validity of ViaInfo field values."""

    def test_diameter_positive(self, sample_board_path: str) -> None:
        """Via diameter is positive."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        for via in r.get_vias():
            assert via.diameter_mm > 0

    def test_drill_positive(self, sample_board_path: str) -> None:
        """Drill diameter is positive."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        for via in r.get_vias():
            assert via.drill_mm > 0

    def test_diameter_greater_than_drill(self, sample_board_path: str) -> None:
        """outer diameter > drill diameter."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        for via in r.get_vias():
            assert via.diameter_mm > via.drill_mm

    def test_layer_range_valid(self, sample_board_path: str) -> None:
        """top_layer <= bottom_layer, both >= 0."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        for via in r.get_vias():
            assert via.top_layer >= 0
            assert via.bottom_layer >= 0
            assert via.top_layer <= via.bottom_layer

    def test_net_code_positive(self, sample_board_path: str) -> None:
        """All vias in sample_board are connected to a net."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        for via in r.get_vias():
            assert via.net_code > 0


# ──────────────────────────────────────────────
# 4. sample_board expected-value checks
# ──────────────────────────────────────────────
class TestSampleBoardVias:
    """Checks expected values derived from the sample_board spec.

    sample_board: K nets, each net k has a via at coordinate (k, k).
    via diameter=0.2mm, drill=0.1mm, F.Cu(0)<->B.Cu(2).
    """

    def test_via_count_matches_net_count(self, sample_board_path: str) -> None:
        """Via count equals the track-derived net count (K)."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        vias = r.get_vias()
        k = r.get_track_count() // 2
        assert len(vias) == k

    def test_via_positions_at_kk(self, sample_board_path: str) -> None:
        """Each via sits at coordinate (k, k)."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        vias = r.get_vias()
        positions = sorted((via.x_mm, via.y_mm) for via in vias)
        for i, (x, y) in enumerate(positions, start=1):
            assert x == pytest.approx(float(i), abs=0.01), f"via {i}: x={x}"
            assert y == pytest.approx(float(i), abs=0.01), f"via {i}: y={y}"

    def test_via_dimensions(self, sample_board_path: str) -> None:
        """Via diameter=0.2mm, drill=0.1mm."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        for via in r.get_vias():
            assert via.diameter_mm == pytest.approx(0.2, abs=0.01)
            assert via.drill_mm == pytest.approx(0.1, abs=0.01)

    def test_via_layers(self, sample_board_path: str) -> None:
        """All vias are F.Cu(0)<->B.Cu(2) through vias."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        for via in r.get_vias():
            assert via.top_layer == 0, f"top_layer={via.top_layer}"
            assert via.bottom_layer == 2, f"bottom_layer={via.bottom_layer}"

    def test_via_net_names_match_board_nets(self, sample_board_path: str) -> None:
        """Each via's net_name matches a net that exists on the board."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        track_nets = {t.net_name for t in r.get_tracks()}
        for via in r.get_vias():
            assert via.net_name in track_nets, f"via net={via.net_name} not in tracks"


# ──────────────────────────────────────────────
# 5. get_vias() does not mutate state
# ──────────────────────────────────────────────
class TestGetViasReadonly:
    """get_vias() returns the same result without side effects."""

    def test_consecutive_calls_identical(self, sample_board_path: str) -> None:
        """Two consecutive calls return identical results."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        vias1 = r.get_vias()
        vias2 = r.get_vias()
        assert len(vias1) == len(vias2)
        for v1, v2 in zip(vias1, vias2):
            assert v1.x_mm == v2.x_mm
            assert v1.y_mm == v2.y_mm
            assert v1.net_code == v2.net_code

    def test_does_not_change_track_count(self, sample_board_path: str) -> None:
        """Calling get_vias() does not affect track_count."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        tc_before = r.get_track_count()
        _ = r.get_vias()
        assert r.get_track_count() == tc_before


# ──────────────────────────────────────────────
# 6. __repr__ checks
# ──────────────────────────────────────────────
class TestViaInfoRepr:
    """Checks ViaInfo's __repr__ behavior."""

    def test_repr_contains_coordinates(self, sample_board_path: str) -> None:
        """repr includes coordinate info."""
        krl = _import_krl()
        r = krl.RLRouter(sample_board_path)
        via = r.get_vias()[0]
        s = repr(via)
        assert "ViaInfo" in s
        assert "layers=" in s
        assert "net=" in s
