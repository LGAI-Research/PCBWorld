"""DRC (Design Rule Check) API tests.

Verifies that the Python API's DRC results match the KiCad GUI's DRC
results for sample_drc_violation.kicad_pcb + sample_drc_violation.kicad_dru.

GUI reference values:
  - Clearance violation: 8 (error)
  - Missing connection between items: 1 (error)
  - Track has unconnected end: 2 (warning)
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

BOARD = {
    "pcb": PROJECT_ROOT / "tests/fixtures/sample_drc_violation.kicad_pcb",
    "dru": PROJECT_ROOT / "tests/fixtures/sample_drc_violation.kicad_dru",
}

# Expected values from the GUI
EXPECTED_VIOLATIONS = {
    "Clearance violation": 8,
    "Missing connection between items": 1,
    "Track has unconnected end": 2,
}
EXPECTED_TOTAL = sum(EXPECTED_VIOLATIONS.values())  # 11
EXPECTED_SEVERITY = {
    0x20: 9,   # error: clearance 8 + missing connection 1
    0x10: 2,   # warning: track unconnected end 2
}


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


@pytest.fixture
def board_path() -> str:
    if not BOARD["pcb"].exists():
        pytest.skip(f"Test board not found: {BOARD['pcb']}")
    return str(BOARD["pcb"])


@pytest.fixture
def dru_path() -> str:
    if not BOARD["dru"].exists():
        pytest.skip(f"DRU file not found: {BOARD['dru']}")
    return str(BOARD["dru"])


class TestDRCMatchesGUI:
    """Verifies that the DRC results for sample_drc_violation match the KiCad GUI."""

    def test_total_violation_count(self, board_path: str, dru_path: str) -> None:
        """Total violation count must match the GUI result (11)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc(dru_path)
        assert len(violations) == EXPECTED_TOTAL, (
            f"Expected {EXPECTED_TOTAL} violations, got {len(violations)}"
        )

    def test_violation_type_distribution(self, board_path: str, dru_path: str) -> None:
        """Per-error-type counts must match the GUI result."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc(dru_path)

        actual_counts = Counter(v.error_type for v in violations)

        for error_type, expected_count in EXPECTED_VIOLATIONS.items():
            actual_count = actual_counts.get(error_type, 0)
            assert actual_count == expected_count, (
                f"{error_type}: expected {expected_count}, got {actual_count}"
            )

        # Fail if an unexpected violation type appears.
        for error_type, count in actual_counts.items():
            assert error_type in EXPECTED_VIOLATIONS, (
                f"Unexpected violation type: {error_type} ({count})"
            )

    def test_severity_distribution(self, board_path: str, dru_path: str) -> None:
        """Per-severity counts must match the GUI result."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc(dru_path)

        actual_sev = Counter(v.severity for v in violations)
        sev_labels = {0x10: "warning", 0x20: "error"}

        for sev_code, expected_count in EXPECTED_SEVERITY.items():
            actual_count = actual_sev.get(sev_code, 0)
            label = sev_labels.get(sev_code, hex(sev_code))
            assert actual_count == expected_count, (
                f"{label}: expected {expected_count}, got {actual_count}"
            )

    def test_clearance_violations_are_errors(self, board_path: str, dru_path: str) -> None:
        """Every Clearance violation must have error (0x20) severity."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc(dru_path)

        clearance = [v for v in violations if v.error_type == "Clearance violation"]
        assert len(clearance) == 8
        for v in clearance:
            assert v.severity == 0x20, (
                f"Clearance violation should be error, got severity={v.severity}"
            )

    def test_dangling_tracks_are_warnings(self, board_path: str, dru_path: str) -> None:
        """Every Track has unconnected end must have warning (0x10) severity."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc(dru_path)

        dangling = [v for v in violations if v.error_type == "Track has unconnected end"]
        assert len(dangling) == 2
        for v in dangling:
            assert v.severity == 0x10, (
                f"Dangling track should be warning, got severity={v.severity}"
            )

    def test_violation_fields_valid(self, board_path: str, dru_path: str) -> None:
        """Every violation's fields must have the correct type and value."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc(dru_path)

        for v in violations:
            assert isinstance(v.error_code, int) and v.error_code > 0
            assert isinstance(v.error_type, str) and len(v.error_type) > 0
            assert isinstance(v.message, str) and len(v.message) > 0
            assert isinstance(v.x_mm, float)
            assert isinstance(v.y_mm, float)
            assert isinstance(v.layer, int)
            assert isinstance(v.net_names, list)
            assert isinstance(v.severity, int) and v.severity in (0x10, 0x20)

    def test_missing_connection_net(self, board_path: str, dru_path: str) -> None:
        """Missing connection must occur on the UCTS2 net (confirmed in the GUI)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc(dru_path)

        missing = [v for v in violations if v.error_type == "Missing connection between items"]
        assert len(missing) == 1
        assert "/UCTS2" in missing[0].net_names


class TestDRCUtils:
    """Verifies that KiCadEngine's DRCUtils caching works correctly."""

    def test_cache_matches_run_result(self, board_path: str, dru_path: str) -> None:
        """run_drc()'s return value must match the DRCUtils cache."""
        from pcb_world.engine.kicad_engine import KiCadEngine
        engine = KiCadEngine(board_path)
        drc_run = engine.run_drc(dru_path)
        drc_cached = engine.drc_helper.get_violations()
        assert drc_run == drc_cached

    def test_violation_count_matches(self, board_path: str, dru_path: str) -> None:
        """get_violation_count() must match the length of the violations list."""
        from pcb_world.engine.kicad_engine import KiCadEngine
        engine = KiCadEngine(board_path)
        engine.run_drc(dru_path)
        assert engine.drc_helper.get_violation_count() == EXPECTED_TOTAL

    def test_violations_by_net(self, board_path: str, dru_path: str) -> None:
        """get_violations_by_net() must correctly group error types by net."""
        from pcb_world.engine.kicad_engine import KiCadEngine
        engine = KiCadEngine(board_path)
        engine.run_drc(dru_path)
        by_net = engine.drc_helper.get_violations_by_net()
        assert isinstance(by_net, dict)
        # The UCTS2 net must have a Missing connection entry.
        assert "/UCTS2" in by_net

    def test_filter_by_severity(self, board_path: str, dru_path: str) -> None:
        """Severity filtering must work correctly."""
        from pcb_world.engine.kicad_engine import KiCadEngine
        engine = KiCadEngine(board_path)
        engine.run_drc(dru_path)
        errors = engine.drc_helper.get_filtered_by_severity(0x20)
        warnings = engine.drc_helper.get_filtered_by_severity(0x10)
        assert len(errors) == EXPECTED_SEVERITY[0x20]
        assert len(warnings) == EXPECTED_SEVERITY[0x10]


class TestDRCConsistency:
    """Verifies consistency and stability of DRC runs."""

    def test_multiple_calls_same_result(self, board_path: str, dru_path: str) -> None:
        """Calling run_drc() multiple times on the same board must give the same result."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        c1 = r.run_drc(dru_path)
        c2 = r.run_drc(dru_path)
        assert len(c1) == len(c2)

    def test_drc_does_not_modify_board(self, board_path: str, dru_path: str) -> None:
        """run_drc() must not change the track count or routing state."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        tracks_before = r.get_track_count()
        is_routing_before = r.is_routing()
        r.run_drc(dru_path)
        assert r.get_track_count() == tracks_before
        assert r.is_routing() == is_routing_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
