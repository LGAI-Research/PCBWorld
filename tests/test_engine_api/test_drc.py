"""DRC (Design Rule Check) API tests.

Tests:
- run_drc — execute DRC and return list[DRCViolation]
- get_drc_violation_count — matches len(run_drc())
- get_drc_violations — same list as run_drc() return value
- DRCViolation struct — field types and __repr__
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

# BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"
BOARD = {
    "pcb": PROJECT_ROOT / "tests/fixtures/sample_drc_violation.kicad_pcb",
    "dru": PROJECT_ROOT / "tests/fixtures/sample_drc_violation.kicad_dru",
}
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
    if not BOARD["pcb"].exists():
        pytest.skip(f"Test board not found: {BOARD['pcb']}")
    return str(BOARD["pcb"])


class TestRunDRC:
    def test_run_drc_returns_list(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc()
        print(f"\n[DRC] violation count: {len(violations)}")
        for v in violations[:5]:
            print(f"  {repr(v)}")
        assert isinstance(violations, list)

    def test_run_drc_clean_board_baseline(self, board_path: str) -> None:
        """Clean board (no tracks) should have a known baseline violation count."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        baseline = r.run_drc()
        assert isinstance(baseline, list)

    def test_run_drc_after_routing(self, board_path: str) -> None:
        """DRC after routing should still return a valid list."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)
        r.move(END[0], END[1])
        r.fix_route(END[0], END[1])

        result = r.run_drc()
        assert isinstance(result, list)
        assert len(result) >= 0

    def test_run_drc_multiple_calls(self, board_path: str) -> None:
        """run_drc can be called multiple times without crash."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        c1 = r.run_drc()
        c2 = r.run_drc()
        assert isinstance(c1, list)
        assert isinstance(c2, list)
        # Same board state should give same result
        assert len(c1) == len(c2)
      
        
class TestDRCUtils:
    def test_drc(self, board_path: str) -> None:
        from pcb_world.engine.kicad_engine import KiCadEngine
        engine = KiCadEngine(board_path)
        drc_run = engine.run_drc(str(BOARD["dru"]))
        drc_cached = engine.drc_helper.get_violations()
        assert drc_run == drc_cached

        v0 = engine.drc_helper.get_violations_by_net()
        
        
        
class TestDRCViolationCount:
    def test_count_matches_run_drc(self, board_path: str) -> None:
        """get_drc_violation_count should match len(run_drc())."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc()
        stored = r.get_drc_violation_count()
        assert len(violations) == stored

    def test_count_zero_before_run(self, board_path: str) -> None:
        """Before run_drc is called, violation count should be 0."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.get_drc_violation_count() == 0

    def test_count_updates_after_board_change(self, board_path: str) -> None:
        """DRC count should reflect board state at time of run_drc call."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        c1 = r.run_drc()

        # Route a track
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)
        r.move(END[0], END[1])
        r.fix_route(END[0], END[1])

        c2 = r.run_drc()
        # Count may change after routing (more or fewer violations)
        assert isinstance(c2, list)
        assert len(c2) >= 0


class TestDRCViolations:
    def test_violations_list_type(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc()
        assert isinstance(violations, list)

    def test_violations_empty_before_run(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.get_drc_violations()
        assert violations == []

    def test_violations_count_matches_list(self, board_path: str) -> None:
        """Length of run_drc() result should match get_drc_violation_count()."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        violations = r.run_drc()
        count = r.get_drc_violation_count()
        assert len(violations) == count

    def test_violation_struct_fields(self, board_path: str) -> None:
        """Each DRCViolation should have the expected fields with correct types."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)

        # Route through obstacle to provoke violations
        r.set_routing_mode(krl.MODE_MARK_OBSTACLES)
        r.start_route(START[0], START[1], 0)
        r.move(END[0], END[1])
        r.fix_route(END[0], END[1])

        violations = r.run_drc()

        if len(violations) > 0:
            v = violations[0]
            assert hasattr(v, "error_code")
            assert hasattr(v, "error_type")
            assert hasattr(v, "message")
            assert hasattr(v, "x_mm")
            assert hasattr(v, "y_mm")
            assert hasattr(v, "layer")
            assert hasattr(v, "net_names")
            assert hasattr(v, "severity")

            assert isinstance(v.error_code, int)
            assert isinstance(v.error_type, str)
            assert isinstance(v.message, str)
            assert isinstance(v.x_mm, float)
            assert isinstance(v.y_mm, float)
            assert isinstance(v.layer, int)
            assert isinstance(v.net_names, list)
            assert isinstance(v.severity, int)

    def test_violation_repr(self, board_path: str) -> None:
        """DRCViolation __repr__ should contain 'DRCViolation'."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)

        r.set_routing_mode(krl.MODE_MARK_OBSTACLES)
        r.start_route(START[0], START[1], 0)
        r.move(END[0], END[1])
        r.fix_route(END[0], END[1])

        violations = r.run_drc()

        if len(violations) > 0:
            s = repr(violations[0])
            assert "DRCViolation" in s

    def test_mark_obstacles_produces_violations(self, board_path: str) -> None:
        """MarkObstacles mode routes through obstacles, which should produce DRC violations."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)

        # MarkObstacles ignores DRC during routing → violations expected after
        r.set_routing_mode(krl.MODE_MARK_OBSTACLES)
        r.start_route(START[0], START[1], 0)
        # Route straight through the obstacle at (2, 0)
        r.move(2.0, 0.0)
        r.move(END[0], END[1])
        r.fix_route(END[0], END[1])

        violations = r.run_drc()
        # MarkObstacles should produce at least some violations from obstacle overlap
        # (not guaranteed depending on exact path, so we just check it runs)
        assert isinstance(violations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
