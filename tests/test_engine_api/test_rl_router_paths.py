"""Tests routing the same goal via different methods.

Test board: simple_obstacle_board.kicad_pcb
- P1 (0,0) -> P2 (3,5) route, NET1
- Obstacle OBS1: 2mm x 2mm rectangle centered at (2,0), [1,-1]~[3,1]

Routing pattern:
  move(wp)  -> fix_route(wp, force_finish=False)   # fix an intermediate waypoint
  move(end) -> fix_route(end)                       # final commit
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build_rl"
RL_MODULE_DIR = BUILD_DIR / "pcbnew" / "python" / "rl"
BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"
OUTPUT_DIR = PROJECT_ROOT / "var" / "tests" / "output"

sys.path.insert(0, str(RL_MODULE_DIR))


@pytest.fixture(autouse=True)
def _ensure_output_dir() -> None:
    """Creates the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def board_path() -> str:
    """Path to the test board."""
    if not BOARD_PATH.exists():
        pytest.skip(f"test board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


def _import_krl():
    """Imports the kicad_rl_router module."""
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip(
            f"kicad_rl_router module not found. "
            f"build path: {RL_MODULE_DIR}"
        )


START = (0.0, 0.0)
END = (3.0, 5.0)
DISTANCE = ((END[0] - START[0]) ** 2 + (END[1] - START[1]) ** 2) ** 0.5


class TestSameGoalDifferentModes:
    """Same goal, different routing modes."""

    @pytest.mark.parametrize(
        "mode_name,mode_value",
        [
            ("walkaround", 2),
            ("shove", 1),
            ("mark_obstacles", 0),
        ],
    )
    def test_routing_mode(
        self, board_path: str, mode_name: str, mode_value: int
    ) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(mode_value)

        started = r.start_route(START[0], START[1], 0)
        assert started, f"start_route failed (mode={mode_name})"

        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)
        success = r.fix_route(END[0], END[1])

        assert success, f"fix_route failed (mode={mode_name})"
        assert r.get_track_count() > 0

        output = OUTPUT_DIR / f"same_goal_{mode_name}.kicad_pcb"
        r.save(str(output))
        assert output.exists()


class TestSameGoalDifferentWaypoints:
    """Same goal, different intermediate waypoints."""

    def test_waypoint_left(self, board_path: str) -> None:
        """Detour to the left (avoiding the obstacle)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        started = r.start_route(START[0], START[1], 0)
        assert started

        r.move(-0.5, 1.0)
        r.fix_route(-0.5, 1.0, force_finish=False)
        r.move(0.0, 3.0)
        r.fix_route(0.0, 3.0, force_finish=False)
        success = r.fix_route(END[0], END[1])

        assert success
        assert r.get_track_count() > 0
        r.save(str(OUTPUT_DIR / "waypoint_left.kicad_pcb"))

    def test_waypoint_right(self, board_path: str) -> None:
        """Detour to the right."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        started = r.start_route(START[0], START[1], 0)
        assert started

        r.move(6.0, 0.0)
        r.fix_route(6.0, 0.0, force_finish=False)
        r.move(4.0, 3.0)
        r.fix_route(4.0, 3.0, force_finish=False)
        success = r.fix_route(END[0], END[1])

        assert success
        assert r.get_track_count() > 0
        r.save(str(OUTPUT_DIR / "waypoint_right.kicad_pcb"))

    def test_waypoint_direct(self, board_path: str) -> None:
        """Near-straight path (minimal detour around the obstacle)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        started = r.start_route(START[0], START[1], 0)
        assert started

        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)
        success = r.fix_route(END[0], END[1])

        assert success
        assert r.get_track_count() > 0
        r.save(str(OUTPUT_DIR / "waypoint_direct.kicad_pcb"))


class TestSameGoalDifferentStepSizes:
    """Same goal, different move() step sizes."""

    @pytest.mark.parametrize("num_steps", [2, 5, 10])
    def test_step_size(self, board_path: str, num_steps: int) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        started = r.start_route(START[0], START[1], 0)
        assert started

        total_steps = num_steps + 1
        for i in range(1, num_steps + 1):
            x = START[0] + (END[0] - START[0]) * i / total_steps
            y = START[1] + (END[1] - START[1]) * i / total_steps
            r.move(x, y)
            r.fix_route(x, y, force_finish=False)

        success = r.fix_route(END[0], END[1])

        assert success
        assert r.get_track_count() > 0
        r.save(str(OUTPUT_DIR / f"step_{num_steps}.kicad_pcb"))


class TestPathComparison:
    """Checks path comparisons — that different settings produce different results."""

    def test_left_vs_right_waypoint_differ(self, board_path: str) -> None:
        """Checks that left/right detour waypoints produce different tracks."""
        krl = _import_krl()

        # Left detour
        r1 = krl.RLRouter(board_path)
        r1.set_routing_mode(krl.MODE_WALKAROUND)
        r1.start_route(START[0], START[1], 0)
        r1.move(-0.5, 1.0)
        r1.fix_route(-0.5, 1.0, force_finish=False)
        r1.move(0.0, 3.0)
        r1.fix_route(0.0, 3.0, force_finish=False)
        r1.fix_route(END[0], END[1])
        tracks_left = r1.get_tracks()

        # Right detour
        r2 = krl.RLRouter(board_path)
        r2.set_routing_mode(krl.MODE_WALKAROUND)
        r2.start_route(START[0], START[1], 0)
        r2.move(6.0, 0.0)
        r2.fix_route(6.0, 0.0, force_finish=False)
        r2.move(4.0, 3.0)
        r2.fix_route(4.0, 3.0, force_finish=False)
        r2.fix_route(END[0], END[1])
        tracks_right = r2.get_tracks()

        assert len(tracks_left) > 0 and len(tracks_right) > 0

        # Coordinates or segment counts must differ.
        coords_left = {
            (t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm) for t in tracks_left
        }
        coords_right = {
            (t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm) for t in tracks_right
        }
        assert coords_left != coords_right, "left/right paths produced identical tracks"

    @pytest.mark.parametrize("num_steps", [2, 10])
    def test_step_count_produces_tracks(
        self, board_path: str, num_steps: int
    ) -> None:
        """Checks that each step count produces valid tracks."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)

        total = num_steps + 1
        for i in range(1, num_steps + 1):
            x = START[0] + (END[0] - START[0]) * i / total
            y = START[1] + (END[1] - START[1]) * i / total
            r.move(x, y)
            r.fix_route(x, y, force_finish=False)

        r.fix_route(END[0], END[1])
        assert r.get_track_count() > 0, f"no result for {num_steps} steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
