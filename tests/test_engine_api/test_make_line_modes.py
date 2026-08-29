"""Fixes make_line behavior per routing mode — mark_obstacles / walkaround / shove.

Board: tests/fixtures/simple_obstacle_board.kicad_pcb
  - NET1: P1(0,0) -> P2(3,5)
  - NET_OBSTACLE: SMD pad (2,0)

Scenarios
  A. Clean straight path: all three modes succeed and commit the same
     straight line.
  B. A vertical NET_OBSTACLE blocker (2,0)->(2,5) blocks the straight path —
     the point where the per-mode contracts diverge:
     - mark_obstacles: refuses the commit (fix_route=False, board unchanged).
       The RL binding does not expose KiCad's "Allow DRC violations" switch,
       so LINE_PLACER::FixRoute's collision rejection always applies — the
       GUI's commit-through-collisions behavior (highlight collisions + allow
       violations) is not reachable here.
     - walkaround: routes around the blocker (succeeds, blocker unchanged,
       path length > direct).
     - shove: pushes the blocker out of the way (succeeds, blocker moved,
       path stays near-direct).
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"

sys.path.insert(0, str(RL_MODULE_DIR))

START = (0.0, 0.0)
END = (3.0, 5.0)

# Vertical blocker from NET_OBSTACLE pad(2,0) to (2,5).
# The P1->P2 straight line crosses x=2 at y~=3.33, so it must intersect.
OBSTACLE_PAD = (2.0, 0.0)
BLOCKER_END = (2.0, 5.0)

# P1->P2 shortest path length (45 degrees) = 3*sqrt(2) + 2
DIRECT_LEN_MM = 3.0 * math.sqrt(2.0) + 2.0

LAYER_FCU = 0


def _import_krl():
    try:
        import kicad_rl_router as krl

        return krl
    except ImportError:
        pytest.skip(f"kicad_rl_router not found. Build path: {RL_MODULE_DIR}")


@pytest.fixture
def krl():
    return _import_krl()


@pytest.fixture
def router(board_path: str, krl):
    r = krl.RLRouter(board_path)
    r.build_connectivity()
    return r


def _make_line(router, mode: int, p1, p2) -> bool:
    """Runs one start->move->fix cycle. Returns whether fix_route succeeded."""
    router.set_routing_mode(mode)
    if not router.start_route(p1[0], p1[1], LAYER_FCU):
        return False
    router.move(p2[0], p2[1])
    return router.fix_route(p2[0], p2[1], force_finish=True)


def _net_tracks(router, net_name: str):
    return [t for t in router.get_tracks() if t.net_name == net_name]


def _total_len_mm(tracks) -> float:
    return sum(math.hypot(t.x2_mm - t.x1_mm, t.y2_mm - t.y1_mm) for t in tracks)


def _track_coords(tracks):
    return sorted(
        (round(t.x1_mm, 3), round(t.y1_mm, 3), round(t.x2_mm, 3), round(t.y2_mm, 3))
        for t in tracks
    )


def _lay_blocker(router, krl) -> None:
    assert _make_line(router, krl.MODE_WALKAROUND, OBSTACLE_PAD, BLOCKER_END), (
        "failed to route the blocker — check whether the fixture board changed"
    )
    router.build_connectivity()


class TestCleanDirectPath:
    """A. When the straight path is clean, all modes commit the same
    straight line and DRC stays clean, regardless of mode."""

    @pytest.mark.parametrize(
        "mode_attr", ["MODE_MARK_OBSTACLES", "MODE_WALKAROUND", "MODE_SHOVE"]
    )
    def test_direct_route_succeeds(self, router, krl, mode_attr):
        assert _make_line(router, getattr(krl, mode_attr), START, END)
        router.build_connectivity()

        tracks = _net_tracks(router, "NET1")
        assert tracks
        assert _total_len_mm(tracks) == pytest.approx(DIRECT_LEN_MM, abs=0.05)
        assert len(router.run_drc()) == 0


class TestBlockedPath:
    """B. Locks in the diverging per-mode contracts when a blocker blocks the
    straight path."""

    def test_mark_obstacles_refuses_commit(self, router, krl):
        """mark_obstacles refuses the commit rather than committing through
        the blocked straight path.

        walkaround can route around the same blocker (see below), but
        mark_obstacles never attempts a detour — it is rejected by the
        collision check at fix time. The failure is atomic: neither a track
        nor a DRC violation is added.
        """
        _lay_blocker(router, krl)
        base_violations = len(router.run_drc())

        ok = _make_line(router, krl.MODE_MARK_OBSTACLES, START, END)
        router.build_connectivity()

        assert ok is False
        assert _net_tracks(router, "NET1") == []
        assert len(router.run_drc()) == base_violations

    def test_walkaround_detours_leaving_blocker(self, router, krl):
        _lay_blocker(router, krl)
        blocker_before = _track_coords(_net_tracks(router, "NET_OBSTACLE"))

        assert _make_line(router, krl.MODE_WALKAROUND, START, END)
        router.build_connectivity()

        tracks = _net_tracks(router, "NET1")
        assert tracks
        assert _total_len_mm(tracks) > DIRECT_LEN_MM + 0.1
        assert _track_coords(_net_tracks(router, "NET_OBSTACLE")) == blocker_before

    def test_shove_pushes_blocker_and_stays_near_direct(self, router, krl):
        _lay_blocker(router, krl)
        blocker_before = _track_coords(_net_tracks(router, "NET_OBSTACLE"))

        assert _make_line(router, krl.MODE_SHOVE, START, END)
        router.build_connectivity()

        tracks = _net_tracks(router, "NET1")
        assert tracks
        assert _total_len_mm(tracks) == pytest.approx(DIRECT_LEN_MM, abs=0.5)
        assert _track_coords(_net_tracks(router, "NET_OBSTACLE")) != blocker_before
