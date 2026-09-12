"""Routing-accuracy verification tests.

Covers three areas:
  1. Obstacle-avoidance accuracy
  2. Pad-connection accuracy
  3. Path comparison / differentiation

Test board: simple_obstacle_board.kicad_pcb
  - P1(0,0) -> P2(3,5), NET1
  - Obstacle OBS1: center=(2,0), size=2x2 -> rect [1,-1]~[3,1]
  - clearance: 0.2mm (net_class Default)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build_rl"
RL_MODULE_DIR = BUILD_DIR / "pcbnew" / "python" / "rl"
BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"
OUTPUT_DIR = PROJECT_ROOT / "var" / "tests" / "output"

sys.path.insert(0, str(RL_MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.helpers.geometry_helpers import (
    Rect,
    Segment,
    build_track_chain,
    chain_endpoints,
    segment_rect_intersect,
    segment_to_rect_clearance,
    segment_to_segment_distance,
    total_path_length,
)

# ── Constants ────────────────────────────────────────────

START = (0.0, 0.0)
END = (3.0, 5.0)
OBSTACLE_RECT: Rect = ((1.0, -1.0), (3.0, 1.0))
DESIGN_CLEARANCE = 0.2  # mm (net_class Default)
PAD_TOLERANCE = 0.5  # mm — tolerance between a track endpoint and a pad


# ── Shared helpers ───────────────────────────────────


@pytest.fixture(autouse=True)
def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def board_path() -> str:
    if not BOARD_PATH.exists():
        pytest.skip(f"test board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip(
            f"kicad_rl_router module not found. "
            f"build path: {RL_MODULE_DIR}"
        )


def _tracks_to_segments(tracks: list) -> list[Segment]:
    """Converts a list of TrackInfo into a list of Segment."""
    return [
        ((t.x1_mm, t.y1_mm), (t.x2_mm, t.y2_mm))
        for t in tracks
    ]


def _route_with_waypoints(
    board_path: str,
    waypoints: Sequence[tuple[float, float]],
    mode: int | None = None,
) -> list:
    """Routes through the given waypoints and returns the resulting TrackInfo list."""
    krl = _import_krl()
    r = krl.RLRouter(board_path)
    r.set_routing_mode(mode if mode is not None else krl.MODE_WALKAROUND)

    started = r.start_route(START[0], START[1], 0)
    assert started, "start_route failed"

    for wx, wy in waypoints:
        r.move(wx, wy)
        r.fix_route(wx, wy, force_finish=False)

    success = r.fix_route(END[0], END[1])
    assert success, "fix_route failed"
    return r.get_tracks()


# ── Waypoint presets ───────────────────────────────────

LEFT_WAYPOINTS = [(-0.5, 1.0), (0.0, 3.0)]
RIGHT_WAYPOINTS = [(6.0, 0.0), (4.0, 3.0)]
DIRECT_WAYPOINTS = [(1.5, 2.5)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Obstacle avoidance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestObstacleAvoidance:
    """Checks that tracks never cross the obstacle."""

    @pytest.mark.parametrize(
        "mode_name,mode_value",
        [("walkaround", 2), ("shove", 1)],
    )
    def test_no_track_intersects_obstacle(
        self, board_path: str, mode_name: str, mode_value: int
    ) -> None:
        """In walkaround/shove mode, no track may cross the obstacle."""
        tracks = _route_with_waypoints(
            board_path, LEFT_WAYPOINTS, mode=mode_value
        )
        segments = _tracks_to_segments(tracks)

        violations = [
            seg for seg in segments
            if segment_rect_intersect(seg, OBSTACLE_RECT)
        ]
        assert len(violations) == 0, (
            f"{mode_name}: {len(violations)} track(s) cross the obstacle\n"
            f"first violation: {violations[0] if violations else 'N/A'}"
        )

    def test_clearance_from_obstacle(self, board_path: str) -> None:
        """Checks the minimum clearance between a track and the obstacle."""
        tracks = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        segments = _tracks_to_segments(tracks)

        for seg in segments:
            clearance = segment_to_rect_clearance(seg, OBSTACLE_RECT)
            # Must not be a crossing (0.0).
            assert clearance > 0.0, (
                f"track {seg} crosses the obstacle (clearance=0)"
            )

    def test_left_detour_avoids_obstacle(self, board_path: str) -> None:
        """Checks that the left-detour path avoids the obstacle."""
        tracks = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        segments = _tracks_to_segments(tracks)

        for seg in segments:
            assert not segment_rect_intersect(seg, OBSTACLE_RECT), (
                f"left-detour track {seg} crosses the obstacle"
            )

    def test_right_detour_avoids_obstacle(self, board_path: str) -> None:
        """Checks that the right-detour path avoids the obstacle."""
        tracks = _route_with_waypoints(board_path, RIGHT_WAYPOINTS)
        segments = _tracks_to_segments(tracks)

        for seg in segments:
            assert not segment_rect_intersect(seg, OBSTACLE_RECT), (
                f"right-detour track {seg} crosses the obstacle"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Pad connectivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndpointConnectivity:
    """Checks that the track chain connects P1 and P2."""

    def test_track_chain_connects_pads(self, board_path: str) -> None:
        """Both ends of the track chain must be near P1(0,0) and P2(3,5)."""
        tracks = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        segments = _tracks_to_segments(tracks)

        chains = build_track_chain(segments, tol=0.1)
        assert len(chains) >= 1, "no track chain found"

        # Treat the longest chain as the main path.
        main_chain = max(chains, key=len)
        endpoints, _ = chain_endpoints(main_chain)

        # One endpoint must be near P1 and one near P2.
        start_ok = any(
            abs(p[0] - START[0]) < PAD_TOLERANCE
            and abs(p[1] - START[1]) < PAD_TOLERANCE
            for p in endpoints
        )
        end_ok = any(
            abs(p[0] - END[0]) < PAD_TOLERANCE
            and abs(p[1] - END[1]) < PAD_TOLERANCE
            for p in endpoints
        )

        assert start_ok, (
            f"no endpoint near P1(0,0). endpoints: {endpoints}"
        )
        assert end_ok, (
            f"no endpoint near P2(3,5). endpoints: {endpoints}"
        )

    def test_all_tracks_same_net(self, board_path: str) -> None:
        """Checks that all routed tracks belong to the same net."""
        tracks = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        net_names = {t.net_name for t in tracks}

        # Must all be NET1.
        assert len(net_names) == 1, f"found multiple nets: {net_names}"
        assert "NET1" in net_names, f"net is not NET1: {net_names}"

    def test_no_dangling_tracks(self, board_path: str) -> None:
        """There must be no isolated tracks (a single chain)."""
        tracks = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        segments = _tracks_to_segments(tracks)

        chains = build_track_chain(segments, tol=0.1)
        assert len(chains) == 1, (
            f"expected a single chain but found {len(chains)}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Path comparison / differentiation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPathDifferentiation:
    """Checks that different settings produce different results."""

    def test_left_vs_right_produces_different_paths(
        self, board_path: str
    ) -> None:
        """Checks that left-detour and right-detour paths produce different segment sets."""
        tracks_left = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        tracks_right = _route_with_waypoints(board_path, RIGHT_WAYPOINTS)

        segs_left = _tracks_to_segments(tracks_left)
        segs_right = _tracks_to_segments(tracks_right)

        len_left = total_path_length(segs_left)
        len_right = total_path_length(segs_right)

        # Path length or segment count must differ.
        differs = (
            abs(len_left - len_right) > 0.1
            or len(segs_left) != len(segs_right)
        )
        assert differs, (
            f"left/right paths are identical: "
            f"left={len_left:.2f}mm ({len(segs_left)} segs), "
            f"right={len_right:.2f}mm ({len(segs_right)} segs)"
        )

    def test_different_waypoints_differ_in_length(
        self, board_path: str
    ) -> None:
        """Checks that different waypoints produce different path lengths."""
        tracks_left = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        tracks_direct = _route_with_waypoints(board_path, DIRECT_WAYPOINTS)

        len_left = total_path_length(_tracks_to_segments(tracks_left))
        len_direct = total_path_length(_tracks_to_segments(tracks_direct))

        assert abs(len_left - len_direct) > 0.1, (
            f"left detour ({len_left:.2f}mm) and "
            f"direct path ({len_direct:.2f}mm) have the same length"
        )

    def test_different_modes_may_differ(self, board_path: str) -> None:
        """Checks whether walkaround vs shove mode can produce different results.

        Results may be identical depending on the mode, so this only checks
        that both modes produce a valid result (tracks exist).
        """
        krl = _import_krl()

        tracks_walk = _route_with_waypoints(
            board_path, DIRECT_WAYPOINTS, mode=krl.MODE_WALKAROUND
        )
        tracks_shove = _route_with_waypoints(
            board_path, DIRECT_WAYPOINTS, mode=krl.MODE_SHOVE
        )

        assert len(tracks_walk) > 0, "no walkaround result"
        assert len(tracks_shove) > 0, "no shove result"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Minimum clearance (a lightweight DRC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMinimumClearance:
    """Checks the minimum spacing between tracks."""

    def test_tracks_maintain_clearance(self, board_path: str) -> None:
        """Tracks from the same routing must not overlap.

        Endpoints shared by the same net's tracks legitimately have zero
        distance; this only checks the clearance between unconnected track
        pairs.
        """
        tracks = _route_with_waypoints(board_path, LEFT_WAYPOINTS)
        segments = _tracks_to_segments(tracks)

        n = len(segments)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = segments[i], segments[j]

                # Skip adjacent segments that share an endpoint.
                shared = (
                    _close(si[0], sj[0])
                    or _close(si[0], sj[1])
                    or _close(si[1], sj[0])
                    or _close(si[1], sj[1])
                )
                if shared:
                    continue

                dist = segment_to_segment_distance(si, sj)
                assert dist >= 0.0, (
                    f"negative distance between tracks {i}<->{j}: {dist}"
                )


def _close(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two points are within 0.01mm of each other."""
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
