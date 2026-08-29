"""Unit tests for the geometry_helpers module.

Runs without a build of kicad_rl_router.so.
"""

import math

import pytest

from tests.helpers.geometry_helpers import (
    build_track_chain,
    chain_endpoints,
    point_to_segment_distance,
    segment_rect_intersect,
    segment_to_rect_clearance,
    segment_to_segment_distance,
    segments_intersect,
    total_path_length,
)

# ── Obstacle reference from simple_obstacle_board ───────────────
# OBS1: center=(2,0), size=2x2 -> rect [1,-1]~[3,1]
OBSTACLE_RECT = ((1.0, -1.0), (3.0, 1.0))


# ── TestSegmentsIntersect ──────────────────────────


class TestSegmentsIntersect:
    """Segment-intersection tests."""

    def test_x_cross(self) -> None:
        """An X-shaped crossing."""
        s1 = ((0.0, 0.0), (2.0, 2.0))
        s2 = ((0.0, 2.0), (2.0, 0.0))
        assert segments_intersect(s1, s2) is True

    def test_parallel_no_overlap(self) -> None:
        """Parallel, non-overlapping segments."""
        s1 = ((0.0, 0.0), (2.0, 0.0))
        s2 = ((0.0, 1.0), (2.0, 1.0))
        assert segments_intersect(s1, s2) is False

    def test_t_touch(self) -> None:
        """A T-shaped touch (an endpoint lies on the other segment)."""
        s1 = ((0.0, 0.0), (2.0, 0.0))
        s2 = ((1.0, 0.0), (1.0, 2.0))
        assert segments_intersect(s1, s2) is True

    def test_no_intersection(self) -> None:
        """Non-intersecting segments."""
        s1 = ((0.0, 0.0), (1.0, 0.0))
        s2 = ((2.0, 1.0), (3.0, 1.0))
        assert segments_intersect(s1, s2) is False

    def test_collinear_overlapping(self) -> None:
        """Collinear, overlapping segments."""
        s1 = ((0.0, 0.0), (2.0, 0.0))
        s2 = ((1.0, 0.0), (3.0, 0.0))
        assert segments_intersect(s1, s2) is True

    def test_collinear_disjoint(self) -> None:
        """Collinear but disjoint segments."""
        s1 = ((0.0, 0.0), (1.0, 0.0))
        s2 = ((2.0, 0.0), (3.0, 0.0))
        assert segments_intersect(s1, s2) is False


# ── TestSegmentRectIntersect ───────────────────────


class TestSegmentRectIntersect:
    """Segment-rectangle intersection tests (obstacle-crossing checks)."""

    def test_segment_through_obstacle(self) -> None:
        """A segment that crosses straight through the obstacle."""
        seg = ((0.0, 0.0), (4.0, 0.0))  # horizontal line at y=0, crosses obstacle [1,-1]~[3,1]
        assert segment_rect_intersect(seg, OBSTACLE_RECT) is True

    def test_segment_avoids_obstacle_left(self) -> None:
        """A segment that detours to the left of the obstacle."""
        seg = ((0.0, 0.0), (0.5, 2.0))  # left of the obstacle
        assert segment_rect_intersect(seg, OBSTACLE_RECT) is False

    def test_segment_inside_obstacle(self) -> None:
        """A segment fully inside the obstacle."""
        seg = ((1.5, -0.5), (2.5, 0.5))
        assert segment_rect_intersect(seg, OBSTACLE_RECT) is True

    def test_segment_above_obstacle(self) -> None:
        """A segment that passes above the obstacle."""
        seg = ((0.0, 2.0), (4.0, 2.0))  # y=2, obstacle ymax=1
        assert segment_rect_intersect(seg, OBSTACLE_RECT) is False


# ── TestPointToSegmentDistance ──────────────────────


class TestPointToSegmentDistance:
    """Point-to-segment distance tests."""

    def test_perpendicular_foot(self) -> None:
        """The perpendicular foot lies on the segment."""
        dist = point_to_segment_distance((1.0, 1.0), ((0.0, 0.0), (2.0, 0.0)))
        assert dist == pytest.approx(1.0, abs=1e-9)

    def test_closest_to_endpoint(self) -> None:
        """The closest point is a segment endpoint."""
        dist = point_to_segment_distance((3.0, 0.0), ((0.0, 0.0), (1.0, 0.0)))
        assert dist == pytest.approx(2.0, abs=1e-9)

    def test_point_on_segment(self) -> None:
        """Distance is 0 when the point lies on the segment."""
        dist = point_to_segment_distance((0.5, 0.0), ((0.0, 0.0), (1.0, 0.0)))
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_segment(self) -> None:
        """A degenerate segment (length 0)."""
        dist = point_to_segment_distance((3.0, 4.0), ((0.0, 0.0), (0.0, 0.0)))
        assert dist == pytest.approx(5.0, abs=1e-9)


# ── TestBuildTrackChain ────────────────────────────


class TestBuildTrackChain:
    """Track-chain construction tests."""

    def test_single_chain(self) -> None:
        """Connected segments form a single chain."""
        tracks = [
            ((0.0, 0.0), (1.0, 0.0)),
            ((1.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (2.0, 1.0)),
        ]
        chains = build_track_chain(tracks)
        assert len(chains) == 1
        assert len(chains[0]) == 3

    def test_two_separate_chains(self) -> None:
        """Two disconnected chains."""
        tracks = [
            ((0.0, 0.0), (1.0, 0.0)),
            ((1.0, 0.0), (2.0, 0.0)),
            ((10.0, 10.0), (11.0, 10.0)),
        ]
        chains = build_track_chain(tracks)
        assert len(chains) == 2

    def test_empty_input(self) -> None:
        """Empty input."""
        assert build_track_chain([]) == []


# ── TestChainEndpoints ─────────────────────────────


class TestChainEndpoints:
    """Chain-endpoint extraction tests."""

    def test_linear_chain_has_two_endpoints(self) -> None:
        """A linear chain has two endpoints."""
        chain = [
            ((0.0, 0.0), (1.0, 0.0)),
            ((1.0, 0.0), (2.0, 0.0)),
        ]
        endpoints, _ = chain_endpoints(chain)
        assert len(endpoints) == 2


# ── TestTotalPathLength ────────────────────────────


class TestTotalPathLength:
    """Total path length tests."""

    def test_simple_path(self) -> None:
        """Length of a simple path."""
        tracks = [
            ((0.0, 0.0), (3.0, 0.0)),
            ((3.0, 0.0), (3.0, 4.0)),
        ]
        assert total_path_length(tracks) == pytest.approx(7.0, abs=1e-9)

    def test_diagonal(self) -> None:
        """A diagonal path."""
        tracks = [((0.0, 0.0), (3.0, 4.0))]
        assert total_path_length(tracks) == pytest.approx(5.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
