"""Pure-Python geometry utility module.

Geometric validation functions that work without kicad_rl_router.so.
Used for track intersection, obstacle avoidance, and connectivity checks.

Coordinate conventions:
    point = (x, y)  — float tuple
    segment = ((x1, y1), (x2, y2))
    rect = ((x_min, y_min), (x_max, y_max))  — axis-aligned rectangle
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]
Segment = tuple[Point, Point]
Rect = tuple[Point, Point]  # ((xmin, ymin), (xmax, ymax))


# -- Basic geometric operations --------------------------------------


def _cross(o: Point, a: Point, b: Point) -> float:
    """OA x OB cross product (z component)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """Whether q lies on segment pr (assumes collinearity)."""
    return (
        min(p[0], r[0]) <= q[0] + 1e-9 <= max(p[0], r[0]) + 1e-9
        and min(p[1], r[1]) <= q[1] + 1e-9 <= max(p[1], r[1]) + 1e-9
    )


def segments_intersect(s1: Segment, s2: Segment) -> bool:
    """Determine whether two segments intersect (CCW-orientation based).

    Args:
        s1: first segment ((x1,y1),(x2,y2)).
        s2: second segment ((x1,y1),(x2,y2)).

    Returns:
        True if they intersect.
    """
    p1, q1 = s1
    p2, q2 = s2

    d1 = _cross(p2, q2, p1)
    d2 = _cross(p2, q2, q1)
    d3 = _cross(p1, q1, p2)
    d4 = _cross(p1, q1, q2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    eps = 1e-9
    if abs(d1) < eps and _on_segment(p2, p1, q2):
        return True
    if abs(d2) < eps and _on_segment(p2, q1, q2):
        return True
    if abs(d3) < eps and _on_segment(p1, p2, q1):
        return True
    if abs(d4) < eps and _on_segment(p1, q2, q1):
        return True

    return False


# -- Segment <-> rectangle --------------------------------------------


def segment_rect_intersect(seg: Segment, rect: Rect) -> bool:
    """Determine whether a segment passes through an axis-aligned rectangle.

    True on intersection with a rectangle edge, or when the segment lies
    entirely inside the rectangle.

    Args:
        seg: segment ((x1,y1),(x2,y2)).
        rect: rectangle ((xmin,ymin),(xmax,ymax)).

    Returns:
        True if it passes through.
    """
    (xmin, ymin), (xmax, ymax) = rect

    # A segment endpoint inside the rectangle counts as passing through.
    for p in seg:
        if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax:
            return True

    # Check intersection against the rectangle's 4 edges.
    edges: list[Segment] = [
        ((xmin, ymin), (xmax, ymin)),  # bottom
        ((xmax, ymin), (xmax, ymax)),  # right
        ((xmax, ymax), (xmin, ymax)),  # top
        ((xmin, ymax), (xmin, ymin)),  # left
    ]
    return any(segments_intersect(seg, edge) for edge in edges)


def segment_to_rect_clearance(seg: Segment, rect: Rect) -> float:
    """Minimum distance between a segment and an axis-aligned rectangle.

    Returns 0.0 when they intersect.

    Args:
        seg: the segment.
        rect: the rectangle.

    Returns:
        Minimum distance (mm).
    """
    if segment_rect_intersect(seg, rect):
        return 0.0

    (xmin, ymin), (xmax, ymax) = rect
    edges: list[Segment] = [
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    ]
    return min(segment_to_segment_distance(seg, e) for e in edges)


# -- Point <-> segment distance -----------------------------------------


def point_to_segment_distance(p: Point, seg: Segment) -> float:
    """Minimum distance from a point to a segment.

    Args:
        p: point (x, y).
        seg: segment ((x1,y1),(x2,y2)).

    Returns:
        Minimum distance.
    """
    a, b = seg
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    len_sq = dx * dx + dy * dy

    if len_sq < 1e-18:
        # Degenerate segment (a point).
        return math.hypot(p[0] - a[0], p[1] - a[1])

    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq
    t = max(0.0, min(1.0, t))

    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.hypot(p[0] - proj_x, p[1] - proj_y)


# -- Segment <-> segment distance ---------------------------------------


def segment_to_segment_distance(s1: Segment, s2: Segment) -> float:
    """Minimum distance between two segments.

    Args:
        s1: first segment.
        s2: second segment.

    Returns:
        Minimum distance. 0.0 if they intersect.
    """
    if segments_intersect(s1, s2):
        return 0.0

    return min(
        point_to_segment_distance(s1[0], s2),
        point_to_segment_distance(s1[1], s2),
        point_to_segment_distance(s2[0], s1),
        point_to_segment_distance(s2[1], s1),
    )


# -- Track chain building -------------------------------------------


def _points_close(a: Point, b: Point, tol: float) -> bool:
    """Whether two points are within tolerance of each other."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def build_track_chain(
    tracks: Sequence[Segment], tol: float = 0.01
) -> list[list[Segment]]:
    """Group track segments into connected chains.

    Two segments belong to the same chain when an endpoint of one is
    within tol of an endpoint of the other.

    Args:
        tracks: list of segments.
        tol: connection tolerance (mm).

    Returns:
        List of chains, each chain a list of segments.
    """
    if not tracks:
        return []

    n = len(tracks)
    visited = [False] * n
    chains: list[list[Segment]] = []

    for start_idx in range(n):
        if visited[start_idx]:
            continue
        chain: list[Segment] = [tracks[start_idx]]
        visited[start_idx] = True

        changed = True
        while changed:
            changed = False
            for i in range(n):
                if visited[i]:
                    continue
                seg = tracks[i]
                for c_seg in chain:
                    if (
                        _points_close(seg[0], c_seg[0], tol)
                        or _points_close(seg[0], c_seg[1], tol)
                        or _points_close(seg[1], c_seg[0], tol)
                        or _points_close(seg[1], c_seg[1], tol)
                    ):
                        chain.append(seg)
                        visited[i] = True
                        changed = True
                        break
        chains.append(chain)

    return chains


def chain_endpoints(chain: Sequence[Segment]) -> tuple[set[Point], set[Point]]:
    """Split a chain's points into endpoints (degree 1) and internal junctions (degree 2+).

    Returns:
        (endpoints, junctions) — the set of endpoints and the set of junctions.
        endpoints are the points at the start/end of the chain.
    """
    point_count: dict[tuple[float, float], int] = {}
    for seg in chain:
        for p in seg:
            # Round to absorb floating-point noise.
            key = (round(p[0], 4), round(p[1], 4))
            point_count[key] = point_count.get(key, 0) + 1

    endpoints = {p for p, c in point_count.items() if c == 1}
    junctions = {p for p, c in point_count.items() if c >= 2}
    return endpoints, junctions


def total_path_length(tracks: Sequence[Segment]) -> float:
    """Sum of the lengths of the given track segments.

    Args:
        tracks: list of segments.

    Returns:
        Total length (mm).
    """
    return sum(
        math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) for s in tracks
    )
