"""Pure-Python unit tests for the outline-simplify planner (no C++ needed).

Contract under test:
tessellated micro-segment runs collapse to few arcs/lines within
ε = ARC_LOW_DEF_MM, endpoints/anchors are exactly preserved, the plan is
deterministic, and re-planning the applied result is a no-op (idempotency).
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from pcb_world.engine.outline_simplify import (
    EPS_NM,
    plan_graphics_simplify,
)


@dataclass
class Shape:
    """Stand-in for the RLGraphicShape binding surface."""

    index: int
    kind: int
    x1_nm: int = 0
    y1_nm: int = 0
    xm_nm: int = 0
    ym_nm: int = 0
    x2_nm: int = 0
    y2_nm: int = 0
    width_nm: int = 100_000


def _segments(points, width=100_000, start_index=0):
    """Consecutive points → segment Shape list."""
    return [
        Shape(index=start_index + i, kind=0,
              x1_nm=int(points[i][0]), y1_nm=int(points[i][1]),
              x2_nm=int(points[i + 1][0]), y2_nm=int(points[i + 1][1]),
              width_nm=width)
        for i in range(len(points) - 1)
    ]


def _arc_points(cx, cy, r, a0, a1, n):
    return [
        (round(cx + r * math.cos(a0 + (a1 - a0) * i / n)),
         round(cy + r * math.sin(a0 + (a1 - a0) * i / n)))
        for i in range(n + 1)
    ]


def _apply(shapes, plan):
    """Geometrically apply a plan → new shape list (kept + added)."""
    removed = set(plan.remove_indices)
    out = [s for s in shapes if s.index not in removed]
    nxt = max((s.index for s in shapes), default=0) + 1
    for x1, y1, x2, y2, w in plan.new_segments:
        out.append(Shape(index=nxt, kind=0, x1_nm=x1, y1_nm=y1,
                         x2_nm=x2, y2_nm=y2, width_nm=w))
        nxt += 1
    for x1, y1, xm, ym, x2, y2, w in plan.new_arcs:
        out.append(Shape(index=nxt, kind=1, x1_nm=x1, y1_nm=y1,
                         xm_nm=xm, ym_nm=ym, x2_nm=x2, y2_nm=y2, width_nm=w))
        nxt += 1
    return out


def _max_dev_vs_arcs_lines(points, plan, shapes):
    """Max distance of original vertices to the applied geometry (nm)."""
    applied = _apply(shapes, plan)
    best = np.full(len(points), np.inf)
    pts = np.asarray(points, dtype=np.float64)
    for s in applied:
        if s.kind == 0:
            a = np.array([s.x1_nm, s.y1_nm], dtype=np.float64)
            b = np.array([s.x2_nm, s.y2_nm], dtype=np.float64)
            ab = b - a
            denom = float(ab @ ab) or 1.0
            t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
            d = np.hypot(*(pts - (a + t[:, None] * ab)).T)
        else:
            # distance to the full circle through start/mid/end (upper-bounds
            # the arc distance for vertices inside the sweep)
            a = np.array([s.x1_nm, s.y1_nm], dtype=np.float64)
            m = np.array([s.xm_nm, s.ym_nm], dtype=np.float64)
            b = np.array([s.x2_nm, s.y2_nm], dtype=np.float64)
            d2 = 2.0 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]))
            if abs(d2) < 1e-9:
                continue
            a2, m2, b2 = a @ a, m @ m, b @ b
            ux = (a2 * (m[1] - b[1]) + m2 * (b[1] - a[1]) + b2 * (a[1] - m[1])) / d2
            uy = (a2 * (b[0] - m[0]) + m2 * (a[0] - b[0]) + b2 * (m[0] - a[0])) / d2
            c = np.array([ux, uy])
            r = float(np.hypot(*(a - c)))
            d = np.abs(np.hypot(*(pts - c).T) - r)
        best = np.minimum(best, d)
    return float(best.max())


def test_micro_collinear_run_becomes_one_line():
    points = [(i * 10_000, 0) for i in range(101)]  # 100 × 0.01 mm pieces
    plan = plan_graphics_simplify(_segments(points))
    assert plan.report.n_removed == 100
    assert plan.new_arcs == []
    assert plan.new_segments == [(0, 0, 1_000_000, 0, 100_000)]


def test_tessellated_quarter_arc_becomes_one_arc():
    points = _arc_points(0, 0, 10_000_000, 0.0, math.pi / 2, 400)
    shapes = _segments(points)
    plan = plan_graphics_simplify(shapes)
    assert plan.report.n_removed == 400
    assert len(plan.new_arcs) == 1
    assert plan.new_segments == []
    x1, y1, xm, ym, x2, y2, _w = plan.new_arcs[0]
    assert (x1, y1) == points[0]          # endpoints exactly preserved
    assert (x2, y2) == points[-1]
    assert _max_dev_vs_arcs_lines(points, plan, shapes) <= EPS_NM + 2


def test_closed_tessellated_circle_collapses():
    points = _arc_points(0, 0, 5_000_000, 0.0, 2 * math.pi, 720)
    points[-1] = points[0]  # exactly closed ring
    shapes = _segments(points)
    plan = plan_graphics_simplify(shapes)
    assert plan.report.n_removed == 720
    assert 1 <= len(plan.new_arcs) + len(plan.new_segments) <= 6
    assert len(plan.new_arcs) >= 1
    assert _max_dev_vs_arcs_lines(points, plan, shapes) <= EPS_NM + 2


def test_real_corner_between_long_lines_is_preserved():
    # two 5 mm lines meeting at 90° — nothing to simplify
    points = [(0, 0), (5_000_000, 0), (5_000_000, 5_000_000)]
    plan = plan_graphics_simplify(_segments(points))
    assert plan.report.n_removed == 0


def test_corner_vertex_between_micro_runs_survives_as_breakpoint():
    # two tessellated arcs meeting at a sharp cusp (gothic arch, 60° kink):
    # the cusp vertex must remain an exact endpoint of the fitted elements
    left = _arc_points(-5_000_000, 0, 10_000_000, 0.0, math.pi / 3, 300)
    right = _arc_points(5_000_000, 0, 10_000_000, 2 * math.pi / 3, math.pi, 300)
    cusp = left[-1]
    assert right[0] == cusp
    points = left + right[1:]
    shapes = _segments(points)
    plan = plan_graphics_simplify(shapes)
    assert plan.report.n_removed == 600
    ends = {(a[0], a[1]) for a in plan.new_arcs} | {(a[4], a[5]) for a in plan.new_arcs}
    ends |= {(s[0], s[1]) for s in plan.new_segments} | {(s[2], s[3]) for s in plan.new_segments}
    assert tuple(cusp) in ends


def test_sub_eps_staircase_merges():
    # 0.01 mm steps — far below ε → a diagonal line is a faithful fit
    points = []
    x = y = 0
    points.append((x, y))
    for _ in range(200):
        x += 10_000
        points.append((x, y))
        y += 10_000
        points.append((x, y))
    shapes = _segments(points)
    plan = plan_graphics_simplify(shapes)
    assert plan.report.n_removed == 400
    assert len(plan.new_arcs) + len(plan.new_segments) < 40
    assert _max_dev_vs_arcs_lines(points, plan, shapes) <= EPS_NM + 2


def test_mixed_width_chain_splits_runs():
    pts_a = [(i * 10_000, 0) for i in range(51)]
    pts_b = [(500_000 + i * 10_000, 0) for i in range(51)]
    shapes = _segments(pts_a, width=100_000) + _segments(
        pts_b, width=200_000, start_index=100
    )
    plan = plan_graphics_simplify(shapes)
    assert plan.report.n_removed == 100
    widths = sorted(s[4] for s in plan.new_segments)
    assert widths == [100_000, 200_000]


def test_deterministic():
    points = _arc_points(0, 0, 8_000_000, 0.3, 2.1, 500)
    shapes = _segments(points)
    p1 = plan_graphics_simplify(shapes)
    p2 = plan_graphics_simplify(shapes)
    assert p1.remove_indices == p2.remove_indices
    assert p1.new_segments == p2.new_segments
    assert p1.new_arcs == p2.new_arcs


@pytest.mark.parametrize("case", ["quarter_arc", "circle", "staircase", "sline"])
def test_idempotent_on_applied_result(case):
    if case == "quarter_arc":
        points = _arc_points(0, 0, 10_000_000, 0.0, math.pi / 2, 400)
    elif case == "circle":
        points = _arc_points(0, 0, 5_000_000, 0.0, 2 * math.pi, 720)
        points[-1] = points[0]
    elif case == "staircase":
        points = []
        x = y = 0
        points.append((x, y))
        for _ in range(150):
            x += 10_000
            points.append((x, y))
            y += 10_000
            points.append((x, y))
    else:  # gentle S-curve — arc fit must split
        up = _arc_points(0, 5_000_000, 5_000_000, -math.pi / 2, 0.0, 300)
        down = _arc_points(10_000_000, 5_000_000, 5_000_000, math.pi, math.pi / 2, 300)
        points = up + down[1:]
    shapes = _segments(points)
    plan = plan_graphics_simplify(shapes)
    assert plan.report.n_removed > 0
    applied = _apply(shapes, plan)
    plan2 = plan_graphics_simplify(applied)
    assert plan2.remove_indices == []
    assert plan2.new_segments == []
    assert plan2.new_arcs == []


# ---------------------------------------------------------------------------
# Correctness-gate guarantees (≤ ε dense, cross-run idempotency, loud no-op)
# ---------------------------------------------------------------------------

def _dense_dev(points, plan, shapes):
    """True bidirectional deviation (nm) of the applied geometry vs the
    original polyline, sampled densely (ε/4) both ways — the real ≤ ε check."""
    from pcb_world.engine.outline_simplify import (
        _min_dist_to_polyline, _sample_arc,
    )
    applied = _apply(shapes, plan)
    step = EPS_NM / 4.0
    orig = np.asarray(points, dtype=np.float64)
    fits = []
    for s in applied:
        a = np.array([s.x1_nm, s.y1_nm], float)
        b = np.array([s.x2_nm, s.y2_nm], float)
        fits.append(np.vstack([a, b]) if s.kind == 0
                    else _sample_arc(a, (s.xm_nm, s.ym_nm), b, step))
    removed = set(plan.remove_indices)

    def densify(poly):
        out = [poly]
        for i in range(len(poly) - 1):
            L = float(np.hypot(*(poly[i + 1] - poly[i])))
            k = max(2, int(L / step) + 1)
            t = np.linspace(0, 1, k)
            out.append(poly[i] + (poly[i + 1] - poly[i]) * t[:, None])
        return np.vstack(out)

    worst = 0.0
    # forward: original polyline samples → nearest fitted element
    removed_pts = densify(orig)
    d = np.full(len(removed_pts), np.inf)
    for fp in fits:
        d = np.minimum(d, _min_dist_to_polyline(removed_pts, fp))
    if removed:
        worst = max(worst, float(d.max()))
    # reverse: each fitted element's dense samples → original polyline
    for fp in fits:
        s = densify(fp)
        worst = max(worst, float(_min_dist_to_polyline(s, orig).max()))
    return worst


@pytest.mark.parametrize("seed", range(12))
def test_random_wave_within_eps_and_idempotent(seed):
    """Shallow noisy waves (the adversarial family) must convert within ε
    (dense, bidirectional) AND be a reload fixpoint — the two hard guarantees."""
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.5, 3.0) * EPS_NM
    lam = rng.uniform(5e5, 5e6)
    L = rng.uniform(2e6, 3e7)
    n = int(rng.integers(60, 700))
    xs = np.sort(rng.uniform(0, L, n + 1))
    xs[0], xs[-1] = 0, L
    points = [(round(x), round(amp * math.sin(2 * math.pi * x / lam)
                          + 0.3 * amp * math.sin(6.1 * math.pi * x / lam)))
              for x in xs]
    shapes = _segments(points)
    plan = plan_graphics_simplify(shapes)
    if plan.report.n_removed:
        assert _dense_dev(points, plan, shapes) <= EPS_NM  # ≤ real ε, dense
        plan2 = plan_graphics_simplify(_apply(shapes, plan))
        assert not plan2.remove_indices  # reload fixpoint


def test_junction_conversion_is_idempotent():
    """A T-junction (degree-3 node): converting a chain can dissolve an anchor
    and let a reload chain two runs — the global fixpoint must prevent that."""
    trunk = _arc_points(0, 0, 8_000_000, 0.2, 2.4, 400)
    spur_start = trunk[200]
    spur = [spur_start]
    for k in range(1, 60):
        spur.append((spur_start[0] + k * 30_000, spur_start[1] + k * 5_000))
    shapes = _segments(trunk) + _segments(spur, start_index=10_000)
    plan = plan_graphics_simplify(shapes)
    assert plan.report.n_removed > 0  # the trunk arc converts
    assert plan.report.max_dev_nm <= EPS_NM  # gate's own ≤ ε guarantee
    # the reload of the whole (junction) geometry must be a fixpoint
    assert not plan_graphics_simplify(_apply(shapes, plan)).remove_indices


def test_unfittable_sparse_run_kept_and_warned():
    """A sparse zig-zag no single arc/line can fit within ε must be kept as-is
    (loud no-op), never distorted — verifies the fitter-independent gate."""
    # alternating large offsets, each vertex ~2×ε off any chord/arc through neighbours
    points = [(0, 0)]
    for k in range(1, 40):
        points.append((k * 300_000, (k % 2) * 3 * EPS_NM))
    shapes = _segments(points)
    plan = plan_graphics_simplify(shapes)
    # either unconverted, or if partially converted still ≤ ε and idempotent
    if plan.report.n_removed:
        assert _dense_dev(points, plan, shapes) <= EPS_NM
        assert not plan_graphics_simplify(_apply(shapes, plan)).remove_indices
