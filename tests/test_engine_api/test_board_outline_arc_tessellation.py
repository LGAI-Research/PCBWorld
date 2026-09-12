"""Board-outline arc tessellation is error-bounded (KiCad-native).

``PNS_RL_ROUTER::getBoardOutline`` tessellates Edge.Cuts arcs through KiCad's
own ``SHAPE_ARC``/``GetArcToSegmentCount`` path, so the chord-to-arc deviation
(sagitta) of every emitted segment stays within the board's ``m_MaxError``
(default 0.005 mm) — regardless of radius or sweep. This replaced a fixed
16-segments-per-90-degrees scheme that under-resolved large arcs and
over-tessellated small fillets.

This is the regression guard for that contract.
"""

import math

import pytest

from tests.test_engine_api.conftest import FIXTURES_DIR

# KiCad default ARC_HIGH_DEF; the tessellation guarantees sagitta <= this.
MAX_ERROR_MM = 0.005
# Endpoints land on an int-nm grid; allow a small margin over the exact bound.
TOL_MM = MAX_ERROR_MM + 0.002


@pytest.fixture(scope="module")
def outline_segments():
    """Edge.Cuts outline of a board whose boundary contains 90-degree arcs."""
    krl = pytest.importorskip("kicad_rl_router")
    board = FIXTURES_DIR / "crossover_legacy.kicad_pcb"
    if not board.exists():
        pytest.skip(f"Board not found: {board}")
    router = krl.RLRouter(str(board))
    segs = [
        ((e.x1_mm, e.y1_mm), (e.x2_mm, e.y2_mm)) for e in router.get_board_outline()
    ]
    assert segs, "outline must contain segments"
    return segs


def _chain_runs(segs):
    """Group segments into ordered polylines by shared endpoints.

    Within one source shape ``getBoardOutline`` emits segments sequentially
    (``p2[i] == p1[i+1]``), so an arc becomes one contiguous run of vertices.
    """
    runs = []
    cur = None
    for p1, p2 in segs:
        if cur is not None and _close(cur[-1], p1):
            cur.append(p2)
        else:
            cur = [p1, p2]
            runs.append(cur)
    return runs


def _close(a, b, eps=1e-6):
    return math.hypot(a[0] - b[0], a[1] - b[1]) < eps


def _circumcenter(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:  # collinear -> not an arc
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy


def test_arc_segments_within_max_error(outline_segments):
    """Every curved run's chord sagitta stays within MaxError."""
    runs = _chain_runs(outline_segments)
    curved_runs_checked = 0

    for run in runs:
        if len(run) < 3:
            continue  # a straight edge is a single segment; nothing to bound
        center = _circumcenter(run[0], run[len(run) // 2], run[-1])
        if center is None:
            continue  # collinear (straight) run

        cx, cy = center
        radius = math.hypot(run[0][0] - cx, run[0][1] - cy)

        # Vertices sit on the true arc: radial deviation must be ~0.
        for vx, vy in run:
            radial = abs(math.hypot(vx - cx, vy - cy) - radius)
            assert radial < TOL_MM, f"vertex off the arc by {radial * 1000:.2f} um"

        # Contract: each chord's sagitta (midpoint gap to the arc) <= MaxError.
        for (x1, y1), (x2, y2) in zip(run, run[1:]):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            sagitta = radius - math.hypot(mx - cx, my - cy)
            assert sagitta <= TOL_MM, (
                f"chord sagitta {sagitta * 1000:.2f} um exceeds "
                f"MaxError {MAX_ERROR_MM * 1000:.1f} um"
            )
        curved_runs_checked += 1

    assert curved_runs_checked > 0, "fixture must exercise at least one arc"
