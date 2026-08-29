"""Load-time board-outline simplification — micro-segment chains → native arcs/lines.

Boards whose curved Edge.Cuts outlines were baked into thousands of micro
``gr_line`` segments make PNS walkaround assemble the whole perimeter into one
cluster and iterate it per item (finish actions 5–7 s). The same shape stored as a
handful of native 3-point arcs has no such pathology: an arc is ONE item in
the router world and its collision checks are analytic.

This module plans that rewrite on the *loaded* board (exact integer-nm
geometry read via ``RLRouter.get_graphic_shapes``) and applies it via
``RLRouter.replace_graphic_shapes``:

1. assemble segment chains by shared endpoints,
2. split chains into runs at *anchors* — sharp corners, long segments (≥1 mm,
   classified by true endpoint length), chain ends, junctions (anchors are
   never modified, so their endpoints are exactly preserved; existing arcs
   never join a chain at all),
3. per run, fit with the {line, arc} extension of Douglas–Peucker
   (Rosin–West style): try a single line, then a single 3-point arc pinned to
   the run's endpoints, else split at the worst vertex and recurse,
4. apply only where it strictly reduces the item count.

Two guarantees are ENFORCED here, not trusted from the fit — the fitter is
just a proposer:

- **≤ ε versus the true original.** Every converted run passes an independent
  analytic deviation gate (:func:`_max_deviation`): a line is bounded by its
  vertices' perpendicular distance (exact); an arc by radial residual + max
  sub-chord sagitta (a true upper bound). Fits land inside ``ε −
  EPS_FIT_MARGIN_NM`` so the *stored* integer geometry stays ≤ ε after
  rounding. A run that can't be fit within that keeps its original
  micro-segments and warns — never silently distorted (project loud-fail).
- **Idempotency.** A single pass is not idempotent (emitted lines re-chain on
  reload and can merge across a corner/junction that dissolved). So the
  single-pass planner is iterated to a fixpoint on a Python model of the reload
  (apply → re-plan → …); the returned geometry satisfies ``plan(apply(x)) ==
  ∅``, a true fixpoint. Multi-pass composition is re-validated ≤ ε against the
  original by :func:`_within_eps` (dense, ε/4 sampling); if it drifts, the
  layer is left unconverted with a loud warning.

Also: deterministic — the same input always yields the same plan.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ε — single source: KiCad include/base_units.h:116 ``ARC_LOW_DEF_MM = 0.02``,
# the tolerance the router itself uses when it converts an arc into a hull
# polyline (SHAPE_ARC::ConvertToPolyline at ARC_LOW_DEF). Matching it means
# the converted outline deviates from the file geometry no more than the
# router's own view of a native arc deviates from that arc.
ARC_LOW_DEF_MM = 0.02
EPS_NM = int(ARC_LOW_DEF_MM * 1_000_000)  # 20 000 nm

# The per-run gate bounds deviation analytically at the run vertices; a few
# effects slip a hair past it — integer-nm rounding of an arc's stored mid, and
# an original point near an arc endpoint that projects just outside the sweep
# (measured to the circle, not the arc, it reads a touch short). Observed ≤ 30 nm
# across 5 000 adversarial fuzz cases. Fitting to ``EPS_NM − this`` margin keeps
# the *stored* geometry provably ≤ ε with headroom (0.2 µm = 1 % of ε, ~600× below
# any DRC clearance). The global multi-pass recheck still uses the full ε.
EPS_FIT_MARGIN_NM = 200

# Segments at least this long (50×ε) are immutable anchors, endpoints
# included: genuine outline features, never part of a tessellated micro-run.
ANCHOR_SEG_LEN_NM = 1_000_000
# A direction change of at least this much at a vertex is a corner anchor —
# but only when both adjacent segments are at least CORNER_MIN_LEN_NM (10×ε):
# a sharp turn between micro-jaggies is tessellation noise, not geometry, and
# anchoring it would pin e.g. a sub-ε staircase at every step.
ANCHOR_ANGLE_DEG = 25.0
CORNER_MIN_LEN_NM = 200_000
# Runs shorter than this many segments are never converted (marginal gain,
# and skipping them keeps re-running the pass on its own output a no-op).
MIN_RUN_SEGS = 3
# Endpoints within this distance are treated as the same chain node (chain
# assembly only — coordinates are never moved). Real connected KiCad segments
# share a vertex EXACTLY (integer nm); this only bridges sub-nm mm↔nm round-trip
# jitter (v0.20 fix). Kept tight (8 nm) on purpose: a looser tolerance let the
# ``pts`` polyline pick up multi-nm discontinuities at joins, which perturbed a
# segment's length enough to flip its ≥1 mm anchor classification.
JOIN_TOL_NM = 8
# An unconverted run at least this many segments long is worth a warning:
# it keeps feeding the walkaround-cluster pathology this pass exists to fix.
WARN_UNCONVERTED_RUN = 50
# Reject arc fits flatter than this radius (10 m): after nm rounding their
# midpoint would be collinear with the endpoints, which KiCad cannot store.
MAX_ARC_RADIUS_NM = 1e10


@dataclass
class SimplifyReport:
    """Per-board conversion summary (one per apply/plan call)."""

    n_layer_shapes: int = 0        #: PCB_SHAPEs on the target layer
    n_input_segments: int = 0      #: straight segments among them
    n_removed: int = 0             #: segments replaced
    n_out_arcs: int = 0            #: arcs added
    n_out_lines: int = 0           #: lines added
    max_dev_nm: float = 0.0        #: worst fit deviation vs original vertices
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.n_removed > 0

    def summary(self) -> str:
        return (
            f"{self.n_input_segments} seg -> {self.n_out_arcs} arc + "
            f"{self.n_out_lines} line (replaced {self.n_removed}, "
            f"kept {self.n_input_segments - self.n_removed}), "
            f"max_dev={self.max_dev_nm / 1e6:.4f} mm, "
            f"{self.elapsed_s * 1e3:.0f} ms"
        )


@dataclass
class SimplifyPlan:
    """Rewrite plan in ``replace_graphic_shapes`` argument form (nm ints)."""

    remove_indices: list[int] = field(default_factory=list)
    new_segments: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    new_arcs: list[tuple[int, int, int, int, int, int, int]] = field(default_factory=list)
    report: SimplifyReport = field(default_factory=SimplifyReport)


# ---------------------------------------------------------------------------
# Chain assembly
# ---------------------------------------------------------------------------

def _join_nodes(endpoints: np.ndarray) -> np.ndarray:
    """Map each endpoint to a joint-node id (exact or ≤JOIN_TOL_NM matches).

    ``endpoints``: (2n, 2) int64 array (both ends of n segments). Returns a
    (2n,) array of node ids. Exact-coordinate sharing is the common case; a
    union-find pass then merges any degree-1 coordinate with its nearest
    partner within JOIN_TOL_NM, so 1 nm export jitter cannot break a chain.
    """
    node_of: dict[tuple[int, int], int] = {}
    ids = np.empty(len(endpoints), dtype=np.int64)
    for i, (x, y) in enumerate(endpoints):
        key = (int(x), int(y))
        if key not in node_of:
            node_of[key] = len(node_of)
        ids[i] = node_of[key]

    n_nodes = len(node_of)
    degree = np.bincount(ids, minlength=n_nodes)
    lonely = np.flatnonzero(degree == 1)
    if len(lonely) == 0:
        return ids

    coords = {v: k for k, v in node_of.items()}
    parent = list(range(n_nodes))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:  # smaller root wins → deterministic
            parent[max(ra, rb)] = min(ra, rb)

    buckets: dict[tuple[int, int], list[int]] = {}
    for n, (x, y) in coords.items():
        buckets.setdefault((x // JOIN_TOL_NM, y // JOIN_TOL_NM), []).append(n)
    for n in lonely:
        x, y = coords[int(n)]
        bx, by = x // JOIN_TOL_NM, y // JOIN_TOL_NM
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for m in buckets.get((bx + dx, by + dy), ()):
                    if m == n:
                        continue
                    mx, my = coords[m]
                    if abs(mx - x) <= JOIN_TOL_NM and abs(my - y) <= JOIN_TOL_NM:
                        d = (mx - x) ** 2 + (my - y) ** 2
                        if best is None or (d, m) < best:
                            best = (d, m)
        if best is not None:
            union(int(n), best[1])

    return np.array([find(int(i)) for i in ids], dtype=np.int64)


def _walk_chains(n_segs: int, node_ids: np.ndarray) -> list[list[tuple[int, bool]]]:
    """Group segments into maximal chains through degree-2 joints.

    ``node_ids``: (2n,) joint-node id per endpoint (segment i owns entries 2i,
    2i+1). Returns chains as lists of ``(seg_index, forward)`` — ``forward``
    False means the segment is traversed end→start. Deterministic: open
    chains start at the lowest-index unused segment with a non-degree-2 end,
    then remaining pure cycles in index order.
    """
    incident: dict[int, list[tuple[int, int]]] = {}
    for i in range(n_segs):
        incident.setdefault(int(node_ids[2 * i]), []).append((i, 0))
        incident.setdefault(int(node_ids[2 * i + 1]), []).append((i, 1))
    degree = {n: len(v) for n, v in incident.items()}

    used = np.zeros(n_segs, dtype=bool)
    chains: list[list[tuple[int, bool]]] = []

    def walk(seg: int, end_in: int) -> list[tuple[int, bool]]:
        chain: list[tuple[int, bool]] = []
        while True:
            used[seg] = True
            forward = end_in == 0
            chain.append((seg, forward))
            out_node = int(node_ids[2 * seg + (1 if forward else 0)])
            if degree[out_node] != 2:
                return chain
            nxt = [(s, e) for s, e in incident[out_node] if s != seg]
            if len(nxt) != 1 or used[nxt[0][0]]:
                return chain
            seg, end_in = nxt[0]

    for i in range(n_segs):
        if used[i]:
            continue
        for end in (0, 1):
            if degree[int(node_ids[2 * i + end])] != 2:
                chains.append(walk(i, end))
                break
    for i in range(n_segs):
        if not used[i]:
            chains.append(walk(i, 0))  # pure cycle
    return chains


# ---------------------------------------------------------------------------
# {line, arc} fitting
# ---------------------------------------------------------------------------

def _dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of each point to the closed segment a–b (float nm)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.hypot(*(pts - a).T)
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.hypot(*(pts - proj).T)


def _try_arc(pts: np.ndarray, eps: float):
    """Fit one 3-point arc pinned to pts[0]/pts[-1]; return (mid, dev) or None.

    The single free parameter (the sagitta) starts at the offset of the
    vertex farthest from the chord and is refined by a golden-section search
    on the max radial residual. Every vertex must fall inside the arc's
    angular span (so the radial residual is the true point-to-arc distance),
    and the arc's own bulge point must stay within ε of the source polyline
    (so the arc cannot balloon between sparse vertices).
    """
    a, b = pts[0], pts[-1]
    chord = b - a
    chord_len = float(np.hypot(*chord))
    if chord_len < 1.0:
        return None  # closed/degenerate run — no single arc can span it

    normal = np.array([-chord[1], chord[0]]) / chord_len
    offs = (pts - a) @ normal
    s0 = float(offs[int(np.argmax(np.abs(offs)))])
    if abs(s0) < 1e-6:
        return None  # straight — the line fit already had its chance

    interior = pts[1:-1]
    mid_chord = (a + b) / 2.0

    def eval_sagitta(s: float):
        """Max radial residual of the arc with signed sagitta ``s``."""
        r = (s * s + (chord_len / 2.0) ** 2) / (2.0 * abs(s))
        # centre offset from the chord midpoint along the bulge direction:
        # |s| − r (negative for a minor arc → opposite side, positive major)
        c = mid_chord + normal * math.copysign(1.0, s) * (abs(s) - r)
        res = float(np.max(np.abs(np.hypot(*(interior - c).T) - r)))
        return res, c, r

    best_res, best_s = eval_sagitta(s0)[0], s0
    lo, hi = 0.5 * s0, 1.5 * s0
    if lo > hi:
        lo, hi = hi, lo
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1, f2 = eval_sagitta(x1)[0], eval_sagitta(x2)[0]
    for _ in range(24):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = eval_sagitta(x1)[0]
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = eval_sagitta(x2)[0]
        if min(f1, f2) < best_res:
            best_res, best_s = (f1, x1) if f1 <= f2 else (f2, x2)
    if best_res > eps:
        return None

    res, c, r = eval_sagitta(best_s)
    if r > MAX_ARC_RADIUS_NM:
        return None

    # Angular containment. The data side (sign of s0) selects the ccw or cw
    # sweep from A to B; every interior vertex must lie inside that sweep.
    ang = np.arctan2((pts - c)[:, 1], (pts - c)[:, 0])
    rel = (ang - ang[0]) % (2.0 * math.pi)
    sweep_ccw = float(rel[-1])
    if sweep_ccw <= 1e-9 or sweep_ccw >= 2.0 * math.pi - 1e-9:
        return None
    inner = rel[1:-1]
    if np.all((inner > 1e-9) & (inner < sweep_ccw - 1e-9)):
        sweep = sweep_ccw
    elif np.all((inner > sweep_ccw + 1e-9) & (inner < 2.0 * math.pi - 1e-9)):
        sweep = sweep_ccw - 2.0 * math.pi  # data sits on the cw sweep
    else:
        return None

    mid_ang = float(ang[0]) + sweep / 2.0
    mid = c + r * np.array([math.cos(mid_ang), math.sin(mid_ang)])
    mid_i = (int(round(mid[0])), int(round(mid[1])))
    if mid_i == (int(pts[0, 0]), int(pts[0, 1])) or mid_i == (int(pts[-1, 0]), int(pts[-1, 1])):
        return None  # would collapse to a degenerate arc after rounding
    # An arc pinned only to the run vertices can bulge > ε from the chords
    # between them. Bound that bulge cheaply (O(n), no dense sampling): between
    # two consecutive vertices the chord-to-arc gap is at most the sub-chord
    # sagitta r − √(r² − (L/2)²); adding the vertices' radial residual ``res``
    # upper-bounds the true arc-to-polyline deviation. If that exceeds ε the
    # caller splits into a finer (valid) fit. Conservative, so a rejected arc
    # is always genuinely too coarse — never a silent > ε acceptance.
    seg_len = np.hypot(*(pts[1:] - pts[:-1]).T)
    half = np.minimum(seg_len / 2.0, r)
    sagitta = r - np.sqrt(np.maximum(r * r - half * half, 0.0))
    if res + float(sagitta.max()) > eps:
        return None
    return mid_i, res


def _fit_run(pts: np.ndarray, eps: float, depth: int = 0) -> tuple[list, float]:
    """Recursively fit ``pts`` (float nm, (m+1, 2)) with lines and arcs.

    Returns ``(elements, max_dev)`` where elements are ``('line', i0, i1)`` /
    ``('arc', i0, i1, (mx, my))`` with indices into ``pts``. All deviations
    are measured against these original vertices.
    """
    n = len(pts) - 1
    if n <= 1 or depth > 48:
        return [("line", i, i + 1) for i in range(n)], 0.0

    dev = _dist_to_segment(pts[1:-1], pts[0], pts[-1])
    worst = float(np.max(dev))
    if worst <= eps:
        return [("line", 0, n)], worst

    arc = _try_arc(pts, eps)
    if arc is not None:
        mid, res = arc
        return [("arc", 0, n, mid)], res

    split = 1 + int(np.argmax(dev))
    left, d1 = _fit_run(pts[: split + 1], eps, depth + 1)
    right, d2 = _fit_run(pts[split:], eps, depth + 1)
    shifted = [
        (el[0], el[1] + split, el[2] + split, *el[3:]) for el in right
    ]
    return left + shifted, max(d1, d2)


def _circle_centre(a, m, b):
    """Centre of the circle through three points, or None if collinear."""
    d = 2.0 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]))
    if abs(d) < 1e-9:
        return None
    a2, m2, b2 = a @ a, m @ m, b @ b
    ux = (a2 * (m[1] - b[1]) + m2 * (b[1] - a[1]) + b2 * (a[1] - m[1])) / d
    uy = (a2 * (b[0] - m[0]) + m2 * (a[0] - b[0]) + b2 * (m[0] - a[0])) / d
    return np.array([ux, uy])


# ---------------------------------------------------------------------------
# Correctness gate — fitter-independent hard guarantee
# ---------------------------------------------------------------------------

def _sample_arc(a, mid, b, max_step):
    """Dense points along the 3-point arc a→mid→b (float, spacing ≤ max_step)."""
    c = _circle_centre(a, np.asarray(mid, float), b)
    if c is None:
        return np.vstack([a, b])
    r = float(np.hypot(*(a - c)))
    aa = math.atan2(a[1] - c[1], a[0] - c[0])
    am = math.atan2(mid[1] - c[1], mid[0] - c[0])
    ab = math.atan2(b[1] - c[1], b[0] - c[0])
    sweep = (ab - aa) % (2.0 * math.pi)
    if (am - aa) % (2.0 * math.pi) > sweep:
        sweep -= 2.0 * math.pi  # the mid point fixes which way the arc goes
    n = max(2, int(abs(sweep) * r / max_step) + 1)
    ang = aa + sweep * np.linspace(0.0, 1.0, n)
    return c + r * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _min_dist_to_polyline(samples: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """For each sample, distance to the nearest segment of ``poly`` (vectorized)."""
    best = np.full(len(samples), np.inf)
    for i in range(len(poly) - 1):
        best = np.minimum(best, _dist_to_segment(samples, poly[i], poly[i + 1]))
    return best


def _max_deviation(orig: np.ndarray, elements: list, eps: float) -> float:
    """Upper bound (nm) on the deviation between the original polyline and the
    fit — an independent, analytic re-confirmation of the ≤ ε contract.

    Elements partition the run (shared endpoints), so each is bounded against
    only the local original span ``orig[i0..i1]`` it covers, in O(local) with no
    sampling loops:

    - line: exact — the max perpendicular distance of the local vertices to the
      segment (distance is convex along each original sub-segment, so vertex
      maxima bound the whole polyline).
    - arc: forward = max radial residual of the local vertices (all in-sweep by
      construction, so this is the true vertex→arc distance); reverse ≤ the max
      sub-chord sagitta (chord-to-arc gap between consecutive vertices). Their
      sum upper-bounds the arc↔polyline Hausdorff distance.

    Conservative, so a value ≤ ε is a genuine guarantee; short-circuits once ε
    is exceeded.
    """
    worst = 0.0
    for el in elements:
        ia, ib = el[1], el[2]
        local = orig[ia : ib + 1]
        a, b = orig[ia], orig[ib]
        if el[0] == "line":
            d = float(_dist_to_segment(local, a, b).max())
        else:
            c = _circle_centre(a, np.asarray(el[3], dtype=np.float64), b)
            if c is None:
                d = float(_dist_to_segment(local, a, b).max())
            else:
                r = float(np.hypot(*(a - c)))
                radial = float(np.abs(np.hypot(*(local - c).T) - r).max())
                seg_len = np.hypot(*(local[1:] - local[:-1]).T)
                half = np.minimum(seg_len / 2.0, r)
                sag = float((r - np.sqrt(np.maximum(r * r - half * half, 0.0))).max())
                d = radial + sag
        worst = max(worst, d)
        if worst > eps:
            return worst
    return worst


class _Stub:
    """Minimal RLGraphicShape stand-in for the reload-fixpoint iteration."""

    __slots__ = ("index", "kind", "x1_nm", "y1_nm", "xm_nm", "ym_nm",
                 "x2_nm", "y2_nm", "width_nm")

    def __init__(self, index, kind, x1, y1, xm, ym, x2, y2, w):
        self.index, self.kind = index, kind
        self.x1_nm, self.y1_nm, self.xm_nm, self.ym_nm = x1, y1, xm, ym
        self.x2_nm, self.y2_nm, self.width_nm = x2, y2, w


def _seg_grid(segs: np.ndarray, cell: float) -> dict:
    """Bucket segments (n,4 x1y1x2y2) into a grid keyed by every cell their
    bounding box spans, so a point query need only scan its 3×3 neighbourhood."""
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (x1, y1, x2, y2) in enumerate(segs):
        cx0, cx1 = sorted((int(x1 // cell), int(x2 // cell)))
        cy0, cy1 = sorted((int(y1 // cell), int(y2 // cell)))
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                grid.setdefault((cx, cy), []).append(i)
    return grid


def _grid_min_dist(pts: np.ndarray, segs: np.ndarray, grid: dict, cell: float,
                   eps: float) -> float:
    """Max over ``pts`` of the distance to the nearest segment (grid-accelerated).

    Returns early with ``inf`` intent by short-circuiting: caller compares to ε.
    Each point scans only segments registered in its 3×3 cell neighbourhood;
    a point with no candidate segment nearby yields a large distance (a real
    miss — the geometries genuinely diverge there).
    """
    worst = 0.0
    for p in pts:
        cx, cy = int(p[0] // cell), int(p[1] // cell)
        cand: set[int] = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.update(grid.get((cx + dx, cy + dy), ()))
        if not cand:
            return float("inf")
        idx = np.fromiter(cand, dtype=np.int64)
        a = segs[idx, 0:2].astype(np.float64)
        b = segs[idx, 2:4].astype(np.float64)
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom[denom == 0] = 1.0
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
        d = np.hypot(*(p - (a + t[:, None] * ab)).T)
        dm = float(d.min())
        if dm > worst:
            worst = dm
            if worst > eps:
                return worst
    return worst


def _elements_to_segments(stubs: list, step: float) -> np.ndarray:
    """Densify a list of line/arc stubs into (n,4) segments at spacing ≤ step."""
    out = []
    for st in stubs:
        a = np.array([st.x1_nm, st.y1_nm], dtype=np.float64)
        b = np.array([st.x2_nm, st.y2_nm], dtype=np.float64)
        if st.kind == 0:
            poly = np.vstack([a, b])
        else:
            poly = _sample_arc(a, (st.xm_nm, st.ym_nm), b, step)
        out.append(np.hstack([poly[:-1], poly[1:]]))
    return np.vstack(out) if out else np.empty((0, 4))


def _subdivide(segs: np.ndarray, step: float) -> np.ndarray:
    """Points along each segment at spacing ≤ step (per-segment count)."""
    a = segs[:, 0:2].astype(np.float64)
    b = segs[:, 2:4].astype(np.float64)
    lens = np.hypot(*(b - a).T)
    ncut = np.maximum(1, np.ceil(lens / step).astype(np.int64))
    pts = [a, b]
    for j in range(1, int(ncut.max())):
        m = ncut > j
        f = (j / ncut[m])[:, None]
        pts.append(a[m] + (b[m] - a[m]) * f)
    return np.vstack(pts)


def _within_eps(orig_segs: list, final_stubs: list, eps: float) -> bool:
    """Bidirectional deviation between the original removed geometry and the
    final fixpoint geometry ≤ ε (grid-accelerated, spatial). Sets
    ``_within_eps.last_dev`` for reporting.

    Samples at ε/4 (not ε): a coarser step can miss a deviation peak *between*
    samples by a fraction of ε, which matters here because this is the rigorous
    gate for multi-pass composition, where the drift can sit right at ε."""
    step = max(eps / 4.0, 1.0)
    O = np.array(orig_segs, dtype=np.int64)
    F = _elements_to_segments(final_stubs, step)  # arcs pre-densified ≤ step
    if len(O) == 0 or len(F) == 0:
        _within_eps.last_dev = 0.0
        return True
    cell = max(step * 4.0, 1.0)
    grid_o = _seg_grid(O, cell)
    Fi = F.astype(np.int64)
    grid_f = _seg_grid(Fi, cell)
    # reverse: every final-geometry point ≤ ε from the original polyline
    d1 = _grid_min_dist(_subdivide(F, step), O, grid_o, cell, eps)
    # forward: every original point ≤ ε from the final geometry
    d2 = _grid_min_dist(_subdivide(O.astype(np.float64), step), Fi, grid_f, cell, eps)
    _within_eps.last_dev = max(d1, d2)
    return _within_eps.last_dev <= eps


_within_eps.last_dev = 0.0


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _plan_pass(
    shapes,
    eps_nm: float,
    anchor_seg_len_nm: int,
    anchor_angle_deg: float,
    corner_min_len_nm: int,
    min_run_segs: int,
) -> SimplifyPlan:
    """One simplification pass over ``shapes`` (no idempotency iteration).

    Assembles segment chains, splits each into anchor-bounded runs, and
    converts a run only when the fit reduces the item count AND stays ≤ ε from
    that run's original vertices everywhere (``_max_deviation``). Returns a plan
    whose ``remove_indices`` reference the input shapes' ``.index``. Global
    idempotency + ε-vs-true-original are enforced by the ``plan_graphics_simplify``
    wrapper around this.
    """
    plan = SimplifyPlan()
    rep = plan.report

    segs = []
    for s in shapes:
        rep.n_layer_shapes += 1
        if s.kind != 0:
            continue
        rep.n_input_segments += 1
        if (s.x1_nm, s.y1_nm) != (s.x2_nm, s.y2_nm):
            segs.append((s.index, s.x1_nm, s.y1_nm, s.x2_nm, s.y2_nm, s.width_nm))

    if len(segs) < min_run_segs:
        return plan

    arr = np.array([s[1:6] for s in segs], dtype=np.int64)  # x1 y1 x2 y2 w
    indices = [s[0] for s in segs]
    endpoints = np.empty((2 * len(segs), 2), dtype=np.int64)
    endpoints[0::2] = arr[:, 0:2]
    endpoints[1::2] = arr[:, 2:4]

    chains = _walk_chains(len(segs), _join_nodes(endpoints))
    cos_anchor = math.cos(math.radians(anchor_angle_deg))

    for chain in chains:
        # split the chain at width changes (rare; widths must stay verbatim)
        parts: list[list[tuple[int, bool]]] = [[]]
        for seg, fwd in chain:
            if parts[-1] and arr[parts[-1][-1][0], 4] != arr[seg, 4]:
                parts.append([])
            parts[-1].append((seg, fwd))

        for part in parts:
            n = len(part)
            if n < min_run_segs:
                continue
            # polyline vertices in walk order; fuzzy joints keep the incoming
            # endpoint's coordinates (≤JOIN_TOL_NM apart, far below ε)
            pts = np.empty((n + 1, 2), dtype=np.int64)
            for j, (seg, fwd) in enumerate(part):
                pts[j] = arr[seg, 0:2] if fwd else arr[seg, 2:4]
            last_seg, last_fwd = part[-1]
            pts[n] = arr[last_seg, 2:4] if last_fwd else arr[last_seg, 0:2]

            seg_vecs = (pts[1:] - pts[:-1]).astype(np.float64)
            seg_lens = np.hypot(*seg_vecs.T)
            # Anchor classification uses each segment's TRUE length from its own
            # endpoints (arr), not the pts-derived length: at a fuzzy join pts
            # picks the next segment's start, so a pts-length can differ from the
            # true length by up to the join gap and flip the ≥1 mm test.
            true_len = np.hypot(*(arr[[s for s, _ in part], 2:4]
                                  - arr[[s for s, _ in part], 0:2]).T.astype(np.float64))

            anchor = np.zeros(n + 1, dtype=bool)
            anchor[0] = anchor[n] = True
            long_seg = true_len >= anchor_seg_len_nm
            anchor[:-1] |= long_seg
            anchor[1:] |= long_seg
            if n >= 2:
                cosang = np.einsum("ij,ij->i", seg_vecs[:-1], seg_vecs[1:]) / (
                    seg_lens[:-1] * seg_lens[1:]
                )
                real_corner = np.minimum(seg_lens[:-1], seg_lens[1:]) >= corner_min_len_nm
                anchor[1:-1] |= ~(cosang > cos_anchor) & real_corner

            if (
                bool(np.all(pts[0] == pts[n]))
                and not anchor[1:-1].any()
                and not long_seg.any()
            ):
                # anchorless ring (e.g. fully tessellated blob): rotate to a
                # deterministic vertex and split at the farthest one, so two
                # open, fittable runs remain (a 360° arc cannot be stored)
                k0 = int(np.lexsort((pts[:-1, 1], pts[:-1, 0]))[0])
                pts = np.vstack([pts[k0:-1], pts[: k0 + 1]])
                part = part[k0:] + part[:k0]
                anchor = np.zeros(n + 1, dtype=bool)
                anchor[0] = anchor[n] = True
                anchor[int(np.argmax(np.hypot(*(pts - pts[0]).T)))] = True

            fpts = pts.astype(np.float64)
            for b0, b1 in zip(np.flatnonzero(anchor)[:-1], np.flatnonzero(anchor)[1:]):
                m = int(b1 - b0)
                if m < min_run_segs:
                    continue
                run_pts = fpts[b0 : b1 + 1]
                elements, dev = _fit_run(run_pts, float(eps_nm))
                width = int(arr[part[0][0], 4])
                ipts = pts[b0 : b1 + 1]
                # Per-run deviation gate (fitter-independent): only convert a run
                # when the emitted geometry is provably ≤ ε from the original
                # everywhere. A run that fails keeps its original micro-segments
                # and warns — never silently distorted (project loud-fail policy).
                if (len(elements) >= m
                        or _max_deviation(run_pts, elements, float(eps_nm)) > eps_nm):
                    if m >= WARN_UNCONVERTED_RUN:
                        rep.warnings.append(
                            f"run of {m} micro-segments near "
                            f"({pts[b0, 0] / 1e6:.2f}, {pts[b0, 1] / 1e6:.2f}) mm "
                            f"stayed unconverted (fit gave {len(elements)} elements)"
                        )
                    continue
                for el in elements:
                    if el[0] == "line":
                        p, q = ipts[el[1]], ipts[el[2]]
                        plan.new_segments.append(
                            (int(p[0]), int(p[1]), int(q[0]), int(q[1]), width)
                        )
                    else:
                        p, q, mid = ipts[el[1]], ipts[el[2]], el[3]
                        plan.new_arcs.append(
                            (int(p[0]), int(p[1]), mid[0], mid[1],
                             int(q[0]), int(q[1]), width)
                        )
                for seg, _fwd in part[b0:b1]:
                    plan.remove_indices.append(int(indices[seg]))
                rep.max_dev_nm = max(rep.max_dev_nm, dev)

    rep.n_removed = len(plan.remove_indices)
    return plan


def plan_graphics_simplify(
    shapes,
    eps_nm: float = EPS_NM,
    anchor_seg_len_nm: int = ANCHOR_SEG_LEN_NM,
    anchor_angle_deg: float = ANCHOR_ANGLE_DEG,
    corner_min_len_nm: int = CORNER_MIN_LEN_NM,
    min_run_segs: int = MIN_RUN_SEGS,
    max_passes: int = 8,
) -> SimplifyPlan:
    """Plan the micro-segment → arc/line rewrite for one layer's shapes.

    ``shapes``: iterable with the ``RLGraphicShape`` attribute surface
    (``index/kind/x1_nm/y1_nm/x2_nm/y2_nm/width_nm``) — the binding objects
    or any stand-in (tests, file-level dry scans).

    Two hard guarantees, both enforced here rather than trusted from the fit:

    - **Idempotency** — the single-pass planner (:func:`_plan_pass`) is iterated
      to a fixpoint on a Python model of the reload (apply → re-plan → …). A
      single pass is *not* idempotent: emitted lines re-chain on reload and can
      merge across a corner that straightened out or a junction that dissolved.
      Iterating until a pass removes nothing yields geometry that satisfies
      ``plan(apply(x)) == ∅`` — a true fixpoint, so every later reload is a
      no-op.
    - **≤ ε versus the true original** — each pass keeps runs ≤ ε from *that
      pass's* input, so a multi-pass fixpoint could in principle drift up to
      ``passes × ε`` from the real board. The final fixpoint geometry is
      therefore re-checked, bidirectionally and densely, against the *original*
      removed segments (:func:`_within_eps`). If it exceeds ε (only reachable
      through multi-pass composition on adversarial geometry) the whole layer
      is left unconverted with a loud warning — never silently distorted.

    On real boards the fixpoint is reached in one pass with wide ε margin, so
    the extra passes and the final check are effectively free.
    """
    t0 = time.perf_counter()
    # Fit to a slightly tighter ε so the STORED geometry stays ≤ the real ε
    # after rounding / arc-endpoint slop (see EPS_FIT_MARGIN_NM). The global
    # multi-pass recheck below still validates against the full ε.
    eps_fit = max(1.0, float(eps_nm) - EPS_FIT_MARGIN_NM)
    kwargs = dict(eps_nm=eps_fit, anchor_seg_len_nm=anchor_seg_len_nm,
                  anchor_angle_deg=anchor_angle_deg,
                  corner_min_len_nm=corner_min_len_nm, min_run_segs=min_run_segs)

    orig = list(shapes)
    orig_ids = {s.index for s in orig}
    orig_removed_segs: list[tuple] = []  # true-original geometry that got replaced

    working = {s.index: s for s in orig}
    syn = (max(working) if working else 0) + 1
    synth: dict[int, _Stub] = {}

    first = _plan_pass(orig, **kwargs)
    plan = SimplifyPlan()
    plan.report.n_layer_shapes = first.report.n_layer_shapes
    plan.report.n_input_segments = first.report.n_input_segments
    plan.report.warnings = list(first.report.warnings)

    cur = first
    passes = 0
    converged = True
    while cur.remove_indices:
        passes += 1
        if passes > max_passes:
            converged = False
            break
        for idx in cur.remove_indices:
            s = working.pop(idx, None)
            if idx in orig_ids and s is not None:
                orig_removed_segs.append((s.x1_nm, s.y1_nm, s.x2_nm, s.y2_nm))
            else:
                synth.pop(idx, None)
        for x1, y1, x2, y2, w in cur.new_segments:
            working[syn] = synth[syn] = _Stub(syn, 0, x1, y1, 0, 0, x2, y2, w)
            syn += 1
        for x1, y1, xm, ym, x2, y2, w in cur.new_arcs:
            working[syn] = synth[syn] = _Stub(syn, 1, x1, y1, xm, ym, x2, y2, w)
            syn += 1
        cur = _plan_pass(list(working.values()), **kwargs)

    final_syn = [synth[i] for i in synth if i in working]
    rep = plan.report

    # Accept the fixpoint only if it also stays ≤ ε from the TRUE original.
    # Single-pass fixpoint (the common case — real boards): the per-run gate
    # already proved that, so skip the (redundant) global recheck. Multi-pass
    # fixpoints can compose error, so re-validate globally.
    if not orig_removed_segs:
        accept, dev, reason = True, 0.0, ""
    elif not converged:
        accept, dev, reason = False, 0.0, "did not converge"
    elif passes <= 1:
        accept, dev, reason = True, first.report.max_dev_nm, ""
    else:
        # Same margin as the fit (EPS_FIT_MARGIN_NM): the grid recheck samples
        # at ε spacing and shares the arc-endpoint slop, so validate against the
        # tightened ε to keep the true stored deviation ≤ the real ε.
        accept = _within_eps(orig_removed_segs, final_syn, eps_fit)
        dev = _within_eps.last_dev
        reason = "" if accept else f"drifted > ε ({dev / 1e6:.4f} mm)"

    if orig_removed_segs and accept:
        for i in sorted(orig_ids - set(working)):
            plan.remove_indices.append(i)
        for st in final_syn:
            if st.kind == 0:
                plan.new_segments.append(
                    (st.x1_nm, st.y1_nm, st.x2_nm, st.y2_nm, st.width_nm))
            else:
                plan.new_arcs.append(
                    (st.x1_nm, st.y1_nm, st.xm_nm, st.ym_nm,
                     st.x2_nm, st.y2_nm, st.width_nm))
        rep.max_dev_nm = dev
    elif orig_removed_segs:
        rep.warnings.append(
            f"outline conversion left unconverted: reload fixpoint {reason}")

    plan.remove_indices.sort()
    rep.n_removed = len(plan.remove_indices)
    rep.n_out_arcs = len(plan.new_arcs)
    rep.n_out_lines = len(plan.new_segments)
    rep.elapsed_s = time.perf_counter() - t0
    return plan


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------

def apply_graphics_simplify(native_router, layers=None) -> SimplifyReport:
    """Run the simplify pass on a live ``RLRouter`` (one layer at a time).

    Reads shapes with ``get_graphic_shapes``, plans, and — when the plan
    shrinks the item count — applies it via ``replace_graphic_shapes`` (which
    rebuilds connectivity + the PNS world and re-anchors the episode KIID
    rewind point). Returns the merged report; logs one summary line per
    changed board and every warning (loud by contract).

    ``layers`` defaults to the two graphic layers the PNS world syncs as
    full-stack unroutable obstacles (Edge.Cuts + Margin), read from the
    binding's authoritative PCB_LAYER_ID attrs — KiCad 9 renumbered the enum,
    so hardcoding here would silently scan the wrong layer.
    """
    if layers is None:
        import kicad_rl_router as krl
        layers = (krl.LAYER_EDGE_CUTS, krl.LAYER_MARGIN)
    total = SimplifyReport()
    t0 = time.perf_counter()
    for layer in layers:
        shapes = native_router.get_graphic_shapes(layer)
        plan = plan_graphics_simplify(shapes)
        rep = plan.report
        total.n_layer_shapes += rep.n_layer_shapes
        total.n_input_segments += rep.n_input_segments
        total.warnings.extend(rep.warnings)
        if plan.remove_indices:
            native_router.replace_graphic_shapes(
                layer, plan.remove_indices, plan.new_segments, plan.new_arcs
            )
            total.n_removed += rep.n_removed
            total.n_out_arcs += rep.n_out_arcs
            total.n_out_lines += rep.n_out_lines
            total.max_dev_nm = max(total.max_dev_nm, rep.max_dev_nm)
    total.elapsed_s = time.perf_counter() - t0
    if total.changed:
        logger.info("outline_simplify: %s", total.summary())
    for w in total.warnings:
        logger.warning("outline_simplify: %s", w)
    return total
