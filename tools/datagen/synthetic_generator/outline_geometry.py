"""Board-geometry sampling for ``--geo`` d2b boards (outline / holes / THT mix).

Adds the geometry object families the real d3a/d3b boards carry but the
original d2b generator lacked: non-``gr_rect`` outlines (4-line rect, corner
fillets, rectilinear polygons, circles), internal cutouts, NPTH mounting
holes, oval-drill slots, and a diversified THT pad profile.

Target rates are the 2026-08-16 census of the real boards (script:
``sandbox/d2b_midboard/260816_geo_census.py``, data:
``var/results/d2b_midboard/260816_geo_census.json``; d3b basis, per board):

  outline   rect(4 gr_line) ~51% | lines+arcs ~26% | rectilinear ~20% |
            circle ~1%   (``gr_rect`` primitive: 0% on real boards)
  cutouts   Edge.Cuts loops>=2 on ~5%; gr_circle dia 2.2-4.1 mm typical
  NPTH      ~27% of boards, 2-4 near corners; drill 3.2 (M3) dominant,
            then 2.2 / 2.6 / 3.0
  slots     oval drill on ~10%; 0.6x0.8 .. 1.0x3.0 mm
  THT       real drills 0.8-1.0 mm; pad shapes circle/oval/rect

Boards are ``w x h`` mm — square unless the d2b sampler draws an aspect
ratio (``--aspect-sigma``), in which case every outline family stretches with
the box (the circle family excepted: it collapses back to a square bbox).
Coordinates are mm, y-down (KiCad). Every ``emit_*`` draws uuids from the
render rng it is given, so paired d2b/d2bv twins stay identical.
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field

# --- sampling weights (census-sourced; see module docstring) ---------------
OUTLINE_WEIGHTS = (("rect", 0.53), ("fillet", 0.25), ("poly", 0.20),
                   ("circle", 0.02))
FILLET_RADIUS_WEIGHTS = ((1.0, 201), (2.0, 71), (3.0, 52), (4.0, 35),
                         (5.0, 25))                      # d3b histogram (mm)
POLY_TEMPLATE_WEIGHTS = (("L", 0.40), ("notch", 0.25), ("T", 0.12),
                         ("diag", 0.13), ("L_notch", 0.10))
CUTOUT_PROB = 0.05
CUTOUT_DIA = (2.2, 4.1)          # mm; the rare 34-48 mm giants are excluded
NPTH_PROB = 0.25
NPTH_COUNT_WEIGHTS = ((2, 0.45), (3, 0.15), (4, 0.40))
NPTH_DRILL_WEIGHTS = ((3.2, 0.50), (2.2, 0.20), (2.6, 0.15), (3.0, 0.15))
SLOT_PROB = 0.10
SLOT_W = (0.6, 1.0)              # narrow dim; long dim = w * U[1.3, 3.0]
SLOT_LEN_MULT = (1.3, 3.0)
THT_MODERN_PROB = 0.5            # boards with realistic THT drills/shapes
THT_DRILL_WEIGHTS = ((0.8, 0.5), (0.9, 0.3), (1.0, 0.2))
THT_ANNULUS = (0.4, 0.6)         # pad dia = drill + annulus  -> 1.2..1.6
THT_SHAPE_WEIGHTS = (("circle", 0.5), ("oval", 0.3), ("rect", 0.2))
THT_OVAL_EXTRA = 0.4             # oval pad long dim = pad + 0.4 (capped)

# --- shape-census mode (--shape-census; 2026-08-18 census of d3b_train 275
# boards: 8,691 SMD / 11,847 THT pads). Opt-in so existing dataset seeds stay
# byte-reproducible; when on, THT weights switch to the census mix and SMD
# pads (legacy: always roundrect) draw a per-net census shape. Rare families
# (trapezoid/custom, <1%) are folded into the dominant buckets.
THT_SHAPE_WEIGHTS_CENSUS = (("oval", 0.49), ("circle", 0.40), ("rect", 0.11))
SMD_SHAPE_WEIGHTS_CENSUS = (("rect", 0.92), ("oval", 0.04),
                            ("roundrect", 0.03), ("circle", 0.01))
SMD_OVAL_LEN_MULT = (1.4, 2.2)   # oval SMD long dim = pad_sz * U[...]

EDGE_CLEAR = 0.3                 # copper-to-Edge.Cuts margin for placement
STROKE = '    (stroke (width 0.15) (type solid))'


def _u(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _weighted(rng: random.Random, pairs):
    vals = [v for v, _ in pairs]
    wts = [w for _, w in pairs]
    return rng.choices(vals, weights=wts, k=1)[0]


def _fmt(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _emit_line(rng, out, x1, y1, x2, y2):
    out.append("  (gr_line")
    out.append(f"    (start {_fmt(x1)} {_fmt(y1)})")
    out.append(f"    (end {_fmt(x2)} {_fmt(y2)})")
    out.append(STROKE)
    out.append('    (layer "Edge.Cuts")')
    out.append(f'    (uuid "{_u(rng)}")')
    out.append("  )")


def _emit_arc(rng, out, x1, y1, xm, ym, x2, y2):
    out.append("  (gr_arc")
    out.append(f"    (start {_fmt(x1)} {_fmt(y1)})")
    out.append(f"    (mid {_fmt(xm)} {_fmt(ym)})")
    out.append(f"    (end {_fmt(x2)} {_fmt(y2)})")
    out.append(STROKE)
    out.append('    (layer "Edge.Cuts")')
    out.append(f'    (uuid "{_u(rng)}")')
    out.append("  )")


def _emit_circle(rng, out, cx, cy, r):
    out.append("  (gr_circle")
    out.append(f"    (center {_fmt(cx)} {_fmt(cy)})")
    out.append(f"    (end {_fmt(cx + r)} {_fmt(cy)})")
    out.append(STROKE)
    out.append("    (fill none)")
    out.append('    (layer "Edge.Cuts")')
    out.append(f'    (uuid "{_u(rng)}")')
    out.append("  )")


# --- outlines --------------------------------------------------------------


@dataclass(frozen=True)
class RectOutline:
    """Plain rectangle emitted as 4 gr_line segments (never gr_rect)."""
    w: float
    h: float

    def contains(self, x: float, y: float, margin: float) -> bool:
        return (margin <= x <= self.w - margin
                and margin <= y <= self.h - margin)

    def area(self) -> float:
        return self.w * self.h

    def corners(self):
        w, h = self.w, self.h
        return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    def emit(self, rng, out) -> None:
        w, h = self.w, self.h
        pts = [(0, 0), (w, 0), (w, h), (0, h)]
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            _emit_line(rng, out, ax, ay, bx, by)


@dataclass(frozen=True)
class FilletRectOutline:
    """Rectangle with all four corners filleted (4 gr_line + 4 gr_arc)."""
    w: float
    h: float
    r: float

    def contains(self, x: float, y: float, margin: float) -> bool:
        w, h, r = self.w, self.h, self.r
        if not (margin <= x <= w - margin and margin <= y <= h - margin):
            return False
        for cx, cy in ((r, r), (w - r, r), (w - r, h - r), (r, h - r)):
            # is (x, y) in the corner quadrant beyond this arc center?
            ox = x < cx if cx < w / 2 else x > cx
            oy = y < cy if cy < h / 2 else y > cy
            if ox and oy and math.hypot(x - cx, y - cy) > r - margin:
                return False
        return True

    def area(self) -> float:
        return self.w * self.h - (4.0 - math.pi) * self.r * self.r

    def corners(self):
        w, h = self.w, self.h
        return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    def emit(self, rng, out) -> None:
        w, h, r = self.w, self.h, self.r
        k = r * (1.0 - 1.0 / math.sqrt(2.0))
        _emit_line(rng, out, r, 0, w - r, 0)          # top
        _emit_arc(rng, out, w - r, 0, w - k, k, w, r)  # top-right
        _emit_line(rng, out, w, r, w, h - r)          # right
        _emit_arc(rng, out, w, h - r, w - k, h - k, w - r, h)  # bottom-right
        _emit_line(rng, out, w - r, h, r, h)          # bottom
        _emit_arc(rng, out, r, h, k, h - k, 0, h - r)  # bottom-left
        _emit_line(rng, out, 0, h - r, 0, r)          # left
        _emit_arc(rng, out, 0, r, k, k, r, 0)         # top-left


@dataclass(frozen=True)
class PolyOutline:
    """Rectilinear polygon: base rectangle minus axis-aligned cut rects."""
    w: float
    h: float
    verts: tuple            # closed boundary, consecutive pairs = gr_lines
    cuts: tuple             # removed rects (x0, y0, x1, y1) for contains()

    def contains(self, x: float, y: float, margin: float) -> bool:
        if not (margin <= x <= self.w - margin
                and margin <= y <= self.h - margin):
            return False
        for x0, y0, x1, y1 in self.cuts:
            if (x0 - margin < x < x1 + margin
                    and y0 - margin < y < y1 + margin):
                return False
        return True

    def area(self) -> float:
        a = self.w * self.h
        for x0, y0, x1, y1 in self.cuts:
            a -= (x1 - x0) * (y1 - y0)
        return a

    def corners(self):
        w, h = self.w, self.h
        return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    def emit(self, rng, out) -> None:
        vs = list(self.verts)
        for (ax, ay), (bx, by) in zip(vs, vs[1:] + vs[:1]):
            _emit_line(rng, out, ax, ay, bx, by)


@dataclass(frozen=True)
class CircleOutline:
    """Circular board: one gr_circle. Its bbox is square by construction, so
    an aspect draw does not apply — ``sample_geo`` feeds it the geometric mean
    side and ``w``/``h`` report that square box back to the caller."""
    dia: float

    @property
    def w(self) -> float:
        return self.dia

    @property
    def h(self) -> float:
        return self.dia

    def contains(self, x: float, y: float, margin: float) -> bool:
        r = self.dia / 2.0
        return math.hypot(x - r, y - r) <= r - margin

    def area(self) -> float:
        return math.pi * (self.dia / 2.0) ** 2

    def corners(self):
        # ring positions at the four diagonals stand in for corners
        r = self.dia / 2.0
        d = r / math.sqrt(2.0)
        return [(r - d, r - d), (r + d, r - d), (r + d, r + d), (r - d, r + d)]

    def emit(self, rng, out) -> None:
        r = self.dia / 2.0
        _emit_circle(rng, out, r, r, r)


def _poly_verts(rng: random.Random, w: float, h: float, template: str):
    """Vertex list + cut rects for one rectilinear template (base box w x h).

    Cut sizes are fractions of the axis they run along (``fx`` of the width,
    ``fy`` of the height), so a stretched board keeps proportionate cuts. The
    draw order is unchanged from the square-only version: with ``w == h`` the
    emitted geometry is identical.
    """
    fx = lambda lo, hi: rng.uniform(lo, hi) * w
    fy = lambda lo, hi: rng.uniform(lo, hi) * h

    if template == "L":            # one corner cut (6 lines)
        cw, ch = fx(0.25, 0.45), fy(0.25, 0.45)
        verts = [(cw, 0), (w, 0), (w, h), (0, h), (0, ch), (cw, ch)]
        cuts = [(0, 0, cw, ch)]
    elif template == "notch":      # one edge notch (8 lines)
        nw, nd = fx(0.20, 0.45), fy(0.15, 0.40)
        t = rng.uniform(0.15 + nw / w / 2, 0.85 - nw / w / 2) * w
        x0, x1 = t - nw / 2, t + nw / 2
        verts = [(0, 0), (x0, 0), (x0, nd), (x1, nd), (x1, 0), (w, 0),
                 (w, h), (0, h)]
        cuts = [(x0, 0, x1, nd)]
    elif template == "T":          # both top corners cut (8 lines)
        cw0, cw1 = fx(0.20, 0.40), fx(0.20, 0.40)
        ch = fy(0.20, 0.40)
        verts = [(cw0, 0), (w - cw1, 0), (w - cw1, ch), (w, ch), (w, h),
                 (0, h), (0, ch), (cw0, ch)]
        cuts = [(0, 0, cw0, ch), (w - cw1, 0, w, ch)]
    elif template == "diag":       # two diagonal corners cut (8 lines)
        cw0, ch0 = fx(0.20, 0.40), fy(0.20, 0.40)
        cw2, ch2 = fx(0.20, 0.40), fy(0.20, 0.40)
        verts = [(cw0, 0), (w, 0), (w, h - ch2), (w - cw2, h - ch2),
                 (w - cw2, h), (0, h), (0, ch0), (cw0, ch0)]
        cuts = [(0, 0, cw0, ch0), (w - cw2, h - ch2, w, h)]
    elif template == "L_notch":    # corner cut + opposite-edge notch (10)
        cw, ch = fx(0.20, 0.40), fy(0.20, 0.40)
        nw, nd = fx(0.20, 0.40), fy(0.15, 0.35)
        t = rng.uniform(0.15 + nw / w / 2, 0.85 - nw / w / 2) * w
        x0, x1 = t - nw / 2, t + nw / 2
        verts = [(cw, 0), (w, 0), (w, h), (x1, h), (x1, h - nd), (x0, h - nd),
                 (x0, h), (0, h), (0, ch), (cw, ch)]
        cuts = [(0, 0, cw, ch), (x0, h - nd, x1, h)]
    else:
        raise ValueError(f"unknown poly template {template!r}")
    return tuple(verts), tuple(cuts)


# --- holes / THT profile ---------------------------------------------------


@dataclass(frozen=True)
class Hole:
    """NPTH hole; round when w == h, else an axis-aligned oval slot."""
    x: float
    y: float
    w: float
    h: float

    def blocks(self, x: float, y: float, margin: float) -> bool:
        # capsule distance: segment along the long axis, radius = short/2
        rw = min(self.w, self.h) / 2.0
        half = (max(self.w, self.h) - min(self.w, self.h)) / 2.0
        if self.w >= self.h:
            dx = min(max(x, self.x - half), self.x + half) - x
            dy = self.y - y
        else:
            dx = self.x - x
            dy = min(max(y, self.y - half), self.y + half) - y
        return math.hypot(dx, dy) < rw + margin

    def emit(self, rng, out, ref: str) -> None:
        shape = "circle" if self.w == self.h else "oval"
        out.append('  (footprint "SamplePad:NPTH"')
        out.append('    (layer "F.Cu")')
        out.append(f"    (at {_fmt(self.x)} {_fmt(self.y)})")
        out.append(f'    (uuid "{_u(rng)}")')
        out.append(f'    (property "Reference" "{ref}"')
        out.append("      (at 0 -1)")
        out.append('      (layer "F.SilkS")')
        out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))")
        out.append("    )")
        out.append(f'    (property "Value" "Hole"')
        out.append("      (at 0 1)")
        out.append('      (layer "F.Fab")')
        out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))")
        out.append("    )")
        out.append(f'    (pad "" np_thru_hole {shape}')
        out.append("      (at 0 0)")
        out.append(f"      (size {_fmt(self.w)} {_fmt(self.h)})")
        if self.w == self.h:
            out.append(f"      (drill {_fmt(self.w)})")
        else:
            out.append(f"      (drill oval {_fmt(self.w)} {_fmt(self.h)})")
        out.append('      (layers "*.Cu" "*.Mask")')
        out.append(f'      (uuid "{_u(rng)}")')
        out.append("    )")
        out.append("  )")
        out.append("")


@dataclass(frozen=True)
class Cutout:
    x: float
    y: float
    r: float

    def blocks(self, x: float, y: float, margin: float) -> bool:
        return math.hypot(x - self.x, y - self.y) < self.r + margin

    def emit(self, rng, out) -> None:
        _emit_circle(rng, out, self.x, self.y, self.r)


@dataclass(frozen=True)
class ThtProfile:
    """Per-board THT pad family: realistic drill + shape mix.

    ``None`` profile keeps the legacy pads (circle, rule pad size, via drill).
    """
    drill: float
    pad: float           # circle dia / rect side / oval short dim
    oval_len: float      # oval long dim
    # --shape-census swaps in THT_SHAPE_WEIGHTS_CENSUS (default keeps the
    # legacy mix so existing dataset seeds stay byte-reproducible).
    shape_weights: tuple = THT_SHAPE_WEIGHTS

    @property
    def max_dim(self) -> float:
        return self.oval_len

    def sample_net_pad(self, rng: random.Random):
        shape = _weighted(rng, self.shape_weights)
        if shape == "oval":
            horizontal = rng.random() < 0.5
            w, h = ((self.oval_len, self.pad) if horizontal
                    else (self.pad, self.oval_len))
            return "oval", w, h, self.drill
        return shape, self.pad, self.pad, self.drill


@dataclass
class BoardGeo:
    """Sampled geometry for one board; consumed by placement and render."""
    outline: object
    cutouts: list = field(default_factory=list)
    holes: list = field(default_factory=list)
    tht: ThtProfile | None = None

    def allows_pad(self, x: float, y: float, pad_half: float,
                   clearance: float) -> bool:
        if not self.outline.contains(x, y, pad_half + EDGE_CLEAR):
            return False
        m = pad_half + clearance
        for c in self.cutouts:
            if c.blocks(x, y, m):
                return False
        for hole in self.holes:
            if hole.blocks(x, y, m):
                return False
        return True

    def usable_frac(self) -> float:
        """Approximate usable-area fraction vs the base box (capacity derate)."""
        box = self.outline.w * self.outline.h
        frac = self.outline.area() / box
        blocked = 0.0
        for c in self.cutouts:
            blocked += math.pi * (c.r + 1.5) ** 2
        for hole in self.holes:
            blocked += math.pi * (max(hole.w, hole.h) / 2.0 + 1.5) ** 2
        return max(0.3, frac - blocked / box)

    def emit_edge_cuts(self, rng, out) -> None:
        self.outline.emit(rng, out)
        for c in self.cutouts:
            c.emit(rng, out)

    def emit_hole_footprints(self, rng, out) -> None:
        for i, hole in enumerate(self.holes):
            hole.emit(rng, out, ref=f"H{i + 1}")


def sample_smd_pad(rng: random.Random, pad_sz: float):
    """Per-net SMD shape draw for --shape-census: (shape, w, h).

    Always consumes exactly 3 rng draws (shape, orientation coin, oval
    elongation) regardless of the chosen shape, so the downstream sample
    stream is independent of the outcome.
    """
    shape = _weighted(rng, SMD_SHAPE_WEIGHTS_CENSUS)
    horizontal = rng.random() < 0.5
    length = pad_sz * rng.uniform(*SMD_OVAL_LEN_MULT)
    if shape == "oval":
        w, h = (length, pad_sz) if horizontal else (pad_sz, length)
        return "oval", w, h
    return shape, pad_sz, pad_sz


def sample_geo(rng: random.Random, w: float, h: float,
               shape_census: bool = False) -> BoardGeo:
    """Draw one board's geometry inside a ``w x h`` box. Draw order is FIXED —
    changing it changes every downstream sample for a given seed.

    The circle family ignores the box shape (a circle's bbox is square): it
    takes the geometric mean side, so callers must read the realized box back
    from ``geo.outline.w/h`` rather than assuming the requested one."""
    kind = _weighted(rng, OUTLINE_WEIGHTS)
    if kind == "fillet":
        r = _weighted(rng, FILLET_RADIUS_WEIGHTS)
        r = min(r, 0.15 * min(w, h))
        outline = FilletRectOutline(w, h, r)
    elif kind == "poly":
        template = _weighted(rng, POLY_TEMPLATE_WEIGHTS)
        verts, cuts = _poly_verts(rng, w, h, template)
        outline = PolyOutline(w, h, verts, cuts)
    elif kind == "circle":
        outline = CircleOutline(math.sqrt(w * h))
    else:
        outline = RectOutline(w, h)

    geo = BoardGeo(outline=outline)
    w, h = outline.w, outline.h   # circle collapses back to a square box

    # NPTH mounting holes near (surviving) corners
    if rng.random() < NPTH_PROB:
        count = _weighted(rng, NPTH_COUNT_WEIGHTS)
        drill = _weighted(rng, NPTH_DRILL_WEIGHTS)
        inset = drill / 2.0 + EDGE_CLEAR + rng.uniform(1.5, 3.5)
        corners = list(outline.corners())
        rng.shuffle(corners)
        for cx, cy in corners[:count]:
            hx = cx + (inset if cx < w / 2 else -inset)
            hy = cy + (inset if cy < h / 2 else -inset)
            if outline.contains(hx, hy, drill / 2.0 + EDGE_CLEAR):
                geo.holes.append(Hole(hx, hy, drill, drill))

    # oval-drill slots along a random edge
    if rng.random() < SLOT_PROB:
        n_slots = 1 if rng.random() < 0.7 else 2
        for _ in range(n_slots):
            sw = rng.uniform(*SLOT_W)
            length = sw * rng.uniform(*SLOT_LEN_MULT)
            edge = rng.randrange(4)
            t = rng.uniform(0.2, 0.8) * (w if edge in (0, 2) else h)
            inset = length / 2.0 + EDGE_CLEAR + rng.uniform(1.0, 3.0)
            if edge == 0:
                hx, hy, hw, hh = t, inset, length, sw
            elif edge == 1:
                hx, hy, hw, hh = w - inset, t, sw, length
            elif edge == 2:
                hx, hy, hw, hh = t, h - inset, length, sw
            else:
                hx, hy, hw, hh = inset, t, sw, length
            if outline.contains(hx, hy, max(hw, hh) / 2.0 + EDGE_CLEAR):
                ok = all(not h.blocks(hx, hy, max(hw, hh)) for h in geo.holes)
                if ok:
                    geo.holes.append(Hole(hx, hy, hw, hh))

    # internal circular cutouts
    if rng.random() < CUTOUT_PROB:
        n_cut = 1 if rng.random() < 0.8 else 2
        for _ in range(n_cut):
            r = rng.uniform(*CUTOUT_DIA) / 2.0
            for _try in range(50):
                cx = rng.uniform(r + 3.0, w - r - 3.0)
                cy = rng.uniform(r + 3.0, h - r - 3.0)
                if not outline.contains(cx, cy, r + 2.0):
                    continue
                if any(h.blocks(cx, cy, r + 1.0) for h in geo.holes):
                    continue
                if any(c.blocks(cx, cy, r + 1.0) for c in geo.cutouts):
                    continue
                geo.cutouts.append(Cutout(cx, cy, r))
                break

    # THT pad profile. Census mode swaps only the SHAPE weights — profile
    # presence stays coin-gated (THT_MODERN_PROB): forcing the profile on
    # every board would also change drill/pad-size distributions and derate
    # capacity on the former-fallback half (density is the loaded variable —
    # the d2b_geo lesson). Cost: aggregate THT shape mix stays circle-diluted
    # by the legacy fallback (~circle .70/oval .25 vs real .40/.49); the
    # census weights are exact within profile boards only.
    if rng.random() < THT_MODERN_PROB:
        drill = _weighted(rng, THT_DRILL_WEIGHTS)
        pad = drill + rng.uniform(*THT_ANNULUS)
        geo.tht = ThtProfile(drill=drill, pad=pad,
                             oval_len=pad + THT_OVAL_EXTRA,
                             shape_weights=(THT_SHAPE_WEIGHTS_CENSUS
                                            if shape_census
                                            else THT_SHAPE_WEIGHTS))
    return geo
