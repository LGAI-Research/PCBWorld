"""Engine-backed view of board state in legacy parser dict shape.

``KiCadEngine`` (kicad_rl_router) is the single source of truth for board
state. This module reshapes the engine's output into the dict layout that
``BoardStatic.from_board`` and existing tests expect.

Pads carry **human** layer IDs (1=Top, N=Bottom) — engine natives use
``PCB_LAYER_ID`` (board IDs), so we wrap them in thin shims that
present the same attribute names the legacy parser produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pcb_world.engine.containers import BoardMeta, BoardSnapshot
from pcb_world.engine.utils import (  # noqa: F401  re-exports
    load_and_save_via_engine,
)

#: Valid values for the ``outline_mode`` of :func:`parse_pcb_file` — how
#: Edge.Cuts arcs/circles are represented in ``board_edges`` (and therefore in
#: the obs ``boardlines``):
#:   "tess"   — C++-side error-bounded tessellation into straight segments
#:              (m_MaxError = 0.005 mm; circles as 32-gons). Legacy/current path.
#:   "poly16" — fixed 16-segments-per-90° re-tessellation, rebuilt Python-side
#:              from engine arc primitives.
#:   "arc"    — one entry per arc/circle, carrying the on-arc midpoint
#:              (``x3_mm``/``y3_mm``); straight segments unchanged.
OUTLINE_MODES = ("poly16", "tess", "arc")


@dataclass
class BoardEdge:
    """One outline (Edge.Cuts) segment in mm — legacy dict shape.

    ``x3_mm``/``y3_mm`` (outline_mode="arc" only): the on-arc midpoint of an
    arc entry — KiCad's native 3-point arc form. ``None`` = straight segment.
    A full circle is encoded as ``p1 == p2`` (a point on the circle) with the
    midpoint at its antipode, so the diameter is spanned exactly.
    """
    x1_mm: float = 0.0
    y1_mm: float = 0.0
    x2_mm: float = 0.0
    y2_mm: float = 0.0
    width_mm: float = 0.0
    x3_mm: float | None = None
    y3_mm: float | None = None


class _PadView:
    """Engine ``PadInfo`` re-exposed with a *human* ``layer`` ID.

    The legacy parser placed thru-hole pads on layer ``0`` (sentinel:
    "spans the whole copper stack"); SMD/connect pads on the human
    layer they sit on. This wrapper preserves that contract.
    """

    __slots__ = ("_p", "layer")

    def __init__(self, pad, human_layer: int) -> None:
        self._p = pad
        self.layer = human_layer

    @property
    def x_mm(self):       return self._p.x_mm
    @property
    def y_mm(self):       return self._p.y_mm
    @property
    def width_mm(self):   return self._p.width_mm
    @property
    def height_mm(self):  return self._p.height_mm
    @property
    def net_code(self):   return self._p.net_code
    @property
    def net_name(self):   return self._p.net_name
    @property
    def shape(self):      return self._p.shape
    @property
    def pad_type(self):   return self._p.pad_type
    @property
    def pad_name(self):   return self._p.pad_name
    @property
    def footprint_ref(self): return self._p.footprint_ref


class _TrackView:
    __slots__ = ("_t", "layer")

    def __init__(self, track, layer_map) -> None:
        self._t = track
        try:
            self.layer = layer_map.board_to_human(track.layer)
        except KeyError as e:
            raise ValueError(
                f"Track parsing failed: cannot map board layer {track.layer!r} "
                f"to human layer (net={track.net_name!r}, "
                f"x1={track.x1_mm}, y1={track.y1_mm}, "
                f"x2={track.x2_mm}, y2={track.y2_mm})"
            ) from e

    @property
    def x1_mm(self):    return self._t.x1_mm
    @property
    def y1_mm(self):    return self._t.y1_mm
    @property
    def x2_mm(self):    return self._t.x2_mm
    @property
    def y2_mm(self):    return self._t.y2_mm
    @property
    def width_mm(self): return self._t.width_mm
    @property
    def net_code(self): return self._t.net_code
    @property
    def net_name(self): return self._t.net_name


class _ViaView:
    __slots__ = ("_v", "top_layer", "bottom_layer")

    def __init__(self, via, layer_map) -> None:
        self._v = via
        try:
            self.top_layer = layer_map.board_to_human(via.top_layer)
        except KeyError as e:
            raise ValueError(
                f"Via parsing failed: cannot map top board layer {via.top_layer!r} "
                f"to human layer (net={via.net_name!r}, "
                f"x={via.x_mm}, y={via.y_mm})"
            ) from e
        try:
            self.bottom_layer = layer_map.board_to_human(via.bottom_layer)
        except KeyError as e:
            raise ValueError(
                f"Via parsing failed: cannot map bottom board layer {via.bottom_layer!r} "
                f"to human layer (net={via.net_name!r}, "
                f"x={via.x_mm}, y={via.y_mm})"
            ) from e

    @property
    def x_mm(self):        return self._v.x_mm
    @property
    def y_mm(self):        return self._v.y_mm
    @property
    def diameter_mm(self): return self._v.diameter_mm
    @property
    def drill_mm(self):    return self._v.drill_mm
    @property
    def net_code(self):    return self._v.net_code
    @property
    def net_name(self):    return self._v.net_name


class _KeepoutView:
    """Rule-area keepout zone exposed as a polygon obstacle.

    Presents the same geometry attributes as ``_PadView`` (bbox centre + size,
    for consumers that only understand axis-aligned rectangles) plus ``pts``
    — the true outline polygon — and ``shape == "polygon"``. Never attached to
    a net (``net_code == -1``). ``layer`` is a *human* layer ID.
    """

    __slots__ = (
        "pts", "layer", "name",
        "keepout_tracks", "keepout_vias", "keepout_pads",
        "_cx", "_cy", "_w", "_h",
    )

    def __init__(self, zone, human_layer: int) -> None:
        self.pts = [(float(x), float(y)) for x, y in zone.pts]
        self.layer = human_layer
        self.name = zone.name
        self.keepout_tracks = zone.keepout_tracks
        self.keepout_vias = zone.keepout_vias
        self.keepout_pads = zone.keepout_pads
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        self._cx = (min(xs) + max(xs)) / 2.0
        self._cy = (min(ys) + max(ys)) / 2.0
        self._w = max(xs) - min(xs)
        self._h = max(ys) - min(ys)

    @property
    def x_mm(self):      return self._cx
    @property
    def y_mm(self):      return self._cy
    @property
    def width_mm(self):  return self._w
    @property
    def height_mm(self): return self._h
    @property
    def shape(self):     return "polygon"
    @property
    def net_code(self):  return -1
    @property
    def net_name(self):  return ""


# ---------------------------------------------------------------------------
# Outline arc geometry (outline_mode "poly16" / "arc")
# ---------------------------------------------------------------------------

def _arc_center(x1: float, y1: float, xm: float, ym: float,
                x2: float, y2: float) -> tuple[float, float] | None:
    """Circumcenter of the 3 arc points, or ``None`` when collinear."""
    d = 2.0 * (x1 * (ym - y2) + xm * (y2 - y1) + x2 * (y1 - ym))
    chord2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if abs(d) < 1e-9 * max(1.0, chord2):
        return None
    s1 = x1 * x1 + y1 * y1
    sm = xm * xm + ym * ym
    s2 = x2 * x2 + y2 * y2
    cx = (s1 * (ym - y2) + sm * (y2 - y1) + s2 * (y1 - ym)) / d
    cy = (s1 * (x2 - xm) + sm * (x1 - x2) + s2 * (xm - x1)) / d
    return cx, cy


def _arc_angles(x1: float, y1: float, xm: float, ym: float,
                x2: float, y2: float,
                cx: float, cy: float) -> tuple[float, float]:
    """(start_angle, signed_sweep) with the sweep direction passing through mid."""
    two_pi = 2.0 * math.pi
    a1 = math.atan2(y1 - cy, x1 - cx)
    am = math.atan2(ym - cy, xm - cx)
    a2 = math.atan2(y2 - cy, x2 - cx)
    sweep_ccw = (a2 - a1) % two_pi
    mid_ccw = (am - a1) % two_pi
    if sweep_ccw == 0.0:  # p1 == p2: full circle, direction from mid
        return a1, two_pi if mid_ccw > 0.0 else -two_pi
    if mid_ccw <= sweep_ccw:
        return a1, sweep_ccw
    return a1, sweep_ccw - two_pi


def _arc_extreme_points(x1: float, y1: float, xm: float, ym: float,
                        x2: float, y2: float) -> list[tuple[float, float]]:
    """Axis-extreme points the arc traverses (for exact bbox; endpoints excluded)."""
    if (x1, y1) == (x2, y2):
        # Full circle (p1 == p2, mid = antipode): the three points are
        # collinear so the circumcenter path below cannot apply — derive the
        # circle from the diameter and emit all four axis extremes.
        cx, cy = (x1 + xm) / 2.0, (y1 + ym) / 2.0
        r = math.hypot(x1 - cx, y1 - cy)
        return [(cx + r, cy), (cx - r, cy), (cx, cy + r), (cx, cy - r)]
    c = _arc_center(x1, y1, xm, ym, x2, y2)
    if c is None:
        return []
    cx, cy = c
    r = math.hypot(x1 - cx, y1 - cy)
    a1, sweep = _arc_angles(x1, y1, xm, ym, x2, y2, cx, cy)
    two_pi = 2.0 * math.pi
    pts = []
    for k in range(4):  # angles 0, π/2, π, 3π/2
        a = k * math.pi / 2.0
        off = (a - a1) % two_pi if sweep >= 0 else (a1 - a) % two_pi
        if off <= abs(sweep):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _tessellate_arc_poly16(x1: float, y1: float, xm: float, ym: float,
                           x2: float, y2: float) -> list[tuple[float, float]]:
    """Polyline points of the fixed 16-segments-per-90° tessellation scheme."""
    c = _arc_center(x1, y1, xm, ym, x2, y2)
    if c is None:  # collinear/degenerate arc: emit the chord
        return [(x1, y1), (x2, y2)]
    cx, cy = c
    r = math.hypot(x1 - cx, y1 - cy)
    a1, sweep = _arc_angles(x1, y1, xm, ym, x2, y2, cx, cy)
    # KiCad's native C++ computes the sweep from the exact int-nm center, so
    # an exact quarter fillet always yields n = 16. Recomputing the
    # circumcenter here from mm-rounded doubles can land the sweep ~1e-7 deg
    # below 90, which bare int() would truncate to 15 — the epsilon corrects
    # for that without affecting genuinely sub-90 arcs.
    n = max(2, int(16 * abs(sweep) * 180.0 / math.pi / 90.0 + 1e-6))
    return [
        (cx + r * math.cos(a1 + sweep * i / n), cy + r * math.sin(a1 + sweep * i / n))
        for i in range(n + 1)
    ]


def _board_edges_from_shapes(shapes, mode: str) -> tuple[list[BoardEdge], list[tuple[float, float]]]:
    """Convert engine outline primitives to ``BoardEdge`` entries for one mode.

    Returns ``(edges, extra_bbox_pts)`` — the extra points are arc/circle
    axis-extremes that endpoints alone would miss when computing the board bbox
    (mode "arc" only; tessellated modes cover them with segment endpoints).
    """
    edges: list[BoardEdge] = []
    bbox_pts: list[tuple[float, float]] = []
    for s in shapes:
        if s.kind == 0:  # straight segment
            edges.append(BoardEdge(s.x1_mm, s.y1_mm, s.x2_mm, s.y2_mm, s.width_mm))
        elif s.kind in (1, 2):  # arc / circle (circle: p1 == p2, mid = antipode)
            if mode == "arc":
                edges.append(BoardEdge(
                    s.x1_mm, s.y1_mm, s.x2_mm, s.y2_mm, s.width_mm,
                    x3_mm=s.x3_mm, y3_mm=s.y3_mm,
                ))
                bbox_pts.extend(_arc_extreme_points(
                    s.x1_mm, s.y1_mm, s.x3_mm, s.y3_mm, s.x2_mm, s.y2_mm,
                ))
            else:  # poly16
                if s.kind == 2:
                    # Circle: 32-gon starting at p1, same as the C++ CIRCLE case
                    # (16-per-90° over a full turn == 32 segments).
                    cx = (s.x1_mm + s.x3_mm) / 2.0
                    cy = (s.y1_mm + s.y3_mm) / 2.0
                    r = math.hypot(s.x1_mm - cx, s.y1_mm - cy)
                    a0 = math.atan2(s.y1_mm - cy, s.x1_mm - cx)
                    two_pi = 2.0 * math.pi
                    pts = [
                        (cx + r * math.cos(a0 + two_pi * i / 32),
                         cy + r * math.sin(a0 + two_pi * i / 32))
                        for i in range(33)
                    ]
                else:
                    pts = _tessellate_arc_poly16(
                        s.x1_mm, s.y1_mm, s.x3_mm, s.y3_mm, s.x2_mm, s.y2_mm,
                    )
                for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                    edges.append(BoardEdge(ax, ay, bx, by, s.width_mm))
        else:
            raise ValueError(f"Unknown BoardOutlineShape kind: {s.kind!r}")
    return edges, bbox_pts


def parse_pcb_file(pcb_path: str | Path, engine, outline_mode: str = "tess") -> dict:
    """Build the legacy parser dict from a live ``KiCadEngine``.

    ``pcb_path`` is accepted only to keep the call-site signature stable
    across the migration; it is not read here — the engine has already
    loaded the file.

    ``outline_mode`` selects the Edge.Cuts representation in ``board_edges``
    (see :data:`OUTLINE_MODES`); everything downstream of ``board_edges`` is
    mode-agnostic.
    """
    engine_meta = engine.get_board_meta()
    layer_map = engine.layer_map

    pads: list = []
    obstacles: list = []
    for p in engine.get_pads():
        if p.pad_type == "thru_hole" or p.layer < 0:
            # 0 = spans-copper-stack sentinel. The engine reports multi-layer
            # copper pads (including NPTH) as RL_LAYER_SPANS_COPPER (-2).
            human_layer = 0
        else:
            try:
                human_layer = layer_map.board_to_human(p.layer)
            except KeyError as e:
                raise ValueError(
                    f"Pad parsing failed: cannot map board layer {p.layer!r} "
                    f"to human layer (pad_type={p.pad_type}, net={p.net_name!r}, "
                    f"x={p.x_mm}, y={p.y_mm})"
                ) from e
        wrapped = _PadView(p, human_layer)
        if p.pad_type == "np_thru_hole":
            obstacles.append(wrapped)
        else:
            pads.append(wrapped)

    # Rule-area keepout zones, exposed as polygon obstacles alongside the
    # NPTH pads above. Each engine entry is already one zone per copper layer.
    keepout_count = 0
    for z in getattr(engine, "get_keepouts", lambda: [])():
        try:
            human_layer = layer_map.board_to_human(z.layer)
        except KeyError:
            continue  # non-copper / unmapped layer: skip (router still honors it)
        obstacles.append(_KeepoutView(z, human_layer))
        keepout_count += 1

    tracks = [_TrackView(t, layer_map) for t in engine.get_tracks()]
    vias   = [_ViaView(v, layer_map)   for v in engine.get_vias()]
    rats   = list(engine.get_ratsnest())

    snapshot = BoardSnapshot(
        tracks=tracks,
        vias=vias,
        pads=pads,
        ratsnest=rats,
        track_count=engine.get_track_count(),
        unrouted_count=engine.get_unrouted_count(),
    )

    if outline_mode == "tess":
        board_edges = [
            BoardEdge(e.x1_mm, e.y1_mm, e.x2_mm, e.y2_mm, e.width_mm)
            for e in engine.get_board_outline()
        ]
        outline_bbox_pts: list[tuple[float, float]] = []
    elif outline_mode in ("poly16", "arc"):
        board_edges, outline_bbox_pts = _board_edges_from_shapes(
            engine.get_board_outline_shapes(), outline_mode,
        )
    else:
        raise ValueError(
            f"outline_mode must be one of {OUTLINE_MODES}, got {outline_mode!r}"
        )

    # Match the legacy parser: bbox is computed from the board outline
    # (Edge.Cuts polylines), not from the engine's whole-board bbox.
    if not board_edges:
        raise ValueError(
            f"Board parsing failed: no Edge.Cuts outline segments found "
            f"(pcb_path={pcb_path!r})"
        )
    all_x = ([e.x1_mm for e in board_edges] + [e.x2_mm for e in board_edges]
             + [p[0] for p in outline_bbox_pts])
    all_y = ([e.y1_mm for e in board_edges] + [e.y2_mm for e in board_edges]
             + [p[1] for p in outline_bbox_pts])
    bbox_x = min(all_x)
    bbox_y = min(all_y)
    bbox_w = max(all_x) - bbox_x
    bbox_h = max(all_y) - bbox_y

    meta = BoardMeta(
        bbox_x=bbox_x, bbox_y=bbox_y,
        bbox_w=bbox_w, bbox_h=bbox_h,
        net_count=engine_meta.net_count,
        copper_layers=engine_meta.copper_layers,
    )

    # Match legacy: include net_code 0 ("") so callers that iterate
    # over net_names see the same set of keys as before.
    net_names = {0: ""}
    net_names.update(engine.get_net_names())

    parse_stats = {
        "nets": meta.net_count,
        "copper_layers": meta.copper_layers,
        "pads": len(pads),
        "tracks": len(tracks),
        "vias": len(vias),
        "ratsnest_edges": len(rats),
        "board_edges": len(board_edges),
        "obstacles": len(obstacles),
        "keepouts": keepout_count,
        "thru_hole_pads": sum(1 for p in pads if p.layer == 0),
        "bbox": (
            f"({meta.bbox_x:.1f}, {meta.bbox_y:.1f}) "
            f"{meta.bbox_w:.1f}x{meta.bbox_h:.1f} mm"
        ),
    }

    return {
        "board_meta": meta,
        "board_snapshot": snapshot,
        "net_names": net_names,
        "board_edges": board_edges,
        "obstacles": obstacles,
        "parse_stats": parse_stats,
    }
