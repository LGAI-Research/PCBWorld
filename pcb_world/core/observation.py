"""Observation builder for the KiCad RL environment (JSON format).

Converts raw data from engine_api types into json-friendly observations.
No C++ imports — operates purely on BoardSnapshot, RoutingSessionState, DRCResult.

Coordinate system:
- All coordinates are in **mm** (real PCB units). No normalization applied.
- Consumers (MLP wrappers, LLM agents) normalize as needed using board_meta.

Architecture:
- Flat dataclass primitives composed via containment (no inheritance).
- Point -> Line, Rectangle, Circle, Via (composition)
- Same primitive reused across contexts (e.g., Rectangle for both pads and obstacles).
- Semantic meaning comes from position in hierarchy, not from class type.

ID scheme:
- Hierarchical, net-scoped string IDs with type prefixes.
- Static context (board edges, pads): global scope with prefixes E, D, P.
- Dynamic state (tracks, vias, ratsnest): net-scoped with prefixes P, T, V, Q.
- Within a net, shared coordinates receive the same Point ID (deduplication).
- Numbers restart from 0 per net per type, keeping IDs small for LLM consumption.

Output format:
- build_json_observation() -> dict with board_meta, board_static, routing_geometry, router_head
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pcb_world.engine.containers import (
    BoardMeta,
    BoardSnapshot,
    RoutingSessionState,
)

# Flag: warn once per process when the loaded .so lacks
# RatsnestEdge.layer1/2 — see _rats_layers.
_WARNED_LEGACY_RATS_LAYER = False


# ---------------------------------------------------------------------------
# Layer 1: Geometric Primitives
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    """2D position (x, y) without layer context."""
    id: str = ""
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "xy": [self.x, self.y]}


@dataclass
class Point3D:
    """3D position (x, y, layer) in PCB space."""
    id: str = ""
    x: float = 0.0
    y: float = 0.0
    layer: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "xy": [self.x, self.y], "layer": self.layer}


@dataclass
class Line:
    """Track segment or board edge. Both endpoints share the same layer.

    ``mid`` (board-outline arcs, outline_obs="arc" only): the on-arc midpoint —
    the entry is an arc through p1→mid→p2 (KiCad's native 3-point form; a full
    circle has p1 == p2 with mid at the antipode). ``None`` = straight segment.
    The key is emitted only when set, so straight-only obs stay byte-identical.
    """
    id: str = ""
    p1: Point2D = field(default_factory=Point2D)
    p2: Point2D = field(default_factory=Point2D)
    width: float = 0.0
    layer: int = 0
    mid: Point2D | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "p1": self.p1.to_dict(),
            "p2": self.p2.to_dict(),
            "width": self.width,
            "layer": self.layer,
        }
        if self.mid is not None:
            d["mid"] = self.mid.to_dict()
        return d


@dataclass
class Rectangle:
    """Pad or obstacle. Center is 2D; layer is a property of the pad."""
    id: str = ""
    center: Point2D = field(default_factory=Point2D)
    width: float = 0.0
    height: float = 0.0
    layer: int = 0
    # KiCad shape keyword ("rect", "circle", "oval", "roundrect", ...).
    # "" when unknown (obstacles / C++ bindings that don't carry shape).
    # Consumers: the visualizer, and the RL tokenizer — shape_bucket_id
    # feeds the shape_obs channel, and shape == "polygon" marks rule-area
    # keepout entries (excluded from OBSTACLE token emission).
    shape: str = ""
    # Absolute (x_mm, y_mm) outline vertices — only for polygon shapes such as
    # rule-area keepout zones (shape == "polygon"). Empty for plain rectangles;
    # width/height/center still hold the bounding box so rect-only consumers work.
    pts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "center": self.center.to_dict(),
            "width": self.width,
            "height": self.height,
            "layer": self.layer,
            "shape": self.shape,
        }
        # pts holds polygon outline vertices only for polygon shapes (rule-area
        # keepout zones); plain pads/rects leave it empty. Emit the key ONLY when
        # populated, so pad obs stay byte-identical to the RL indexed-obs path — which
        # never carries pts, a visualization-only field the RL/LLM paths ignore.
        if self.pts:
            d["pts"] = [list(p) for p in self.pts]
        return d


@dataclass
class Circle:
    """Circular shape. Center is 2D; layer is a property of the shape."""
    id: str = ""
    center: Point2D = field(default_factory=Point2D)
    r: float = 0.0
    width: float = 0.0
    layer: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "center": self.center.to_dict(),
            "r": self.r,
            "width": self.width,
            "layer": self.layer,
        }


@dataclass
class Via:
    """Via spanning layers. Center is 2D; layer range is separate."""
    id: str = ""
    center: Point2D = field(default_factory=Point2D)
    layer_start: int = 1     # human layer (1=Top)
    layer_end: int = 2       # human layer (N=Bottom)
    via_width: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "center": self.center.to_dict(),
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "via_width": self.via_width,
        }


# Board-level design rules are exposed via ``board_static["board_constraints"]``
# (populated by env.py from the engine). Per-net constraints live on
# ``NetContext.constraints`` — real per-netclass resolved values (opt-in via
# the env ``net_constraint_obs`` knob).


# ---------------------------------------------------------------------------
# ID Allocation (net-scoped, type-prefixed string IDs)
# ---------------------------------------------------------------------------

class IDAllocator:
    """Type-prefixed ID allocator with point deduplication.

    Each (prefix) type has an independent counter starting from 0,
    producing IDs like "P0", "P1", "T0", "T1", "V0", etc.

    Points sharing the same normalized coordinates (within tolerance)
    receive the same ID, enabling topology reasoning.
    """

    def __init__(self, point_tol: int = 3) -> None:
        self._point_tol = point_tol
        self._counters: dict[str, int] = defaultdict(int)
        self._point2d_registry: dict[tuple[float, float], Point2D] = {}
        self._point3d_registry: dict[tuple[float, float, int], Point3D] = {}

    def next_id(self, prefix: str) -> str:
        """Allocate and return the next unique ID for the given prefix."""
        idx = self._counters[prefix]
        self._counters[prefix] += 1
        return f"{prefix}{idx}"

    def get_or_create_point2d(
        self, x: float, y: float, prefix: str = "P",
    ) -> Point2D:
        """Return existing Point2D if (x,y) match, else create new one."""
        key = (round(x, self._point_tol), round(y, self._point_tol))
        if key not in self._point2d_registry:
            self._point2d_registry[key] = Point2D(
                id=self.next_id(prefix), x=x, y=y,
            )
        return self._point2d_registry[key]

    def get_or_create_point3d(
        self, x: float, y: float, layer: int, prefix: str = "Q",
    ) -> Point3D:
        """Return existing Point3D if (x,y,layer) match, else create new one."""
        key = (round(x, self._point_tol), round(y, self._point_tol), layer)
        if key not in self._point3d_registry:
            self._point3d_registry[key] = Point3D(
                id=self.next_id(prefix), x=x, y=y, layer=layer,
            )
        return self._point3d_registry[key]


# ---------------------------------------------------------------------------
# Layer 2: Semantic Containers (compose primitives)
# ---------------------------------------------------------------------------

@dataclass
class NetContext:
    """Static per-net info (built once at reset)."""
    net_code: int = 0
    net_name: str = ""
    pads: list[Rectangle] = field(default_factory=list)
    # Resolved per-net DRC constraints (netclass value, KiCad inherit →
    # Default fallback, BDS global-min clamp) — the values the engine is
    # actually driven with on net_select. ``None`` (default) leaves them out
    # of the observation entirely; populated by env.py only when
    # ``net_constraint_obs`` is on. Keys: track_width / clearance /
    # via_diameter / via_drill (mm).
    constraints: dict | None = None

    def to_dict(self) -> dict:
        out = {
            "net_name": self.net_name,
            "pads": {f"pad_{i}": p.to_dict() for i, p in enumerate(self.pads)},
        }
        if self.constraints is not None:
            out["constraints"] = dict(self.constraints)
        return out


@dataclass
class NetGeometry:
    """Dynamic per-net geometric objects (updated every step)."""
    net_code: int = 0
    tracks: list[Line] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    points: list[Point3D] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tracks": {f"track_{i}": t.to_dict() for i, t in enumerate(self.tracks)},
            "vias": {f"via_{i}": v.to_dict() for i, v in enumerate(self.vias)},
            "points": [p.to_dict() for p in self.points],
        }


# ---------------------------------------------------------------------------
# Layer 3: Board-level container
# ---------------------------------------------------------------------------

@dataclass
class BoardStatic:
    """Board metadata + static context (cached at reset).

    ``board_constraints`` is the engine-sourced strictest-per-netclass
    dict produced by ``compute_hardest_per_netclass`` — the single source
    of board-level DRC constraints. Fields unset across BDS + every
    netclass are reported as ``-1.0`` so consumers can skip them.
    """
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_w: float = 1.0
    bbox_h: float = 1.0
    scale: float = 1.0
    net_count: int = 1
    copper_layers: int = 2
    # Static context (primitive composition)
    boardlines: list[Line] = field(default_factory=list)
    nets: dict[int, NetContext] = field(default_factory=dict)
    obstacles: list[Rectangle] = field(default_factory=list)
    # Pads that have no net assignment (net_code <= 0): NC SMD/PTH pads
    # and ``connect`` pads that the schematic left dangling. They must
    # not show up in ``nets`` (which is keyed by net_code), but tracks
    # still need to clear from them physically. Surfaced separately so
    # the visualizer can render them as obstacles without polluting the
    # LLM-facing per-net pad list.
    unconnected_pads: list[Rectangle] = field(default_factory=list)
    board_constraints: dict = field(default_factory=dict)

    @classmethod
    def from_meta(cls, meta: BoardMeta) -> BoardStatic:
        """Create with metadata only (no static context)."""
        w = max(meta.bbox_w, 1e-6)
        h = max(meta.bbox_h, 1e-6)
        return cls(
            bbox_x=meta.bbox_x,
            bbox_y=meta.bbox_y,
            bbox_w=w,
            bbox_h=h,
            scale=max(w, h),
            net_count=meta.net_count,
            copper_layers=meta.copper_layers,
        )

    @classmethod
    def from_board(
        cls,
        meta: BoardMeta,
        pads: list,
        board_edges: list,
        net_names: dict[int, str],
        board_constraints: dict,
        obstacles: list | None = None,
        target_nets: frozenset[int] | None = None,
    ) -> BoardStatic:
        """Create with full static context from parsed board data.

        ``board_constraints`` is expected to be the engine-sourced
        strictest-per-netclass dict (see ``compute_hardest_per_netclass``);
        it's stored on the instance and surfaced via
        ``_build_board_static`` without further interpretation here.

        ``obstacles`` is an optional list of pad-shaped blockers — the
        parser routes ``np_thru_hole`` pads here (non-plated mounting
        holes). Each element is duck-typed like ``PadInfoCompat``
        (``x_mm``/``y_mm``/``width_mm``/``height_mm``/``layer``/``shape``).

        ``target_nets`` (net-subset / partial routing): when given, only
        these net codes populate ``nets`` (the per-net routable-target view
        the model sees). Pads of any other net are demoted to
        ``unconnected_pads`` — they stay visible as obstacles (the router
        still physically clears from them) but drop out of the routable-net
        list, ratsnest, and termination accounting. ``None`` (default) keeps
        every net_code>0 pad as a routable net (whole-board path).

        All coordinates and dimensions are in mm (no normalization).

        Static context uses global-scope prefixed IDs:
          E = board edge, D = pad rectangle, O = obstacle rectangle,
          P = shared 2D point.
        """
        w = max(meta.bbox_w, 1e-6)
        h = max(meta.bbox_h, 1e-6)
        scale = max(w, h)

        alloc = IDAllocator()

        # Board edge lines (mm)
        boardlines = []
        for e in board_edges:
            p1 = alloc.get_or_create_point2d(
                e.x1_mm, e.y1_mm, prefix="P",
            )
            p2 = alloc.get_or_create_point2d(
                e.x2_mm, e.y2_mm, prefix="P",
            )
            mid = None
            if getattr(e, "x3_mm", None) is not None:
                mid = alloc.get_or_create_point2d(
                    e.x3_mm, e.y3_mm, prefix="P",
                )
            boardlines.append(Line(
                id=alloc.next_id("E"), p1=p1, p2=p2,
                width=e.width_mm, layer=44, mid=mid,
            ))

        # Group pads by net and build NetContext. Pads without a net
        # (net_code <= 0) are collected separately below for the
        # visualizer; they're routing obstacles even though no net
        # owns them.
        pads_by_net: dict[int, list] = defaultdict(list)
        unconnected_pads_input: list = []
        for p in pads:
            if p.net_code > 0 and (target_nets is None or p.net_code in target_nets):
                pads_by_net[p.net_code].append(p)
            else:
                # net_code <= 0 (NC pads) OR a non-target net under net-subset
                # routing: kept as an obstacle-pad, dropped from the routable
                # per-net view.
                unconnected_pads_input.append(p)

        nets: dict[int, NetContext] = {}
        for nc, net_pads in pads_by_net.items():
            pad_rects = []
            for p in net_pads:
                center = alloc.get_or_create_point2d(
                    p.x_mm, p.y_mm, prefix="P",
                )
                pad_rects.append(Rectangle(
                    id=alloc.next_id("D"), center=center,
                    width=p.width_mm, height=p.height_mm,
                    layer=p.layer,
                    shape=getattr(p, "shape", ""),
                ))
            nets[nc] = NetContext(
                net_code=nc,
                net_name=net_names.get(nc, f"net_{nc}"),
                pads=pad_rects,
            )

        # Obstacle rectangles — NPTH mounting holes etc. Same geometry
        # fields as a pad but never attached to a net.
        obstacle_rects: list[Rectangle] = []
        for o in (obstacles or []):
            center = alloc.get_or_create_point2d(o.x_mm, o.y_mm, prefix="P")
            obstacle_rects.append(Rectangle(
                id=alloc.next_id("O"), center=center,
                width=o.width_mm, height=o.height_mm,
                layer=o.layer,
                shape=getattr(o, "shape", ""),
                pts=[tuple(p) for p in getattr(o, "pts", [])],
            ))

        # Unconnected pads — same geometry as a pad, no net. The "U"
        # prefix keeps their IDs distinguishable from real pads ("D")
        # and obstacles ("O") in any downstream debug output.
        unconnected_pad_rects: list[Rectangle] = []
        for p in unconnected_pads_input:
            center = alloc.get_or_create_point2d(p.x_mm, p.y_mm, prefix="P")
            unconnected_pad_rects.append(Rectangle(
                id=alloc.next_id("U"), center=center,
                width=p.width_mm, height=p.height_mm,
                layer=p.layer,
                shape=getattr(p, "shape", ""),
            ))

        return cls(
            bbox_x=meta.bbox_x, bbox_y=meta.bbox_y, bbox_w=w, bbox_h=h,
            scale=scale, net_count=meta.net_count,
            copper_layers=meta.copper_layers,
            boardlines=boardlines, nets=nets,
            obstacles=obstacle_rects,
            unconnected_pads=unconnected_pad_rects,
            board_constraints=board_constraints,
        )


# ---------------------------------------------------------------------------
# Observation Builders
# ---------------------------------------------------------------------------

def build_net_geometry(
    snapshot: BoardSnapshot,
    board_info: BoardStatic,
    layer_map=None,
    target_nets: frozenset[int] | None = None,
) -> dict[int, NetGeometry]:
    """Build per-net dynamic state with net-scoped, type-prefixed IDs.

    All coordinates are in mm (no normalization).

    Each net gets its own IDAllocator, so IDs restart from 0 per net:
      P = Point2D (track endpoints, via centers)
      T = Track (Line)
      V = Via
      Q = Point3D (ratsnest endpoints)

    Points sharing the same coordinates within a net are deduplicated.

    ``target_nets`` (net-subset / partial routing): when given, only the
    RATSNEST of nets outside this set is dropped, so the "still-to-connect"
    signal (and ``net_valid_mask``) covers only the target nets. Tracks/vias are
    kept for EVERY net regardless — existing copper on non-target / kept nets
    stays visible (obstacle context + viewer render). ``None`` (default) keeps
    every net's ratsnest too (whole-board path).

    Returns:
        dict[net_code -> NetGeometry] with mm coordinates and
        net-scoped string IDs (e.g. "P0", "T1", "V0", "Q2").
    """
    def _b2h(board_layer: int) -> int:
        """Board layer -> human layer (safe fallback to 1)."""
        if layer_map is None:
            return board_layer
        try:
            return layer_map.board_to_human(board_layer)
        except KeyError:
            return 1

    def _rats_layers(e) -> tuple[int, int]:
        """Ratsnest endpoint human layer (0 = multi-layer/thru sentinel,
        matching pad_layer convention).

        The engine reports edge.layer1/layer2 (PCB_LAYER_ID, -1 = multi-layer)
        when the binding provides them; a binding lacking the fields falls
        back to the constant 1 and warns once per process.
        """
        lay1 = getattr(e, "layer1", None)
        if lay1 is None:
            global _WARNED_LEGACY_RATS_LAYER
            if not _WARNED_LEGACY_RATS_LAYER:
                _WARNED_LEGACY_RATS_LAYER = True
                import warnings
                warnings.warn(
                    "kicad_rl_router.so predates RatsnestEdge.layer1/2 — "
                    "ratsnest Q tokens fall back to legacy layer=1; rebuild "
                    "the engine (engine/build_rl_router.sh).",
                    RuntimeWarning, stacklevel=2,
                )
            return 1, 1
        # Negative = RL_LAYER_SPANS_COPPER (-2, multi-layer parent) → human
        # 0 (thru) sentinel.
        return (0 if lay1 < 0 else _b2h(lay1),
                0 if e.layer2 < 0 else _b2h(e.layer2))

    # Group raw data by net_code first, then build with per-net allocators.
    #
    # IMPORTANT: snapshot.tracks / .vias / .ratsnest come from the C++ engine
    # in a non-deterministic order (e.g. std::unordered_* iteration or
    # pointer-sorted containers). Identical board states can yield different
    # list orders across calls. To make the observation deterministic — and
    # therefore make argmax-deterministic policy rollouts reproducible —
    # we sort each per-net bucket here by a stable geometric key.
    tracks_by_net: dict[int, list] = defaultdict(list)
    vias_by_net: dict[int, list] = defaultdict(list)
    ratsnest_by_net: dict[int, list] = defaultdict(list)

    # Tracks / vias are shown for EVERY net (existing copper stays visible even
    # for non-target / kept nets — the router must clear from it and the viewer
    # must render it). Only the RATSNEST (the "still-to-connect" signal that
    # drives net selection) is restricted to the target set, so non-target /
    # kept nets are never routing targets.
    def _rats_in_scope(net_code: int) -> bool:
        return net_code > 0 and (target_nets is None or net_code in target_nets)

    for t in snapshot.tracks:
        if t.net_code > 0:
            tracks_by_net[t.net_code].append(t)
    for v in snapshot.vias:
        if v.net_code > 0:
            vias_by_net[v.net_code].append(v)
    for e in snapshot.ratsnest:
        if _rats_in_scope(e.net_code):
            ratsnest_by_net[e.net_code].append(e)

    for k in tracks_by_net:
        tracks_by_net[k].sort(key=lambda t: (
            t.layer, t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm, t.width_mm,
        ))
    for k in vias_by_net:
        vias_by_net[k].sort(key=lambda v: (
            v.x_mm, v.y_mm, v.top_layer, v.bottom_layer, v.diameter_mm,
        ))
    for k in ratsnest_by_net:
        ratsnest_by_net[k].sort(key=lambda e: (
            e.x1_mm, e.y1_mm, e.x2_mm, e.y2_mm,
        ))

    # set() iteration order is also a latent source of nondeterminism
    # across runs; sort the net codes too.
    all_net_codes = sorted(
        set(tracks_by_net) | set(vias_by_net) | set(ratsnest_by_net)
    )
    nets: dict[int, NetGeometry] = {}

    for nc in all_net_codes:
        alloc = IDAllocator()
        ns = NetGeometry(net_code=nc)

        for t in tracks_by_net.get(nc, []):
            p1 = alloc.get_or_create_point2d(
                t.x1_mm, t.y1_mm, prefix="P",
            )
            p2 = alloc.get_or_create_point2d(
                t.x2_mm, t.y2_mm, prefix="P",
            )
            ns.tracks.append(Line(
                id=alloc.next_id("T"), p1=p1, p2=p2,
                width=t.width_mm, layer=_b2h(t.layer),
            ))

        for v in vias_by_net.get(nc, []):
            center = alloc.get_or_create_point2d(
                v.x_mm, v.y_mm, prefix="P",
            )
            ns.vias.append(Via(
                id=alloc.next_id("V"), center=center,
                layer_start=_b2h(v.top_layer), layer_end=_b2h(v.bottom_layer),
                via_width=v.diameter_mm,
            ))

        for e in ratsnest_by_net.get(nc, []):
            l1, l2 = _rats_layers(e)
            p1 = alloc.get_or_create_point3d(
                e.x1_mm, e.y1_mm, layer=l1, prefix="Q",
            )
            p2 = alloc.get_or_create_point3d(
                e.x2_mm, e.y2_mm, layer=l2, prefix="Q",
            )
            ns.points.append(p1)
            ns.points.append(p2)

        nets[nc] = ns

    return nets


def _build_routing_geometry(snapshot: BoardSnapshot, board_info: BoardStatic, layer_map=None) -> dict:
    """Build dynamic routing state as JSON dict (called every step)."""
    net_states = build_net_geometry(snapshot, board_info, layer_map=layer_map)
    return {f"net_{nc}": ns.to_dict() for nc, ns in net_states.items()}


def _build_board_static(board_info: BoardStatic) -> dict:
    """Build static context including board metadata (called once at reset, or cached)."""
    return {
        # Board metadata (normalization reference)
        "bbox_x": board_info.bbox_x,
        "bbox_y": board_info.bbox_y,
        "bbox_w": board_info.bbox_w,
        "bbox_h": board_info.bbox_h,
        "scale": board_info.scale,
        "net_count": board_info.net_count,
        "copper_layers": board_info.copper_layers,
        # Static geometry
        "boardlines": {
            f"edge_{i}": {k: v for k, v in l.to_dict().items() if k != "layer"}
            for i, l in enumerate(board_info.boardlines)
        },
        "nets": {f"net_{nc}": ctx.to_dict() for nc, ctx in board_info.nets.items()},
        "obstacles": {f"obs_{i}": o.to_dict() for i, o in enumerate(board_info.obstacles)},
        "unconnected_pads": {
            f"upad_{i}": p.to_dict()
            for i, p in enumerate(board_info.unconnected_pads)
        },
        "board_constraints": dict(board_info.board_constraints),
    }


def _build_router_head(
    router_state: RoutingSessionState,
    step_count: int,
    max_steps: int,
    current_net_id: int | None = None,
    routing_mode: int = 2,
) -> dict:
    """Build router session state.

    Derives current_net_phase from observable state:
        0 = no net selected (current_net_id is None)
        1 = net selected, not actively routing
        2 = actively routing
    """
    # Derive phase from observable state
    if current_net_id is None:
        net_phase = 0
    elif not router_state.is_routing:
        net_phase = 1
    else:
        net_phase = 2

    return {
        "current_xy": [router_state.route_head[0], router_state.route_head[1]],
        "current_layer": router_state.current_layer,
        "current_net": current_net_id if current_net_id is not None else -1,
        "router_state_code": router_state.state_code,
        "via_toggle": router_state.is_placing_via,
        "is_routing": router_state.is_routing,
        "is_dragging": router_state.is_dragging,
        "routing_mode": routing_mode,
        "current_net_phase": net_phase,
        "step": step_count,
        "step_ratio": step_count / max(max_steps, 1),
        "steps_remaining": max(max_steps - step_count, 0),
    }


def build_json_observation(
    snapshot: BoardSnapshot,
    router_state: RoutingSessionState,
    board_info: BoardStatic,
    step_count: int = 0,
    max_steps: int = 100,
    current_net_id: int | None = None,
    routing_mode: int = 2,
    *,
    board_static: dict | None = None,
    net_geometry: dict[int, NetGeometry] | None = None,
    drc_violations: list[dict] | None = None,
    action_history: list[dict] | None = None,
    closed_nets: list[int] | None = None,
) -> dict:
    """Build the full hierarchical JSON state dictionary.

    All coordinates are in mm (no normalization). Consumers can normalize
    using board_static (bbox, scale).

    Args:
        snapshot: BoardSnapshot from engine_api.
        router_state: RoutingSessionState from engine_api.
        board_info: Cached board metadata (with static context).
        step_count: Current step number in the episode.
        max_steps: Maximum steps per episode.
        current_net_id: Currently selected net (None if no net selected).
            Used to derive current_net_phase from observable state.
        routing_mode: Current routing mode (0=MarkObstacles, 1=Shove, 2=Walkaround).
        board_static: Pre-built static context dict (skip rebuild if provided).
        net_geometry: Pre-built canonical IR (skip rebuild if provided).

    Returns:
        Dict with three top-level keys:
        - board_static: board metadata (bbox, scale, net_count, copper_layers)
          and static context (boardlines, nets/pads, obstacles, constraints)
        - routing_geometry: dynamic routing geometry (tracks, vias, ratsnest points per net)
        - router_head: router session state (position, layer, net, flags, phase)
    """
    if net_geometry is None:
        net_geometry = build_net_geometry(snapshot, board_info)
    routing_geometry = {f"net_{nc}": ns.to_dict() for nc, ns in net_geometry.items()}

    return {
        "board_static": board_static if board_static is not None else _build_board_static(board_info),
        "routing_geometry": routing_geometry,
        "router_head": _build_router_head(
            router_state, step_count, max_steps, current_net_id, routing_mode,
        ),
        "drc_violations": drc_violations if drc_violations is not None else [],
        # Recent action records, newest first (up to env action_history_len;
        # shorter near episode start, [] right after reset).
        "action_history": action_history if action_history is not None else [],
        # Nets consumed by net_end this episode (re-selection is masked out;
        # every routable net closed => early `terminated`). PCBWorld.step §3.
        "closed_nets": closed_nets if closed_nets is not None else [],
    }
