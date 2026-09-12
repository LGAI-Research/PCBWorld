"""Aggregate data containers assembled by ``KiCadEngine``.

The per-item wire mirrors (``TrackInfo``, ``PadInfo``, …) and the wire codec
live in :mod:`pcb_world.engine.wire` — the protocol module shared with the
engine repository; they are re-exported here so existing consumers keep one
import site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pcb_world.engine.wire import (  # noqa: F401  re-exports
    KRL_CONSTANT_NAMES,
    KRL_FIELDS,
    BoardEdge,
    BoardOutlineShape,
    BoundingBox,
    CleanupItem,
    CleanupResult,
    ClusterPoint,
    DesignRules,
    DRCViolation,
    FootprintInfo,
    GraphicShape,
    NetClassInfo,
    PadInfo,
    RatsnestEdge,
    TrackInfo,
    ViaInfo,
    ZoneInfo,
    from_wire,
    to_wire,
)


@dataclass
class BoardMeta:
    """Board metadata (static, captured once at reset)."""

    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_w: float = 1.0
    bbox_h: float = 1.0
    net_count: int = 1
    copper_layers: int = 2


@dataclass
class BoardSnapshot:
    """Board state snapshot (captured every step).

    Lists contain TrackInfo / PadInfo / RatsnestEdge items: plain wire
    mirrors (above) in engine-IPC mode, pybind11 objects passed by
    reference in in-process mode. Field access is identical either way.
    """

    tracks: list = field(default_factory=list)
    vias: list = field(default_factory=list)
    pads: list = field(default_factory=list)
    ratsnest: list = field(default_factory=list)
    track_count: int = 0
    unrouted_count: int = 0


@dataclass
class RoutingSessionState:
    """Routing/dragging session state."""

    state_code: int = 0           # 0=IDLE, 1=DRAG_SEGMENT, 2=DRAG_COMPONENT, 3=ROUTE_TRACK
    is_routing: bool = False
    is_dragging: bool = False
    is_placing_via: bool = False
    current_layer: int = -1       # human layer (1=Top, N=Bottom, -1=idle)
    route_head: tuple = (0.0, 0.0, -1.0)      # (x_mm, y_mm, human_layer)
    current_net_code: int = -1
    routing_target: tuple = (0.0, 0.0, -1.0)  # (x_mm, y_mm, human_layer)


@dataclass
class DRCResult:
    """DRC execution result."""

    violation_count: int = 0
    violations: list = field(default_factory=list)  # DRCViolation items (wire mirror or pybind)


@dataclass
class RewardSnapshot:
    """Minimal state for reward computation."""

    unrouted_count: int = 0
    track_count: int = 0
    via_count: int = 0
    total_wirelength: float = 0.0
    drc_violation_count: int = 0
    drc_violations_per_net: dict[str, int] = field(default_factory=dict)
    # Severity-split counts (error == 0x20, warning == 0x10).
    # Sum stays in the fields above for backward compatibility.
    drc_error_count: int = 0
    drc_warning_count: int = 0
    drc_errors_per_net: dict[str, int] = field(default_factory=dict)
    drc_warnings_per_net: dict[str, int] = field(default_factory=dict)
    # Errors + promoted warnings (track/via_dangling, net_conflict) — see
    # pcb_world.engine.drc.DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED. Populated
    # unconditionally alongside the other splits so reward/state can switch
    # modes without re-running DRC.
    drc_promoted_count: int = 0
    drc_promoted_per_net: dict[str, int] = field(default_factory=dict)
    # Per-net connectivity (ladder reward bonuses). Universe = target nets
    # under net-subset routing, else all pad-bearing nets. -1 only on legacy
    # construction paths; reward configs that need these fail loudly then.
    connected_net_count: int = -1
    target_net_count: int = -1
    # Net codes (same universe) with >=1 open ratsnest edge — the identity
    # behind ``connected_net_count``. None only on legacy construction paths;
    # size-weighted ladder bonuses fail loudly then.
    unconnected_net_codes: frozenset[int] | None = None
