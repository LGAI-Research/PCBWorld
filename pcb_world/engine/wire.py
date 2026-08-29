"""Engine-IPC wire schema — the interface definition between the two programs.

Plain mirror types for every pybind class the ``kicad_rl_router`` binding
returns, the authoritative field-order registry (``KRL_FIELDS``), the module
constants snapshotted into the handshake, and the ``to_wire``/``from_wire``
codec. The engine server serializes binding objects with ``to_wire``; the
environment reconstructs them with ``from_wire``. Field names mirror the
binding's declaration order exactly, so consumers (observation builders, DRC
helpers) never see a KiCad type.

This module is stdlib-only and carries no logic from either program: it is
the protocol both of them speak. An identical copy ships in the other
program's repository — the engine's ``engine_server/wire.py`` and the
environment's ``pcb_world/engine/wire.py`` — so neither program has to
import a module from the other at runtime. The two copies are kept byte for
byte identical (checked by the environment's ``tools/check_separation.py``),
which is why this file carries no per-file licence header: each copy is
covered by the LICENSE of the repository it sits in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


# ===========================================================================
# Engine-IPC wire schema: plain mirrors of the pybind binding types
# ===========================================================================

class TrackInfo(NamedTuple):
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    width_mm: float
    layer: int
    net_code: int
    net_name: str
    uuid: str


class ViaInfo(NamedTuple):
    x_mm: float
    y_mm: float
    diameter_mm: float
    drill_mm: float
    top_layer: int
    bottom_layer: int
    net_code: int
    net_name: str
    uuid: str


class PadInfo(NamedTuple):
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    layer: int
    net_code: int
    net_name: str
    pad_name: str
    footprint_ref: str
    pad_type: str
    shape: str


class RatsnestEdge(NamedTuple):
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    net_code: int
    layer1: int
    layer2: int


class ClusterPoint(NamedTuple):
    x_mm: float
    y_mm: float
    layer: int


class ZoneInfo(NamedTuple):
    pts: list          # [(x_mm, y_mm), ...]
    layer: int
    keepout_tracks: bool
    keepout_vias: bool
    keepout_pads: bool
    name: str


class BoardEdge(NamedTuple):
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    width_mm: float


class BoardOutlineShape(NamedTuple):
    kind: str
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    x3_mm: float
    y3_mm: float
    width_mm: float


class GraphicShape(NamedTuple):
    index: int
    kind: str
    x1_nm: int
    y1_nm: int
    xm_nm: int
    ym_nm: int
    x2_nm: int
    y2_nm: int
    width_nm: int


class DRCViolation(NamedTuple):
    error_code: int
    error_type: str
    message: str
    x_mm: float
    y_mm: float
    layer: int
    net_names: list
    severity: int
    item_a: str
    item_b: str


class BoundingBox(NamedTuple):
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float


class FootprintInfo(NamedTuple):
    ref: str
    value: str
    fpid: str
    x_mm: float
    y_mm: float
    orientation_deg: float
    flipped: bool
    layer: int
    courtyard: list       # [[(x_mm, y_mm), ...], ...] — one closed contour each


class CleanupItem(NamedTuple):
    code: int
    code_name: str
    item_a: str
    item_b: str


# Mutable mirrors: consumers edit fields then pass back (set_design_rules).
@dataclass
class NetClassInfo:
    name: str = ""
    clearance_mm: float = -1.0
    track_width_mm: float = -1.0
    via_diameter_mm: float = -1.0
    via_drill_mm: float = -1.0
    uvia_diameter_mm: float = -1.0
    uvia_drill_mm: float = -1.0


@dataclass
class DesignRules:
    min_clearance_mm: float = -1.0
    min_track_width_mm: float = -1.0
    min_via_diameter_mm: float = -1.0
    min_through_hole_mm: float = -1.0
    min_via_annular_width_mm: float = -1.0
    min_hole_to_hole_mm: float = -1.0
    min_uvia_diameter_mm: float = -1.0
    min_uvia_drill_mm: float = -1.0
    copper_edge_clearance_mm: float = -1.0
    track_width_presets_mm: list = field(default_factory=list)
    via_presets_mm: list = field(default_factory=list)
    default_netclass: NetClassInfo = field(default_factory=NetClassInfo)
    netclasses: list = field(default_factory=list)  # [NetClassInfo, ...]


@dataclass
class CleanupResult:
    """Track-cleaner result: the wire mirror of the binding's ``CleanupResult``.

    Field declaration order = binding declaration order (registered below).

    ``ran`` is False when a precondition rejected the call — today only an open
    routing/drag session (``reject_reason``); nothing was inspected or changed.
    ``items`` holds the CleanupItem entries (pybind objects in-process, plain
    :class:`CleanupItem` mirrors over IPC — ``code_name``/``item_a``/``item_b``
    access is identical) in execution order; ``removed`` / ``modified`` are the
    affected item UUIDs as strings and stay empty on a dry run.
    """

    ran: bool = False
    reject_reason: str = ""
    items: list = field(default_factory=list)      # CleanupItem entries
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when the board was actually mutated (never true for a dry run)."""
        return bool(self.removed or self.modified)

    def counts(self) -> dict[str, int]:
        """Operation count per cleanup code name ("merge_tracks": 3, …)."""
        out: dict[str, int] = {}
        for item in self.items:
            out[item.code_name] = out.get(item.code_name, 0) + 1
        return out


# Authoritative field ORDER per wire type (= declaration order in
# pns_rl_bindings.cpp). The server validates this registry against the live
# binding at startup (constant handshake) — a binding field added/renamed
# without updating this registry fails loudly on both sides.
_WIRE_TYPES = {
    "TrackInfo": TrackInfo,
    "ViaInfo": ViaInfo,
    "PadInfo": PadInfo,
    "RatsnestEdge": RatsnestEdge,
    "ClusterPoint": ClusterPoint,
    "ZoneInfo": ZoneInfo,
    "BoardEdge": BoardEdge,
    "BoardOutlineShape": BoardOutlineShape,
    "GraphicShape": GraphicShape,
    "DRCViolation": DRCViolation,
    "BoundingBox": BoundingBox,
    "FootprintInfo": FootprintInfo,
    "CleanupItem": CleanupItem,
    "NetClassInfo": NetClassInfo,
    "DesignRules": DesignRules,
    "CleanupResult": CleanupResult,
}

KRL_FIELDS: dict[str, tuple] = {
    name: (
        cls._fields if issubclass(cls, tuple)
        else tuple(cls.__dataclass_fields__)
    )
    for name, cls in _WIRE_TYPES.items()
}

# Module-level constants of kicad_rl_router snapshotted into the handshake
# (the NC client never imports the module, so these come over the wire).
KRL_CONSTANT_NAMES = (
    "LAYER_EDGE_CUTS", "LAYER_MARGIN",
    "MODE_MARK_OBSTACLES", "MODE_SHOVE", "MODE_WALKAROUND",
    "CORNER_MITERED_45", "CORNER_ROUNDED_45", "CORNER_MITERED_90",
    "CORNER_ROUNDED_90",
    "DM_CORNER", "DM_SEGMENT", "DM_VIA", "DM_FREE_ANGLE", "DM_ARC",
    "DM_ANY", "DM_COMPONENT",
    "STATE_IDLE", "STATE_DRAG_SEGMENT", "STATE_DRAG_COMPONENT",
    "STATE_ROUTE_TRACK",
    "F_Cu", "B_Cu",
)

_PRIMITIVES = (type(None), bool, int, float, str, bytes)
_WIRE_TAG = "__krl__"


def to_wire(value):
    """Encode a binding return value into primitives-only structures.

    Accepts primitives, lists/tuples/dicts (recursed), and any object
    whose type name is registered in ``KRL_FIELDS`` (pybind object or
    plain mirror alike — encoding is getattr-based). Unknown object types
    raise: the wire never silently degrades.
    """
    if isinstance(value, _PRIMITIVES):
        return value
    tname = type(value).__name__
    if tname in KRL_FIELDS:
        return (_WIRE_TAG, tname,
                tuple(to_wire(getattr(value, f)) for f in KRL_FIELDS[tname]))
    if isinstance(value, (list, tuple)):
        encoded = [to_wire(v) for v in value]
        # A plain tuple must not alias the tagged-tuple form.
        if isinstance(value, tuple):
            return ("__tuple__", encoded)
        return encoded
    if isinstance(value, dict):
        return {k: to_wire(v) for k, v in value.items()}
    raise TypeError(
        f"engine IPC: unserializable return type {type(value)!r} — register "
        "it in KRL_FIELDS (wire.py, both copies)")


def from_wire(value):
    """Decode ``to_wire`` output into plain mirror objects."""
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, tuple):
        if value and value[0] == _WIRE_TAG:
            _, tname, fields = value
            return _WIRE_TYPES[tname](*(from_wire(f) for f in fields))
        if value and value[0] == "__tuple__":
            return tuple(from_wire(v) for v in value[1])
        return tuple(from_wire(v) for v in value)
    if isinstance(value, list):
        return [from_wire(v) for v in value]
    if isinstance(value, dict):
        return {k: from_wire(v) for k, v in value.items()}
    raise TypeError(f"engine IPC: cannot decode wire value {value!r}")
