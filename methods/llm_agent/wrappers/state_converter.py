"""LLM state converter — observation dict → text (XML / S-expression).

One half of ``projection.py`` (with ``action_converter``), mirroring
the RL codec ``methods.rl_agent.models.v1.encoding``. Encodes a PCBWorld observation
into the prompt-facing string, plus the inverse split helpers and the
valid-action presentation derived from the action mask.

What reaches the text is decided per JSON obs element by :data:`OBS_ELEMENTS`
(render or drop, and for which prompt versions): the frozen ``legacy`` versions
render a fixed set, so a new obs field never changes their text; ``latest``
renders that set plus the elements marked ``latest``. An element a knob-gated
obs field feeds is rendered only when the env produced it: the per-net
``constraints`` need ``net_constraint_obs``, which every serializer takes from
the env's :class:`~configs.loader.schema.EnvConfig`.

Public: OBS_ELEMENTS / obs_elements / serialize_obs /
format_state / format_state_split / format_state_sexpr /
format_state_split_sexpr / format_valid_actions / split_board_static.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Tuple

from methods.llm_agent.wrappers.prompts.dispatcher import LATEST, PCBWORLD_PROMPT_VERSIONS
from pcb_world.core.action_schema import ACT_IDLE, ACTION_NAMES
from pcb_world.core.action_schema import MODE_INT_TO_LETTER as _MODE_INT_TO_LETTER


# ===========================================================================
# Obs element table — which JSON obs elements reach the prompt text
# ===========================================================================

class ObsElement(NamedTuple):
    """One JSON obs element: ``kind`` ``render`` (``note`` = how it is drawn) or
    ``drop`` (``note`` = why not; its whole subtree is dropped). A rendered element
    reaches the text of the prompt versions ``since`` … ``until`` (inclusive; ``None``
    = open-ended, ``latest`` counts as newer than every legacy version)."""

    kind: str
    note: str
    since: str = "v1"
    until: Optional[str] = None


def _r(note: str, since: str = "v1", until: Optional[str] = None) -> ObsElement:
    return ObsElement("render", note, since, until)


def _d(reason: str) -> ObsElement:
    return ObsElement("drop", reason)


def _version_rank(version: str) -> float:
    """``v<N>`` -> N; ``latest`` -> newer than every legacy version."""
    return float("inf") if version == LATEST else float(version[1:])


# Keys are JSON paths: ``.`` between dict keys, ``*`` for the numbered key of a
# dict entry (``net_3`` -> ``net_*``), ``[]`` for a list item. Every key of an
# all-obs-knobs-on observation must be listed here or under a dropped parent
# (test_llm_prompt_render.py) — a new obs field needs a decision before it ships.
# The frozen legacy versions render a fixed set (their text is pinned); ``latest``
# renders exactly what the RL tokenizer reads, apart from :data:`PARITY_EXCEPTIONS`
# (tests/test_llm_obs_parity.py). The rendered set feeds each prompt bundle's
# ``content_hash``.
L = LATEST
OBS_ELEMENTS: Dict[str, ObsElement] = {
    "board_static": _r("board_static section"),
    "board_static.bbox_x": _r("bbox"), "board_static.bbox_y": _r("bbox"),
    "board_static.bbox_w": _r("bbox"), "board_static.bbox_h": _r("bbox"),
    "board_static.scale": _r("scale"),
    "board_static.net_count": _r("net_count"),
    "board_static.copper_layers": _r("copper_layers"),
    "board_static.boardlines": _r("boardlines"),
    "board_static.boardlines.edge_*": _r("edge, or arc when it has a mid point"),
    "board_static.boardlines.edge_*.id": _r("edge / arc id"),
    "board_static.boardlines.edge_*.p1": _r("p1"),
    "board_static.boardlines.edge_*.p1.id": _r("p1 id"),
    "board_static.boardlines.edge_*.p1.xy": _r("p1 x y"),
    "board_static.boardlines.edge_*.p2": _r("p2"),
    "board_static.boardlines.edge_*.p2.id": _r("p2 id"),
    "board_static.boardlines.edge_*.p2.xy": _r("p2 x y"),
    "board_static.boardlines.edge_*.mid": _r("mid (arc outline)"),
    "board_static.boardlines.edge_*.mid.id": _r("mid id"),
    "board_static.boardlines.edge_*.mid.xy": _r("mid x y"),
    "board_static.boardlines.edge_*.width": _r("(width W) — outline stroke width", L),
    "board_static.nets": _r("nets"),
    "board_static.nets.net_*": _r("net code + name"),
    "board_static.nets.net_*.net_name": _r("net name"),
    "board_static.nets.net_*.pads": _r("pads"),
    "board_static.nets.net_*.pads.pad_*": _r("pad"),
    "board_static.nets.net_*.pads.pad_*.id": _r("pad id"),
    "board_static.nets.net_*.pads.pad_*.center": _r("pad x y"),
    "board_static.nets.net_*.pads.pad_*.center.xy": _r("pad x y"),
    "board_static.nets.net_*.pads.pad_*.center.id": _d("the pad id names the pad"),
    "board_static.nets.net_*.pads.pad_*.layer": _r("pad layer (th = through-hole)"),
    "board_static.nets.net_*.pads.pad_*.width": _r("(size w h); v1-v8 drop it", L),
    "board_static.nets.net_*.pads.pad_*.height": _r("(size w h); v1-v8 drop it", L),
    "board_static.nets.net_*.pads.pad_*.shape": _r("(shape S); v1-v8 drop it", L),
    "board_static.nets.net_*.constraints": _r("constraints", "v8"),
    "board_static.nets.net_*.constraints.track_width": _r("track_width", "v8"),
    "board_static.nets.net_*.constraints.clearance": _r("clearance", "v8"),
    "board_static.nets.net_*.constraints.via_diameter": _r("via_diameter", "v8"),
    "board_static.nets.net_*.constraints.via_drill": _r("via_drill", "v8", "v8"),
    "board_static.obstacles": _r("obstacles"),
    "board_static.obstacles.obs_*": _r("obs (latest skips rule-area keepouts, shape polygon)"),
    "board_static.obstacles.obs_*.id": _r("obs id"),
    "board_static.obstacles.obs_*.center": _r("obs x y"),
    "board_static.obstacles.obs_*.center.xy": _r("obs x y"),
    "board_static.obstacles.obs_*.center.id": _d("the obs id names the obstacle"),
    "board_static.obstacles.obs_*.width": _r("size w"),
    "board_static.obstacles.obs_*.height": _r("size h"),
    "board_static.obstacles.obs_*.layer": _r("obs layer"),
    "board_static.obstacles.obs_*.shape": _r("(shape S); v1-v8 draw the bounding box", L),
    "board_static.obstacles.obs_*.pts": _d("keepout polygon outline (latest skips keepouts; v1-v8 draw the bbox)"),
    "board_static.unconnected_pads": _r("unconnected_pads; v1-v8 drop them", L),
    "board_static.unconnected_pads.upad_*": _r("upad", L),
    "board_static.unconnected_pads.upad_*.id": _r("upad id", L),
    "board_static.unconnected_pads.upad_*.center": _r("upad x y", L),
    "board_static.unconnected_pads.upad_*.center.xy": _r("upad x y", L),
    "board_static.unconnected_pads.upad_*.center.id": _d("the upad id names the pad"),
    "board_static.unconnected_pads.upad_*.width": _r("(size w h)", L),
    "board_static.unconnected_pads.upad_*.height": _r("(size w h)", L),
    "board_static.unconnected_pads.upad_*.layer": _r("upad layer", L),
    "board_static.unconnected_pads.upad_*.shape": _r("(shape S)", L),
    "board_static.board_constraints": _r("board_constraints", until="v8"),
    "board_static.board_constraints.clearance_mm": _r("clearance", until="v8"),
    "board_static.board_constraints.track_width_mm": _r("track_width", until="v8"),
    "board_static.board_constraints.via_diameter_mm": _r("via_diameter", until="v8"),
    "board_static.board_constraints.via_drill_mm": _r("via_drill", until="v8"),
    "board_static.board_constraints.uvia_diameter_mm": _r("uvia_diameter", until="v8"),
    "board_static.board_constraints.uvia_drill_mm": _r("uvia_drill", until="v8"),
    "routing_geometry": _r("routing_geometry section"),
    "routing_geometry.net_*": _r("net code"),
    "routing_geometry.net_*.tracks": _r("tracks"),
    "routing_geometry.net_*.tracks.track_*": _r("track"),
    "routing_geometry.net_*.tracks.track_*.id": _r("track id"),
    "routing_geometry.net_*.tracks.track_*.layer": _r("track layer"),
    "routing_geometry.net_*.tracks.track_*.p1": _r("p1"),
    "routing_geometry.net_*.tracks.track_*.p1.id": _r("p1 id"),
    "routing_geometry.net_*.tracks.track_*.p1.xy": _r("p1 x y"),
    "routing_geometry.net_*.tracks.track_*.p2": _r("p2"),
    "routing_geometry.net_*.tracks.track_*.p2.id": _r("p2 id"),
    "routing_geometry.net_*.tracks.track_*.p2.xy": _r("p2 x y"),
    "routing_geometry.net_*.tracks.track_*.width": _r("(width W); v1-v8 drop it", L),
    "routing_geometry.net_*.vias": _r("vias"),
    "routing_geometry.net_*.vias.via_*": _r("via"),
    "routing_geometry.net_*.vias.via_*.id": _r("via id"),
    "routing_geometry.net_*.vias.via_*.center": _r("center"),
    "routing_geometry.net_*.vias.via_*.center.id": _r("center id"),
    "routing_geometry.net_*.vias.via_*.center.xy": _r("center x y"),
    "routing_geometry.net_*.vias.via_*.layer_start": _r("layers start"),
    "routing_geometry.net_*.vias.via_*.layer_end": _r("layers end"),
    "routing_geometry.net_*.vias.via_*.via_width": _r("(diameter D); v1-v8 drop it", L),
    "routing_geometry.net_*.points": _r("points"),
    "routing_geometry.net_*.points[].id": _r("point id"),
    "routing_geometry.net_*.points[].xy": _r("point x y"),
    "routing_geometry.net_*.points[].layer": _d("the prompt asks the agent to infer it from the pad"),
    "router_head": _r("router_head section"),
    "router_head.current_xy": _r("xy"),
    "router_head.current_layer": _r("layer"),
    "router_head.current_net": _r("net"),
    "router_head.current_net_phase": _r("phase"),
    "router_head.is_routing": _r("is_routing"),
    "router_head.routing_mode": _r("routing_mode letter"),
    "router_head.step": _r("step"),
    "router_head.step_ratio": _r("step ratio"),
    "router_head.steps_remaining": _d("step + ratio already give the budget"),
    "router_head.router_state_code": _d("engine-internal state"),
    "router_head.via_toggle": _d("engine-internal state"),
    "router_head.is_dragging": _d("engine-internal state"),
    "drc_violations": _r("(drc_violations ...); v1-v8 drop them", L),
    "drc_violations[].x_mm": _r("drc x", L),
    "drc_violations[].y_mm": _r("drc y", L),
    "drc_violations[].layer": _r("drc layer", L),
    "drc_violations[].type_id": _r("drc type bucket name", L),
    "drc_violations[].severity": _r("error | warning", L),
    "drc_violations[].net_names": _r("first net name (the RL token binds the first net)", L),
    "drc_violations[].error_type": _d("free KiCad text; the type bucket names it"),
    "action_history": _d("the prompt's History block comes from the rollout memory"),
    "closed_nets": _r("(closed_nets CODE ...); v1-v8 drop it", L),
}
del L

#: Obs paths where ``latest`` and the RL tokenizer (configs/rl.yaml policy defaults)
#: deliberately differ, with the reason. ``llm_only``: rendered for the LLM, never read
#: by the tokenizer; ``rl_only``: read by the tokenizer, not rendered.
PARITY_EXCEPTIONS: Dict[str, Dict[str, str]] = {
    "llm_only": {
        **{p: "ids name items in the text (the RL tokens are positional)" for p in (
            "board_static.boardlines.edge_*.id", "board_static.boardlines.edge_*.p1.id",
            "board_static.boardlines.edge_*.p2.id", "board_static.boardlines.edge_*.mid.id",
            "board_static.nets.net_*.pads.pad_*.id", "board_static.obstacles.obs_*.id",
            "board_static.unconnected_pads.upad_*.id",
            "routing_geometry.net_*.tracks.track_*.id",
            "routing_geometry.net_*.tracks.track_*.p1.id",
            "routing_geometry.net_*.tracks.track_*.p2.id",
            "routing_geometry.net_*.vias.via_*.id", "routing_geometry.net_*.vias.via_*.center.id",
            "routing_geometry.net_*.points[].id")},
        "board_static.scale": "max(bbox w, h) — derived from the bbox both see",
        "board_static.net_count": "the number of nets — derived from the nets both see",
        "router_head.step": "the step budget: the RL head reads steps_remaining",
        "router_head.step_ratio": "the step budget: the RL head reads steps_remaining",
    },
    "rl_only": {
        "router_head.steps_remaining": "the step budget: the LLM reads step and step/max_steps",
        "action_history": "both see their past actions: the RL tokens read the env's window, "
                          "the LLM prompt's History block comes from the rollout memory "
                          "(history_length)",
    },
}


@lru_cache(maxsize=None)
def obs_elements(prompt_version: str) -> FrozenSet[str]:
    """The rendered element paths of ``prompt_version`` (case-insensitive, like
    ``get_pcbworld_prompts``): every ``render`` element whose ``since`` … ``until``
    range holds the version."""
    key = prompt_version.lower()
    if key not in PCBWORLD_PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {prompt_version!r}; "
                         f"choose from: {', '.join(PCBWORLD_PROMPT_VERSIONS)}")
    rank = _version_rank(key)
    return frozenset(path for path, e in OBS_ELEMENTS.items()
                     if e.kind == "render" and _version_rank(e.since) <= rank
                     and (e.until is None or rank <= _version_rank(e.until)))


_NET_CONSTRAINTS = "board_static.nets.net_*.constraints"
_NET_CONSTRAINT_KEYS = ("track_width", "clearance", "via_diameter", "via_drill")
_NET_CONSTRAINT_PATHS = frozenset(p for p in OBS_ELEMENTS
                                  if p == _NET_CONSTRAINTS or p.startswith(_NET_CONSTRAINTS + "."))


_DRC_PATHS = frozenset(p for p in OBS_ELEMENTS
                      if p == "drc_violations" or p.startswith("drc_violations["))


def _rendered(prompt_version: str, net_constraint_obs: bool,
              emit_drc_tokens: bool = True) -> FrozenSet[str]:
    """:func:`obs_elements` of ``prompt_version`` minus the elements the env did not
    produce: the per-net ``constraints`` exist only with ``net_constraint_obs`` on, the
    DRC rows only with ``emit_drc_tokens`` on."""
    elements = obs_elements(prompt_version)
    if not net_constraint_obs:
        elements -= _NET_CONSTRAINT_PATHS
    return elements if emit_drc_tokens else elements - _DRC_PATHS


_EDGE_WIDTH = "board_static.boardlines.edge_*.width"
_PAD_SIZE = "board_static.nets.net_*.pads.pad_*.width"
_OBS_SHAPE = "board_static.obstacles.obs_*.shape"
_UPADS = "board_static.unconnected_pads"
_BOARD_CONSTRAINTS = "board_static.board_constraints"
_TRACK_WIDTH = "routing_geometry.net_*.tracks.track_*.width"
_VIA_DIAMETER = "routing_geometry.net_*.vias.via_*.via_width"
_DRC = "drc_violations"
_CLOSED_NETS = "closed_nets"
_KEEPOUT_SHAPE = "polygon"   # a rule-area keepout entry in board_static.obstacles


def _net_constraint_keys(elements: FrozenSet[str]) -> List[str]:
    return [k for k in _NET_CONSTRAINT_KEYS if f"{_NET_CONSTRAINTS}.{k}" in elements]


def _drawn_obstacles(bs: Dict[str, Any], elements: FrozenSet[str]) -> List[Dict[str, Any]]:
    """board_static obstacles in render order; a version that renders the obstacle shape
    skips rule-area keepouts, as the RL tokenizer does."""
    items = list(bs.get("obstacles", {}).values())
    if _OBS_SHAPE in elements:
        items = [o for o in items if o.get("shape", "") != _KEEPOUT_SHAPE]
    return items


def _drc_type_name(type_id: Any) -> str:
    """DRC type bucket name (the RL token's 7 buckets): clearance … copper_edge, other."""
    from pcb_world.engine.drc import DRC_TYPE_ID_DEFAULT, VIOLATION_KEYWORDS

    tid = int(type_id)
    if not 0 <= tid < DRC_TYPE_ID_DEFAULT:
        return "other"
    return VIOLATION_KEYWORDS[tid].replace(" ", "_")


def _drc_rows(violations: List[Dict[str, Any]],
              copper_layers: int) -> List[Tuple[str, str, str, str, str, str]]:
    """(x, y, layer, type, severity, net) per violation — the fields the RL token reads
    (layer as the prompt's copper number 1 = Top … N = Bottom, ``-`` when the marker sits
    on no copper layer; severity error | warning; the first net, ``-`` when none)."""
    from pcb_world.engine.drc import DRC_SEVERITY_ERROR
    from pcb_world.engine.layer_mapping import LayerMapping

    human = {lid: n for n, lid in enumerate(LayerMapping(copper_layers).board_layer_order, 1)}
    return [(_fmt(float(v["x_mm"])), _fmt(float(v["y_mm"])), str(human.get(v["layer"], "-")),
             _drc_type_name(v["type_id"]),
             "error" if int(v["severity"]) == DRC_SEVERITY_ERROR else "warning",
             (v.get("net_names") or ["-"])[0]) for v in violations]


# ===========================================================================
# State dict → XML-like string
# ===========================================================================

def _is_leaf(v: Any) -> bool:
    """Leaf = scalar or list of scalars (e.g. xy=[10.0, 20.0])."""
    if isinstance(v, (int, float, bool, str)):
        return True
    if isinstance(v, list) and all(isinstance(x, (int, float, bool, str)) for x in v):
        return True
    return False


def _fmt_scalar(x: Any) -> str:
    """Format a single scalar value, rounding floats to 3 decimal places."""
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


# Layer sentinel shared with ``envs/engine_api/pcb_file_parser.py`` — a pad
# whose ``layer`` is ``0`` is plated through every copper layer
# (``thru_hole``). Both prompt formats render this as ``th`` so the LLM
# sees a distinct token rather than a number it might confuse with layer 0.
_THRU_HOLE_LAYER = 0
_THRU_HOLE_TAG = "th"


def _fmt_pad_layer(layer: Any) -> str:
    """Render a pad layer value, substituting the thru-hole sentinel."""
    if layer == _THRU_HOLE_LAYER:
        return _THRU_HOLE_TAG
    return str(layer)


def _fmt_val(v: Any) -> str:
    """Format a leaf value for an XML attribute."""
    if isinstance(v, (bool, int, float)):
        return _fmt_scalar(v)
    if isinstance(v, list):
        return '"' + " ".join(_fmt_scalar(x) for x in v) + '"'
    return f'"{v}"'


def _to_xml(tag: str, data: Any, indent: int = 0) -> str:
    """Recursively convert a value into a self-closing XML-like string.

    All float values are rounded to 3 decimal places (.3f).
    Leaf values and empty elements use self-closing tags (<tag ... />).
    """
    prefix = "  " * indent

    # leaf (shouldn't be called directly, but safe fallback)
    if _is_leaf(data):
        return f"{prefix}<{tag} value={_fmt_val(data)} />"

    # list of dicts  (e.g. points → <point ... />)
    if isinstance(data, list):
        if not data:
            return f"{prefix}<{tag} />"
        item_tag = tag.rstrip("s") if tag.endswith("s") and len(tag) > 1 else tag
        lines = [f"{prefix}<{tag}>"]
        for item in data:
            lines.append(_to_xml(item_tag, item, indent + 1))
        lines.append(f"{prefix}</{tag}>")
        return "\n".join(lines)

    # dict
    if not isinstance(data, dict):
        return f"{prefix}<{tag} value={_fmt_val(data)} />"

    if not data:
        return f"{prefix}<{tag} />"

    attrs: list[str] = []
    children: list[tuple[str, Any]] = []
    for k, v in data.items():
        if _is_leaf(v):
            attrs.append(f"{k}={_fmt_val(v)}")
        else:
            children.append((k, v))

    attr_str = " ".join(attrs)
    opening = f"{tag} {attr_str}" if attr_str else tag

    if not children:
        return f"{prefix}<{opening} />"

    lines = [f"{prefix}<{opening}>"]
    for ck, cv in children:
        # "pad_0": {"id": "D0", ...} → <pad id="D0" ... />
        child_tag = ck.rsplit("_", 1)[0] if (
            isinstance(cv, dict) and "id" in cv and
            "_" in ck and ck.rsplit("_", 1)[1].isdigit()
        ) else ck
        lines.append(_to_xml(child_tag, cv, indent + 1))
    lines.append(f"{prefix}</{tag}>")
    return "\n".join(lines)


def _normalize_net(net_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a per-net routing dict.

    Ensure tracks/vias/points become empty dict/list when absent,
    so _to_xml renders them as self-closing tags.
    """
    return {
        "tracks": net_dict.get("tracks") or {},
        "vias": net_dict.get("vias") or {},
        "points": net_dict.get("points") or [],
    }


def _tag_thru_hole_pads(board_static: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied board_static where pad layer==0 is swapped
    for the ``th`` tag, so the generic XML rendering surfaces the
    thru-hole sentinel as a distinct token.
    """
    nets = board_static.get("nets") or {}
    if not nets:
        return board_static
    new_nets = {}
    for net_key, net_val in nets.items():
        pads = net_val.get("pads") or {}
        new_pads = {}
        for pk, pad in pads.items():
            if pad.get("layer") == _THRU_HOLE_LAYER:
                # shallow-copy only the pad dict that needs mutation
                new_pads[pk] = {**pad, "layer": _THRU_HOLE_TAG}
            else:
                new_pads[pk] = pad
        new_nets[net_key] = {**net_val, "pads": new_pads}
    return {**board_static, "nets": new_nets}


def _xml_indent(level: int) -> str:
    return "  " * level


def _pad_layer_attr(layer: Any) -> str:
    """Format a pad layer attribute value — bare digit or quoted ``"th"``."""
    pl = _fmt_pad_layer(layer)
    return f'"{pl}"' if not str(pl).isdigit() else str(pl)


def _board_static_to_xml(bs: Dict[str, Any], elements: FrozenSet[str], ind: int = 0) -> str:
    """Hand-rolled XML serializer for board_static.

    Applies the **same simplifications** as
    :func:`_board_static_to_sexpr` so XML and sexpr renders carry the
    same information (only the surface syntax differs):
      - pad: keeps only id / x / y / layer; drops width/height/shape and
        the nested ``<center>`` wrapper
      - obs: drops shape; ``size="W H"`` attribute; nested ``<center>``
        unrolled to inline x/y
      - boardlines: drops edge width (arc entries keep their mid point)
      - bbox / scale / net_count / copper_layers as root attributes
      - board_constraints: same field-prefix and "<0 skip" filter as sexpr
    """
    p = _xml_indent(ind)
    p1 = _xml_indent(ind + 1)
    p2 = _xml_indent(ind + 2)
    p3 = _xml_indent(ind + 3)
    p4 = _xml_indent(ind + 4)

    lines: list[str] = [
        f"{p}<board_static"
        f" bbox_x={_fmt(bs['bbox_x'])} bbox_y={_fmt(bs['bbox_y'])}"
        f" bbox_w={_fmt(bs['bbox_w'])} bbox_h={_fmt(bs['bbox_h'])}"
        f" scale={_fmt(bs['scale'])}"
        f" net_count={bs['net_count']}"
        f" copper_layers={bs['copper_layers']}>"
    ]

    # boardlines — match sexpr's `(edge ...)` / `(arc ...)` tags (not
    # `boardline`). An entry with a "mid" point is an arc through
    # p1 -> mid -> p2 (KiCad 3-point form; full circle: p1 == p2, mid =
    # antipode) and renders as <arc> with the mid between the endpoints.
    boardlines = bs.get("boardlines", {})
    if boardlines:
        lines.append(f"{p1}<boardlines>")
        for _key, edge in boardlines.items():
            x1, y1 = edge["p1"]["xy"]
            x2, y2 = edge["p2"]["xy"]
            mid = edge.get("mid")
            tag = "arc" if mid is not None else "edge"
            lines.append(f'{p2}<{tag} id="{edge["id"]}">')
            lines.append(
                f'{p3}<p1 id="{edge["p1"]["id"]}" x={_fmt(x1)} y={_fmt(y1)} />'
            )
            if mid is not None:
                mx, my = mid["xy"]
                lines.append(
                    f'{p3}<mid id="{mid["id"]}" x={_fmt(mx)} y={_fmt(my)} />'
                )
            lines.append(
                f'{p3}<p2 id="{edge["p2"]["id"]}" x={_fmt(x2)} y={_fmt(y2)} />'
            )
            lines.append(f"{p2}</{tag}>")
        lines.append(f"{p1}</boardlines>")
    else:
        lines.append(f"{p1}<boardlines />")

    # nets — `<net code=K name="...">` (parallel to sexpr `(net K "NAME" ...)`)
    nets = bs.get("nets", {})
    if nets:
        lines.append(f"{p1}<nets>")
        for net_key, net_val in nets.items():
            code = net_key.split("_", 1)[1]
            name = net_val["net_name"]
            lines.append(f'{p2}<net code={code} name="{name}">')
            pads = net_val.get("pads", {})
            if pads:
                lines.append(f"{p3}<pads>")
                for _pk, pad in pads.items():
                    cx, cy = pad["center"]["xy"]
                    lines.append(
                        f'{p4}<pad id="{pad["id"]}"'
                        f" x={_fmt(cx)} y={_fmt(cy)}"
                        f" layer={_pad_layer_attr(pad['layer'])} />"
                    )
                lines.append(f"{p3}</pads>")
            else:
                lines.append(f"{p3}<pads />")
            if _NET_CONSTRAINTS in elements:
                c = net_val["constraints"]
                lines.append(f"{p3}<constraints " + " ".join(
                    f"{k}={_fmt(float(c[k]))}" for k in _NET_CONSTRAINT_KEYS) + " />")
            lines.append(f"{p2}</net>")
        lines.append(f"{p1}</nets>")
    else:
        lines.append(f"{p1}<nets />")

    # obstacles — `<obs ...>` (parallel to sexpr `(obs ...)`)
    obstacles = bs.get("obstacles", {})
    if obstacles:
        lines.append(f"{p1}<obstacles>")
        for _ok, obs_item in obstacles.items():
            cx, cy = obs_item["center"]["xy"]
            w, h = obs_item["width"], obs_item["height"]
            lines.append(
                f'{p2}<obs id="{obs_item["id"]}"'
                f" x={_fmt(cx)} y={_fmt(cy)}"
                f' size="{_fmt(w)} {_fmt(h)}"'
                f" layer={obs_item['layer']} />"
            )
        lines.append(f"{p1}</obstacles>")
    else:
        lines.append(f"{p1}<obstacles />")

    # board_constraints — same filter as sexpr (skip unset / negative).
    bc = bs.get("board_constraints", {})
    if bc:
        _BC_LABELS = {
            "clearance_mm":     "clearance",
            "track_width_mm":   "track_width",
            "via_diameter_mm":  "via_diameter",
            "via_drill_mm":     "via_drill",
            "uvia_diameter_mm": "uvia_diameter",
            "uvia_drill_mm":    "uvia_drill",
        }
        attrs: list[str] = []
        for field, label in _BC_LABELS.items():
            v = bc.get(field, -1.0)
            if v is None or v < 0:
                continue
            attrs.append(f"{label}={_fmt(float(v))}")
        if attrs:
            lines.append(f"{p1}<board_constraints {' '.join(attrs)} />")

    lines.append(f"{p}</board_static>")
    return "\n".join(lines)


def _routing_geometry_to_xml(rg: Dict[str, Any], ind: int = 0) -> str:
    """Hand-rolled XML serializer for routing_geometry.

    Same simplifications as :func:`_routing_geometry_to_sexpr`:
      - track: drops width
      - via: drops via_width; ``layers="start end"`` attribute
      - point: drops layer (matches sexpr; layer must be inferred from
        the target pad / nearby tracks)
    """
    p = _xml_indent(ind)
    p1 = _xml_indent(ind + 1)
    p2 = _xml_indent(ind + 2)
    p3 = _xml_indent(ind + 3)
    p4 = _xml_indent(ind + 4)

    lines: list[str] = [f"{p}<routing_geometry>"]

    for net_key, net_val in rg.items():
        code = net_key.split("_", 1)[1]
        tracks = net_val.get("tracks") or {}
        vias = net_val.get("vias") or {}
        points = net_val.get("points") or []

        lines.append(f"{p1}<net code={code}>")

        # tracks
        if tracks:
            lines.append(f"{p2}<tracks>")
            for _tk, trk in tracks.items():
                x1, y1 = trk["p1"]["xy"]
                x2, y2 = trk["p2"]["xy"]
                lines.append(
                    f'{p3}<track id="{trk["id"]}" layer={trk["layer"]}>'
                )
                lines.append(
                    f'{p4}<p1 id="{trk["p1"]["id"]}" x={_fmt(x1)} y={_fmt(y1)} />'
                )
                lines.append(
                    f'{p4}<p2 id="{trk["p2"]["id"]}" x={_fmt(x2)} y={_fmt(y2)} />'
                )
                lines.append(f"{p3}</track>")
            lines.append(f"{p2}</tracks>")
        else:
            lines.append(f"{p2}<tracks />")

        # vias
        if vias:
            lines.append(f"{p2}<vias>")
            for _vk, via in vias.items():
                cx, cy = via["center"]["xy"]
                ls, le = via["layer_start"], via["layer_end"]
                lines.append(
                    f'{p3}<via id="{via["id"]}" layers="{ls} {le}">'
                )
                lines.append(
                    f'{p4}<center id="{via["center"]["id"]}"'
                    f" x={_fmt(cx)} y={_fmt(cy)} />"
                )
                lines.append(f"{p3}</via>")
            lines.append(f"{p2}</vias>")
        else:
            lines.append(f"{p2}<vias />")

        # points
        if points:
            lines.append(f"{p2}<points>")
            for pt in points:
                px, py = pt["xy"]
                lines.append(
                    f'{p3}<point id="{pt["id"]}" x={_fmt(px)} y={_fmt(py)} />'
                )
            lines.append(f"{p2}</points>")
        else:
            lines.append(f"{p2}<points />")

        lines.append(f"{p1}</net>")

    lines.append(f"{p}</routing_geometry>")
    return "\n".join(lines)


def _router_head_to_xml(rh: Dict[str, Any], ind: int = 0) -> str:
    """Hand-rolled XML serializer for router_head.

    Mirrors :func:`_router_head_to_sexpr`: drops internal-only fields
    (router_state_code, via_toggle, is_dragging), translates routing_mode
    integer to its letter (m/p/w), and merges step + step_ratio.
    """
    p = _xml_indent(ind)
    xy = rh["current_xy"]
    return (
        f"{p}<router_head"
        f' xy="{_fmt(xy[0])} {_fmt(xy[1])}"'
        f" layer={rh['current_layer']}"
        f" net={rh['current_net']}"
        f" phase={rh['current_net_phase']}"
        f" is_routing={_fmt(rh['is_routing'])}"
        f" routing_mode={_MODE_INT_TO_LETTER.get(rh['routing_mode'], rh['routing_mode'])}"
        f' step="{rh["step"]} {_fmt(rh["step_ratio"])}"'
        f" />"
    )


def _xml_elements(prompt_version: str, net_constraint_obs: bool,
                  emit_drc_tokens: bool = True) -> FrozenSet[str]:
    """:func:`_rendered`, for the versions with an XML form: ``latest`` describes and renders
    the S-expression form only."""
    if prompt_version.lower() == LATEST:
        raise ValueError("prompt_version latest renders the sexpr state format only "
                         "(its glossary describes that syntax); use state_format sexpr")
    return _rendered(prompt_version, net_constraint_obs, emit_drc_tokens)


def format_state(obs: Dict[str, Any], prompt_version: str, *,
                 net_constraint_obs: bool, emit_drc_tokens: bool = True) -> str:
    """Convert build_json_observation() output to an XML-like string.

    Carries the same information as :func:`format_state_sexpr` — the
    two serializers apply identical field-level simplifications; only
    the surface syntax (XML tags vs. S-expression parens) differs.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.
        prompt_version: selects the rendered elements (:func:`obs_elements`).
        net_constraint_obs: the env's ``EnvConfig.net_constraint_obs`` — whether
            the per-net ``constraints`` exist to be rendered.
        emit_drc_tokens: the env's ``EnvConfig.emit_drc_tokens`` — whether the DRC
            rows exist to be rendered (off: no ``drc_violations`` element).

    Returns:
        Multi-line XML-like string.
    """
    obs = dict(obs)
    elements = _xml_elements(prompt_version, net_constraint_obs, emit_drc_tokens)
    sections = []
    if "board_static" in obs:
        sections.append(_board_static_to_xml(obs["board_static"], elements))
    if "routing_geometry" in obs:
        sections.append(_routing_geometry_to_xml(obs["routing_geometry"]))
    if "router_head" in obs:
        sections.append(_router_head_to_xml(obs["router_head"]))
    return "\n".join(sections)


def format_state_split(obs: Dict[str, Any], prompt_version: str, *,
                       net_constraint_obs: bool, emit_drc_tokens: bool = True) -> Tuple[str, str]:
    """Like :func:`format_state`, but returns board_static separately.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.
        prompt_version: selects the rendered elements (:func:`obs_elements`).
        net_constraint_obs: the env's ``EnvConfig.net_constraint_obs`` — whether
            the per-net ``constraints`` exist to be rendered.
        emit_drc_tokens: the env's ``EnvConfig.emit_drc_tokens`` — whether the DRC
            rows exist to be rendered (off: no ``drc_violations`` element).

    Returns:
        (board_static_xml, dynamic_xml) where dynamic_xml contains
        routing_geometry and router_head.
    """
    obs = dict(obs)
    static = (_board_static_to_xml(obs["board_static"],
                                   _xml_elements(prompt_version, net_constraint_obs,
                                                 emit_drc_tokens))
              if "board_static" in obs else "")
    dynamic_sections = []
    if "routing_geometry" in obs:
        dynamic_sections.append(_routing_geometry_to_xml(obs["routing_geometry"]))
    if "router_head" in obs:
        dynamic_sections.append(_router_head_to_xml(obs["router_head"]))
    return static, "\n".join(dynamic_sections)


# ===========================================================================
# State dict → S-expression string
# ===========================================================================

def _fmt(x: Any) -> str:
    """Format a single scalar for S-expression output."""
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _sindent(level: int) -> str:
    return "  " * level


def _sexpr_size_shape(pad: Dict[str, Any], elements: FrozenSet[str]) -> str:
    """`` (size w h) (shape S)`` when the version renders pad extents, else ``""``."""
    if _PAD_SIZE not in elements:
        return ""
    return f" (size {_fmt(float(pad['width']))} {_fmt(float(pad['height']))}) (shape {pad['shape']})"


def _board_static_to_sexpr(bs: Dict[str, Any], elements: FrozenSet[str], ind: int = 0) -> str:
    """Convert board_static dict to S-expression.

    Simplifications vs raw dict:
      - bbox_x/y/w/h merged into (bbox x y w h)
      - pad: (pad <id> <x> <y> <layer>), dropping width/height/center-id
      - per-net constraints as (constraints ...) only when ``elements``
        holds them (``latest`` on an env with ``net_constraint_obs``)
      - board_constraints emitted as (board_constraints ...); fields
        unset across BDS and every netclass (value < 0) are omitted so
        the prompt only mentions real constraints (e.g. uvia_* on boards
        that don't use microvias stays out)
    """
    p = _sindent(ind)
    p1 = _sindent(ind + 1)
    p2 = _sindent(ind + 2)
    p3 = _sindent(ind + 3)
    p4 = _sindent(ind + 4)
    lines: list[str] = [f"{p}(board_static"]

    # bbox (merged)
    lines.append(f"{p1}(bbox {_fmt(bs['bbox_x'])} {_fmt(bs['bbox_y'])} {_fmt(bs['bbox_w'])} {_fmt(bs['bbox_h'])})")
    lines.append(f"{p1}(scale {_fmt(bs['scale'])})")
    lines.append(f"{p1}(net_count {bs['net_count']})")
    lines.append(f"{p1}(copper_layers {bs['copper_layers']})")

    # boardlines — a "mid"-bearing entry is an arc p1 -> mid -> p2 (KiCad
    # 3-point form; full circle: p1 == p2, mid = antipode) and renders as
    # (arc ID (p1 ...) (mid ...) (p2 ...)), mirroring the file's gr_arc.
    boardlines = bs.get("boardlines", {})
    if boardlines:
        lines.append(f"{p1}(boardlines")
        for _key, edge in boardlines.items():
            e_id = edge["id"]
            x1, y1 = edge["p1"]["xy"]
            x2, y2 = edge["p2"]["xy"]
            mid = edge.get("mid")
            width = f" (width {_fmt(float(edge['width']))})" if _EDGE_WIDTH in elements else ""
            if mid is not None:
                mx, my = mid["xy"]
                lines.append(
                    f"{p2}(arc {e_id}"
                    f" (p1 {edge['p1']['id']} {_fmt(x1)} {_fmt(y1)})"
                    f" (mid {mid['id']} {_fmt(mx)} {_fmt(my)})"
                    f" (p2 {edge['p2']['id']} {_fmt(x2)} {_fmt(y2)}){width})"
                )
            else:
                lines.append(
                    f"{p2}(edge {e_id}"
                    f" (p1 {edge['p1']['id']} {_fmt(x1)} {_fmt(y1)})"
                    f" (p2 {edge['p2']['id']} {_fmt(x2)} {_fmt(y2)}){width})"
                )
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(boardlines)")

    # nets — pad simplified to (pad <id> <x> <y> <layer>)
    nets = bs.get("nets", {})
    if nets:
        lines.append(f"{p1}(nets")
        for net_key, net_val in nets.items():
            net_code = net_key.split("_", 1)[1]
            net_name = net_val["net_name"]
            lines.append(f"{p2}(net {net_code} \"{net_name}\"")
            pads = net_val.get("pads", {})
            if pads:
                lines.append(f"{p3}(pads")
                for _pk, pad in pads.items():
                    pad_id = pad["id"]
                    cx, cy = pad["center"]["xy"]
                    pl = _fmt_pad_layer(pad["layer"])
                    lines.append(f"{p4}(pad {pad_id} {_fmt(cx)} {_fmt(cy)} {pl}"
                                 f"{_sexpr_size_shape(pad, elements)})")
                lines.append(f"{p3})")
            else:
                lines.append(f"{p3}(pads)")
            if _NET_CONSTRAINTS in elements:
                c = net_val["constraints"]
                lines.append(f"{p3}(constraints " + " ".join(
                    f"({k} {_fmt(float(c[k]))})" for k in _net_constraint_keys(elements)) + ")")
            lines.append(f"{p2})")
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(nets)")

    # obstacles
    obstacles = _drawn_obstacles(bs, elements)
    if obstacles:
        lines.append(f"{p1}(obstacles")
        for obs in obstacles:
            o_id = obs["id"]
            cx, cy = obs["center"]["xy"]
            w, h = obs["width"], obs["height"]
            layer = obs["layer"]
            shape = f" (shape {obs['shape']})" if _OBS_SHAPE in elements else ""
            lines.append(
                f"{p2}(obs {o_id} {_fmt(cx)} {_fmt(cy)}"
                f" (size {_fmt(w)} {_fmt(h)}) (layer {layer}){shape})"
            )
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(obstacles)")

    # unconnected_pads — net-less pads: copper every net must avoid
    if _UPADS in elements:
        upads = bs.get("unconnected_pads", {})
        if upads:
            lines.append(f"{p1}(unconnected_pads")
            for up in upads.values():
                cx, cy = up["center"]["xy"]
                lines.append(f"{p2}(upad {up['id']} {_fmt(cx)} {_fmt(cy)} "
                             f"{_fmt_pad_layer(up['layer'])}{_sexpr_size_shape(up, elements)})")
            lines.append(f"{p1})")
        else:
            lines.append(f"{p1}(unconnected_pads)")

    # board_constraints — strictest effective DRC constraints on this
    # board. Keys use full field names (not abbreviations) so the LLM
    # sees unambiguous labels; the trailing ``_mm`` is dropped since every
    # value is in millimetres by convention across this prompt format.
    bc = bs.get("board_constraints", {}) if _BOARD_CONSTRAINTS in elements else {}
    if bc:
        _BC_LABELS = {
            "clearance_mm":     "clearance",
            "track_width_mm":   "track_width",
            "via_diameter_mm":  "via_diameter",
            "via_drill_mm":     "via_drill",
            "uvia_diameter_mm": "uvia_diameter",
            "uvia_drill_mm":    "uvia_drill",
        }
        emitted = []
        for field, label in _BC_LABELS.items():
            v = bc.get(field, -1.0)
            if v is None or v < 0:
                continue
            emitted.append(f"({label} {_fmt(float(v))})")
        if emitted:
            lines.append(f"{p1}(board_constraints " + " ".join(emitted) + ")")

    lines.append(f"{p})")
    return "\n".join(lines)


def _routing_geometry_to_sexpr(rg: Dict[str, Any], elements: FrozenSet[str], ind: int = 0) -> str:
    """Convert routing_geometry dict to S-expression.

    Track width / via diameter only when ``elements`` holds them (``latest``).
    """
    p = _sindent(ind)
    p1 = _sindent(ind + 1)
    p2 = _sindent(ind + 2)
    p3 = _sindent(ind + 3)
    lines: list[str] = [f"{p}(routing_geometry"]

    for net_key, net_val in rg.items():
        net_code = net_key.split("_", 1)[1]
        tracks = net_val.get("tracks") or {}
        vias = net_val.get("vias") or {}
        points = net_val.get("points") or []

        lines.append(f"{p1}(net {net_code}")

        # tracks — width removed
        if tracks:
            lines.append(f"{p2}(tracks")
            for _tk, trk in tracks.items():
                t_id = trk["id"]
                x1, y1 = trk["p1"]["xy"]
                x2, y2 = trk["p2"]["xy"]
                layer = trk["layer"]
                lines.append(
                    f"{p3}(track {t_id}"
                    f" (p1 {trk['p1']['id']} {_fmt(x1)} {_fmt(y1)})"
                    f" (p2 {trk['p2']['id']} {_fmt(x2)} {_fmt(y2)})"
                    f" (layer {layer})"
                    + (f" (width {_fmt(float(trk['width']))})" if _TRACK_WIDTH in elements else "")
                    + ")"
                )
            lines.append(f"{p2})")
        else:
            lines.append(f"{p2}(tracks)")

        # vias — via_width removed
        if vias:
            lines.append(f"{p2}(vias")
            for _vk, via in vias.items():
                v_id = via["id"]
                cx, cy = via["center"]["xy"]
                ls, le = via["layer_start"], via["layer_end"]
                lines.append(
                    f"{p3}(via {v_id}"
                    f" (center {via['center']['id']} {_fmt(cx)} {_fmt(cy)})"
                    f" (layers {ls} {le})"
                    + (f" (diameter {_fmt(float(via['via_width']))})" if _VIA_DIAMETER in elements else "")
                    + ")"
                )
            lines.append(f"{p2})")
        else:
            lines.append(f"{p2}(vias)")

        # points (ratsnest)
        if points:
            lines.append(f"{p2}(points")
            for pt in points:
                pt_id = pt["id"]
                px, py = pt["xy"]
                lines.append(f"{p3}(point {pt_id} {_fmt(px)} {_fmt(py)})")
            lines.append(f"{p2})")
        else:
            lines.append(f"{p2}(points)")

        lines.append(f"{p1})")

    lines.append(f"{p})")
    return "\n".join(lines)


def _router_head_to_sexpr(rh: Dict[str, Any], ind: int = 0) -> str:
    """Convert router_head dict to S-expression.

    Drops internal-only fields: router_state_code, via_toggle, is_dragging.
    Merges step + step_ratio into (step N ratio).
    """
    p = _sindent(ind)
    p1 = _sindent(ind + 1)
    xy = rh["current_xy"]
    lines: list[str] = [
        f"{p}(router_head",
        f"{p1}(xy {_fmt(xy[0])} {_fmt(xy[1])})",
        f"{p1}(layer {rh['current_layer']})",
        f"{p1}(net {rh['current_net']})",
        f"{p1}(phase {rh['current_net_phase']})",
        f"{p1}(is_routing {_fmt(rh['is_routing'])})",
        f"{p1}(routing_mode {_MODE_INT_TO_LETTER.get(rh['routing_mode'], rh['routing_mode'])})",
        f"{p1}(step {rh['step']} {_fmt(rh['step_ratio'])})",
        f"{p})",
    ]
    return "\n".join(lines)


def _dynamic_extras_to_sexpr(obs: Dict[str, Any], elements: FrozenSet[str], ind: int = 0) -> List[str]:
    """``(drc_violations ...)`` and ``(closed_nets ...)`` when ``elements`` holds them."""
    p, p1 = _sindent(ind), _sindent(ind + 1)
    out: List[str] = []
    if _DRC in elements:
        rows = _drc_rows(obs.get("drc_violations") or [], obs["board_static"]["copper_layers"])
        if rows:
            out.append("\n".join([f"{p}(drc_violations"] + [
                f'{p1}(drc {x} {y} {layer} {t} {sev} "{net}")' for x, y, layer, t, sev, net in rows
            ] + [f"{p})"]))
        else:
            out.append(f"{p}(drc_violations)")
    if _CLOSED_NETS in elements:
        out.append(f"{p}(closed_nets" + "".join(f" {c}" for c in obs.get("closed_nets") or []) + ")")
    return out


def _dynamic_sexpr(obs: Dict[str, Any], elements: FrozenSet[str]) -> List[str]:
    sections = []
    if "routing_geometry" in obs:
        sections.append(_routing_geometry_to_sexpr(obs["routing_geometry"], elements))
    sections += _dynamic_extras_to_sexpr(obs, elements)
    if "router_head" in obs:
        sections.append(_router_head_to_sexpr(obs["router_head"]))
    return sections


def format_state_sexpr(obs: Dict[str, Any], prompt_version: str, *,
                       net_constraint_obs: bool, emit_drc_tokens: bool = True) -> str:
    """Convert build_json_observation() output to S-expression string.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.
        prompt_version: selects the rendered elements (:func:`obs_elements`).
        net_constraint_obs: the env's ``EnvConfig.net_constraint_obs`` — whether
            the per-net ``constraints`` exist to be rendered.
        emit_drc_tokens: the env's ``EnvConfig.emit_drc_tokens`` — whether the DRC
            rows exist to be rendered (off: no ``drc_violations`` element).

    Returns:
        Multi-line S-expression string.
    """
    obs = dict(obs)
    if "routing_geometry" in obs:
        obs["routing_geometry"] = {
            net_key: _normalize_net(net_val)
            for net_key, net_val in obs["routing_geometry"].items()
        }
    elements = _rendered(prompt_version, net_constraint_obs, emit_drc_tokens)
    sections = []
    if "board_static" in obs:
        sections.append(_board_static_to_sexpr(obs["board_static"], elements))
    return "\n".join(sections + _dynamic_sexpr(obs, elements))


def format_state_split_sexpr(obs: Dict[str, Any], prompt_version: str, *,
                             net_constraint_obs: bool, emit_drc_tokens: bool = True) -> Tuple[str, str]:
    """Like format_state_sexpr(), but returns board_static separately.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.
        prompt_version: selects the rendered elements (:func:`obs_elements`).
        net_constraint_obs: the env's ``EnvConfig.net_constraint_obs`` — whether
            the per-net ``constraints`` exist to be rendered.
        emit_drc_tokens: the env's ``EnvConfig.emit_drc_tokens`` — whether the DRC
            rows exist to be rendered (off: no ``drc_violations`` element).

    Returns:
        (board_static_sexpr, dynamic_sexpr) where dynamic_sexpr contains
        routing_geometry and router_head.
    """
    obs = dict(obs)
    if "routing_geometry" in obs:
        obs["routing_geometry"] = {
            net_key: _normalize_net(net_val)
            for net_key, net_val in obs["routing_geometry"].items()
        }
    elements = _rendered(prompt_version, net_constraint_obs, emit_drc_tokens)
    static = (_board_static_to_sexpr(obs["board_static"], elements)
              if "board_static" in obs else "")
    return static, "\n".join(_dynamic_sexpr(obs, elements))


_FORMAT_FUNCS = {"xml": format_state, "sexpr": format_state_sexpr}


def serialize_obs(obs: Dict[str, Any], state_format: str, prompt_version: str, *,
                  net_constraint_obs: bool, emit_drc_tokens: bool = True) -> str:
    """Observation dict → prompt text in ``state_format`` (``sexpr`` | ``xml``)."""
    return _FORMAT_FUNCS[state_format](obs, prompt_version,
                                       net_constraint_obs=net_constraint_obs,
                                       emit_drc_tokens=emit_drc_tokens)


# ===========================================================================
# Action mask → valid action name list
# ===========================================================================

# Action names, usage descriptions, and examples in registry order (index 0..5).
# idle (index 6) is omitted — it is a fallback-only action, never shown to the LLM.
# Each entry: (name, description, example)
_ACTION_USAGE: List[Tuple[str, str, str]] = [
    ("net_select",
     "net_select <net_id> — Select a net to route.",
     "net_select 3"),
    ("start_route",
     "start_route <x> <y> <layer> — Begin routing from pad at (x, y) on the given layer.",
     "start_route 25.0 40.0 0"),
    ("net_end",
     "net_end — Mark the current net as fully routed.",
     "net_end"),
    ("make_line",
     "make_line <x> <y> <mode> — Draw a track from the current position to (x, y).",
     "make_line 30.0 45.0 w"),
    ("make_via",
     "make_via <x> <y> <mode> — Draw a track to (x, y), then place a via to switch layers.",
     "make_via 30.0 45.0 w"),
    ("finish",
     "finish <mode> — Auto-complete the route on the current layer to the nearest unconnected pad (does not place a via or switch layers).",
     "finish w"),
]

_MODE_ACTIONS = {"make_line", "make_via", "finish"}
_MODE_DESC = "mode: m=MarkObstacles, p=PushAndShove, w=Walkaround"
_USAGE_LINES = {name: f"{usage}  e.g. {example}" for name, usage, example in _ACTION_USAGE}


def format_valid_actions(mask: List[bool], action_lines: Optional[Dict[str, str]] = None,
                         mode_lines: Tuple[str, ...] = (_MODE_DESC,)) -> str:
    """Convert a boolean action mask to a usage string for valid actions.

    Args:
        mask: List of bools aligned with ACTION_REGISTRY order.
              Only the first 6 entries (index 0..5) are used; idle (6) is excluded.
        action_lines: each action's line (default: the legacy ``_ACTION_USAGE`` table;
              ``latest`` passes its generated lines).
        mode_lines: appended when a mode-using action is valid.

    Returns:
        Newline-separated action usage strings for valid actions.
        Appends the mode lines if any mode-using action is valid.
    """
    if action_lines is None:
        action_lines = _USAGE_LINES
    names = [n for n, ok in zip(ACTION_NAMES[:ACT_IDLE], mask) if ok]
    lines = [action_lines[n] for n in names]
    if any(n in _MODE_ACTIONS for n in names):
        lines += mode_lines
    return "\n".join(lines)


def split_board_static(text_obs: str, state_format: str = "sexpr") -> Tuple[str, str]:
    """Split a full observation string into (board_static, dynamic).

    Inverse of format_state_split / format_state_split_sexpr for post-serialization
    splitting (when only the full serialized string is available).

    Args:
        text_obs:     Full serialized observation string.
        state_format: "sexpr" or "xml".

    Returns:
        (board_static_str, dynamic_str). Returns ("", text_obs) if the split fails.
    """
    if state_format == "sexpr":
        # S-expr: (board_static ...) followed by (routing_geometry ...)
        m = re.split(r'(\(board_static\b.*?^\))\n?', text_obs,
                     maxsplit=1, flags=re.DOTALL | re.MULTILINE)
        if len(m) >= 3:
            return m[1], m[2]
        return "", text_obs
    # XML: <board_static>…</board_static>
    m = re.split(r'(</board_static>)\n?', text_obs, maxsplit=1)
    if len(m) >= 3:
        return m[0] + m[1], m[2]
    return "", text_obs


# ===========================================================================
# Candidate targets → S-expression (EnvConfig action_target: candidates)
# ===========================================================================

def _fmt_mm_exact(x: float) -> str:
    """``_fmt`` of a coordinate when 3 decimals give it back, else the shortest decimal
    that parses back to the same float (no exponent, no trailing dot)."""
    import numpy as np

    s = _fmt(float(x))
    return s if abs(float(s) - x) < 1e-9 else np.format_float_positional(float(x), trim="-")


def format_candidates(rows: Dict[str, List[Tuple[float, float, int, int]]]) -> str:
    """``{action name: [(x_mm, y_mm, layer, ctype), ...]}`` → the ``(candidates ...)``
    block, one ``(c X Y LAYER TYPE)`` line per allowed target; an action with no
    target renders bare (``(make_line)``). A coordinate that 3 decimals would round
    (KiCad nm values, 100/N grid spacings) is printed in full, since the env matches a
    candidate target exactly."""
    from pcb_world.vec.candidate_pool import CTYPE_DIRECTIONAL, CTYPE_PAD, CTYPE_TRACK, CTYPE_VIA

    names = {CTYPE_PAD: "pad", CTYPE_TRACK: "track", CTYPE_VIA: "via",
             CTYPE_DIRECTIONAL: "directional"}
    p1, p2 = _sindent(1), _sindent(2)
    lines = ["(candidates"]
    for action, cands in rows.items():
        if not cands:
            lines.append(f"{p1}({action})")
            continue
        lines.append(f"{p1}({action}")
        lines += [f"{p2}(c {_fmt_mm_exact(x)} {_fmt_mm_exact(y)} {int(layer)} {names[ct]})"
                  for x, y, layer, ct in cands]
        lines.append(f"{p1})")
    lines.append(")")
    return "\n".join(lines)
