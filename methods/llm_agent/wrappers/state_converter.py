"""LLM state converter — observation dict → text (XML / S-expression).

One half of ``projection.py`` (with ``action_converter``), mirroring
the RL codec ``methods.rl_agent.models.v1.encoding``. Encodes a PCBWorld observation
into the prompt-facing string, plus the inverse split helpers and the
valid-action presentation derived from the action mask.

Public: format_state / format_state_split / format_state_sexpr /
format_state_split_sexpr / format_valid_actions / split_board_static.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pcb_world.core.action_schema import MODE_INT_TO_LETTER as _MODE_INT_TO_LETTER


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


def _board_static_to_xml(bs: Dict[str, Any], ind: int = 0) -> str:
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


def format_state(obs: Dict[str, Any]) -> str:
    """Convert build_json_observation() output to an XML-like string.

    Carries the same information as :func:`format_state_sexpr` — the
    two serializers apply identical field-level simplifications; only
    the surface syntax (XML tags vs. S-expression parens) differs.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.

    Returns:
        Multi-line XML-like string.
    """
    obs = dict(obs)
    sections = []
    if "board_static" in obs:
        sections.append(_board_static_to_xml(obs["board_static"]))
    if "routing_geometry" in obs:
        sections.append(_routing_geometry_to_xml(obs["routing_geometry"]))
    if "router_head" in obs:
        sections.append(_router_head_to_xml(obs["router_head"]))
    return "\n".join(sections)


def format_state_split(obs: Dict[str, Any]) -> Tuple[str, str]:
    """Like :func:`format_state`, but returns board_static separately.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.

    Returns:
        (board_static_xml, dynamic_xml) where dynamic_xml contains
        routing_geometry and router_head.
    """
    obs = dict(obs)
    static = _board_static_to_xml(obs["board_static"]) if "board_static" in obs else ""
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


def _board_static_to_sexpr(bs: Dict[str, Any], ind: int = 0) -> str:
    """Convert board_static dict to S-expression.

    Simplifications vs raw dict:
      - bbox_x/y/w/h merged into (bbox x y w h)
      - pad: (pad <id> <x> <y> <layer>), dropping width/height/center-id
      - per-net constraints removed
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
            if mid is not None:
                mx, my = mid["xy"]
                lines.append(
                    f"{p2}(arc {e_id}"
                    f" (p1 {edge['p1']['id']} {_fmt(x1)} {_fmt(y1)})"
                    f" (mid {mid['id']} {_fmt(mx)} {_fmt(my)})"
                    f" (p2 {edge['p2']['id']} {_fmt(x2)} {_fmt(y2)}))"
                )
            else:
                lines.append(
                    f"{p2}(edge {e_id}"
                    f" (p1 {edge['p1']['id']} {_fmt(x1)} {_fmt(y1)})"
                    f" (p2 {edge['p2']['id']} {_fmt(x2)} {_fmt(y2)}))"
                )
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(boardlines)")

    # nets — pad simplified to (pad <id> <x> <y>), constraints removed
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
                    lines.append(f"{p4}(pad {pad_id} {_fmt(cx)} {_fmt(cy)} {pl})")
                lines.append(f"{p3})")
            else:
                lines.append(f"{p3}(pads)")
            lines.append(f"{p2})")
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(nets)")

    # obstacles
    obstacles = bs.get("obstacles", {})
    if obstacles:
        lines.append(f"{p1}(obstacles")
        for _ok, obs in obstacles.items():
            o_id = obs["id"]
            cx, cy = obs["center"]["xy"]
            w, h = obs["width"], obs["height"]
            layer = obs["layer"]
            lines.append(
                f"{p2}(obs {o_id} {_fmt(cx)} {_fmt(cy)}"
                f" (size {_fmt(w)} {_fmt(h)}) (layer {layer}))"
            )
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(obstacles)")

    # board_constraints — strictest effective DRC constraints on this
    # board. Keys use full field names (not abbreviations) so the LLM
    # sees unambiguous labels; the trailing ``_mm`` is dropped since every
    # value is in millimetres by convention across this prompt format.
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


def _routing_geometry_to_sexpr(rg: Dict[str, Any], ind: int = 0) -> str:
    """Convert routing_geometry dict to S-expression.

    Simplifications: track width removed, via_width removed.
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
                    f" (layer {layer}))"
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
                    f" (layers {ls} {le}))"
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


def format_state_sexpr(obs: Dict[str, Any]) -> str:
    """Convert build_json_observation() output to S-expression string.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.

    Returns:
        Multi-line S-expression string.
    """
    obs = dict(obs)
    if "routing_geometry" in obs:
        obs["routing_geometry"] = {
            net_key: _normalize_net(net_val)
            for net_key, net_val in obs["routing_geometry"].items()
        }
    sections = []
    if "board_static" in obs:
        sections.append(_board_static_to_sexpr(obs["board_static"]))
    if "routing_geometry" in obs:
        sections.append(_routing_geometry_to_sexpr(obs["routing_geometry"]))
    if "router_head" in obs:
        sections.append(_router_head_to_sexpr(obs["router_head"]))
    return "\n".join(sections)


def format_state_split_sexpr(obs: Dict[str, Any]) -> Tuple[str, str]:
    """Like format_state_sexpr(), but returns board_static separately.

    Args:
        obs: Dict with keys board_static, routing_geometry, router_head.

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
    static = _board_static_to_sexpr(obs["board_static"]) if "board_static" in obs else ""
    dynamic_sections = []
    if "routing_geometry" in obs:
        dynamic_sections.append(_routing_geometry_to_sexpr(obs["routing_geometry"]))
    if "router_head" in obs:
        dynamic_sections.append(_router_head_to_sexpr(obs["router_head"]))
    return static, "\n".join(dynamic_sections)


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


def format_valid_actions(mask: List[bool]) -> str:
    """Convert a boolean action mask to a usage string for valid actions.

    Args:
        mask: List of bools aligned with ACTION_REGISTRY order.
              Only the first 6 entries (index 0..5) are used; idle (6) is excluded.

    Returns:
        Newline-separated action usage strings for valid actions.
        Appends mode description if any mode-using action is valid.
    """
    valid_actions = [(name, usage, example) for (name, usage, example), valid in zip(_ACTION_USAGE, mask) if valid]
    lines = [f"{usage}  e.g. {example}" for _, usage, example in valid_actions]
    if any(name in _MODE_ACTIONS for name, _, _ in valid_actions):
        lines.append(_MODE_DESC)
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
