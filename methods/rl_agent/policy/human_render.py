"""Human-readable text renderers for the RL policy's input/output.

These functions are the visualizer's bridge between the raw env obs dict
that the policy actually consumes (via
:class:`methods.rl_agent.models.v1.tokenizer.BatchedStateTokenizer`)
and the GUI panes that a human reads.

Two outputs:

  - :func:`render_obs_human` walks the obs dict in the SAME ORDER as the
    tokenizer's ``forward`` loop, so the lines line up 1:1 with the tokens
    the policy sees. CAND lines are numbered with the same index space the
    policy's pointer head samples against — printing ``[CAND 7]`` means
    pointer_idx=7 will land there.

  - :func:`render_action_human` formats ``(action_type, pointer_idx,
    routing_mode)`` plus the resolved candidate coordinate into one line
    suitable for the history pane.

Lives next to the policy so any tokenizer schema change is reviewed
alongside the renderer that mirrors it.
"""

from __future__ import annotations

from pcb_world.core.masking import ACTION_NAMES
from methods.rl_agent.models.v1.encoding import _sorted_net_keys
from methods.rl_agent.models.v1.embedding import CandidateType, EntityType, StructuralToken


# ---------------------------------------------------------------------------
# Enum-ish integer → label maps
# ---------------------------------------------------------------------------

# KiCad PNS router strategies (pcb_world/engine/kicad_engine.py).
_ROUTING_MODE_NAMES = {
    0: "mark_obstacles",
    1: "shove",
    2: "walkaround",
}

# net_phase derived in pcb_world/core/observation.py.
_NET_PHASE_NAMES = {
    0: "idle",            # no net selected
    1: "net_selected",    # selected, not routing
    2: "routing",         # actively routing
}

# CandidateType labels (methods/rl_agent/models/v1/embedding.py).
_CAND_TYPE_NAMES = {
    CandidateType.PAD_POINT.value:     "pad",
    CandidateType.TRACK_ENDPOINT.value: "track_end",
    CandidateType.VIA_CENTER.value:    "via",
    CandidateType.RATSNEST.value:      "rat",
    CandidateType.DIRECTIONAL.value:   "dir",
}


def _mode_name(rm: int) -> str:
    return _ROUTING_MODE_NAMES.get(int(rm), f"mode={int(rm)}")


def _phase_name(np_: int) -> str:
    return _NET_PHASE_NAMES.get(int(np_), f"phase={int(np_)}")


def _cand_name(ct: int) -> str:
    return _CAND_TYPE_NAMES.get(int(ct), f"ct={int(ct)}")


# ---------------------------------------------------------------------------
# Action renderer
# ---------------------------------------------------------------------------

def render_action_human(
    action_type: int,
    pointer_idx: int,
    routing_mode: int,
    cand_mm: list[tuple[float, float, int]],
) -> str:
    """Format a single policy action for the history pane.

    ``cand_mm[k] = (x_mm, y_mm, layer)`` is the physical resolution of
    ``pointer_idx = k`` — same ordering the tokenizer used to embed the
    CAND tokens, so the printout matches what
    :func:`render_obs_human` showed.
    """
    if 0 <= action_type < len(ACTION_NAMES):
        name = ACTION_NAMES[action_type]
    else:
        name = f"?action_type={action_type}"
    if pointer_idx < 0:
        ptr = "(no pointer)"
    elif pointer_idx < len(cand_mm):
        x, y, ly = cand_mm[pointer_idx]
        ptr = f"CAND[{pointer_idx}] → ({x:.2f},{y:.2f})mm L{ly}"
    else:
        ptr = f"CAND[{pointer_idx}] (out of range; {len(cand_mm)} cands)"
    mode = _mode_name(routing_mode)
    return f"{name}  {ptr}  mode={mode}"


# ---------------------------------------------------------------------------
# Observation renderer
# ---------------------------------------------------------------------------

def _fmt(x) -> str:
    """Local clone of :func:`methods.llm_agent.wrappers.state_converter._fmt` so this module
    never reaches into LLM-side privates. Floats → 3 decimals, bools
    → lowercase, everything else → str."""
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _indent(level: int) -> str:
    return "  " * level


def _format_state_sexpr_rl(obs: dict) -> str:
    """Emit the obs as an S-expression mirroring exactly what
    :class:`BatchedStateTokenizer` reads. Always raw mm values.

    Intentionally diverges from :func:`methods.llm_agent.wrappers.state_converter.format_state_sexpr`:

    Omitted (LLM sexpr shows; RL tokenizer never reads):
      - ``(scale ...)``, ``(net_count ...)``, ``(obstacles ...)``,
        ``(board_constraints ...)``
      - ``net["net_name"]`` string (RL uses ``net_code`` only)
      - All entity IDs (pad_id / edge_id / track_id / via_id / point_id)

    Restored (LLM sexpr drops; RL tokenizer reads):
      - per-net ``constraints`` (track_width / clearance / via_diameter)
      - ``edge["width"]``, ``pad["width"] / ["height"]``,
        ``track["width"]``, ``via["via_width"]``
    """
    bs = obs.get("board_static", {}) or {}
    rg = obs.get("routing_geometry", {}) or {}
    rh = obs.get("router_head", {}) or {}
    n_copper = int(bs.get("copper_layers", 0))

    p1, p2, p3, p4 = _indent(1), _indent(2), _indent(3), _indent(4)
    lines: list[str] = ["(board_static"]
    lines.append(
        f"{p1}(bbox {_fmt(bs.get('bbox_x', 0.0))} {_fmt(bs.get('bbox_y', 0.0))} "
        f"{_fmt(bs.get('bbox_w', 0.0))} {_fmt(bs.get('bbox_h', 0.0))})"
    )
    lines.append(f"{p1}(copper_layers {n_copper})")

    edges = bs.get("boardlines") or {}
    if edges:
        lines.append(f"{p1}(boardlines")
        for e in edges.values():
            x1, y1 = e["p1"]["xy"]
            x2, y2 = e["p2"]["xy"]
            m = e.get("mid")
            if m is not None:  # arc entry: p1 -> mid -> p2 (KiCad 3-point form)
                mx, my = m["xy"]
                lines.append(
                    f"{p2}(arc (p1 {_fmt(x1)} {_fmt(y1)}) "
                    f"(mid {_fmt(mx)} {_fmt(my)}) "
                    f"(p2 {_fmt(x2)} {_fmt(y2)}) (width {_fmt(e.get('width', 0))}))"
                )
            else:
                lines.append(
                    f"{p2}(edge (p1 {_fmt(x1)} {_fmt(y1)}) "
                    f"(p2 {_fmt(x2)} {_fmt(y2)}) (width {_fmt(e.get('width', 0))}))"
                )
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(boardlines)")

    nets = bs.get("nets") or {}
    if nets:
        lines.append(f"{p1}(nets")
        for nk in _sorted_net_keys(nets):
            net = nets[nk]
            code = nk.split("_", 1)[1]
            lines.append(f"{p2}(net {code}")
            c = net.get("constraints", {}) or {}
            lines.append(
                f"{p3}(constraints (track_width {_fmt(c.get('track_width', 0))}) "
                f"(clearance {_fmt(c.get('clearance', 0))}) "
                f"(via_diameter {_fmt(c.get('via_diameter', 0))}))"
            )
            pads = net.get("pads") or {}
            if pads:
                lines.append(f"{p3}(pads")
                for pad in pads.values():
                    cx, cy = pad["center"]["xy"]
                    ly = int(pad.get("layer", -1))
                    layer_str = "th" if ly == 0 else str(ly)
                    lines.append(
                        f"{p4}(pad (xy {_fmt(cx)} {_fmt(cy)}) "
                        f"(size {_fmt(pad.get('width', 0))} {_fmt(pad.get('height', 0))}) "
                        f"(layer {layer_str}))"
                    )
                lines.append(f"{p3})")
            else:
                lines.append(f"{p3}(pads)")
            lines.append(f"{p2})")
        lines.append(f"{p1})")
    else:
        lines.append(f"{p1}(nets)")
    lines.append(")")

    rg_lines: list[str] = ["(routing_geometry"]
    for nk in _sorted_net_keys(rg):
        ng = rg[nk] or {}
        code = nk.split("_", 1)[1]
        rg_lines.append(f"{p1}(net {code}")
        tracks = ng.get("tracks") or {}
        if tracks:
            rg_lines.append(f"{p2}(tracks")
            for tr in tracks.values():
                x1, y1 = tr["p1"]["xy"]
                x2, y2 = tr["p2"]["xy"]
                rg_lines.append(
                    f"{p3}(track (p1 {_fmt(x1)} {_fmt(y1)}) "
                    f"(p2 {_fmt(x2)} {_fmt(y2)}) "
                    f"(width {_fmt(tr.get('width', 0))}) "
                    f"(layer {tr.get('layer', '?')}))"
                )
            rg_lines.append(f"{p2})")
        else:
            rg_lines.append(f"{p2}(tracks)")
        vias = ng.get("vias") or {}
        if vias:
            rg_lines.append(f"{p2}(vias")
            for v in vias.values():
                cx, cy = v["center"]["xy"]
                rg_lines.append(
                    f"{p3}(via (xy {_fmt(cx)} {_fmt(cy)}) "
                    f"(layers {v.get('layer_start', '?')} {v.get('layer_end', '?')}) "
                    f"(dia {_fmt(v.get('via_width', 0))}))"
                )
            rg_lines.append(f"{p2})")
        else:
            rg_lines.append(f"{p2}(vias)")
        points = ng.get("points") or []
        if points:
            rg_lines.append(f"{p2}(points")
            for pt in points:
                x, y = pt["xy"]
                rg_lines.append(f"{p3}(point {_fmt(x)} {_fmt(y)})")
            rg_lines.append(f"{p2})")
        else:
            rg_lines.append(f"{p2}(points)")
        rg_lines.append(f"{p1})")
    rg_lines.append(")")
    lines.extend(rg_lines)

    lines.append("(router_head")
    xy = rh.get("current_xy", (0.0, 0.0))
    lines.append(f"{p1}(xy {_fmt(xy[0])} {_fmt(xy[1])})")
    lines.append(f"{p1}(layer {rh.get('current_layer', 0)})")
    lines.append(f"{p1}(net {rh.get('current_net', -1)})")
    lines.append(f"{p1}(phase {rh.get('current_net_phase', 0)})")
    lines.append(f"{p1}(is_routing {_fmt(bool(rh.get('is_routing', False)))})")
    lines.append(
        f"{p1}(routing_mode {_mode_name(int(rh.get('routing_mode', 2)))})"
    )
    lines.append(
        f"{p1}(step_ratio {_fmt(float(rh.get('step_ratio', 0.0)))})"
    )
    lines.append(")")

    return "\n".join(lines)


def _format_state_token_ids(obs: dict, *, obstacle_obs: bool = False) -> str:
    """Demo view: state as a raw integer token-ID sequence.

    Each position emits the corresponding ``EntityType`` (or structural
    / action-history marker) ID, in the exact order
    :class:`BatchedStateTokenizer` produces them. Continuous features
    (pos/size/widths/layer-distances) have NO discrete IDs in this
    tokenizer — they're absent from the sequence by construction.
    ``obstacle_obs`` mirrors the tokenizer knob (OBSTACLE tokens are
    emitted only when the policy was built with it on).
    """
    bs = obs.get("board_static", {}) or {}
    rg = obs.get("routing_geometry", {}) or {}
    rh = obs.get("router_head", {}) or {}
    n_copper = int(bs.get("copper_layers", 0))

    # Walk in tokenizer's emit order. Each entry: (id, label) for the legend
    # alignment; the final flat sequence is just the id stream.
    seq: list[tuple[int | str, str]] = []
    seq.append((int(EntityType.BOARD), "BOARD"))
    for _e in (bs.get("boardlines") or {}).values():
        seq.append((int(EntityType.EDGE), "EDGE"))
    for nk in _sorted_net_keys(bs.get("nets") or {}):
        seq.append((int(EntityType.NET), f"NET[{nk}]"))
        for _pad in ((bs["nets"][nk].get("pads") or {}).values()):
            seq.append((int(EntityType.PAD), "PAD"))
    if obstacle_obs:
        for src_key in ("obstacles", "unconnected_pads"):
            for _o in (bs.get(src_key) or {}).values():
                if _o.get("shape", "") == "polygon":  # rule-area keepout
                    continue
                seq.append((int(EntityType.OBSTACLE), "OBST"))
    for nk in _sorted_net_keys(rg):
        ng = rg[nk] or {}
        for _tr in (ng.get("tracks") or {}).values():
            seq.append((int(EntityType.TRACK), "TRACK"))
        for _v in (ng.get("vias") or {}).values():
            seq.append((int(EntityType.VIA), "VIA"))
        for _pt in (ng.get("points") or []):
            seq.append((int(EntityType.RAT), "RAT"))
    for _v in (obs.get("drc_violations") or []):
        seq.append((int(EntityType.DRC_VIOLATION), "DRC"))
    seq.append((int(EntityType.HEAD), "HEAD"))

    from pcb_world.vec.candidate_pool import (
        collect_raw_candidates, build_directional_candidates,
    )
    cn = rh.get("current_net", -1)
    cur_net = cn if isinstance(cn, int) and cn > 0 else None
    extra = None
    if rh.get("is_routing", False):
        hxy = rh.get("current_xy", (0.0, 0.0))
        _ha = obs.get("_aug") or {}
        extra = build_directional_candidates(
            (hxy[0], hxy[1]), rh.get("current_layer", 1),
            mode=_ha.get("directional_candidates"),
        )
    raw_cands = collect_raw_candidates(obs, cur_net, extra)
    _CAND_ETYPE = {
        int(CandidateType.PAD_POINT):      int(EntityType.CAND_PAD),
        int(CandidateType.TRACK_ENDPOINT): int(EntityType.CAND_TRACK_END),
        int(CandidateType.VIA_CENTER):     int(EntityType.CAND_VIA),
        int(CandidateType.RATSNEST):       int(EntityType.CAND_RAT),
        int(CandidateType.DIRECTIONAL):    int(EntityType.CAND_DIR),
    }
    for k, (_x, _y, _ly, ct) in enumerate(raw_cands):
        ct_int = int(ct) if isinstance(ct, int) else int(ct.value)
        seq.append((_CAND_ETYPE[ct_int], f"CAND[{k}:{_cand_name(ct_int)}]"))

    # ACTION_HISTORY emits 3 sub-tokens (at, pt, mode) per entry, newest
    # first. Each uses a weight-tied / categorical embed: at→
    # action_type_head[k] (id=k), pt→continuous-only (no id, marked '*'),
    # mode→routing_mode_embed[m] (id=m). Entries beyond the obs history are
    # idle sentinels (model pads to its configured window; here we render
    # one sentinel entry when the history is empty).
    hist = obs.get("action_history") or []
    if not hist:
        seq.append((6, "hist0_at(=idle sentinel)"))
        seq.append(("*", "hist0_pt(continuous)"))
        seq.append((0, "hist0_mode(=m sentinel)"))
    for k, pa in enumerate(hist):
        at = int(pa.get("action_type", 6))
        pname = ACTION_NAMES[at] if 0 <= at < len(ACTION_NAMES) else f"?{at}"
        seq.append((at, f"hist{k}_at(={pname})"))
        seq.append(("*", f"hist{k}_pt(continuous)"))
        m = int(pa.get("routing_mode", 0))
        seq.append((m, f"hist{k}_mode(={_mode_name(m)})"))

    seq.append((int(StructuralToken.VAL), "VAL"))
    seq.append((int(StructuralToken.SOD), "SOD"))

    # Build output: legend + per-position table + flat sequence.
    lines: list[str] = [
        "; RL state — raw token ID sequence (demo view)",
        f"; n_copper={n_copper}  len={len(seq)}",
        "; EntityType: BOARD=0 EDGE=1 NET=2 PAD=3 TRACK=4 VIA=5 RAT=6 HEAD=7",
        ";             CAND_PAD=8 CAND_TRACK_END=9 CAND_VIA=10 CAND_RAT=11 CAND_DIR=12",
        ";             DRC_VIOLATION=13  |  StructuralToken: VAL=0 SOD=1",
        "; '*' = continuous features only, no discrete ID (feeds Fourier)",
        "",
    ]
    ids = [str(t[0]) for t in seq]
    lines.append("[" + ", ".join(ids) + "]")
    lines.append("")
    lines.append("; per-position labels:")
    for pos, (tok_id, label) in enumerate(seq):
        lines.append(f"  pos={pos:>3}  id={str(tok_id):>3}  {label}")
    return "\n".join(lines)


def _format_action_token_ids(
    action_type: int, pointer_idx: int, routing_mode: int,
) -> str:
    """Demo view: action as the raw ``[action_type, pointer_idx, routing_mode]``
    integer triple — exactly what ``policy.act_and_value`` returns.
    """
    at = int(action_type)
    ptr = int(pointer_idx)
    rm = int(routing_mode)
    at_name = ACTION_NAMES[at] if 0 <= at < len(ACTION_NAMES) else f"?{at}"
    return (
        f"; action tokens [action_type, pointer_idx, routing_mode]\n"
        f"; ActionType: net_select=0 start_route=1 net_end=2 make_line=3 "
        f"make_via=4 finish=5 idle=6\n"
        f"; RoutingMode: m=0 p=1 w=2\n"
        f"[{at}, {ptr}, {rm}]   ; "
        f"action_type={at_name}, pointer_idx={ptr}, "
        f"routing_mode={_mode_name(rm)}"
    )


def render_action_human(
    action_type: int,
    pointer_idx: int,
    routing_mode: int,
    cand_mm: list[tuple[float, float, int]],
    *,
    raw_tokens: bool = False,
) -> str:
    """Format a single policy action for the history pane.

    Default (``raw_tokens=False``): verbose ``ACTION  CAND[k] → (x,y)mm L<l>
    mode=...`` form used inside the ``<think>`` envelope.

    ``raw_tokens=True``: demo view — the bare integer triple
    ``[action_type, pointer_idx, routing_mode]`` with a one-line label.
    """
    if raw_tokens:
        return _format_action_token_ids(action_type, pointer_idx, routing_mode)
    if 0 <= action_type < len(ACTION_NAMES):
        name = ACTION_NAMES[action_type]
    else:
        name = f"?action_type={action_type}"
    if pointer_idx < 0:
        ptr = "(no pointer)"
    elif pointer_idx < len(cand_mm):
        x, y, ly = cand_mm[pointer_idx]
        ptr = f"CAND[{pointer_idx}] → ({x:.2f},{y:.2f})mm L{ly}"
    else:
        ptr = f"CAND[{pointer_idx}] (out of range; {len(cand_mm)} cands)"
    mode = _mode_name(routing_mode)
    return f"{name}  {ptr}  mode={mode}"


def render_obs_human(
    obs: dict, *, max_cands: int = 32, raw_tokens: bool = False,
    obstacle_obs: bool = False,
) -> str:
    """Human-readable view of the obs, limited to what the RL policy sees.

    Default (``raw_tokens=False``): S-expression with raw mm values for
    state + RL-only appendix blocks (drc_violations, candidates,
    action_history). Closer to env's structured output; easy to cross-
    check against the board GUI.

    ``raw_tokens=True``: demo view — pure integer token-ID sequence.
    Bypasses every readable rendering and just emits the entity_type
    IDs in tokenizer order. Useful for demos / debugging the
    tokenizer's positional layout.
    """
    if raw_tokens:
        return _format_state_token_ids(obs, obstacle_obs=obstacle_obs)

    lines: list[str] = [_format_state_sexpr_rl(obs)]

    # ----- (drc_violations ...) -----
    drc_list = obs.get("drc_violations") or []
    if drc_list:
        lines.append("(drc_violations")
        for v in drc_list:
            sev_int = int(v.get("severity", 0))
            sev_name = (
                "error" if sev_int == 2
                else "warn" if sev_int == 1
                else "info"
            )
            nets = " ".join(f'"{n}"' for n in (v.get("net_names") or []))
            lines.append(
                f"  (violation"
                f" (type_id {v.get('type_id', '?')})"
                f" (xy {float(v.get('x_mm', 0)):.3f} {float(v.get('y_mm', 0)):.3f})"
                f" (layer {v.get('layer', 1)})"
                f" (severity {sev_name})"
                f" (nets {nets}))"
            )
        lines.append(")")

    # ----- (candidates ...) -----
    from pcb_world.vec.candidate_pool import (
        collect_raw_candidates, build_directional_candidates,
    )
    rh = obs.get("router_head", {}) or {}
    cn = rh.get("current_net", -1)
    cur_net = cn if isinstance(cn, int) and cn > 0 else None
    extra = None
    if rh.get("is_routing", False):
        hxy = rh.get("current_xy", (0.0, 0.0))
        _ha = obs.get("_aug") or {}
        extra = build_directional_candidates(
            (hxy[0], hxy[1]), rh.get("current_layer", 1),
            mode=_ha.get("directional_candidates"),
        )
    raw_cands = collect_raw_candidates(obs, cur_net, extra)
    lines.append("(candidates")
    for k, (x, y, ly, ct) in enumerate(raw_cands[:max_cands]):
        ct_int = int(ct) if isinstance(ct, int) else int(ct.value)
        lines.append(
            f"  (cand {k} {_cand_name(ct_int)}"
            f" {float(x):.3f} {float(y):.3f} {ly})"
        )
    if len(raw_cands) > max_cands:
        lines.append(
            f"  ; ... {len(raw_cands) - max_cands} more (truncated at "
            f"max_cands={max_cands})"
        )
    lines.append(")")

    # ----- (action_history ...) -----
    hist = obs.get("action_history") or []
    if not hist:
        lines.append("(action_history (none))")
    for k, pa in enumerate(hist):
        at = int(pa.get("action_type", 6))
        name = ACTION_NAMES[at] if 0 <= at < len(ACTION_NAMES) else f"?{at}"
        succ = "true" if pa.get("success") else "false"
        net = pa.get("net_id")
        net_s = "-" if net is None else str(int(net))
        lines.append(
            f"(action_history (age {k}) (type {name}) (success {succ}) (net {net_s}))"
        )

    return "\n".join(lines)



def count_state_entities(
    obs: dict, *, obstacle_obs: bool = False, action_history_len: int = 1,
) -> int:
    """Approximate ``seq_len`` (token count) without invoking the tokenizer.

    Mirrors the per-entity branches in ``render_obs_human`` so callers can
    populate ``token_counts[i] = (0, seq_len, 1)`` without pulling torch.
    Does not deduct truncations applied by ``max_*`` rendering caps — the
    intent is to report the count the *policy* sees, not the *user*.
    ``obstacle_obs`` / ``action_history_len`` mirror the policy's tokenizer
    knobs (defaults = a single-entry action history, no OBSTACLE tokens).
    """
    n = 0
    bs = obs.get("board_static", {}) or {}
    rg = obs.get("routing_geometry", {}) or {}
    n += 1  # BOARD
    n += len(bs.get("boardlines") or {})  # EDGE
    nets = bs.get("nets") or {}
    for nk in nets:
        n += 1  # NET
        n += len(nets[nk].get("pads") or {})  # PAD
    if obstacle_obs:
        for src_key in ("obstacles", "unconnected_pads"):
            for _o in (bs.get(src_key) or {}).values():
                if _o.get("shape", "") == "polygon":  # rule-area keepout
                    continue
                n += 1  # OBSTACLE
    for nk in rg:
        ng = rg[nk] or {}
        n += len(ng.get("tracks") or {})
        n += len(ng.get("vias") or {})
        n += len(ng.get("points") or [])
    n += len(obs.get("drc_violations") or [])
    n += 1  # HEAD
    # CAND count is per-step variable; counted via candidate_pool.
    from pcb_world.vec.candidate_pool import (
        collect_raw_candidates, build_directional_candidates,
    )
    rh = obs.get("router_head", {}) or {}
    cn = rh.get("current_net", -1)
    cur_net = cn if isinstance(cn, int) and cn > 0 else None
    extra = None
    if rh.get("is_routing", False):
        hxy = rh.get("current_xy", (0.0, 0.0))
        _ha = obs.get("_aug") or {}
        extra = build_directional_candidates(
            (hxy[0], hxy[1]), rh.get("current_layer", 1),
            mode=_ha.get("directional_candidates"),
        )
    n += len(collect_raw_candidates(obs, cur_net, extra))
    # ACTION_HISTORY: 3 sub-tokens (at, pt, mode) per entry; the model pads
    # to its configured window with idle sentinels, so the policy always
    # consumes 3*K regardless of how full the obs history is (K=1 == a single
    # previous-action triple).
    n += 3 * max(int(action_history_len), 1)
    n += 2  # VAL + SOD structural
    return n
