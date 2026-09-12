"""Indexed (columnar + index) observation format.

Array-table twin of the nested-dict observation produced by
``build_json_observation()``. The nested dict's *small-Python-object count*
dominates both the subproc step barrier (main-thread unpickle, 56-63% at
n_envs=128) and the tokenizer forward (``_walk_obs`` dict traversal).
Replacing the two object-heavy groups — ``board_static`` and
``routing_geometry`` — with numpy tables collapses unpickle to buffer copies
and lets the tokenizer gather/vectorize instead of walking dicts.

Format ("indexed_v1") — a plain dict so existing container-level code
(``_aug`` injection, ``board_static`` reference-sharing) keeps working:

    {
      "_fmt": "indexed_v1",
      "board_static":     {scalars + static tables},   # episode-invariant
      "routing_geometry": {dynamic tables},            # rebuilt every step
      "router_head":      <same dict as legacy>,       # small: passthrough
      "drc_violations":   <same list as legacy>,       # ≤32 rows: passthrough
      "action_history":   <same list[dict] as legacy>,  # newest first
      "closed_nets":      <same list[int] as legacy>,
    }

Static tables (all coords raw mm, **float64** — see precision note):
    pt_xy (Ps,2) f64      global 2D point pool, registration order; row k ↔ id "P{k}"
    edge_pt (E,2) i64     p1/p2 → pt_xy rows            edge_w (E,) f64
    edge_mid (E,) i64     on-arc midpoint → pt_xy row; -1 = straight segment
    net_code (S,) i64     PARSER order (not sorted)     net_name  list[str]
    net_constraints       list[dict|None] (passthrough; None unless the
                          env net_constraint_obs knob fills them)
    net_pad_start/count (S,) i64                        (CSR into pad rows)
    pad_pt (P,) i64  pad_wh (P,2) f64  pad_layer (P,) i64 (0 = thru sentinel)
    pad_shape list[str]   (tokenizer shape-id source + viz; obs_/upad_shape
                           additionally gate OBSTACLE emission — polygon
                           rows = rule-area keepouts, excluded)
    obs_* / upad_*        same columns as pads (obstacles / unconnected pads)
    + scalar passthroughs: bbox_x/y/w/h, scale, net_count, copper_layers,
      board_constraints (opaque dict ref)

Dynamic tables (net grouping = ascending net_code, matching
``build_net_geometry``'s determinism sort):
    pt_xy (Pd,2) f64      per-net pools concatenated; local row k ↔ id "P{k}"
    net_code (Sd,) i64    ascending
    net_pt_start/count, trk_start/count, via_start/count, rat_start/count (Sd,) i64
    trk_pt (L,2) i64 (GLOBAL pt rows)  trk_w (L,) f64  trk_layer (L,) i64
    via_pt (V,) i64  via_ls/via_le (V,) i64  via_dia (V,) f64
    rat_xy (Q,2) f64 (duplicates preserved = points[] list membership)
    rat_q (Q,) i64 (local Q id)  rat_layer (Q,) i64

Precision invariant: coordinate pools are **float64** and hold the
dict-path values bit-for-bit (dedup stores the FIRST-registered
unrounded value, key = round(·,3) — same as ``IDAllocator``). The
tokenizer normalizes in f64 and casts to f32 exactly once, preserving
the legacy single-rounding anchor.

``arrays_to_dict()`` losslessly reconstructs the legacy nested dict
(key order, string IDs, nesting) for cold consumers (LLM serializer,
viz/debug, JSON dumps); hot consumers (tokenizer, candidate pool,
pointer masks) read the tables directly.

Builders re-run the coordinate dedup instead of trusting id strings
(test mocks carry cosmetic ids); on canonical production dicts this
reproduces the original ids exactly — enforced by the round-trip tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pcb_world.core.observation import (
    BoardStatic,
    NetGeometry,
    _build_router_head,
)
from pcb_world.engine.containers import RoutingSessionState

OBS_FMT = "indexed_v1"

_F8 = np.float64
_I8 = np.int64


def is_indexed(obs: dict) -> bool:
    """True when *obs* is an indexed_v1 observation."""
    return obs.get("_fmt") == OBS_FMT


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class _PointPool:
    """Registration-order point pool with IDAllocator-equivalent dedup:
    key = (round(x,3), round(y,3)), stored value = first-registered
    unrounded (x, y)."""

    __slots__ = ("xs", "ys", "_key_to_idx")

    def __init__(self) -> None:
        self.xs: list[float] = []
        self.ys: list[float] = []
        self._key_to_idx: dict[tuple[float, float], int] = {}

    def register(self, x: float, y: float) -> int:
        key = (round(x, 3), round(y, 3))
        idx = self._key_to_idx.get(key)
        if idx is None:
            idx = len(self.xs)
            self._key_to_idx[key] = idx
            self.xs.append(x)
            self.ys.append(y)
        return idx

    def to_array(self) -> np.ndarray:
        arr = np.empty((len(self.xs), 2), dtype=_F8)
        arr[:, 0] = self.xs
        arr[:, 1] = self.ys
        return arr


def _rect_columns(rects: list[dict], pool: _PointPool) -> dict[str, Any]:
    """Columns for a Rectangle-shaped collection (pads/obstacles/upads).
    ``rects`` = list of Rectangle.to_dict()-shaped dicts."""
    n = len(rects)
    pt = np.empty((n,), dtype=_I8)
    wh = np.empty((n, 2), dtype=_F8)
    layer = np.empty((n,), dtype=_I8)
    shape: list[str] = []
    for i, r in enumerate(rects):
        cx, cy = r["center"]["xy"]
        pt[i] = pool.register(cx, cy)
        wh[i, 0] = r["width"]
        wh[i, 1] = r["height"]
        layer[i] = r["layer"]
        shape.append(r.get("shape", ""))
    return {"pt": pt, "wh": wh, "layer": layer, "shape": shape}


# ---------------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------------

def static_tables_from_dict(board_static: dict) -> dict:
    """Build the static table group from a legacy ``board_static`` dict.

    Iteration order mirrors ``BoardStatic.from_board`` exactly
    (edges → nets/pads in dict order → obstacles → unconnected pads),
    so on canonical dicts the reconstructed pool rows equal the original
    "P{k}" ids and pad rows equal the "D{k}" ids.
    """
    pool = _PointPool()

    edges = board_static.get("boardlines", {})
    E = len(edges)
    edge_pt = np.empty((E, 2), dtype=_I8)
    edge_w = np.empty((E,), dtype=_F8)
    edge_mid = np.full((E,), -1, dtype=_I8)  # on-arc midpoint row; -1 = straight
    for i, e in enumerate(edges.values()):
        edge_pt[i, 0] = pool.register(*e["p1"]["xy"])
        edge_pt[i, 1] = pool.register(*e["p2"]["xy"])
        if "mid" in e:  # arc entry (outline_obs="arc"); order matches IDAllocator
            edge_mid[i] = pool.register(*e["mid"]["xy"])
        edge_w[i] = e["width"]

    nets = board_static.get("nets", {})
    net_code: list[int] = []
    net_name: list[str] = []
    net_constraints: list[dict | None] = []
    all_pads: list[dict] = []
    pad_start = np.empty((len(nets),), dtype=_I8)
    pad_count = np.empty((len(nets),), dtype=_I8)
    for j, (nk, nv) in enumerate(nets.items()):
        net_code.append(int(nk.split("_", 1)[1]))
        net_name.append(nv.get("net_name", ""))
        net_constraints.append(nv.get("constraints"))
        pads = list(nv.get("pads", {}).values())
        pad_start[j] = len(all_pads)
        pad_count[j] = len(pads)
        all_pads.extend(pads)
    pad_cols = _rect_columns(all_pads, pool)

    obs_cols = _rect_columns(
        list(board_static.get("obstacles", {}).values()), pool,
    )
    upad_cols = _rect_columns(
        list(board_static.get("unconnected_pads", {}).values()), pool,
    )

    return {
        # Scalar passthroughs (same objects — bit-exact round-trip).
        "bbox_x": board_static["bbox_x"],
        "bbox_y": board_static["bbox_y"],
        "bbox_w": board_static["bbox_w"],
        "bbox_h": board_static["bbox_h"],
        "scale": board_static.get("scale", 1.0),
        "net_count": board_static.get("net_count", len(nets)),
        "copper_layers": board_static["copper_layers"],
        "board_constraints": board_static.get("board_constraints", {}),
        # Tables.
        "pt_xy": pool.to_array(),
        "edge_pt": edge_pt,
        "edge_w": edge_w,
        "edge_mid": edge_mid,
        "net_code": np.asarray(net_code, dtype=_I8),
        "net_name": net_name,
        "net_constraints": net_constraints,
        "net_pad_start": pad_start,
        "net_pad_count": pad_count,
        "pad_pt": pad_cols["pt"],
        "pad_wh": pad_cols["wh"],
        "pad_layer": pad_cols["layer"],
        "pad_shape": pad_cols["shape"],
        "obs_pt": obs_cols["pt"],
        "obs_wh": obs_cols["wh"],
        "obs_layer": obs_cols["layer"],
        "obs_shape": obs_cols["shape"],
        "upad_pt": upad_cols["pt"],
        "upad_wh": upad_cols["wh"],
        "upad_layer": upad_cols["layer"],
        "upad_shape": upad_cols["shape"],
    }


# ---------------------------------------------------------------------------
# Dynamic tables
# ---------------------------------------------------------------------------

def _dynamic_from_per_net(
    per_net: list[tuple[int, list, list, list]],
) -> dict:
    """Assemble dynamic tables from per-net entity lists.

    ``per_net`` = [(net_code, tracks, vias, points)] in ascending
    net_code order. Entities are duck-typed accessors:
      track  -> (x1, y1, x2, y2, width, layer)
      via    -> (x, y, layer_start, layer_end, via_width)
      point  -> (x, y, q_id, layer)
    supplied as plain tuples by the two front-ends below.
    """
    Sd = len(per_net)
    net_code = np.empty((Sd,), dtype=_I8)
    net_pt_start = np.empty((Sd,), dtype=_I8)
    net_pt_count = np.empty((Sd,), dtype=_I8)
    trk_start = np.empty((Sd,), dtype=_I8)
    trk_count = np.empty((Sd,), dtype=_I8)
    via_start = np.empty((Sd,), dtype=_I8)
    via_count = np.empty((Sd,), dtype=_I8)
    rat_start = np.empty((Sd,), dtype=_I8)
    rat_count = np.empty((Sd,), dtype=_I8)

    pool_xs: list[float] = []
    pool_ys: list[float] = []
    trk_pt: list[tuple[int, int]] = []
    trk_w: list[float] = []
    trk_layer: list[int] = []
    via_pt: list[int] = []
    via_ls: list[int] = []
    via_le: list[int] = []
    via_dia: list[float] = []
    rat_x: list[float] = []
    rat_y: list[float] = []
    rat_q: list[int] = []
    rat_layer: list[int] = []

    for j, (code, tracks, vias, points) in enumerate(per_net):
        net_code[j] = code
        base = len(pool_xs)
        net_pt_start[j] = base
        trk_start[j] = len(trk_pt)
        via_start[j] = len(via_pt)
        rat_start[j] = len(rat_x)

        # Per-net 2D pool: track endpoints (p1, p2 per track) then via
        # centers — the IDAllocator registration order in
        # build_net_geometry.
        key_to_idx: dict[tuple[float, float], int] = {}

        def _reg(x: float, y: float) -> int:
            key = (round(x, 3), round(y, 3))
            idx = key_to_idx.get(key)
            if idx is None:
                idx = len(pool_xs) - base
                key_to_idx[key] = idx
                pool_xs.append(x)
                pool_ys.append(y)
            return base + idx

        for (x1, y1, x2, y2, w, layer) in tracks:
            trk_pt.append((_reg(x1, y1), _reg(x2, y2)))
            trk_w.append(w)
            trk_layer.append(layer)
        for (x, y, ls, le, dia) in vias:
            via_pt.append(_reg(x, y))
            via_ls.append(ls)
            via_le.append(le)
            via_dia.append(dia)
        for (x, y, qid, layer) in points:
            rat_x.append(x)
            rat_y.append(y)
            rat_q.append(qid)
            rat_layer.append(layer)

        net_pt_count[j] = len(pool_xs) - base
        trk_count[j] = len(trk_pt) - trk_start[j]
        via_count[j] = len(via_pt) - via_start[j]
        rat_count[j] = len(rat_x) - rat_start[j]

    pt_xy = np.empty((len(pool_xs), 2), dtype=_F8)
    pt_xy[:, 0] = pool_xs
    pt_xy[:, 1] = pool_ys
    rat_xy = np.empty((len(rat_x), 2), dtype=_F8)
    rat_xy[:, 0] = rat_x
    rat_xy[:, 1] = rat_y
    return {
        "pt_xy": pt_xy,
        "net_code": net_code,
        "net_pt_start": net_pt_start,
        "net_pt_count": net_pt_count,
        "trk_start": trk_start,
        "trk_count": trk_count,
        "via_start": via_start,
        "via_count": via_count,
        "rat_start": rat_start,
        "rat_count": rat_count,
        "trk_pt": np.asarray(trk_pt, dtype=_I8).reshape(-1, 2),
        "trk_w": np.asarray(trk_w, dtype=_F8),
        "trk_layer": np.asarray(trk_layer, dtype=_I8),
        "via_pt": np.asarray(via_pt, dtype=_I8),
        "via_ls": np.asarray(via_ls, dtype=_I8),
        "via_le": np.asarray(via_le, dtype=_I8),
        "via_dia": np.asarray(via_dia, dtype=_F8),
        "rat_xy": rat_xy,
        "rat_q": np.asarray(rat_q, dtype=_I8),
        "rat_layer": np.asarray(rat_layer, dtype=_I8),
    }


def _q_ids_for_points(points_xy: list[tuple[float, float, int]]) -> list[int]:
    """Per-net Q-id assignment: dedup on (round(x,3), round(y,3), layer),
    first occurrence wins — mirrors ``IDAllocator.get_or_create_point3d``."""
    key_to_id: dict[tuple[float, float, int], int] = {}
    out: list[int] = []
    for x, y, layer in points_xy:
        key = (round(x, 3), round(y, 3), layer)
        qid = key_to_id.get(key)
        if qid is None:
            qid = len(key_to_id)
            key_to_id[key] = qid
        out.append(qid)
    return out


def dynamic_tables_from_ir(net_geometry: dict[int, NetGeometry]) -> dict:
    """Dynamic tables from the ``build_net_geometry`` dataclass IR
    (the env hot path — skips the ``to_dict()`` materialization).

    The IR already carries the determinism sort, per-net dedup'd shared
    ``Point2D``/``Point3D`` objects and dense per-net ids, so refs/ids
    are taken from the objects directly.
    """
    per_net = []
    for code in sorted(net_geometry):
        ns = net_geometry[code]
        tracks = [
            (t.p1.x, t.p1.y, t.p2.x, t.p2.y, t.width, t.layer)
            for t in ns.tracks
        ]
        vias = [
            (v.center.x, v.center.y, v.layer_start, v.layer_end, v.via_width)
            for v in ns.vias
        ]
        points = [
            (p.x, p.y, int(p.id[1:]), p.layer) for p in ns.points
        ]
        per_net.append((code, tracks, vias, points))
    return _dynamic_from_per_net(per_net)


def dynamic_tables_from_dict(routing_geometry: dict) -> dict:
    """Dynamic tables from a legacy ``routing_geometry`` dict.

    Ids are re-derived by re-running the dedup (ids in test mocks are
    cosmetic); on canonical production dicts this reproduces the
    original per-net "P{k}"/"Q{k}" ids exactly.
    """
    per_net = []
    for nk in sorted(routing_geometry, key=lambda k: int(k.split("_", 1)[1])):
        nv = routing_geometry[nk]
        code = int(nk.split("_", 1)[1])
        tracks = [
            (t["p1"]["xy"][0], t["p1"]["xy"][1],
             t["p2"]["xy"][0], t["p2"]["xy"][1],
             t["width"], t["layer"])
            for t in (nv.get("tracks") or {}).values()
        ]
        vias = [
            (v["center"]["xy"][0], v["center"]["xy"][1],
             v["layer_start"], v["layer_end"], v["via_width"])
            for v in (nv.get("vias") or {}).values()
        ]
        pts_raw = [
            (p["xy"][0], p["xy"][1], p.get("layer", 1))
            for p in (nv.get("points") or [])
        ]
        qids = _q_ids_for_points(pts_raw)
        points = [
            (x, y, qid, layer)
            for (x, y, layer), qid in zip(pts_raw, qids)
        ]
        per_net.append((code, tracks, vias, points))
    return _dynamic_from_per_net(per_net)


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

def dict_to_arrays(obs: dict) -> dict:
    """Convert a legacy nested-dict observation to indexed_v1.

    ``router_head`` / ``drc_violations`` / ``action_history`` /
    ``closed_nets`` / ``_aug`` are passed through by reference — they
    are small and identical in both formats.
    """
    out = {
        "_fmt": OBS_FMT,
        "board_static": static_tables_from_dict(obs["board_static"]),
        "routing_geometry": dynamic_tables_from_dict(
            obs.get("routing_geometry") or {},
        ),
        "router_head": obs["router_head"],
        "drc_violations": obs.get("drc_violations") or [],
        "action_history": obs.get("action_history") or [],
        "closed_nets": obs.get("closed_nets") or [],
    }
    if "_aug" in obs:
        out["_aug"] = obs["_aug"]
    return out


def build_indexed_observation(
    static_tables: dict,
    net_geometry: dict[int, NetGeometry],
    router_state: RoutingSessionState,
    step_count: int = 0,
    max_steps: int = 100,
    current_net_id: int | None = None,
    routing_mode: int = 2,
    *,
    drc_violations: list[dict] | None = None,
    action_history: list[dict] | None = None,
    closed_nets: list[int] | None = None,
) -> dict:
    """Assemble an indexed_v1 observation (env hot path).

    ``static_tables`` is the cached ``static_tables_from_dict()`` result
    (episode-invariant — shared by reference across steps);
    ``net_geometry`` is the per-step ``build_net_geometry`` IR.
    ``router_head`` reuses ``observation._build_router_head`` verbatim.
    """
    return {
        "_fmt": OBS_FMT,
        "board_static": static_tables,
        "routing_geometry": dynamic_tables_from_ir(net_geometry),
        "router_head": _build_router_head(
            router_state, step_count, max_steps, current_net_id,
            routing_mode,
        ),
        "drc_violations": drc_violations if drc_violations is not None else [],
        "action_history": action_history if action_history is not None else [],
        "closed_nets": closed_nets if closed_nets is not None else [],
    }


def make_empty_indexed_obs() -> dict:
    """Minimal schema-valid empty indexed obs (0 nets, no geometry) — a test
    helper (schema-validity / tokenizer-passthrough edge-case guard for an
    empty obs). Production crash recovery does not use this: the subproc
    ``_safe_recv_step`` reuses the last good obs, or ``_recover_worker``
    supplies a real reset obs."""
    empty_static: dict = {
        "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 1.0, "bbox_h": 1.0,
        "scale": 1.0, "net_count": 0, "copper_layers": 2,
        "board_constraints": {},
        "boardlines": {}, "nets": {}, "obstacles": {},
        "unconnected_pads": {},
    }
    obs = dict_to_arrays({
        "board_static": empty_static,
        "routing_geometry": {},
        "router_head": {
            "current_xy": [0.0, 0.0], "current_layer": -1,
            "current_net": -1, "router_state_code": 0,
            "via_toggle": False, "is_routing": False,
            "is_dragging": False, "routing_mode": 2,
            "current_net_phase": 0, "step": 0, "step_ratio": 0.0,
            "steps_remaining": 0,
        },
        "drc_violations": [], "action_history": [], "closed_nets": [],
    })
    return obs


# ---------------------------------------------------------------------------
# Lossless inverse: indexed -> legacy nested dict
# ---------------------------------------------------------------------------

def _point_dict(idx: int, pt_xy: np.ndarray) -> dict:
    return {
        "id": f"P{idx}",
        "xy": [float(pt_xy[idx, 0]), float(pt_xy[idx, 1])],
    }


def _rect_dicts(
    prefix: str, cols_prefix: str, bs: dict, start: int, count: int,
    id_base: int,
) -> dict:
    """Rebuild a {key_i: Rectangle.to_dict()} block. ``id_base`` = global
    id counter offset for the "D{k}" style ids (row index)."""
    pt_xy = bs["pt_xy"]
    pt = bs[f"{cols_prefix}_pt"]
    wh = bs[f"{cols_prefix}_wh"]
    layer = bs[f"{cols_prefix}_layer"]
    shape = bs[f"{cols_prefix}_shape"]
    id_prefix = {"pad": "D", "obs": "O", "upad": "U"}[cols_prefix]
    out = {}
    for i in range(start, start + count):
        out[f"{prefix}_{i - start}"] = {
            "id": f"{id_prefix}{id_base + i}",
            "center": _point_dict(int(pt[i]), pt_xy),
            "width": float(wh[i, 0]),
            "height": float(wh[i, 1]),
            "layer": int(layer[i]),
            "shape": shape[i],
        }
    return out


def arrays_to_dict(iobs: dict) -> dict:
    """Losslessly reconstruct the legacy nested-dict observation.

    Key order, string ids ("P/E/D/O/U/T/V/Q{k}") and nesting match
    ``build_json_observation()`` byte-for-byte on canonical inputs —
    the contract cold consumers (LLM serializer, viz/debug, JSON dumps)
    rely on. Passthrough fields are returned by reference.
    """
    if not is_indexed(iobs):
        return iobs

    bs = iobs["board_static"]
    pt_xy = bs["pt_xy"]

    boardlines = {}
    edge_pt, edge_w = bs["edge_pt"], bs["edge_w"]
    edge_mid = bs.get("edge_mid")
    for i in range(len(edge_pt)):
        entry = {
            "id": f"E{i}",
            "p1": _point_dict(int(edge_pt[i, 0]), pt_xy),
            "p2": _point_dict(int(edge_pt[i, 1]), pt_xy),
            "width": float(edge_w[i]),
        }
        if edge_mid is not None and edge_mid[i] >= 0:
            entry["mid"] = _point_dict(int(edge_mid[i]), pt_xy)
        boardlines[f"edge_{i}"] = entry

    nets = {}
    for j, code in enumerate(bs["net_code"]):
        start = int(bs["net_pad_start"][j])
        count = int(bs["net_pad_count"][j])
        net_val: dict = {
            "net_name": bs["net_name"][j],
            "pads": {
                f"pad_{k}": pd
                for k, pd in enumerate(
                    _rect_dicts("pad", "pad", bs, start, count, 0).values()
                )
            },
        }
        constraints = bs["net_constraints"][j]
        if constraints is not None:
            net_val["constraints"] = constraints
        nets[f"net_{int(code)}"] = net_val

    board_static = {
        "bbox_x": bs["bbox_x"],
        "bbox_y": bs["bbox_y"],
        "bbox_w": bs["bbox_w"],
        "bbox_h": bs["bbox_h"],
        "scale": bs["scale"],
        "net_count": bs["net_count"],
        "copper_layers": bs["copper_layers"],
        "boardlines": boardlines,
        "nets": nets,
        "obstacles": _rect_dicts(
            "obs", "obs", bs, 0, len(bs["obs_pt"]), 0,
        ),
        "unconnected_pads": _rect_dicts(
            "upad", "upad", bs, 0, len(bs["upad_pt"]), 0,
        ),
        "board_constraints": bs["board_constraints"],
    }

    rg = iobs["routing_geometry"]
    d_pt = rg["pt_xy"]
    routing_geometry = {}
    for j, code in enumerate(rg["net_code"]):
        pt_base = int(rg["net_pt_start"][j])

        def _local_point(gidx: int) -> dict:
            return {
                "id": f"P{gidx - pt_base}",
                "xy": [float(d_pt[gidx, 0]), float(d_pt[gidx, 1])],
            }

        tracks = {}
        t0, tc = int(rg["trk_start"][j]), int(rg["trk_count"][j])
        for k in range(tc):
            r = t0 + k
            tracks[f"track_{k}"] = {
                "id": f"T{k}",
                "p1": _local_point(int(rg["trk_pt"][r, 0])),
                "p2": _local_point(int(rg["trk_pt"][r, 1])),
                "width": float(rg["trk_w"][r]),
                "layer": int(rg["trk_layer"][r]),
            }
        vias = {}
        v0, vc = int(rg["via_start"][j]), int(rg["via_count"][j])
        for k in range(vc):
            r = v0 + k
            vias[f"via_{k}"] = {
                "id": f"V{k}",
                "center": _local_point(int(rg["via_pt"][r])),
                "layer_start": int(rg["via_ls"][r]),
                "layer_end": int(rg["via_le"][r]),
                "via_width": float(rg["via_dia"][r]),
            }
        points = []
        r0, rc = int(rg["rat_start"][j]), int(rg["rat_count"][j])
        for k in range(rc):
            r = r0 + k
            points.append({
                "id": f"Q{int(rg['rat_q'][r])}",
                "xy": [float(rg["rat_xy"][r, 0]), float(rg["rat_xy"][r, 1])],
                "layer": int(rg["rat_layer"][r]),
            })
        routing_geometry[f"net_{int(code)}"] = {
            "tracks": tracks,
            "vias": vias,
            "points": points,
        }

    out = {
        "board_static": board_static,
        "routing_geometry": routing_geometry,
        "router_head": iobs["router_head"],
        "drc_violations": iobs["drc_violations"],
        "action_history": iobs["action_history"],
        "closed_nets": iobs["closed_nets"],
    }
    if "_aug" in iobs:
        out["_aug"] = iobs["_aug"]
    return out
