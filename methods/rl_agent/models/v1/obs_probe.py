"""Obs-semantics self-probe — ckpt-embedded digest of the tokenizer walk.

Why this exists: a change of obs *semantics* that keeps tensor shapes (e.g. a
different coordinate normalization) leaves no evidence in ckpt args (no knob
changed) or in weight shapes (no tensor resized), so a checkpoint would
silently eval under a different coordinate meaning. The other two defenses —
args replay and weight-shape self-detection — cannot see this class by
construction.

Mechanism: every checkpoint embeds a PROBE — a fixed synthetic observation
plus the sha256 of the tokenizer's pure-CPU Phase-1 walk (``_walk_obs``) over
it, computed by the code that trained the checkpoint. Loading re-encodes the
*stored* probe obs with the current code and compares digests. The contract
is exactly "the obs semantics this policy was trained under still hold":

- semantics change (values, normalization, ordering)  → digest differs → error
- code change that preserves encoder output           → digest equal   → silent
- an obs element the stored probe does not carry       → the elements it does
  carry still encode the same → pass (matching how the compat shims treat
  such additions)
- current code cannot encode the stored probe at all   → error (that inability
  is itself the incompatibility evidence)

Because the comparison baseline travels inside the checkpoint, the repo-side
probe builder can evolve without mis-judging any checkpoint, and no version
constant is ever bumped.

Escape hatch (deliberate archaeology only): ``CADAGENT_ALLOW_OBS_MISMATCH=1``
downgrades the load-time hard error to a loud warning and stamps
``policy.obs_schema_mismatch`` so result writers can mark the output.
A checkpoint that carries no probe cannot be checked — loaders print a note
and skip it.

Determinism: the walk is pure CPU/numpy; floats are rounded to 1e-6 before
hashing to absorb libm/BLAS jitter across hosts. Unknown types in the walk
output raise (never silently repr'd) so a future walk-format change that this
canonicalizer cannot faithfully hash fails loud instead of hashing garbage.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Probe observation
# ---------------------------------------------------------------------------

def build_probe_obs() -> dict:
    """Synthetic mid-route observation touching every tokenizer walk path.

    Coverage: BOARD · NET (open + closed, distinct constraints) · PAD (SMD on
    both copper layers, THT ``layer == 0``, non-rect shape bucket) · TRACK ·
    VIA · RATSNEST · EDGE (rect segments + one arc with midpoint) · OBSTACLE
    (NPTH circle, roundrect NC pad, plus a polygon keep-out that must be
    FILTERED) · DRC (error + warning, net-bound + orphan) · HEAD (mid-route)
    · ACTION_HISTORY (pointer + pointer-less entries) · candidates (8-dir from
    the live head) · ``_aug`` with every transform axis non-identity so the
    normalization/aug math is inside the digest.

    Values are deliberately non-round/asymmetric so coordinate-role swaps
    can't cancel out.
    """
    bbox_x, bbox_y, bbox_w, bbox_h = 93.7, 41.3, 61.4, 38.9
    cx, cy = bbox_x + bbox_w / 2, bbox_y + bbox_h / 2

    def pt(pid: str, x: float, y: float) -> dict:
        return {"id": pid, "xy": [x, y]}

    edges: dict[str, Any] = {}
    corners = [
        (bbox_x, bbox_y, bbox_x + bbox_w, bbox_y),
        (bbox_x + bbox_w, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h),
        (bbox_x + bbox_w, bbox_y + bbox_h, bbox_x, bbox_y + bbox_h),
        (bbox_x, bbox_y + bbox_h, bbox_x, bbox_y),
    ]
    for i, (x1, y1, x2, y2) in enumerate(corners):
        edges[f"edge_{i}"] = {
            "id": f"E{i}", "p1": pt(f"P{2*i}", x1, y1),
            "p2": pt(f"P{2*i+1}", x2, y2), "width": 0.1,
        }
    edges["edge_4"] = {  # arc outline entry (outline_obs="arc" path)
        "id": "E4", "p1": pt("P8", bbox_x + 7.3, bbox_y),
        "p2": pt("P9", bbox_x + 12.1, bbox_y), "width": 0.1,
        "mid": pt("PM4", bbox_x + 9.7, bbox_y - 2.3),
    }

    def pad(pid: str, x: float, y: float, layer: int, w: float, h: float,
            shape: str = "rect") -> dict:
        return {
            "id": f"D{pid}", "center": pt(f"PP{pid}", x, y),
            "width": w, "height": h, "layer": layer, "shape": shape,
        }

    nets = {
        "net_1": {
            "net_name": "NET1",
            "pads": {
                "pad_0": pad("10", cx - 11.3, cy - 5.7, 1, 1.1, 0.7),
                "pad_1": pad("11", cx + 9.1, cy + 6.3, 2, 0.9, 0.9,
                             shape="circle"),
                "pad_2": pad("12", cx + 3.7, cy - 8.9, 0, 1.7, 1.7,
                             shape="oval"),  # THT: spans all copper
            },
            "constraints": {
                "track_width": 0.23, "clearance": 0.19,
                "via_diameter": 0.61, "via_drill": 0.31,
            },
        },
        "net_2": {  # closed net — exercises the `closed` NET channel
            "net_name": "NET2",
            "pads": {
                "pad_0": pad("20", cx - 4.9, cy + 11.7, 1, 0.6, 1.3,
                             shape="roundrect"),
                "pad_1": pad("21", cx + 13.9, cy - 3.1, 2, 0.6, 1.3),
            },
            "constraints": {
                "track_width": 0.31, "clearance": 0.27,
                "via_diameter": 0.73, "via_drill": 0.37,
            },
        },
    }

    board_static = {
        "bbox_x": bbox_x, "bbox_y": bbox_y,
        "bbox_w": bbox_w, "bbox_h": bbox_h,
        "scale": max(bbox_w, bbox_h),
        "net_count": 2,
        "copper_layers": 2,
        "boardlines": edges,
        "nets": nets,
        "obstacles": {
            "obst_0": {  # NPTH hole — thru sentinel layer 0
                "id": "O0", "center": pt("PO0", cx - 17.9, cy + 2.3),
                "width": 2.1, "height": 2.1, "layer": 0, "shape": "circle",
            },
            "obst_1": {  # rule-area keep-out — MUST be filtered by the walk
                "id": "O1", "center": pt("PO1", cx, cy - 13.1),
                "width": 8.0, "height": 5.0, "layer": 1, "shape": "polygon",
            },
        },
        "unconnected_pads": {
            "nc_0": {
                "id": "NC0", "center": pt("PN0", cx + 16.3, cy + 9.7),
                "width": 1.3, "height": 0.8, "layer": 2, "shape": "roundrect",
            },
        },
        "board_constraints": {
            "clearance_mm": 0.19, "track_width_mm": 0.23,
            "via_diameter_mm": 0.61, "via_drill_mm": 0.31,
            "uvia_diameter_mm": -1.0, "uvia_drill_mm": -1.0,
        },
    }

    routing_geometry = {
        "net_1": {
            "tracks": {
                "track_0": {
                    "id": "T0", "p1": pt("PT0", cx - 11.3, cy - 5.7),
                    "p2": pt("PT1", cx - 2.9, cy - 1.3),
                    "width": 0.23, "layer": 1,
                },
            },
            "vias": {
                "via_0": {
                    "id": "V0", "center": pt("PV0", cx - 2.9, cy - 1.3),
                    "layer_start": 1, "layer_end": 2, "via_width": 0.61,
                },
            },
            "points": [
                {"id": "Q0", "xy": [cx + 9.1, cy + 6.3], "layer": 2},
            ],
        },
    }

    router_head = {
        "current_xy": [cx - 2.9, cy - 1.3],
        "current_layer": 2,
        "current_net": 1,
        "router_state_code": 1,
        "via_toggle": True,
        "is_routing": True,
        "is_dragging": False,
        "routing_mode": 1,
        "current_net_phase": 2,
        "step": 17,
        "step_ratio": 0.35,
        "steps_remaining": 333,
    }

    return {
        "board_static": board_static,
        "routing_geometry": routing_geometry,
        "router_head": router_head,
        "closed_nets": [2],
        "drc_violations": [
            {"x_mm": cx + 1.7, "y_mm": cy + 0.9, "layer": 2, "type_id": 1,
             "severity": 0x20, "net_names": ["NET1"]},
            {"x_mm": cx - 6.1, "y_mm": cy + 4.3, "layer": 1, "type_id": 3,
             "severity": 0, "net_names": []},  # warning + orphan
        ],
        "action_history": [
            {"action_type": 2, "success": True, "has_pointer": True,
             "pointer_xy": [cx - 2.9, cy - 1.3], "pointer_layer": 1,
             "routing_mode": 1, "net_id": 1},
            {"action_type": 0, "success": False, "has_pointer": False,
             "routing_mode": -1, "net_id": None},
        ],
        # Every aug axis non-identity so the transform math is digested.
        "_aug": {
            "scale_x": 1.15, "scale_y": 0.9,
            "aug_cx": cx - 3.3, "aug_cy": cy + 2.9,
            "axis_swap": True, "flip_x": -1, "flip_y": 1,
            "nn_dx": 0.05, "nn_dy": -0.03, "nn_zoom": 1.07,
            "slot_perm": [1, 0],
            "directional_candidates": None,
            "connectivity_filter": False,
        },
    }


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

_FLOAT_DECIMALS = 6


def _canon(x: Any) -> Any:
    """Canonicalize a walk value tree for stable JSON hashing.

    Floats round to 1e-6 (absorbs cross-host libm/BLAS jitter); unknown types
    raise so a walk-format change this cannot faithfully hash fails loud.
    """
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, torch.Tensor):
        return _canon(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.floating):
            return np.round(x.astype(np.float64), _FLOAT_DECIMALS).tolist()
        return x.tolist()
    if isinstance(x, np.generic):
        return _canon(x.item())
    if isinstance(x, float):
        return round(x, _FLOAT_DECIMALS)
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    raise TypeError(
        f"obs_probe cannot canonicalize walk value of type {type(x)!r} — "
        "extend _canon() deliberately rather than hashing garbage"
    )


def probe_digest(tokenizer: Any, obs: dict | None = None) -> str:
    """sha256 over the tokenizer's Phase-1 CPU walk of ``obs``.

    ``tokenizer`` is a ``BatchedStateTokenizer``; only its pure-CPU
    ``_walk_obs`` runs (no weights touched), so the digest fingerprints obs
    *semantics* under that tokenizer's config — which is replayed from ckpt
    args at load time, making save-side and load-side configs identical.
    """
    probe = build_probe_obs() if obs is None else copy.deepcopy(obs)
    walk = tokenizer._walk_obs([probe])
    blob = json.dumps(_canon(walk), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
