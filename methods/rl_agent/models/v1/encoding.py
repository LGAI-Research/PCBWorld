"""v1 codec — stateless obs↔token / action / mask transforms.

The variant's **stateless** half (no learnable parameters): it turns the
observation dict into the structural quantities the network and the rollout
buffer consume, and decodes the policy's ``(action_type, pointer, mode)`` triple
back to env-space. The learnable embedding tables + transformer live in
``net`` ("trainable weights belong to net").

Three sections, relocated byte-for-byte (C1-models, "logic 0" per §7 B-rule)
from the former scattered codec modules:
  * **Geometry** — ``encode_layer`` (from ``policy/token_vocabulary``)
  * **State**    — norm context + dataclasses (from ``policy/state_tokenizer_common``)
                   + obs→net/cand ordering (from ``wrappers/state_converter``)
  * **Action**   — triple decode + pointer→value (from ``wrappers/action_converter``)
  * **Mask**     — the wrapper-less serving ``DistInputs`` (``dist_from_obs``); the masks
                   and the candidate pool themselves are env rules in
                   ``pcb_world.core.constraints`` (re-exported here)

Those source modules have been removed; import the codec from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pcb_world.core.action_schema import MODE_WALKAROUND
# The candidate pool and the act-time masks are env rules (pcb_world.core.constraints);
# re-exported here so the codec keeps one import surface.
from pcb_world.core.constraints import (  # noqa: F401
    FORCE_ROUTING_MODE,
    _parse_net_code,
    _sorted_net_keys,
    act_constraints,
    cand_mm_list_from_obs,
    cand_raw_list_from_obs,
    mask_payload,
    mode_mask,
    net_valid_mask,
    offboard_pointer_indices,
    offlayer_pointer_indices,
    sorted_net_codes_from_obs,
    start_route_pointer_indices,
)
from methods.rl_agent.models.v1.spec import (
    ACTION_ARITY, MAX_COPPER, NUM_ROUTING_MODES, DistInputs,
)


# ===========================================================================
# Geometry helpers (stateless scalar encoders)
# ===========================================================================
def encode_layer(layer_human: int, n_copper: int) -> tuple[float, float]:
    """Encode a copper layer as ``(dist_top, dist_bottom)`` in [0, 1]."""
    if not (1 <= layer_human <= n_copper):
        raise ValueError(
            f"layer_human={layer_human} must be in [1, {n_copper}]"
        )
    dist_top = (layer_human - 1) / MAX_COPPER
    dist_bot = (n_copper - layer_human) / MAX_COPPER
    return (dist_top, dist_bot)


# Boundary-shape family → shape_obs bucket id (rows of shape_embed — a
# checkpoint contract, see spec.NUM_SHAPE_BUCKETS). Source strings are the
# KiCad tokens the engine emits (pns_rl_router.cpp getPads) plus the
# parser-synthesized "polygon" for rule-area keepouts (never tokenized —
# filtered out before emission, mapped here only for totality).
_SHAPE_TO_BUCKET = {
    "rect": 0,
    "roundrect": 1,
    "circle": 2,
    "oval": 3,
    "trapezoid": 4,       # → other
    "chamfered_rect": 4,  # → other
    "custom": 4,          # → other
    "polygon": 4,         # → other (keepout entries; excluded upstream)
    "": 5,                # → unknown
}


def shape_bucket_id(shape: str) -> int:
    """Map a KiCad shape token to its shape_obs bucket (unseen → unknown)."""
    return _SHAPE_TO_BUCKET.get(shape, 5)


# ===========================================================================
# State — normalization context, output dataclass, obs→ordering
# ===========================================================================
@dataclass
class TokenizerOutput:
    """Batched tokenizer output."""

    token_embeddings: torch.Tensor  # (B, max_seq, d_model)
    net_indices: torch.Tensor       # (B, max_nets) int64, -1 = pad
    cand_indices: torch.Tensor      # (B, max_cands) int64, -1 = pad
    key_padding_mask: torch.Tensor  # (B, max_seq) bool, True = padded
    seq_lens: torch.Tensor          # (B,) int64
    cand_mm_list: list[list[tuple[float, float, int]]]
    slot_ids: torch.Tensor | None = None  # (B, max_seq) int64, -1 = no-slot


@dataclass
class NormContext:
    # Normalization divisor: exact post-aug bbox half-extent max(bw, bh)/2,
    # divided by the nn_zoom jitter — the bbox long axis maps to exactly
    # [-nn_zoom, +nn_zoom] in normalized coords ([-1, 1] at zoom 1).
    norm_scale: float
    cx: float
    cy: float
    n_copper: int
    # Post-aug bbox size values to emit in BOARD token features. For "none"
    # this equals obs bw/bh; for "new" this is (sx * bw, sy * bh).
    bbox_w_eff: float = 0.0
    bbox_h_eff: float = 0.0
    # Scheme tag — "new" or "none". Drives _norm_pos_edge dispatch.
    scheme: str = "none"
    # New-scheme params.
    aug_scale_x: float = 1.0
    aug_scale_y: float = 1.0
    aug_cx: float = 0.0
    aug_cy: float = 0.0
    # Scheme-independent post-transforms (apply after the scheme branch).
    axis_swap: bool = False
    flip_x: int = 1
    flip_y: int = 1
    nn_dx: float = 0.0
    nn_dy: float = 0.0


def _compute_norm_ctx(
    board_static: dict, aug: dict | None = None,
) -> NormContext:
    bw = board_static["bbox_w"]
    bh = board_static["bbox_h"]
    orig_cx = board_static["bbox_x"] + bw / 2
    orig_cy = board_static["bbox_y"] + bh / 2

    # Orthogonal post-transforms — read regardless of scheme. Absent keys
    # default to identity so old serialized aug dicts continue to work.
    axis_swap = bool(aug.get("axis_swap", False)) if aug else False
    flip_x = int(aug.get("flip_x", 1)) if aug else 1
    flip_y = int(aug.get("flip_y", 1)) if aug else 1
    nn_dx = float(aug.get("nn_dx", 0.0)) if aug else 0.0
    nn_dy = float(aug.get("nn_dy", 0.0)) if aug else 0.0
    # Feature-space uniform zoom (obs-only, like nn_dx/nn_dy): folded into
    # the divisor so positions AND dims scale together — a consistent zoom
    # of the whole scene, never a geometry distortion.
    nn_zoom = float(aug.get("nn_zoom", 1.0)) if aug else 1.0

    if aug and "scale_x" in aug:
        # New scheme.
        sx = float(aug.get("scale_x", 1.0))
        sy = float(aug.get("scale_y", 1.0))
        cxa = float(aug.get("aug_cx", orig_cx))
        cya = float(aug.get("aug_cy", orig_cy))
        new_bw = sx * bw
        new_bh = sy * bh
        new_cx = cxa + sx * (orig_cx - cxa)
        new_cy = cya + sy * (orig_cy - cya)
        return NormContext(
            norm_scale=max(new_bw, new_bh) / 2 / nn_zoom,
            cx=new_cx, cy=new_cy,
            n_copper=board_static["copper_layers"],
            bbox_w_eff=new_bw, bbox_h_eff=new_bh,
            scheme="new",
            aug_scale_x=sx, aug_scale_y=sy,
            aug_cx=cxa, aug_cy=cya,
            axis_swap=axis_swap,
            flip_x=flip_x, flip_y=flip_y,
            nn_dx=nn_dx, nn_dy=nn_dy,
        )

    # No scheme-specific aug — orthogonal axes may still be active.
    return NormContext(
        norm_scale=max(bw, bh) / 2 / nn_zoom,
        cx=orig_cx, cy=orig_cy,
        n_copper=board_static["copper_layers"],
        bbox_w_eff=bw, bbox_h_eff=bh,
        scheme="none",
        axis_swap=axis_swap,
        flip_x=flip_x, flip_y=flip_y,
        nn_dx=nn_dx, nn_dy=nn_dy,
    )


def _apply_post_transform(
    nx: float, ny: float, ctx: NormContext,
) -> tuple[float, float]:
    """Apply orthogonal scheme-independent post-transforms in order:
    axis_swap → sign_reflection → nn_input_trans. Combined with axis_swap,
    sign_reflection spans the D4 dihedral group.
    """
    if ctx.axis_swap:
        nx, ny = ny, nx
    return (
        ctx.flip_x * nx + ctx.nn_dx,
        ctx.flip_y * ny + ctx.nn_dy,
    )


def _maybe_swap_pair(a: float, b: float, ctx: NormContext) -> tuple[float, float]:
    """Swap an x/y-separated scalar feature pair (e.g. bbox_w/bbox_h,
    pad_width/pad_height) when axis_swap is active. Must be applied to
    every (x-scalar, y-scalar) pair emitted as token features, otherwise
    the model sees a scene whose coords are swapped but whose scalar
    extents are not — an inconsistent geometry.
    """
    return (b, a) if ctx.axis_swap else (a, b)


def _norm_pos(x: float, y: float, ctx: NormContext) -> tuple[float, float]:
    """Normalize a physical point that is NOT on the board boundary.

    Under "new" / "none" the tokenizer frame already is the post-aug frame,
    so we just center and divide.  A scheme-independent post-transform
    (sign_reflection + nn_input_trans) is applied last.
    """
    nx = (x - ctx.cx) / ctx.norm_scale
    ny = (y - ctx.cy) / ctx.norm_scale
    return _apply_post_transform(nx, ny, ctx)


def _norm_pos_edge(x: float, y: float, ctx: NormContext) -> tuple[float, float]:
    """Normalize a point that lies on the board boundary (edge p1/p2 or
    bbox origin). Under "new" the physical edge is still at its original
    position, so we apply the virtual scale around ``c_aug`` first.

    The scheme-independent post-transform is applied uniformly at the end.
    """
    if ctx.scheme == "new":
        vx = ctx.aug_cx + ctx.aug_scale_x * (x - ctx.aug_cx)
        vy = ctx.aug_cy + ctx.aug_scale_y * (y - ctx.aug_cy)
        nx = (vx - ctx.cx) / ctx.norm_scale
        ny = (vy - ctx.cy) / ctx.norm_scale
        return _apply_post_transform(nx, ny, ctx)
    return _norm_pos(x, y, ctx)


def _norm_dim(val: float, ctx: NormContext) -> float:
    return val / ctx.norm_scale * 100


def _safe_encode_layer(layer_human: int, n_copper: int) -> tuple[float, float]:
    """encode_layer with an (0,0) "unknown layer" sentinel for out-of-range ids.

    DRC violations on real boards can sit on NON-copper layers (silk/edge/…,
    e.g. layer_human=6 on a 4-copper board) — those must not crash the walk.
    Same sentinel the <1 case always used.
    """
    if layer_human < 1 or layer_human > n_copper:
        return (0.0, 0.0)
    return encode_layer(layer_human, n_copper)


# ---------------------------------------------------------------------------
# Per-ELEMENT ctx twins (batch-level Indexed-Obs walk). Same expressions and
# op ORDER as the scalar path, but every ctx parameter is a per-entity f64/bool
# array (np.repeat of per-obs values), so ONE numpy call chain covers a whole
# mixed-obs batch. np.where only SELECTS between fully computed branches —
# the selected element's value is bit-identical to the scalar branch.
# ---------------------------------------------------------------------------
def _norm_pos_elem(
    x, y, *, cx, cy, scale, swap, flip_x, flip_y, nn_dx, nn_dy,
):
    nx = (x - cx) / scale
    ny = (y - cy) / scale
    if swap.any():
        nx, ny = np.where(swap, ny, nx), np.where(swap, nx, ny)
    return (flip_x * nx + nn_dx, flip_y * ny + nn_dy)


def _norm_pos_edge_elem(
    x, y, *, is_new, aug_cx, aug_cy, aug_sx, aug_sy,
    cx, cy, scale, swap, flip_x, flip_y, nn_dx, nn_dy,
):
    if is_new.any():
        vx = aug_cx + aug_sx * (x - aug_cx)
        vy = aug_cy + aug_sy * (y - aug_cy)
        x = np.where(is_new, vx, x)
        y = np.where(is_new, vy, y)
    return _norm_pos_elem(
        x, y, cx=cx, cy=cy, scale=scale, swap=swap,
        flip_x=flip_x, flip_y=flip_y, nn_dx=nn_dx, nn_dy=nn_dy,
    )


def _norm_dim_elem(val, scale):
    return val / scale * 100


def _safe_encode_layer_elem(layer_human, n_copper):
    """Per-element twin: ``n_copper`` is a per-entity array."""
    over = layer_human > n_copper
    if over.any():
        i = int(np.flatnonzero(over)[0])
        raise ValueError(
            f"layer_human={int(layer_human[i])} must be in "
            f"[1, {int(n_copper[i])}]"
        )
    dist_top = (layer_human - 1) / MAX_COPPER
    dist_bot = (n_copper - layer_human) / MAX_COPPER
    low = layer_human < 1
    if low.any():
        dist_top = np.where(low, 0.0, dist_top)
        dist_bot = np.where(low, 0.0, dist_bot)
    return dist_top, dist_bot


# ===========================================================================
# Action — normalize the policy triple, resolve pointers to env-space
# ===========================================================================
def unpack_action(action) -> tuple[int, int, int]:
    """Normalise a policy action to a ``(action_type, ptr, mode)`` tuple.

    Accepts numpy arrays, torch tensors, and plain sequences of length 3.
    """
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if arr.size != ACTION_ARITY:
        raise ValueError(
            f"KiCadRLWrapper expects a length-{ACTION_ARITY} action "
            f"[action_type, pointer_idx, routing_mode], got shape {arr.shape}"
        )
    return int(arr[0]), int(arr[1]), int(arr[2])


def clamp_mode(routing_mode: int, num_modes: int = NUM_ROUTING_MODES) -> int:
    """Clamp routing_mode to ``[0, num_modes-1]`` (policy may send -1 when unused)."""
    if routing_mode < 0:
        return MODE_WALKAROUND  # default fallback for unused slots
    return min(int(routing_mode), num_modes - 1)


def pointer_to_net_id(sorted_net_codes: list[int], pointer_idx: int) -> int:
    """Map a pointer index into the tokenizer net pool → env net_id (-1 if OOR)."""
    if 0 <= pointer_idx < len(sorted_net_codes):
        return int(sorted_net_codes[pointer_idx])
    return -1


def pointer_to_cand(
    cand_mm: list[tuple[float, float, int]], pointer_idx: int,
) -> tuple[float, float, int]:
    """Map a pointer index into the candidate pool → (x_mm, y_mm, layer).

    Out-of-range → ``(nan, nan, -1)``.
    """
    if 0 <= pointer_idx < len(cand_mm):
        x, y, layer = cand_mm[pointer_idx]
        return float(x), float(y), int(layer)
    return float("nan"), float("nan"), -1


# ===========================================================================
# Mask — hard masks the policy applies before sampling
# ===========================================================================
def dist_from_obs(
    obs_list: list[dict],
    action_masks: list[np.ndarray],
    start_route_xys: list[tuple[float, float, int] | None],
    knobs: Any,
    *,
    policy_net_select: bool,
) -> tuple[DistInputs, list[list[tuple[float, float, int]]], list[list[int]]]:
    """Serving path without the wrapper (bare PCBWorld / JSON obs): per obs the
    candidate pool and net order the tokenizer sees, the :func:`mask_payload`
    under the checkpoint's mask ``knobs`` (``mask_start_point`` ·
    ``offboard_mask`` · ``force_walkaround`` · ``net_reselect``), stacked into
    one ``DistInputs``. Returns ``(dist, cand_mm per obs, sorted net codes per
    obs)`` — the pointer-decode tables of each row."""
    # The obs already carries its pool ``_aug`` (or none, on bare PCBWorld): nothing to add.
    acts = [
        act_constraints(
            obs, action_mask=am, cand_aug={}, start_route_xy=xy,
            mask_start_point=knobs.mask_start_point,
            offboard_mask=knobs.offboard_mask,
            force_walkaround=knobs.force_walkaround,
            net_reselect=knobs.net_reselect,
        )
        for obs, am, xy in zip(obs_list, action_masks, start_route_xys, strict=True)
    ]
    return (
        DistInputs.from_payloads([a.masks for a in acts], policy_net_select=policy_net_select),
        [a.cand_mm for a in acts], [a.sorted_net_codes for a in acts],
    )
