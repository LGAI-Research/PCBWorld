"""Batched State Tokenizer.

Turns a batch of observations into token embeddings with **one
``vocab.encode_TYPE`` call per entity type** instead of one call per
observation, so the GPU launch count scales with the number of entity
types rather than with the number of observations.

Produces **bit-equivalent** ``TokenizerOutput`` to the per-obs reference
tokenizer (``tests/helpers/reference_tokenizer.py``) when both share the
same ``vocab`` weights — checked by
``tests/test_state_tokenizer_batched.py``.

Phases (per ``forward(obs_list)``):
  1. CPU walk: one Python pass over obs_list, building per-type buffers
     and per-obs entity counts. Closed-form position layout.
  2. Numpy → single H2D copy per type.
  3. ``vocab.encode_TYPE`` once per type with per-entity head_xy + has_head.
  4. Scatter to padded ``(B, max_seq, d_model)`` via ``index_copy_``.
  5. Slot embedding + LayerNorm batched over the whole tensor.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from methods.rl_agent.models.v1.encoding import (
    TokenizerOutput,
    _compute_norm_ctx,
    _maybe_swap_pair,
    _norm_dim,
    _norm_pos,
    _norm_pos_edge,
    _norm_dim_elem,
    _norm_pos_edge_elem,
    _norm_pos_elem,
    _safe_encode_layer_elem,
    _safe_encode_layer,
    _sorted_net_keys,
    shape_bucket_id,
)
from methods.rl_agent.models.v1.spec import (
    ACT_IDLE,
    CandidateType,
    N_MAX_SLOTS,
    NUM_DRC_TYPES,
    StructuralToken as ST,
)
from methods.rl_agent.models.v1.embedding import TokenVocabulary

# DRC severity: 0x20 = error (emits 1.0 flag), anything else (warning, etc.) = 0.0.
# Single source: pcb_world.engine.drc (import-safe — the C++ router is imported lazily,
# and the codec already loads pcb_world.engine transitively via spec).
from pcb_world.core.indexed_obs import is_indexed
from pcb_world.engine.drc import DRC_SEVERITY_ERROR as _DRC_SEVERITY_ERROR
from pcb_world.vec.candidate_pool import (
    build_directional_candidates,
    collect_raw_candidates,
)


# Optional profiling hook. When non-None (set by an external profiler),
# BatchedStateTokenizer.forward records sub-region wall-clock seconds
# into the dict (keys: "walk", "h2d_encode", "scatter", "slot_emb").
# Default None → zero overhead in production.
_BATCHED_TIMER_HOOK: dict[str, list[float]] | None = None


def _head_state(rh: dict, ctx) -> tuple[list[float], float]:
    """Head xy in normalized coords + has-head flag (None/idle → (0,0)/0.0).
    Shared by ``_walk_dict``/``_walk_indexed`` (same operation order as each
    walk's inline code)."""
    is_head_active = (
        rh.get("is_routing", False)
        or int(rh.get("current_net_phase", 0)) > 0
    )
    if is_head_active:
        hxy_n = _norm_pos(rh["current_xy"][0], rh["current_xy"][1], ctx)
        return [hxy_n[0], hxy_n[1]], 1.0
    return [0.0, 0.0], 0.0


def _collect_cands_raw(obs: dict, rh: dict, aug, current_net_id):
    """Collect raw CAND material (including directional candidates) — the
    shared prep step for both walks."""
    cur_net_for_cands = current_net_id if (
        isinstance(current_net_id, int) and current_net_id > 0
    ) else None

    extra = None
    if rh.get("is_routing", False):
        head_xy_mm = rh["current_xy"]
        extra = build_directional_candidates(
            (head_xy_mm[0], head_xy_mm[1]), rh["current_layer"],
            mode=(aug.get("directional_candidates") if aug else None),
        )

    return collect_raw_candidates(obs, cur_net_for_cands, extra)


def _np_f8(lst: list, cols: int) -> np.ndarray:
    """float64 ``(N, cols)`` — preserves column count even for an empty list,
    so it stays concat/gather-compatible."""
    return np.asarray(lst, dtype=np.float64).reshape(len(lst), cols)


def _np_f8v(lst: list) -> np.ndarray:
    """float64 ``(N,)`` scalar column (e.g. the has-head flag)."""
    return np.asarray(lst, dtype=np.float64)


def _np_i8(lst: list) -> np.ndarray:
    """int64 ``(N,)`` (obs_idx/pos/slot/type id)."""
    return np.asarray(lst, dtype=np.int64)


class _SmallBlockBufs:
    """Shared buffers + per-obs emission for the small blocks
    (BOARD/NET/DRC/HEAD/ACTION_HISTORY/VAL·SOD).

    ``_walk_dict`` (json) and ``_walk_indexed`` (indexed arrays) differ
    only in the data **source**; their small-block token emission is
    identical, so it lives here once and a token-layout change is made in
    one place. Heavy types (EDGE/PAD/TRACK/VIA/RAT + the CAND body) are
    not here — each walk handles them its own way (per-entity loop vs
    vectorized finalize).

    Every ``emit_*`` returns the updated ``pos``. The properties finalize
    each field as a **numpy array** (f64/i64), the same container the
    heavy types use, so the walk cache (merge/bounds/gather) runs through
    a single numpy path with no per-type branching. Bit-identity between
    the two walks is guarded by tests/test_indexed_tokenizer.py.
    """

    def __init__(self, tok) -> None:
        self._tok = tok
        # BOARD: 1 per obs
        self.board_xy: list[list[float]] = []
        self.board_wh: list[list[float]] = []
        self.board_nc: list[list[float]] = []
        self.board_obs_idx: list[int] = []
        self.board_pos: list[int] = []
        # NET
        self.net_tw: list[list[float]] = []
        self.net_cl: list[list[float]] = []
        self.net_vd: list[list[float]] = []
        self.net_closed: list[list[float]] = []  # 1.0 = consumed by net_end this episode
        self.net_obs_idx: list[int] = []
        self.net_pos: list[int] = []
        self.net_slot: list[int] = []
        # DRC violation
        self.drc_xy: list[list[float]] = []
        self.drc_ld: list[list[float]] = []
        self.drc_type: list[int] = []
        self.drc_sev: list[float] = []
        self.drc_head: list[list[float]] = []
        self.drc_has: list[float] = []
        self.drc_obs_idx: list[int] = []
        self.drc_pos: list[int] = []
        self.drc_slot: list[int] = []
        # HEAD: 1 per obs
        self.head_xy_b: list[list[float]] = []
        self.head_layer: list[list[float]] = []
        self.head_rm: list[int] = []
        self.head_np: list[int] = []
        self.head_sr: list[list[float]] = []
        self.head_obs_idx: list[int] = []
        self.head_pos: list[int] = []
        self.head_slot: list[int] = []
        # ACTION_HISTORY: 3 per entry (at, pt, mode) × K entries per obs,
        # newest first. Always emitted; short/empty histories pad with the
        # idle sentinel (K=1 in legacy prev-action mode).
        self.pa_type: list[int] = []
        self.pa_succ: list[float] = []
        self.pa_xy: list[list[float]] = []
        self.pa_ld: list[list[float]] = []
        self.pa_has_ptr: list[float] = []
        self.pa_mode: list[int] = []
        self.pa_age: list[int] = []
        self.pa_slot: list[int] = []
        self.pa_obs_idx: list[int] = []
        self.pa_at_pos: list[int] = []
        self.pa_pt_pos: list[int] = []
        self.pa_mo_pos: list[int] = []
        # Structural VAL+SOD: 2 per obs
        self.val_obs_idx: list[int] = []
        self.val_pos: list[int] = []
        self.sod_obs_idx: list[int] = []
        self.sod_pos: list[int] = []

    # ---- Emit (per-obs; 1:1 correspondence with each walk's inline block) ----

    def emit_board(self, b: int, pos: int, bs: dict, ctx) -> int:
        self.board_xy.append(list(_norm_pos_edge(bs["bbox_x"], bs["bbox_y"], ctx)))
        self.board_wh.append(list(_maybe_swap_pair(
            _norm_dim(ctx.bbox_w_eff, ctx),
            _norm_dim(ctx.bbox_h_eff, ctx),
            ctx,
        )))
        self.board_nc.append([float(ctx.n_copper)])
        self.board_obs_idx.append(b)
        self.board_pos.append(pos)
        return pos + 1

    def emit_net_row(self, b: int, pos: int, slot: int, constraints: dict,
                     is_closed: bool, ctx, net_positions_b: list[int]) -> int:
        c = constraints
        self.net_tw.append([_norm_dim(c.get("track_width", 0), ctx)])
        self.net_cl.append([_norm_dim(c.get("clearance", 0), ctx)])
        self.net_vd.append([_norm_dim(c.get("via_diameter", 0), ctx)])
        self.net_closed.append([1.0 if is_closed else 0.0])
        self.net_obs_idx.append(b)
        self.net_pos.append(pos)
        self.net_slot.append(slot)
        net_positions_b.append(pos)
        return pos + 1

    def emit_drc(self, b: int, pos: int, violations, ctx,
                 head_xy_norm: list[float], has_head_obs: float,
                 real_name_to_slot: dict[str, int]) -> int:
        for vio in violations:
            self.drc_xy.append(list(_norm_pos(
                vio["x_mm"], vio["y_mm"], ctx,
            )))
            vdt, vdb = _safe_encode_layer(
                int(vio.get("layer", 1)), ctx.n_copper,
            )
            self.drc_ld.append([vdt, vdb])
            tid = int(vio.get("type_id", NUM_DRC_TYPES - 1))
            if tid < 0 or tid >= NUM_DRC_TYPES:
                tid = NUM_DRC_TYPES - 1
            self.drc_type.append(tid)
            self.drc_sev.append(
                1.0 if int(vio.get("severity", 0)) == _DRC_SEVERITY_ERROR else 0.0
            )
            self.drc_head.append(head_xy_norm)
            self.drc_has.append(has_head_obs)
            self.drc_obs_idx.append(b)
            self.drc_pos.append(pos)
            # Bind to first net's slot; orphan (empty net_names) → -1.
            nets_vio = vio.get("net_names") or []
            if nets_vio:
                self.drc_slot.append(real_name_to_slot.get(nets_vio[0], -1))
            else:
                self.drc_slot.append(-1)
            pos += 1
        return pos

    def emit_head(self, b: int, pos: int, rh: dict, ctx,
                  head_xy_norm: list[float], has_head_obs: float,
                  head_slot_obs: int) -> int:
        h_layer = rh["current_layer"]
        hdt, hdb = _safe_encode_layer(h_layer, ctx.n_copper)
        self.head_xy_b.append(head_xy_norm if has_head_obs else [0.0, 0.0])
        self.head_layer.append([hdt, hdb])
        self.head_rm.append(int(rh["routing_mode"]))
        self.head_np.append(int(rh["current_net_phase"]))
        tok = self._tok
        if tok.time_feature == "log_remaining":
            remaining = max(float(rh["steps_remaining"]), 0.0)
            self.head_sr.append([math.log1p(remaining) / tok._time_cap_log])
        elif tok.time_feature == "sin_remaining":
            remaining = max(float(rh["steps_remaining"]), 0.0)
            self.head_sr.append([remaining / tok._time_cap])
        elif tok.time_feature == "none":
            # Time-blind ablation: constant 0 through the same-width Fourier
            # slot — the policy cannot observe episode progress/deadline.
            self.head_sr.append([0.0])
        else:
            self.head_sr.append([float(rh["step_ratio"])])
        self.head_obs_idx.append(b)
        self.head_pos.append(pos)
        self.head_slot.append(head_slot_obs)
        return pos + 1

    def emit_action_history(self, b: int, pos: int, history, ctx,
                            slot_of) -> int:
        # Always emits exactly K entries (fixed token count). Entries the
        # obs doesn't have yet (episode start / short history) emit an
        # "idle" sentinel with success=True so the policy can distinguish
        # "no record" from a real past step; the age index still advances.
        # ``slot_of(net_code) -> slot`` binds an entry to its net's slot
        # embedding (SameNetBias); in legacy prev-action mode the single
        # entry is slot-free.
        vocab = self._tok.vocab
        legacy = vocab.legacy_action_history
        for age in range(vocab.action_history_len):
            pa = history[age] if age < len(history) else None
            if pa is None:
                self.pa_type.append(ACT_IDLE)
                self.pa_succ.append(1.0)
                self.pa_xy.append([0.0, 0.0])
                self.pa_ld.append([0.0, 0.0])
                self.pa_has_ptr.append(0.0)
                self.pa_mode.append(0)
                self.pa_slot.append(-1)
            else:
                self.pa_type.append(int(pa["action_type"]))
                self.pa_succ.append(1.0 if pa.get("success") else 0.0)
                # Normalize xy into the current board frame. When the
                # action had no pointer (net_select/net_end/finish/idle),
                # has_ptr=0 masks the coords — we still emit a numeric (0,0).
                if pa.get("has_pointer"):
                    pxy = _norm_pos(
                        pa["pointer_xy"][0], pa["pointer_xy"][1], ctx,
                    )
                    self.pa_xy.append([pxy[0], pxy[1]])
                    pdt, pdb = _safe_encode_layer(
                        int(pa.get("pointer_layer", 0)), ctx.n_copper,
                    )
                    self.pa_ld.append([pdt, pdb])
                    self.pa_has_ptr.append(1.0)
                else:
                    self.pa_xy.append([0.0, 0.0])
                    self.pa_ld.append([0.0, 0.0])
                    self.pa_has_ptr.append(0.0)
                rm = int(pa.get("routing_mode", -1))
                self.pa_mode.append(rm if rm >= 0 else 0)
                nid = pa.get("net_id")
                self.pa_slot.append(
                    -1 if (legacy or nid is None) else slot_of(int(nid))
                )
            self.pa_age.append(age)
            self.pa_obs_idx.append(b)
            self.pa_at_pos.append(pos); pos += 1
            self.pa_pt_pos.append(pos); pos += 1
            self.pa_mo_pos.append(pos); pos += 1
        return pos

    def emit_val_sod(self, b: int, pos: int) -> int:
        self.val_obs_idx.append(b)
        self.val_pos.append(pos)
        pos += 1
        self.sod_obs_idx.append(b)
        self.sod_pos.append(pos)
        pos += 1
        return pos

    # ---- Tuples for the _walk_obs return dict (same layout; containers finalized as numpy) ----

    @property
    def board(self):
        return (_np_f8(self.board_xy, 2), _np_f8(self.board_wh, 2),
                _np_f8(self.board_nc, 1),
                _np_i8(self.board_obs_idx), _np_i8(self.board_pos))

    @property
    def net(self):
        return (_np_f8(self.net_tw, 1), _np_f8(self.net_cl, 1),
                _np_f8(self.net_vd, 1), _np_f8(self.net_closed, 1),
                _np_i8(self.net_obs_idx), _np_i8(self.net_pos),
                _np_i8(self.net_slot))

    @property
    def drc(self):
        return (_np_f8(self.drc_xy, 2), _np_f8(self.drc_ld, 2),
                _np_i8(self.drc_type), _np_f8v(self.drc_sev),
                _np_f8(self.drc_head, 2), _np_f8v(self.drc_has),
                _np_i8(self.drc_obs_idx), _np_i8(self.drc_pos),
                _np_i8(self.drc_slot))

    @property
    def head(self):
        return (_np_f8(self.head_xy_b, 2), _np_f8(self.head_layer, 2),
                _np_i8(self.head_rm), _np_i8(self.head_np),
                _np_f8(self.head_sr, 1), _np_i8(self.head_obs_idx),
                _np_i8(self.head_pos), _np_i8(self.head_slot))

    @property
    def action_history(self):
        return (_np_i8(self.pa_type), _np_f8v(self.pa_succ),
                _np_f8(self.pa_xy, 2), _np_f8(self.pa_ld, 2),
                _np_f8v(self.pa_has_ptr), _np_i8(self.pa_mode),
                _np_i8(self.pa_age), _np_i8(self.pa_slot),
                _np_i8(self.pa_obs_idx),
                _np_i8(self.pa_at_pos), _np_i8(self.pa_pt_pos),
                _np_i8(self.pa_mo_pos))

    @property
    def val(self):
        return (_np_i8(self.val_obs_idx), _np_i8(self.val_pos))

    @property
    def sod(self):
        return (_np_i8(self.sod_obs_idx), _np_i8(self.sod_pos))


class BatchedStateTokenizer(nn.Module):
    """Batched alternative to :class:`StateTokenizer`."""

    def __init__(
        self,
        d_model: int = 128,
        n_freq: int = 32,
        max_seq_len: int = 10000,
        coord_encoding: str = "fourier",
        mlp_hidden: int = 128,
        disable_slot_emb: bool = False,
        legacy_pad_layer_encoding: bool = False,
        legacy_net_encoding: bool = False,
        legacy_edge_encoding: bool = False,
        time_feature: str = "step_ratio",
        time_feature_cap: int = 10000,
        n_max_slots: int = N_MAX_SLOTS,
        action_history_len: int = 1,
        legacy_action_history: bool = False,
        obstacle_obs: bool = False,
        shape_obs: bool = False,
    ) -> None:
        super().__init__()
        if time_feature not in ("step_ratio", "log_remaining", "sin_remaining",
                                "none"):
            raise ValueError(
                f"time_feature must be 'step_ratio', 'log_remaining', "
                f"'sin_remaining', or 'none', got {time_feature!r}"
            )
        # "sin_remaining": linear u = remaining/cap through a dedicated
        # ladder whose top rung has period 2 steps (base^(2·n_freq−1) = cap)
        # — transformer-PE anchoring, uniform ±1-step resolution.
        time_fourier_base = None
        if time_feature == "sin_remaining":
            time_fourier_base = float(time_feature_cap) ** (
                1.0 / (2 * n_freq - 1)
            )
        self.vocab = TokenVocabulary(
            n_max_slots=n_max_slots,
            d_model=d_model,
            n_freq=n_freq,
            coord_encoding=coord_encoding,
            mlp_hidden=mlp_hidden,
            disable_slot_emb=disable_slot_emb,
            legacy_pad_layer_encoding=legacy_pad_layer_encoding,
            legacy_net_encoding=legacy_net_encoding,
            legacy_edge_encoding=legacy_edge_encoding,
            time_fourier_base=time_fourier_base,
            action_history_len=action_history_len,
            legacy_action_history=legacy_action_history,
            obstacle_obs=obstacle_obs,
            shape_obs=shape_obs,
        )
        # Emission gate for OBSTACLE tokens (netless blockers): off (default)
        # ⇒ no OBSTACLE token is emitted. shape_obs gates only the
        # encoder-side additive channel; the walk always carries the
        # shape_id columns (fixed tuple arity).
        self.obstacle_obs = bool(obstacle_obs)
        self.shape_obs = bool(shape_obs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # HEAD-token time scalar: all modes are a scalar into the same-width
        # Fourier slot and add no weights, so checkpoints stay
        # weight-compatible across modes (sin_remaining's ladder is a
        # non-persistent buffer rebuilt from ckpt args).
        self.time_feature = time_feature
        self._time_cap = float(time_feature_cap)
        self._time_cap_log = math.log1p(float(time_feature_cap))

    # No static cache to clear (API parity with the reference tokenizer).
    def clear_static_cache(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Phase 1: CPU walk + position layout
    # ------------------------------------------------------------------
    def _walk_obs(
        self, obs_list: list[dict],
    ) -> dict[str, Any]:
        """Phase-1 walk — dispatches on the observation format.

        indexed_v1 array observations take the vectorized ``_walk_indexed``
        path (numpy gathers, no per-entity Python loop); legacy nested
        dicts take ``_walk_dict``. Both return the identical buffer dict
        (bit-identical token content/order — see tests/test_indexed_tokenizer.py).
        Mixed-format batches are rejected: buffers must stay homogeneous.
        """
        if obs_list and is_indexed(obs_list[0]):
            return self._walk_indexed(obs_list)
        return self._walk_dict(obs_list)

    def _walk_dict(
        self, obs_list: list[dict],
    ) -> dict[str, Any]:
        """Single Python pass over obs_list. Builds per-type Python lists
        of raw values + per-obs metadata (positions, slot ids, head_xy).

        Returns a dict keyed by entity type and metadata. Nothing here
        touches torch / GPU.
        """
        B = len(obs_list)

        # ---------- Per-type raw buffers (lists of plain values) ----------
        # Each entry corresponds to one ENTITY (one token).
        # *_pos and *_obs_idx track where each entity goes in the final
        # padded (B, max_seq, d_model) output.
        # The small blocks (BOARD/NET/DRC/HEAD/ACTION_HISTORY/VAL·SOD) are
        # shared with _walk_indexed (_SmallBlockBufs); only the heavy-type
        # buffers (EDGE/PAD/OBSTACLE/TRACK/VIA/RAT/CAND) are built
        # per-entity here.
        sb = _SmallBlockBufs(self)

        # EDGE — xy_mid = the on-arc midpoint of arc entries (obs "mid" key,
        # outline_obs="arc"), chord midpoint for straight edges (degenerate
        # arc); is_arc marks entries whose mid came from the obs.
        edge_xy1: list[list[float]] = []
        edge_xy2: list[list[float]] = []
        edge_mid: list[list[float]] = []
        edge_arc: list[float] = []
        edge_w: list[list[float]] = []
        edge_obs_idx: list[int] = []
        edge_pos: list[int] = []

        # PAD — via-style layer range (layer_start, layer_end) so single-layer
        # pads use ls == le and thru-hole pads use (1, n_copper). Mirrors the
        # via encoding so the policy net sees one "copper layer span" primitive.
        # shape_id (boundary-shape bucket) is always collected for fixed tuple
        # arity; the encoder consumes it only under shape_obs.
        pad_xy: list[list[float]] = []
        pad_wh: list[list[float]] = []
        pad_ls: list[list[float]] = []
        pad_le: list[list[float]] = []
        pad_head: list[list[float]] = []
        pad_has: list[float] = []
        pad_obs_idx: list[int] = []
        pad_pos: list[int] = []
        pad_slot: list[int] = []
        pad_shape: list[int] = []

        # OBSTACLE — netless immovable blockers (obstacle_obs knob): NPTH
        # mounting holes / slots (board_static "obstacles", rule-area keepout
        # entries excluded by shape == "polygon") then net-less NC pads
        # ("unconnected_pads"). Pad-shaped channels, no net slot.
        obst_xy: list[list[float]] = []
        obst_wh: list[list[float]] = []
        obst_ls: list[list[float]] = []
        obst_le: list[list[float]] = []
        obst_head: list[list[float]] = []
        obst_has: list[float] = []
        obst_obs_idx: list[int] = []
        obst_pos: list[int] = []
        obst_shape: list[int] = []

        # TRACK
        tr_xy1: list[list[float]] = []
        tr_xy2: list[list[float]] = []
        tr_w: list[list[float]] = []
        tr_ld: list[list[float]] = []
        tr_head: list[list[float]] = []
        tr_has: list[float] = []
        tr_obs_idx: list[int] = []
        tr_pos: list[int] = []
        tr_slot: list[int] = []

        # VIA
        via_xy: list[list[float]] = []
        via_ls: list[list[float]] = []
        via_le: list[list[float]] = []
        via_dia: list[list[float]] = []
        via_head: list[list[float]] = []
        via_has: list[float] = []
        via_obs_idx: list[int] = []
        via_pos: list[int] = []
        via_slot: list[int] = []

        # RAT
        rat_xy: list[list[float]] = []
        rat_head: list[list[float]] = []
        rat_has: list[float] = []
        rat_obs_idx: list[int] = []
        rat_pos: list[int] = []
        rat_slot: list[int] = []

        # CAND
        cand_type: list[int] = []
        cand_xy: list[list[float]] = []
        cand_ld: list[list[float]] = []
        cand_head: list[list[float]] = []
        cand_has: list[float] = []
        cand_obs_idx: list[int] = []
        cand_pos: list[int] = []
        cand_slot: list[int] = []

        # ---------- Per-obs outputs ----------
        seq_lens = [0] * B
        net_positions: list[list[int]] = [[] for _ in range(B)]
        cand_positions: list[list[int]] = [[] for _ in range(B)]
        cand_mm_list: list[list[tuple[float, float, int]]] = [[] for _ in range(B)]
        slot_perm_per_obs: list[list[int] | None] = [None] * B

        # ---------- Walk each obs ----------
        for b, obs in enumerate(obs_list):
            bs = obs["board_static"]
            rg = obs.get("routing_geometry", {})
            rh = obs["router_head"]
            aug = obs.get("_aug")

            ctx = _compute_norm_ctx(bs, aug)
            net_keys = _sorted_net_keys(bs.get("nets", {}))
            if len(net_keys) > self.vocab.n_max_slots:
                raise ValueError(
                    f"Board has {len(net_keys)} nets but slot table has "
                    f"only {self.vocab.n_max_slots} slots."
                )
            net_to_slot = {nk: k for k, nk in enumerate(net_keys)}

            # DRC violations arrive with real net-name strings (KiCad side),
            # while slot assignment is keyed by "net_<code>". Build the real
            # name → slot map so DRC tokens can bind to the correct slot.
            real_name_to_slot: dict[str, int] = {}
            for nk in net_keys:
                real = bs["nets"][nk].get("net_name")
                if isinstance(real, str) and real:
                    real_name_to_slot.setdefault(real, net_to_slot[nk])

            slot_perm_per_obs[b] = (
                aug.get("slot_perm") if aug is not None else None
            )

            # Head xy in normalized coords (None when idle).
            head_xy_norm, has_head_obs = _head_state(rh, ctx)

            pos = 0  # current token position within this obs's sequence

            # ---------- BOARD ----------
            pos = sb.emit_board(b, pos, bs, ctx)

            # ---------- EDGE ----------
            edges = bs.get("boardlines", {})
            for e in edges.values():
                x1, y1 = e["p1"]["xy"]
                x2, y2 = e["p2"]["xy"]
                m = e.get("mid")
                if m is not None:
                    mx, my = m["xy"]
                    edge_arc.append(1.0)
                else:
                    mx = (x1 + x2) / 2.0
                    my = (y1 + y2) / 2.0
                    edge_arc.append(0.0)
                edge_xy1.append(list(_norm_pos_edge(x1, y1, ctx)))
                edge_xy2.append(list(_norm_pos_edge(x2, y2, ctx)))
                edge_mid.append(list(_norm_pos_edge(mx, my, ctx)))
                edge_w.append([_norm_dim(e["width"], ctx)])
                edge_obs_idx.append(b)
                edge_pos.append(pos)
                pos += 1

            # ---------- NET + PAD ----------
            # Nets consumed by net_end this episode (dynamic; obs top-level).
            closed_codes = {f"net_{c}" for c in (obs.get("closed_nets") or [])}
            for nk in net_keys:
                slot = net_to_slot[nk]
                net = bs["nets"][nk]
                pos = sb.emit_net_row(
                    b, pos, slot, net.get("constraints", {}),
                    nk in closed_codes, ctx, net_positions[b],
                )

                pads = net.get("pads", {})
                for pad in pads.values():
                    pad_xy.append(list(_norm_pos(
                        pad["center"]["xy"][0], pad["center"]["xy"][1], ctx,
                    )))
                    pad_wh.append(list(_maybe_swap_pair(
                        _norm_dim(pad["width"], ctx),
                        _norm_dim(pad["height"], ctx),
                        ctx,
                    )))
                    # Via-style (layer_start, layer_end) encoding. Single-layer
                    # SMD/connect pads use ls == le == pad["layer"]. Thru-hole
                    # pads (parser sentinel ``layer == 0`` = "spans every copper
                    # layer") get expanded to (1, n_copper) — same representation
                    # the engine emits for a thru via, so the policy net learns
                    # one geometric primitive for both.
                    pad_layer = pad["layer"]
                    if self.vocab.legacy_pad_layer_encoding:
                        layer_start = layer_end = pad_layer
                    elif pad_layer == 0:
                        layer_start, layer_end = 1, ctx.n_copper
                    else:
                        layer_start = layer_end = pad_layer
                    ls_dt, ls_db = _safe_encode_layer(layer_start, ctx.n_copper)
                    le_dt, le_db = _safe_encode_layer(layer_end, ctx.n_copper)
                    pad_ls.append([ls_dt, ls_db])
                    pad_le.append([le_dt, le_db])
                    pad_head.append(head_xy_norm)
                    pad_has.append(has_head_obs)
                    pad_obs_idx.append(b)
                    pad_pos.append(pos)
                    pad_slot.append(slot)
                    pad_shape.append(shape_bucket_id(pad.get("shape", "")))
                    pos += 1

            # ---------- OBSTACLE (obstacle_obs knob; netless blockers) ----------
            # Static-zone tail: NPTH holes/slots then NC pads, mirroring the
            # indexed static-table order (edges → nets/pads → obstacles →
            # unconnected pads). Rule-area keepout entries share the
            # "obstacles" dict but are engine-only — filtered by shape.
            if self.obstacle_obs:
                for src_key in ("obstacles", "unconnected_pads"):
                    for o in bs.get(src_key, {}).values():
                        shape = o.get("shape", "")
                        if shape == "polygon":  # rule-area keepout — not tokenized
                            continue
                        obst_xy.append(list(_norm_pos(
                            o["center"]["xy"][0], o["center"]["xy"][1], ctx,
                        )))
                        obst_wh.append(list(_maybe_swap_pair(
                            _norm_dim(o["width"], ctx),
                            _norm_dim(o["height"], ctx),
                            ctx,
                        )))
                        # Same layer-span primitive as PAD: parser sentinel
                        # layer == 0 (NPTH / thru NC pad) spans all copper.
                        o_layer = o["layer"]
                        if o_layer == 0:
                            layer_start, layer_end = 1, ctx.n_copper
                        else:
                            layer_start = layer_end = o_layer
                        ls_dt, ls_db = _safe_encode_layer(layer_start, ctx.n_copper)
                        le_dt, le_db = _safe_encode_layer(layer_end, ctx.n_copper)
                        obst_ls.append([ls_dt, ls_db])
                        obst_le.append([le_dt, le_db])
                        obst_head.append(head_xy_norm)
                        obst_has.append(has_head_obs)
                        obst_obs_idx.append(b)
                        obst_pos.append(pos)
                        obst_shape.append(shape_bucket_id(shape))
                        pos += 1

            # ---------- TRACK / VIA / RAT (per net in routing_geometry) ----------
            for nk in _sorted_net_keys(rg):
                net_geom = rg[nk]
                slot = net_to_slot.get(nk, -1)

                tracks = net_geom.get("tracks", {})
                for tr in tracks.values():
                    tr_xy1.append(list(_norm_pos(
                        tr["p1"]["xy"][0], tr["p1"]["xy"][1], ctx,
                    )))
                    tr_xy2.append(list(_norm_pos(
                        tr["p2"]["xy"][0], tr["p2"]["xy"][1], ctx,
                    )))
                    tr_w.append([_norm_dim(tr["width"], ctx)])
                    dt, db = _safe_encode_layer(tr["layer"], ctx.n_copper)
                    tr_ld.append([dt, db])
                    tr_head.append(head_xy_norm)
                    tr_has.append(has_head_obs)
                    tr_obs_idx.append(b)
                    tr_pos.append(pos)
                    tr_slot.append(slot)
                    pos += 1

                vias = net_geom.get("vias", {})
                for via in vias.values():
                    via_xy.append(list(_norm_pos(
                        via["center"]["xy"][0], via["center"]["xy"][1], ctx,
                    )))
                    ls_dt, ls_db = _safe_encode_layer(
                        via["layer_start"], ctx.n_copper,
                    )
                    le_dt, le_db = _safe_encode_layer(
                        via["layer_end"], ctx.n_copper,
                    )
                    via_ls.append([ls_dt, ls_db])
                    via_le.append([le_dt, le_db])
                    via_dia.append([_norm_dim(via.get("via_width", 0), ctx)])
                    via_head.append(head_xy_norm)
                    via_has.append(has_head_obs)
                    via_obs_idx.append(b)
                    via_pos.append(pos)
                    via_slot.append(slot)
                    pos += 1

                points = net_geom.get("points", [])
                for pt in points:
                    rat_xy.append(list(_norm_pos(
                        pt["xy"][0], pt["xy"][1], ctx,
                    )))
                    rat_head.append(head_xy_norm)
                    rat_has.append(has_head_obs)
                    rat_obs_idx.append(b)
                    rat_pos.append(pos)
                    rat_slot.append(slot)
                    pos += 1

            # ---------- DRC VIOLATIONS ----------
            # Inserted after TRACK/VIA/RAT and before HEAD so the decoder
            # treats them as part of the dynamic zone.
            violations = obs.get("drc_violations", []) or []
            pos = sb.emit_drc(b, pos, violations, ctx,
                              head_xy_norm, has_head_obs, real_name_to_slot)

            # ---------- HEAD ----------
            current_net_id = rh.get("current_net", -1)
            if isinstance(current_net_id, int) and current_net_id > 0:
                head_slot_obs = net_to_slot.get(f"net_{current_net_id}", -1)
            else:
                head_slot_obs = -1
            pos = sb.emit_head(b, pos, rh, ctx,
                               head_xy_norm, has_head_obs, head_slot_obs)

            # ---------- CAND ----------
            raw_cands = _collect_cands_raw(obs, rh, aug, current_net_id)

            for k, (x_mm, y_mm, ly, ct) in enumerate(raw_cands):
                ct_int = int(ct) if isinstance(ct, int) else int(ct.value)
                cand_type.append(ct_int)
                cand_xy.append(list(_norm_pos(x_mm, y_mm, ctx)))
                dt, db = _safe_encode_layer(ly, ctx.n_copper)
                cand_ld.append([dt, db])
                cand_head.append(head_xy_norm)
                cand_has.append(has_head_obs)
                cand_obs_idx.append(b)
                cand_pos.append(pos)
                cand_slot.append(head_slot_obs)
                cand_positions[b].append(pos)
                cand_mm_list[b].append((x_mm, y_mm, ly))
                pos += 1

            # ---------- ACTION_HISTORY (3 tokens per entry: at, pt, mode) ----------
            pos = sb.emit_action_history(
                b, pos, obs.get("action_history") or [], ctx,
                lambda nid: net_to_slot.get(f"net_{nid}", -1),
            )

            # ---------- VAL + SOD ----------
            pos = sb.emit_val_sod(b, pos)

            seq_lens[b] = pos

        return {
            "B": B,
            "seq_lens": seq_lens,
            "net_positions": net_positions,
            "cand_positions": cand_positions,
            "cand_mm_list": cand_mm_list,
            "slot_perm_per_obs": slot_perm_per_obs,
            "board": sb.board,
            # Heavy types are also finalized as numpy — same container/dtype
            # as _walk_indexed's vectorized finalize (so the walk cache runs
            # through a single numpy path).
            "edge": (_np_f8(edge_xy1, 2), _np_f8(edge_xy2, 2),
                     _np_f8(edge_w, 1),
                     _np_i8(edge_obs_idx), _np_i8(edge_pos),
                     _np_f8(edge_mid, 2), _np_f8v(edge_arc)),
            "net": sb.net,
            "pad": (_np_f8(pad_xy, 2), _np_f8(pad_wh, 2), _np_f8(pad_ls, 2),
                    _np_f8(pad_le, 2), _np_f8(pad_head, 2), _np_f8v(pad_has),
                    _np_i8(pad_obs_idx), _np_i8(pad_pos), _np_i8(pad_slot),
                    _np_i8(pad_shape)),
            "obstacle": (_np_f8(obst_xy, 2), _np_f8(obst_wh, 2),
                         _np_f8(obst_ls, 2), _np_f8(obst_le, 2),
                         _np_f8(obst_head, 2), _np_f8v(obst_has),
                         _np_i8(obst_obs_idx), _np_i8(obst_pos),
                         _np_i8(obst_shape)),
            "track": (_np_f8(tr_xy1, 2), _np_f8(tr_xy2, 2), _np_f8(tr_w, 1),
                      _np_f8(tr_ld, 2), _np_f8(tr_head, 2), _np_f8v(tr_has),
                      _np_i8(tr_obs_idx), _np_i8(tr_pos), _np_i8(tr_slot)),
            "via": (_np_f8(via_xy, 2), _np_f8(via_ls, 2), _np_f8(via_le, 2),
                    _np_f8(via_dia, 1), _np_f8(via_head, 2), _np_f8v(via_has),
                    _np_i8(via_obs_idx), _np_i8(via_pos), _np_i8(via_slot)),
            "rat": (_np_f8(rat_xy, 2), _np_f8(rat_head, 2), _np_f8v(rat_has),
                    _np_i8(rat_obs_idx), _np_i8(rat_pos), _np_i8(rat_slot)),
            "drc": sb.drc,
            "head": sb.head,
            "cand": (_np_i8(cand_type), _np_f8(cand_xy, 2), _np_f8(cand_ld, 2),
                     _np_f8(cand_head, 2), _np_f8v(cand_has),
                     _np_i8(cand_obs_idx), _np_i8(cand_pos), _np_i8(cand_slot)),
            "action_history": sb.action_history,
            "val": sb.val,
            "sod": sb.sod,
        }

    def _walk_indexed(
        self, obs_list: list[dict],
    ) -> dict[str, Any]:
        """indexed_v1 twin of ``_walk_dict`` — batch-level vectorization.

        Entity-heavy types (EDGE/PAD/OBSTACLE/TRACK/VIA/RAT) are processed with a
        CONSTANT number of numpy calls per type for the whole batch
        (PyG-Batch pattern): per-obs tables are concatenated once with
        index-shifted point-pool references, per-obs NormContext values
        become per-entity parameter columns via ``np.repeat``, and the
        per-ELEMENT ctx twins in ``encoding`` reproduce the scalar math
        op-for-op (``np.where`` only selects between fully computed
        branches). Bit-identity vs ``_walk_dict`` is enforced by
        tests/test_indexed_tokenizer.py.

        Small blocks (BOARD/NET/DRC/HEAD/ACTION_HISTORY/VAL/SOD) emit through the
        shared per-obs single source ``_SmallBlockBufs`` (identical to the
        ``_walk_dict`` path). Per-board derived orderings are memoized on
        the (episode-shared) static table dict under ``"_walk_cache"``.
        """
        B = len(obs_list)

        # Small blocks (BOARD/NET/DRC/HEAD/ACTION_HISTORY/VAL·SOD) use the
        # buffers shared with _walk_dict.
        sb = _SmallBlockBufs(self)

        # CAND — segment collectors (vectorized finalize, like the other
        # heavy types). Raw (x, y, layer, ctype) tuples accumulate batch-wide.
        c_b: list[int] = []; c_cnt: list[int] = []; c_base: list[int] = []
        c_slotl: list[int] = []
        c_raw: list[tuple] = []

        seq_lens = [0] * B
        net_positions: list[list[int]] = [[] for _ in range(B)]
        cand_positions: list[list[int]] = [[] for _ in range(B)]
        cand_mm_list: list[list[tuple[float, float, int]]] = [[] for _ in range(B)]
        slot_perm_per_obs: list[list[int] | None] = [None] * B

        # Per-obs ctx parameter rows -> (B, 17) f64. Column layout:
        # 0 cx | 1 cy | 2 norm_scale | 3 flip_x | 4 flip_y | 5 nn_dx | 6 nn_dy
        # | 7 axis_swap | 8 n_copper | 9 is_new | 10 aug_cx | 11 aug_cy
        # | 12 aug_sx | 13 aug_sy | 14 head_x | 15 head_y | 16 has_head
        ctx_rows: list[tuple] = []

        # Concatenated point pools with per-source offsets (dedup by id()
        # so update minibatches sharing one board pay its pool once).
        s_parts: list[np.ndarray] = []
        s_off_by_id: dict[int, int] = {}
        s_total = 0
        d_parts: list[np.ndarray] = []
        d_off_by_id: dict[int, int] = {}
        d_total = 0

        # Heavy-type segments: parallel python lists (cheap appends).
        e_b: list[int] = []; e_cnt: list[int] = []; e_base: list[int] = []
        e_off: list[int] = []; e_refs: list[np.ndarray] = []
        e_w: list[np.ndarray] = []; e_mid: list[np.ndarray] = []

        p_b: list[int] = []; p_cnt: list[int] = []; p_base: list[int] = []
        p_slotl: list[int] = []; p_off: list[int] = []
        p_refs: list[np.ndarray] = []; p_wh: list[np.ndarray] = []
        p_lay: list[np.ndarray] = []; p_shape: list[np.ndarray] = []

        o_b: list[int] = []; o_cnt: list[int] = []; o_base: list[int] = []
        o_off: list[int] = []; o_refs: list[np.ndarray] = []
        o_wh: list[np.ndarray] = []; o_lay: list[np.ndarray] = []
        o_shape: list[np.ndarray] = []

        t_b: list[int] = []; t_cnt: list[int] = []; t_base: list[int] = []
        t_slotl: list[int] = []; t_off: list[int] = []
        t_refs: list[np.ndarray] = []; t_wv: list[np.ndarray] = []
        t_lay: list[np.ndarray] = []

        v_b: list[int] = []; v_cnt: list[int] = []; v_base: list[int] = []
        v_slotl: list[int] = []; v_off: list[int] = []
        v_refs: list[np.ndarray] = []; v_dia: list[np.ndarray] = []
        v_lsv: list[np.ndarray] = []; v_lev: list[np.ndarray] = []

        r_b: list[int] = []; r_cnt: list[int] = []; r_base: list[int] = []
        r_slotl: list[int] = []; r_xy: list[np.ndarray] = []

        for b, obs in enumerate(obs_list):
            bs = obs["board_static"]
            rg = obs["routing_geometry"]
            rh = obs["router_head"]
            aug = obs.get("_aug")

            ctx = _compute_norm_ctx(bs, aug)

            cache = bs.get("_walk_cache")
            if cache is None:
                codes = bs["net_code"]
                order = np.argsort(codes, kind="stable")
                code_sorted = codes[order]
                cache = {
                    "order": order,
                    "code_sorted": code_sorted,
                    "net_to_slot": {
                        int(code_sorted[k]): k for k in range(len(order))
                    },
                    # Boundary-shape buckets, mapped once per board (strings
                    # → int64; the only per-entity Python work on the static
                    # tables, amortized by this cache).
                    "pad_shape_ids": np.fromiter(
                        (shape_bucket_id(s) for s in bs["pad_shape"]),
                        dtype=np.int64, count=len(bs["pad_shape"]),
                    ),
                }
                bs["_walk_cache"] = cache
            if self.obstacle_obs and "obstacle_tables" not in cache:
                # Merged netless-blocker table: NPTH holes/slots (rule-area
                # keepout rows excluded by shape == "polygon") then NC pads —
                # same order AND same polygon filter as the dict walk (the
                # filter is symmetric even though real pads can never carry
                # "polygon" — bit-identity by construction, not by data).
                # Refs point into the shared static point pool (registration
                # order edges → pads → obstacles → unconnected pads).
                keep = [i for i, s in enumerate(bs["obs_shape"])
                        if s != "polygon"]
                ukeep = [i for i, s in enumerate(bs["upad_shape"])
                         if s != "polygon"]
                cache["obstacle_tables"] = (
                    np.concatenate([bs["obs_pt"][keep], bs["upad_pt"][ukeep]]),
                    np.concatenate([bs["obs_wh"][keep], bs["upad_wh"][ukeep]],
                                   axis=0),
                    np.concatenate([bs["obs_layer"][keep],
                                    bs["upad_layer"][ukeep]]),
                    np.array(
                        [shape_bucket_id(bs["obs_shape"][i]) for i in keep]
                        + [shape_bucket_id(bs["upad_shape"][i])
                           for i in ukeep],
                        dtype=np.int64,
                    ),
                )
            order = cache["order"]
            code_sorted = cache["code_sorted"]
            net_to_slot = cache["net_to_slot"]
            S = len(order)
            if S > self.vocab.n_max_slots:
                raise ValueError(
                    f"Board has {S} nets but slot table has "
                    f"only {self.vocab.n_max_slots} slots."
                )

            slot_perm_per_obs[b] = (
                aug.get("slot_perm") if aug is not None else None
            )

            head_xy_norm, has_head_obs = _head_state(rh, ctx)

            ctx_rows.append((
                ctx.cx, ctx.cy, ctx.norm_scale,
                float(ctx.flip_x), float(ctx.flip_y), ctx.nn_dx, ctx.nn_dy,
                1.0 if ctx.axis_swap else 0.0, float(ctx.n_copper),
                1.0 if ctx.scheme == "new" else 0.0,
                ctx.aug_cx, ctx.aug_cy, ctx.aug_scale_x, ctx.aug_scale_y,
                head_xy_norm[0], head_xy_norm[1], has_head_obs,
            ))

            s_off = s_off_by_id.get(id(bs))
            if s_off is None:
                s_off = s_total
                s_off_by_id[id(bs)] = s_off
                s_parts.append(bs["pt_xy"])
                s_total += len(bs["pt_xy"])
            d_off = d_off_by_id.get(id(rg))
            if d_off is None:
                d_off = d_total
                d_off_by_id[id(rg)] = d_off
                d_parts.append(rg["pt_xy"])
                d_total += len(rg["pt_xy"])

            pos = 0

            # ---------- BOARD ----------
            pos = sb.emit_board(b, pos, bs, ctx)

            # ---------- EDGE (segment) ----------
            E = len(bs["edge_pt"])
            if E:
                e_b.append(b); e_cnt.append(E); e_base.append(pos)
                e_off.append(s_off)
                e_refs.append(bs["edge_pt"]); e_w.append(bs["edge_w"])
                em = bs.get("edge_mid")
                if em is None:  # indexed obs with no edge_mid table: all straight
                    em = np.full((E,), -1, dtype=np.int64)
                e_mid.append(em)
                pos += E

            # ---------- NET + PAD ----------
            closed_set = {int(c) for c in (obs.get("closed_nets") or [])}
            for k in range(S):
                j = int(order[k])
                code = int(code_sorted[k])
                pos = sb.emit_net_row(
                    b, pos, k, bs["net_constraints"][j] or {},
                    code in closed_set, ctx, net_positions[b],
                )

                n_pads = int(bs["net_pad_count"][j])
                if n_pads:
                    s0 = int(bs["net_pad_start"][j])
                    rows = slice(s0, s0 + n_pads)
                    p_b.append(b); p_cnt.append(n_pads); p_base.append(pos)
                    p_slotl.append(k); p_off.append(s_off)
                    p_refs.append(bs["pad_pt"][rows])
                    p_wh.append(bs["pad_wh"][rows])
                    p_lay.append(bs["pad_layer"][rows])
                    p_shape.append(cache["pad_shape_ids"][rows])
                    pos += n_pads

            # ---------- OBSTACLE (obstacle_obs knob; netless blockers) ----------
            if self.obstacle_obs:
                ot_refs, ot_wh, ot_lay, ot_shape = cache["obstacle_tables"]
                n_obst = len(ot_refs)
                if n_obst:
                    o_b.append(b); o_cnt.append(n_obst); o_base.append(pos)
                    o_off.append(s_off)
                    o_refs.append(ot_refs); o_wh.append(ot_wh)
                    o_lay.append(ot_lay); o_shape.append(ot_shape)
                    pos += n_obst

            # ---------- TRACK / VIA / RAT (per dyn net, ascending) ----------
            d_codes = rg["net_code"]
            for jd in range(len(d_codes)):
                slot = net_to_slot.get(int(d_codes[jd]), -1)

                tc = int(rg["trk_count"][jd])
                if tc:
                    t0 = int(rg["trk_start"][jd])
                    rows = slice(t0, t0 + tc)
                    t_b.append(b); t_cnt.append(tc); t_base.append(pos)
                    t_slotl.append(slot); t_off.append(d_off)
                    t_refs.append(rg["trk_pt"][rows])
                    t_wv.append(rg["trk_w"][rows])
                    t_lay.append(rg["trk_layer"][rows])
                    pos += tc

                vc = int(rg["via_count"][jd])
                if vc:
                    v0 = int(rg["via_start"][jd])
                    rows = slice(v0, v0 + vc)
                    v_b.append(b); v_cnt.append(vc); v_base.append(pos)
                    v_slotl.append(slot); v_off.append(d_off)
                    v_refs.append(rg["via_pt"][rows])
                    v_dia.append(rg["via_dia"][rows])
                    v_lsv.append(rg["via_ls"][rows])
                    v_lev.append(rg["via_le"][rows])
                    pos += vc

                rc = int(rg["rat_count"][jd])
                if rc:
                    r0 = int(rg["rat_start"][jd])
                    r_b.append(b); r_cnt.append(rc); r_base.append(pos)
                    r_slotl.append(slot)
                    r_xy.append(rg["rat_xy"][r0:r0 + rc])
                    pos += rc

            # ---------- DRC VIOLATIONS (legacy loop; list is small) ----------
            violations = obs.get("drc_violations", []) or []
            real_name_to_slot: dict[str, int] = {}
            if violations:
                for k in range(S):
                    real = bs["net_name"][int(order[k])]
                    if isinstance(real, str) and real:
                        real_name_to_slot.setdefault(real, k)
            pos = sb.emit_drc(b, pos, violations, ctx,
                              head_xy_norm, has_head_obs, real_name_to_slot)

            # ---------- HEAD ----------
            current_net_id = rh.get("current_net", -1)
            if isinstance(current_net_id, int) and current_net_id > 0:
                head_slot_obs = net_to_slot.get(current_net_id, -1)
            else:
                head_slot_obs = -1
            pos = sb.emit_head(b, pos, rh, ctx,
                               head_xy_norm, has_head_obs, head_slot_obs)

            # ---------- CAND (shared pool builder; small per-net loop) ----------
            raw_cands = _collect_cands_raw(obs, rh, aug, current_net_id)

            n_c = len(raw_cands)
            if n_c:
                c_b.append(b); c_cnt.append(n_c); c_base.append(pos)
                c_slotl.append(head_slot_obs)
                c_raw.extend(raw_cands)
                cand_positions[b].extend(range(pos, pos + n_c))
                cand_mm_list[b].extend(
                    (x_mm, y_mm, ly) for (x_mm, y_mm, ly, _ct) in raw_cands
                )
                pos += n_c

            # ---------- ACTION_HISTORY (3 tokens per entry: at, pt, mode) ----------
            pos = sb.emit_action_history(
                b, pos, obs.get("action_history") or [], ctx,
                lambda nid: net_to_slot.get(nid, -1),
            )

            # ---------- VAL + SOD ----------
            pos = sb.emit_val_sod(b, pos)

            seq_lens[b] = pos

        # ================= batched finalize =================
        P = (np.asarray(ctx_rows, dtype=np.float64) if ctx_rows
             else np.empty((0, 17), dtype=np.float64))
        s_pool = (np.concatenate(s_parts, axis=0) if s_parts
                  else np.empty((0, 2), dtype=np.float64))
        d_pool = (np.concatenate(d_parts, axis=0) if d_parts
                  else np.empty((0, 2), dtype=np.float64))

        def _seg_expand(bl, cntl, basel, slotl=None):
            """Segment lists -> per-entity (counts, obs_idx, pos, slot|None)."""
            cnt = np.asarray(cntl, dtype=np.int64)
            total = int(cnt.sum())
            obs_ent = np.repeat(np.asarray(bl, dtype=np.int64), cnt)
            cum0 = np.concatenate(([0], np.cumsum(cnt)[:-1]))
            pos_ent = (np.repeat(np.asarray(basel, dtype=np.int64), cnt)
                       + np.arange(total, dtype=np.int64)
                       - np.repeat(cum0, cnt))
            slot_ent = (np.repeat(np.asarray(slotl, dtype=np.int64), cnt)
                        if slotl is not None else None)
            return cnt, obs_ent, pos_ent, slot_ent

        def _pos_elem(x, y, cp):
            return _norm_pos_elem(
                x, y, cx=cp[:, 0], cy=cp[:, 1], scale=cp[:, 2],
                swap=cp[:, 7] != 0.0, flip_x=cp[:, 3], flip_y=cp[:, 4],
                nn_dx=cp[:, 5], nn_dy=cp[:, 6],
            )

        # ---- EDGE ----
        if e_b:
            cnt, obs_ent, pos_ent, _ = _seg_expand(e_b, e_cnt, e_base)
            cp = P[obs_ent]
            refs = (np.concatenate(e_refs, axis=0)
                    + np.repeat(np.asarray(e_off, dtype=np.int64), cnt)[:, None])
            g = s_pool[refs]                                   # (E, 2, 2)
            edge_kw = dict(
                is_new=cp[:, 9] != 0.0, aug_cx=cp[:, 10], aug_cy=cp[:, 11],
                aug_sx=cp[:, 12], aug_sy=cp[:, 13],
                cx=cp[:, 0], cy=cp[:, 1], scale=cp[:, 2],
                swap=cp[:, 7] != 0.0, flip_x=cp[:, 3], flip_y=cp[:, 4],
                nn_dx=cp[:, 5], nn_dy=cp[:, 6],
            )
            x1n, y1n = _norm_pos_edge_elem(g[:, 0, 0], g[:, 0, 1], **edge_kw)
            x2n, y2n = _norm_pos_edge_elem(g[:, 1, 0], g[:, 1, 1], **edge_kw)
            # On-arc midpoint where edge_mid >= 0, else the chord midpoint
            # (raw-coordinate math first, then the same normalizer — matches
            # the dict walk op-for-op).
            mid_rows = np.concatenate(e_mid, axis=0)
            has_mid = mid_rows >= 0
            mraw = s_pool[np.where(
                has_mid,
                mid_rows + np.repeat(np.asarray(e_off, dtype=np.int64), cnt),
                0,
            )]
            mx = np.where(has_mid, mraw[:, 0], (g[:, 0, 0] + g[:, 1, 0]) / 2.0)
            my = np.where(has_mid, mraw[:, 1], (g[:, 0, 1] + g[:, 1, 1]) / 2.0)
            xmn, ymn = _norm_pos_edge_elem(mx, my, **edge_kw)
            edge_bufs = (
                np.stack([x1n, y1n], axis=1),
                np.stack([x2n, y2n], axis=1),
                _norm_dim_elem(np.concatenate(e_w, axis=0), cp[:, 2])[:, None],
                obs_ent, pos_ent,
                np.stack([xmn, ymn], axis=1),
                has_mid.astype(np.float64),
            )
        else:
            edge_bufs = (np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 1)),
                         np.empty((0,), np.int64), np.empty((0,), np.int64),
                         np.empty((0, 2)), np.empty((0,)))

        # ---- PAD ----
        if p_b:
            cnt, obs_ent, pos_ent, slot_ent = _seg_expand(p_b, p_cnt, p_base, p_slotl)
            cp = P[obs_ent]
            refs = (np.concatenate(p_refs, axis=0)
                    + np.repeat(np.asarray(p_off, dtype=np.int64), cnt))
            g = s_pool[refs]                                   # (Pd, 2)
            nx, ny = _pos_elem(g[:, 0], g[:, 1], cp)
            wh = np.concatenate(p_wh, axis=0)
            w_n = _norm_dim_elem(wh[:, 0], cp[:, 2])
            h_n = _norm_dim_elem(wh[:, 1], cp[:, 2])
            swap = cp[:, 7] != 0.0
            if swap.any():                                     # _maybe_swap_pair
                w_n, h_n = np.where(swap, h_n, w_n), np.where(swap, w_n, h_n)
            layer = np.concatenate(p_lay, axis=0)
            nc_ent = cp[:, 8]
            if self.vocab.legacy_pad_layer_encoding:
                ls = layer
                le = layer.astype(np.float64)
            else:
                thru = layer == 0
                ls = np.where(thru, 1, layer)
                le = np.where(thru, nc_ent, layer)
            ls_dt, ls_db = _safe_encode_layer_elem(ls, nc_ent)
            le_dt, le_db = _safe_encode_layer_elem(le, nc_ent)
            pad_bufs = (
                np.stack([nx, ny], axis=1),
                np.stack([w_n, h_n], axis=1),
                np.stack([ls_dt, ls_db], axis=1),
                np.stack([le_dt, le_db], axis=1),
                cp[:, 14:16].copy(), cp[:, 16],
                obs_ent, pos_ent, slot_ent,
                np.concatenate(p_shape),
            )
        else:
            pad_bufs = (np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)),
                        np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)),
                        np.empty((0,), np.int64), np.empty((0,), np.int64),
                        np.empty((0,), np.int64), np.empty((0,), np.int64))

        # ---- OBSTACLE ----
        # PAD finalize twin minus slot/legacy-layer (knob is new-format only);
        # shape ids come pre-bucketed from the per-board obstacle_tables cache.
        if o_b:
            cnt, obs_ent, pos_ent, _ = _seg_expand(o_b, o_cnt, o_base)
            cp = P[obs_ent]
            refs = (np.concatenate(o_refs, axis=0)
                    + np.repeat(np.asarray(o_off, dtype=np.int64), cnt))
            g = s_pool[refs]                                   # (Od, 2)
            nx, ny = _pos_elem(g[:, 0], g[:, 1], cp)
            wh = np.concatenate(o_wh, axis=0)
            w_n = _norm_dim_elem(wh[:, 0], cp[:, 2])
            h_n = _norm_dim_elem(wh[:, 1], cp[:, 2])
            swap = cp[:, 7] != 0.0
            if swap.any():                                     # _maybe_swap_pair
                w_n, h_n = np.where(swap, h_n, w_n), np.where(swap, w_n, h_n)
            layer = np.concatenate(o_lay, axis=0)
            nc_ent = cp[:, 8]
            thru = layer == 0
            ls = np.where(thru, 1, layer)
            le = np.where(thru, nc_ent, layer)
            ls_dt, ls_db = _safe_encode_layer_elem(ls, nc_ent)
            le_dt, le_db = _safe_encode_layer_elem(le, nc_ent)
            obst_bufs = (
                np.stack([nx, ny], axis=1),
                np.stack([w_n, h_n], axis=1),
                np.stack([ls_dt, ls_db], axis=1),
                np.stack([le_dt, le_db], axis=1),
                cp[:, 14:16].copy(), cp[:, 16],
                obs_ent, pos_ent,
                np.concatenate(o_shape),
            )
        else:
            obst_bufs = (np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)),
                         np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)),
                         np.empty((0,), np.int64), np.empty((0,), np.int64),
                         np.empty((0,), np.int64))

        # ---- TRACK ----
        if t_b:
            cnt, obs_ent, pos_ent, slot_ent = _seg_expand(t_b, t_cnt, t_base, t_slotl)
            cp = P[obs_ent]
            refs = (np.concatenate(t_refs, axis=0)
                    + np.repeat(np.asarray(t_off, dtype=np.int64), cnt)[:, None])
            g = d_pool[refs]                                   # (L, 2, 2)
            x1n, y1n = _pos_elem(g[:, 0, 0], g[:, 0, 1], cp)
            x2n, y2n = _pos_elem(g[:, 1, 0], g[:, 1, 1], cp)
            dt, db = _safe_encode_layer_elem(
                np.concatenate(t_lay, axis=0), cp[:, 8],
            )
            track_bufs = (
                np.stack([x1n, y1n], axis=1),
                np.stack([x2n, y2n], axis=1),
                _norm_dim_elem(np.concatenate(t_wv, axis=0), cp[:, 2])[:, None],
                np.stack([dt, db], axis=1),
                cp[:, 14:16].copy(), cp[:, 16],
                obs_ent, pos_ent, slot_ent,
            )
        else:
            track_bufs = (np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 1)),
                          np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)),
                          np.empty((0,), np.int64), np.empty((0,), np.int64),
                          np.empty((0,), np.int64))

        # ---- VIA ----
        if v_b:
            cnt, obs_ent, pos_ent, slot_ent = _seg_expand(v_b, v_cnt, v_base, v_slotl)
            cp = P[obs_ent]
            refs = (np.concatenate(v_refs, axis=0)
                    + np.repeat(np.asarray(v_off, dtype=np.int64), cnt))
            g = d_pool[refs]                                   # (V, 2)
            nx, ny = _pos_elem(g[:, 0], g[:, 1], cp)
            ls_dt, ls_db = _safe_encode_layer_elem(
                np.concatenate(v_lsv, axis=0), cp[:, 8],
            )
            le_dt, le_db = _safe_encode_layer_elem(
                np.concatenate(v_lev, axis=0), cp[:, 8],
            )
            via_bufs = (
                np.stack([nx, ny], axis=1),
                np.stack([ls_dt, ls_db], axis=1),
                np.stack([le_dt, le_db], axis=1),
                _norm_dim_elem(np.concatenate(v_dia, axis=0), cp[:, 2])[:, None],
                cp[:, 14:16].copy(), cp[:, 16],
                obs_ent, pos_ent, slot_ent,
            )
        else:
            via_bufs = (np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)),
                        np.empty((0, 1)), np.empty((0, 2)), np.empty((0,)),
                        np.empty((0,), np.int64), np.empty((0,), np.int64),
                        np.empty((0,), np.int64))

        # ---- RAT ----
        if r_b:
            cnt, obs_ent, pos_ent, slot_ent = _seg_expand(r_b, r_cnt, r_base, r_slotl)
            cp = P[obs_ent]
            g = np.concatenate(r_xy, axis=0)                   # (Q, 2)
            nx, ny = _pos_elem(g[:, 0], g[:, 1], cp)
            rat_bufs = (
                np.stack([nx, ny], axis=1),
                cp[:, 14:16].copy(), cp[:, 16],
                obs_ent, pos_ent, slot_ent,
            )
        else:
            rat_bufs = (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)),
                        np.empty((0,), np.int64), np.empty((0,), np.int64),
                        np.empty((0,), np.int64))

        # ---- CAND ----
        if c_b:
            cnt, obs_ent, pos_ent, slot_ent = _seg_expand(c_b, c_cnt, c_base, c_slotl)
            cp = P[obs_ent]
            xs, ys, lys, cts = zip(*c_raw)
            cx_arr = np.asarray(xs, dtype=np.float64)
            cy_arr = np.asarray(ys, dtype=np.float64)
            cl_arr = np.asarray(lys, dtype=np.int64)
            ctype_arr = np.asarray(
                [int(c) if isinstance(c, int) else int(c.value) for c in cts],
                dtype=np.int64,
            )
            nx, ny = _pos_elem(cx_arr, cy_arr, cp)
            dt, db = _safe_encode_layer_elem(cl_arr, cp[:, 8])
            cand_bufs = (
                ctype_arr,
                np.stack([nx, ny], axis=1),
                np.stack([dt, db], axis=1),
                cp[:, 14:16].copy(), cp[:, 16],
                obs_ent, pos_ent, slot_ent,
            )
        else:
            cand_bufs = (np.empty((0,), np.int64), np.empty((0, 2)),
                         np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)),
                         np.empty((0,), np.int64), np.empty((0,), np.int64),
                         np.empty((0,), np.int64))

        return {
            "B": B,
            "seq_lens": seq_lens,
            "net_positions": net_positions,
            "cand_positions": cand_positions,
            "cand_mm_list": cand_mm_list,
            "slot_perm_per_obs": slot_perm_per_obs,
            "board": sb.board,
            "edge": edge_bufs,
            "net": sb.net,
            "pad": pad_bufs,
            "obstacle": obst_bufs,
            "track": track_bufs,
            "via": via_bufs,
            "rat": rat_bufs,
            "drc": sb.drc,
            "head": sb.head,
            "cand": cand_bufs,
            "action_history": sb.action_history,
            "val": sb.val,
            "sod": sb.sod,
        }

    # ------------------------------------------------------------------
    # Walk cache — eliminates re-tokenizing (walking) on the update path.
    #
    # walk is a pure function of the raw obs, and obs is immutable for the
    # duration of one PPO update, so the batched walk collect already
    # performed is kept **flat as-is** (_walk_obs's output is already
    # flat-batched: an entity-type concat + obs_idx per type). The update
    # then does only an index-gather per minibatch. Because _walk_obs
    # traverses obs in order and appends (per-obs state like pos resets for
    # each obs), concatenating batch walks (merge_walked) equals walking
    # the whole set directly, and rearranging per-sample contiguous ranges
    # (gather_walked) equals walking that subset directly (identical
    # elements and order → bit-equal downstream tensors).
    # ------------------------------------------------------------------
    # _walk_obs return schema is fixed: token-type key → the index of the
    # *_obs_idx list within the parallel-list tuple. If the schema changes,
    # merge_walked's guard fails loudly (prevents a silent mis-merge).
    _WALK_PER_OBS_KEYS = ("seq_lens", "net_positions", "cand_positions",
                          "cand_mm_list", "slot_perm_per_obs")
    _WALK_OBS_IDX_SLOT = {
        "board": 3, "edge": 3, "net": 4, "pad": 6, "obstacle": 6, "track": 6,
        "via": 6, "rat": 3, "drc": 6, "head": 5, "cand": 5,
        "action_history": 8, "val": 0, "sod": 0,
    }

    def walk_timed(self, obs_list: list[dict]) -> dict[str, Any]:
        """Batched ``_walk_obs`` + profiler walk-bucket timing.

        For walks performed OUTSIDE ``forward`` (e.g. the rollout
        ``budgeted_forward`` path, which then passes the result back via
        ``walked=``): keeps the speed-profiler's CPU-walk bucket truthful —
        without this the external walk lands in the launch/sync residual.
        Adds one extra ``walk`` bucket entry per call (the diagnostic
        forward-count read is approximate under this path).
        """
        bucket = _BATCHED_TIMER_HOOK
        t0 = time.perf_counter() if bucket is not None else 0.0
        walk = self._walk_obs(obs_list)
        if bucket is not None:
            bucket.setdefault("walk", []).append(time.perf_counter() - t0)
        return walk

    def merge_walked(self, walks: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge batch walk dicts (each with B ≥ 1) into one flat walk.

        Returns the same dict ``_walk_obs`` would produce on the list formed
        by concatenating each batch's obs in order. Only obs_idx is
        rewritten to the merge position (cumulative B offset); the
        remaining fields are concatenated. Used by collect to merge T
        per-step batch walks once at the end of a rollout into
        ``walk_flat``.
        """
        known = {"B", *self._WALK_PER_OBS_KEYS, *self._WALK_OBS_IDX_SLOT}
        extra = set(walks[0].keys()) - known
        assert not extra, (
            f"_walk_obs schema changed — merge_walked cannot see keys {extra}; "
            "update _WALK_PER_OBS_KEYS/_WALK_OBS_IDX_SLOT."
        )
        offs = [0]
        for w in walks:
            offs.append(offs[-1] + w["B"])
        merged: dict[str, Any] = {"B": offs[-1]}
        for k in self._WALK_PER_OBS_KEYS:
            merged[k] = [x for w in walks for x in w[k]]
        # Every type field is a numpy column (shared by both walks) — the
        # merged result must match a direct batch-walk in elements, order,
        # and dtype (tests/test_walk_cache.py parametrizes both formats).
        for k, oi in self._WALK_OBS_IDX_SLOT.items():
            n_fields = len(walks[0][k])
            fields = []
            for f in range(n_fields):
                parts = [w[k][f] for w in walks]
                assert isinstance(parts[0], np.ndarray), (
                    f"walk['{k}'][{f}] is not ndarray — walk outputs are uniformly np"
                )
                if f == oi:
                    fields.append(np.concatenate(
                        [p + off for p, off in zip(parts, offs)]))
                else:
                    fields.append(np.concatenate(parts, axis=0))
            merged[k] = tuple(fields)
        return merged

    def walk_sample_bounds(
        self, walk: dict[str, Any],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-type per-sample boundaries ``{type: (starts, ends)}`` for a flat walk.

        Sample ``i``'s entities are the ``[starts[i], ends[i])`` contiguous
        range within each type's field arrays (``_walk_obs`` appends in obs
        order → obs_idx is non-decreasing). This is the index table
        ``gather_walked`` uses — computed once at update entry.

        Two loud guards (this repo runs no ``-O``): an unknown top-level key
        means a schema change this cache would silently drop; a non-monotone
        ``obs_idx`` means ``_walk_obs`` no longer appends in obs order and the
        boundary slicing would be wrong. ``tests/test_walk_cache.py`` proves
        both the gather equivalence and that the guards fire.
        """
        known = {"B", *self._WALK_PER_OBS_KEYS, *self._WALK_OBS_IDX_SLOT}
        extra = set(walk.keys()) - known
        assert not extra, (
            f"_walk_obs schema changed — walk_sample_bounds cannot see keys "
            f"{extra}; update _WALK_PER_OBS_KEYS/_WALK_OBS_IDX_SLOT."
        )
        ar = np.arange(walk["B"])
        bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for k, oi in self._WALK_OBS_IDX_SLOT.items():
            arr = walk[k][oi]
            assert isinstance(arr, np.ndarray), (
                f"walk['{k}'] obs_idx is not ndarray — walk outputs are uniformly np"
            )
            assert arr.size == 0 or bool(np.all(np.diff(arr) >= 0)), (
                f"walk['{k}'] obs_idx is not non-decreasing — gather_walked "
                "assumes _walk_obs appends entities in obs order."
            )
            ends = np.searchsorted(arr, ar, side="right")
            starts = np.concatenate(([0], ends[:-1]))
            bounds[k] = (starts, ends)
        return bounds

    def gather_walked(
        self,
        walk: dict[str, Any],
        bounds: dict[str, tuple[np.ndarray, np.ndarray]],
        indices: list[int],
    ) -> dict[str, Any]:
        """Extract the minibatch walk for an arbitrary-order subset
        ``indices`` from the flat walk.

        Returns a dict that is **byte-identical** to
        ``_walk_obs([obs[i] for i in indices])`` — a true drop-in for
        per-minibatch re-walking on the update path. Every type field is
        numpy (shared by both walks): reassembled with one vectorized
        fancy-index, with only obs_idx rewritten to minibatch-local
        positions (0..m-1).
        """
        m = len(indices)
        idx = np.asarray(indices, dtype=np.int64)
        out: dict[str, Any] = {"B": m}
        for k in self._WALK_PER_OBS_KEYS:
            col = walk[k]
            out[k] = [col[i] for i in indices]
        for k, oi in self._WALK_OBS_IDX_SLOT.items():
            starts, ends = bounds[k]
            s, e = starts[idx], ends[idx]
            lens = e - s
            fields = walk[k]
            # Block j = flat[s[j] : s[j]+lens[j]] — build the fancy index
            # once via repeat/cumsum and reuse it across all fields.
            total = int(lens.sum())
            if total:
                cum0 = np.concatenate(([0], np.cumsum(lens)[:-1]))
                fancy = (np.repeat(s - cum0, lens)
                         + np.arange(total, dtype=np.int64))
            else:
                fancy = np.empty(0, dtype=np.int64)
            rec = []
            for f, fld in enumerate(fields):
                if f == oi:
                    # Match the direct walk's obs_idx dtype to keep byte-identity.
                    rec.append(np.repeat(np.arange(m, dtype=fld.dtype), lens))
                else:
                    rec.append(fld[fancy])
            out[k] = tuple(rec)
        return out

    # ------------------------------------------------------------------
    # Phase 2 + 3: Encode each entity type (single batched call per type).
    # Returns (embeddings, flat_indices) per type for scatter step.
    # ------------------------------------------------------------------
    def _encode_all(
        self, walk: dict[str, Any], device: torch.device, max_seq: int,
        *,
        action_type_weight: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, list[int]]]:
        """For each entity type with at least one entity, returns
        ``(emb, flat_idx, slot_ids)`` where ``emb`` is ``(K, d_model)``,
        ``flat_idx`` is ``(K,)`` int64 = ``obs_idx * max_seq + pos``, and
        ``slot_ids`` is a Python list[int] of length K (for the slot
        embedding step).
        """
        results: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []
        f32 = torch.float32

        def _t(buf: list, dtype=f32) -> torch.Tensor:
            arr = np.asarray(buf, dtype=np.float32 if dtype is f32 else np.int64)
            return torch.from_numpy(arr).to(device)

        def _flat(obs_idx: list[int], pos: list[int]) -> torch.Tensor:
            arr = np.asarray(obs_idx, dtype=np.int64) * max_seq + np.asarray(
                pos, dtype=np.int64,
            )
            return torch.from_numpy(arr).to(device)

        # ---- BOARD (1 per obs, slot=-1) ----
        bxy, bwh, bnc, b_obs, b_pos = walk["board"]
        if len(bxy):
            emb = self.vocab.encode_board(_t(bxy), _t(bwh), _t(bnc))
            results.append((emb, _flat(b_obs, b_pos), [-1] * len(bxy)))

        # ---- EDGE (slot=-1) ----
        # NOTE: every walk field is a uniform ndarray — guard with len(),
        # never bare truthiness (ambiguous for ndarrays).
        e_xy1, e_xy2, e_w, e_obs, e_pos, e_mid, e_arc = walk["edge"]
        if len(e_xy1):
            if self.vocab.legacy_edge_encoding:
                if e_arc.any():
                    raise RuntimeError(
                        "obs contains arc board-outline entries (outline_obs="
                        "'arc') but this policy uses legacy 2-point edge "
                        "tokens — re-eval with outline_obs='tess'/'poly16' or "
                        "use an arc-capable checkpoint."
                    )
                emb = self.vocab.encode_edge(_t(e_xy1), _t(e_xy2), _t(e_w))
            else:
                emb = self.vocab.encode_edge(
                    _t(e_xy1), _t(e_xy2), _t(e_w), xy_mid=_t(e_mid),
                )
            results.append((emb, _flat(e_obs, e_pos), [-1] * len(e_xy1)))

        # ---- NET ----
        n_tw, n_cl, n_vd, n_closed, n_obs, n_pos, n_slot = walk["net"]
        if len(n_tw):
            emb = self.vocab.encode_net(
                _t(n_tw), _t(n_cl), _t(n_vd), closed=_t(n_closed),
            )
            results.append((emb, _flat(n_obs, n_pos), n_slot))

        def _t_shape(buf) -> torch.Tensor | None:
            # shape_id column: int64 rows into shape_embed (DRC type_t style);
            # None when the channel is off (encoder must not consume it).
            if not self.shape_obs:
                return None
            return torch.from_numpy(np.asarray(buf, dtype=np.int64)).to(device)

        # ---- PAD ----
        (p_xy, p_wh, p_ls, p_le, p_head, p_has,
         p_obs, p_pos, p_slot, p_shape) = walk["pad"]
        if len(p_xy):
            emb = self.vocab.encode_pad(
                _t(p_xy), _t(p_wh), _t(p_ls), _t(p_le),
                head_xy=_t(p_head),
                has_head=_t(p_has).unsqueeze(-1),
                shape_id=_t_shape(p_shape),
            )
            results.append((emb, _flat(p_obs, p_pos), p_slot))

        # ---- OBSTACLE (netless blockers, slot=-1) ----
        (o_xy, o_wh, o_ls, o_le, o_head, o_has,
         o_obs, o_pos, o_shape) = walk["obstacle"]
        if len(o_xy):
            emb = self.vocab.encode_obstacle(
                _t(o_xy), _t(o_wh), _t(o_ls), _t(o_le),
                head_xy=_t(o_head),
                has_head=_t(o_has).unsqueeze(-1),
                shape_id=_t_shape(o_shape),
            )
            results.append((emb, _flat(o_obs, o_pos), [-1] * len(o_xy)))

        # ---- TRACK ----
        (t_xy1, t_xy2, t_w, t_ld, t_head, t_has,
         t_obs, t_pos, t_slot) = walk["track"]
        if len(t_xy1):
            emb = self.vocab.encode_track(
                _t(t_xy1), _t(t_xy2), _t(t_w), _t(t_ld),
                head_xy=_t(t_head),
                has_head=_t(t_has).unsqueeze(-1),
            )
            results.append((emb, _flat(t_obs, t_pos), t_slot))

        # ---- VIA ----
        (v_xy, v_ls, v_le, v_dia, v_head, v_has,
         v_obs, v_pos, v_slot) = walk["via"]
        if len(v_xy):
            emb = self.vocab.encode_via(
                _t(v_xy), _t(v_ls), _t(v_le), _t(v_dia),
                head_xy=_t(v_head),
                has_head=_t(v_has).unsqueeze(-1),
            )
            results.append((emb, _flat(v_obs, v_pos), v_slot))

        # ---- RAT ----
        r_xy, r_head, r_has, r_obs, r_pos, r_slot = walk["rat"]
        if len(r_xy):
            emb = self.vocab.encode_rat(
                _t(r_xy),
                head_xy=_t(r_head),
                has_head=_t(r_has).unsqueeze(-1),
            )
            results.append((emb, _flat(r_obs, r_pos), r_slot))

        # ---- DRC ----
        (d_xy, d_ld, d_type, d_sev, d_head, d_has,
         d_obs, d_pos, d_slot) = walk["drc"]
        if len(d_xy):
            type_t = torch.from_numpy(
                np.asarray(d_type, dtype=np.int64),
            ).to(device)
            emb = self.vocab.encode_drc(
                _t(d_xy), _t(d_ld), type_t,
                severity_flag=_t(d_sev).unsqueeze(-1),
                head_xy=_t(d_head),
                has_head=_t(d_has).unsqueeze(-1),
            )
            results.append((emb, _flat(d_obs, d_pos), d_slot))

        # ---- HEAD ----
        (h_xy, h_ld, h_rm, h_np, h_sr,
         h_obs, h_pos, h_slot) = walk["head"]
        if len(h_xy):
            rm_t = torch.from_numpy(
                np.asarray(h_rm, dtype=np.int64),
            ).to(device)
            np_t = torch.from_numpy(
                np.asarray(h_np, dtype=np.int64),
            ).to(device)
            emb = self.vocab.encode_head(
                _t(h_xy), _t(h_ld), rm_t, np_t, _t(h_sr),
            )
            results.append((emb, _flat(h_obs, h_pos), h_slot))

        # ---- CAND ----
        (c_type, c_xy, c_ld, c_head, c_has,
         c_obs, c_pos, c_slot) = walk["cand"]
        if len(c_type):
            type_t = torch.from_numpy(
                np.asarray(c_type, dtype=np.int64),
            ).to(device)
            emb = self.vocab.encode_cand(
                type_t, _t(c_xy), _t(c_ld),
                head_xy=_t(c_head),
                has_head=_t(c_has).unsqueeze(-1),
            )
            results.append((emb, _flat(c_obs, c_pos), c_slot))

        # ---- ACTION_HISTORY (3 tokens per entry, K entries per obs) ----
        # Only emitted when the policy threads its action_type_head.weight
        # through forward() for weight tying. Without it the positions are
        # left as zeros (pre-LN), which the transformer treats as noise.
        # Each entry's 3 tokens share the entry's net slot (SameNetBias /
        # slot embedding); legacy prev-action mode emits slot=-1 throughout.
        (pa_type, pa_succ, pa_xy, pa_ld, pa_has_ptr, pa_mode, pa_age,
         pa_slot, pa_obs, pa_at_p, pa_pt_p, pa_mo_p) = walk["action_history"]
        if len(pa_type) and action_type_weight is not None:
            type_t = torch.from_numpy(
                np.asarray(pa_type, dtype=np.int64),
            ).to(device)
            mode_t = torch.from_numpy(
                np.asarray(pa_mode, dtype=np.int64),
            ).to(device)
            succ_t = torch.from_numpy(
                np.asarray(pa_succ, dtype=np.float32),
            ).to(device)
            xy_t = torch.from_numpy(
                np.asarray(pa_xy, dtype=np.float32),
            ).to(device)
            ld_t = torch.from_numpy(
                np.asarray(pa_ld, dtype=np.float32),
            ).to(device)
            has_t = torch.from_numpy(
                np.asarray(pa_has_ptr, dtype=np.float32),
            ).to(device)
            age_t = torch.from_numpy(
                np.asarray(pa_age, dtype=np.int64),
            ).to(device)
            # (N, 3, d) -> split into 3 flat emb streams (N = B*K entries).
            pa_emb = self.vocab.encode_action_history(
                type_t, succ_t, xy_t, ld_t, has_t, mode_t, age_t,
                action_type_weight,
            )
            # Scatter each of the 3 token slots separately.
            at_emb = pa_emb[:, 0, :]; pt_emb = pa_emb[:, 1, :]; mo_emb = pa_emb[:, 2, :]
            pa_slot_l = pa_slot.tolist()
            results.append((at_emb, _flat(pa_obs, pa_at_p), pa_slot_l))
            results.append((pt_emb, _flat(pa_obs, pa_pt_p), pa_slot_l))
            results.append((mo_emb, _flat(pa_obs, pa_mo_p), pa_slot_l))

        # ---- VAL + SOD (structural, slot=-1) ----
        for tok_id, key, slot_default in [
            (int(ST.VAL), "val", -1),
            (int(ST.SOD), "sod", -1),
        ]:
            obs_idx, posv = walk[key]
            if not len(obs_idx):
                continue
            ids = torch.full(
                (len(obs_idx),), tok_id, dtype=torch.long, device=device,
            )
            emb = self.vocab.embed_structural(ids)
            results.append((emb, _flat(obs_idx, posv), [slot_default] * len(obs_idx)))

        return results

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self, obs_list: list[dict], *,
        action_type_weight: torch.Tensor | None = None,
        walked: dict[str, Any] | None = None,
    ) -> TokenizerOutput:
        """When ``walked`` is given, skips Phase 1 (the CPU walk) and uses
        that walk dict instead (the update path's walk cache — built by
        ``gather_walked``/``merge_walked``; ``obs_list`` is ignored in that
        case)."""
        device = next(self.parameters()).device
        d_model = self.d_model
        bucket = _BATCHED_TIMER_HOOK

        # Phase 1
        t0 = time.perf_counter() if bucket is not None else 0.0
        walk = self._walk_obs(obs_list) if walked is None else walked
        if bucket is not None:
            bucket.setdefault("walk", []).append(time.perf_counter() - t0)
        B = walk["B"]
        seq_lens_py: list[int] = walk["seq_lens"]
        net_positions: list[list[int]] = walk["net_positions"]
        cand_positions: list[list[int]] = walk["cand_positions"]
        cand_mm_list: list[list[tuple[float, float, int]]] = walk["cand_mm_list"]
        slot_perm_per_obs: list[list[int] | None] = walk["slot_perm_per_obs"]

        max_seq = min(max(seq_lens_py) if seq_lens_py else 0, self.max_seq_len)

        # Phase 2 + 3
        t0 = time.perf_counter() if bucket is not None else 0.0
        encoded = self._encode_all(
            walk, device, max_seq, action_type_weight=action_type_weight,
        )
        if bucket is not None:
            bucket.setdefault("h2d_encode", []).append(
                time.perf_counter() - t0,
            )

        # Phase 4: scatter into padded output via index_copy_
        t0 = time.perf_counter() if bucket is not None else 0.0
        out = torch.zeros(B, max_seq, d_model, device=device)
        flat_out = out.view(B * max_seq, d_model)

        # SEQ_PAD embedding for positions beyond seq_lens[b].
        pad_emb = self.vocab.embed_structural(
            torch.tensor(int(ST.SEQ_PAD), device=device),
        )

        # Build flat slot_ids tensor (B, max_seq) initialized to -1.
        slot_ids_flat = torch.full(
            (B * max_seq,), -1, dtype=torch.long, device=device,
        )

        # Positions are always < max_seq by construction (walk pos < seq_len,
        # max_seq = min(max(seq_lens), max_seq_len)); the only violation is an
        # obs exceeding the cap, which flat-index truncation cannot express
        # (it would silently write into the NEXT row's range) — fail loudly
        # instead.
        if seq_lens_py and max(seq_lens_py) > max_seq:
            raise ValueError(
                f"seq_len {max(seq_lens_py)} exceeds max_seq_len "
                f"{self.max_seq_len} — cannot scatter without corrupting "
                f"neighbouring rows"
            )
        for emb, flat_idx, slot_ids in encoded:
            flat_out.index_copy_(0, flat_idx, emb)
            if len(slot_ids):
                slot_t = torch.tensor(
                    slot_ids, dtype=torch.long, device=device,
                )
                slot_ids_flat.index_copy_(0, flat_idx, slot_t)

        # Build key_padding_mask. (Padded positions are filled AFTER
        # LayerNorm: a padded slot holds the raw pad_embed, with no
        # LayerNorm and no slot contribution.)
        seq_lens = torch.tensor(seq_lens_py, dtype=torch.long, device=device)
        seq_lens = seq_lens.clamp(max=max_seq)
        positions = torch.arange(max_seq, device=device).unsqueeze(0)  # (1, S)
        key_padding_mask = positions >= seq_lens.unsqueeze(-1)         # (B, S)

        slot_ids_t = slot_ids_flat.view(B, max_seq)
        if bucket is not None:
            bucket.setdefault("scatter", []).append(time.perf_counter() - t0)

        # Phase 5: slot embedding + LayerNorm
        t0 = time.perf_counter() if bucket is not None else 0.0
        # Apply per-obs slot_perm: for obs b with perm, remap slot_ids_t[b].
        any_perm = any(p is not None for p in slot_perm_per_obs)
        if any_perm:
            # Build (B, n_max_slots) gather table; rows without perm are
            # identity. Per-obs slot_perm may be shorter than
            # n_max_slots (one entry per net) — write only over the
            # length the user provided; remaining entries stay identity
            # and are never gathered (no slot_id >= len(perm) exists).
            n_slots = self.vocab.n_max_slots
            perm_table = torch.arange(
                n_slots, dtype=torch.long, device=device,
            ).unsqueeze(0).expand(B, n_slots).clone()
            for b, perm in enumerate(slot_perm_per_obs):
                if perm is not None:
                    L = len(perm)
                    perm_table[b, :L] = torch.as_tensor(
                        perm, dtype=torch.long, device=device,
                    )
            valid = slot_ids_t >= 0
            safe = slot_ids_t.clamp(min=0)
            # gather: remapped[b, s] = perm_table[b, safe[b, s]]
            remapped = torch.gather(perm_table, 1, safe)
            slot_ids_t = torch.where(valid, remapped, slot_ids_t)

        if self.vocab.disable_slot_emb:
            out = self.vocab.embed_ln(out)
        else:
            valid_mask = (slot_ids_t >= 0).unsqueeze(-1).to(out.dtype)
            safe_ids = slot_ids_t.clamp(min=0)
            slot_contrib = self.vocab.slot_emb_table[safe_ids] * valid_mask
            out = self.vocab.embed_ln(
                out + self.vocab.slot_scale * slot_contrib,
            )

        # Fill padded positions with raw pad_emb AFTER LayerNorm — these
        # positions stay un-normalized.
        if seq_lens_py and min(seq_lens_py) < max_seq:  # == key_padding_mask.any(), no sync
            out = out.masked_scatter(
                key_padding_mask.unsqueeze(-1),
                pad_emb.expand_as(out)[key_padding_mask],
            )
        if bucket is not None:
            bucket.setdefault("slot_emb", []).append(time.perf_counter() - t0)

        # Build pointer index tensors (B, max_*) padded with -1.
        max_nets = max((len(p) for p in net_positions), default=0)
        max_cands = max((len(p) for p in cand_positions), default=0)

        def _idx_tensor(pos_lists: list[list[int]], width: int) -> torch.Tensor:
            # Fill the (B, width) int64 array on CPU for one H2D copy,
            # avoiding a per-element CUDA assignment (one kernel launch per
            # element). p >= max_seq stays -1 (same meaning).
            arr = np.full((B, width), -1, dtype=np.int64)
            for b, plist in enumerate(pos_lists):
                if plist:
                    row = np.asarray(plist, dtype=np.int64)
                    arr[b, :len(row)] = np.where(row < max_seq, row, -1)
            return torch.from_numpy(arr).to(device)

        net_idx = _idx_tensor(net_positions, max_nets)
        cand_idx = _idx_tensor(cand_positions, max_cands)

        return TokenizerOutput(
            token_embeddings=out,
            net_indices=net_idx,
            cand_indices=cand_idx,
            key_padding_mask=key_padding_mask,
            seq_lens=seq_lens,
            cand_mm_list=cand_mm_list,
            slot_ids=slot_ids_t,
        )
