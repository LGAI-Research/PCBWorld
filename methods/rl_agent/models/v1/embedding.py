"""Token vocabulary for the Decoder-Only PCB routing policy.

Single-token-per-entity design:

  * Each entity (pad, track, via, head, candidate, ratsnest, net, edge,
    board) collapses into ONE token whose embedding is::

        token = entity_type_embed[type] + entity_proj(concat([features]))

  * Track tokens use symmetric endpoint pooling so a swap of the two
    endpoints yields the same embedding (Deep-Sets style).
  * Geometric tokens (pad, track, via, cand, rat) carry a head-relative
    distance feature so the policy gets that inductive bias for free.
  * Slot embedding stays as a fixed orthogonal table; magnitude is now
    governed by a learnable ``slot_scale`` parameter (init 0.3).
  * :class:`StructuralToken` keeps only ``VAL``, ``SOD``, ``SEQ_PAD``;
    every entity type is enumerated by :class:`EntityType`.

All ``encode_*`` methods accept a leading batch dimension ``K`` (the number
of entities of the same type in a single observation) and return
``(K, d_model)``. ``head_xy`` is per-observation, shared across the batch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Shape / vocab contract relocated to the v1 spec (C1-models step 1). Re-exported
# here so existing ``from ...token_vocabulary import EntityType`` etc. keep working.
from methods.rl_agent.models.v1.spec import (  # noqa: F401
    MAX_COPPER,
    MAX_HISTORY,
    N_MAX_SLOTS,
    StructuralToken,
    NUM_STRUCTURAL_TOKENS,
    EntityType,
    NUM_ENTITY_TYPES,
    NUM_DRC_TYPES,
    NUM_SHAPE_BUCKETS,
    CandidateType,
    NUM_CAND_TYPES,
    cand_type_to_entity,
)


# encode_layer (stateless geometry helper) relocated to the codec
# (``models.v1.encoding``, C1-models step 3). Re-imported so this module's
# internal references + the token_vocabulary shim keep resolving it.
from methods.rl_agent.models.v1.encoding import encode_layer  # noqa: F401


# ---------------------------------------------------------------------------
# Fourier Encoding
# ---------------------------------------------------------------------------
class FourierEncoding(nn.Module):
    """Sin / cos multi-frequency encoding for continuous values.

    ``(*, D) → (*, D × 2 × n_freq)``
    """

    def __init__(self, n_freq: int = 6, base: float = 2.0) -> None:
        super().__init__()
        self.n_freq = n_freq
        freqs = base ** torch.arange(n_freq, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x.unsqueeze(-1) * self.freqs * math.pi
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return encoded.flatten(-2)

    def output_dim(self, input_dim: int) -> int:
        return input_dim * 2 * self.n_freq


# ---------------------------------------------------------------------------
# Token Vocabulary
# ---------------------------------------------------------------------------
class TokenVocabulary(nn.Module):
    """Embedding vocabulary for the single-token-per-entity tokenizer.

    All ``encode_*`` methods take a leading batch dim ``K`` (entities of
    the same type in one observation) and return ``(K, d_model)``. They
    accept ``head_xy`` as a per-observation shared ``(2,)`` tensor (or
    ``None`` to indicate "no head"). The Linear projections (``pad_proj``,
    ``track_proj`` ... ) own all the learnable feature-fusion weights.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_freq: int = 32,
        coord_encoding: str = "fourier",
        mlp_hidden: int = 128,
        fourier_base: float = 1.20,
        n_max_slots: int = N_MAX_SLOTS,
        disable_slot_emb: bool = False,
        slot_scale_init: float = 0.3,
        legacy_pad_layer_encoding: bool = False,
        legacy_net_encoding: bool = False,
        legacy_edge_encoding: bool = False,
        time_fourier_base: float | None = None,
        action_history_len: int = 1,
        legacy_action_history: bool = False,
        obstacle_obs: bool = False,
        shape_obs: bool = False,
    ) -> None:
        super().__init__()
        if coord_encoding not in ("fourier", "mlp", "linear"):
            raise ValueError(
                f"coord_encoding must be 'fourier', 'mlp', or 'linear', got {coord_encoding!r}"
            )
        if not (1 <= action_history_len <= MAX_HISTORY):
            raise ValueError(
                f"action_history_len must be in [1, {MAX_HISTORY}], "
                f"got {action_history_len}"
            )
        if legacy_action_history and action_history_len != 1:
            raise ValueError(
                "legacy_action_history (old prev-action checkpoints) implies "
                f"action_history_len=1, got {action_history_len}"
            )
        self.d_model = d_model
        self.n_freq = n_freq
        self.coord_encoding = coord_encoding
        self.n_max_slots = n_max_slots
        self.disable_slot_emb = disable_slot_emb
        self.legacy_pad_layer_encoding = bool(legacy_pad_layer_encoding)
        self.legacy_net_encoding = bool(legacy_net_encoding)
        self.legacy_edge_encoding = bool(legacy_edge_encoding)
        self.action_history_len = int(action_history_len)
        self.legacy_action_history = bool(legacy_action_history)
        self.obstacle_obs = bool(obstacle_obs)
        self.shape_obs = bool(shape_obs)

        # --- Type embeddings ---
        self.entity_type_embed = nn.Embedding(NUM_ENTITY_TYPES, d_model)
        self.structural_embed = nn.Embedding(NUM_STRUCTURAL_TOKENS, d_model)

        # --- Slot embedding ---
        slot_table = torch.empty(n_max_slots, d_model)
        nn.init.orthogonal_(slot_table)
        self.register_buffer("slot_emb_table", slot_table)
        self.slot_scale = nn.Parameter(torch.tensor(float(slot_scale_init)))
        self.embed_ln = nn.LayerNorm(d_model)

        # --- Coord encoders ---
        if coord_encoding == "fourier":
            self.fourier_2d = FourierEncoding(n_freq=n_freq, base=fourier_base)
            base_1d = fourier_base ** ((n_freq - 1) / (2 * n_freq - 1))
            self.fourier_1d = FourierEncoding(n_freq=n_freq * 2, base=base_1d)
            f = self.fourier_2d.output_dim(2)
            assert self.fourier_1d.output_dim(1) == f
        elif coord_encoding == "linear":
            # Bare nn.Linear with no activation; output dim == d_model so
            # the downstream *_proj shapes match the "mlp" branch.
            self.lin_xy = nn.Linear(2, d_model)
            self.lin_1d = nn.Linear(1, d_model)
            self.lin_2d = nn.Linear(2, d_model)
            f = d_model
        else:
            def _mlp(in_dim: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(in_dim, mlp_hidden),
                    nn.GELU(),
                    nn.Linear(mlp_hidden, d_model),
                )
            self.mlp_xy = _mlp(2)
            self.mlp_1d = _mlp(1)
            self.mlp_2d = _mlp(2)
            f = d_model
        self.fenc_dim = f

        # Dedicated ladder for the HEAD time scalar (time_feature
        # "sin_remaining"): same sin/cos family and output width f as
        # fourier_1d, but frequencies anchored to step units — with
        # u = remaining/cap and base = cap^(1/(2·n_freq−1)) the top rung
        # is sin(remaining·π), period 2 steps, so ±1 step resolves at any
        # horizon. Non-persistent buffer (and zero new weights) keeps the
        # state_dict identical across time_feature modes. Only consulted
        # in "fourier" coord mode; None ⇒ legacy path via fourier_1d.
        time_freqs = None
        if time_fourier_base is not None:
            time_freqs = float(time_fourier_base) ** torch.arange(
                n_freq * 2, dtype=torch.float32,
            )
        self.register_buffer("time_freqs", time_freqs, persistent=False)

        # Routing-mode embedding kept for action-side weight tying.
        self.routing_mode_embed = nn.Embedding(3, d_model)
        self.net_phase_embed = nn.Embedding(3, d_model)

        # --- Per-entity feature-fusion projections ---
        # head-rel extras (d_min_enc + d_avg_enc + has_head_mask)
        head_rel_extra = 2 * f + 1
        # Old v51 checkpoints encoded pads with one layer-distance pair:
        #   xy + wh + layer + head_rel = 3f + (2f + 1)
        # Latest checkpoints encode a layer span for thru-hole pads:
        #   xy + wh + layer_start + layer_end + head_rel = 4f + (2f + 1)
        pad_layer_terms = 1 if self.legacy_pad_layer_encoding else 2
        # Old checkpoints encode nets with the 3 static netclass constraints:
        #   track_width + clearance + via_diameter = 3f
        # Latest checkpoints add the per-episode closed flag (net consumed by
        # net_end; obs["closed_nets"]):
        #   track_width + clearance + via_diameter + closed = 4f
        net_terms = 3 if self.legacy_net_encoding else 4
        self.pad_proj = nn.Linear((2 + pad_layer_terms) * f + head_rel_extra, d_model)
        self.via_proj = nn.Linear(4 * f + head_rel_extra, d_model)
        self.endpoint_proj = nn.Linear(f, d_model)
        self.track_proj = nn.Linear(d_model + 2 * f + head_rel_extra, d_model)
        self.edge_proj = nn.Linear(d_model + f, d_model)
        # Old checkpoints encode edges from the 2 endpoints only; latest add a
        # 3rd on-path midpoint through its OWN projection, summed into the
        # endpoint pool: straight edge -> chord midpoint, board-outline arc ->
        # the on-arc midpoint (KiCad 3-point form). The separate projection is
        # what keeps the mid role identifiable — running mid through the shared
        # endpoint_proj plus an additive role vector would commute back into a
        # fully 3-point-symmetric sum and lose which point bulges. edge_proj
        # shape is unchanged, so checkpoint detection keys on this module's
        # presence, not a shape.
        if not self.legacy_edge_encoding:
            self.edge_mid_proj = nn.Linear(f, d_model)
        self.rat_proj = nn.Linear(f + head_rel_extra, d_model)
        self.head_proj = nn.Linear(3 * f + 2 * d_model, d_model)
        self.cand_proj = nn.Linear(3 * f + 1, d_model)
        self.net_proj = nn.Linear(net_terms * f, d_model)
        self.board_proj = nn.Linear(3 * f, d_model)

        # --- DRC violation token ---
        # 7-bucket taxonomy; features = xy(f) + layer(f) + type_embed(d_model)
        #                            + head_rel(2f+1) + severity_flag(1)
        self.drc_type_embed = nn.Embedding(NUM_DRC_TYPES, d_model)
        self.drc_proj = nn.Linear(
            2 * f + d_model + head_rel_extra + 1, d_model,
        )

        # --- Boundary-shape channel (shape_obs knob) ---
        # Additive categorical embedding on PAD and OBSTACLE tokens:
        #   token = type_vec + pad_proj(feats) + shape_embed(shape_id)
        # Additive (not concat) so pad_proj keeps its width — the loader's
        # pad-width legacy detection stays valid and knob-off state_dicts are
        # unchanged. Created only when on: checkpoint detection keys on this
        # module's presence (the edge_mid_proj / history_age_proj pattern).
        # OBSTACLE tokens reuse pad_proj (endpoint_proj-style tying): the
        # obstacle_obs knob adds no weights beyond the EntityType row.
        if self.shape_obs:
            self.shape_embed = nn.Embedding(NUM_SHAPE_BUCKETS, d_model)

        # --- Action-history tokens (3 per entry, K entries per obs) ---
        # Weight-tied with the policy's action_type_head (passed in at
        # forward time) and our routing_mode_embed. Only introduces:
        #   * 3-row slot embedding distinguishing at/pt/mode positions
        #     (shared across entries — the "this is a history token" marker),
        #   * scalar projection for the success flag (adds to at-token),
        #   * pt-feature projection (xy fourier + layer dist + has_ptr),
        #   * age projection (below) — non-legacy only.
        self.prev_action_slot_emb = nn.Parameter(
            torch.randn(3, d_model) * 0.02,
        )
        self.prev_action_success_proj = nn.Linear(1, d_model)
        # xy_enc (f) + layer dist (2) + has_ptr flag (1)
        self.prev_action_pt_proj = nn.Linear(f + 2 + 1, d_model)
        # Age (recency) marker, added to all 3 tokens of entry k: raw age k
        # normalized by the frozen MAX_HISTORY denominator, then a small
        # Fourier ladder + projection (the encode_layer / pad-layer pattern).
        # Deliberately K-shape-free so action_history_len stays a config knob
        # (checkpoints carry no K-sized weights). Absent on legacy prev-action
        # checkpoints — the loader keys legacy detection on this module.
        if not self.legacy_action_history:
            self.history_age_fourier = FourierEncoding(n_freq=6, base=2.0)
            self.history_age_proj = nn.Linear(
                self.history_age_fourier.output_dim(1), d_model,
            )

    # ------------------------------------------------------------------
    # Coordinate encoders (always batched: (K, D) -> (K, fenc_dim))
    # ------------------------------------------------------------------
    def _enc_xy(self, xy: torch.Tensor) -> torch.Tensor:
        if self.coord_encoding == "fourier":
            return self.fourier_2d(xy)
        if self.coord_encoding == "linear":
            return self.lin_xy(xy)
        return self.mlp_xy(xy)

    def _enc_2d(self, vals: torch.Tensor) -> torch.Tensor:
        if self.coord_encoding == "fourier":
            return self.fourier_2d(vals)
        if self.coord_encoding == "linear":
            return self.lin_2d(vals)
        return self.mlp_2d(vals)

    def _enc_1d(self, vals: torch.Tensor) -> torch.Tensor:
        if self.coord_encoding == "fourier":
            return self.fourier_1d(vals)
        if self.coord_encoding == "linear":
            return self.lin_1d(vals)
        return self.mlp_1d(vals)

    def _enc_time(self, vals: torch.Tensor) -> torch.Tensor:
        """HEAD time scalar: dedicated ladder when configured, else _enc_1d."""
        if self.time_freqs is not None and self.coord_encoding == "fourier":
            scaled = vals.unsqueeze(-1) * self.time_freqs * math.pi
            return torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(-2)
        return self._enc_1d(vals)

    # ------------------------------------------------------------------
    # Head-relative features for a batch of entities.
    #
    # Three accepted shapes for ``head_xy`` (legacy + batched):
    #   * ``None``                     → all entries treated as has_head=0
    #   * ``(2,)`` or ``(1, 2)``       → broadcast across K entities; has_head=1
    #   * ``(K, 2)`` per-entity        → caller MUST also pass ``has_head``
    #
    # ``has_head`` is a ``(K, 1)`` mask in {0., 1.} — 1 where the entity's
    # observation has an active routing head, else 0. The distance encoding
    # is multiplied by the mask so masked rows produce all-zero output,
    # matching the legacy ``head_xy is None`` branch exactly.
    # ------------------------------------------------------------------
    def _normalize_head_input(
        self,
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None,
        K: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if head_xy is None:
            return (
                torch.zeros(K, 2, device=device),
                torch.zeros(K, 1, device=device),
            )
        if head_xy.dim() == 1:
            head_xy_K2 = head_xy.unsqueeze(0).expand(K, 2)
        elif head_xy.dim() == 2 and head_xy.size(0) == 1:
            head_xy_K2 = head_xy.expand(K, 2)
        elif head_xy.dim() == 2 and head_xy.size(0) == K:
            head_xy_K2 = head_xy
        else:
            raise ValueError(
                f"head_xy must be None, (2,), (1, 2), or (K, 2); "
                f"got shape {tuple(head_xy.shape)} with K={K}",
            )
        if has_head is None:
            has_head = torch.ones(K, 1, device=device)
        return head_xy_K2, has_head

    def _head_rel_pointwise(
        self,
        xy: torch.Tensor,           # (K, 2)
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(d_min, d_avg, has_head)`` for entities with one xy each.

        For 1-point entities ``d_min == d_avg == ||xy - head||``.
        Returns ``(K, 2*f + 1)``.
        """
        K = xy.size(0)
        head_xy_K2, has = self._normalize_head_input(
            head_xy, has_head, K, xy.device,
        )
        diff = xy - head_xy_K2                        # (K, 2)
        d = diff.norm(dim=-1, keepdim=True)           # (K, 1)
        d_enc = self._enc_1d(d) * has                 # (K, fenc_dim), masked
        return torch.cat([d_enc, d_enc, has], dim=-1)

    def _head_rel_track(
        self,
        xy1: torch.Tensor,          # (K, 2)
        xy2: torch.Tensor,          # (K, 2)
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(d_min, d_avg, has_head)`` for two-endpoint tracks."""
        K = xy1.size(0)
        head_xy_K2, has = self._normalize_head_input(
            head_xy, has_head, K, xy1.device,
        )
        d1 = (xy1 - head_xy_K2).norm(dim=-1)          # (K,)
        d2 = (xy2 - head_xy_K2).norm(dim=-1)
        d_min = torch.minimum(d1, d2).unsqueeze(-1)
        d_avg = ((d1 + d2) * 0.5).unsqueeze(-1)
        d_min_enc = self._enc_1d(d_min) * has
        d_avg_enc = self._enc_1d(d_avg) * has
        return torch.cat([d_min_enc, d_avg_enc, has], dim=-1)

    # ------------------------------------------------------------------
    # Entity-type vector helper (broadcast K times).
    # ------------------------------------------------------------------
    def _type_vec(self, kind: EntityType, K: int, device) -> torch.Tensor:
        idx = torch.full((K,), int(kind), dtype=torch.long, device=device)
        return self.entity_type_embed(idx)

    # ------------------------------------------------------------------
    # Per-entity encoders
    # ------------------------------------------------------------------
    def _rect_token(
        self,
        entity_type: EntityType,
        xy: torch.Tensor,
        wh: torch.Tensor,
        layer_start_dt_db: torch.Tensor,
        layer_end_dt_db: torch.Tensor,
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None,
        shape_id: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shared pad-shaped token body (PAD and OBSTACLE — pad_proj tied)."""
        K = xy.size(0)
        layer_feats = [self._enc_2d(layer_start_dt_db)]
        if not self.legacy_pad_layer_encoding:
            layer_feats.append(self._enc_2d(layer_end_dt_db))
        feats = torch.cat([
            self._enc_xy(xy),
            self._enc_2d(wh),
            *layer_feats,
            self._head_rel_pointwise(xy, head_xy, has_head),
        ], dim=-1)
        tok = self._type_vec(entity_type, K, xy.device) + self.pad_proj(feats)
        if self.shape_obs:
            if shape_id is None:
                raise ValueError(
                    f"shape_obs is on but no shape_id column reached "
                    f"encode_{entity_type.name.lower()} — walk/encoder drift"
                )
            tok = tok + self.shape_embed(shape_id)
        return tok

    def encode_pad(
        self,
        xy: torch.Tensor,                # (K, 2)
        wh: torch.Tensor,                # (K, 2)
        layer_start_dt_db: torch.Tensor, # (K, 2)
        layer_end_dt_db: torch.Tensor,   # (K, 2)
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
        shape_id: torch.Tensor | None = None,  # (K,) int64, shape_obs only
    ) -> torch.Tensor:
        """Pad encoder. SMD/connect pads pass ``layer_start == layer_end``;
        thru-hole pads pass ``(layer_start=1, layer_end=n_copper)`` so the
        feature shape matches a thru via barrel — pads and vias share one
        copper-layer-span primitive.
        """
        return self._rect_token(
            EntityType.PAD, xy, wh, layer_start_dt_db, layer_end_dt_db,
            head_xy, has_head, shape_id,
        )

    def encode_obstacle(
        self,
        xy: torch.Tensor,                # (K, 2)
        wh: torch.Tensor,                # (K, 2)
        layer_start_dt_db: torch.Tensor, # (K, 2)
        layer_end_dt_db: torch.Tensor,   # (K, 2)
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
        shape_id: torch.Tensor | None = None,  # (K,) int64, shape_obs only
    ) -> torch.Tensor:
        """Netless immovable blocker (NPTH hole / slot / NC pad): identical
        geometry channels to a pad through the tied ``pad_proj``; identity
        comes from the OBSTACLE type row (and the absent net slot).
        """
        return self._rect_token(
            EntityType.OBSTACLE, xy, wh, layer_start_dt_db, layer_end_dt_db,
            head_xy, has_head, shape_id,
        )

    def encode_via(
        self,
        xy: torch.Tensor,
        layer_start_dt_db: torch.Tensor,
        layer_end_dt_db: torch.Tensor,
        via_dia: torch.Tensor,     # (K, 1)
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        K = xy.size(0)
        feats = torch.cat([
            self._enc_xy(xy),
            self._enc_2d(layer_start_dt_db),
            self._enc_2d(layer_end_dt_db),
            self._enc_1d(via_dia),
            self._head_rel_pointwise(xy, head_xy, has_head),
        ], dim=-1)
        return self._type_vec(EntityType.VIA, K, xy.device) + self.via_proj(feats)

    def encode_track(
        self,
        xy1: torch.Tensor,         # (K, 2)
        xy2: torch.Tensor,         # (K, 2)
        width: torch.Tensor,       # (K, 1)
        layer_dt_db: torch.Tensor, # (K, 2)
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Symmetric endpoint pooling: encode_track(xy1, xy2, ...) ==
        encode_track(xy2, xy1, ...).
        """
        K = xy1.size(0)
        e1 = self.endpoint_proj(self._enc_xy(xy1))  # (K, d_model)
        e2 = self.endpoint_proj(self._enc_xy(xy2))
        endpoint_sym = e1 + e2                       # symmetric
        feats = torch.cat([
            endpoint_sym,
            self._enc_1d(width),
            self._enc_2d(layer_dt_db),
            self._head_rel_track(xy1, xy2, head_xy, has_head),
        ], dim=-1)
        return self._type_vec(EntityType.TRACK, K, xy1.device) + self.track_proj(feats)

    def encode_edge(
        self,
        xy1: torch.Tensor,
        xy2: torch.Tensor,
        width: torch.Tensor,       # (K, 1)
        xy_mid: torch.Tensor | None = None,  # (K, 2); required unless legacy
    ) -> torch.Tensor:
        """Endpoint-symmetric: encode_edge(xy1, xy2, w, m) == encode_edge(xy2,
        xy1, w, m) — an arc through (p1, mid, p2) equals its reversal. The mid
        point rides its own edge_mid_proj (not endpoint_proj), so which point
        bulges stays identifiable in the pooled sum.
        """
        K = xy1.size(0)
        e1 = self.endpoint_proj(self._enc_xy(xy1))
        e2 = self.endpoint_proj(self._enc_xy(xy2))
        endpoint_sym = e1 + e2
        if self.legacy_edge_encoding:
            if xy_mid is not None:
                raise RuntimeError(
                    "legacy_edge_encoding policy got an edge midpoint: this "
                    "checkpoint predates 3-point edge tokens and cannot "
                    "represent arc outlines — eval it with outline_obs="
                    "'tess'/'poly16', not 'arc'."
                )
        else:
            if xy_mid is None:
                raise RuntimeError(
                    "encode_edge requires xy_mid (chord midpoint for straight "
                    "edges) unless legacy_edge_encoding is set"
                )
            endpoint_sym = endpoint_sym + self.edge_mid_proj(self._enc_xy(xy_mid))
        feats = torch.cat([endpoint_sym, self._enc_1d(width)], dim=-1)
        return self._type_vec(EntityType.EDGE, K, xy1.device) + self.edge_proj(feats)

    def encode_rat(
        self,
        xy: torch.Tensor,
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Ratsnest Q-points carry NO layer feature: upstream RatsnestEdge has
        # no per-endpoint layer, and the prior layer=1 fallback injected a
        # fake constant signal. Layer is recovered via co-located pad tokens.
        K = xy.size(0)
        feats = torch.cat([
            self._enc_xy(xy),
            self._head_rel_pointwise(xy, head_xy, has_head),
        ], dim=-1)
        return self._type_vec(EntityType.RAT, K, xy.device) + self.rat_proj(feats)

    def encode_head(
        self,
        xy: torch.Tensor,            # (1, 2)
        layer_dt_db: torch.Tensor,   # (1, 2)
        routing_mode: torch.Tensor,  # (1,) long
        net_phase: torch.Tensor,     # (1,) long
        step_ratio: torch.Tensor,    # (1, 1)
    ) -> torch.Tensor:
        K = xy.size(0)
        rm = self.routing_mode_embed(routing_mode)
        np_ = self.net_phase_embed(net_phase)
        feats = torch.cat([
            self._enc_xy(xy),
            self._enc_2d(layer_dt_db),
            self._enc_time(step_ratio),
            rm,
            np_,
        ], dim=-1)
        return self._type_vec(EntityType.HEAD, K, xy.device) + self.head_proj(feats)

    def encode_cand(
        self,
        cand_type_ints: torch.Tensor,   # (K,) int64 — CandidateType values
        xy: torch.Tensor,
        layer_dt_db: torch.Tensor,
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        K = xy.size(0)
        device = xy.device
        head_xy_K2, has = self._normalize_head_input(
            head_xy, has_head, K, device,
        )
        d = (xy - head_xy_K2).norm(dim=-1, keepdim=True)  # (K, 1)
        d_enc = self._enc_1d(d) * has                     # masked
        feats = torch.cat([
            self._enc_xy(xy),
            self._enc_2d(layer_dt_db),
            d_enc,
            has,
        ], dim=-1)
        proj = self.cand_proj(feats)
        # Per-row entity-type lookup.
        cand_entity_ids = torch.tensor(
            [cand_type_to_entity(int(c)) for c in cand_type_ints.tolist()],
            dtype=torch.long, device=device,
        )
        return self.entity_type_embed(cand_entity_ids) + proj

    def encode_net(
        self,
        track_w: torch.Tensor,    # (K, 1)
        clearance: torch.Tensor,  # (K, 1)
        via_dia: torch.Tensor,    # (K, 1)
        closed: torch.Tensor | None = None,  # (K, 1) float in {0., 1.}
    ) -> torch.Tensor:
        K = track_w.size(0)
        parts = [
            self._enc_1d(track_w),
            self._enc_1d(clearance),
            self._enc_1d(via_dia),
        ]
        if not self.legacy_net_encoding:
            if closed is None:
                raise ValueError(
                    "encode_net requires the per-net `closed` flag "
                    "(legacy_net_encoding=False)."
                )
            parts.append(self._enc_1d(closed))
        feats = torch.cat(parts, dim=-1)
        return self._type_vec(EntityType.NET, K, track_w.device) + self.net_proj(feats)

    def encode_drc(
        self,
        xy: torch.Tensor,              # (K, 2)
        layer_dt_db: torch.Tensor,     # (K, 2)
        type_id: torch.Tensor,         # (K,) long, 0..NUM_DRC_TYPES-1
        severity_flag: torch.Tensor,   # (K, 1) float in {0., 1.}  (1 = error)
        head_xy: torch.Tensor | None,
        has_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        K = xy.size(0)
        type_embed = self.drc_type_embed(type_id)          # (K, d_model)
        feats = torch.cat([
            self._enc_xy(xy),
            self._enc_2d(layer_dt_db),
            type_embed,
            self._head_rel_pointwise(xy, head_xy, has_head),
            severity_flag,
        ], dim=-1)
        return (
            self._type_vec(EntityType.DRC_VIOLATION, K, xy.device)
            + self.drc_proj(feats)
        )

    def encode_board(
        self,
        bbox_origin: torch.Tensor,    # (1, 2)
        bbox_size: torch.Tensor,      # (1, 2)
        n_copper: torch.Tensor,       # (1, 1)
    ) -> torch.Tensor:
        K = bbox_origin.size(0)
        feats = torch.cat([
            self._enc_xy(bbox_origin),
            self._enc_2d(bbox_size),
            self._enc_1d(n_copper),
        ], dim=-1)
        return self._type_vec(EntityType.BOARD, K, bbox_origin.device) + self.board_proj(feats)

    def encode_action_history(
        self,
        prev_type: torch.Tensor,          # (N,) int64        N = B*K entries
        prev_success: torch.Tensor,       # (N,) float
        prev_xy: torch.Tensor,            # (N, 2) float (normalized mm)
        prev_layer_d: torch.Tensor,       # (N, 2) float (layer dist encoding)
        prev_has_ptr: torch.Tensor,       # (N,) float in {0, 1}
        prev_mode: torch.Tensor,          # (N,) int64 (clamped ≥0)
        age: torch.Tensor,                # (N,) int64: 0 = newest
        action_type_weight: torch.Tensor, # (NUM_ACTIONS, d_model) — from policy
    ) -> torch.Tensor:
        """Emit 3 tokens per history entry: (N, 3, d_model).

        Weight-tying: ``action_type_weight`` is the policy's
        ``action_type_head.weight`` so the state-zone history at-token and
        the action-zone slot-0 token share embeddings for the same action
        ids. ``routing_mode_embed`` is shared likewise for the mode token.

        The age embedding (Fourier(age / MAX_HISTORY) → proj) is added to
        all 3 tokens of an entry — the only order signal in the otherwise
        permutation-equivariant state zone. Legacy prev-action checkpoints
        (``legacy_action_history``) skip it: single entry, bit-identical to
        the historical 3-token encoding.
        """
        import torch.nn.functional as F
        at_emb = F.embedding(prev_type, action_type_weight)              # (N, d)
        succ_emb = self.prev_action_success_proj(
            prev_success.unsqueeze(-1),
        )                                                                 # (N, d)
        prev_at = at_emb + succ_emb + self.prev_action_slot_emb[0]       # (N, d)

        xy_enc = self._enc_xy(prev_xy)                                   # (N, f)
        pt_feat = torch.cat(
            [xy_enc, prev_layer_d, prev_has_ptr.unsqueeze(-1)], dim=-1,
        )                                                                 # (N, f+3)
        prev_pt = self.prev_action_pt_proj(pt_feat) + self.prev_action_slot_emb[1]

        mode_emb = self.routing_mode_embed(prev_mode)                    # (N, d)
        prev_mo = mode_emb + self.prev_action_slot_emb[2]

        out = torch.stack([prev_at, prev_pt, prev_mo], dim=1)            # (N, 3, d)
        if not self.legacy_action_history:
            age_norm = age.to(out.dtype) / float(MAX_HISTORY)            # (N,)
            age_emb = self.history_age_proj(
                self.history_age_fourier(age_norm.unsqueeze(-1)),
            )                                                             # (N, d)
            out = out + age_emb.unsqueeze(1)
        return out

    # ------------------------------------------------------------------
    # Structural / control token lookup
    # ------------------------------------------------------------------
    def embed_structural(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.structural_embed(token_ids)

    def embed_routing_mode(self, mode_ids: torch.Tensor) -> torch.Tensor:
        return self.routing_mode_embed(mode_ids)
