"""Frozen per-observation reference StateTokenizer (test fixture).

Retired from production (the decoder-v1 per-obs tokenizer) and kept ONLY as
the reference implementation that
parity tests compare the live
``methods.rl_agent.models.v1.tokenizer.BatchedStateTokenizer`` against
(``tests/test_state_tokenizer_batched.py`` and friends). Do not use outside
tests; do not "improve" — bit-equivalence with history is the point.

Single-token-per-entity layout:

  * Each pad / track / via / ratsnest / cand / head / net / edge / board
    becomes ONE token.
  * Inside a single observation, all entities of the same type are
    encoded together (one matrix multiply per type).
  * ``net_indices`` points at the per-net NET token (one per net).
    ``cand_indices`` points at the per-cand CAND token.
  * Slot embedding is added with ``slot_scale * slot_emb_table`` and the
    entire sequence is then LayerNorm'd.

Sequence layout (per observation)::

    static zone:
        [BOARD]
        [EDGE_1 .. EDGE_E]
        [NET_1 PAD_1_1 PAD_1_2 .. NET_2 PAD_2_1 ..]   ← ★ NET tokens are pointer targets
        [OBST_1 .. OBST_O]                             ← obstacle_obs knob only

    dynamic + cand zone:
        [TRACK_* VIA_* RAT_*]                          ← per-net routing geometry
        [HEAD]
        [CAND_1 .. CAND_C]                             ← ★ CAND tokens are pointer targets
        [VAL] [SOD]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from methods.rl_agent.models.v1.embedding import (
    CandidateType,
    StructuralToken as ST,
    TokenVocabulary,
)
from methods.rl_agent.models.v1.encoding import (
    NormContext,
    TokenizerOutput,
    _apply_post_transform,
    _compute_norm_ctx,
    _maybe_swap_pair,
    _norm_dim,
    _norm_pos,
    _norm_pos_edge,
    _parse_net_code,
    _safe_encode_layer,
    _sorted_net_keys,
    shape_bucket_id,
)
from pcb_world.vec.candidate_pool import (
    build_directional_candidates,
    collect_raw_candidates,
)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SingleTokenResult:
    """Per-observation tokenization result (before batch padding)."""

    embeddings: torch.Tensor       # (N, d_model)
    net_end_positions: list[int]   # NET token positions
    cand_positions: list[int]      # CAND token positions
    n_static_tokens: int           # boundary between static and dynamic+cand
    cand_mm: list[tuple[float, float, int]]
    debug_labels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# StateTokenizer
# ---------------------------------------------------------------------------
class StateTokenizer(nn.Module):
    """Convert JSON observation → (B, max_seq, d_model) embeddings.

    Owns a :class:`TokenVocabulary` and produces :class:`TokenizerOutput`
    in the layout documented at the top of this module.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_freq: int = 32,
        max_seq_len: int = 10000,
        coord_encoding: str = "fourier",
        mlp_hidden: int = 128,
        disable_slot_emb: bool = False,
        action_history_len: int = 1,
        legacy_action_history: bool = False,
        obstacle_obs: bool = False,
        shape_obs: bool = False,
    ) -> None:
        super().__init__()
        self.vocab = TokenVocabulary(
            d_model=d_model,
            n_freq=n_freq,
            coord_encoding=coord_encoding,
            mlp_hidden=mlp_hidden,
            disable_slot_emb=disable_slot_emb,
            action_history_len=action_history_len,
            legacy_action_history=legacy_action_history,
            obstacle_obs=obstacle_obs,
            shape_obs=shape_obs,
        )
        self.obstacle_obs = bool(obstacle_obs)
        self.shape_obs = bool(shape_obs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def clear_static_cache(self) -> None:
        # No-op kept for backwards-compatibility with callers that try to
        # invalidate the prior tokenizer's static cache. The new tokenizer
        # rebuilds every step, so there is nothing to invalidate.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self, obs_list: list[dict], *,
        action_type_weight: torch.Tensor | None = None,
    ) -> TokenizerOutput:
        device = next(self.parameters()).device
        results = [
            self._tokenize_single(obs, device, action_type_weight=action_type_weight)
            for obs in obs_list
        ]

        batch_size = len(results)
        seq_lens = [r.embeddings.size(0) for r in results]
        max_seq = min(max(seq_lens), self.max_seq_len)
        max_nets = max((len(r.net_end_positions) for r in results), default=0)
        max_cands = max((len(r.cand_positions) for r in results), default=0)

        pad_embed = self.vocab.embed_structural(
            torch.tensor(int(ST.SEQ_PAD), device=device),
        )

        token_embs = torch.zeros(batch_size, max_seq, self.d_model, device=device)
        key_padding_mask = torch.ones(batch_size, max_seq, dtype=torch.bool, device=device)
        net_idx = torch.full((batch_size, max_nets), -1, dtype=torch.long, device=device)
        cand_idx = torch.full((batch_size, max_cands), -1, dtype=torch.long, device=device)
        s_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

        for b, r in enumerate(results):
            slen = min(r.embeddings.size(0), max_seq)
            token_embs[b, :slen] = r.embeddings[:slen]
            if slen < max_seq:
                token_embs[b, slen:] = pad_embed.unsqueeze(0)
            key_padding_mask[b, :slen] = False
            for i, pos in enumerate(r.net_end_positions):
                if i < max_nets and pos < max_seq:
                    net_idx[b, i] = pos
            for i, pos in enumerate(r.cand_positions):
                if i < max_cands and pos < max_seq:
                    cand_idx[b, i] = pos
            s_lens[b] = slen

        return TokenizerOutput(
            token_embeddings=token_embs,
            net_indices=net_idx,
            cand_indices=cand_idx,
            key_padding_mask=key_padding_mask,
            seq_lens=s_lens,
            cand_mm_list=[r.cand_mm for r in results],
        )

    # ------------------------------------------------------------------
    # Per-observation tokenization
    # ------------------------------------------------------------------
    def _tokenize_single(
        self, obs: dict, device: torch.device,
        *,
        action_type_weight: torch.Tensor | None = None,
    ) -> SingleTokenResult:
        bs = obs["board_static"]
        rg = obs.get("routing_geometry", {})
        rh = obs["router_head"]
        aug = obs.get("_aug")
        closed_codes = {f"net_{c}" for c in (obs.get("closed_nets") or [])}

        ctx = _compute_norm_ctx(bs, aug)
        net_keys = _sorted_net_keys(bs.get("nets", {}))
        if len(net_keys) > self.vocab.n_max_slots:
            raise ValueError(
                f"Board has {len(net_keys)} nets but slot table has only "
                f"{self.vocab.n_max_slots} slots."
            )
        net_to_slot: dict[str, int] = {nk: k for k, nk in enumerate(net_keys)}

        # DRC violations carry real net-name strings. Build real → slot map.
        real_name_to_slot: dict[str, int] = {}
        for _nk in net_keys:
            _real = bs["nets"][_nk].get("net_name")
            if isinstance(_real, str) and _real:
                real_name_to_slot.setdefault(_real, net_to_slot[_nk])

        # Head xy in normalized coordinates (None when idle).
        is_head_active = (
            rh.get("is_routing", False)
            or int(rh.get("current_net_phase", 0)) > 0
        )
        if is_head_active:
            hxy_n = _norm_pos(rh["current_xy"][0], rh["current_xy"][1], ctx)
            head_xy = torch.tensor(
                [hxy_n[0], hxy_n[1]], device=device, dtype=torch.float32,
            )
        else:
            head_xy = None

        token_chunks: list[torch.Tensor] = []
        slot_ids: list[int] = []
        labels: list[str] = []
        net_positions: list[int] = []

        # ============================================================
        # Static zone: BOARD, EDGEs, NET+PADs (per net)
        # ============================================================
        # 1) BOARD token
        bbox_origin = torch.tensor(
            [_norm_pos_edge(bs["bbox_x"], bs["bbox_y"], ctx)],
            device=device, dtype=torch.float32,
        )
        bbox_w_n, bbox_h_n = _maybe_swap_pair(
            _norm_dim(ctx.bbox_w_eff, ctx),
            _norm_dim(ctx.bbox_h_eff, ctx),
            ctx,
        )
        bbox_size = torch.tensor(
            [[bbox_w_n, bbox_h_n]],
            device=device, dtype=torch.float32,
        )
        n_copper_t = torch.tensor(
            [[float(ctx.n_copper)]], device=device, dtype=torch.float32,
        )
        token_chunks.append(self.vocab.encode_board(bbox_origin, bbox_size, n_copper_t))
        slot_ids.append(-1)
        labels.append("[BOARD]")

        # 2) EDGE tokens
        edges = bs.get("boardlines", {})
        if edges:
            xy1_list, xy2_list, mid_list, w_list = [], [], [], []
            for _eid, e in edges.items():
                x1, y1 = e["p1"]["xy"]
                x2, y2 = e["p2"]["xy"]
                m = e.get("mid")
                mx, my = m["xy"] if m is not None else ((x1 + x2) / 2.0,
                                                        (y1 + y2) / 2.0)
                xy1_list.append(_norm_pos_edge(x1, y1, ctx))
                xy2_list.append(_norm_pos_edge(x2, y2, ctx))
                mid_list.append(_norm_pos_edge(mx, my, ctx))
                w_list.append([_norm_dim(e["width"], ctx)])
            xy1 = torch.tensor(xy1_list, device=device, dtype=torch.float32)
            xy2 = torch.tensor(xy2_list, device=device, dtype=torch.float32)
            w = torch.tensor(w_list, device=device, dtype=torch.float32)
            if self.vocab.legacy_edge_encoding:
                token_chunks.append(self.vocab.encode_edge(xy1, xy2, w))
            else:
                mid = torch.tensor(mid_list, device=device, dtype=torch.float32)
                token_chunks.append(self.vocab.encode_edge(xy1, xy2, w, xy_mid=mid))
            slot_ids.extend([-1] * xy1.size(0))
            labels.extend([f"[EDGE_{i}]" for i in range(xy1.size(0))])

        # Helper to update running token offset.
        def _current_offset() -> int:
            return sum(c.size(0) for c in token_chunks)

        # 3) NET + PAD tokens (per net)
        for nk in net_keys:
            slot = net_to_slot[nk]
            net = bs["nets"][nk]
            c = net.get("constraints", {})
            tw = torch.tensor(
                [[_norm_dim(c.get("track_width", 0), ctx)]],
                device=device, dtype=torch.float32,
            )
            cl = torch.tensor(
                [[_norm_dim(c.get("clearance", 0), ctx)]],
                device=device, dtype=torch.float32,
            )
            vd = torch.tensor(
                [[_norm_dim(c.get("via_diameter", 0), ctx)]],
                device=device, dtype=torch.float32,
            )
            closed = torch.tensor(
                [[1.0 if nk in closed_codes else 0.0]],
                device=device, dtype=torch.float32,
            )
            net_positions.append(_current_offset())
            token_chunks.append(self.vocab.encode_net(tw, cl, vd, closed=closed))
            slot_ids.append(slot)
            labels.append(f"[NET_{nk}]")

            pads = net.get("pads", {})
            if pads:
                xy_list, wh_list, ls_list, le_list, sh_list = [], [], [], [], []
                for _pk, pad in pads.items():
                    xy_list.append(_norm_pos(pad["center"]["xy"][0], pad["center"]["xy"][1], ctx))
                    pad_w_n, pad_h_n = _maybe_swap_pair(
                        _norm_dim(pad["width"], ctx),
                        _norm_dim(pad["height"], ctx),
                        ctx,
                    )
                    wh_list.append([pad_w_n, pad_h_n])
                    # Via-style (layer_start, layer_end). Single-layer pads use
                    # ls == le; thru-hole pads (layer == 0) span (1, n_copper).
                    pad_layer = pad["layer"]
                    if pad_layer == 0:
                        layer_start, layer_end = 1, ctx.n_copper
                    else:
                        layer_start = layer_end = pad_layer
                    ls_dt, ls_db = _safe_encode_layer(layer_start, ctx.n_copper)
                    le_dt, le_db = _safe_encode_layer(layer_end, ctx.n_copper)
                    ls_list.append([ls_dt, ls_db])
                    le_list.append([le_dt, le_db])
                    sh_list.append(shape_bucket_id(pad.get("shape", "")))
                xy = torch.tensor(xy_list, device=device, dtype=torch.float32)
                wh = torch.tensor(wh_list, device=device, dtype=torch.float32)
                ls = torch.tensor(ls_list, device=device, dtype=torch.float32)
                le = torch.tensor(le_list, device=device, dtype=torch.float32)
                sh = (torch.tensor(sh_list, device=device, dtype=torch.int64)
                      if self.shape_obs else None)
                token_chunks.append(
                    self.vocab.encode_pad(xy, wh, ls, le, head_xy, shape_id=sh)
                )
                slot_ids.extend([slot] * xy.size(0))
                labels.extend([f"[PAD_{nk}_{i}]" for i in range(xy.size(0))])

        # -------- OBSTACLE (obstacle_obs knob; netless blockers) --------
        # NPTH holes/slots (rule-area keepout entries filtered by
        # shape == "polygon") then NC pads — mirrors the batched walks.
        if self.obstacle_obs:
            o_xy, o_wh, o_ls, o_le, o_sh = [], [], [], [], []
            for src_key in ("obstacles", "unconnected_pads"):
                for o in bs.get(src_key, {}).values():
                    shape = o.get("shape", "")
                    if shape == "polygon":
                        continue
                    o_xy.append(_norm_pos(o["center"]["xy"][0], o["center"]["xy"][1], ctx))
                    o_w_n, o_h_n = _maybe_swap_pair(
                        _norm_dim(o["width"], ctx),
                        _norm_dim(o["height"], ctx),
                        ctx,
                    )
                    o_wh.append([o_w_n, o_h_n])
                    o_layer = o["layer"]
                    if o_layer == 0:
                        layer_start, layer_end = 1, ctx.n_copper
                    else:
                        layer_start = layer_end = o_layer
                    ls_dt, ls_db = _safe_encode_layer(layer_start, ctx.n_copper)
                    le_dt, le_db = _safe_encode_layer(layer_end, ctx.n_copper)
                    o_ls.append([ls_dt, ls_db])
                    o_le.append([le_dt, le_db])
                    o_sh.append(shape_bucket_id(shape))
            if o_xy:
                xy = torch.tensor(o_xy, device=device, dtype=torch.float32)
                wh = torch.tensor(o_wh, device=device, dtype=torch.float32)
                ls = torch.tensor(o_ls, device=device, dtype=torch.float32)
                le = torch.tensor(o_le, device=device, dtype=torch.float32)
                sh = (torch.tensor(o_sh, device=device, dtype=torch.int64)
                      if self.shape_obs else None)
                token_chunks.append(
                    self.vocab.encode_obstacle(xy, wh, ls, le, head_xy, shape_id=sh)
                )
                slot_ids.extend([-1] * xy.size(0))
                labels.extend([f"[OBST_{i}]" for i in range(xy.size(0))])

        n_static_tokens = _current_offset()

        # ============================================================
        # Dynamic+Cand zone: tracks/vias/rats per net, head, cands, VAL/SOD
        # ============================================================
        for nk in _sorted_net_keys(rg):
            net_geom = rg[nk]
            slot = net_to_slot.get(nk, -1)

            tracks = net_geom.get("tracks", {})
            if tracks:
                xy1_l, xy2_l, w_l, ly_l = [], [], [], []
                for _tk, tr in tracks.items():
                    xy1_l.append(_norm_pos(tr["p1"]["xy"][0], tr["p1"]["xy"][1], ctx))
                    xy2_l.append(_norm_pos(tr["p2"]["xy"][0], tr["p2"]["xy"][1], ctx))
                    w_l.append([_norm_dim(tr["width"], ctx)])
                    dt, db = _safe_encode_layer(tr["layer"], ctx.n_copper)
                    ly_l.append([dt, db])
                xy1 = torch.tensor(xy1_l, device=device, dtype=torch.float32)
                xy2 = torch.tensor(xy2_l, device=device, dtype=torch.float32)
                w = torch.tensor(w_l, device=device, dtype=torch.float32)
                ld = torch.tensor(ly_l, device=device, dtype=torch.float32)
                token_chunks.append(self.vocab.encode_track(xy1, xy2, w, ld, head_xy))
                slot_ids.extend([slot] * xy1.size(0))
                labels.extend([f"[TRACK_{nk}_{i}]" for i in range(xy1.size(0))])

            vias = net_geom.get("vias", {})
            if vias:
                xy_l, ls_l, le_l, dia_l = [], [], [], []
                for _vk, via in vias.items():
                    xy_l.append(_norm_pos(via["center"]["xy"][0], via["center"]["xy"][1], ctx))
                    ls_dt, ls_db = _safe_encode_layer(via["layer_start"], ctx.n_copper)
                    le_dt, le_db = _safe_encode_layer(via["layer_end"], ctx.n_copper)
                    ls_l.append([ls_dt, ls_db])
                    le_l.append([le_dt, le_db])
                    dia_l.append([_norm_dim(via.get("via_width", 0), ctx)])
                xy = torch.tensor(xy_l, device=device, dtype=torch.float32)
                ls = torch.tensor(ls_l, device=device, dtype=torch.float32)
                le = torch.tensor(le_l, device=device, dtype=torch.float32)
                dia = torch.tensor(dia_l, device=device, dtype=torch.float32)
                token_chunks.append(self.vocab.encode_via(xy, ls, le, dia, head_xy))
                slot_ids.extend([slot] * xy.size(0))
                labels.extend([f"[VIA_{nk}_{i}]" for i in range(xy.size(0))])

            points = net_geom.get("points", [])
            if points:
                xy_l = []
                for pt in points:
                    xy_l.append(_norm_pos(pt["xy"][0], pt["xy"][1], ctx))
                xy = torch.tensor(xy_l, device=device, dtype=torch.float32)
                token_chunks.append(self.vocab.encode_rat(xy, head_xy))
                slot_ids.extend([slot] * xy.size(0))
                labels.extend([f"[RAT_{nk}_{i}]" for i in range(xy.size(0))])

        # DRC violation tokens (between RAT and HEAD).
        from methods.rl_agent.models.v1.embedding import NUM_DRC_TYPES as _NDRC
        _violations = obs.get("drc_violations", []) or []
        if _violations:
            _xy_l, _ld_l, _tid_l, _sev_l, _slot_l = [], [], [], [], []
            for _v in _violations:
                _xy_l.append(_norm_pos(_v["x_mm"], _v["y_mm"], ctx))
                _vdt, _vdb = _safe_encode_layer(
                    int(_v.get("layer", 1)), ctx.n_copper,
                )
                _ld_l.append([_vdt, _vdb])
                _tid = int(_v.get("type_id", _NDRC - 1))
                if _tid < 0 or _tid >= _NDRC:
                    _tid = _NDRC - 1
                _tid_l.append(_tid)
                _sev_l.append(
                    [1.0 if int(_v.get("severity", 0)) == 0x20 else 0.0],
                )
                _nets_v = _v.get("net_names") or []
                _slot_l.append(
                    real_name_to_slot.get(_nets_v[0], -1) if _nets_v else -1,
                )
            _xy = torch.tensor(_xy_l, device=device, dtype=torch.float32)
            _ld = torch.tensor(_ld_l, device=device, dtype=torch.float32)
            _tid_t = torch.tensor(_tid_l, device=device, dtype=torch.long)
            _sev_t = torch.tensor(_sev_l, device=device, dtype=torch.float32)
            token_chunks.append(
                self.vocab.encode_drc(
                    _xy, _ld, _tid_t, _sev_t, head_xy,
                ),
            )
            slot_ids.extend(_slot_l)
            labels.extend([f"[DRC_{i}]" for i in range(_xy.size(0))])

        # HEAD token
        current_net_id = rh.get("current_net", -1)
        if isinstance(current_net_id, int) and current_net_id > 0:
            head_slot = net_to_slot.get(f"net_{current_net_id}", -1)
        else:
            head_slot = -1
        h_layer = rh["current_layer"]
        hdt, hdb = _safe_encode_layer(h_layer, ctx.n_copper)
        # When idle (head not active) we still emit a HEAD token but with
        # zeros for xy / layer so the policy always gets one. Matches the
        # head_xy=None contract used by encode_pad / encode_track above.
        if head_xy is None:
            hxy_t = torch.zeros(1, 2, device=device, dtype=torch.float32)
        else:
            hxy_t = head_xy.unsqueeze(0)
        h_layer_t = torch.tensor([[hdt, hdb]], device=device, dtype=torch.float32)
        rm_t = torch.tensor([int(rh["routing_mode"])], device=device, dtype=torch.long)
        np_t = torch.tensor([int(rh["current_net_phase"])], device=device, dtype=torch.long)
        sr_t = torch.tensor([[float(rh["step_ratio"])]], device=device, dtype=torch.float32)
        token_chunks.append(
            self.vocab.encode_head(hxy_t, h_layer_t, rm_t, np_t, sr_t),
        )
        slot_ids.append(head_slot)
        labels.append("[HEAD]")

        # CAND tokens
        cand_positions, cand_mm = self._build_candidate_pool(
            obs, ctx, head_xy, head_slot, device, token_chunks, slot_ids, labels,
        )

        # ACTION_HISTORY: 3 tokens (at, pt, mo) per entry, K entries per obs,
        # always emitted (idle-sentinel padded; K=1 slot/age-free in legacy
        # prev-action mode). When action_type_weight is not supplied, reserve
        # zero-vector positions (matches BatchedStateTokenizer's behavior so
        # parity tests hold without having to pass the policy weight).
        history = obs.get("action_history") or []
        legacy_hist = self.vocab.legacy_action_history
        for age in range(self.vocab.action_history_len):
            pa = history[age] if age < len(history) else None
            if pa is None:
                pa_type_v = 6  # ACT_IDLE
                pa_succ_v = 1.0
                pa_xy_v = (0.0, 0.0)
                pa_ld_v = (0.0, 0.0)
                pa_has_v = 0.0
                pa_mode_v = 0
                pa_slot_v = -1
            else:
                pa_type_v = int(pa["action_type"])
                pa_succ_v = 1.0 if pa.get("success") else 0.0
                if pa.get("has_pointer"):
                    _pxy = _norm_pos(pa["pointer_xy"][0], pa["pointer_xy"][1], ctx)
                    pa_xy_v = (_pxy[0], _pxy[1])
                    _pdt, _pdb = _safe_encode_layer(
                        int(pa.get("pointer_layer", 0)), ctx.n_copper,
                    )
                    pa_ld_v = (_pdt, _pdb)
                    pa_has_v = 1.0
                else:
                    pa_xy_v = (0.0, 0.0)
                    pa_ld_v = (0.0, 0.0)
                    pa_has_v = 0.0
                _rm = int(pa.get("routing_mode", -1))
                pa_mode_v = _rm if _rm >= 0 else 0
                _nid = pa.get("net_id")
                pa_slot_v = (
                    -1 if (legacy_hist or _nid is None)
                    else net_to_slot.get(f"net_{int(_nid)}", -1)
                )

            if action_type_weight is not None:
                type_t = torch.tensor([pa_type_v], dtype=torch.long, device=device)
                succ_t = torch.tensor([pa_succ_v], dtype=torch.float32, device=device)
                xy_t = torch.tensor([list(pa_xy_v)], dtype=torch.float32, device=device)
                ld_t = torch.tensor([list(pa_ld_v)], dtype=torch.float32, device=device)
                has_t = torch.tensor([pa_has_v], dtype=torch.float32, device=device)
                mode_t = torch.tensor([pa_mode_v], dtype=torch.long, device=device)
                age_t = torch.tensor([age], dtype=torch.long, device=device)
                pa_emb = self.vocab.encode_action_history(
                    type_t, succ_t, xy_t, ld_t, has_t, mode_t, age_t,
                    action_type_weight,
                )  # (1, 3, d)
                token_chunks.append(pa_emb.squeeze(0))
            else:
                token_chunks.append(
                    torch.zeros(3, self.d_model, device=device, dtype=torch.float32),
                )
            slot_ids.extend([pa_slot_v] * 3)
            labels.extend(
                [f"[HIST{age}_AT]", f"[HIST{age}_PT]", f"[HIST{age}_MO]"],
            )

        # VAL + SOD
        val_id = torch.tensor(int(ST.VAL), device=device, dtype=torch.long)
        sod_id = torch.tensor(int(ST.SOD), device=device, dtype=torch.long)
        token_chunks.append(self.vocab.embed_structural(val_id).unsqueeze(0))
        slot_ids.append(-1)
        labels.append("[VAL]")
        token_chunks.append(self.vocab.embed_structural(sod_id).unsqueeze(0))
        slot_ids.append(-1)
        labels.append("[SOD]")

        # ============================================================
        # Stack and apply slot embedding + LayerNorm
        # ============================================================
        all_embs = torch.cat(token_chunks, dim=0)  # (N, d_model)
        slot_ids_t = torch.tensor(slot_ids, dtype=torch.long, device=device)

        slot_perm = aug.get("slot_perm") if aug else None
        if slot_perm is not None:
            perm_t = torch.as_tensor(slot_perm, dtype=torch.long, device=device)
            valid = slot_ids_t >= 0
            remapped = perm_t[slot_ids_t.clamp(min=0)]
            slot_ids_t = torch.where(valid, remapped, slot_ids_t)

        if self.vocab.disable_slot_emb:
            all_embs = self.vocab.embed_ln(all_embs)
        else:
            valid_mask = (slot_ids_t >= 0).unsqueeze(-1).to(all_embs.dtype)
            safe_ids = slot_ids_t.clamp(min=0)
            slot_contrib = self.vocab.slot_emb_table[safe_ids] * valid_mask
            all_embs = self.vocab.embed_ln(
                all_embs + self.vocab.slot_scale * slot_contrib,
            )

        return SingleTokenResult(
            embeddings=all_embs,
            net_end_positions=net_positions,
            cand_positions=cand_positions,
            n_static_tokens=n_static_tokens,
            cand_mm=cand_mm,
            debug_labels=labels,
        )

    # ------------------------------------------------------------------
    # Candidate pool builder (single-token-per-cand)
    # ------------------------------------------------------------------
    def _build_candidate_pool(
        self,
        obs: dict,
        ctx: NormContext,
        head_xy: torch.Tensor | None,
        head_slot: int,
        device: torch.device,
        token_chunks: list[torch.Tensor],
        slot_ids: list[int],
        labels: list[str],
    ) -> tuple[list[int], list[tuple[float, float, int]]]:
        rh = obs["router_head"]
        current_net_id = rh.get("current_net", -1)
        if current_net_id <= 0:
            current_net_id = None

        extra = None
        if rh.get("is_routing", False):
            head_xy_mm = rh["current_xy"]
            _ra = obs.get("_aug") or {}
            extra = build_directional_candidates(
                (head_xy_mm[0], head_xy_mm[1]), rh["current_layer"],
                mode=_ra.get("directional_candidates"),
            )

        raw_cands = collect_raw_candidates(obs, current_net_id, extra)

        if not raw_cands:
            return [], []

        # Compute starting position for cand tokens.
        offset = sum(c.size(0) for c in token_chunks)

        type_l: list[int] = []
        xy_l: list[tuple[float, float]] = []
        layer_l: list[list[float]] = []
        cand_positions: list[int] = []
        cand_mm: list[tuple[float, float, int]] = []

        for k, (x_mm, y_mm, ly, ct) in enumerate(raw_cands):
            ct_int = int(ct) if isinstance(ct, int) else int(ct.value)
            type_l.append(ct_int)
            xy_l.append(_norm_pos(x_mm, y_mm, ctx))
            dt, db = _safe_encode_layer(ly, ctx.n_copper)
            layer_l.append([dt, db])
            cand_positions.append(offset + k)
            cand_mm.append((x_mm, y_mm, ly))

        type_t = torch.tensor(type_l, dtype=torch.long, device=device)
        xy_t = torch.tensor(xy_l, dtype=torch.float32, device=device)
        ld_t = torch.tensor(layer_l, dtype=torch.float32, device=device)
        token_chunks.append(self.vocab.encode_cand(type_t, xy_t, ld_t, head_xy))
        slot_ids.extend([head_slot] * xy_t.size(0))
        labels.extend([f"[CAND_{k}({CandidateType(t).name})]" for k, t in enumerate(type_l)])

        return cand_positions, cand_mm


# ---------------------------------------------------------------------------
# Debug dump (kept for ad-hoc inspection)
# ---------------------------------------------------------------------------
def dump_tokenized(
    tokenizer: StateTokenizer,
    obs: dict,
    path: str | None = None,
) -> list[str]:
    device = next(tokenizer.parameters()).device
    result = tokenizer._tokenize_single(obs, device)

    net_set = set(result.net_end_positions)
    cand_set = set(result.cand_positions)
    labels = result.debug_labels
    n_embs = result.embeddings.size(0)

    lines: list[str] = []
    lines.append(f"Total tokens: {n_embs}")
    lines.append(
        f"Static prefix: 0..{result.n_static_tokens - 1} "
        f"({result.n_static_tokens} tokens)"
    )
    lines.append("")
    lines.append("Net Pointer Indices:")
    for i, pos in enumerate(result.net_end_positions):
        tok_label = labels[pos] if pos < len(labels) else "?"
        lines.append(f"  {i:>3}  pos={pos:>4}  {tok_label}")
    lines.append("")
    lines.append("Candidate Pointer Indices:")
    for i, pos in enumerate(result.cand_positions):
        tok_label = labels[pos] if pos < len(labels) else "?"
        mm = result.cand_mm[i] if i < len(result.cand_mm) else None
        mm_str = f"({mm[0]:.1f}, {mm[1]:.1f}, L{mm[2]})" if mm else "?"
        lines.append(f"  {i:>3}  pos={pos:>4}  {mm_str:>22}  {tok_label}")
    lines.append("")
    lines.append(f"{'idx':>4}  {'marker':>6}  token")
    lines.append("=" * 72)
    for i in range(n_embs):
        marker = ""
        if i in net_set:
            marker = "★net"
        elif i in cand_set:
            marker = "★cand"
        elif i == result.n_static_tokens - 1:
            marker = "──end"
        label = labels[i] if i < len(labels) else "?"
        lines.append(f"{i:4d}  {marker:>6}  {label}")

    if path is not None:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    return lines
