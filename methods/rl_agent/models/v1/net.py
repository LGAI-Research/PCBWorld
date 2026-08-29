"""Decoder-Only Transformer RL policy for PCB routing (policy layer).

Owns the RL-specific surface: action constants / SLOT_USAGE, the
``KiCadRLModel`` (tokenizer + transformer stack + pointer/value heads,
autoregressive ``act`` / ``evaluate_actions`` — one state pass (K/V cache) +
incremental action-token decode). The pure network
building blocks (ReZero / SameNetBias / MultiHeadAttention /
GatedTransformerLayer / build_2zone_mask / combine_masks / init_weights) live
in :mod:`methods.rl_agent.models.v1.blocks`.

Architecture notes:
  * Each residual sublayer is gated by ReZero (one learned scalar per
    sublayer, initialized to 0).
  * No rotary positional encoding — the state zone is permutation-equivariant;
    a learned ``action_pos_emb`` is injected on the action-zone tokens only.
  * Pointer / action_type logits are gated through ``clip_C * tanh(.)`` to
    suppress entropy collapse and gradient blow-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from methods.rl_agent.models.v1.blocks import (
    GatedTransformerLayer,
    SameNetBias,
    build_2zone_mask,
    build_slot_membership,
    combine_masks,
    init_weights,
)
from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from methods.rl_agent.models.v1.encoding import TokenizerOutput

# The action-space contract lives in the v1 spec; re-exported here so callers
# can import it (``SLOT_USAGE`` etc.) alongside the model.
from methods.rl_agent.models.v1.spec import (  # noqa: F401
    NUM_ACTION_TYPES,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACT_NET_END,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_FINISH,
    ACT_IDLE,
    NUM_ROUTING_MODES,
    SLOT_USAGE,
)


# ---------------------------------------------------------------------------
# Incremental-decode cache
# ---------------------------------------------------------------------------
@dataclass
class _DecodeCache:
    """State-pass K/V + padding mask for exact incremental action decoding.

    ``kv``: per-layer ``(k, v)``, each ``(B, H, L_prefix, d_head)`` — raw
    (un-augmented) projections captured by ``_run_transformer(...,
    return_cache=True)``. They remain autograd-graph nodes, so a training
    decode reuses the state pass instead of recomputing it.
    ``key_padding_mask``: ``(B, L_prefix)`` bool, True = padded prefix key.
    """

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    key_padding_mask: torch.Tensor


@dataclass
class _StateEnc:
    """Shared state-encoding result — the return value of ``_encode_state``.

    The Pass-0 output shared by the three public forwards
    (``act_and_value``/``evaluate_actions_and_value``/``factored_action_logits``):
    tokenize → state pass → VAL critic + SOD action-type logits. ``cache`` is
    set only when ``return_cache=True`` (for incremental action-token
    decode); otherwise None.
    """

    tok_out: "TokenizerOutput"
    H_state: torch.Tensor
    cache: "_DecodeCache | None"
    values: torch.Tensor
    at_logits: torch.Tensor
    B: int
    device: torch.device
    arange_B: torch.Tensor
    seq_lens: torch.Tensor
    n_state_max: int


# ---------------------------------------------------------------------------
# KiCadRLModel
# ---------------------------------------------------------------------------
class KiCadRLModel(nn.Module):
    """Decoder-Only Transformer policy with pointer-network action selection.

    Architecture: BatchedStateTokenizer → GatedTransformerLayer stack → pointer/embedding heads.

    Action layout (``(B, 3)`` int64)::

        [0] action_type   in [0..5]
        [1] pointer_idx   in [-1, max_pointer) — -1 if unused
        [2] routing_mode  in [-1, 3)           — -1 if unused

    Action-type → parameters mapping is defined by :data:`SLOT_USAGE`.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,  # head_dim 16 at d_model=128
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 10000,
        n_freq: int = 32,
        use_critic: bool = False,
        detach_critic: bool = False,
        coord_encoding: str = "fourier",
        mlp_hidden: int = 128,
        disable_slot_emb: bool = False,
        policy_net_select: bool = False,
        same_net_bias: bool = False,
        legacy_pad_layer_encoding: bool = False,
        legacy_net_encoding: bool = False,
        legacy_edge_encoding: bool = False,
        time_feature: str = "step_ratio",
        time_feature_cap: int = 10000,
        n_max_slots: int = 64,
        action_history_len: int = 1,
        legacy_action_history: bool = False,
        obstacle_obs: bool = False,
        shape_obs: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.scale = math.sqrt(d_model)
        self.use_critic = use_critic
        self.detach_critic = detach_critic
        # When True, the policy's net_select pointer is propagated to the env
        # (learned net ordering). When False, KiCadRLWrapper overrides it
        # with a random unrouted-net pick.
        # Training-loop helpers read this to decide whether to fetch
        # per-env ``net_valid_mask`` from the vec env pool.
        self.policy_net_select = bool(policy_net_select)
        # AM-style logit clipping (pointer + action_type heads).
        # Why: scaled dot-product logits can blow up early in training,
        # causing entropy collapse and gradient explosions on pointer heads.
        self.clip_C = 10.0

        # Tokenizer owns the vocabulary (structural + Fourier/MLP projections).
        self.tokenizer = BatchedStateTokenizer(
            d_model=d_model,
            n_freq=n_freq,
            max_seq_len=max_seq_len,
            coord_encoding=coord_encoding,
            mlp_hidden=mlp_hidden,
            disable_slot_emb=disable_slot_emb,
            legacy_pad_layer_encoding=legacy_pad_layer_encoding,
            legacy_net_encoding=legacy_net_encoding,
            legacy_edge_encoding=legacy_edge_encoding,
            time_feature=time_feature,
            time_feature_cap=time_feature_cap,
            n_max_slots=n_max_slots,
            action_history_len=action_history_len,
            legacy_action_history=legacy_action_history,
            obstacle_obs=obstacle_obs,
            shape_obs=shape_obs,
        )

        # Transformer stack (ReZero residual, prefix-LM 2-zone mask, no RoPE).
        self.layers = nn.ModuleList(
            [
                GatedTransformerLayer(d_model, n_heads, d_ff)
                for _ in range(n_layers)
            ]
        )

        # Optional per-head same-net attention bias (opt-in). Zero-init so
        # that turning it on doesn't disturb early training. Shared across
        # all transformer layers.
        self.same_net_bias = SameNetBias(n_heads) if same_net_bias else None
        # Speed-experiment knobs (plain attrs, NOT state_dict entries; wired
        # by :meth:`configure_speed`). ``bf16_compute`` wraps the transformer
        # compute (stack + incremental decode) in torch.autocast(bfloat16)
        # and hands fp32 hiddens back to the heads — probability/log-prob
        # math, GAE and the optimizer stay fp32. ``_stack_fn``/``_decode_fn``
        # are the torch.compile seams ('stack'/'decode' regions); the 'heads'
        # region compiles the pointer/value helpers in place.
        self.bf16_compute = False
        self._stack_fn = self._stack_impl
        self._decode_fn = self._decode_impl

        # Learned absolute position embedding for the action zone only.
        # State zone is a permutation-equivariant set; ordering of action
        # tokens is meaningful (action_type → pointer), so we inject one
        # learned embedding per action slot:
        #   slot 0: at_tok      (action_type token)
        #   slot 1: point_tok   (chosen-pointer state token re-emitted)
        # Mode is read off h_at / h_pt without a dedicated token, so two
        # slots suffice.
        self.action_pos_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)

        # Action-type embedding (input token + output scoring).
        # Used both as input token (when action_type is chosen) and as output
        # projection (dot-producted with h_SOD to produce action_type logits).
        self.action_type_head = nn.Embedding(NUM_ACTION_TYPES, d_model)

        # Routing-mode embedding is reused from the tokenizer vocabulary
        # (weight tying between state encoding and action output).
        # Access via self.tokenizer.vocab.routing_mode_embed.

        # Critic head for PPO (optional). Uses the VAL-position hidden state
        # (h_VAL = H_state[arange_B, seq_lens - 2]) as state summary.
        # The VAL token sits one position before SOD: [..., VAL, SOD].
        # This separates value representation from policy representation,
        # reducing gradient interference between the two objectives.
        if use_critic:
            self.critic_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        # SB3-style orthogonal init for all nn.Linear (gain=0.01 for *head*).
        init_weights(self)
        # action_type_head is an nn.Embedding, not Linear → init explicitly
        # small so the initial action_type distribution is near-uniform
        # (matches the "head" convention in init_weights).
        nn.init.orthogonal_(self.action_type_head.weight, gain=0.01)

    # ------------------------------------------------------------------
    # Speed knobs (experiment wiring — profile.py --bf16/--compile-*)
    # ------------------------------------------------------------------
    def configure_speed(
        self,
        bf16: bool = False,
        compile_regions: tuple[str, ...] | list[str] = (),
        compile_mode: str = "default",
    ) -> None:
        """Wire the bf16/torch.compile experiment knobs (idempotent-ish; call
        once, right after construction and .to(device)).

        Args:
            bf16: autocast(bfloat16) around stack + incremental decode; heads
                and all probability math stay fp32.
            compile_regions: subset of {'stack', 'decode', 'heads', 'encode'} —
                'stack' = full-pass layer loop, 'decode' = incremental
                K/V-append loop, 'heads' = pointer/value helpers, 'encode' =
                tokenizer ``vocab.encode_*`` entity encoders (Fourier ladders
                + projections; the dict/H2D glue in ``_encode_all`` stays
                eager — only the pure tensor encoders are compiled).
                The alias ``'efficient'`` expands to
                {'stack', 'decode', 'heads'}; ``'encode'`` is deliberately
                excluded from it — compiling the encoders measures slower.
            compile_mode: 'default' | 'reduce-overhead' | 'max-autotune'
                (torch.compile mode; 'default' passes mode=None).
        """
        regions = set(compile_regions)
        if "efficient" in regions:
            regions = (regions - {"efficient"}) | {"stack", "decode", "heads"}
        unknown = regions - {"stack", "decode", "heads", "encode"}
        assert not unknown, f"unknown compile regions: {unknown}"
        self.bf16_compute = bool(bf16)
        mode = None if compile_mode == "default" else compile_mode
        if "stack" in regions:
            self._stack_fn = torch.compile(
                self._stack_impl, mode=mode, dynamic=True,
            )
        if "decode" in regions:
            self._decode_fn = torch.compile(
                self._decode_impl, mode=mode, dynamic=True,
            )
        if "heads" in regions:
            self._combined_ptr_logits = torch.compile(
                self._combined_ptr_logits, mode=mode, dynamic=True,
            )
            self._compute_value = torch.compile(
                self._compute_value, mode=mode, dynamic=True,
            )
        if "encode" in regions:
            vocab = self.tokenizer.vocab
            for name in dir(vocab):
                if name.startswith("encode_"):
                    setattr(
                        vocab, name,
                        torch.compile(getattr(vocab, name), mode=mode, dynamic=True),
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_transformer(
        self,
        embs: torch.Tensor,
        n_state: int,
        key_padding_mask: torch.Tensor,
        slot_ids: torch.Tensor | None = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, "_DecodeCache"]:
        """Run the full Transformer stack with a 2-zone prefix-LM mask.

        Args:
            embs: ``(B, L, d)`` token embeddings.
            n_state: Scalar number of state tokens (BOARD/EDGE/NET/PAD/OBSTACLE/
                TRACK/VIA/RAT/DRC/HEAD/CAND/ACTION_HISTORY/VAL/SOD).
            key_padding_mask: ``(B, L)`` bool, True = padded position.
            return_cache: also return a :class:`_DecodeCache` (per-layer K/V +
                padding mask) for :meth:`_decode_appended`. The K/V tensors
                stay in the autograd graph, so training reuses — not
                recomputes — the state pass.

        Returns:
            ``(B, L, d)`` hidden states after all layers; with
            ``return_cache``, a ``(hidden, cache)`` tuple.
        """
        _, L, _ = embs.shape
        zone = build_2zone_mask(n_state, L).to(embs.device)
        attn_mask = combine_masks(zone, key_padding_mask)  # (B, 1, L, L)
        if attn_mask.dtype != embs.dtype:
            # combine_masks builds float32; match embs so SDPA accepts the
            # mask under non-default dtypes (e.g. float64 equivalence tests).
            attn_mask = attn_mask.to(embs.dtype)
        # Per-head same-net additive bias (opt-in). slot_ids covers state
        # tokens; appended action tokens get slot -1 (no-slot).
        same_net_aug = None
        if self.same_net_bias is not None and slot_ids is not None:
            B = embs.size(0)
            if slot_ids.size(1) < L:
                pad = torch.full(
                    (B, L - slot_ids.size(1)), -1,
                    dtype=slot_ids.dtype, device=slot_ids.device,
                )
                slot_ids = torch.cat([slot_ids, pad], dim=1)
            # Absorb α_h·MMᵀ into q/k channels — exact reformulation of the
            # additive same-net bias (blocks.build_slot_membership); the
            # (B,H,L,L) bias tensor is never built. Shared M across all layers.
            same_net_aug = (build_slot_membership(slot_ids), self.same_net_bias.alpha)
        if self.bf16_compute:
            # SDPA requires mask dtype == query dtype; under autocast the
            # projections emit bf16. -inf is representable in bf16.
            attn_mask = attn_mask.to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, kv = self._stack_fn(embs, attn_mask, same_net_aug, return_cache)
            x = x.float()
        else:
            x, kv = self._stack_fn(embs, attn_mask, same_net_aug, return_cache)
        if not return_cache:
            return x
        return x, _DecodeCache(kv=kv, key_padding_mask=key_padding_mask)

    def _stack_impl(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        same_net_aug: tuple[torch.Tensor, torch.Tensor] | None,
        return_kv: bool,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Layer-stack loop — the 'stack' torch.compile target (pure tensors)."""
        if not return_kv:
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask, same_net_aug=same_net_aug)
            return x, None
        kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            x, layer_kv = layer(
                x, attn_mask=attn_mask, same_net_aug=same_net_aug,
                return_kv=True,
            )
            kv.append(layer_kv)
        return x, kv

    def _decode_appended(
        self,
        cache: "_DecodeCache",
        new_embs: torch.Tensor,
        appended_mask: torch.Tensor | None = None,
        extend_cache: bool = True,
    ) -> tuple[torch.Tensor, "_DecodeCache | None"]:
        """Exact incremental decode: append action token(s) to a cached pass.

        Equivalent to re-running the full stack on ``[prefix, new_embs]`` with
        the 2-zone mask and reading the hidden states at the appended
        positions: prefix hiddens cannot change (state→action attention is
        blocked), appended tokens attend to all prefix tokens (minus padding)
        and causally among themselves, and the same-net bias contributes
        exactly zero to them (appended tokens carry slot ``-1`` — see
        :meth:`MultiHeadAttention.forward_incremental`).

        Args:
            cache: From ``_run_transformer(..., return_cache=True)`` or a
                prior ``_decode_appended`` call.
            new_embs: ``(B, n_new, d)`` appended token embeddings, in action
                order (earlier tokens are attendable by later ones).
            appended_mask: optional ``(n_new, n_new)`` ADDITIVE mask among the
                appended tokens (0 = visible, ``-inf`` = blocked), replacing the
                default causal one. Use it to pack several INDEPENDENT action
                branches that share this prefix into one decode call — a
                block-structured mask keeps each branch blind to the others, so
                every appended hidden equals what a per-branch full rerun would
                give (see :meth:`factored_action_logits`). Rows must keep their
                own diagonal visible.

            extend_cache: build ``cache'`` (default). ``False`` returns ``None``
                instead and skips the per-layer prefix concatenation — for
                one-shot decodes that discard it (see :meth:`_decode_impl`).

        Returns:
            ``(h_new, cache')`` — ``h_new`` is ``(B, n_new, d)`` final-layer
            hiddens of the appended tokens; ``cache'`` extends the prefix by
            ``n_new`` (never-padded) positions for further decoding (only
            meaningful for the default single-branch causal mask), or ``None``
            under ``extend_cache=False``.
        """
        B, n_new, _ = new_embs.shape
        kpm = cache.key_padding_mask  # (B, L_prefix) bool
        device = new_embs.device
        dtype = torch.bfloat16 if self.bf16_compute else new_embs.dtype
        # Additive mask (B, 1, n_new, L_prefix + n_new): -inf at padded prefix
        # keys; causal upper-triangle (or the caller's branch structure) among
        # the appended tokens.
        prefix_mask = torch.where(
            kpm, float("-inf"), 0.0,
        ).to(dtype).view(B, 1, 1, -1).expand(B, 1, n_new, -1)
        if appended_mask is None:
            appended = torch.triu(
                torch.full((n_new, n_new), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
        else:
            appended = appended_mask.to(device=device, dtype=dtype)
        appended = appended.view(1, 1, n_new, n_new).expand(B, 1, n_new, n_new)
        attn_mask = torch.cat([prefix_mask, appended], dim=-1)

        if self.bf16_compute:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, new_kv = self._decode_fn(new_embs, cache.kv, attn_mask,
                                            extend_cache)
            x = x.float()
        else:
            x, new_kv = self._decode_fn(new_embs, cache.kv, attn_mask,
                                        extend_cache)
        if not extend_cache:
            return x, None
        new_kpm = torch.cat(
            [kpm, torch.zeros(B, n_new, dtype=torch.bool, device=kpm.device)],
            dim=1,
        )
        return x, _DecodeCache(kv=new_kv, key_padding_mask=new_kpm)

    def _decode_impl(
        self,
        x: torch.Tensor,
        kv_prefix: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        extend_cache: bool = True,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Incremental-decode layer loop — the 'decode' torch.compile target.

        ``extend_cache=False`` skips building the extended prefix cache. The
        concatenation allocates two ``(B, H, L_prefix + n_new, d_head)`` tensors
        PER LAYER, so a caller that discards the returned cache (the one-shot
        decodes in ``factored_action_logits`` / ``evaluate_actions_and_value``)
        pays that bandwidth for nothing. The hidden states are unaffected —
        ``k_n``/``v_n`` are consumed inside the layer either way.
        """
        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (k_p, v_p) in zip(self.layers, kv_prefix):
            x, (k_n, v_n) = layer.forward_incremental(
                x, (k_p, v_p), attn_mask=attn_mask,
            )
            if extend_cache:
                new_kv.append(
                    (torch.cat([k_p, k_n], dim=2), torch.cat([v_p, v_n], dim=2)),
                )
        return x, new_kv

    @staticmethod
    def _unblocked_cands(
        pointer_masks: torch.Tensor | None,
        B: int,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Candidate columns still selectable somewhere in the batch → indices.

        ``pointer_masks`` is the ``(B, K_blk)`` int64 block list (right-padded
        with -1) that ``_combined_ptr_logits`` turns into -inf pointer columns.
        A column blocked for EVERY row can never be pointed at, so its
        (type, candidate) mode branch is unreachable and need not be decoded.
        Returns all ``K`` indices when nothing is blocked.
        """
        if K == 0:
            return torch.zeros(0, dtype=torch.int64, device=device)
        if pointer_masks is None or pointer_masks.numel() == 0:
            return torch.arange(K, device=device)
        pm = pointer_masks.view(B, -1) if pointer_masks.dim() != 2 else pointer_masks
        blocked = torch.zeros(B, K, dtype=torch.bool, device=device)
        valid = (pm >= 0) & (pm < K)
        if bool(valid.any()):
            rows = torch.arange(B, device=device).unsqueeze(1).expand_as(pm)[valid]
            blocked[rows, pm[valid]] = True
        return torch.nonzero(~blocked.all(dim=0), as_tuple=False).flatten()

    def _pointer_logits(
        self,
        h_query: torch.Tensor,  # (B, d)
        H_state: torch.Tensor,  # (B, L, d)
        indices: torch.Tensor,  # (B, P) int64, -1 = pad
    ) -> torch.Tensor:
        """Dot-product pointer scoring with -inf masking on -1 indices.

        Returns:
            ``(B, P)`` float logits. Empty pool (P == 0) returns ``(B, 0)``.
        """
        B, P = indices.shape
        if P == 0:
            return torch.empty(B, 0, device=h_query.device, dtype=h_query.dtype)
        d = H_state.size(-1)
        safe_idx = indices.clamp(min=0).unsqueeze(-1).expand(B, P, d)  # (B, P, d)
        pts = torch.gather(H_state, 1, safe_idx)  # (B, P, d)
        raw = (h_query.unsqueeze(1) @ pts.transpose(1, 2)).squeeze(1) / self.scale
        # Clip BEFORE masking so the -inf positions stay -inf.
        logits = self.clip_C * torch.tanh(raw)
        logits = logits.masked_fill(indices == -1, float("-inf"))
        return logits

    def _combined_ptr_logits(
        self,
        h_at: torch.Tensor,  # (B, d)
        H_state: torch.Tensor,  # (B, L, d)
        net_indices: torch.Tensor,  # (B, M)
        cand_indices: torch.Tensor,  # (B, N)
        is_net_select: torch.Tensor,  # (B,) bool
        cand_block_idx: torch.Tensor | None = None,  # (B, K) int64, -1 pad
        net_valid_mask: torch.Tensor | None = None,  # (B, M_mask) bool
        row_block_idx: torch.Tensor | None = None,  # (B, K) int64, -1 pad
        row_block_gate: torch.Tensor | None = None,  # (B,) bool
    ) -> torch.Tensor:
        """Combine net and cand pointer logits into a single ``(B, max(M, N))`` tensor.

        For ``is_net_select`` batches, net logits are used; otherwise cand logits.
        Padded slots are filled with -inf.

        Args:
            cand_block_idx: Optional ``(B, K)`` int64 tensor; every entry
                ``>= 0`` is a column in ``cand_logits`` to set to ``-inf``
                before combining. Right-padded with ``-1``. Used by
                :class:`KiCadRLWrapper` to block the start_route
                ``(x, y)`` across **all layers** — neither make_line nor
                make_via nor a follow-up start_route can re-pick the
                same point. Only affects cand-based rows;
                ``is_net_select`` rows are unchanged.

                A ``(B,)`` tensor is also accepted (auto-promoted to
                ``(B, 1)``).
        """
        net_logits = self._pointer_logits(h_at, H_state, net_indices)   # (B, M)
        cand_logits = self._pointer_logits(h_at, H_state, cand_indices)  # (B, N)

        # Per-row net validity mask (policy-driven net selection): disallow
        # already-routed nets by setting their logits to -inf. Only affects
        # is_net_select rows (the net/cand where() below preserves cand rows).
        # Accepts a mask shorter/longer than M; we align to M by truncate/pad.
        if net_valid_mask is not None and net_indices.size(1) > 0:
            M = net_indices.size(1)
            nvm = net_valid_mask
            if nvm.size(1) < M:
                pad = torch.zeros(
                    nvm.size(0), M - nvm.size(1),
                    dtype=torch.bool, device=nvm.device,
                )
                nvm = torch.cat([nvm, pad], dim=1)
            elif nvm.size(1) > M:
                nvm = nvm[:, :M]
            net_logits = net_logits.masked_fill(~nvm, float("-inf"))

        # Per-ROW candidate block, gated by the row's action type. Unlike
        # cand_block_idx (which applies to every cand row), this one is consumed
        # only where ``row_block_gate`` is set — the caller decides per row.
        # Used for the make_line off-layer rule: make_line cannot change layer, so
        # a candidate on another layer routes to the SAME (x, y) on the current
        # one and duplicates the same-layer candidate; make_via changes layer by
        # design and is left unrestricted. The first (same-layer) candidate is
        # never blocked and directional candidates are generated at the head's
        # layer, so a gated row cannot go all-(-inf) from this.
        if (row_block_idx is not None and row_block_gate is not None
                and cand_logits.size(1) > 0 and row_block_idx.numel()):
            # reshape, not view: rep()/expand can hand us a non-contiguous
            # tensor and view would raise on it.
            rb = row_block_idx.reshape(cand_logits.size(0), -1)
            N = cand_logits.size(1)
            valid = (rb >= 0) & (rb < N) & row_block_gate.reshape(-1, 1)
            if bool(valid.any()):
                cand_logits = cand_logits.clone()
                rows = (torch.arange(cand_logits.size(0), device=cand_logits.device)
                        .unsqueeze(1).expand_as(rb)[valid])
                cand_logits[rows, rb[valid]] = float("-inf")

        # Same-point masking: zero out every (row, col) where col >= 0.
        # Must happen BEFORE the net/cand where() so net_select rows stay
        # untouched. Accepts (B, K) or (B,) shape.
        if cand_block_idx is not None and cand_logits.size(1) > 0:
            cbi = cand_block_idx
            if cbi.dim() == 1:
                cbi = cbi.unsqueeze(1)  # (B,) → (B, 1)
            if cbi.size(1) > 0:
                valid = cbi >= 0  # (B, K) bool
                if valid.any():
                    cand_logits = cand_logits.clone()
                    N = cand_logits.size(1)
                    # Build flat (row, col) index lists for each valid entry.
                    rows_grid = (
                        torch.arange(cbi.size(0), device=cbi.device)
                        .unsqueeze(1).expand_as(cbi)
                    )
                    rows = rows_grid[valid]
                    cols = cbi[valid].clamp(min=0, max=N - 1)
                    cand_logits[rows, cols] = float("-inf")

        M = net_indices.size(1)
        N = cand_indices.size(1)
        K = max(M, N, 1)  # guard against M==N==0

        if M < K:
            net_logits = F.pad(net_logits, (0, K - M), value=float("-inf"))
        if N < K:
            cand_logits = F.pad(cand_logits, (0, K - N), value=float("-inf"))

        is_net = is_net_select.unsqueeze(-1).expand(-1, K)
        return torch.where(is_net, net_logits, cand_logits)

    def _compute_value(self, h_summary: torch.Tensor) -> torch.Tensor:
        """Compute state-value from a (B, d_model) state summary tensor.

        Standard PPO: value loss flows back through ``critic_head`` *and* the
        transformer backbone, exactly like SB3 / CleanRL / RLHF (Trlx / TRL).
        The shared backbone is shaped jointly by policy and value losses
        (balanced by ``vf_coef`` in the trainer).

        Args:
            h_summary: ``(B, d_model)`` — the hidden at the VAL position
                ``H_state[arange_B, seq_lens - 2]``, which has bidirectionally
                attended over the entire state via the 2-zone prefix-LM mask.

        Returns:
            ``(B,)`` float values. If ``use_critic`` is False, returns a
            zero tensor with no grad.
        """
        if not self.use_critic:
            return torch.zeros(
                h_summary.size(0), device=h_summary.device, dtype=h_summary.dtype,
            )
        if self.detach_critic:
            h_summary = h_summary.detach()
        return self.critic_head(h_summary).squeeze(-1)

    @staticmethod
    def _extract_scalar_bounds(
        tok_out: TokenizerOutput,
    ) -> int:
        """Extract scalar ``n_state_max`` for the batch.

        ``seq_lens`` may differ per row because dynamic + cand token counts
        depend on env state. We pad to ``max(seq_lens)`` and let the
        ``key_padding_mask`` handle the gap, so the 2-zone mask uses
        ``n_state_max`` as a single scalar state/action boundary.

        Per-row "real" SOD positions (``H[:, seq_lens - 1]``) must be
        extracted by callers using ``arange_B`` / ``seq_lens`` indexing —
        the scalar returned here is only the upper bound.
        """
        # token_embeddings is padded to exactly max(seq_lens) (tokenizer
        # clamps seq_lens to that width), so the shape read IS the max —
        # without a GPU->CPU sync per forward.
        return tok_out.token_embeddings.size(1)

    # ------------------------------------------------------------------
    # Shared core — pieces used by all three public forwards (act/evaluate/
    # factored); whether no_grad applies is decided by each public forward
    # (evaluate needs grad).
    # ------------------------------------------------------------------
    def _encode_state(
        self,
        obs_list: list[dict],
        walked: dict | None = None,
        *,
        action_masks: torch.Tensor | None = None,
        return_cache: bool = True,
    ) -> _StateEnc:
        """Pass-0 shared core: tokenize → state-only pass → VAL critic +
        SOD action-type logits (through mask application).

        When ``return_cache=True``, also returns the state pass's per-layer
        K/V cache so the following action-token decode
        (``_decode_appended``) can reuse it.
        """
        tok_out: TokenizerOutput = self.tokenizer(
            obs_list, action_type_weight=self.action_type_head.weight,
            walked=walked,
        )
        state_embs = tok_out.token_embeddings          # (B, L_s, d)
        kpm = tok_out.key_padding_mask                 # (B, L_s)
        n_state_max = self._extract_scalar_bounds(tok_out)
        B = state_embs.size(0)
        assert state_embs.size(1) == n_state_max, (
            f"expected L_s ({state_embs.size(1)}) == n_state_max ({n_state_max})"
        )
        device = state_embs.device
        arange_B = torch.arange(B, device=device)
        seq_lens = tok_out.seq_lens.to(device)         # (B,)

        cache = None
        if return_cache:
            H_state, cache = self._run_transformer(
                state_embs, n_state_max, kpm, slot_ids=tok_out.slot_ids,
                return_cache=True,
            )
        else:
            H_state = self._run_transformer(
                state_embs, n_state_max, kpm, slot_ids=tok_out.slot_ids,
            )

        # Per-row SOD/VAL: row i's real state ends at seq_lens[i] —
        # SOD = seq_lens-1 (policy), VAL = seq_lens-2 (critic); padding is
        # blocked by kpm.
        h_SOD = H_state[arange_B, seq_lens - 1]        # (B, d)
        values = self._compute_value(H_state[arange_B, seq_lens - 2])  # (B,)

        at_raw = h_SOD @ self.action_type_head.weight.T / self.scale  # (B, 6)
        at_logits = self.clip_C * torch.tanh(at_raw)
        if action_masks is not None:
            at_logits = at_logits.masked_fill(~action_masks, float("-inf"))
        return _StateEnc(tok_out, H_state, cache, values, at_logits,
                         B, device, arange_B, seq_lens, n_state_max)

    def _at_token(self, action_type_ids: torch.Tensor) -> torch.Tensor:
        """action-type embedding + action-zone pos-emb[0] → the at token ``(B, d)``."""
        return self.action_type_head(action_type_ids) + self.action_pos_emb[0]

    def _decode_one_token(
        self, cache: _DecodeCache, tok: torch.Tensor,
    ) -> tuple[torch.Tensor, _DecodeCache]:
        """Append one action token to the cached state pass and return its final hidden."""
        h_new, cache = self._decode_appended(cache, tok.unsqueeze(1))
        return h_new[:, 0], cache

    def _point_token(
        self,
        H_state: torch.Tensor,
        cand_indices: torch.Tensor,
        pointer_idx: torch.Tensor,
        arange_B: torch.Tensor,
    ) -> torch.Tensor:
        """Re-emit the chosen-pointer state hidden + pos-emb[1] → the point
        token ``(B, d)``.

        When the cand pool is empty (idle), uses a position-0 dummy — the
        corresponding actions (make_line/via) are mask-blocked in that
        state, so the value is unused.
        """
        if cand_indices.size(1) == 0:
            chosen_cand_pos = torch.zeros(
                arange_B.size(0), dtype=torch.long, device=H_state.device,
            )
        else:
            safe_ptr_idx = pointer_idx.clamp(
                min=0, max=cand_indices.size(1) - 1,
            )
            chosen_cand_pos = torch.gather(
                cand_indices, 1, safe_ptr_idx.unsqueeze(1),
            ).squeeze(1).clamp(min=0)  # (B,)
        return H_state[arange_B, chosen_cand_pos] + self.action_pos_emb[1]

    def _mode_logits(
        self, h: torch.Tensor, mode_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Routing-mode logits (weight-tied to the vocab embedding) + an optional mask."""
        logits = h @ self.tokenizer.vocab.routing_mode_embed.weight.T / self.scale
        if mode_mask is not None:
            logits = logits.masked_fill(~mode_mask, float("-inf"))
        return logits

    # ------------------------------------------------------------------
    # Sampling (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act_and_value(
        self,
        obs_list: list[dict],
        action_masks: torch.Tensor | None = None,
        deterministic: bool = False,
        pointer_masks: torch.Tensor | None = None,
        offlayer_masks: torch.Tensor | None = None,
        mode_mask: torch.Tensor | None = None,
        net_valid_mask: torch.Tensor | None = None,
        allow_net_select_lp: bool = False,
        walked: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions autoregressively and compute state values.

        Args:
            obs_list: List of JSON observation dicts.
            walked: Optional pre-walked tokenizer dict for this batch
                (``BatchedStateTokenizer.merge_walked`` output) — skips the
                CPU walk. Must correspond 1:1 (same order) to ``obs_list``.
            action_masks: ``(B, 6)`` bool, True = valid action_type.
            deterministic: If True, argmax instead of sampling.
            pointer_masks: Optional ``(B, K)`` int64 tensor (right-padded
                with ``-1``). Every non-negative entry is a cand index
                whose pointer logit is set to ``-inf``. This is how
                :class:`KiCadRLWrapper` implements strict same-
                point masking: after ``start_route(x, y, l)`` every cand
                with the same ``(x, y)`` (any layer) is blocked, so
                make_line / make_via / next start_route cannot re-pick
                it. The trainer collects
                :meth:`KiCadRLWrapper.start_route_pointer_indices`
                per env and stacks the result. A ``(B,)`` tensor is also
                accepted (auto-promoted to ``(B, 1)``).

        Returns:
            actions: ``(B, 3)`` int64 — [action_type, pointer_idx, routing_mode].
                Unused slots are set to -1.
            log_probs: ``(B,)`` float — joint log-probability of the action.
            values: ``(B,)`` float — critic estimate of the state value.
                Zero tensor (no grad) if ``use_critic`` is False.
        """
        enc = self._encode_state(
            obs_list, walked, action_masks=action_masks, return_cache=True,
        )
        H_state, cache = enc.H_state, enc.cache
        net_indices = enc.tok_out.net_indices          # (B, M)
        cand_indices = enc.tok_out.cand_indices        # (B, N)
        device, arange_B = enc.device, enc.arange_B
        values = enc.values

        at_dist = Categorical(logits=enc.at_logits)
        action_type = (
            enc.at_logits.argmax(-1) if deterministic else at_dist.sample()
        )
        log_prob = at_dist.log_prob(action_type)

        # === Pass 1 (incremental): state + at_tok ===
        # action_pos_emb[0] gives the action_type slot a learned absolute
        # position so the action zone has meaningful order without RoPE.
        h_at, cache = self._decode_one_token(cache, self._at_token(action_type))

        is_net_select = action_type == ACT_NET_SELECT  # (B,)
        is_finish = action_type == ACT_FINISH          # (B,)
        slot_usage_dev = SLOT_USAGE.to(device)
        needs_ptr = slot_usage_dev[action_type, 0]     # (B,) bool
        needs_mode = slot_usage_dev[action_type, 1]    # (B,) bool

        # Combined pointer logits (net for net_select, cand otherwise).
        ptr_logits = self._combined_ptr_logits(
            h_at, H_state, net_indices, cand_indices, is_net_select,
            cand_block_idx=pointer_masks,
            net_valid_mask=net_valid_mask,
            row_block_idx=offlayer_masks,
            row_block_gate=(action_type == ACT_MAKE_LINE),
        )  # (B, K)
        # Diagnostic guard (not a fallback): on the normal path, a
        # non-net_select row's cand block cannot be all-(-inf) — while
        # routing, directional candidates always exist from geometry
        # generation, and the net-select-state pool includes the current
        # net's pad. If this occurs — regardless of whether the pointer
        # would be consumed — fail immediately with row context instead of
        # Categorical's uninformative ValueError; this is a loud guard, not
        # a silent mitigation that swallows an abnormal state.
        dead = torch.isinf(ptr_logits).all(dim=-1)
        if bool(dead.any()):
            from pcb_world.diag import guard_fail

            i = int(dead.nonzero()[0, 0])
            guard_fail(
                "act_dead_ptr_row",
                "act: all-(-inf) cand pointer row — "
                f"batch_row={i} action_type={int(action_type[i])} "
                f"needs_ptr={bool(needs_ptr[i])} "
                f"K={int(ptr_logits.shape[1])} "
                f"n_dead_rows={int(dead.sum())}",
                obs_list=obs_list,
                action_type=action_type,
                ptr_logits=ptr_logits,
                dead_rows=dead.nonzero(),
                pointer_masks=pointer_masks,
                offlayer_masks=offlayer_masks,
                net_valid_mask=net_valid_mask,
                action_masks=action_masks,
                mode_mask=mode_mask,
                net_indices=net_indices,
                cand_indices=cand_indices,
            )
        ptr_dist = Categorical(logits=ptr_logits)
        pointer_idx = (
            ptr_logits.argmax(-1) if deterministic else ptr_dist.sample()
        )

        # Mode logits at h_at (used for finish).
        mode_logits_at = self._mode_logits(h_at, mode_mask)  # (B, 3)
        mode_at_dist = Categorical(logits=mode_logits_at)
        mode_at_pick = (
            mode_logits_at.argmax(-1) if deterministic else mode_at_dist.sample()
        )

        # When mode_mask is provided (masking active), net_select's pointer
        # is external/random by default — exclude it from log-prob unless
        # allow_net_select_lp is set (policy-driven net selection).
        if mode_mask is not None and not allow_net_select_lp:
            needs_ptr = needs_ptr & ~is_net_select

        log_prob = log_prob + needs_ptr.float() * ptr_dist.log_prob(pointer_idx)
        log_prob = log_prob + is_finish.float() * mode_at_dist.log_prob(mode_at_pick)

        # === Pass 2 (incremental): state + at_tok + point_tok ===
        # pointer_idx for cand-based actions indexes cand_indices[:, pointer_idx].
        point_tok = self._point_token(H_state, cand_indices, pointer_idx, arange_B)
        h_pt, cache = self._decode_one_token(cache, point_tok)

        mode_logits_pt = self._mode_logits(h_pt, mode_mask)  # (B, 3)
        mode_pt_dist = Categorical(logits=mode_logits_pt)
        mode_pt_pick = (
            mode_logits_pt.argmax(-1) if deterministic else mode_pt_dist.sample()
        )

        needs_pt_mode = needs_ptr & needs_mode  # make_line/make_via
        log_prob = log_prob + needs_pt_mode.float() * mode_pt_dist.log_prob(
            mode_pt_pick,
        )

        # Final action assembly: unused slots → -1.
        neg_ones = torch.full_like(action_type, -1)
        routing_mode = torch.where(
            needs_pt_mode,
            mode_pt_pick,
            torch.where(is_finish, mode_at_pick, neg_ones),
        )
        final_pointer_idx = torch.where(needs_ptr, pointer_idx, neg_ones)

        actions = torch.stack(
            [action_type, final_pointer_idx, routing_mode], dim=-1,
        )  # (B, 3) int64
        return actions, log_prob, values

    def act(
        self,
        obs_list: list[dict],
        action_masks: torch.Tensor | None = None,
        deterministic: bool = False,
        pointer_masks: torch.Tensor | None = None,
        offlayer_masks: torch.Tensor | None = None,
        mode_mask: torch.Tensor | None = None,
        net_valid_mask: torch.Tensor | None = None,
        allow_net_select_lp: bool = False,
        walked: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thin wrapper around :meth:`act_and_value`.

        Returns ``(actions, log_probs)`` — drops the value tensor.
        """
        actions, log_probs, _ = self.act_and_value(
            obs_list,
            action_masks=action_masks,
            deterministic=deterministic,
            pointer_masks=pointer_masks,
            offlayer_masks=offlayer_masks,
            mode_mask=mode_mask,
            net_valid_mask=net_valid_mask,
            allow_net_select_lp=allow_net_select_lp,
            walked=walked,
        )
        return actions, log_probs

    # ------------------------------------------------------------------
    # Teacher-forced evaluation (training)
    # ------------------------------------------------------------------
    def evaluate_actions_and_value(
        self,
        obs_list: list[dict],
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        pointer_masks: torch.Tensor | None = None,
        offlayer_masks: torch.Tensor | None = None,
        mode_mask: torch.Tensor | None = None,
        net_valid_mask: torch.Tensor | None = None,
        allow_net_select_lp: bool = False,
        walked: dict | None = None,
        entropy_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score actions under the current policy and compute state values.

        Args:
            obs_list: List of JSON observation dicts (same order as actions).
            walked: Optional pre-walked tokenizer dict for this minibatch
                (``BatchedStateTokenizer.gather_walked``/``merge_walked``
                output) — skips the CPU walk and ignores ``obs_list``. Must
                correspond 1:1 (same order) to the minibatch rows.
            actions: ``(B, 3)`` int64 from a prior :meth:`act` call.
            action_masks: ``(B, 6)`` bool, True = valid action_type.
            pointer_masks: Optional ``(B, K)`` int64 tensor — same semantics
                as in :meth:`act_and_value`. Must match the value used
                during rollout so the teacher-forced log_prob sees the
                same (masked) distribution.
            entropy_norm: If True, divide each sample's joint entropy by its
                max achievable entropy — sum of ``ln(N_valid)`` over the same
                ``needs_*``-weighted heads the entropy sums — yielding a
                [0, 1] relative entropy invariant to action-space size
                (``--entropy-norm``). log_probs/values are unaffected.

        Returns:
            log_probs: ``(B,)`` float — joint log-probability of actions.
            entropy:   ``(B,)`` float — joint entropy of the policy distribution.
            values:    ``(B,)`` float — critic estimate of the state value.
                Zero tensor (no grad) if ``use_critic`` is False.
        """
        enc = self._encode_state(
            obs_list, walked, action_masks=action_masks, return_cache=True,
        )
        H_state = enc.H_state
        net_indices = enc.tok_out.net_indices
        cand_indices = enc.tok_out.cand_indices
        device, arange_B = enc.device, enc.arange_B
        values = enc.values

        action_type = actions[:, 0].long()
        action_ptr = actions[:, 1].long()
        action_mode = actions[:, 2].long()

        # === Pass 2 (incremental): state + at_tok + point_tok ===
        # Teacher-forced: both action tokens are known upfront, so decode
        # them in ONE incremental call (at_tok first — point_tok attends
        # to it causally). SOD is a state-zone hidden, identical in
        # H_state (state cannot attend to appended action tokens).
        # point_tok: only meaningful for make_line/make_via; others use dummy.
        point_tok = self._point_token(H_state, cand_indices, action_ptr, arange_B)
        at_tok = self._at_token(action_type)
        h_new, _ = self._decode_appended(
            enc.cache, torch.stack([at_tok, point_tok], dim=1),
            extend_cache=False,
        )
        h_at = h_new[:, 0]
        h_pt = h_new[:, 1]

        # --- action_type logits (SOD; masking already applied in the shared core) ---
        at_dist = Categorical(logits=enc.at_logits)
        log_prob = at_dist.log_prob(action_type)
        entropy = at_dist.entropy()

        # --- pointer logits (combined) ---
        is_net_select = action_type == ACT_NET_SELECT
        ptr_logits = self._combined_ptr_logits(
            h_at, H_state, net_indices, cand_indices, is_net_select,
            cand_block_idx=pointer_masks,
            net_valid_mask=net_valid_mask,
            row_block_idx=offlayer_masks,
            row_block_gate=(action_type == ACT_MAKE_LINE),
        )
        # Diagnostic guard — same as the act path: an all-(-inf) cand block
        # on a non-net_select row is an abnormal state regardless of pointer
        # consumption → fail immediately with context.
        dead = torch.isinf(ptr_logits).all(dim=-1)
        if bool(dead.any()):
            from pcb_world.diag import guard_fail

            i = int(dead.nonzero()[0, 0])
            guard_fail(  # grads are live here — the dump detaches
                "evaluate_dead_ptr_row",
                "evaluate: all-(-inf) cand pointer row — "
                f"batch_row={i} action_type={int(action_type[i])} "
                f"K={int(ptr_logits.shape[1])} "
                f"n_dead_rows={int(dead.sum())}",
                obs_list=obs_list,
                actions=actions,
                action_type=action_type,
                ptr_logits=ptr_logits,
                dead_rows=dead.nonzero(),
                pointer_masks=pointer_masks,
                offlayer_masks=offlayer_masks,
                net_valid_mask=net_valid_mask,
                action_masks=action_masks,
                mode_mask=mode_mask,
                net_indices=net_indices,
                cand_indices=cand_indices,
            )
        ptr_dist = Categorical(logits=ptr_logits)

        # --- mode logits (both finish and make_line/via branches) ---
        mode_at_logits = self._mode_logits(h_at, mode_mask)
        mode_pt_logits = self._mode_logits(h_pt, mode_mask)
        mode_at_dist = Categorical(logits=mode_at_logits)
        mode_pt_dist = Categorical(logits=mode_pt_logits)

        # --- SLOT_USAGE weighting ---
        slot_usage_dev = SLOT_USAGE.to(device)
        needs_ptr = slot_usage_dev[action_type, 0].float()
        needs_mode = slot_usage_dev[action_type, 1].float()
        is_finish = (action_type == ACT_FINISH).float()

        # When mode_mask is provided (masking active), net_select's pointer
        # is external/random by default — exclude it from log-prob unless
        # allow_net_select_lp is set (policy-driven net selection).
        if mode_mask is not None and not allow_net_select_lp:
            needs_ptr = needs_ptr * (~(action_type == ACT_NET_SELECT)).float()

        # Pointer contribution (log_prob & entropy).
        #
        # For actions that don't use the pointer slot (net_end / finish)
        # ``action_ptr`` is ``-1`` and clamps to ``0``. With same-point
        # masking (``pointer_masks``) an arbitrary cand index may be
        # ``-inf`` in ``ptr_logits``, so ``ptr_dist.log_prob(0)`` can be
        # ``-inf``. The plain ``needs_ptr * log_prob`` form then produces
        # ``0.0 * -inf = NaN``, which cascades through the loss and
        # corrupts every weight on the next optimizer step. Guard with
        # ``torch.where`` so pointerless rows contribute exactly zero.
        safe_ptr_idx = action_ptr.clamp(min=0)
        ptr_log_prob = ptr_dist.log_prob(safe_ptr_idx)
        ptr_log_prob = torch.where(
            needs_ptr > 0, ptr_log_prob, torch.zeros_like(ptr_log_prob),
        )
        log_prob = log_prob + ptr_log_prob
        entropy = entropy + needs_ptr * ptr_dist.entropy()

        # Mode contribution (finish → h_at, make_line/via → h_pt).
        # Use torch.where to prevent 0 * -inf = NaN when mode_mask is
        # active and one branch's log_prob is -inf for the sampled mode.
        safe_mode = action_mode.clamp(min=0)
        mode_at_lp = mode_at_dist.log_prob(safe_mode)
        mode_pt_lp = mode_pt_dist.log_prob(safe_mode)
        mode_log_prob = torch.where(
            is_finish > 0, mode_at_lp, mode_pt_lp,
        )
        mode_log_prob = torch.where(
            needs_mode > 0, mode_log_prob, torch.zeros_like(mode_log_prob),
        )
        mode_entropy = torch.where(
            is_finish > 0, mode_at_dist.entropy(), mode_pt_dist.entropy(),
        )
        mode_entropy = torch.where(
            needs_mode > 0, mode_entropy, torch.zeros_like(mode_entropy),
        )
        log_prob = log_prob + mode_log_prob
        entropy = entropy + mode_entropy

        if entropy_norm:
            # Per-sample max achievable joint entropy: ln(N_valid) per head,
            # combined with exactly the needs_*/is_finish weights the entropy
            # sums above use. -inf-masked entries drop out of the finite
            # count. clamp(min=1) guards the log; a fully deterministic row
            # (max_ent == 0) has entropy 0, so 0 / eps -> 0.
            def _ln_nvalid(logits: torch.Tensor) -> torch.Tensor:
                return torch.log(
                    torch.isfinite(logits).sum(-1).clamp(min=1).float()
                )

            max_ent = _ln_nvalid(enc.at_logits)
            max_ent = max_ent + needs_ptr * _ln_nvalid(ptr_logits)
            mode_ln = torch.where(
                is_finish > 0,
                _ln_nvalid(mode_at_logits), _ln_nvalid(mode_pt_logits),
            )
            max_ent = max_ent + needs_mode * mode_ln
            # A deterministic row (max_ent == 0) has entropy 0 as well, but
            # an eps-clamped division amplifies that row's residual bf16 ~ε
            # gradient by 1/eps → grad inf → clip_grad_norm's inf*0=NaN
            # chain corrupts the weights. max_ent is the log of an integer
            # count, so it is either 0 or ≥ ln2: only the zero rows get
            # denominator 1, which removes the amplification channel. The
            # division runs in fp32.
            denom = torch.where(
                max_ent > 0, max_ent, torch.ones_like(max_ent),
            )
            entropy = entropy.float() / denom

        return log_prob, entropy, values

    # ------------------------------------------------------------------
    # Factored prior (state encoded once)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def factored_action_logits(
        self,
        obs_list: list[dict],
        action_masks: torch.Tensor | None = None,
        pointer_masks: torch.Tensor | None = None,
        offlayer_masks: torch.Tensor | None = None,
        mode_mask: torch.Tensor | None = None,
        net_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive factor logits with the state encoded ONCE.

        Mirrors Pass-0/Pass-1/2 of :meth:`act_and_value` but vectorized over ALL
        action types (instead of one sampled type), so a caller can build the
        full sampler-exact ``P(at)·P(ptr|at)·P(mode|·)`` prior over every candidate
        WITHOUT a per-candidate forward — the pointer logits already span all
        pointers, and the mode is read off the POST-pointer hidden per candidate.

        Returns (masks pre-applied; ready for ``log_softmax``):
          ``at_logits``      (B, T)      action-type logits
          ``ptr_logits``     (B, T, K)   pointer logits per type (from ``h_at``)
          ``mode_at_logits`` (B, T, 3)   PRE-pointer mode logits (h_at) — EXACT for
              ``finish`` (its sampler reads h_at); the caller uses these for finish.
          ``mode_pt_logits`` (B, T, K, 3) POST-pointer mode logits per (type,
              candidate) — sampler-EXACT for make_line/make_via, which the
              caller uses for them. Non-pointer/mode types fall back to the
              ``mode_at`` broadcast (never indexed by a real pointer).
          ``values``         (B,)        critic value

        Cost: ONE state pass + ONE incremental decode of ``T + P·Kk`` appended
        tokens over the cached prefix — Θ(L²d) + Θ((T+P·Kk)·L·d), versus the
        ``(1 + T + P·K)·Θ(L²d)`` a per-branch full rerun costs. ``P``/``Kk``
        count only the branches a legal action can read (see the pruning
        below), so a non-routing node decodes ``T`` tokens instead of
        ``T + 2K`` — measured 3x (K=32) to 74x (K=512) on that decode.

        Masked-out (type, candidate) slots of ``mode_pt_logits`` hold the
        ``mode_at`` broadcast rather than a decoded hidden — they are -inf in
        ``at_logits``/``ptr_logits`` under the same masks, so no reachable
        action reads them. The KEPT rows are mathematically unchanged (each
        branch attends only to the prefix, its own at token and itself), but not
        bit-identical: dropping rows shortens the SDPA key axis, so the softmax /
        GEMM reduction order shifts. Measured residual on the kept rows is fp32
        epsilon (~5e-7) and shrinks to ~7e-16 in fp64, i.e. rounding only — at
        tokens come out bit-identical.
        """
        # Pass 0 (shared core): state only → at_logits + critic value. The K/V cache
        # is what makes the branch decode below cheap — the state prefix is
        # encoded once and every action branch attends to it without recompute.
        enc = self._encode_state(
            obs_list, action_masks=action_masks, return_cache=True,
        )
        tok_out = enc.tok_out
        net_indices = tok_out.net_indices
        cand_indices = tok_out.cand_indices
        H_state = enc.H_state
        at_logits = enc.at_logits                        # (B, T)
        values = enc.values
        B, _, d = H_state.shape
        device = enc.device
        T = self.action_type_head.weight.size(0)

        # Pointer+mode types (make_line/make_via): their sampler reads the mode
        # off the POST-pointer hidden, so they need one branch per candidate.
        # PRUNED to the branches a legal action can actually read: a (type,
        # candidate) branch exists ONLY to supply mode_pt_logits[type, cand],
        # and the caller scores an action as P(at)·P(ptr|at)·P(mode|·). A type
        # masked out of at_logits, or a candidate masked out of ptr_logits, is
        # -inf there — its joint is 0 and it never enters the legal set, so its
        # branch is pure waste. The dropped slots keep the mode_at broadcast
        # fallback below (unreachable under the same masks). Biggest win: at a
        # non-routing node (net_select / start_route) both pointer+mode types
        # are masked, so the whole P·K decode disappears.
        pt_types = (SLOT_USAGE[:, 0] & SLOT_USAGE[:, 1]).to(device)
        if action_masks is not None:
            # union over the batch — a branch survives if ANY row can use it
            pt_types = pt_types & action_masks.any(dim=0)
        pt_type_ids = torch.nonzero(pt_types, as_tuple=False).flatten()   # (P,)
        P = int(pt_type_ids.numel())
        K = cand_indices.size(1)
        cand_ids = self._unblocked_cands(pointer_masks, B, K, device)     # (Kk,)
        Kk = int(cand_ids.numel())
        n_pt = P * Kk

        # Pass 1+2 fused — ONE incremental decode over the cached state prefix.
        # Appended tokens (n_new = T + P·Kk), in order:
        #   [0, T)                 at_tok(t)      for every action type t
        #   [T, T + P·Kk)          point_tok(k)   for branch (p, k), p-outer
        # Each is an INDEPENDENT branch of the autoregressive action tree, so the
        # appended-mask is block-structured instead of causal: an at token sees
        # only the prefix + itself; a point token sees the prefix + its own type's
        # at token + itself — exactly the context a per-branch full rerun would
        # build ([state | at_tok(p) | point_tok(k)]). Cross-branch attention is
        # blocked, so the K/V prefix is shared instead of duplicated P·Kk times.
        type_ids = torch.arange(T, device=device)
        new_embs = self._at_token(type_ids).unsqueeze(0).expand(B, T, d)
        if n_pt:
            cand_pos = cand_indices.clamp(min=0).index_select(1, cand_ids)
            point_toks = torch.gather(
                H_state, 1, cand_pos.unsqueeze(-1).expand(B, Kk, d),
            ) + self.action_pos_emb[1]                                 # (B, Kk, d)
            new_embs = torch.cat(
                [new_embs,
                 point_toks.unsqueeze(1).expand(B, P, Kk, d).reshape(B, n_pt, d)],
                dim=1,
            )
        n_new = T + n_pt
        app_mask = torch.full((n_new, n_new), float("-inf"), device=device)
        app_mask.fill_diagonal_(0.0)                     # every token sees itself
        if n_pt:
            app_mask[
                torch.arange(T, n_new, device=device),
                pt_type_ids.repeat_interleave(Kk),       # branch (p, k) → at_tok(p)
            ] = 0.0
        h_new, _ = self._decode_appended(
            enc.cache, new_embs, appended_mask=app_mask, extend_cache=False,
        )

        h_at = h_new[:, :T].reshape(B * T, d)            # B-outer / T-inner
        h_pt = h_new[:, T:]                              # (B, P·Kk, d)

        def rep(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1).expand(B, T, *x.shape[1:]).reshape(B * T, *x.shape[1:])

        is_net_select = (type_ids == ACT_NET_SELECT).repeat(B)             # (B*T,)
        ptr_logits = self._combined_ptr_logits(
            h_at, rep(H_state), rep(net_indices), rep(cand_indices), is_net_select,
            cand_block_idx=(rep(pointer_masks) if pointer_masks is not None else None),
            net_valid_mask=(rep(net_valid_mask) if net_valid_mask is not None else None),
            row_block_idx=(rep(offlayer_masks) if offlayer_masks is not None else None),
            row_block_gate=(type_ids == ACT_MAKE_LINE).repeat(B),
        ).reshape(B, T, -1)

        mode_at = self._mode_logits(h_at, None).reshape(B, T, -1)
        if mode_mask is not None:
            mode_at = mode_at.masked_fill(~mode_mask.unsqueeze(1), float("-inf"))

        # POST-pointer mode for the pointer+mode types, read off the branch
        # hiddens decoded above. Fallback = the pre-pointer mode broadcast over
        # candidates, so non-pt types and the K axis stay defined.
        mode_pt_logits = mode_at.unsqueeze(2).expand(
            B, T, K, mode_at.size(-1)).clone()
        if n_pt:
            mode_pt = self._mode_logits(
                h_pt.reshape(B * n_pt, d), None,
            ).reshape(B, P, Kk, -1)
            if mode_mask is not None:
                mode_pt = mode_pt.masked_fill(
                    ~mode_mask.view(B, 1, 1, -1), float("-inf"))
            # scatter the decoded branches back onto the full (T, K) grid; the
            # pruned slots keep the mode_at fallback set above
            ti = pt_type_ids.view(P, 1).expand(P, Kk).reshape(-1)
            ci = cand_ids.view(1, Kk).expand(P, Kk).reshape(-1)
            mode_pt_logits[:, ti, ci] = mode_pt.reshape(B, P * Kk, -1)

        return {
            "at_logits": at_logits, "ptr_logits": ptr_logits,
            "mode_at_logits": mode_at, "mode_pt_logits": mode_pt_logits,
            "values": values,
        }

    def evaluate(
        self,
        obs_list: list[dict],
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        pointer_masks: torch.Tensor | None = None,
        offlayer_masks: torch.Tensor | None = None,
        mode_mask: torch.Tensor | None = None,
        net_valid_mask: torch.Tensor | None = None,
        allow_net_select_lp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thin wrapper around :meth:`evaluate_actions_and_value`.

        Returns ``(log_probs, entropy)`` — drops the value tensor.
        """
        log_probs, entropy, _ = self.evaluate_actions_and_value(
            obs_list,
            actions,
            action_masks=action_masks,
            pointer_masks=pointer_masks,
            offlayer_masks=offlayer_masks,
            mode_mask=mode_mask,
            net_valid_mask=net_valid_mask,
            allow_net_select_lp=allow_net_select_lp,
        )
        return log_probs, entropy
