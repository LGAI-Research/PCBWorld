"""Transformer building blocks for the PCB routing policy (model layer).

Pure network architecture — no RL/action semantics. The policy layer
(:mod:`methods.rl_agent.models.v1.net`, ``KiCadRLModel``) assembles these
blocks with the state tokenizer and pointer/value heads and owns everything
RL-specific (act / log-prob / entropy / critic).

Components: ReZero, SameNetBias, MultiHeadAttention (SDPA, or flex_attention
over a key-padding BlockMask), GatedTransformerLayer, build_2zone_mask,
combine_masks, padding_attn_mask, flex_padding_block_mask, same_net_score_mod,
init_weights.

Design reference: DECODER_ONLY_DESIGN §2.4-§2.9 (internal design note).
"""

from __future__ import annotations

import math
import os

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

# --- SDP backend override (activated by FORCE_SDP_EFFICIENT=1 env var) ---
if os.environ.get("FORCE_SDP_EFFICIENT"):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


# ---------------------------------------------------------------------------
# 2-Zone Attention Mask (State bidir / Action causal)
# ---------------------------------------------------------------------------
def build_2zone_mask(
    n_state: int,
    seq_len: int,
) -> torch.Tensor:
    """Build prefix-LM style 2-zone attention mask.

    Zone boundaries::

        State:   [0, n_state)         BOARD/EDGE/NET/PAD/TRACK/VIA/RAT/DRC
                                      /HEAD/CAND/ACTION_HISTORY/VAL/SOD
        Action:  [n_state, seq_len)   action tokens (at / pt / mode)

    Rules::

                      ┌── State ──┬── Action ──┐
        State         │  bidir    │   -inf     │
        Action        │  attend   │  causal    │
                      └───────────┴────────────┘

    Net scoping (same-net vs different-net) is handled implicitly by the
    orthogonal slot embedding in ``TokenVocabulary``; board-global tokens
    (BOARD / EDGE) carry slot=-1 and therefore receive no slot embedding.

    Args:
        n_state: Number of state tokens.
        seq_len: Total sequence length (state + action).

    Returns:
        ``(seq_len, seq_len)`` float tensor, 0.0 = attend, -inf = block.
    """
    mask = torch.zeros(seq_len, seq_len)

    # State zone cannot see action
    if n_state < seq_len:
        mask[:n_state, n_state:] = float("-inf")

    # Action zone: causal within itself
    for i in range(n_state, seq_len):
        if i + 1 < seq_len:
            mask[i, i + 1 : seq_len] = float("-inf")

    return mask


def combine_masks(
    zone_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Combine 2-zone mask with key-padding mask for SDPA.

    Args:
        zone_mask: ``(seq_len, seq_len)`` from :func:`build_2zone_mask`.
        key_padding_mask: ``(B, seq_len)`` bool, True = padded.

    Returns:
        ``(B, 1, seq_len, seq_len)`` float tensor for SDPA ``attn_mask``.
    """
    # zone_mask → (1, 1, L, L)
    combined = zone_mask.unsqueeze(0).unsqueeze(0)

    # key_padding_mask → (B, 1, 1, L): -inf for padded keys
    pad_mask = torch.where(
        key_padding_mask.unsqueeze(1).unsqueeze(2),
        torch.tensor(float("-inf"), device=key_padding_mask.device),
        torch.tensor(0.0, device=key_padding_mask.device),
    )

    return combined + pad_mask  # (B, 1, L, L) via broadcast


def padding_attn_mask(
    key_padding_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Key-padding-only SDPA mask, kept in its broadcast ``(B, 1, 1, L)`` form.

    For a sequence that is *all* state tokens (``n_state >= seq_len``)
    :func:`build_2zone_mask` is all-zero — there is no action zone to block —
    so :func:`combine_masks` returns nothing but this row repeated ``seq_len``
    times. Same values, ``seq_len`` times the memory, and the broadcast is
    materialized on every forward. The policy's state pass is always in that
    regime: action tokens never enter it, they are decoded afterwards over the
    K/V cache (``KiCadRLModel._decode_appended``).

    An unpadded batch could go further and pass no mask at all — an all-zero
    additive mask is a no-op, and only a null ``attn_mask`` lets SDPA reach the
    flash kernel ("Flash Attention does not support non-null attn_mask"), worth
    ~1.6x there. We deliberately do NOT: swapping kernels changes the
    floating-point reduction order, and under bf16 that moves outputs by ~5e-3
    (on outputs of scale ~0.8) — enough for a greedy argmax to flip and send an
    episode down a different trajectory. Always emitting the mask keeps this
    rewrite bit-identical to the dense form it replaces. The flash path is only
    reachable when a batch has zero padding, which update minibatches (256
    random samples) essentially never are, so the speed left on the table is
    ~0 where the GPU time actually is. The sanctioned kernel change is the
    opt-in flex_attention path (:func:`flex_padding_block_mask`), which is
    faster on padded batches too.

    Args:
        key_padding_mask: ``(B, seq_len)`` bool, True = padded.
        dtype: query dtype; SDPA requires the additive mask to match it.

    Returns:
        ``(B, 1, 1, seq_len)`` additive mask.
    """
    return torch.where(
        key_padding_mask[:, None, None, :],
        torch.tensor(float("-inf"), device=key_padding_mask.device, dtype=dtype),
        torch.tensor(0.0, device=key_padding_mask.device, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# flex_attention over a key-padding BlockMask (opt-in state-pass kernel)
# ---------------------------------------------------------------------------
# Block granularity of the BlockMask; sequences are padded up to a multiple of
# it (flex_padding_block_mask assumes whole blocks).
FLEX_BLOCK = 128

_FLEX_COMPILED = None


def padded_len(L: int, block: int = FLEX_BLOCK) -> int:
    """``L`` rounded up to a multiple of ``block``."""
    return -(-L // block) * block


def compiled_flex_attention():
    """The process-wide ``torch.compile(flex_attention, dynamic=True)``.

    Uncompiled flex_attention runs a generic fallback that is ~3x slower than
    SDPA at these sizes and materializes the score matrix (a 21 GB / OOM
    blow-up was traced to exactly that) — only the compiled kernel wins.
    ``dynamic=True`` is essential: with ``dynamic=False`` every distinct
    ``(B, L_pad)`` is a ~2.5 s recompile, and ``--mem-budget`` chunking (chunk
    sizes ``B`` vary per minibatch) exhausted the 128-recompile limit within
    minutes, after which dynamo silently ran flex eager. Measured: one dynamic
    graph serves every shape at the same or better steady-state speed than
    the static graphs (B64/L1024: 4.1 vs 6.4 ms). Built lazily so importing
    this module stays free of the flex import.
    """
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=True)
    return _FLEX_COMPILED


@torch._dynamo.disable
def flex_attention_call(q, k, v, block_mask=None, score_mod=None, scale=None):
    """Call the compiled flex kernel as a graph break inside a compiled stack.

    When flex is traced into an enclosing torch.compile graph (the 'stack'
    region), its backward returns ``None`` for the gradients of tensors the
    ``score_mod`` captured (torch 2.8) — the same-net ``alpha`` would get no
    gradient and the joint graph fails on the first grad accumulation. Skipped
    by dynamo, this frame runs the nested compiled flex with its own autograd,
    which handles captured buffers correctly; the rest of the layer stays in
    the outer graph. The break is not free — without a score_mod the inlined
    form is faster (body d3b 1.26x vs 1.05x, d3c 1.75x vs 1.23x over sdpa) —
    so :class:`MultiHeadAttention` takes this route only when a ``score_mod``
    is present.
    """
    return compiled_flex_attention()(
        q, k, v, block_mask=block_mask, score_mod=score_mod, scale=scale,
    )


def flex_padding_block_mask(lens: torch.Tensor, L_pad: int, block: int = FLEX_BLOCK):
    """Key-padding BlockMask for flex_attention, built straight from lengths.

    Row ``b`` may attend keys ``[0, lens[b])``; every query row (padded ones
    included — their outputs are discarded downstream) sees the same key set.
    Because padding is a suffix, the block structure is known without scanning
    the mask: key blocks below ``lens[b] // block`` are full, the one holding
    ``lens[b]`` (if any) is partial and gets ``mask_mod`` applied inside the
    kernel, and the rest are skipped by the kernel entirely. This is the ~0.05
    ms path; ``create_block_mask`` evaluates ``mask_mod`` over the whole
    ``(B, L, L)`` grid for the same result.

    Contract: ``lens`` must describe SUFFIX padding (positions ``>= lens[b]``
    padded), which is what the tokenizer emits. A non-suffix mask would mark a
    block as full while it contains padding, and the kernel does not apply
    ``mask_mod`` to full blocks.

    Args:
        lens: ``(B,)`` int64 real lengths, on the attention device.
        L_pad: padded sequence length, a multiple of ``block``.
        block: BlockMask block size (``BLOCK_SIZE``).

    Returns:
        ``BlockMask`` with ``seq_lengths == (L_pad, L_pad)`` and a head
        dimension of 1 (broadcast over heads).
    """
    from torch.nn.attention.flex_attention import BlockMask

    assert L_pad % block == 0, (L_pad, block)
    B = lens.size(0)
    n = L_pad // block
    full = (lens // block).to(torch.int32)                       # (B,)
    partial = ((lens % block) > 0).to(torch.int32)               # (B,) 0/1
    # Same key set for every query block -> expand the per-row counts over
    # the n query blocks. The partial block (at most one) sits at index
    # ``full[b]``; its slot is only read when partial[b] == 1, so clamp the
    # no-partial rows (full == n) into range.
    full_num = full.view(B, 1, 1).expand(B, 1, n).contiguous()
    part_num = partial.view(B, 1, 1).expand(B, 1, n).contiguous()
    full_idx = (torch.arange(n, device=lens.device, dtype=torch.int32)
                .view(1, 1, 1, n).expand(B, 1, n, n).contiguous())
    part_idx = (full.clamp(max=n - 1).view(B, 1, 1, 1)
                .expand(B, 1, n, n).contiguous())

    def mask_mod(b, h, q, kv):
        return kv < lens[b]

    return BlockMask.from_kv_blocks(
        part_num, part_idx, full_num, full_idx,
        BLOCK_SIZE=block, mask_mod=mask_mod, seq_lengths=(L_pad, L_pad),
    )


def same_net_score_mod(slot_ids: torch.Tensor, alpha: torch.Tensor):
    """flex ``score_mod`` for the per-head same-net bias.

    ``score + alpha[h]`` where ``slot_ids[b, q] == slot_ids[b, kv]`` and both
    are valid (``>= 0``) — the same additive bias :class:`SameNetBias` defines,
    applied inside the kernel. The sdpa path absorbs it into extra q/k
    channels instead (``MultiHeadAttention``'s ``same_net_aug``); flex must NOT
    take that route: it widens the head dim by ``K_pad`` (up to the number of
    nets on the board), which the flex template rounds up to a power of two —
    a ``K_pad`` of 64 turns head dim 16 into 128 and the kernel runs out of
    shared memory ("No valid triton configs") — and every distinct ``K_pad``
    is another static recompile. ``alpha`` (fp32 Parameter) receives
    gradients through the captured tensor.

    ``alpha`` is captured spread out as a ``(B, H, L)`` tensor and read at
    ``[b, h, q]``, not as the ``(H,)`` parameter itself: the flex backward
    accumulates a captured buffer's gradient element-wise (atomically), so an
    ``(H,)`` capture funnels every (b, q, kv) element onto H addresses and
    serializes — measured 10-20x SLOWER than sdpa. Per-(b, h, q) addresses
    make it 1.7x (d3b) to 4.9x (L=1024, ~100 nets) FASTER than the sdpa
    absorption; the reduction back to ``(H,)`` is the expand's backward.

    Args:
        slot_ids: ``(B, L)`` int64, ``-1`` = padded / no-slot (never same-net).
        alpha: ``(H,)`` per-head bias scale.
    """
    B, L = slot_ids.shape
    alpha_bhq = alpha.view(1, -1, 1).expand(B, alpha.numel(), L).contiguous()

    def score_mod(score, b, h, q, kv):
        sq = slot_ids[b, q]
        same = (sq == slot_ids[b, kv]) & (sq >= 0)
        return torch.where(same, score + alpha_bhq[b, h, q], score)

    return score_mod


# ---------------------------------------------------------------------------
# ReZero residual (replaces GTrXL GRUGate)
# ---------------------------------------------------------------------------
class ReZero(nn.Module):
    """ReZero residual gate — ``out = x + alpha * y`` with ``alpha`` init 0.

    Why this exists:
        GTrXL's per-layer ``GRUGate`` carries 6·d_model² + d_model params and
        adds 6 matrix multiplies per sublayer. ReZero achieves the same
        "initial identity pass-through" property with a single scalar, cuts
        ~49K params per gate at d_model=128, and removes 6 GEMMs per layer.

    Initial state:
        ``alpha = 0`` → output equals the residual input ``x`` exactly. The
        sublayer ``y`` is dynamically gated up only as gradients open it.
    """

    def __init__(self, _d_model: int | None = None) -> None:
        # The d_model arg is accepted for API parity with GRUGate; ReZero
        # has no per-channel parameters so it is unused.
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * y


# ---------------------------------------------------------------------------
# Per-head same-net attention bias (opt-in)
# ---------------------------------------------------------------------------
class SameNetBias(nn.Module):
    """Per-head same-net attention bias — the learnable ``alpha`` container.

    Semantics: ``bias[b, h, i, j] = alpha[h] * 1[slot_i == slot_j & both
    valid]`` (slot id ``-1`` = padded/no-slot, never same-net). ``alpha`` is
    consumed exclusively through the exact q/k-channel absorption
    (:func:`build_slot_membership` + ``MultiHeadAttention``'s ``same_net_aug``)
    — the dense ``(B,H,L,L)`` materialization was removed 2026-07-16 (the
    absorbed form is the mathematically identical, memory-safe rewrite).
    ``alpha`` is initialised to 0 (ReZero-style) so the initial forward is
    bit-identical to a no-bias SDPA, then grows only if gradients open it.
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_heads))  # (H,)


def build_slot_membership(
    slot_ids: torch.Tensor,
    align: int = 8,
) -> torch.Tensor:
    """One-hot slot-membership matrix ``M`` ``(B, L, K_pad)`` with
    ``M @ Mᵀ == 1[slot_i == slot_j & both valid]`` (the same-net indicator).

    This is the factor that lets :class:`MultiHeadAttention` absorb the
    per-head same-net bias ``α_h · MMᵀ`` into extra q/k channels instead of
    materializing the ``(B, H, L, L)`` bias tensor::

        QKᵀ/√d + α_h·MMᵀ  =  [Q, α_h√d·M][K, M]ᵀ / √d

    ``K_pad`` = (highest present slot id + 1) rounded up to ``align``. In the
    default configuration (no ``--slot-perm``) slot ids are compact
    ``[0, net_count)``, so ``K_pad`` is just a handful of channels. ``--slot-perm``
    spreads ids across the full ``N_MAX_SLOTS`` (=64) table, which would push
    ``K_pad`` toward 64 — re-evaluate the absorption there.

    Args:
        slot_ids: ``(B, L)`` int64, ``-1`` = padded / no-slot (never same-net).
        align: ``K_pad`` is rounded up to a multiple of this. 8 keeps the
            augmented head-dim ``d_head + K_pad`` aligned for the CUDA
            mem-efficient kernel (unaligned strides raise "LSE is not correctly
            aligned"); ``d_head`` is already a multiple of 8 in this model.

    Returns:
        ``(B, L, K_pad)`` float32, ``K_pad >= align``. Rows at padded / no-slot
        positions are all-zero (contribute nothing to ``M @ Mᵀ``).
    """
    valid = slot_ids >= 0                                   # (B, L)
    K = int(slot_ids.max().item()) + 1                     # highest present slot id + 1 (0 if all -1)
    K_pad = max(align, -(-K // align) * align)             # ceil(K/align)*align, >= align
    M = F.one_hot(slot_ids.clamp(min=0), K_pad).to(torch.float32)   # (B, L, K_pad)
    return M * valid.unsqueeze(-1).to(torch.float32)


# ---------------------------------------------------------------------------
# Multi-Head Attention (standard SDPA, no positional encoding)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Standard pre-norm multi-head attention over PyTorch SDPA.

    The state zone is a permutation-equivariant set, so RoPE has been
    removed from the QK path; ordering for the action zone is supplied
    via the learned ``action_pos_emb`` injected at the embedding layer.

    Kernel: SDPA with an additive ``attn_mask``, or — when a ``block_mask``
    is passed instead — compiled flex_attention, which skips key blocks that
    are entirely padding (:func:`flex_padding_block_mask`). The same-net bias
    rides on ``same_net_aug`` (q/k-channel absorption) for SDPA and on
    ``score_mod`` (:func:`same_net_score_mod`) for flex — never both.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        same_net_aug: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
        block_mask=None,
        score_mod=None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """SDPA (or flex) attention over (optionally bias-augmented) q/k/v.

        Args:
            x: ``(B, L, d_model)`` input.
            attn_mask: additive SDPA mask; ``(B, 1, L, L)`` zone+padding only
                (the same-net bias is NOT baked in here when ``same_net_aug``
                is given — it is absorbed into q/k channels instead).
                Ignored when ``block_mask`` is given.
            block_mask: flex_attention ``BlockMask`` (key padding) — selects
                the flex kernel; ``L`` must equal its ``seq_lengths``.
                ``same_net_aug`` must be ``None`` on this path.
            score_mod: flex ``score_mod`` (flex path only), e.g. the same-net
                bias from :func:`same_net_score_mod`.
            same_net_aug: optional ``(M, alpha)`` where ``M`` is the
                ``(B, L, K_pad)`` slot-membership matrix from
                :func:`build_slot_membership` and ``alpha`` is the per-head
                ``(H,)`` bias scale. When present, ``α_h·MMᵀ`` is folded into
                extra q/k channels — an exact reformulation of adding
                ``α_h·1[same-net]`` to the pre-softmax logits, but without ever
                materializing the ``(B, H, L, L)`` bias.
            return_kv: also return the raw (un-augmented) per-head ``(k, v)``,
                each ``(B, H, L, d_head)``, for reuse as a prefix cache in
                :meth:`forward_incremental`.
        """
        B, L, D = x.shape
        qkv = self.qkv_proj(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        if block_mask is not None:
            assert same_net_aug is None, "flex takes the same-net bias as score_mod"
            # Captured-grad score_mod -> graph break (see flex_attention_call);
            # otherwise let the outer compile inline the kernel (faster).
            flex = flex_attention_call if score_mod is not None else compiled_flex_attention()
            attend = lambda q_, k_, v_, scale=None: flex(  # noqa: E731
                q_, k_, v_, block_mask=block_mask, score_mod=score_mod, scale=scale,
            )
        else:
            attend = lambda q_, k_, v_, scale=None: F.scaled_dot_product_attention(  # noqa: E731
                q_, k_, v_, attn_mask=attn_mask, scale=scale,
            )
        if same_net_aug is None:
            out = attend(q, k, v)
        else:
            # QKᵀ/√d + α_h·MMᵀ = [Q, α_h√d·M][K, M]ᵀ / √d. Append M as extra
            # key/query channels (q scaled by α_h√d), zero-pad v to the same
            # width (kernel needs matching head-dim), slice the real output back
            # off. `scale` MUST stay 1/√d_head (not 1/√(d_head+K_pad)).
            M, alpha = same_net_aug
            K_pad = M.size(-1)
            m = M.to(q.dtype).unsqueeze(1).expand(B, self.n_heads, L, K_pad)
            # .to(q.dtype): alpha is an fp32 Parameter — under bf16 autocast
            # the product promotes to fp32 and torch.cat would dtype-mismatch.
            q_aug = torch.cat(
                [q, (alpha.view(1, -1, 1, 1) * math.sqrt(self.d_head) * m).to(q.dtype)],
                dim=-1,
            )
            k_aug = torch.cat([k, m], dim=-1)
            v_aug = torch.cat([v, v.new_zeros(B, self.n_heads, L, K_pad)], dim=-1)
            out = attend(
                q_aug, k_aug, v_aug, scale=1.0 / math.sqrt(self.d_head),
            )[..., : self.d_head]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        if return_kv:
            return out, (k, v)
        return out

    def forward_incremental(
        self,
        x_new: torch.Tensor,
        kv_prefix: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attention for appended query tokens over cached prefix K/V.

        Exact under two conditions the policy layer guarantees: (i) prefix
        tokens never attend to appended tokens (2-zone mask), so the cached
        prefix K/V stay valid; (ii) appended action tokens carry slot ``-1``,
        so the same-net-bias q-channels (``α_h√d·m``) are all-zero for them —
        the augmented path contributes exactly 0 to values AND grads and is
        skipped here. Gradients flow through ``kv_prefix`` when it requires
        grad (training reuses the state pass instead of recomputing it).

        Args:
            x_new: ``(B, n_new, d_model)`` appended token embeddings.
            kv_prefix: per-head ``(k, v)``, each ``(B, H, L_prefix, d_head)``,
                from a prior ``forward(..., return_kv=True)`` /
                ``forward_incremental`` call.
            attn_mask: additive SDPA mask ``(B, 1, n_new, L_prefix + n_new)``.

        Returns:
            ``(out, (k_new, v_new))`` — ``out`` is ``(B, n_new, d_model)``;
            ``k_new`` / ``v_new`` are ``(B, H, n_new, d_head)`` for extending
            the prefix cache.
        """
        B, n_new, D = x_new.shape
        qkv = self.qkv_proj(x_new)  # (B, n_new, 3*D)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        q = q.view(B, n_new, self.n_heads, self.d_head).transpose(1, 2)
        k_new = k_new.view(B, n_new, self.n_heads, self.d_head).transpose(1, 2)
        v_new = v_new.view(B, n_new, self.n_heads, self.d_head).transpose(1, 2)
        k = torch.cat([kv_prefix[0], k_new], dim=2)
        v = torch.cat([kv_prefix[1], v_new], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, n_new, D)
        return self.out_proj(out), (k_new, v_new)


# ---------------------------------------------------------------------------
# Gated Transformer Layer
# ---------------------------------------------------------------------------
class GatedTransformerLayer(nn.Module):
    """Pre-norm Transformer layer with ReZero residual scaling.

    Uses :class:`MultiHeadAttention` (standard SDPA, no RoPE). The residual
    gates (``res_attn`` / ``res_ff``) are :class:`ReZero` scalars, init 0,
    so the layer is initially an identity over the residual stream.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.res_attn = ReZero(d_model)
        self.res_ff = ReZero(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        same_net_aug: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
        block_mask=None,
        score_mod=None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention + ReZero residual
        if return_kv:
            y, kv = self.attn(
                self.norm1(x), attn_mask=attn_mask, same_net_aug=same_net_aug,
                return_kv=True, block_mask=block_mask, score_mod=score_mod,
            )
        else:
            y = self.attn(
                self.norm1(x), attn_mask=attn_mask, same_net_aug=same_net_aug,
                block_mask=block_mask, score_mod=score_mod,
            )
        x = self.res_attn(x, y)
        # Pre-norm FFN + ReZero residual
        y = self.ff(self.norm2(x))
        x = self.res_ff(x, y)
        return (x, kv) if return_kv else x

    def forward_incremental(
        self,
        x_new: torch.Tensor,
        kv_prefix: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Layer forward for appended tokens over a cached K/V prefix.

        See :meth:`MultiHeadAttention.forward_incremental` for the exactness
        conditions. Returns ``(x_new_out, (k_new, v_new))``.
        """
        y, kv_new = self.attn.forward_incremental(
            self.norm1(x_new), kv_prefix, attn_mask=attn_mask,
        )
        x = self.res_attn(x_new, y)
        y = self.ff(self.norm2(x))
        x = self.res_ff(x, y)
        return x, kv_new


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------
def init_weights(model: nn.Module) -> None:
    """Orthogonal weight initialisation (SB3-style).

    - Backbone ``nn.Linear``: ``gain=1.0``
    - Policy head ``nn.Linear`` (name contains ``'policy_head'``): ``gain=0.01``
      → keeps the initial action distribution near-uniform.
    - ``ReZero.alpha``: untouched (already 0; we only init ``nn.Linear``).
    - ``nn.Linear`` bias: zeros

    Note: the substring is ``'policy_head'`` (not ``'head'``) so that
    multi-layer critic MLPs named ``critic_head.*`` are NOT shrunk to 0.01.
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            gain = 0.01 if "policy_head" in name else 1.0
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
