"""Transformer building blocks for the PCB routing policy (model layer).

Pure network architecture — no RL/action semantics. The policy layer
(:mod:`methods.rl_agent.models.v1.net`, ``KiCadRLModel``) assembles these
blocks with the state tokenizer and pointer/value heads and owns everything
RL-specific (act / log-prob / entropy / critic).

Components: ReZero, SameNetBias, MultiHeadAttention (standard SDPA),
GatedTransformerLayer, build_2zone_mask, combine_masks, init_weights.
"""

from __future__ import annotations

import math
import os

import torch
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
    (:func:`build_slot_membership` + ``MultiHeadAttention``'s ``same_net_aug``);
    the dense ``(B,H,L,L)`` bias tensor is never materialized, and the absorbed
    form is mathematically identical to adding it.
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

    The state zone is a permutation-equivariant set, so the QK path carries
    no positional encoding; ordering for the action zone is supplied via the
    learned ``action_pos_emb`` injected at the embedding layer.

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
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """SDPA over (optionally bias-augmented) q/k/v.

        Args:
            x: ``(B, L, d_model)`` input.
            attn_mask: additive SDPA mask; ``(B, 1, L, L)`` zone+padding only
                (the same-net bias is NOT baked in here when ``same_net_aug``
                is given — it is absorbed into q/k channels instead).
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
        if same_net_aug is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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
            out = F.scaled_dot_product_attention(
                q_aug, k_aug, v_aug, attn_mask=attn_mask,
                scale=1.0 / math.sqrt(self.d_head),
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
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention + ReZero residual
        if return_kv:
            y, kv = self.attn(
                self.norm1(x), attn_mask=attn_mask, same_net_aug=same_net_aug,
                return_kv=True,
            )
        else:
            y = self.attn(
                self.norm1(x), attn_mask=attn_mask, same_net_aug=same_net_aug,
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
