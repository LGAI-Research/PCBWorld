"""flex_attention state pass (``configure_speed(attn='flex')``) vs the sdpa path.

The flex path pads the sequence to a ``FLEX_BLOCK`` multiple, builds a
key-padding ``BlockMask`` straight from the per-row lengths (no mask scan) and
lets the kernel skip all-padding key blocks. It swaps the attention kernel, so
the contract is *fp32 agreement to rounding* (1e-5, outputs and grads), not the
bit-identity ``tests/test_attn_mask_form.py`` pins for the sdpa mask rewrite.
The BlockMask construction itself is exact and is checked against
``create_block_mask`` on CPU; everything that runs the kernel is CUDA-only
like the other speed knobs.
"""

from __future__ import annotations

import pytest
import torch

from methods.rl_agent.models.v1 import blocks
from methods.rl_agent.models.v1.blocks import (
    FLEX_BLOCK,
    flex_padding_block_mask,
    padded_len,
)
from tests.test_speed_knobs._helpers import (
    assert_compile_matches_eager,
    batch,
    opened_model,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="speed knobs are CUDA-only",
)


def _kpm(lens: torch.Tensor, L: int) -> torch.Tensor:
    return torch.arange(L, device=lens.device)[None, :] >= lens[:, None]


class TestBlockMaskFromLens:
    def test_padded_len_rounds_up_to_block(self):
        assert padded_len(1) == FLEX_BLOCK
        assert padded_len(FLEX_BLOCK) == FLEX_BLOCK
        assert padded_len(FLEX_BLOCK + 1) == 2 * FLEX_BLOCK
        assert padded_len(453) == 512 and padded_len(891) == 896

    @pytest.mark.parametrize("L_pad", [FLEX_BLOCK, 3 * FLEX_BLOCK])
    def test_matches_create_block_mask(self, L_pad):
        from torch.nn.attention.flex_attention import create_block_mask

        # Edge cases: exact multiples, one token, one past a boundary, full.
        lens = torch.tensor([1, FLEX_BLOCK, FLEX_BLOCK + 1, L_pad, L_pad - 1, 5])
        lens = lens.clamp(max=L_pad)
        ours = flex_padding_block_mask(lens, L_pad)

        def mask_mod(b, h, q, kv):
            return kv < lens[b]

        ref = create_block_mask(mask_mod, B=lens.numel(), H=None, Q_LEN=L_pad,
                                KV_LEN=L_pad, device="cpu")
        assert ours.seq_lengths == (L_pad, L_pad)
        assert torch.equal(ours.to_dense(), ref.to_dense())
        # Same block structure too — a full block mis-filed as partial would
        # still give the same dense mask but cost a mask_mod pass.
        assert torch.equal(ours.kv_num_blocks, ref.kv_num_blocks)
        assert torch.equal(ours.full_kv_num_blocks, ref.full_kv_num_blocks)

    def test_block_table_and_mask_mod_reproduce_the_key_padding_mask(self):
        L_pad = 2 * FLEX_BLOCK
        lens = torch.tensor([L_pad, 130, 7])
        B, nb = lens.numel(), L_pad // FLEX_BLOCK
        bm = flex_padding_block_mask(lens, L_pad)
        # Block level (to_dense is the block table): row b keeps every key
        # block below ceil(lens[b] / block), for every query block.
        n_keep = -(-lens // FLEX_BLOCK)
        blocks_kept = (torch.arange(nb)[None, :] < n_keep[:, None])
        assert torch.equal(bm.to_dense().bool(),
                           blocks_kept[:, None, None, :].expand(B, 1, nb, nb))
        # Token level: mask_mod (applied inside partial blocks) is exactly the
        # key-padding mask, for padded query rows too.
        b = torch.arange(B)[:, None, None]
        q = torch.arange(L_pad)[None, :, None]
        kv = torch.arange(L_pad)[None, None, :]
        tok = bm.mask_mod(b, torch.zeros((), dtype=torch.long), q, kv)
        assert torch.equal(tok.expand(B, L_pad, L_pad),
                           (~_kpm(lens, L_pad))[:, None, :].expand(B, L_pad, L_pad))

    def test_flex_needs_head_dim_16(self):
        # The flex kernel rejects E < 16 deep inside inductor; fail at the knob.
        from methods.rl_agent.models.v1.net import KiCadRLModel
        m = KiCadRLModel(d_model=32, n_heads=4, n_layers=1, d_ff=64,
                         max_seq_len=2000, n_freq=4)
        with pytest.raises(AssertionError, match="head_dim >= 16"):
            m.configure_speed(attn="flex")


@cuda_only
class TestFlexMatchesSdpa:
    @pytest.mark.parametrize("same_net", [False, True])
    @pytest.mark.parametrize("L", [FLEX_BLOCK, FLEX_BLOCK + 37])
    def test_run_transformer_fp32(self, L, same_net):
        # opened_model: ReZero open (else the stack is an identity and the
        # kernel is never exercised); n_heads=2 -> d_head 16 (flex minimum).
        # same_net=True routes flex through the q/k-augmented path as well.
        m = opened_model(n_heads=2)
        B, d = 4, m.d_model
        torch.manual_seed(1)
        embs = torch.randn(B, L, d, device="cuda", requires_grad=True)
        lens = torch.tensor([L, L - 1, L // 2, 3], device="cuda")
        kpm = _kpm(lens, L)
        slots = torch.randint(-1, 4, (B, L), device="cuda") if same_net else None

        out = {}
        for impl in ("sdpa", "flex"):
            m.attn_impl = impl
            m.zero_grad(set_to_none=True)
            embs.grad = None
            x, cache = m._run_transformer(embs, L, kpm, slot_ids=slots,
                                          return_cache=True)
            x[~kpm].pow(2).mean().backward()
            out[impl] = (
                x.detach(), embs.grad.clone(),
                {n: p.grad.clone() for n, p in m.named_parameters()
                 if p.grad is not None},
                cache,
            )
        xs, gs, ps, cs = out["sdpa"]
        xf, gf, pf, cf = out["flex"]
        real = ~kpm
        assert torch.allclose(xs[real], xf[real], atol=1e-5, rtol=0)
        assert torch.allclose(gs[real], gf[real], atol=1e-5, rtol=0)
        assert ps.keys() == pf.keys()
        for n in ps:
            scale = ps[n].abs().max().clamp(min=1.0)
            assert torch.allclose(ps[n] / scale, pf[n] / scale, atol=1e-5, rtol=0), n
        # Nothing downstream may see L_pad: hiddens, cache K/V and the cache's
        # padding mask keep the exact shapes the sdpa path produces.
        assert xf.shape == xs.shape
        assert torch.equal(cf.key_padding_mask, cs.key_padding_mask)
        for (ks, vs), (kf, vf) in zip(cs.kv, cf.kv):
            assert kf.shape == ks.shape and vf.shape == vs.shape
            keep = real[:, None, :, None].expand_as(ks)
            assert torch.allclose(ks[keep], kf[keep], atol=1e-5, rtol=0)

    def test_flex_same_net_never_widens_the_head_dim(self, monkeypatch):
        # The q/k absorption is the sdpa mechanism; on flex it inflates the
        # head dim by K_pad (-> shared-memory failure at K_pad 64, and a static
        # recompile per K_pad). flex must take the score_mod route instead.
        from methods.rl_agent.models.v1 import net as net_mod

        def _boom(*a, **k):
            raise AssertionError("flex path built the q/k slot-membership augmentation")
        monkeypatch.setattr(net_mod, "build_slot_membership", _boom)
        m = opened_model(n_heads=2)
        m.attn_impl = "flex"
        B, L, d = 2, FLEX_BLOCK, m.d_model
        embs = torch.randn(B, L, d, device="cuda")
        kpm = _kpm(torch.tensor([L, L // 2], device="cuda"), L)
        slots = torch.randint(-1, 60, (B, L), device="cuda")  # K_pad would be 64
        with torch.no_grad():
            m._run_transformer(embs, L, kpm, slot_ids=slots)

    def test_padding_is_inert_under_flex(self):
        # Block skipping is driven by lens; a wrong block table would let
        # padded keys leak into real rows. Perturb padding with randn (not a
        # constant — LayerNorm absorbs offsets) and require identical real
        # rows.
        m = opened_model(n_heads=2)
        m.attn_impl = "flex"
        B, L, d = 3, FLEX_BLOCK + 9, m.d_model
        torch.manual_seed(2)
        embs = torch.randn(B, L, d, device="cuda")
        kpm = _kpm(torch.tensor([L, FLEX_BLOCK, 20], device="cuda"), L)
        with torch.no_grad():
            a = m._run_transformer(embs, L, kpm)
            noisy = embs.clone()
            noisy[kpm] = torch.randn_like(noisy[kpm]) * 5.0
            b = m._run_transformer(noisy, L, kpm)
        assert torch.equal(a[~kpm], b[~kpm])

    def test_policy_flex_eager_matches_sdpa(self):
        assert_compile_matches_eager((), attn="flex")

    def test_policy_flex_with_efficient_regions_matches_sdpa(self):
        # The adopted training combo: static-compiled stack around flex.
        assert_compile_matches_eager(("stack", "decode", "heads"), attn="flex")


@cuda_only
class TestFlexShapeBucketing:
    def test_lengths_in_one_block_share_one_graph(self):
        # flex is compiled with dynamic=False, so every distinct (B, L) would be
        # a ~2 s recompile; padding to FLEX_BLOCK multiples is what bounds it.
        import torch._dynamo
        import torch._dynamo.utils as du

        m = opened_model(n_heads=2)
        m.configure_speed(attn="flex")
        assert torch._dynamo.config.recompile_limit >= 128
        torch._dynamo.reset()
        blocks._FLEX_COMPILED = None
        du.counters.clear()

        def graphs() -> int:
            return int(du.counters["stats"]["unique_graphs"])

        B, d = 2, m.d_model
        counts = []
        for L in (FLEX_BLOCK - 30, FLEX_BLOCK - 1, FLEX_BLOCK, FLEX_BLOCK + 1):
            embs = torch.randn(B, L, d, device="cuda")
            kpm = _kpm(torch.tensor([L, L // 2], device="cuda"), L)
            with torch.no_grad():
                m._run_transformer(embs, L, kpm)
            counts.append(graphs())
        # three lengths in the first bucket -> one graph; the fourth crosses
        # the block boundary -> exactly one more.
        assert counts == [1, 1, 1, 2], counts
        # Same-net bias on: the slot-id range (K_pad on the sdpa path) must not
        # be a shape the flex graph specializes on.
        L = FLEX_BLOCK
        for hi in (3, 40, 500):
            embs = torch.randn(B, L, d, device="cuda")
            kpm = _kpm(torch.tensor([L, L // 2], device="cuda"), L)
            slots = torch.randint(-1, hi, (B, L), device="cuda")
            with torch.no_grad():
                m._run_transformer(embs, L, kpm, slot_ids=slots)
        assert graphs() == 3, graphs()  # one new graph for the score_mod variant, then stable

    def test_recompile_limit_raised_only_with_compile_or_flex(self):
        # No compiled region and sdpa -> dynamo config untouched; any compiled
        # region raises it (the 'heads' region alone exceeds the default 8 via
        # 0/1-size specializations and would silently run eager).
        import torch._dynamo
        cfg = torch._dynamo.config
        cfg.recompile_limit = 8
        opened_model().configure_speed(bf16=True)
        assert cfg.recompile_limit == 8
        opened_model().configure_speed(compile_regions=("heads",))
        assert cfg.recompile_limit >= 128
