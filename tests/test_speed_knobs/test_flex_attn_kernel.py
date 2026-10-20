"""flex_attention vs sdpa at the transformer stack (``_run_transformer``): fp32 agreement to
1e-5 of hiddens, input and parameter grads, and the KV cache, with and without the same-net
path. Split from ``test_flex_attn.py`` (its first case pays the ~5 s flex compile) so the two
files run on separate pytest-xdist workers.
"""

from __future__ import annotations

import pytest
import torch

from methods.rl_agent.models.v1.blocks import FLEX_BLOCK
from tests.test_speed_knobs._helpers import key_padding_mask, opened_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="speed knobs are CUDA-only")
@pytest.mark.parametrize("same_net", [False, True])
@pytest.mark.parametrize("L", [FLEX_BLOCK, FLEX_BLOCK + 37])
def test_run_transformer_fp32(L, same_net):
    # opened_model: ReZero open (else the stack is an identity and the
    # kernel is never exercised); n_heads=2 -> d_head 16 (flex minimum).
    # same_net=True routes flex through the q/k-augmented path as well.
    m = opened_model(n_heads=2)
    B, d = 4, m.d_model
    torch.manual_seed(1)
    embs = torch.randn(B, L, d, device="cuda", requires_grad=True)
    lens = torch.tensor([L, L - 1, L // 2, 3], device="cuda")
    kpm = key_padding_mask(lens, L)
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
