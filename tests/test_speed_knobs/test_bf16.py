"""bf16 autocast sanity — intentionally changes numerics, loose guards only."""

from __future__ import annotations

import pytest
import torch

from tests.test_speed_knobs._helpers import batch, opened_model, run_eval

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="speed knobs are CUDA-only",
)


class TestBF16Sanity:
    def test_bf16_close_and_grads_finite(self):
        m = opened_model()
        obs = batch()
        acts, _ = m.act(obs, deterministic=True)

        lp_f, ent_f, val_f, g_f = run_eval(m, obs, acts)

        m.configure_speed(bf16=True)
        lp_b, ent_b, val_b, g_b = run_eval(m, obs, acts)

        # bf16 (~8-bit mantissa) through 2 layers — loose agreement only.
        assert torch.allclose(lp_f, lp_b, atol=0.15, rtol=0.05)
        assert torch.allclose(val_f, val_b, atol=0.15, rtol=0.05)
        assert torch.allclose(ent_f, ent_b, atol=0.15, rtol=0.05)
        for name, g in g_b.items():
            assert torch.isfinite(g).all(), name
        assert g_b["layers.0.attn.qkv_proj.weight"].abs().max() > 1e-6
        # Outputs must be fp32 at the head boundary (autocast stays internal).
        assert lp_b.dtype == torch.float32 and val_b.dtype == torch.float32
