"""Shared fixtures/assertions for the speed-knob guards.

``KiCadRLModel.configure_speed`` wires two experiment knobs:

  * torch.compile regions ('stack'/'decode'/'heads'/'encode') — must be
    numerically equivalent to eager within fp32 kernel-reordering tolerance
    (allclose 1e-4), forward AND gradients.
  * bf16 autocast (stack + decode compute only) — intentionally changes
    numerics; guarded with loose relative agreement + finite/nonzero grads.

CUDA-only (autocast target + inductor compile time on CPU is prohibitive).
"""

from __future__ import annotations

import torch

from methods.rl_agent.models.v1.net import KiCadRLModel
from tests._mock_obs import make_mock_obs


def opened_model() -> KiCadRLModel:
    torch.manual_seed(0)
    m = KiCadRLModel(
        d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=2000,
        n_freq=4, use_critic=True, same_net_bias=True,
    ).to("cuda")
    with torch.no_grad():
        m.same_net_bias.alpha.copy_(torch.tensor([-0.3, 0.7, -0.5, 1.1]))
        for layer in m.layers:
            layer.res_attn.alpha.fill_(0.7)
            layer.res_ff.alpha.fill_(0.5)
    return m


def batch() -> list[dict]:
    return [
        make_mock_obs(
            n_nets=nn, pads_per_net=2, n_ratsnest_per_net=2,
            is_routing=(i % 2 == 0), current_net_phase=1, current_layer=1,
            n_tracks=i, n_vias=i // 2,
        )
        for i, nn in enumerate([2, 4, 6, 3])
    ]


def run_eval(m: KiCadRLModel, obs, acts):
    m.zero_grad(set_to_none=True)
    lp, ent, val = m.evaluate_actions_and_value(obs, acts)
    (lp.sum() + ent.sum() + val.pow(2).sum()).backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in m.named_parameters() if p.grad is not None
    }
    return lp.detach(), ent.detach(), val.detach(), grads


def assert_compile_matches_eager(regions: tuple[str, ...]) -> None:
    """Eager vs compiled-region equivalence: actions exact, outputs/grads 1e-4."""
    m = opened_model()
    obs = batch()
    acts, _ = m.act(obs, deterministic=True)

    lp_e, ent_e, val_e, g_e = run_eval(m, obs, acts)
    a_e, alp_e, av_e = m.act_and_value(obs, deterministic=True)

    m.configure_speed(compile_regions=regions)
    lp_c, ent_c, val_c, g_c = run_eval(m, obs, acts)
    a_c, alp_c, av_c = m.act_and_value(obs, deterministic=True)

    assert torch.equal(a_e, a_c)
    assert torch.allclose(alp_e, alp_c, atol=1e-4)
    assert torch.allclose(av_e, av_c, atol=1e-4)
    assert torch.allclose(lp_e, lp_c, atol=1e-4)
    assert torch.allclose(ent_e, ent_c, atol=1e-4)
    assert torch.allclose(val_e, val_c, atol=1e-4)
    assert g_e.keys() == g_c.keys()
    for name in g_e:
        assert torch.allclose(g_e[name], g_c[name], atol=1e-4), name
    assert g_e["layers.0.attn.qkv_proj.weight"].abs().max() > 1e-6
