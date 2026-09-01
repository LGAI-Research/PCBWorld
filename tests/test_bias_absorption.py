"""B1 regression: same-net attention bias absorbed into q/k channels.

Absorbing ``α_h·MMᵀ`` into extra q/k channels
(:func:`methods.rl_agent.models.v1.blocks.build_slot_membership` +
``MultiHeadAttention``'s ``same_net_aug``) is an exact rewrite of adding
``α_h·1[same-net]`` to the pre-softmax logits. The dense reference path was
REMOVED (2026-07-16) after the equivalence sign-off; the surviving guards pin
the mathematical core (``M @ Mᵀ`` == same-net indicator, channel alignment)
and the ckpt contract (``same_net_bias.alpha`` unchanged in ``state_dict``;
old checkpoints load strict and reproduce).

No C++ dependency — pure PyTorch. CPU always runs; CUDA runs too when present
(the CUDA mem-efficient kernel is where the channel-alignment constraint bites).
"""

from __future__ import annotations

import pytest
import torch

from methods.rl_agent.models.v1.blocks import build_slot_membership
from methods.rl_agent.models.v1.net import KiCadRLModel
from tests._mock_obs import make_mock_obs


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestSameNetBiasAbsorption:
    """Absorbing ``α_h·MMᵀ`` into extra q/k channels is an *exact* rewrite of
    adding ``α_h·1[same-net]`` to the pre-softmax logits.

    Critical: ReZero residual gates init to 0, which gates the attention
    sublayer (and hence the bias) completely out of the output. The gates MUST
    be opened or the whole comparison is vacuous (identity passthrough, zero
    ``alpha`` gradient). ``_opened_model`` does this and the test asserts the
    gradient is actually non-trivial.
    """

    def _opened_model(self, device: str) -> KiCadRLModel:
        torch.manual_seed(0)
        m = KiCadRLModel(
            d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=2000,
            n_freq=4, use_critic=True, same_net_bias=True,
        ).to(device)
        with torch.no_grad():
            m.same_net_bias.alpha.copy_(torch.tensor([-0.3, 0.7, -0.5, 1.1]))
            for layer in m.layers:
                layer.res_attn.alpha.fill_(0.7)   # open attention residual
                layer.res_ff.alpha.fill_(0.5)
        return m

    def _varied_batch(self) -> list[dict]:
        # Different net counts + routing states → variable seq_len and a
        # non-trivial same-net structure (slots spread, off-diagonal pairs).
        return [
            make_mock_obs(
                n_nets=nn, pads_per_net=2, n_ratsnest_per_net=2,
                is_routing=(i % 2 == 0), current_net_phase=1, current_layer=1,
                n_tracks=i, n_vias=i // 2,
            )
            for i, nn in enumerate([2, 4, 6, 3])
        ]

    def test_membership_gram_matches_indicator(self):
        # M @ Mᵀ must reproduce 1[slot_i == slot_j & both valid] exactly, and
        # K_pad must stay a multiple of 8 (mem-efficient kernel alignment).
        torch.manual_seed(1)
        slot_ids = torch.randint(-1, 64, (3, 40))
        M = build_slot_membership(slot_ids)
        assert M.size(-1) % 8 == 0
        gram = M @ M.transpose(1, 2)  # (B, L, L)
        valid = slot_ids >= 0
        same = (slot_ids.unsqueeze(-1) == slot_ids.unsqueeze(-2))
        same = (same & valid.unsqueeze(-1) & valid.unsqueeze(-2)).float()
        assert torch.equal(gram, same)

    def test_membership_all_no_slot(self):
        # All -1 → empty membership (no same-net pairs), still shape-valid.
        M = build_slot_membership(torch.full((2, 5), -1))
        assert M.size(-1) == 8 and float(M.sum()) == 0.0

    @pytest.mark.parametrize("device", _devices())
    def test_alpha_reaches_loss(self, device):
        # The absorbed bias must actually flow into the loss: a non-trivial
        # ``alpha`` gradient proves the q/k-channel path is wired (not gated
        # out by ReZero or dropped by the attention kernel).
        m = self._opened_model(device)
        obs = self._varied_batch()
        acts, _ = m.act(obs, deterministic=True)
        m.zero_grad(set_to_none=True)
        lp, ent, val = m.evaluate_actions_and_value(obs, acts)
        (lp.sum() + ent.sum() + val.pow(2).sum()).backward()
        assert m.same_net_bias.alpha.grad.abs().max() > 1e-4

    def test_alpha_in_state_dict(self):
        # ckpt hard requirement: ``alpha`` is the only bias state; the removed
        # dense/absorb runtime toggle was never a parameter/buffer.
        m = self._opened_model("cpu")
        sd = m.state_dict()
        assert "same_net_bias.alpha" in sd

    def test_checkpoint_roundtrip(self):
        # A saved state_dict loads cleanly into a fresh model (no missing /
        # unexpected keys) and reproduces identical outputs — dense-era
        # checkpoints are byte-identical, so backward compat holds.
        src = self._opened_model("cpu")
        obs = self._varied_batch()
        acts, _ = src.act(obs, deterministic=True)

        dst = KiCadRLModel(
            d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=2000,
            n_freq=4, use_critic=True, same_net_bias=True,
        )
        missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
        assert not missing and not unexpected

        with torch.no_grad():
            lp_s, _, v_s = src.evaluate_actions_and_value(obs, acts)
            lp_d, _, v_d = dst.evaluate_actions_and_value(obs, acts)
        assert torch.allclose(lp_s, lp_d, atol=1e-5)
        assert torch.allclose(v_s, v_d, atol=1e-5)
