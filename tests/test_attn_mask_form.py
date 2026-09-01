"""SDPA mask form for the state pass — the (B,1,L,L) -> (B,1,1,L) rewrite.

The state pass is always all-state (``n_state == L``; action tokens are decoded
separately over the K/V cache), so ``build_2zone_mask`` is all-zero there and
the dense mask it feeds ``combine_masks`` is the key-padding row repeated L
times. Passing that row broadcast instead is the same computation with no L^2
tensor built per forward.

The contract these tests pin is **bit-identity**, not "close enough": an
unpadded batch could drop the mask entirely and reach the flash kernel, but
swapping SDPA kernels changes the floating-point reduction order (~5e-3 under
bf16), which is enough to flip a greedy argmax and send an episode down a
different trajectory. ``padding_attn_mask`` therefore always emits the mask.

No C++ dependency — pure PyTorch tests.
"""

from __future__ import annotations

import pytest
import torch

from methods.rl_agent.models.v1 import net as net_mod
from methods.rl_agent.models.v1.blocks import (
    build_2zone_mask,
    combine_masks,
    padding_attn_mask,
)
from methods.rl_agent.models.v1.net import KiCadRLModel
from tests._mock_obs import make_mock_obs


def _kpm(seq_lens: list[int], width: int) -> torch.Tensor:
    return torch.arange(width)[None, :] >= torch.tensor(seq_lens)[:, None]


def _tiny_policy() -> KiCadRLModel:
    return KiCadRLModel(d_model=32, n_heads=4, n_layers=2, d_ff=64,
                        max_seq_len=2000, n_freq=4)


def _open_rezero(model: KiCadRLModel, alpha: float = 0.5) -> None:
    """ReZero alphas init to 0, which makes every layer an exact identity —
    an attention-level regression is then invisible. Open them so the tests
    actually exercise the attention path."""
    with torch.no_grad():
        for layer in model.layers:
            layer.res_attn.alpha.fill_(alpha)
            layer.res_ff.alpha.fill_(alpha)


class TestMaskEquivalence:
    @pytest.mark.parametrize("seq_lens", [[12, 7, 5], [12, 12, 12]])
    def test_broadcast_form_equals_dense_all_state_mask(self, seq_lens):
        L = 12
        kpm = _kpm(seq_lens, L)
        dense = combine_masks(build_2zone_mask(L, L), kpm)   # (B, 1, L, L)
        bcast = padding_attn_mask(kpm, torch.float32)        # (B, 1, 1, L)
        assert dense.shape == (len(seq_lens), 1, L, L)
        assert bcast.shape == (len(seq_lens), 1, 1, L)
        assert torch.equal(dense, bcast.expand_as(dense))

    def test_zone_mask_is_all_zero_when_n_state_equals_len(self):
        # The premise of the rewrite: with no action zone there is nothing for
        # build_2zone_mask to block.
        assert torch.equal(build_2zone_mask(16, 16), torch.zeros(16, 16))

    def test_mask_is_emitted_even_when_nothing_is_padded(self):
        # Dropping it would reach flash and change the kernel's arithmetic.
        kpm = _kpm([9, 9], 9)
        assert not kpm.any()
        m = padding_attn_mask(kpm, torch.float32)
        assert m is not None
        assert torch.equal(m, torch.zeros_like(m))


class TestStatePassEquivalence:
    @staticmethod
    def _legacy(model, embs, kpm):
        L = embs.size(1)
        mask = combine_masks(build_2zone_mask(L, L), kpm).to(embs.dtype)
        x = embs
        for layer in model.layers:
            x = layer(x, attn_mask=mask, same_net_aug=None)
        return x

    @pytest.mark.parametrize("seq_lens", [[16, 11, 4], [16, 16, 16]])
    def test_run_transformer_is_bit_identical_to_dense_path(self, seq_lens):
        torch.manual_seed(0)
        model = _tiny_policy().double()
        _open_rezero(model)
        B, L = len(seq_lens), 16
        embs = torch.randn(B, L, 32, dtype=torch.float64)
        kpm = _kpm(seq_lens, L)
        new = model._run_transformer(embs, L, kpm)
        assert torch.equal(new, self._legacy(model, embs, kpm))

    def test_policy_outputs_are_bit_identical_to_dense_path(self, monkeypatch):
        torch.manual_seed(0)
        model = _tiny_policy()
        _open_rezero(model)
        obs_list = [make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1),
                    make_mock_obs(n_nets=3, pads_per_net=3, n_ratsnest_per_net=2)]
        with torch.no_grad():
            enc_new = model._encode_state(obs_list, return_cache=False)

        # Force the legacy dense mask back in and re-encode.
        def _dense(kpm, dtype):
            L = kpm.size(1)
            return combine_masks(build_2zone_mask(L, L), kpm).to(dtype)
        monkeypatch.setattr(net_mod, "padding_attn_mask", _dense)
        with torch.no_grad():
            enc_ref = model._encode_state(obs_list, return_cache=False)

        assert torch.equal(enc_new.H_state, enc_ref.H_state)
        assert torch.equal(enc_new.values, enc_ref.values)
        assert torch.equal(enc_new.at_logits, enc_ref.at_logits)

    def test_state_pass_never_builds_an_LxL_mask(self, monkeypatch):
        # Guard against a silent revert: combine_masks must not be reached.
        def _boom(*a, **k):
            raise AssertionError("state pass materialized a dense (B,1,L,L) mask")
        monkeypatch.setattr(net_mod, "combine_masks", _boom)
        model = _tiny_policy()
        with torch.no_grad():
            model._encode_state([make_mock_obs(n_nets=2)], return_cache=False)


class TestPaddingIsInert:
    def test_perturbing_padded_positions_leaves_real_outputs_intact(self):
        torch.manual_seed(0)
        model = _tiny_policy().double()
        _open_rezero(model)
        B, L = 2, 16
        embs = torch.randn(B, L, 32, dtype=torch.float64)
        kpm = _kpm([16, 9], L)

        out_a = model._run_transformer(embs, L, kpm)
        noisy = embs.clone()
        # Must not be a constant offset: LayerNorm subtracts the per-token
        # mean, so adding c to every channel of a token is a no-op and the
        # test would pass vacuously.
        noisy[kpm] = torch.randn_like(noisy[kpm]) * 5.0
        out_b = model._run_transformer(noisy, L, kpm)

        real = ~kpm
        assert torch.equal(out_a[real], out_b[real])

    def test_dropping_the_mask_would_leak_padding(self):
        # Why padding_attn_mask never returns None: this is what it would cost.
        torch.manual_seed(0)
        model = _tiny_policy().double()
        _open_rezero(model)
        B, L = 2, 16
        embs = torch.randn(B, L, 32, dtype=torch.float64)
        kpm = _kpm([16, 9], L)

        def _no_mask(x):
            h = x
            for layer in model.layers:
                h = layer(h, attn_mask=None, same_net_aug=None)
            return h

        noisy = embs.clone()
        noisy[kpm] = torch.randn_like(noisy[kpm]) * 5.0
        real = ~kpm
        assert not torch.allclose(_no_mask(embs)[real], _no_mask(noisy)[real],
                                  atol=1e-6, rtol=0)
