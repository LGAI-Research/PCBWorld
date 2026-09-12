"""HEAD-token time feature: step_ratio (legacy) vs log_remaining vs
sin_remaining.

Covers the full option chain introduced for per-board step budgets:

* obs builder emits ``steps_remaining`` alongside ``step_ratio``,
* ``BatchedStateTokenizer(time_feature="log_remaining")`` encodes
  ``log1p(steps_remaining)/log1p(cap)`` into the same Fourier slot the
  legacy mode feeds with ``step_ratio`` (weight-shape identical!),
* ``time_feature="sin_remaining"`` feeds the linear ``remaining/cap``
  through a dedicated ladder anchored to step units (top rung period =
  2 steps → ±1 step resolves at any horizon; zero new weights, the
  ladder is a non-persistent buffer),
* ``RLPolicyConfig`` defaults keep old checkpoints (no ``time_feature``
  key) on the legacy path.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from configs.loader.schema import RLPolicyConfig
from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from tests._mock_obs import make_mock_obs


# ===================================================================
# Obs builder
# ===================================================================
def _stub_router_state():
    return SimpleNamespace(
        route_head=(1.0, 2.0, 0.0),
        current_layer=1,
        state_code=0,
        is_placing_via=False,
        is_routing=False,
        is_dragging=False,
    )


class TestObsStepsRemaining:
    def test_router_head_emits_steps_remaining(self):
        from pcb_world.core.observation import _build_router_head

        rh = _build_router_head(_stub_router_state(), step_count=30,
                                max_steps=200)
        assert rh["steps_remaining"] == 170
        assert rh["step_ratio"] == pytest.approx(30 / 200)

    def test_steps_remaining_clamped_at_zero(self):
        from pcb_world.core.observation import _build_router_head

        rh = _build_router_head(_stub_router_state(), step_count=205,
                                max_steps=200)
        assert rh["steps_remaining"] == 0


# ===================================================================
# Tokenizer modes
# ===================================================================
def _tokenizer_pair(seed: int = 0, cap: int = 10000):
    """Two tokenizers with identical weights, differing only in mode."""
    torch.manual_seed(seed)
    legacy = BatchedStateTokenizer(d_model=64, n_freq=8)
    log_rem = BatchedStateTokenizer(d_model=64, n_freq=8,
                                    time_feature="log_remaining",
                                    time_feature_cap=cap)
    log_rem.load_state_dict(legacy.state_dict())
    legacy.eval()
    log_rem.eval()
    return legacy, log_rem


class TestTokenizerTimeFeature:
    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="time_feature"):
            BatchedStateTokenizer(time_feature="remaining")

    def test_state_dict_identical_across_modes(self):
        legacy, log_rem = _tokenizer_pair()
        for k, v in legacy.state_dict().items():
            assert torch.equal(v, log_rem.state_dict()[k])

    def test_none_mode_is_time_blind(self):
        """'none': step_ratio/steps_remaining has zero effect on the
        embedding — two obs differing only in time must produce
        bit-identical token embeddings."""
        torch.manual_seed(0)
        legacy = BatchedStateTokenizer(d_model=64, n_freq=8)
        blind = BatchedStateTokenizer(d_model=64, n_freq=8,
                                      time_feature="none")
        blind.load_state_dict(legacy.state_dict())  # checkpoint-compatible (identical weight shapes)
        blind.eval()
        obs_kwargs = dict(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
        early = [make_mock_obs(step_ratio=0.01, steps_remaining=250,
                               **obs_kwargs)]
        late = [make_mock_obs(step_ratio=0.99, steps_remaining=1,
                              **obs_kwargs)]
        with torch.no_grad():
            out_early = blind(early).token_embeddings
            out_late = blind(late).token_embeddings
        assert torch.equal(out_early, out_late)
        # Contrast: in legacy (step_ratio) mode the same two obs must differ.
        with torch.no_grad():
            legacy.eval()
            l_early = legacy(early).token_embeddings
            l_late = legacy(late).token_embeddings
        assert not torch.equal(l_early, l_late)

    def test_log_remaining_matches_equivalent_step_ratio(self):
        """When log1p(remaining)/log1p(cap) equals the other obs's
        step_ratio, both modes must produce identical embeddings —
        proving the scalar lands in the same slot via the same math."""
        cap = 10000
        target = 0.42
        remaining = math.expm1(target * math.log1p(cap))
        legacy, log_rem = _tokenizer_pair(cap=cap)

        obs_kwargs = dict(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
        obs_legacy = [make_mock_obs(step_ratio=target, **obs_kwargs)]
        obs_logrem = [make_mock_obs(steps_remaining=remaining, **obs_kwargs)]
        with torch.no_grad():
            out_legacy = legacy(obs_legacy)
            out_logrem = log_rem(obs_logrem)
        diff = (out_legacy.token_embeddings
                - out_logrem.token_embeddings).abs().max().item()
        assert diff < 1e-6

    def test_log_remaining_ignores_step_ratio(self):
        """In log_remaining mode the obs's step_ratio must be inert."""
        _, log_rem = _tokenizer_pair()
        kwargs = dict(n_nets=1, pads_per_net=1, n_ratsnest_per_net=0,
                      steps_remaining=500)
        a = [make_mock_obs(step_ratio=0.1, **kwargs)]
        b = [make_mock_obs(step_ratio=0.9, **kwargs)]
        with torch.no_grad():
            assert torch.equal(log_rem(a).token_embeddings,
                               log_rem(b).token_embeddings)

    def test_log_remaining_distinguishes_budgets(self):
        _, log_rem = _tokenizer_pair()
        kwargs = dict(n_nets=1, pads_per_net=1, n_ratsnest_per_net=0)
        a = [make_mock_obs(steps_remaining=10, **kwargs)]
        b = [make_mock_obs(steps_remaining=5000, **kwargs)]
        with torch.no_grad():
            assert not torch.equal(log_rem(a).token_embeddings,
                                   log_rem(b).token_embeddings)


# ===================================================================
# sin_remaining mode
# ===================================================================
def _sin_tokenizer(seed: int = 0, cap: int = 10000, n_freq: int = 8):
    torch.manual_seed(seed)
    tok = BatchedStateTokenizer(d_model=64, n_freq=n_freq,
                                time_feature="sin_remaining",
                                time_feature_cap=cap)
    tok.eval()
    return tok


class TestSinRemaining:
    def test_state_dict_identical_to_legacy(self):
        """The ladder is a non-persistent buffer and adds no weights, so
        state_dicts stay interchangeable across all three modes."""
        legacy, _ = _tokenizer_pair()
        sin = _sin_tokenizer()
        sin.load_state_dict(legacy.state_dict(), strict=True)
        legacy.load_state_dict(sin.state_dict(), strict=True)

    def test_ladder_top_rung_period_two_steps(self):
        """base^(2·n_freq−1) = cap ⇒ top-rung phase advances by π per
        remaining step — the ±1-step-resolution anchoring."""
        cap, n_freq = 10000, 8
        sin = _sin_tokenizer(cap=cap, n_freq=n_freq)
        freqs = sin.vocab.time_freqs
        assert freqs is not None and len(freqs) == 2 * n_freq
        assert freqs[-1].item() == pytest.approx(cap, rel=1e-5)

    def test_resolves_one_step_at_long_horizon(self):
        """The motivating fix: at remaining ≈ cap, one step must move the
        encoding by O(1) — where log_remaining's squash leaves it below
        the ladder's resolution."""
        cap = 10000
        _, log_rem = _tokenizer_pair(cap=cap)
        sin = _sin_tokenizer(cap=cap)
        kwargs = dict(n_nets=1, pads_per_net=1, n_ratsnest_per_net=0)
        a = [make_mock_obs(steps_remaining=cap - 1, **kwargs)]
        b = [make_mock_obs(steps_remaining=cap, **kwargs)]
        with torch.no_grad():
            d_sin = (sin(a).token_embeddings
                     - sin(b).token_embeddings).abs().max().item()
            d_log = (log_rem(a).token_embeddings
                     - log_rem(b).token_embeddings).abs().max().item()
        assert d_sin > 10 * d_log
        assert d_sin > 1e-2

    def test_ignores_step_ratio(self):
        sin = _sin_tokenizer()
        kwargs = dict(n_nets=1, pads_per_net=1, n_ratsnest_per_net=0,
                      steps_remaining=500)
        a = [make_mock_obs(step_ratio=0.1, **kwargs)]
        b = [make_mock_obs(step_ratio=0.9, **kwargs)]
        with torch.no_grad():
            assert torch.equal(sin(a).token_embeddings,
                               sin(b).token_embeddings)

    def test_distinguishes_budgets(self):
        sin = _sin_tokenizer()
        kwargs = dict(n_nets=1, pads_per_net=1, n_ratsnest_per_net=0)
        a = [make_mock_obs(steps_remaining=10, **kwargs)]
        b = [make_mock_obs(steps_remaining=5000, **kwargs)]
        with torch.no_grad():
            assert not torch.equal(sin(a).token_embeddings,
                                   sin(b).token_embeddings)


# ===================================================================
# Config / checkpoint compat
# ===================================================================
class TestPolicyConfigTimeFeature:
    def test_old_checkpoint_defaults_to_step_ratio(self):
        cfg = RLPolicyConfig.from_checkpoint({})
        assert cfg.time_feature == "step_ratio"
        assert cfg.time_feature_cap == 10000

    def test_checkpoint_roundtrip(self):
        cfg = RLPolicyConfig.from_checkpoint(
            {"time_feature": "log_remaining", "time_feature_cap": 2048})
        assert cfg.time_feature == "log_remaining"
        assert cfg.time_feature_cap == 2048

    def test_checkpoint_roundtrip_sin_remaining(self):
        cfg = RLPolicyConfig.from_checkpoint(
            {"time_feature": "sin_remaining", "time_feature_cap": 4096})
        assert cfg.time_feature == "sin_remaining"
        assert cfg.time_feature_cap == 4096

    def test_weights_compatible_across_modes(self):
        """A default-mode state_dict must load strict into a log_remaining
        or sin_remaining model — the guarantee that lets old KDD
        checkpoints keep working."""
        from methods.rl_agent.models.v1.net import KiCadRLModel

        common = dict(d_model=32, n_heads=4, n_layers=1, d_ff=64, n_freq=4)
        old = KiCadRLModel(**common)
        for mode in ("log_remaining", "sin_remaining"):
            new = KiCadRLModel(time_feature=mode,
                               time_feature_cap=2048, **common)
            new.load_state_dict(old.state_dict(), strict=True)
