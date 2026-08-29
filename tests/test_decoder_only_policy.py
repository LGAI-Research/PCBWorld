"""Tests for methods.rl_agent.models.v1.net.

No C++ dependency — pure PyTorch tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from methods.rl_agent.models.v1.net import (
    ACT_FINISH,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    NUM_ACTION_TYPES,
    SLOT_USAGE,
    KiCadRLModel,
)
from methods.rl_agent.models.v1.blocks import (
    GatedTransformerLayer,
    MultiHeadAttention,
    ReZero,
    build_2zone_mask,
    combine_masks,
    init_weights,
)
from tests._mock_obs import make_mock_obs


# ===================================================================
# TestBuild2ZoneMask
# ===================================================================
class TestBuild2ZoneMask:
    def test_shape(self):
        mask = build_2zone_mask(15, 20)
        assert mask.shape == (20, 20)

    def test_state_bidirectional(self):
        mask = build_2zone_mask(15, 20)
        assert (mask[:15, :15] == 0.0).all()

    def test_state_cant_see_action(self):
        mask = build_2zone_mask(15, 20)
        assert (mask[:15, 15:20] == float("-inf")).all()

    def test_action_sees_state(self):
        mask = build_2zone_mask(15, 20)
        assert (mask[15:, :15] == 0.0).all()

    def test_action_causal(self):
        mask = build_2zone_mask(15, 20)
        for i in range(15, 20):
            assert (mask[i, 15 : i + 1] == 0.0).all()
            if i + 1 < 20:
                assert (mask[i, i + 1 : 20] == float("-inf")).all()

    def test_no_action_zone(self):
        mask = build_2zone_mask(15, 15)
        assert mask.shape == (15, 15)
        assert (mask == 0.0).all()

    def test_minimal_zones(self):
        mask = build_2zone_mask(2, 3)
        assert mask.shape == (3, 3)
        # State [0,1] bidirectional
        assert mask[0, 1] == 0.0
        assert mask[1, 0] == 0.0
        # State can't see action [2]
        assert mask[0, 2] == float("-inf")
        assert mask[1, 2] == float("-inf")
        # Action [2] sees everything, no future
        assert mask[2, 0] == 0.0
        assert mask[2, 1] == 0.0
        assert mask[2, 2] == 0.0


# ===================================================================
# TestCombineMasks
# ===================================================================
class TestCombineMasks:
    def test_shape(self):
        zone = build_2zone_mask(8, 12)
        kpm = torch.tensor([[False] * 10 + [True] * 2, [False] * 12])
        combined = combine_masks(zone, kpm)
        assert combined.shape == (2, 1, 12, 12)

    def test_padded_columns_blocked(self):
        zone = torch.zeros(5, 5)
        kpm = torch.tensor([[False, False, False, True, True]])
        combined = combine_masks(zone, kpm)
        # Columns 3 and 4 should be -inf for all rows
        assert (combined[0, 0, :, 3:5] == float("-inf")).all()
        # Columns 0-2 should be 0
        assert (combined[0, 0, :, :3] == 0.0).all()

    def test_no_padding_preserves_zone(self):
        zone = build_2zone_mask(4, 6)
        kpm = torch.zeros(1, 6, dtype=torch.bool)
        combined = combine_masks(zone, kpm)
        assert torch.allclose(combined[0, 0], zone)


# ===================================================================
# TestReZero
# ===================================================================
class TestReZero:
    def test_alpha_init_zero(self):
        rz = ReZero()
        assert rz.alpha.shape == (1,)
        assert rz.alpha.item() == pytest.approx(0.0)

    def test_initial_pass_through_exact(self):
        """alpha=0 → output identical to residual input x."""
        rz = ReZero()
        x = torch.randn(4, 10, 64)
        y = torch.randn(4, 10, 64)
        out = rz(x, y)
        assert torch.equal(out, x)

    def test_after_training_blends_y(self):
        """Once alpha drifts off zero the sublayer y starts contributing."""
        rz = ReZero()
        with torch.no_grad():
            rz.alpha.fill_(0.5)
        x = torch.zeros(2, 4, 16)
        y = torch.ones(2, 4, 16)
        out = rz(x, y)
        assert torch.allclose(out, 0.5 * y)

    def test_output_shape(self):
        rz = ReZero()
        x = torch.randn(3, 7, 32)
        y = torch.randn(3, 7, 32)
        assert rz(x, y).shape == (3, 7, 32)

    def test_gradient_flows(self):
        rz = ReZero()
        x = torch.randn(2, 5, 32, requires_grad=True)
        y = torch.randn(2, 5, 32, requires_grad=True)
        out = rz(x, y)
        out.sum().backward()
        assert x.grad is not None
        assert y.grad is not None
        assert rz.alpha.grad is not None

    def test_param_count(self):
        """ReZero adds exactly 1 scalar — vs GRUGate's 6·d² + d."""
        rz = ReZero()
        total = sum(p.numel() for p in rz.parameters())
        assert total == 1


# ===================================================================
# TestMultiHeadAttention (standard SDPA, no RoPE)
# ===================================================================
class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=128, n_heads=4)
        x = torch.randn(2, 20, 128)
        out = attn(x)
        assert out.shape == (2, 20, 128)

    def test_with_2d_attn_mask(self):
        attn = MultiHeadAttention(d_model=128, n_heads=4)
        x = torch.randn(2, 10, 128)
        mask = build_2zone_mask(7, 10)
        out = attn(x, attn_mask=mask)
        assert out.shape == (2, 10, 128)

    def test_with_4d_attn_mask(self):
        attn = MultiHeadAttention(d_model=128, n_heads=4)
        x = torch.randn(2, 10, 128)
        zone = build_2zone_mask(7, 10)
        kpm = torch.zeros(2, 10, dtype=torch.bool)
        mask = combine_masks(zone, kpm)
        out = attn(x, attn_mask=mask)
        assert out.shape == (2, 10, 128)

    def test_gradient_flow(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_set_permutation_invariant(self):
        """Without RoPE the attention output is permutation-equivariant
        (proven by checking output for shuffled inputs is the shuffle of
        the original output). This is the property the state-zone design
        relies on: state tokens have no positional bias.
        """
        torch.manual_seed(0)
        attn = MultiHeadAttention(d_model=32, n_heads=4)
        attn.eval()
        x = torch.randn(1, 5, 32)
        perm = torch.tensor([2, 0, 4, 1, 3])
        out = attn(x)
        out_perm = attn(x[:, perm])
        assert torch.allclose(out_perm, out[:, perm], atol=1e-5)


# ===================================================================
# TestGatedTransformerLayer
# ===================================================================
class TestGatedTransformerLayer:
    def test_output_shape(self):
        layer = GatedTransformerLayer(128, 4, 512)
        x = torch.randn(2, 20, 128)
        assert layer(x).shape == (2, 20, 128)

    def test_with_2zone_mask(self):
        layer = GatedTransformerLayer(64, 4, 256)
        x = torch.randn(2, 15, 64)
        mask = build_2zone_mask(10, 15)
        out = layer(x, attn_mask=mask)
        assert out.shape == (2, 15, 64)

    def test_initial_output_near_identity(self):
        """ReZero alpha=0 → first forward is exact identity over the residual."""
        layer = GatedTransformerLayer(64, 4, 256)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert torch.allclose(out, x)

    def test_gradient_flow(self):
        layer = GatedTransformerLayer(64, 4, 256)
        x = torch.randn(2, 10, 64, requires_grad=True)
        # Open the ReZero gates so the sublayer gradient flows.
        with torch.no_grad():
            layer.res_attn.alpha.fill_(0.5)
            layer.res_ff.alpha.fill_(0.5)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_stacked_6_layers(self):
        """6 layers stacked (production config) should produce correct shapes."""
        layers = nn.ModuleList(
            [GatedTransformerLayer(128, 4, 512) for _ in range(6)]
        )
        x = torch.randn(2, 20, 128)
        mask = build_2zone_mask(15, 20)
        for layer in layers:
            x = layer(x, attn_mask=mask)
        assert x.shape == (2, 20, 128)

    def test_stacked_6_layers_near_identity(self):
        """ReZero alpha=0 stack: every layer is identity, so cumulative drift = 0."""
        layers = nn.ModuleList(
            [GatedTransformerLayer(128, 4, 512) for _ in range(6)]
        )
        x_orig = torch.randn(2, 20, 128)
        x = x_orig.clone()
        for layer in layers:
            x = layer(x)
        assert torch.allclose(x, x_orig)

    def test_with_combined_mask(self):
        """Full pipeline: 2zone + padding → combined → layer."""
        layer = GatedTransformerLayer(64, 4, 256)
        B, L = 2, 12
        x = torch.randn(B, L, 64)
        zone = build_2zone_mask(8, L)
        kpm = torch.tensor([[False] * 10 + [True] * 2, [False] * 12])
        mask = combine_masks(zone, kpm)
        out = layer(x, attn_mask=mask)
        assert out.shape == (B, L, 64)


# ===================================================================
# TestInitWeights
# ===================================================================
class TestInitWeights:
    def test_preserves_rezero_alpha(self):
        rz = ReZero()
        init_weights(rz)
        assert rz.alpha.item() == pytest.approx(0.0)

    def test_preserves_rezero_alpha_in_layer(self):
        layer = GatedTransformerLayer(64, 4, 256)
        init_weights(layer)
        assert layer.res_attn.alpha.item() == pytest.approx(0.0)
        assert layer.res_ff.alpha.item() == pytest.approx(0.0)

    def test_policy_head_small_gain(self):
        """Modules whose name contains 'policy_head' get gain=0.01."""
        model = nn.Module()
        model.action_policy_head = nn.Linear(32, 6)
        init_weights(model)
        assert model.action_policy_head.weight.abs().max().item() < 0.1

    def test_non_policy_head_default_gain(self):
        """Other heads (e.g. critic_head) get the default gain=1.0."""
        model = nn.Module()
        model.critic_head = nn.Linear(32, 1)
        init_weights(model)
        # gain=1.0 orthogonal init produces values much larger than 0.01.
        assert model.critic_head.weight.abs().max().item() > 0.1


# ===================================================================
# TestKiCadRLModel
# ===================================================================
def _tiny_policy() -> KiCadRLModel:
    """Small policy suitable for fast tests."""
    return KiCadRLModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=2000,
        n_freq=4,
    )


def _batch_obs(n_nets: int = 2, is_routing: bool = False,
               current_net_phase: int = 1) -> list[dict]:
    """Build a 2-element batch of identical mock observations."""
    obs = make_mock_obs(
        n_nets=n_nets,
        pads_per_net=2,
        n_ratsnest_per_net=1,
        is_routing=is_routing,
        current_net_phase=current_net_phase,
        current_layer=1 if current_net_phase > 0 else -1,
    )
    return [obs, obs]


class TestKiCadRLModelStructure:
    def test_construction(self):
        policy = _tiny_policy()
        # Basic attribute presence
        assert hasattr(policy, "tokenizer")
        # No RoPE; the action zone uses a learned action_pos_emb.
        assert not hasattr(policy, "rope")
        assert hasattr(policy, "action_pos_emb")
        assert policy.action_pos_emb.shape == (2, policy.d_model)
        assert len(policy.layers) == 2
        assert policy.action_type_head.num_embeddings == NUM_ACTION_TYPES

    def test_slot_usage_shape(self):
        assert SLOT_USAGE.shape == (NUM_ACTION_TYPES, 2)
        assert SLOT_USAGE.dtype == torch.bool
        # net_end has no params
        assert not SLOT_USAGE[ACT_NET_END, 0]
        assert not SLOT_USAGE[ACT_NET_END, 1]
        # make_line / make_via need both pointer and mode
        assert SLOT_USAGE[ACT_MAKE_LINE, 0] and SLOT_USAGE[ACT_MAKE_LINE, 1]
        assert SLOT_USAGE[ACT_MAKE_VIA, 0] and SLOT_USAGE[ACT_MAKE_VIA, 1]
        # finish needs only mode
        assert not SLOT_USAGE[ACT_FINISH, 0]
        assert SLOT_USAGE[ACT_FINISH, 1]
        # Full table pinned — guards the ACTION_REGISTRY-derived construction
        # against drift from the params definitions.
        expected = torch.tensor(
            [
                [True, False],   # net_select:  net pointer only
                [True, False],   # start_route: cand pointer only
                [False, False],  # net_end:     nothing
                [True, True],    # make_line:   cand pointer + mode
                [True, True],    # make_via:    cand pointer + mode
                [False, True],   # finish:      mode only
                [False, False],  # idle:        nothing (LLM no-op)
            ],
            dtype=torch.bool,
        )
        assert torch.equal(SLOT_USAGE, expected)

    def test_action_type_head_init_small(self):
        policy = _tiny_policy()
        # gain=0.01 → near-uniform initial logits
        assert policy.action_type_head.weight.abs().max().item() < 0.1

    def test_parameter_count_reasonable(self):
        policy = _tiny_policy()
        n_params = sum(p.numel() for p in policy.parameters())
        # Sanity: should be > 10k (trivial) and < 5M (not huge).
        assert 10_000 < n_params < 5_000_000


class TestKiCadRLModelDrcTokens:
    """Forward/backward still succeed when obs carry DRC violation tokens."""

    @staticmethod
    def _drc(x, y, type_id=0, severity=0x20, nets=("NET1",)):
        return {
            "x_mm": x, "y_mm": y, "layer": 1,
            "error_type": "Clearance violation",
            "type_id": type_id, "severity": severity,
            "net_names": list(nets),
        }

    def test_forward_with_drc(self):
        policy = _tiny_policy()
        obs_no = _batch_obs(current_net_phase=2, is_routing=True)
        obs_yes = _batch_obs(current_net_phase=2, is_routing=True)
        for o in obs_yes:
            o["drc_violations"] = [
                self._drc(112.0, 58.0, type_id=0, severity=0x20),
                self._drc(125.0, 65.0, type_id=2, severity=0x10, nets=[]),
            ]
        # Both should forward without error.
        a_no, _ = policy.act(obs_no)
        a_yes, _ = policy.act(obs_yes)
        assert a_no.shape == a_yes.shape

    def test_backward_with_drc(self):
        policy = _tiny_policy()
        # Robustness to prior-test RNG pollution: policy-wide
        # nn.init.orthogonal_ consumes the global generator in module
        # iteration order, so drc_proj can land on a near-degenerate
        # matrix depending on test ordering — producing zero gradient
        # even though the forward path is correct. Overwrite with a
        # locally-seeded small-normal fill so this check is independent
        # of global RNG state.
        drc_proj = policy.tokenizer.vocab.drc_proj
        with torch.no_grad():
            gen = torch.Generator().manual_seed(0)
            drc_proj.weight.normal_(mean=0.0, std=0.1, generator=gen)
            if drc_proj.bias is not None:
                drc_proj.bias.zero_()
        policy.train()
        obs = _batch_obs(current_net_phase=2, is_routing=True)
        for i, o in enumerate(obs):
            o["drc_violations"] = [
                self._drc(105.0 + i, 55.0, type_id=0, severity=0x20),
                self._drc(115.0, 62.0, type_id=6, severity=0x20, nets=[]),
            ]
        # Take gradient through the tokenizer directly (act() samples with
        # a detached categorical, which drops the grad graph).
        out = policy.tokenizer(obs)
        valid = (~out.key_padding_mask).unsqueeze(-1).to(out.token_embeddings.dtype)
        # Use a squared-magnitude loss: the final embed_ln in TokenVocabulary
        # forces each token to zero mean across d_model, so the plain sum
        # ``(emb * valid).sum()`` is structurally ~0 and the gradient becomes
        # pure FP-rounding noise — sometimes exactly 0.0 depending on prior
        # RNG state. Squaring gives a non-vanishing loss.
        loss = (out.token_embeddings.pow(2) * valid).sum()
        loss.backward()
        g = policy.tokenizer.vocab.drc_proj.weight.grad
        assert g is not None
        assert torch.isfinite(g).all()
        assert g.abs().sum().item() > 0

    def test_empty_drc_list_is_noop(self):
        policy = _tiny_policy()
        obs_a = _batch_obs(current_net_phase=2, is_routing=True)
        obs_b = _batch_obs(current_net_phase=2, is_routing=True)
        for o in obs_b:
            o["drc_violations"] = []
        with torch.no_grad():
            a = policy.tokenizer(obs_a)
            b = policy.tokenizer(obs_b)
        assert a.token_embeddings.shape == b.token_embeddings.shape
        assert torch.allclose(a.token_embeddings, b.token_embeddings)


class TestKiCadRLModelAct:
    def test_act_output_shape_and_types(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=1, is_routing=False)
        actions, log_probs = policy.act(obs_list)
        assert actions.shape == (2, 3)
        assert actions.dtype == torch.int64
        assert log_probs.shape == (2,)
        assert log_probs.dtype == torch.float32
        # action_type in range
        assert ((actions[:, 0] >= 0) & (actions[:, 0] < NUM_ACTION_TYPES)).all()
        # Unused slots → -1
        for b in range(2):
            at = int(actions[b, 0])
            needs_ptr = bool(SLOT_USAGE[at, 0])
            needs_mode = bool(SLOT_USAGE[at, 1])
            if not needs_ptr:
                assert int(actions[b, 1]) == -1
            if not needs_mode:
                assert int(actions[b, 2]) == -1

    def test_act_respects_action_mask(self):
        """When only ACT_NET_SELECT is allowed, all sampled action_types must be 0."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=0, is_routing=False)
        mask = torch.zeros(2, NUM_ACTION_TYPES, dtype=torch.bool)
        mask[:, ACT_NET_SELECT] = True
        actions, _ = policy.act(obs_list, action_masks=mask)
        assert (actions[:, 0] == ACT_NET_SELECT).all()
        # net_select needs pointer, not mode
        assert (actions[:, 1] >= 0).all()
        assert (actions[:, 2] == -1).all()

    def test_act_forces_make_line(self):
        """Force make_line → pointer and mode must both be set."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        mask = torch.zeros(2, NUM_ACTION_TYPES, dtype=torch.bool)
        mask[:, ACT_MAKE_LINE] = True
        actions, _ = policy.act(obs_list, action_masks=mask)
        assert (actions[:, 0] == ACT_MAKE_LINE).all()
        assert (actions[:, 1] >= 0).all()
        assert ((actions[:, 2] >= 0) & (actions[:, 2] < 3)).all()

    def test_act_forces_net_end(self):
        """Force net_end → both pointer and mode slots should be -1."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=1, is_routing=False)
        mask = torch.zeros(2, NUM_ACTION_TYPES, dtype=torch.bool)
        mask[:, ACT_NET_END] = True
        actions, _ = policy.act(obs_list, action_masks=mask)
        assert (actions[:, 0] == ACT_NET_END).all()
        assert (actions[:, 1] == -1).all()
        assert (actions[:, 2] == -1).all()

    def test_act_forces_finish(self):
        """Force finish → pointer is -1, mode is set."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        mask = torch.zeros(2, NUM_ACTION_TYPES, dtype=torch.bool)
        mask[:, ACT_FINISH] = True
        actions, _ = policy.act(obs_list, action_masks=mask)
        assert (actions[:, 0] == ACT_FINISH).all()
        assert (actions[:, 1] == -1).all()
        assert ((actions[:, 2] >= 0) & (actions[:, 2] < 3)).all()

    def test_act_deterministic_is_reproducible(self):
        policy = _tiny_policy()
        policy.eval()
        obs_list = _batch_obs(current_net_phase=1)
        a1, lp1 = policy.act(obs_list, deterministic=True)
        a2, lp2 = policy.act(obs_list, deterministic=True)
        assert torch.equal(a1, a2)
        assert torch.allclose(lp1, lp2)

    def test_act_log_prob_finite(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=1)
        _, log_probs = policy.act(obs_list)
        assert torch.isfinite(log_probs).all()
        # log_prob of discrete action must be <= 0
        assert (log_probs <= 1e-5).all()


class TestKiCadRLModelEvaluate:
    def test_evaluate_output_shape_and_types(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=1)
        actions, _ = policy.act(obs_list)
        log_probs, entropy = policy.evaluate(obs_list, actions)
        assert log_probs.shape == (2,)
        assert entropy.shape == (2,)
        assert log_probs.dtype == torch.float32
        assert entropy.dtype == torch.float32

    def test_evaluate_finite_and_entropy_nonneg(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=1)
        actions, _ = policy.act(obs_list)
        log_probs, entropy = policy.evaluate(obs_list, actions)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()
        assert (entropy >= -1e-5).all()
        assert (log_probs <= 1e-5).all()

    def test_evaluate_gradient_flow(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=1)
        actions, _ = policy.act(obs_list)
        log_probs, entropy = policy.evaluate(obs_list, actions)
        loss = -(log_probs.mean()) - 0.01 * entropy.mean()
        loss.backward()
        # action_type_head must have a gradient
        assert policy.action_type_head.weight.grad is not None
        assert policy.action_type_head.weight.grad.abs().sum().item() > 0
        # Transformer backbone must have gradients on some params
        total_backbone_grad = 0.0
        for name, p in policy.layers.named_parameters():
            if p.grad is not None:
                total_backbone_grad += p.grad.abs().sum().item()
        assert total_backbone_grad > 0.0

    def test_evaluate_net_end_action_safe(self):
        """net_end has pointer=-1 and mode=-1 — must not crash or produce NaN."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=1)
        # Hand-craft net_end actions
        actions = torch.tensor(
            [[ACT_NET_END, -1, -1], [ACT_NET_END, -1, -1]], dtype=torch.int64,
        )
        log_probs, entropy = policy.evaluate(obs_list, actions)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()

    def test_evaluate_finish_action_safe(self):
        """finish: pointer=-1, mode in [0,3)."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_FINISH, -1, 0], [ACT_FINISH, -1, 2]], dtype=torch.int64,
        )
        log_probs, entropy = policy.evaluate(obs_list, actions)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()

    def test_evaluate_make_line_action(self):
        """make_line: pointer (cand) + mode both used."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )
        log_probs, entropy = policy.evaluate(obs_list, actions)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()

    def test_act_evaluate_log_prob_consistency(self):
        """act() deterministic + evaluate() with the same action → log_probs match."""
        policy = _tiny_policy()
        policy.eval()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions, act_log_probs = policy.act(obs_list, deterministic=True)
        eval_log_probs, _ = policy.evaluate(obs_list, actions)
        assert torch.allclose(act_log_probs, eval_log_probs, atol=1e-5), (
            f"act log_probs {act_log_probs} vs evaluate {eval_log_probs}"
        )


class TestKiCadRLModelIdle:
    def test_idle_obs_net_select_only(self):
        """Idle observation: net_phase=0, no candidates. net_select must work."""
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=0, is_routing=False)
        mask = torch.zeros(2, NUM_ACTION_TYPES, dtype=torch.bool)
        mask[:, ACT_NET_SELECT] = True
        actions, log_probs = policy.act(obs_list, action_masks=mask)
        assert (actions[:, 0] == ACT_NET_SELECT).all()
        assert torch.isfinite(log_probs).all()

    def test_idle_obs_evaluate(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=0, is_routing=False)
        # net_select with pointer_idx=0 (first net)
        actions = torch.tensor(
            [[ACT_NET_SELECT, 0, -1], [ACT_NET_SELECT, 0, -1]], dtype=torch.int64,
        )
        log_probs, entropy = policy.evaluate(obs_list, actions)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()


# ===================================================================
# Critic head & act_and_value / evaluate_actions_and_value
# ===================================================================
def _tiny_policy_with_critic() -> KiCadRLModel:
    """Small policy with PPO critic head enabled."""
    return KiCadRLModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=2000,
        n_freq=4,
        use_critic=True,
    )


class TestCriticHead:
    def test_no_critic_attribute_when_disabled(self):
        policy = _tiny_policy()
        assert policy.use_critic is False
        assert not hasattr(policy, "critic_head")

    def test_critic_attribute_when_enabled(self):
        policy = _tiny_policy_with_critic()
        assert policy.use_critic is True
        assert hasattr(policy, "critic_head")
        # critic_head should be a non-trivial nn.Module with parameters
        n_params = sum(p.numel() for p in policy.critic_head.parameters())
        assert n_params > 0

    def test_critic_adds_parameters(self):
        a = sum(p.numel() for p in _tiny_policy().parameters())
        b = sum(p.numel() for p in _tiny_policy_with_critic().parameters())
        assert b > a, f"critic should add params (no-critic={a}, critic={b})"

    def test_compute_value_without_critic_returns_zeros(self):
        policy = _tiny_policy()
        h = torch.randn(4, policy.d_model)
        v = policy._compute_value(h)
        assert v.shape == (4,)
        assert torch.all(v == 0)
        assert v.requires_grad is False

    def test_compute_value_with_critic_returns_finite(self):
        policy = _tiny_policy_with_critic()
        h = torch.randn(4, policy.d_model, requires_grad=True)
        v = policy._compute_value(h)
        assert v.shape == (4,)
        assert torch.isfinite(v).all()


class TestActAndValue:
    def test_act_and_value_shapes_no_critic(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions, log_probs, values = policy.act_and_value(obs_list)
        assert actions.shape == (2, 3)
        assert log_probs.shape == (2,)
        assert values.shape == (2,)
        assert torch.all(values == 0)
        assert torch.isfinite(log_probs).all()

    def test_act_and_value_shapes_with_critic(self):
        policy = _tiny_policy_with_critic()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions, log_probs, values = policy.act_and_value(obs_list)
        assert actions.shape == (2, 3)
        assert log_probs.shape == (2,)
        assert values.shape == (2,)
        assert torch.isfinite(values).all()

    def test_act_wrapper_matches_act_and_value(self):
        """act() must produce identical actions/log_probs to act_and_value()."""
        torch.manual_seed(0)
        policy = _tiny_policy()
        policy.eval()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)

        torch.manual_seed(123)
        a1, lp1 = policy.act(obs_list, deterministic=True)
        torch.manual_seed(123)
        a2, lp2, _ = policy.act_and_value(obs_list, deterministic=True)

        assert torch.equal(a1, a2)
        assert torch.allclose(lp1, lp2)


class TestEvaluateActionsAndValue:
    def test_evaluate_and_value_shapes_no_critic(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )
        log_probs, entropy, values = policy.evaluate_actions_and_value(
            obs_list, actions,
        )
        assert log_probs.shape == (2,)
        assert entropy.shape == (2,)
        assert values.shape == (2,)
        assert torch.all(values == 0)

    def test_evaluate_and_value_shapes_with_critic(self):
        policy = _tiny_policy_with_critic()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )
        log_probs, entropy, values = policy.evaluate_actions_and_value(
            obs_list, actions,
        )
        assert log_probs.shape == (2,)
        assert entropy.shape == (2,)
        assert values.shape == (2,)
        assert torch.isfinite(values).all()

    def test_evaluate_wrapper_matches_evaluate_and_value(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )
        lp1, ent1 = policy.evaluate(obs_list, actions)
        lp2, ent2, _ = policy.evaluate_actions_and_value(obs_list, actions)
        assert torch.allclose(lp1, lp2)
        assert torch.allclose(ent1, ent2)


class TestEntropyNorm:
    """--entropy-norm: joint entropy / max achievable entropy (ln N_valid)."""

    def _eval(self, policy, obs_list, actions, **kw):
        return policy.evaluate_actions_and_value(obs_list, actions, **kw)

    def test_normalized_in_unit_interval_and_lp_values_unchanged(self):
        policy = _tiny_policy_with_critic()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )
        lp_raw, ent_raw, v_raw = self._eval(policy, obs_list, actions)
        lp_n, ent_n, v_n = self._eval(
            policy, obs_list, actions, entropy_norm=True,
        )
        # Normalization touches ONLY the entropy channel.
        assert torch.allclose(lp_raw, lp_n)
        assert torch.allclose(v_raw, v_n)
        assert torch.all(ent_n >= 0.0)
        assert torch.all(ent_n <= 1.0 + 1e-5)
        # Identical obs rows share one N_valid layout -> identical denominator.
        denom = ent_raw / ent_n
        assert torch.allclose(denom[0], denom[1], rtol=1e-5)
        # Denominator is a real ln(N_valid) sum: > ln(2) for this open state.
        assert denom[0].item() > 0.5

    def test_single_valid_action_row_is_zero_not_nan(self):
        # Mask down to one valid action type (net_end: no ptr, no mode slot)
        # -> max_ent == 0 -> normalized entropy must be exactly 0, not NaN.
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_NET_END, -1, -1], [ACT_NET_END, -1, -1]], dtype=torch.int64,
        )
        masks = torch.zeros(2, NUM_ACTION_TYPES, dtype=torch.bool)
        masks[:, ACT_NET_END] = True
        _, ent_n, _ = self._eval(
            policy, obs_list, actions, action_masks=masks, entropy_norm=True,
        )
        assert torch.isfinite(ent_n).all()
        assert torch.allclose(ent_n, torch.zeros_like(ent_n), atol=1e-6)

    def test_mixed_deterministic_row_backward_finite_bf16(self):
        # A deterministic row (max_ent == 0) mixed into a batch keeps its
        # denominator at 1, so the eps-clamped division has no channel to
        # amplify bf16 residual gradients into inf/NaN (fwd 0, bwd finite).
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_NET_END, -1, -1], [ACT_MAKE_LINE, 0, 1]], dtype=torch.int64,
        )
        masks = torch.ones(2, NUM_ACTION_TYPES, dtype=torch.bool)
        masks[0] = False
        masks[0, ACT_NET_END] = True   # row0: only one valid action -> max_ent == 0
        policy.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, ent_n, _ = policy.evaluate_actions_and_value(
                obs_list, actions, action_masks=masks, entropy_norm=True,
            )
        assert torch.isfinite(ent_n).all()
        assert ent_n[0].item() == pytest.approx(0.0, abs=1e-5)
        ent_n.sum().backward()
        for p in policy.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_gradient_flows_through_normalized_entropy(self):
        policy = _tiny_policy()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )
        policy.zero_grad(set_to_none=True)
        _, ent_n, _ = self._eval(
            policy, obs_list, actions, entropy_norm=True,
        )
        ent_n.mean().backward()
        grads = [
            p.grad for p in policy.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        ]
        assert grads, "normalized entropy must remain differentiable"
        assert all(torch.isfinite(g).all() for g in grads)


class TestCriticGradientFlow:
    """Standard PPO: value loss flows back to the transformer backbone."""

    def _backbone_modules(self, policy: KiCadRLModel):
        """All backbone parameters that should receive value-loss grads."""
        backbone_params = []
        backbone_params += list(policy.tokenizer.parameters())
        backbone_params += list(policy.layers.parameters())
        backbone_params += list(policy.action_type_head.parameters())
        return backbone_params

    def test_act_and_value_uses_no_grad(self):
        """act_and_value is decorated @torch.no_grad — used for rollouts."""
        policy = _tiny_policy_with_critic()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        _, log_probs, values = policy.act_and_value(obs_list)
        # Both should be detached — they're stored as old_log_probs / old_values
        # in the rollout buffer and re-evaluated under the current policy later.
        assert log_probs.requires_grad is False
        assert values.requires_grad is False

    def test_value_backward_flows_to_backbone(self):
        """Standard PPO: value loss reaches the transformer backbone."""
        policy = _tiny_policy_with_critic()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )

        policy.zero_grad(set_to_none=True)
        _, _, values = policy.evaluate_actions_and_value(obs_list, actions)
        values.sum().backward()

        # Critic head should have grads.
        critic_grads = [
            p.grad for p in policy.critic_head.parameters() if p.grad is not None
        ]
        assert len(critic_grads) > 0
        assert any(g.abs().sum().item() > 0 for g in critic_grads)

        # Backbone should ALSO have grads (gradient is NOT detached).
        backbone = self._backbone_modules(policy)
        any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in backbone
        )
        assert any_grad, (
            "Value loss did not reach backbone — should be the case for "
            "standard PPO with shared trunk."
        )

    def test_policy_loss_still_flows_to_backbone(self):
        """Sanity: policy gradients (log_prob) flow into the backbone."""
        policy = _tiny_policy_with_critic()
        obs_list = _batch_obs(current_net_phase=2, is_routing=True)
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )

        policy.zero_grad(set_to_none=True)
        log_probs, _, _ = policy.evaluate_actions_and_value(obs_list, actions)
        log_probs.sum().backward()

        # At least some backbone params should now have grads.
        backbone = self._backbone_modules(policy)
        any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in backbone
        )
        assert any_grad, "Policy loss did not reach backbone — backbone params must receive gradients"

    def test_critic_intermediate_layers_have_normal_gain(self):
        """Multi-layer critic MLP must NOT be shrunk to gain=0.01.

        An init_weights pattern matching the bare 'head' substring would shrink
        every critic_head.* Linear.
        """
        policy = _tiny_policy_with_critic()
        # The first Linear in critic_head (index 1, after LayerNorm at 0).
        first_linear = policy.critic_head[1]
        assert isinstance(first_linear, nn.Linear)
        # gain=0.01 orthogonal init produces values < 0.05; gain=1.0 is much larger.
        assert first_linear.weight.abs().max().item() > 0.1, (
            "critic_head intermediate Linear was init'd with small gain — "
            "init_weights pattern is too greedy."
        )


# ===================================================================
# Variable seq_lens batching (per-row SOD extraction)
# ===================================================================
def _batch_obs_varying_seq_lens() -> list[dict]:
    """Two obs with different seq_lens.

    Varies ``n_ratsnest_per_net`` (dynamic tokens) while keeping
    ``n_nets`` / ``pads_per_net`` constant.
    """
    obs1 = make_mock_obs(
        n_nets=2, pads_per_net=2,
        n_ratsnest_per_net=1,
        is_routing=True,
        current_net_phase=2, current_layer=1,
    )
    obs2 = make_mock_obs(
        n_nets=2, pads_per_net=2,
        n_ratsnest_per_net=3,
        is_routing=True,
        current_net_phase=2, current_layer=1,
    )
    return [obs1, obs2]


class TestVariableSeqLenBatch:
    """Policy must accept batches with varying seq_lens.

    ``_extract_scalar_bounds`` returns ``n_state_max`` and SOD is extracted
    per-row via ``H_state[arange_B, seq_lens - 1]``.
    """

    def test_tokenizer_produces_different_seq_lens(self):
        policy = _tiny_policy()
        obs_list = _batch_obs_varying_seq_lens()
        tok_out = policy.tokenizer(obs_list)
        assert tok_out.seq_lens[0].item() != tok_out.seq_lens[1].item()

    def test_act_and_value_handles_varying_seq_lens(self):
        policy = _tiny_policy()
        policy.eval()
        obs_list = _batch_obs_varying_seq_lens()
        actions, log_probs, values = policy.act_and_value(obs_list)
        assert actions.shape == (2, 3)
        assert log_probs.shape == (2,)
        assert values.shape == (2,)
        assert torch.isfinite(log_probs).all()

    def test_act_and_value_with_critic_handles_varying_seq_lens(self):
        policy = _tiny_policy_with_critic()
        policy.eval()
        obs_list = _batch_obs_varying_seq_lens()
        actions, log_probs, values = policy.act_and_value(obs_list)
        assert actions.shape == (2, 3)
        assert torch.isfinite(values).all()

    def test_evaluate_actions_and_value_handles_varying_seq_lens(self):
        policy = _tiny_policy_with_critic()
        obs_list = _batch_obs_varying_seq_lens()
        actions = torch.tensor(
            [[ACT_MAKE_LINE, 0, 1], [ACT_MAKE_LINE, 0, 2]], dtype=torch.int64,
        )
        log_probs, entropy, values = policy.evaluate_actions_and_value(
            obs_list, actions,
        )
        assert log_probs.shape == (2,)
        assert entropy.shape == (2,)
        assert values.shape == (2,)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(values).all()

    def test_batched_act_matches_per_row_act(self):
        """Sanity: forwarding 2 obs as a batch must yield results
        consistent with forwarding each obs individually (deterministic mode).
        """
        torch.manual_seed(42)
        policy = _tiny_policy_with_critic()
        policy.eval()
        obs_list = _batch_obs_varying_seq_lens()

        a_batch, lp_batch, v_batch = policy.act_and_value(
            obs_list, deterministic=True,
        )

        a_single_0, lp_single_0, v_single_0 = policy.act_and_value(
            [obs_list[0]], deterministic=True,
        )
        a_single_1, lp_single_1, v_single_1 = policy.act_and_value(
            [obs_list[1]], deterministic=True,
        )

        # Argmax actions must be identical (the network sees the same
        # tokens in either case; padding does not affect the masked
        # attention output at real positions).
        assert torch.equal(a_batch[0:1], a_single_0)
        assert torch.equal(a_batch[1:2], a_single_1)

        # log-probs and values should match within float tolerance.
        # atol=3e-3 accommodates matmul-tiling accumulation differences at
        # current Fourier/d_model widths; argmax equality above is the strict check.
        assert torch.allclose(lp_batch[0:1], lp_single_0, atol=3e-3)
        assert torch.allclose(lp_batch[1:2], lp_single_1, atol=3e-3)
        assert torch.allclose(v_batch[0:1], v_single_0, atol=3e-3)
        assert torch.allclose(v_batch[1:2], v_single_1, atol=3e-3)


def test_dead_end_rows_raise_with_context(tmp_path, monkeypatch):
    """When a non-net_select row's candidate block is entirely -inf — regardless of
    whether the pointer is consumed (net_end or make_line) — evaluation must fail
    immediately with a RuntimeError carrying row context (action_type/K/row index),
    rather than surfacing as Categorical's uninformative ValueError for a state that
    should never occur under normal operation. Right before raising,
    pcb_world.diag.dump_context saves the full tensor context to a .pt file and
    appends its path as dump= at the end of the message. When at least one candidate
    survives (partial mask), a non-consuming row still behaves normally with finite
    logp/entropy."""
    from methods.rl_agent.models.v1.encoding import cand_mm_list_from_obs

    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    torch.manual_seed(0)
    policy = _tiny_policy()
    obs_list = _batch_obs(current_net_phase=2, is_routing=True)
    K = len(cand_mm_list_from_obs(obs_list[0]))
    assert K > 0
    blocked = torch.arange(K).unsqueeze(0).repeat(2, 1)    # every candidate -inf

    for at in (ACT_NET_END, ACT_MAKE_LINE):               # both non-consuming and consuming
        actions = torch.tensor([[at, 0, 1], [at, 0, 1]], dtype=torch.int64)
        with pytest.raises(RuntimeError, match="cand pointer row") as exc_info:
            policy.evaluate_actions_and_value(
                obs_list, actions, pointer_masks=blocked,
            )
        # dump= path round-trip: the tensor context must actually be loadable.
        dump_path = str(exc_info.value).rsplit("dump=", 1)[1]
        payload = torch.load(dump_path, weights_only=False)
        assert payload["ptr_logits"].shape[0] == 2
        assert not payload["ptr_logits"].requires_grad     # confirm detached
        assert torch.isinf(payload["ptr_logits"][int(payload["dead_rows"][0, 0])]).all()
        assert len(payload["obs_list"]) == 2

    # partial mask (candidates remain) -> non-consuming row behaves normally
    partial = torch.tensor([[0], [0]])
    actions_ok = torch.tensor(
        [[ACT_NET_END, -1, -1], [ACT_NET_END, -1, -1]], dtype=torch.int64,
    )
    lp, ent, _ = policy.evaluate_actions_and_value(
        obs_list, actions_ok, pointer_masks=partial,
    )
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ent).all()
