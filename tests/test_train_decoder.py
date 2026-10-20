"""Tests for the decoder-policy RL rollout/buffer/update carve (Stage 6).

Two test groups:

1. **Pure unit tests** for GAE math, flatten, RunningRewardStd, explained
   variance — no env / no policy required, run anywhere.

2. **End-to-end smoke** — runs 2 iterations of GRPO and PPO against the
   real ``simple_routing_board.kicad_pcb`` fixture. Skipped when the C++
   ``kicad_rl_router`` module or the fixture board is missing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pcb_world.engine import engine_available
import torch

from pcb_world.engine.kicad_engine import allow_router_coexistence

from methods.rl_agent.algorithms._common import policy_update_loop
from methods.rl_agent.algorithms.grpo import compute_grpo_advantages
from methods.rl_agent.models.v1.spec import Transition
from methods.rl_agent.training.buffer import (
    compute_gae_flat,
    flatten_group_to_buffer,
    ppo_collector_to_buffer,
)
from methods.rl_agent.training.collect import (
    PPOCollectorOutput,
    collect_group_episodes,
    collect_n_steps_ppo,
)
from methods.rl_agent.training.utils import (
    RewardNormalizer,
    RunningRewardStd,
    auto_device,
    explained_variance,
)
from methods.rl_agent.wrappers.factory import make_decoder_env
from tests.helpers.dist_inputs import permissive_dist_np
from tests.helpers.env_kwargs import full_env_kwargs

_FIXTURE_BOARD = os.path.join(
    os.path.dirname(__file__), "fixtures", "simple_routing_board.kicad_pcb",
)


def _skip_if_no_env() -> None:
    if not os.path.exists(_FIXTURE_BOARD):
        pytest.skip(f"Fixture board not found: {_FIXTURE_BOARD}")
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


# ===================================================================
# Unit: RunningRewardStd
# ===================================================================
class TestRunningRewardStd:
    def test_initial_std_one(self):
        s = RunningRewardStd()
        assert abs(s.std - 1.0) < 1e-3

    def test_constant_batch_std_small(self):
        s = RunningRewardStd()
        for _ in range(10):
            s.update(np.array([5.0, 5.0, 5.0]))
        assert s.std < 0.1

    def test_variable_batch_std_finite(self):
        s = RunningRewardStd()
        rng = np.random.default_rng(0)
        for _ in range(20):
            s.update(rng.normal(0, 2.0, size=8))
        assert 1.0 < s.std < 4.0


# ===================================================================
# Unit: RewardNormalizer (SB3 VecNormalize parity)
# ===================================================================
class TestRewardNormalizer:
    def test_initial_std_one(self):
        rn = RewardNormalizer(n_envs=2, gamma=0.99)
        assert abs(rn.std - 1.0) < 1e-3

    def test_normalized_shape_and_dtype(self):
        rn = RewardNormalizer(n_envs=3, gamma=0.99)
        out = rn.normalize_step(
            rewards=np.array([1.0, 2.0, 3.0]),
            dones=np.array([False, False, False]),
        )
        assert out.shape == (3,)
        assert out.dtype == np.float32

    def test_constant_reward_drives_std_up(self):
        """A constant non-zero reward stream → discounted return grows
        to ~ r/(1-gamma), the running std moves above the initial seed
        of 1.0, and subsequent normalized rewards shrink below raw."""
        rn = RewardNormalizer(n_envs=1, gamma=0.9)
        for _ in range(200):
            rn.normalize_step(
                rewards=np.array([1.0]),
                dones=np.array([False]),
            )
        # Std grew from initial 1.0; the saturating returns near
        # 1/(1-0.9)=10 give a sample variance ~ (1.4)² in steady state.
        assert rn.std > 1.2
        # A fresh raw reward of 1.0 should normalize to < 1.0 (since std > 1).
        normalized = rn.normalize_step(
            rewards=np.array([1.0]), dones=np.array([False]),
        )
        assert abs(normalized[0]) < 1.0

    def test_done_resets_returns(self):
        rn = RewardNormalizer(n_envs=2, gamma=0.9)
        # Build up returns on env 0; env 1 stays at 0.
        for _ in range(10):
            rn.normalize_step(
                rewards=np.array([1.0, 0.0]),
                dones=np.array([False, False]),
            )
        # Sanity: env 0 has accumulated, env 1 is still 0.
        assert rn._returns[0] > 1.0
        assert abs(rn._returns[1]) < 1e-9

        # Mark env 0 done — its return must reset to 0.
        rn.normalize_step(
            rewards=np.array([0.5, 0.0]),
            dones=np.array([True, False]),
        )
        assert abs(rn._returns[0]) < 1e-9

    def test_state_dict_roundtrip(self):
        """Running stats survive a state_dict round-trip (ckpt save/resume),
        so a restored normalizer scales rewards identically; per-env returns
        are transient and start at 0 (n_envs may differ on resume)."""
        rn = RewardNormalizer(n_envs=2, gamma=0.9)
        for _ in range(50):
            rn.normalize_step(
                rewards=np.array([1.0, 0.5]),
                dones=np.array([False, False]),
            )
        restored = RewardNormalizer(n_envs=4, gamma=0.9)
        restored.load_state_dict(rn.state_dict())
        assert abs(restored.std - rn.std) < 1e-12
        assert restored._count == rn._count
        assert np.all(restored._returns == 0.0)

    def test_clip_bounds(self):
        rn = RewardNormalizer(n_envs=1, gamma=0.99, clip=2.0)
        # Force a small std then push a huge reward through.
        for _ in range(5):
            rn.normalize_step(
                rewards=np.array([0.001]), dones=np.array([False]),
            )
        out = rn.normalize_step(
            rewards=np.array([1000.0]), dones=np.array([False]),
        )
        assert out[0] <= 2.0 + 1e-6
        assert out[0] >= -2.0 - 1e-6

    def test_no_mean_subtraction(self):
        """SB3 VecNormalize does NOT subtract the mean — sign is preserved."""
        rn = RewardNormalizer(n_envs=1, gamma=0.99)
        # Build up positive returns to inflate the std.
        for _ in range(50):
            rn.normalize_step(
                rewards=np.array([1.0]), dones=np.array([False]),
            )
        pos = rn.normalize_step(
            rewards=np.array([1.0]), dones=np.array([False]),
        )
        neg = rn.normalize_step(
            rewards=np.array([-1.0]), dones=np.array([False]),
        )
        assert pos[0] > 0
        assert neg[0] < 0

    def test_collector_passes_normalized_rewards(self):
        """When a normalizer is given to collect_n_steps_ppo, the
        rewards stored in the output should be normalized values
        (matching SB3 RolloutBuffer semantics)."""
        _skip_if_no_env()
        from methods.rl_agent.models.v1.net import KiCadRLModel

        torch.manual_seed(0)
        device = torch.device("cpu")
        envs = [make_decoder_env(_FIXTURE_BOARD, **full_env_kwargs(max_steps=8, seed=0))]
        policy = KiCadRLModel(use_critic=True, **_tiny_policy_kwargs()).to(device)
        rn = RewardNormalizer(n_envs=1, gamma=1.0)
        try:
            coll = collect_n_steps_ppo(
                envs, policy, device, n_steps=12, reward_normalizer=rn,
            )
            # Episode rewards are RAW (Monitor semantics).
            assert all(isinstance(r, float) for r in coll.episode_rewards)
            # Stored rewards should respect the clip bound (normalized).
            assert np.all(np.abs(coll.rewards) <= 10.0 + 1e-6)
            # Running stats should have advanced from initial 1.0.
            assert rn.std != 1.0
        finally:
            for e in envs:
                e.close()

    def test_collector_caches_walk_aligned_and_byte_identical(self):
        """The collect-time walk cache (``walk_flat``) must be row-aligned
        with ``obs_list`` and byte-identical to a direct walk — i.e. a true
        drop-in for the update's re-walk. Exercised through the REAL env +
        tokenizer + budgeted/plain forward, so it also pins collect-side
        alignment (flat sample ``i`` ↔ obs ``i``) across multiple envs."""
        _skip_if_no_env()
        from methods.rl_agent.models.v1.net import KiCadRLModel

        def _rec_eq(x, y) -> bool:
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                return (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                        and x.dtype == y.dtype and x.shape == y.shape
                        and np.array_equal(x, y))
            if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                return (type(x) is type(y) and len(x) == len(y)
                        and all(_rec_eq(a, b) for a, b in zip(x, y)))
            return x == y

        torch.manual_seed(0)
        device = torch.device("cpu")
        n_steps, n_envs = 6, 2
        with allow_router_coexistence("collector list mode: n_envs in-process envs"):
            envs = [make_decoder_env(_FIXTURE_BOARD, **full_env_kwargs(max_steps=8, seed=i))
                    for i in range(n_envs)]
        policy = KiCadRLModel(use_critic=True, **_tiny_policy_kwargs()).to(device)
        try:
            coll = collect_n_steps_ppo(envs, policy, device, n_steps=n_steps)
            tok = policy.tokenizer
            assert coll.walk_flat is not None
            assert coll.walk_flat["B"] == n_steps * n_envs == len(coll.obs_list)
            # The merged flat walk == one direct batched walk of obs_list
            # (alignment + byte-identity), and an update-style shuffled gather
            # == the direct walk of that subset.
            direct = tok._walk_obs(coll.obs_list)
            assert coll.walk_flat.keys() == direct.keys()
            assert all(_rec_eq(coll.walk_flat[k], direct[k]) for k in direct)
            bounds = tok.walk_sample_bounds(coll.walk_flat)
            idx = [7, 0, 11, 3]
            got = tok.gather_walked(coll.walk_flat, bounds, idx)
            sub = tok._walk_obs([coll.obs_list[i] for i in idx])
            assert got.keys() == sub.keys()
            assert all(_rec_eq(got[k], sub[k]) for k in sub)
        finally:
            for e in envs:
                e.close()


# ===================================================================
# Unit: compute_grpo_advantages
# ===================================================================
class TestComputeGRPOAdvantages:
    def test_zero_centered(self):
        s = RunningRewardStd()
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        advs = compute_grpo_advantages(rewards, s)
        assert abs(float(advs.mean())) < 1e-6

    def test_positive_for_above_mean(self):
        s = RunningRewardStd()
        rewards = np.array([0.0, 0.0, 10.0])
        advs = compute_grpo_advantages(rewards, s)
        assert advs[2] > advs[0]
        assert advs[2] > advs[1]


# ===================================================================
# Unit: compute_gae_flat (PPO)
# ===================================================================
class TestComputeGAEFlat:
    """All test cases use a single env (N=1) for clarity."""

    def _flat(self, rewards, values, *, terminated_at=None, truncated_at=None,
              terminal_v=None, final_value=0.0):
        """Build single-env (T, 1) flat tensors for compute_gae_flat.

        Args:
            rewards: list of T floats.
            values: list of T floats.
            terminated_at: optional step index of true termination.
            truncated_at: optional step index of truncation (with terminal_v).
            terminal_v: bootstrap value at the truncated boundary.
            final_value: V(s_{T+1}) for non-terminal-at-T case.
        """
        T = len(rewards)
        rew = np.asarray(rewards, dtype=np.float32).reshape(T, 1)
        val = np.asarray(values, dtype=np.float32).reshape(T, 1)
        ep_starts = np.zeros((T, 1), dtype=bool)
        ep_starts[0, 0] = True
        terminal_values = np.full((T, 1), np.nan, dtype=np.float32)
        if terminated_at is not None:
            terminal_values[terminated_at, 0] = 0.0
            if terminated_at + 1 < T:
                ep_starts[terminated_at + 1, 0] = True
        if truncated_at is not None:
            assert terminal_v is not None
            terminal_values[truncated_at, 0] = float(terminal_v)
            if truncated_at + 1 < T:
                ep_starts[truncated_at + 1, 0] = True
        return rew, val, ep_starts, terminal_values, np.array(
            [final_value], dtype=np.float32,
        )

    def test_terminal_no_bootstrap_lambda_one(self):
        """gamma=1, lambda=1, terminal at T-1 → adv = sum(r[t..]) - v[t]."""
        rew, val, ep, term_v, fv = self._flat(
            rewards=[1.0, 2.0, 3.0],
            values=[0.0, 0.0, 0.0],
            terminated_at=2,
        )
        adv, ret = compute_gae_flat(
            rew, val, ep, fv, term_v, gamma=1.0, gae_lambda=1.0,
        )
        # Last step: delta = 3 + 0 - 0 = 3 → adv[2] = 3
        # Second:    delta = 2 + 0 - 0 = 2; gae = 2 + 1*1*1*3 = 5  (but next_non_terminal=1)
        # Actually since none of the intermediate steps are boundaries,
        # next_non_terminal=1 except after the terminated step.
        assert abs(adv[2, 0] - 3.0) < 1e-5
        assert abs(adv[1, 0] - 5.0) < 1e-5
        assert abs(adv[0, 0] - 6.0) < 1e-5
        # returns = adv + values
        assert np.allclose(ret, adv + val)

    def test_truncated_bootstrap(self):
        """truncated at T-1 → bootstrap with terminal_v (not 0)."""
        rew, val, ep, term_v, fv = self._flat(
            rewards=[0.0, 0.0, 5.0],
            values=[1.0, 2.0, 3.0],
            truncated_at=2, terminal_v=10.0,
        )
        adv, _ = compute_gae_flat(
            rew, val, ep, fv, term_v, gamma=1.0, gae_lambda=1.0,
        )
        # Last step: delta = 5 + 10 - 3 = 12; next_non_terminal=0 → gae = 12
        assert abs(adv[2, 0] - 12.0) < 1e-5

    def test_terminated_zero_bootstrap(self):
        """terminated at T-1 → bootstrap with 0."""
        rew, val, ep, term_v, fv = self._flat(
            rewards=[0.0, 0.0, 5.0],
            values=[1.0, 2.0, 3.0],
            terminated_at=2,
        )
        adv, _ = compute_gae_flat(
            rew, val, ep, fv, term_v, gamma=1.0, gae_lambda=1.0,
        )
        # Last: delta = 5 + 0 - 3 = 2 → adv[2] = 2
        assert abs(adv[2, 0] - 2.0) < 1e-5

    def test_gamma_zero_advantage_is_td_error(self):
        """gamma=0 → advantage[t] = r[t] - v[t]."""
        rew, val, ep, term_v, fv = self._flat(
            rewards=[1.0, 2.0, 3.0],
            values=[0.5, 0.5, 0.5],
            terminated_at=2,
        )
        adv, _ = compute_gae_flat(
            rew, val, ep, fv, term_v, gamma=0.0, gae_lambda=1.0,
        )
        for t in range(3):
            assert abs(adv[t, 0] - (rew[t, 0] - val[t, 0])) < 1e-5

    def test_no_boundary_uses_final_values(self):
        """If no termination/truncation in rollout, last step uses final_values."""
        rew, val, ep, term_v, fv = self._flat(
            rewards=[1.0, 1.0],
            values=[0.0, 0.0],
            final_value=2.0,
        )
        adv, _ = compute_gae_flat(
            rew, val, ep, fv, term_v, gamma=1.0, gae_lambda=1.0,
        )
        # Last: delta = 1 + 1*2 - 0 = 3 → adv[1] = 3
        # Second-to-last: delta = 1 + 1*0 - 0 = 1; gae = 1 + 1*1*1*3 = 4
        assert abs(adv[1, 0] - 3.0) < 1e-5
        assert abs(adv[0, 0] - 4.0) < 1e-5

    def test_returns_equal_advantage_plus_values(self):
        rew, val, ep, term_v, fv = self._flat(
            rewards=[0.5, 1.5, 2.5, 3.5],
            values=[0.1, 0.2, 0.3, 0.4],
            terminated_at=3,
        )
        adv, ret = compute_gae_flat(
            rew, val, ep, fv, term_v, gamma=0.99, gae_lambda=0.95,
        )
        assert np.allclose(ret, adv + val, atol=1e-6)


# ===================================================================
# Unit: flatten_group_to_buffer (GRPO)
# ===================================================================
def _mk_step(reward, action_type=0):
    return {
        "obs": {},
        "transition": Transition(
            dist=permissive_dist_np(1),
            actions=np.array([[action_type, 0, 0]], dtype=np.int64),
            old_log_probs=np.zeros(1, dtype=np.float32),
            old_values=np.zeros(1, dtype=np.float32),
            kl_mask=None,
        ),
        "reward": float(reward),
        "terminated": False,
        "truncated": False,
    }


class TestFlattenGroupToBuffer:
    def test_grpo_flatten_broadcast(self):
        traj1 = [_mk_step(0.0), _mk_step(1.0)]
        traj2 = [_mk_step(2.0)]
        per_ep = np.array([0.5, -0.5], dtype=np.float32)
        buf = flatten_group_to_buffer([traj1, traj2], per_ep)
        assert buf["transition"].actions.shape == (3, 3)
        assert buf["transition"].old_log_probs.shape == (3,)
        assert buf["advantages"].shape == (3,)
        # Episode 0 gets 0.5 broadcast to its 2 steps; episode 1 gets -0.5.
        assert np.allclose(buf["advantages"], [0.5, 0.5, -0.5])
        # GRPO buffer has no returns key.
        assert "returns" not in buf
        assert isinstance(buf["obs_list"], list)
        assert len(buf["obs_list"]) == 3

    def test_empty_trajectories_raise(self):
        # A GRPO collect always yields >= 1 step per env; an empty group is a
        # caller bug, not a zero-row buffer.
        with pytest.raises(ValueError, match="GRPO group of 2 trajectories has no steps"):
            flatten_group_to_buffer(
                [[], []], np.array([0.0, 0.0], dtype=np.float32),
            )


# ===================================================================
# Unit: ppo_collector_to_buffer
# ===================================================================
class TestPPOCollectorToBuffer:
    def test_flatten_shapes(self):
        T, N = 3, 2
        coll = PPOCollectorOutput(
            obs_list=[{} for _ in range(T * N)],
            transition=Transition(
                dist=permissive_dist_np(T * N),
                actions=np.zeros((T * N, 3), dtype=np.int64),
                old_log_probs=np.zeros(T * N, dtype=np.float32),
                old_values=np.full(T * N, 0.5, dtype=np.float32),
                kl_mask=None,
            ),
            rewards=np.zeros((T, N), dtype=np.float32),
            raw_rewards=np.zeros((T, N), dtype=np.float32),
            values=np.full((T, N), 0.5, dtype=np.float32),
            episode_starts=np.zeros((T, N), dtype=bool),
            terminated_mask=np.zeros((T, N), dtype=bool),
            terminal_values=np.full((T, N), np.nan, dtype=np.float32),
            final_values=np.zeros(N, dtype=np.float32),
            episode_rewards=[],
            episode_lengths=[],
            episode_drc_violations=[],
            episode_final_potentials=[],
        )
        adv = np.arange(T * N, dtype=np.float32).reshape(T, N)
        ret = adv + 0.5
        buf = ppo_collector_to_buffer(coll, adv, ret)
        assert buf["transition"].actions.shape == (T * N, 3)
        assert buf["advantages"].shape == (T * N,)
        assert buf["returns"].shape == (T * N,)
        assert buf["transition"].old_values.shape == (T * N,)
        # Row-major flatten: adv[0, 0], adv[0, 1], adv[1, 0], ...
        assert np.allclose(buf["advantages"], adv.reshape(-1))


# ===================================================================
# Unit: explained_variance
# ===================================================================
class TestExplainedVariance:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert abs(explained_variance(y, y) - 1.0) < 1e-6

    def test_zero_variance_truth(self):
        y_pred = np.array([1.0, 2.0])
        y_true = np.array([5.0, 5.0])
        assert explained_variance(y_pred, y_true) == 0.0

    def test_random_pred_negative(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # anti-correlated
        ev = explained_variance(y_pred, y_true)
        assert ev < 0


# ===================================================================
# Unit: PPO requires use_critic
# ===================================================================
class TestPPORequiresCritic:
    def test_ppo_without_critic_raises(self):
        from methods.rl_agent.models.v1.net import KiCadRLModel

        device = torch.device("cpu")
        policy = KiCadRLModel(
            disable_slot_emb=False,
            policy_net_select=False,
            d_model=32, n_heads=4, n_layers=2, d_ff=64,
            max_seq_len=2000, n_freq=4, use_critic=False,
        )
        opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
        buf = {
            "obs_list": [{}],
            "transition": Transition(
                dist=permissive_dist_np(1),
                actions=np.zeros((1, 3), dtype=np.int64),
                old_log_probs=np.zeros((1,), dtype=np.float32),
                old_values=np.zeros((1,), dtype=np.float32),
                kl_mask=None,
            ),
            "advantages": np.zeros((1,), dtype=np.float32),
            "returns": np.zeros((1,), dtype=np.float32),
        }
        with pytest.raises(ValueError, match="use_critic"):
            policy_update_loop(policy, opt, buf, device, algo="ppo")


# ===================================================================
# ALLOWED_MISSING_KEYS (checkpoint-load tolerance per trainer)
# ===================================================================
class TestAllowedMissingKeys:
    """Pins each trainer's effective set and its derivation from the base."""

    def test_effective_sets(self):
        from methods.rl_agent.training.loop import (
            GRPOTrainer, PPOTrainer, RLTrainer,
        )

        assert RLTrainer.ALLOWED_MISSING_KEYS == {"prev_action", "drc"}
        assert PPOTrainer.ALLOWED_MISSING_KEYS == {"prev_action", "drc"}
        assert GRPOTrainer.ALLOWED_MISSING_KEYS == {"prev_action", "drc", "same_net_bias"}

    def test_subclasses_derive_from_base(self):
        """A base edit reaches both trainers: PPO inherits the base set and
        GRPO's set is the base set plus its own additions."""
        from methods.rl_agent.training.loop import (
            GRPOTrainer, PPOTrainer, RLTrainer,
        )

        assert "ALLOWED_MISSING_KEYS" not in vars(PPOTrainer)
        assert PPOTrainer.ALLOWED_MISSING_KEYS is RLTrainer.ALLOWED_MISSING_KEYS
        assert RLTrainer.ALLOWED_MISSING_KEYS < GRPOTrainer.ALLOWED_MISSING_KEYS

    def test_sets_are_immutable(self):
        """PPO shares the base object, so an in-place edit through one trainer
        must be impossible rather than silently loosening the other's loads."""
        from methods.rl_agent.training.loop import (
            GRPOTrainer, PPOTrainer, RLTrainer,
        )

        for cls in (RLTrainer, PPOTrainer, GRPOTrainer):
            assert isinstance(cls.ALLOWED_MISSING_KEYS, frozenset), cls.__name__


# ===================================================================
# auto_device
# ===================================================================
class TestAutoDevice:
    def test_explicit_cpu(self):
        assert auto_device("cpu") == torch.device("cpu")

    def test_auto_resolves(self):
        d = auto_device("auto")
        assert isinstance(d, torch.device)


# ===================================================================
# End-to-end smoke (require C++ kicad_rl_router)
# ===================================================================
def _tiny_policy_kwargs():
    return dict(
        d_model=32, n_heads=4, n_layers=2, d_ff=64,
        max_seq_len=2000, n_freq=4, policy_net_select=False, disable_slot_emb=False,
    )


class TestEndToEndGRPO:
    def test_two_iterations(self):
        _skip_if_no_env()
        from methods.rl_agent.models.v1.net import KiCadRLModel

        torch.manual_seed(0)
        device = torch.device("cpu")
        with allow_router_coexistence("collector list mode: 2 in-process envs"):
            envs = [
                make_decoder_env(_FIXTURE_BOARD, **full_env_kwargs(max_steps=20, seed=i))
                for i in range(2)
            ]
        policy = KiCadRLModel(use_critic=False, **_tiny_policy_kwargs()).to(device)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
        rrs = RunningRewardStd()

        try:
            for _ in range(2):
                trajs, rew, *_ = collect_group_episodes(
                    envs, policy, device, max_steps=20,
                )
                assert len(trajs) == 2
                assert sum(len(t) for t in trajs) > 0
                advs = compute_grpo_advantages(rew, rrs)
                buf = flatten_group_to_buffer(trajs, advs)
                metrics = policy_update_loop(
                    policy, opt, buf, device,
                    algo="grpo", n_epochs=2, batch_size=8,
                )
                assert np.isfinite(metrics["loss"])
                assert np.isfinite(metrics["entropy"])
                assert metrics["value_loss"] == 0.0
        finally:
            for e in envs:
                e.close()


class TestEndToEndPPO:
    def test_two_iterations(self):
        _skip_if_no_env()
        from methods.rl_agent.models.v1.net import KiCadRLModel

        torch.manual_seed(0)
        device = torch.device("cpu")
        with allow_router_coexistence("collector list mode: 2 in-process envs"):
            envs = [
                make_decoder_env(_FIXTURE_BOARD, **full_env_kwargs(max_steps=20, seed=i))
                for i in range(2)
            ]
        policy = KiCadRLModel(use_critic=True, **_tiny_policy_kwargs()).to(device)
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

        # Snapshot critic_head params before training to assert they update.
        before = {
            name: p.detach().clone()
            for name, p in policy.critic_head.named_parameters()
        }
        # Also snapshot a backbone parameter — value loss should reach it
        # (standard PPO, no detach).
        backbone_param_name, backbone_param = next(
            iter(policy.layers.named_parameters()),
        )
        backbone_before = backbone_param.detach().clone()

        try:
            for _ in range(2):
                coll = collect_n_steps_ppo(
                    envs, policy, device, n_steps=16,
                )
                assert coll.transition.actions.shape == (16 * 2, 3)
                assert coll.values.shape == (16, 2)

                advs, rets = compute_gae_flat(
                    rewards=coll.rewards,
                    values=coll.values,
                    episode_starts=coll.episode_starts,
                    final_values=coll.final_values,
                    terminal_values=coll.terminal_values,
                    gamma=1.0, gae_lambda=0.95,
                )
                buf = ppo_collector_to_buffer(coll, advs, rets)
                assert "returns" in buf and buf["transition"].old_values.shape == (16 * 2,)

                metrics = policy_update_loop(
                    policy, opt, buf, device,
                    algo="ppo", n_epochs=2, batch_size=8,
                    vf_coef=0.5, normalize_advantages=True,
                )
                assert np.isfinite(metrics["loss"])
                assert np.isfinite(metrics["value_loss"])
                assert metrics["value_loss"] >= 0.0
        finally:
            for e in envs:
                e.close()

        # Critic head must have changed.
        any_critic_changed = False
        for name, p in policy.critic_head.named_parameters():
            if not torch.allclose(p.detach(), before[name]):
                any_critic_changed = True
                break
        assert any_critic_changed, "critic_head was not updated by PPO training"

        # Backbone must also have changed (value loss flows back).
        assert not torch.allclose(backbone_param.detach(), backbone_before), (
            f"Backbone param '{backbone_param_name}' did not update — "
            "value/policy loss is not reaching the backbone."
        )


class TestPPOCollectorTerminalHandling:
    """Verifies collect_n_steps_ppo handles termination/truncation correctly."""

    def test_collector_termination_e2e(self):
        _skip_if_no_env()
        from methods.rl_agent.models.v1.net import KiCadRLModel

        torch.manual_seed(0)
        device = torch.device("cpu")
        # Tiny max_steps so the env is forced to truncate quickly.
        with allow_router_coexistence("collector list mode: 2 in-process envs"):
            envs = [make_decoder_env(_FIXTURE_BOARD, **full_env_kwargs(max_steps=4, seed=i)) for i in range(2)]
        policy = KiCadRLModel(use_critic=True, **_tiny_policy_kwargs()).to(device)
        try:
            coll = collect_n_steps_ppo(envs, policy, device, n_steps=12)
            # episode_starts should have at least one True after t=0 (auto-reset).
            n_resets = int(coll.episode_starts.sum()) - 2  # exclude t=0 init
            assert n_resets >= 1, "Expected at least one auto-reset across 12 steps"
            # terminal_values should be filled (not nan) at the boundary steps.
            n_boundaries = int((~np.isnan(coll.terminal_values)).sum())
            assert n_boundaries >= 1
            # Episode book-keeping records at least one finished episode.
            assert len(coll.episode_rewards) >= 1
            assert len(coll.episode_lengths) == len(coll.episode_rewards)
        finally:
            for e in envs:
                e.close()


# ===================================================================
# Unit: build_evaluators diagnostic-protocol override (eval2..eval5)
# ===================================================================
class TestBuildEvaluatorsDiagOverride:
    """--eval-diag-max-steps / --eval-diag-masking-rule pin eval2..eval5 only.

    eval_transformer is stubbed (kwargs capture) — no env/C++ involved; the
    test verifies which protocol each evaluator's rollout fn would run.
    """

    def _args(self, tmp_path, **over):
        from types import SimpleNamespace

        board_a = tmp_path / "board_a.kicad_pcb"
        board_b = tmp_path / "board_b.kicad_pcb"
        board_a.write_text("")   # loader stats the path; contents never read
        board_b.write_text("")
        manifest = tmp_path / "diag_boards.txt"
        manifest.write_text(f"{board_a}\n{board_b}\n")
        from methods.rl_agent.training.train_ppo import build_arg_parser

        base = vars(build_arg_parser().parse_args(["--board", "b.kicad_pcb"]))
        base.update(
            max_steps=1024, masking_rule="train_rule.yaml",
            reward_rule="default",
            eval_n_rollouts=5, n_envs=8, eval_base_seed=1000,
            eval_boards_per_batch=None,
            eval2_boards=str(manifest), eval2_prefix="val_d3b",
            eval3_boards=None, eval3_prefix="val_d3a",
        )
        base.update(over)
        return SimpleNamespace(**base)

    def _build(self, monkeypatch, args):
        import eval.rollout.rl as rl_mod
        from methods.rl_agent.training.loop import build_evaluators

        calls = []

        def _stub_eval_transformer(agent, device, boards, **kw):
            calls.append(kw)
            return "sentinel"

        monkeypatch.setattr(rl_mod, "eval_transformer", _stub_eval_transformer)
        from methods._shared.board_loader import load_boards_from_dir_or_list
        from pathlib import Path

        eval_boards = load_boards_from_dir_or_list(
            boards_list=Path(args.eval2_boards),
        )
        from methods.rl_agent.training.loop import val_env_config

        primary, extras = build_evaluators(args, None, "cpu", eval_boards,
                                           val_env=val_env_config(args))
        return primary, dict(extras), calls

    def test_override_pins_diag_sets_only(self, monkeypatch, tmp_path):
        args = self._args(
            tmp_path,
            eval_diag_max_steps=512, eval_diag_masking_rule="netfree.yaml",
        )
        primary, extras, calls = self._build(monkeypatch, args)

        primary.rollout_fn(primary.boards)
        native = calls[-1]
        assert native["max_steps"] == 1024
        assert native["env_config"].env.max_steps == 1024
        assert native["env_config"].env.masking_rule == "train_rule.yaml"

        extras["val_d3b"].rollout_fn(extras["val_d3b"].boards)
        diag = calls[-1]
        assert diag["max_steps"] == 512
        assert diag["env_config"].env.max_steps == 512
        assert diag["env_config"].env.masking_rule == "netfree.yaml"
        # Only protocol knobs differ from the native env config.
        rest_native = {k: v for k, v in native["env_config"].to_record().items()
                       if k not in ("max_steps", "masking_rule")}
        rest_diag = {k: v for k, v in diag["env_config"].to_record().items()
                     if k not in ("max_steps", "masking_rule")}
        assert rest_native == rest_diag

        # Greedy pass stays native (primary boards, train protocol).
        extras["val_greedy"].rollout_fn(extras["val_greedy"].boards)
        greedy = calls[-1]
        assert greedy["max_steps"] == 1024
        assert greedy["env_config"].env.masking_rule == "train_rule.yaml"
        assert greedy["deterministic"] is True

    def test_no_override_reuses_native_fn(self, monkeypatch, tmp_path):
        # Flags absent entirely (pre-flag checkpoints via getattr fallback).
        args = self._args(tmp_path)
        primary, extras, _ = self._build(monkeypatch, args)
        assert extras["val_d3b"].rollout_fn is primary.rollout_fn


class TestHandleValOverallNoneTolerance:
    """Regression (260813 A10): None entries in ``overall`` (a board whose every
    rollout crashed) must not kill the consuming path — the summary prints NA
    and best-ckpt selection is skipped by the existing guard."""

    def test_none_metrics_do_not_raise(self, capsys):
        from types import SimpleNamespace
        from methods.rl_agent.training.loop import RLTrainer

        stub = SimpleNamespace(best_eval_fp=float("-inf"), args=None)
        ov = {
            "fp_mean_of_means": None, "fp_mean_of_maxes": None,
            "routability_mean": None, "wirelength_mean": None,
            "via_count_mean": None,
        }
        RLTrainer._handle_val_overall(stub, 10, ov, lambda: {})
        out = capsys.readouterr().out
        assert "rout=NA" in out and "fp_mean_of_means=NA" in out

    def test_normal_metrics_still_format_and_select(self, tmp_path):
        from types import SimpleNamespace
        from methods.rl_agent.training.loop import RLTrainer

        saved = {}
        stub = SimpleNamespace(
            best_eval_fp=float("-inf"),
            args=SimpleNamespace(save_dir=str(tmp_path), ckpt_config=lambda: {"schema": 2, "values": {"rules_source": "board"}}),
            _save_ckpt=lambda path, payload: saved.update({"path": path}),
        )
        ov = {
            "fp_mean_of_means": 0.5, "fp_mean_of_maxes": 0.9,
            "routability_mean": 0.8, "wirelength_mean": 10.0,
            "via_count_mean": 2.0,
        }
        RLTrainer._handle_val_overall(stub, 10, ov, lambda: {})
        assert stub.best_eval_fp == 0.5
        assert saved["path"].endswith("policy_best.pt")


@pytest.mark.parametrize("algo", ["ppo", "grpo"])
class TestResumeKeepsTheBestCheckpoint:
    """``policy_best.pt`` is the best over the whole run, across resumes."""

    @staticmethod
    def _trainer(algo, save_dir, start_iteration=1, resume=None):
        from types import SimpleNamespace
        from configs.loader.schema import DEFAULTS
        from methods.rl_agent.training.loop import GRPOTrainer, PPOTrainer

        cls = {"ppo": PPOTrainer, "grpo": GRPOTrainer}[algo]
        t = cls.__new__(cls)
        t.best_eval_fp = float("-inf") if DEFAULTS.best_metric_mode == "max" else float("inf")
        t.start_iteration = start_iteration
        t.args = SimpleNamespace(
            save_dir=str(save_dir),
            resume=str(resume or save_dir / f"policy_iter_{start_iteration - 1}.pt"),
            ckpt_config=lambda: {"schema": 2, "values": {"rules_source": "board"}})
        t._ckpt_stamp_payload = lambda: {}
        t._schedule_state = lambda: {}
        return t

    @staticmethod
    def _val(t, iteration, score):
        t._handle_val_overall(iteration, {"fp_mean_of_means": score}, lambda: {"w": score})

    @staticmethod
    def _best(save_dir, name="policy_best.pt"):
        return torch.load(save_dir / name, weights_only=False)["policy_state_dict"]["w"]

    def _run_to_iter_10(self, algo, save_dir, score=0.9):
        run = self._trainer(algo, save_dir)
        self._val(run, 10, score)
        run._save_ckpt(str(save_dir / "policy_iter_10.pt"), {"iteration": 10})
        return torch.load(save_dir / "policy_iter_10.pt", weights_only=False)

    def test_a_worse_first_val_after_resume_keeps_policy_best(self, algo, tmp_path):
        ckpt = self._run_to_iter_10(algo, tmp_path)
        assert ckpt["best_val_score"] == pytest.approx(0.9)

        resumed = self._trainer(algo, tmp_path, start_iteration=11)
        resumed._restore_best_score(ckpt)
        self._val(resumed, 20, 0.5)
        assert self._best(tmp_path) == 0.9
        self._val(resumed, 30, 1.0)
        assert self._best(tmp_path) == 1.0

    def test_min_mode_keeps_the_lower_best(self, algo, tmp_path, monkeypatch):
        import dataclasses
        from configs.loader import schema

        monkeypatch.setattr(schema, "DEFAULTS", dataclasses.replace(schema.DEFAULTS, best_metric_mode="min"))
        ckpt = self._run_to_iter_10(algo, tmp_path, score=0.3)
        resumed = self._trainer(algo, tmp_path, start_iteration=11)
        resumed._restore_best_score(ckpt)
        assert resumed.best_eval_fp == pytest.approx(0.3)
        self._val(resumed, 20, 0.5)
        assert self._best(tmp_path) == 0.3
        self._val(resumed, 30, 0.1)
        assert self._best(tmp_path) == 0.1

    def test_a_resume_into_a_new_save_dir_carries_the_run_best(self, algo, tmp_path):
        old, new = tmp_path / "old", tmp_path / "new"
        old.mkdir(), new.mkdir()
        ckpt = self._run_to_iter_10(algo, old)
        resumed = self._trainer(algo, new, start_iteration=11, resume=old / "policy_iter_10.pt")
        resumed._restore_best_score(ckpt)
        assert self._best(new) == 0.9                            # the run's best, from the start
        assert resumed.best_eval_fp == pytest.approx(0.9)
        self._val(resumed, 20, 0.5)                              # worse than the run's 0.9
        assert self._best(new) == 0.9
        self._val(resumed, 30, 1.0)
        assert self._best(new) == 1.0
        assert self._best(old) == 0.9

    def test_an_old_checkpoint_recovers_the_score_of_policy_best(self, algo, tmp_path):
        torch.save({"iteration": 10, "policy_state_dict": {"w": 0.9},
                    "eval_overall": {"fp_mean_of_means": 0.9}}, tmp_path / "policy_best.pt")
        resumed = self._trainer(algo, tmp_path, start_iteration=21)
        resumed._restore_best_score({"iteration": 20})          # pre-best_val_score checkpoint
        assert resumed.best_eval_fp == pytest.approx(0.9)
        self._val(resumed, 30, 0.5)
        assert self._best(tmp_path) == 0.9

    def test_a_fork_into_a_new_save_dir_skips_a_later_best(self, algo, tmp_path, capsys):
        old, new = tmp_path / "old", tmp_path / "new"
        old.mkdir(), new.mkdir()
        ckpt = self._run_to_iter_10(algo, old)
        run = self._trainer(algo, old, start_iteration=11)
        run._restore_best_score(ckpt)
        self._val(run, 20, 1.0)                                  # the abandoned branch's best
        forked = self._trainer(algo, new, start_iteration=11, resume=old / "policy_iter_10.pt")
        forked._restore_best_score(ckpt)
        out = capsys.readouterr().out
        assert str(old / "policy_best.pt") in out and "iteration 20" in out
        assert not (new / "policy_best.pt").exists()
        self._val(forked, 30, 0.5)                               # this branch's first val
        assert self._best(new) == 0.5
        assert self._best(old) == 1.0

    def test_a_policy_best_without_a_score_is_kept_aside_and_replaced(self, algo, tmp_path, capsys):
        torch.save({"iteration": 10, "policy_state_dict": {"w": 0.9}, "metrics": {}},
                   tmp_path / "policy_best.pt")                  # KDD-era: train metrics only
        for _ in range(2):                                       # resumed again before any val
            resumed = self._trainer(algo, tmp_path, start_iteration=21)
            resumed._restore_best_score({"iteration": 20})
            assert "policy_best.unscored.pt" in capsys.readouterr().out
            assert self._best(tmp_path, "policy_best.unscored.pt") == 0.9
            assert self._best(tmp_path) == 0.9
        self._val(resumed, 30, 0.1)                              # the first val of this segment
        assert self._best(tmp_path) == 0.1
        self._val(resumed, 40, 0.05)
        assert self._best(tmp_path) == 0.1
        assert self._best(tmp_path, "policy_best.unscored.pt") == 0.9
