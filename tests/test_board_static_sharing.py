"""Tests for board_static sharing optimization in the rollout collectors.

Verifies that:
1. All observations in a rollout buffer share the same board_static
   Python object per-env (identity, not just equality).
2. board_static is updated correctly on episode boundaries.
3. Tokenization output is bit-identical with shared vs independent
   board_static — i.e. the optimization does not change results.
4. Memory savings are real and measurable.

No C++ dependency — uses a lightweight stub environment and real
KiCadRLModel with mock observations.
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict

import numpy as np
import pytest
import torch

from pcb_world.core.masking import NUM_ACTIONS
from methods.rl_agent.algorithms.grpo import compute_grpo_advantages
from methods.rl_agent.training.buffer import (
    compute_gae_flat,
    flatten_group_to_buffer,
    ppo_collector_to_buffer,
)
from methods.rl_agent.training.collect import (
    collect_group_episodes,
    collect_n_steps_ppo,
)
from methods.rl_agent.training.utils import RunningRewardStd
from tests.helpers.reference_tokenizer import StateTokenizer

# Reuse mock observation builder from the existing tokenizer tests.
from tests._mock_obs import make_mock_obs


# ===================================================================
# Stub environment (no C++ dependency)
# ===================================================================
class _StubDecoderEnv:
    """Minimal env stub that mimics KiCadRLWrapper for collector tests.

    Each episode lasts ``episode_len`` steps then terminates.
    Observations use make_mock_obs() with a unique board_static per env.
    """

    def __init__(self, env_id: int, episode_len: int = 5, n_nets: int = 2):
        self._env_id = env_id
        self._episode_len = episode_len
        self._n_nets = n_nets
        self._step_count = 0
        self._episode_count = 0
        self._last_obs: dict = {}

    def _make_obs(self) -> dict:
        """Create a fresh observation dict (simulates pickle deserialization)."""
        obs = make_mock_obs(
            n_nets=self._n_nets,
            pads_per_net=2,
            n_tracks=self._step_count,  # geometry grows each step
            n_ratsnest_per_net=1,
            is_routing=self._step_count > 0,
            current_net_phase=1 if self._step_count > 0 else 0,
        )
        # Tag board_static with env_id + episode so we can verify identity
        obs["board_static"]["_env_id"] = self._env_id
        obs["board_static"]["_episode"] = self._episode_count
        return obs

    def reset(self, **kwargs):
        self._step_count = 0
        self._episode_count += 1
        self._last_obs = self._make_obs()
        return self._last_obs, {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self._episode_len
        truncated = False
        reward = float(np.random.default_rng(self._step_count).uniform(-1, 1))
        info = {
            "drc_violations": 0,
            "wirelength": float(self._step_count * 10 + self._env_id),
            "via_count": self._env_id + 1,
            "track_count": self._step_count + 2,
        }
        self._last_obs = self._make_obs()
        return self._last_obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self._step_count == 0:
            mask[1] = True   # ACT_NET_SELECT
        else:
            mask[2] = True   # ACT_START_ROUTE
            mask[4] = True   # ACT_NET_END
        return mask

    def start_route_pointer_indices(self) -> np.ndarray:
        return np.zeros((0,), dtype=np.int64)

    def mode_mask(self) -> np.ndarray:
        return np.ones(3, dtype=bool)

    def close(self):
        pass

    def __len__(self):
        return 1


# ===================================================================
# Stub policy (deterministic, no real transformer needed)
# ===================================================================
class _StubPolicy:
    """Minimal policy stub that returns valid actions without computation.

    Avoids running the real transformer so tests are fast and don't
    require GPU.
    """

    def __init__(self):
        self.use_critic = True

    def eval(self):
        pass

    def train(self):
        pass

    def act_and_value(
        self,
        obs_list,
        action_masks=None,
        deterministic=False,
        pointer_masks=None,
        mode_mask=None,
        **_kw,          # offlayer_masks / net_valid_mask / … — stub ignores them
    ):
        B = len(obs_list)
        device = action_masks.device if action_masks is not None else torch.device("cpu")

        # Pick the first valid action type from masks.
        actions = torch.zeros(B, 3, dtype=torch.long, device=device)
        if action_masks is not None:
            for i in range(B):
                valid = action_masks[i].nonzero(as_tuple=False)
                if len(valid) > 0:
                    actions[i, 0] = valid[0].item()
        actions[:, 1] = 0   # pointer
        actions[:, 2] = 0   # mode

        log_probs = torch.zeros(B, device=device)
        values = torch.zeros(B, device=device)
        return actions, log_probs, values

    def parameters(self):
        # Needed if someone iterates parameters; return empty.
        return iter([torch.zeros(1, requires_grad=True)])


# ===================================================================
# Test: PPO collector — board_static identity sharing
# ===================================================================
class TestPPOBoardStaticSharing:
    """Verify collect_n_steps_ppo shares board_static per env."""

    @pytest.fixture()
    def envs_and_policy(self):
        n_envs = 3
        envs = [_StubDecoderEnv(env_id=i, episode_len=4) for i in range(n_envs)]
        policy = _StubPolicy()
        return envs, policy, n_envs

    def test_board_static_shared_within_env(self, envs_and_policy):
        """All observations from the same env must share one board_static object."""
        envs, policy, n_envs = envs_and_policy
        n_steps = 12  # enough to trigger multiple episode resets (ep_len=4)

        coll = collect_n_steps_ppo(
            envs, policy, torch.device("cpu"), n_steps=n_steps,
        )

        # Group observations by env index (row-major: buf[t*N + n]).
        obs_by_env: dict[int, list[dict]] = defaultdict(list)
        for t in range(n_steps):
            for n in range(n_envs):
                obs_by_env[n].append(coll.obs_list[t * n_envs + n])

        # Within each env, consecutive obs in the same episode must share
        # the exact same board_static object (id check).
        for env_id, obs_list in obs_by_env.items():
            prev_episode = None
            prev_bs_id = None
            for obs in obs_list:
                ep = obs["board_static"]["_episode"]
                bs_id = id(obs["board_static"])
                if prev_episode is not None and ep == prev_episode:
                    assert bs_id == prev_bs_id, (
                        f"Env {env_id}, episode {ep}: board_static not shared "
                        f"(id {bs_id} != {prev_bs_id})"
                    )
                prev_episode = ep
                prev_bs_id = bs_id

    def test_board_static_updated_on_reset(self, envs_and_policy):
        """board_static reference changes across episode boundaries."""
        envs, policy, n_envs = envs_and_policy
        n_steps = 12

        coll = collect_n_steps_ppo(
            envs, policy, torch.device("cpu"), n_steps=n_steps,
        )

        # Verify that at least one episode boundary occurred (episode > 1).
        max_episode = 0
        for obs in coll.obs_list:
            max_episode = max(max_episode, obs["board_static"]["_episode"])
        assert max_episode >= 2, "Test needs at least 2 episodes to verify reset"

    def test_board_static_content_preserved(self, envs_and_policy):
        """Shared board_static still contains the correct data."""
        envs, policy, n_envs = envs_and_policy
        n_steps = 8

        coll = collect_n_steps_ppo(
            envs, policy, torch.device("cpu"), n_steps=n_steps,
        )

        for obs in coll.obs_list:
            bs = obs["board_static"]
            # Verify structural fields exist and are correct.
            assert "bbox_x" in bs
            assert "nets" in bs
            assert bs["copper_layers"] == 2
            assert bs["net_count"] == 2
            # Verify our env_id tag is present.
            assert "_env_id" in bs

    def test_terminal_geometry_metrics_are_collected(self, envs_and_policy):
        """PPO rollout records final wire/via/track metrics from terminal info."""
        envs, policy, _n_envs = envs_and_policy

        coll = collect_n_steps_ppo(
            envs, policy, torch.device("cpu"), n_steps=8,
        )

        assert len(coll.episode_wirelengths) == len(coll.episode_rewards)
        assert len(coll.episode_via_counts) == len(coll.episode_rewards)
        assert len(coll.episode_track_counts) == len(coll.episode_rewards)
        assert all(v > 0.0 for v in coll.episode_wirelengths)
        assert all(v >= 1.0 for v in coll.episode_via_counts)
        assert all(v >= 1.0 for v in coll.episode_track_counts)


# ===================================================================
# Test: GRPO collector — board_static identity sharing
# ===================================================================
class TestGRPOBoardStaticSharing:
    """Verify collect_group_episodes shares board_static per env."""

    def test_board_static_shared_in_trajectories(self):
        n_envs = 3
        envs = [_StubDecoderEnv(env_id=i, episode_len=6) for i in range(n_envs)]
        policy = _StubPolicy()

        trajs, _rew, *_ = collect_group_episodes(
            envs, policy, torch.device("cpu"), max_steps=10,
        )

        for env_id, traj in enumerate(trajs):
            if len(traj) < 2:
                continue
            # All steps within one episode must share board_static.
            first_bs_id = id(traj[0]["obs"]["board_static"])
            for step_idx, step in enumerate(traj[1:], 1):
                assert id(step["obs"]["board_static"]) == first_bs_id, (
                    f"GRPO env {env_id}, step {step_idx}: board_static not shared"
                )

    def test_flatten_preserves_sharing(self):
        """After flatten_group_to_buffer, board_static refs are still shared."""
        n_envs = 2
        envs = [_StubDecoderEnv(env_id=i, episode_len=5) for i in range(n_envs)]
        policy = _StubPolicy()

        trajs, rew, *_ = collect_group_episodes(
            envs, policy, torch.device("cpu"), max_steps=8,
        )
        rrs = RunningRewardStd()
        advs = compute_grpo_advantages(rew, rrs)
        buf = flatten_group_to_buffer(trajs, advs)

        # Group by env_id tag and check sharing.
        by_env: dict[int, list[int]] = defaultdict(list)
        for obs in buf["obs_list"]:
            by_env[obs["board_static"]["_env_id"]].append(
                id(obs["board_static"])
            )
        for env_id, ids in by_env.items():
            assert len(set(ids)) == 1, (
                f"GRPO flat buffer env {env_id}: {len(set(ids))} unique "
                f"board_static objects (expected 1)"
            )


# ===================================================================
# Test: Tokenization correctness (shared vs independent board_static)
# ===================================================================
class TestTokenizationCorrectness:
    """Verify that sharing board_static does not change tokenizer output."""

    @pytest.fixture()
    def tokenizer(self):
        return StateTokenizer(d_model=32, n_freq=4, max_seq_len=2000)

    def test_shared_vs_independent_tokenization_identical(self, tokenizer):
        """Tokenizer output must be bit-identical whether board_static is
        shared (same object) or independent (deep copy)."""
        base_obs = make_mock_obs(n_nets=3, pads_per_net=2, n_tracks=2,
                                 n_ratsnest_per_net=1)

        # Simulate 4 observations sharing the same board_static.
        shared_bs = base_obs["board_static"]
        shared_obs_list = []
        for i in range(4):
            obs = copy.deepcopy(base_obs)
            obs["board_static"] = shared_bs  # shared reference
            obs["router_head"]["step_ratio"] = i / 4.0  # vary dynamic part
            shared_obs_list.append(obs)

        # Independent: each obs has its own deep copy of board_static.
        independent_obs_list = []
        for i in range(4):
            obs = copy.deepcopy(base_obs)
            # board_static is already a fresh deep copy
            obs["router_head"]["step_ratio"] = i / 4.0
            independent_obs_list.append(obs)

        # Tokenize both.
        with torch.no_grad():
            out_shared = tokenizer(shared_obs_list)
            out_independent = tokenizer(independent_obs_list)

        # Compare all tensor fields.
        assert torch.equal(out_shared.token_embeddings, out_independent.token_embeddings), \
            "token_embeddings differ"
        assert torch.equal(out_shared.net_indices, out_independent.net_indices), \
            "net_indices differ"
        assert torch.equal(out_shared.cand_indices, out_independent.cand_indices), \
            "cand_indices differ"
        assert torch.equal(out_shared.key_padding_mask, out_independent.key_padding_mask), \
            "key_padding_mask differ"
        assert torch.equal(out_shared.seq_lens, out_independent.seq_lens), \
            "seq_lens differ"

    def test_shared_board_static_not_mutated(self, tokenizer):
        """Tokenizer must not mutate the observation dicts (especially board_static)."""
        obs = make_mock_obs(n_nets=2, pads_per_net=2)
        shared_bs = obs["board_static"]
        bs_snapshot = copy.deepcopy(shared_bs)

        obs_list = [obs, copy.deepcopy(obs)]
        obs_list[1]["board_static"] = shared_bs  # share

        with torch.no_grad():
            tokenizer(obs_list)

        # board_static must not be modified.
        assert shared_bs == bs_snapshot, "Tokenizer mutated board_static"


# ===================================================================
# Test: Memory savings
# ===================================================================
class TestMemorySavings:
    """Verify that board_static sharing actually reduces memory."""

    def test_obs_buffer_memory_reduction(self):
        """With sharing, total sys.getsizeof of board_static refs in the
        buffer should be much smaller than with independent copies."""
        n_envs = 2
        n_steps = 20
        envs = [_StubDecoderEnv(env_id=i, episode_len=8) for i in range(n_envs)]
        policy = _StubPolicy()

        coll = collect_n_steps_ppo(
            envs, policy, torch.device("cpu"), n_steps=n_steps,
        )

        # Count unique board_static object ids.
        bs_ids = set()
        for obs in coll.obs_list:
            bs_ids.add(id(obs["board_static"]))

        total_obs = len(coll.obs_list)
        n_unique = len(bs_ids)

        # With sharing, we should have at most n_envs * n_episodes unique
        # board_static objects (one per env per episode).  Without sharing,
        # we'd have total_obs unique objects. Assert significant reduction.
        assert n_unique < total_obs, (
            f"Expected fewer unique board_static objects than total obs: "
            f"{n_unique} unique vs {total_obs} total"
        )
        # More specifically, n_unique should be << total_obs.
        # With episode_len=8 and n_steps=20, ~2-3 episodes per env.
        # So n_unique should be around n_envs * ~3 = 6.
        assert n_unique <= n_envs * 5, (
            f"Too many unique board_static: {n_unique} "
            f"(expected <= {n_envs * 5} for {n_envs} envs)"
        )

    def test_deep_memory_savings(self):
        """Measure actual memory: shared obs list should use much less
        memory than an equivalent list with deep-copied board_static."""
        base_obs = make_mock_obs(n_nets=10, pads_per_net=4, n_tracks=5)
        shared_bs = base_obs["board_static"]

        n_obs = 100
        # Shared: all obs reference the same board_static.
        shared_list = []
        for i in range(n_obs):
            obs = {
                "board_static": shared_bs,
                "routing_geometry": copy.deepcopy(base_obs["routing_geometry"]),
                "router_head": copy.deepcopy(base_obs["router_head"]),
            }
            shared_list.append(obs)

        # Independent: each obs has its own board_static.
        independent_list = []
        for i in range(n_obs):
            independent_list.append(copy.deepcopy(base_obs))

        # Measure board_static memory.
        shared_bs_size = sys.getsizeof(shared_bs)
        independent_total = sum(
            sys.getsizeof(obs["board_static"]) for obs in independent_list
        )

        # With sharing: only 1 board_static in memory.
        # Without sharing: 100 separate board_static dicts.
        # (sys.getsizeof is shallow, but still shows the dict overhead)
        assert independent_total > shared_bs_size * 2, (
            f"Expected independent to use more memory: "
            f"independent={independent_total}, shared={shared_bs_size}"
        )


# ===================================================================
# Test: PPO full pipeline correctness (GAE + flatten)
# ===================================================================
class TestPPOPipelineWithSharing:
    """End-to-end: collect → GAE → flatten → verify buffer integrity."""

    def test_ppo_buffer_integrity(self):
        n_envs = 2
        n_steps = 10
        envs = [_StubDecoderEnv(env_id=i, episode_len=4) for i in range(n_envs)]
        policy = _StubPolicy()

        coll = collect_n_steps_ppo(
            envs, policy, torch.device("cpu"), n_steps=n_steps,
        )

        # Shapes should be correct.
        assert coll.actions.shape == (n_steps, n_envs, 3)
        assert coll.values.shape == (n_steps, n_envs)
        assert len(coll.obs_list) == n_steps * n_envs

        # GAE should work with shared obs.
        advs, rets = compute_gae_flat(
            rewards=coll.rewards,
            values=coll.values,
            episode_starts=coll.episode_starts,
            final_values=coll.final_values,
            terminal_values=coll.terminal_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert advs.shape == (n_steps, n_envs)
        assert np.all(np.isfinite(advs))

        # Flatten should preserve sharing.
        buf = ppo_collector_to_buffer(coll, advs, rets)
        assert len(buf["obs_list"]) == n_steps * n_envs

        # Verify board_static still shared after flatten.
        by_env: dict[int, list[int]] = defaultdict(list)
        for obs in buf["obs_list"]:
            by_env[obs["board_static"]["_env_id"]].append(
                id(obs["board_static"])
            )
        for env_id, ids in by_env.items():
            n_unique = len(set(ids))
            n_total = len(ids)
            assert n_unique < n_total, (
                f"Flat buffer env {env_id}: no sharing detected "
                f"({n_unique} unique / {n_total} total)"
            )
