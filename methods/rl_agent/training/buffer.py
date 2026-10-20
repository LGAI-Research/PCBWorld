"""Flat replay buffer + advantage/return targets for decoder-policy RL.

The single flat-buffer layout shared by PPO (per-step GAE) and GRPO (grouped
group-relative advantage): GRPO per-env trajectories and PPO ``(T, N)`` collector
output both flatten into :class:`FlatBuffer`; PPO additionally carries
``returns`` (§3.5 invariant — same layout, only the advantage differs).
Carved from ``decoder_common`` (C1-b).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from methods.rl_agent.models.v1.spec import Transition

if TYPE_CHECKING:
    from methods.rl_agent.training.collect import (
        GroupStepRecord,
        PPOCollectorOutput,
    )


class FlatBuffer(TypedDict, total=False):
    # obs_list may arrive as None on DDP update workers when walk_flat is
    # carried (dispatch_update obs-strip) — the walked= update path never
    # reads obs.
    obs_list: list[dict] | None
    # Row-aligned with obs_list: the act-time DistInputs + action, old
    # log-prob/value (+ kl_mask) — what the update re-scores, as one bundle.
    transition: Transition
    advantages: np.ndarray      # (N,)   float32
    returns: np.ndarray         # (N,)   float32 — PPO only
    returns_mc: np.ndarray      # (N,)   float32 — PPO only (diagnostic)
    # Flat batched walk over the whole buffer (one walk dict, B == N,
    # row-aligned with obs_list) — the collect-time cached tokenize walk the
    # update index-gathers per minibatch (see PPOCollectorOutput.walk_flat).
    # Absent on the GRPO path (total=False), which re-walks (one batched
    # walk) at update entry.
    walk_flat: dict


def flatten_group_to_buffer(
    trajectories: list[list[GroupStepRecord]],
    per_episode_advantages: np.ndarray,
) -> FlatBuffer:
    """Flatten per-env GRPO trajectories into a flat replay buffer.

    Each step in episode i broadcasts the same scalar advantage
    ``per_episode_advantages[i]``.
    """
    steps = [(i, step) for i, traj in enumerate(trajectories) for step in traj]
    if not steps:
        raise ValueError(
            f"GRPO group of {len(trajectories)} trajectories has no steps — a "
            "collect yields >= 1 step per env, so the collector returned nothing"
        )
    return {
        "obs_list": [step["obs"] for _, step in steps],
        "transition": Transition.concat([step["transition"] for _, step in steps]),
        "advantages": np.asarray(
            [per_episode_advantages[i] for i, _ in steps], dtype=np.float32,
        ),
    }


def compute_gae_flat(
    rewards: np.ndarray,           # (T, N)
    values: np.ndarray,            # (T, N)
    episode_starts: np.ndarray,    # (T, N) bool
    final_values: np.ndarray,      # (N,)
    terminal_values: np.ndarray,   # (T, N) float, np.nan = not boundary
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation on a (T, N) flat rollout.

    Mirrors SB3's ``RolloutBuffer.compute_returns_and_advantage``, with the
    standard "fold bootstrap into reward" trick at truncated boundaries:

    * **terminated** at step ``t``: ``terminal_values[t,n] = 0`` (true end).
      No adjustment to reward; ``next_non_terminal=0`` cuts the bootstrap.
    * **truncated** at step ``t``: ``terminal_values[t,n] = V(true_next_obs)``.
      We fold ``gamma * V(true_next_obs)`` into ``rewards[t,n]`` so the
      single-step ``delta`` becomes ``r + gamma*V_next - v``, then we still
      set ``next_non_terminal=0`` so GAE does not propagate across the
      episode boundary.
    * **non-boundary**: ``terminal_values[t,n] = nan`` and the rollout
      continues to the next step's value (or ``final_values`` at t=T-1).

    Args:
        rewards, values, episode_starts: per-step rollout tensors.
        final_values: ``V(s_{T+1})`` for each env (the obs that would be
            stepped next if the rollout continued past ``t=T-1``).
        terminal_values: at any (t, n) where the env terminated/truncated
            during step t, contains the bootstrap value to use; ``np.nan``
            otherwise. Use 0 for terminated, ``V(true_next_obs)`` for
            truncated.
        gamma: discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``advantages, returns`` — both ``(T, N)`` float32. ``returns =
        advantages + values``.
    """
    T, N = rewards.shape

    # Step 1: fold truncated bootstrap into rewards.
    # At terminated boundaries terminal_values=0 so this is a no-op.
    boundary_mask = ~np.isnan(terminal_values)  # (T, N)
    bootstrap = np.where(boundary_mask, terminal_values, 0.0).astype(np.float32)
    rewards_adj = rewards.astype(np.float32) + gamma * bootstrap

    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            # Either we're at a boundary at t=T-1 (next_non_terminal=0,
            # next_values irrelevant since the bootstrap is already in
            # rewards_adj) or we bootstrap from final_values.
            at_boundary = boundary_mask[t]
            next_non_terminal = (~at_boundary).astype(np.float32)
            next_values = final_values.astype(np.float32)
        else:
            # SB3 convention: episode_starts[t+1] = True means the env
            # was reset between step t and t+1 (i.e. step t was a
            # boundary). next_non_terminal = 0 at boundaries.
            next_non_terminal = (~episode_starts[t + 1]).astype(np.float32)
            next_values = values[t + 1].astype(np.float32)

        delta = (
            rewards_adj[t]
            + gamma * next_values * next_non_terminal
            - values[t]
        )
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values.astype(np.float32)
    return advantages.astype(np.float32), returns.astype(np.float32)


def ppo_collector_to_buffer(
    coll: PPOCollectorOutput,
    advantages: np.ndarray,   # (T, N)
    returns: np.ndarray,      # (T, N)
) -> FlatBuffer:
    """Flatten a (T, N) PPO rollout into a flat replay buffer.

    Row-major order matches ``coll.obs_list`` and ``coll.transition``.
    """
    T, N = advantages.shape
    assert len(coll.obs_list) == T * N

    out: FlatBuffer = {
        "obs_list": coll.obs_list,
        "transition": coll.transition,
        "advantages": advantages.reshape(T * N).astype(np.float32),
        "returns": returns.reshape(T * N).astype(np.float32),
    }
    if coll.walk_flat is not None:
        assert coll.walk_flat["B"] == T * N
        out["walk_flat"] = coll.walk_flat
    return out
