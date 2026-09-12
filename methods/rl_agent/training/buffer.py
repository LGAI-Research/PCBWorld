"""Flat replay buffer + advantage/return targets for decoder-policy RL.

The single flat-buffer layout shared by PPO (per-step GAE) and GRPO (grouped
group-relative advantage): GRPO per-env trajectories and PPO ``(T, N)`` collector
output both flatten into :class:`FlatBuffer`; PPO additionally carries
``returns``/``old_values`` (§3.5 invariant — same layout, only the advantage differs).
Carved from ``decoder_common`` (C1-b).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from pcb_world.core.masking import NUM_ACTIONS

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
    actions: np.ndarray         # (N, 3) int64
    old_log_probs: np.ndarray   # (N,)   float32
    action_masks: np.ndarray    # (N, NUM_ACTIONS) bool
    pointer_masks: np.ndarray   # (N, K) int64 — start_route cand idxs (-1 pad)
    mode_masks: np.ndarray      # (N, 3) bool — routing-mode mask
    net_valid_masks: np.ndarray # (N, M_max) bool — policy-driven net_select only
    advantages: np.ndarray      # (N,)   float32
    returns: np.ndarray         # (N,)   float32 — PPO only
    old_values: np.ndarray      # (N,)   float32 — PPO only
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
    obs_list: list[dict] = []
    actions: list[np.ndarray] = []
    old_lp: list[float] = []
    masks: list[np.ndarray] = []
    ptr_masks: list[np.ndarray] = []
    m_masks: list[np.ndarray] = []
    nvm_list: list[np.ndarray] = []
    has_nvm = False
    advs: list[float] = []

    for i, traj in enumerate(trajectories):
        for step in traj:
            obs_list.append(step["obs"])
            actions.append(step["action"])
            old_lp.append(step["log_prob"])
            masks.append(step["action_mask"])
            pm = step.get("pointer_mask")
            if pm is None:
                pm = np.zeros((0,), dtype=np.int64)
            else:
                pm = np.asarray(pm, dtype=np.int64).reshape(-1)
            ptr_masks.append(pm)
            m_masks.append(step.get(
                "mode_mask", np.ones(3, dtype=bool),
            ))
            if "net_valid_mask" in step:
                has_nvm = True
                nvm_list.append(step["net_valid_mask"])
            advs.append(float(per_episode_advantages[i]))

    if len(obs_list) == 0:
        return {
            "obs_list": [],
            "actions": np.zeros((0, 3), dtype=np.int64),
            "old_log_probs": np.zeros((0,), dtype=np.float32),
            "action_masks": np.zeros((0, NUM_ACTIONS), dtype=bool),
            "pointer_masks": np.zeros((0, 0), dtype=np.int64),
            "mode_masks": np.zeros((0, 3), dtype=bool),
            "advantages": np.zeros((0,), dtype=np.float32),
        }

    K_max = max((p.shape[0] for p in ptr_masks), default=0)
    if K_max == 0:
        ptr_masks_arr = np.full((len(ptr_masks), 0), -1, dtype=np.int64)
    else:
        ptr_masks_arr = np.full((len(ptr_masks), K_max), -1, dtype=np.int64)
        for j, p in enumerate(ptr_masks):
            if p.shape[0]:
                ptr_masks_arr[j, : p.shape[0]] = p

    out: FlatBuffer = {
        "obs_list": obs_list,
        "actions": np.stack(actions, axis=0).astype(np.int64),
        "old_log_probs": np.asarray(old_lp, dtype=np.float32),
        "action_masks": np.stack(masks, axis=0).astype(bool),
        "pointer_masks": ptr_masks_arr,
        "mode_masks": np.stack(m_masks, axis=0).astype(bool),
        "advantages": np.asarray(advs, dtype=np.float32),
    }
    if has_nvm:
        M_max = max(m.shape[0] for m in nvm_list)
        nvm_arr = np.zeros((len(nvm_list), M_max), dtype=bool)
        for i, m in enumerate(nvm_list):
            nvm_arr[i, : m.shape[0]] = m
        out["net_valid_masks"] = nvm_arr
    return out


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

    Row-major order matches ``coll.obs_list``.
    """
    T, N = advantages.shape
    assert len(coll.obs_list) == T * N

    actions = coll.actions.reshape(T * N, 3)
    log_probs = coll.log_probs.reshape(T * N)
    masks = coll.action_masks.reshape(T * N, NUM_ACTIONS)
    K = coll.pointer_masks.shape[-1] if coll.pointer_masks.ndim == 3 else 0
    ptr_masks = coll.pointer_masks.reshape(T * N, K)
    m_masks = coll.mode_masks.reshape(T * N, 3)
    advs = advantages.reshape(T * N)
    rets = returns.reshape(T * N)
    old_vals = coll.values.reshape(T * N)

    out: FlatBuffer = {
        "obs_list": coll.obs_list,
        "actions": actions.astype(np.int64),
        "old_log_probs": log_probs.astype(np.float32),
        "action_masks": masks.astype(bool),
        "pointer_masks": ptr_masks.astype(np.int64),
        "mode_masks": m_masks.astype(bool),
        "advantages": advs.astype(np.float32),
        "returns": rets.astype(np.float32),
        "old_values": old_vals.astype(np.float32),
    }
    if coll.net_valid_masks is not None:
        M_max = coll.net_valid_masks.shape[-1]
        out["net_valid_masks"] = (
            coll.net_valid_masks.reshape(T * N, M_max).astype(bool)
        )
    if coll.walk_flat is not None:
        assert coll.walk_flat["B"] == T * N
        out["walk_flat"] = coll.walk_flat
    return out
