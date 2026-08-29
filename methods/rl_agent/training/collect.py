"""Rollout collectors for decoder-policy RL — PPO fixed-n_steps + GRPO group.

SB3-style ``collect_rollouts`` for the two algorithms, both thin adapters over
the shared per-step primitive :func:`methods.rl_agent.rollout.primitive.
iter_rollout` (which owns act-time mask gathering, the forward, and stepping):
:func:`collect_n_steps_ppo` (fixed ``n_steps``, auto-reset with reset-obs
re-injection, truncation bootstrap → :class:`PPOCollectorOutput`) and
:func:`collect_group_episodes` (full-episode group rollout → per-env
:class:`GroupStepRecord` trajectories). The mask helpers kept here serve the
adapter-side forwards only (truncation bootstrap + final values). Both feed
:mod:`methods.rl_agent.training.buffer`. Carved from ``decoder_common`` (C1-b).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch

from pcb_world.core.masking import NUM_ACTIONS
from pcb_world.vec.backends.base import VecBackend
from methods.rl_agent.policy.agent import gather_mask_arrays, mask_arrays_to_tensors
from methods.rl_agent.rollout.primitive import budgeted_forward, iter_rollout

if TYPE_CHECKING:
    from pcb_world.vec.backends.subproc import SubprocDecoderVecEnv
    from methods.rl_agent.models.v1.net import KiCadRLModel
    from methods.rl_agent.policy.agent import KiCadRLAgent
    from methods.rl_agent.training.mem_budget import MemBudgetModel
    from methods.rl_agent.training.utils import RewardNormalizer
    from methods.rl_agent.wrappers.adapter import KiCadRLWrapper


def _pad_pointer_mask_steps(
    per_step: list[np.ndarray],
) -> np.ndarray:
    """Pad a per-step list of ``(N, K_t) int64`` arrays into ``(T, N, K_max)``
    right-padded with ``-1``. Returns ``(T, N, 0)`` when no step has any
    match.
    """
    T = len(per_step)
    if T == 0:
        return np.zeros((0, 0, 0), dtype=np.int64)
    N = per_step[0].shape[0]
    K_max = max((p.shape[1] for p in per_step), default=0)
    if K_max == 0:
        return np.full((T, N, 0), -1, dtype=np.int64)
    out = np.full((T, N, K_max), -1, dtype=np.int64)
    for t, p in enumerate(per_step):
        if p.shape[1]:
            out[t, :, : p.shape[1]] = p
    return out


class GroupStepRecord(TypedDict, total=False):
    obs: dict
    action: np.ndarray         # (3,) int64
    log_prob: float
    action_mask: np.ndarray    # (NUM_ACTIONS,) bool
    pointer_mask: int          # start_route cand idx or -1 (same-point mask)
    mode_mask: np.ndarray      # (3,) bool — routing-mode mask
    reward: float
    terminated: bool
    truncated: bool
    net_valid_mask: np.ndarray  # (M,) bool — policy-driven net_select only


def collect_group_episodes(
    envs: SubprocDecoderVecEnv | list[KiCadRLWrapper],
    # Inference surface only (act_and_value / eval / attrs): a
    # methods.rl_agent.policy.agent.KiCadRLAgent facade or a bare KiCadRLModel.
    policy: "KiCadRLAgent | KiCadRLModel",
    device: torch.device,
    *,
    max_steps: int = 200,
    mem_budget: "MemBudgetModel | None" = None,
) -> tuple[
    list[list[GroupStepRecord]],
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Run group_size envs until all are done or max_steps hit (GRPO).

    The per-step act+step transition is the shared primitive
    :func:`methods.rl_agent.rollout.primitive.iter_rollout`; this collector
    owns the GRPO episode semantics on top of it. Done envs are excluded
    from each subsequent step (no further ``action_masks()`` / forward
    calls — terminal-state masks may be all-False which would NaN the
    policy distribution).

    Supports both ``SubprocDecoderVecEnv`` (subprocess-parallel) and
    plain ``list[KiCadRLWrapper]`` (sequential fallback).

    Returns (in order, all of shape ``(group_size,)`` except trajectories):
        trajectories, cumulative_rewards, terminal_drc_violations,
        terminal_final_potentials, terminal_unrouted_counts,
        terminal_terminated, terminal_ratsnest_reduction, terminal_wirelength,
        terminal_via_count, terminal_track_count.

    The four trailing arrays mirror the corresponding ``episode_*`` lists
    on :class:`PPOCollectorOutput` so GRPO can log the same rollout
    scalars as PPO.
    """
    group_size = len(envs)

    # --- Reset (vec-parallel or sequential) ---
    if isinstance(envs, VecBackend):
        obs_list: list[dict] = envs.reset_all()
    else:
        obs_list = [env.reset()[0] for env in envs]

    trajectories: list[list[GroupStepRecord]] = [[] for _ in range(group_size)]
    cumulative_rewards = np.zeros(group_size, dtype=np.float64)
    terminal_drc_violations = np.zeros(group_size, dtype=np.float64)
    terminal_final_potentials = np.zeros(group_size, dtype=np.float64)
    terminal_unrouted_counts = np.zeros(group_size, dtype=np.float64)
    terminal_terminated = np.zeros(group_size, dtype=np.float64)
    terminal_ratsnest_reduction = np.zeros(group_size, dtype=np.float64)
    terminal_wirelength = np.zeros(group_size, dtype=np.float64)
    terminal_via_count = np.zeros(group_size, dtype=np.float64)
    terminal_track_count = np.zeros(group_size, dtype=np.float64)

    # done ownership (iter_rollout contract): GRPO marks env-done here; the
    # primitive only reads it. board_static sharing (memory dedup) is the
    # primitive's obs-advance ref propagation.
    done: dict[int, bool] = {i: False for i in range(group_size)}
    obs_by_slot: dict[int, dict] = dict(enumerate(obs_list))

    policy.eval()
    for batch in iter_rollout(
        envs, policy, device, obs_by_slot, list(range(group_size)), done,
        want_value=True, max_steps=max_steps, mem_budget=mem_budget,
    ):
        for k, i in enumerate(batch.live):
            rec: GroupStepRecord = {
                "obs": batch.obs[k],
                "action": batch.actions[k].copy(),
                "log_prob": float(batch.log_probs[k]),
                "action_mask": batch.action_masks[k].copy(),
                "pointer_mask": batch.pointer_masks[k].copy(),
                "mode_mask": batch.mode_masks[k].copy(),
                "reward": float(batch.rewards[k]),
                "terminated": bool(batch.terminateds[k]),
                "truncated": bool(batch.truncateds[k]),
            }
            if batch.net_valid_masks is not None:
                rec["net_valid_mask"] = batch.net_valid_masks[k].copy()
            trajectories[i].append(rec)
            cumulative_rewards[i] += float(batch.rewards[k])
            if batch.terminateds[k] or batch.truncateds[k]:
                done[i] = True
                info = batch.infos[k]
                terminal_drc_violations[i] = float(
                    info.get("drc_violations", 0)
                )
                terminal_final_potentials[i] = float(
                    info.get("final_potential", 0.0)
                )
                terminal_unrouted_counts[i] = float(
                    info.get("unrouted_count", 0)
                )
                terminal_terminated[i] = (
                    0.0 if info.get("TimeLimit.truncated", False) else 1.0
                )
                terminal_ratsnest_reduction[i] = float(
                    info.get("ratsnest_reduction", 0.0)
                )
                terminal_wirelength[i] = float(
                    info.get("wirelength", 0.0)
                )
                terminal_via_count[i] = float(
                    info.get("via_count", 0)
                )
                terminal_track_count[i] = float(
                    info.get("track_count", 0)
                )

    return (
        trajectories,
        cumulative_rewards,
        terminal_drc_violations,
        terminal_final_potentials,
        terminal_unrouted_counts,
        terminal_terminated,
        terminal_ratsnest_reduction,
        terminal_wirelength,
        terminal_via_count,
        terminal_track_count,
    )


@dataclass
class PPOCollectorOutput:
    """Output of :func:`collect_n_steps_ppo`. Mirrors SB3 RolloutBuffer.

    All arrays have shape ``(T, N)`` where ``T = n_steps`` and
    ``N = n_envs``, except ``obs_list`` (Python list of length ``T*N`` in
    row-major order with ``buffer[t*N + n]`` for env n at step t) and
    ``final_values`` ``(N,)``.
    """
    obs_list: list[dict]                  # length T * N (row-major)
    actions: np.ndarray                   # (T, N, 3) int64
    log_probs: np.ndarray                 # (T, N)    float32
    action_masks: np.ndarray              # (T, N, NUM_ACTIONS) bool
    pointer_masks: np.ndarray             # (T, N, K) int64 — start_route cand idxs (-1 pad)
    mode_masks: np.ndarray                # (T, N, 3) bool — routing-mode mask
    rewards: np.ndarray                   # (T, N)    float32 — possibly normalized
    raw_rewards: np.ndarray               # (T, N)    float32 — env reward, never normalized
    values: np.ndarray                    # (T, N)    float32
    episode_starts: np.ndarray            # (T, N)    bool
    terminated_mask: np.ndarray           # (T, N)    bool — true terminated
    terminal_values: np.ndarray           # (T, N)    float32 — V(true_next_obs)
                                          # at boundaries; np.nan elsewhere
    final_values: np.ndarray              # (N,)      float32 — V(s_{T+1})
    episode_rewards: list[float]          # cumulative reward of every
                                          # episode that completed within
                                          # this rollout (for logging)
    episode_lengths: list[int]            # corresponding lengths
    episode_drc_violations: list[float]   # final DRC violation count per
                                          # completed episode (from info
                                          # dict at terminal/truncated)
    episode_final_potentials: list[float] # Φ(s_final) per completed episode
    episode_unrouted_counts: list[float] = field(default_factory=list)
    episode_wirelengths: list[float] = field(default_factory=list)
    episode_via_counts: list[float] = field(default_factory=list)
    episode_track_counts: list[float] = field(default_factory=list)
    episode_terminated: list[float] = field(default_factory=list)
    # Signed ratsnest shrinkage at episode end ((u_0 - u_t) / u_0; 1.0 = fully
    # routed, negative when the board grew islands). Progress proxy, not the
    # pad-group routability metric — that one is eval-stage only.
    episode_ratsnest_reduction: list[float] = field(default_factory=list)
    # Policy-driven net_select only: (T, N, M_max) bool; None when disabled.
    net_valid_masks: np.ndarray | None = None
    # Fraction of steps the engine rejected the (mask-passing) action
    # (info["action_success"] is False) — the invalid_action_penalty rate.
    invalid_action_ratio: float = 0.0
    # Worker deaths during this rollout's step path (info["engine_crash"] —
    # the synthesized terminated response after a respawn). Each one is also
    # a contaminated episode (reward -1.0 terminal).
    engine_crash_count: int = 0
    # Per-step engine latency (info["step_time_s"]: dispatch → connectivity →
    # snapshot, seconds) for every valid action this rollout — stall telemetry.
    step_times: list[float] = field(default_factory=list)
    # Collect-time flat batched walk over the whole rollout (one walk dict,
    # ``B == T*N``, row-major aligned with obs_list): the per-step batched
    # walks the forward already ran, merged once at collect end — the update
    # index-gathers minibatches from it instead of re-walking
    # (``policy_update_loop`` walk-cache). None when the collect path did not
    # cache it (mock policy without tokenizer; then the update re-walks).
    walk_flat: dict | None = None


def collect_n_steps_ppo(
    envs: SubprocDecoderVecEnv | list[KiCadRLWrapper],
    # Inference surface only (act_and_value / eval / attrs): a
    # methods.rl_agent.policy.agent.KiCadRLAgent facade or a bare KiCadRLModel.
    policy: "KiCadRLAgent | KiCadRLModel",
    device: torch.device,
    *,
    n_steps: int,
    reward_normalizer: "RewardNormalizer | None" = None,
    bootstrap_truncation: bool = True,
    mem_budget: "MemBudgetModel | None" = None,
) -> PPOCollectorOutput:
    """SB3-style PPO rollout: fixed ``n_steps`` per env, auto-reset on done.

    The per-step act+step transition is the shared primitive
    :func:`methods.rl_agent.rollout.primitive.iter_rollout`; this collector
    owns the PPO semantics on top of it (fixed window, autoreset with
    reset-obs re-injection, truncation bootstrap, reward normalization).

    Supports both ``SubprocDecoderVecEnv`` (subprocess-parallel) and
    plain ``list[KiCadRLWrapper]`` (sequential fallback).

    On terminal/truncated:
      - terminated → terminal_value = 0 (true end of episode).
      - truncated (not terminated) → terminal_value = ``V(true_next_obs)``
        (computed via a single-obs forward before the env auto-resets),
        unless ``bootstrap_truncation=False`` (legacy: V=0).

    The ``obs_list`` is row-major: ``[(t=0, n=0), (t=0, n=1), ..., (t=0, n=N-1),
    (t=1, n=0), ...]``.

    Args:
        reward_normalizer: optional :class:`RewardNormalizer`. When provided,
            ``rewards`` stored in the output (and consumed by GAE) are the
            normalized values; ``episode_rewards`` continue to use **raw**
            rewards (matching SB3 ``Monitor`` semantics — Monitor sits below
            ``VecNormalize`` in the wrapper stack).
        bootstrap_truncation: if True (default), compute ``V(s_next)`` at
            truncation boundaries for GAE bootstrap. If False, use V=0
            (legacy behaviour that treats truncation as termination).
    """
    n_envs = len(envs)
    # The per-step act/step/mask machinery (incl. board_static dedup) lives
    # in iter_rollout; only the reset mechanism still branches on the backend.
    use_vec = isinstance(envs, VecBackend)

    # --- Reset ---
    if use_vec:
        obs_list_per_env: list[dict] = envs.reset_all()
    else:
        obs_list_per_env = [env.reset()[0] for env in envs]

    cumulative_rewards_raw = np.zeros(n_envs, dtype=np.float64)
    cumulative_lengths = np.zeros(n_envs, dtype=np.int64)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_drc_violations: list[float] = []
    episode_final_potentials: list[float] = []
    episode_unrouted_counts: list[float] = []
    episode_wirelengths: list[float] = []
    episode_via_counts: list[float] = []
    episode_track_counts: list[float] = []
    episode_terminated: list[float] = []
    episode_ratsnest_reduction: list[float] = []

    obs_buffer: list[dict] = []
    # Per-step batched walks (each B == n_envs, env-index order) — kept flat,
    # merged ONCE after the loop into walk_flat (see PPOCollectorOutput).
    # None tokenizer (scripted/mock policy) → no cache; the update re-walks.
    step_walks: list[dict] = []
    tokenizer = getattr(getattr(policy, "model", policy), "tokenizer", None)
    actions_buf = np.zeros((n_steps, n_envs, 3), dtype=np.int64)
    log_probs_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    masks_buf = np.zeros((n_steps, n_envs, NUM_ACTIONS), dtype=bool)
    # Per-step variable-K (B, K_t) arrays; padded to (T, N, K_max) at end.
    ptr_masks_steps: list[np.ndarray] = []
    mode_masks_buf = np.ones((n_steps, n_envs, 3), dtype=bool)
    # Per-step net_valid_masks list (variable M); padded at end if policy_net_select.
    policy_ns = getattr(policy, "policy_net_select", False)
    nvm_steps: list[np.ndarray] = []
    rewards_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    raw_rewards_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    values_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    episode_starts_buf = np.zeros((n_steps, n_envs), dtype=bool)
    terminated_buf = np.zeros((n_steps, n_envs), dtype=bool)
    terminal_values_buf = np.full((n_steps, n_envs), np.nan, dtype=np.float32)

    # Track which envs are at episode boundary for next step's episode_starts.
    next_episode_start = np.ones(n_envs, dtype=bool)
    invalid_steps = 0  # engine-rejected actions (info["action_success"] False)
    engine_crashes = 0  # worker deaths synthesized as terminated (info["engine_crash"])
    step_times: list[float] = []  # engine latency per valid step (info["step_time_s"])

    # done ownership (iter_rollout contract): PPO never marks done — episodes
    # auto-reset across boundaries and the reset obs is re-injected into the
    # shared obs_by_slot between yields, so every env is live every step.
    done: dict[int, bool] = {i: False for i in range(n_envs)}
    obs_by_slot: dict[int, dict] = dict(enumerate(obs_list_per_env))

    policy.eval()
    for t, batch in enumerate(iter_rollout(
        envs, policy, device, obs_by_slot, list(range(n_envs)), done,
        want_value=True, max_steps=n_steps, mem_budget=mem_budget,
    )):
        # Record episode_start at THIS step (set by previous iteration's done).
        episode_starts_buf[t] = next_episode_start

        # Per-step net_valid_masks (policy-driven net_select only).
        if policy_ns:
            nvm_steps.append(batch.net_valid_masks)

        # Record act-time obs + the exact act-time masks (§3.4: stored, never
        # regenerated at update time). live == all envs, so batch arrays are
        # aligned with env index.
        for i in range(n_envs):
            obs_buffer.append(batch.obs[i])
        # Cache the walk: keep the step's batched walk (over all live == all
        # envs, env-index order) AS-IS — no per-sample split. The forward
        # already computed it.
        if batch.walk is not None:
            step_walks.append(batch.walk)
        actions_buf[t] = batch.actions
        log_probs_buf[t] = batch.log_probs
        masks_buf[t] = batch.action_masks
        ptr_masks_steps.append(batch.pointer_masks)
        mode_masks_buf[t] = batch.mode_masks
        values_buf[t] = batch.values

        # Reset next_episode_start
        next_episode_start = np.zeros(n_envs, dtype=bool)

        # Per-step buffers for this iteration of t (used for batch
        # reward normalization once all envs have been stepped).
        step_raw_rewards = np.zeros(n_envs, dtype=np.float64)
        step_dones = np.zeros(n_envs, dtype=bool)

        # --- Process step results (stepping happened inside iter_rollout) ---
        for i in range(n_envs):
            step_raw_rewards[i] = float(batch.rewards[i])
            terminated = bool(batch.terminateds[i])
            truncated = bool(batch.truncateds[i])
            step_dones[i] = terminated or truncated
            terminated_buf[t, i] = terminated
            cumulative_rewards_raw[i] += float(batch.rewards[i])
            cumulative_lengths[i] += 1
            if not batch.infos[i].get("action_success", True):
                invalid_steps += 1
            if batch.infos[i].get("engine_crash", False):
                engine_crashes += 1
            _st = batch.infos[i].get("step_time_s")
            if _st is not None:
                step_times.append(float(_st))

            if terminated or truncated:
                info = batch.infos[i]
                if truncated and not terminated and bootstrap_truncation:
                    # Truncation: bootstrap V(true_next_obs) from the yielded
                    # obs_next (not-yet-reset state — our custom worker does
                    # NOT auto-reset). Masks come from obs_next's embedded
                    # "_masks" via the shared gather glue; envs without the
                    # payload (stubs) fall through to the query path.
                    t_masks, t_ptr, t_mode, t_nvm, t_off = gather_mask_arrays(
                        envs, [i], policy_net_select=policy_ns,
                        obs_list=[batch.obs_next[i]],
                    )
                    trunc_mask_t, trunc_ptr_t, trunc_mode_t, trunc_nvm_kwargs = (
                        mask_arrays_to_tensors(
                            t_masks, t_ptr, t_mode, t_nvm, device, off_masks=t_off,
                            mode_none_if_all_true=True,
                        )
                    )
                    _, _, trunc_val = policy.act_and_value(
                        [batch.obs_next[i]],
                        action_masks=trunc_mask_t,
                        pointer_masks=trunc_ptr_t,
                        mode_mask=trunc_mode_t,
                        **trunc_nvm_kwargs,
                    )
                    terminal_values_buf[t, i] = trunc_val.item()
                else:
                    terminal_values_buf[t, i] = 0.0

                episode_rewards.append(float(cumulative_rewards_raw[i]))
                episode_lengths.append(int(cumulative_lengths[i]))
                episode_drc_violations.append(
                    float(info.get("drc_violations", 0))
                )
                episode_final_potentials.append(
                    float(info.get("final_potential", 0.0))
                )
                episode_unrouted_counts.append(
                    float(info.get("unrouted_count", 0))
                )
                episode_wirelengths.append(
                    float(info.get("wirelength", 0.0))
                )
                episode_via_counts.append(
                    float(info.get("via_count", 0))
                )
                episode_track_counts.append(
                    float(info.get("track_count", 0))
                )
                episode_terminated.append(
                    0.0 if info.get("TimeLimit.truncated", False)
                    else 1.0
                )
                episode_ratsnest_reduction.append(
                    float(info.get("ratsnest_reduction", 0.0))
                )
                cumulative_rewards_raw[i] = 0.0
                cumulative_lengths[i] = 0

                if not use_vec:
                    # Sequential fallback: reset inline (preserving the
                    # per-env bootstrap->reset order) and re-inject the
                    # reset obs into the shared obs_by_slot (iter_rollout
                    # re-reads it at the top of the next iteration).
                    obs_by_slot[i] = envs[i].reset()[0]
                    next_episode_start[i] = True

        if use_vec:
            # Batch reset all done envs (parallel) and re-inject reset obs.
            done_indices = [i for i in range(n_envs) if step_dones[i]]
            if done_indices:
                reset_results = envs.reset_batch(done_indices)
                for k, i in enumerate(done_indices):
                    obs_by_slot[i] = reset_results[k][0]
                    next_episode_start[i] = True

        # Always store the raw env rewards (for diagnostics).
        raw_rewards_buf[t] = step_raw_rewards.astype(np.float32)

        # Reward normalization (SB3 VecNormalize-equivalent). Done in
        # one batched call per step so the running stats see all envs at
        # once. The normalizer also resets per-env discounted returns at
        # episode boundaries — must use ``step_dones``, not the raw
        # terminated mask, so truncations also reset.
        if reward_normalizer is not None:
            normalized = reward_normalizer.normalize_step(
                step_raw_rewards, step_dones,
            )
            rewards_buf[t] = normalized
        else:
            rewards_buf[t] = step_raw_rewards.astype(np.float32)

    # Compute final values for bootstrap of the last step in non-terminal
    # envs (terminal envs at t=n_steps-1 are handled via terminal_values).
    # Masks via the shared gather glue (obs-embedded "_masks" first, query
    # fallback otherwise); mode_none_if_all_true=True keeps the previous
    # None-if-all-True training convention.
    obs_list_per_env = [obs_by_slot[i] for i in range(n_envs)]
    f_masks, f_ptr, f_mode, f_nvm, f_off = gather_mask_arrays(
        envs, list(range(n_envs)), policy_net_select=policy_ns,
        obs_list=obs_list_per_env,
    )
    final_mask_t, final_ptr_mask_t, final_mode_mask_t, final_nvm_kwargs = (
        mask_arrays_to_tensors(
            f_masks, f_ptr, f_mode, f_nvm, device, mode_none_if_all_true=True,
            off_masks=f_off,
        )
    )
    if mem_budget is not None and mem_budget.ready:
        # Same budget-split as the per-step forward (B = n_envs here too).
        _, _, final_values = budgeted_forward(
            policy, obs_list_per_env, final_mask_t, final_ptr_mask_t,
            final_mode_mask_t, final_nvm_kwargs, mem_budget, want_value=True,
        )
    else:
        _, _, final_values = policy.act_and_value(
            obs_list_per_env,
            action_masks=final_mask_t,
            pointer_masks=final_ptr_mask_t,
            mode_mask=final_mode_mask_t,
            **final_nvm_kwargs,
        )

    # Merge the per-step batched walks (t order == row-major obs_buffer order)
    # into ONE flat walk over the whole rollout — obs_idx becomes the global
    # sample index t*N + n. One concat per type/field; the update gathers
    # minibatches from this without any re-walk.
    walk_flat: dict | None = None
    if step_walks and len(step_walks) == n_steps:
        walk_flat = tokenizer.merge_walked(step_walks)

    # Stack + pad per-step pointer_masks into (T, N, K_max).
    ptr_masks_buf = _pad_pointer_mask_steps(ptr_masks_steps)

    # Stack + pad per-step net_valid_masks to (T, N, M_max). None when disabled.
    nvm_buf: np.ndarray | None = None
    if policy_ns and nvm_steps:
        M_max = max(m.shape[1] for m in nvm_steps)
        nvm_buf = np.zeros((n_steps, n_envs, M_max), dtype=bool)
        for t_i, m in enumerate(nvm_steps):
            nvm_buf[t_i, :, : m.shape[1]] = m

    return PPOCollectorOutput(
        obs_list=obs_buffer,
        actions=actions_buf,
        log_probs=log_probs_buf,
        action_masks=masks_buf,
        pointer_masks=ptr_masks_buf,
        mode_masks=mode_masks_buf,
        rewards=rewards_buf,
        raw_rewards=raw_rewards_buf,
        values=values_buf,
        episode_starts=episode_starts_buf,
        terminated_mask=terminated_buf,
        terminal_values=terminal_values_buf,
        final_values=final_values.cpu().numpy().astype(np.float32),
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_drc_violations=episode_drc_violations,
        episode_final_potentials=episode_final_potentials,
        episode_unrouted_counts=episode_unrouted_counts,
        episode_wirelengths=episode_wirelengths,
        episode_via_counts=episode_via_counts,
        episode_track_counts=episode_track_counts,
        episode_terminated=episode_terminated,
        episode_ratsnest_reduction=episode_ratsnest_reduction,
        net_valid_masks=nvm_buf,
        invalid_action_ratio=float(invalid_steps) / float(max(n_steps * n_envs, 1)),
        engine_crash_count=engine_crashes,
        step_times=step_times,
        walk_flat=walk_flat,
    )

