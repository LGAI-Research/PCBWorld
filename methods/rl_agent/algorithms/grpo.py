"""GRPO per-algorithm pieces — group-relative advantage + critic-free update.

The group-relative baseline (:func:`compute_grpo_advantages` /
:func:`compute_grpo_advantages_grouped`) replaces PPO's value model; the
minibatch loss (:func:`grpo_update_step`) is clipped policy loss + entropy with
no value term. The algorithm-neutral loop lives in
:mod:`methods.rl_agent.algorithms._common`. Carved from ``decoder_common`` (C1-b).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from methods.rl_agent.training.utils import RunningRewardStd


def compute_grpo_advantages(
    terminal_rewards: np.ndarray,
    reward_std_tracker: RunningRewardStd,
) -> np.ndarray:
    """GRPO group-relative baseline + running std scale.

    Returns ``(group_size,)`` per-episode advantage. Each step in episode i
    will use the same scalar :math:`A_i = (R_i - \\bar R) / \\hat\\sigma`.
    """
    reward_std_tracker.update(terminal_rewards)
    group_mean = float(np.mean(terminal_rewards))
    return (terminal_rewards - group_mean) / reward_std_tracker.std


def compute_grpo_advantages_grouped(
    terminal_rewards: np.ndarray,
    group_size: int,
    reward_std_tracker: RunningRewardStd,
) -> np.ndarray:
    """Per-sub-group GRPO advantage when ``n_envs > group_size``.

    The flat ``(n_envs,)`` reward array is sliced into
    ``n_envs // group_size`` contiguous sub-groups of size ``group_size``;
    each sub-group's mean is the local baseline. The running std is
    updated once on the whole batch and shared across sub-groups so the
    scale is stable across iterations (matching :func:`compute_grpo_advantages`).
    """
    n = terminal_rewards.shape[0]
    if n % group_size != 0:
        raise ValueError(
            f"terminal_rewards length {n} not divisible by group_size {group_size}"
        )
    reward_std_tracker.update(terminal_rewards)
    advantages = np.zeros_like(terminal_rewards, dtype=np.float64)
    for g in range(n // group_size):
        sl = slice(g * group_size, (g + 1) * group_size)
        group_mean = float(np.mean(terminal_rewards[sl]))
        advantages[sl] = (terminal_rewards[sl] - group_mean) / reward_std_tracker.std
    return advantages


def grpo_update_step(
    policy_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO minibatch loss: policy + entropy, no value term (``value_loss=0``).

    Returns ``(loss, value_loss)`` — byte-for-byte the GRPO branch of the
    original ``policy_update_loop`` loss assembly.
    """
    value_loss = torch.tensor(0.0, device=device)
    loss = policy_loss + entropy_loss
    return loss, value_loss
