"""PPO per-algorithm update — the value-augmented loss-assembly step.

The algorithm-neutral minibatch loop lives in
:mod:`methods.rl_agent.algorithms._common`; this supplies only PPO's per-minibatch
loss assembly (clipped policy loss + ``vf_coef``·value MSE + entropy), i.e. SB3's
``PPO.train()`` inner step. Carved from ``decoder_common.policy_update_loop``
(C1-b, §3.3 SB3 mapping).
"""
from __future__ import annotations

import torch


def ppo_update_step(
    policy_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    new_values: torch.Tensor,
    mb_ret: torch.Tensor,
    vf_coef: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO minibatch loss: policy + ``vf_coef``·value MSE + entropy.

    Returns ``(loss, value_loss)`` — byte-for-byte the PPO branch of the
    original ``policy_update_loop`` loss assembly.
    """
    value_loss = 0.5 * (new_values - mb_ret).pow(2).mean()
    loss = policy_loss + vf_coef * value_loss + entropy_loss
    return loss, value_loss
