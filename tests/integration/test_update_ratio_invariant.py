"""Rollout == update: an unchanged policy re-scores every buffer row at ratio 1.

The PPO/GRPO update re-runs ``evaluate_actions_and_value`` on the buffer and
divides by the act-time log-prob. If the update forward sees a different
distribution than the rollout forward (a mask the rollout applied and the
update dropped), the first-epoch ratio ``exp(new_lp - old_lp)`` leaves 1 even
though no parameter moved. Here a real ``KiCadRLWrapper`` rollout on a 2-layer
fixture board — with make_line steps whose off-layer block is non-empty — goes
through ``collect_n_steps_ppo`` → ``ppo_collector_to_buffer`` →
``policy_update_loop`` with a zero learning rate, and every row's update-time
log-prob must equal its rollout log-prob.
"""
from __future__ import annotations

import os

import pytest
import torch

from pcb_world.engine import engine_available
from tests.helpers.env_kwargs import full_env_kwargs

_BOARD = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "crossover_board.kicad_pcb",
)

pytestmark = pytest.mark.skipif(
    not engine_available(), reason="kicad_rl_router not available",
)


def test_first_epoch_ratio_is_one_with_offlayer_rows():
    from methods.rl_agent.algorithms._common import policy_update_loop
    from methods.rl_agent.models.v1.net import KiCadRLModel
    from methods.rl_agent.models.v1.spec import ACT_MAKE_LINE
    from methods.rl_agent.training.buffer import ppo_collector_to_buffer
    from methods.rl_agent.training.collect import collect_n_steps_ppo
    from methods.rl_agent.wrappers.factory import make_decoder_env

    torch.manual_seed(0)
    env = make_decoder_env(_BOARD, **full_env_kwargs(max_steps=64))
    try:
        model = KiCadRLModel(disable_slot_emb=False, policy_net_select=False, d_model=32, n_heads=2, n_layers=2, d_ff=64, use_critic=True)
        coll = collect_n_steps_ppo([env], model, torch.device("cpu"), n_steps=96)
    finally:
        env.close()
    zeros = torch.zeros(coll.rewards.shape).numpy()
    buffer = ppo_collector_to_buffer(coll, zeros, zeros)
    tr = buffer["transition"]

    # Not vacuous: make_line rows whose off-layer block is non-empty exist.
    offlayer_make_line = (tr.actions[:, 0] == ACT_MAKE_LINE) & (tr.dist.offlayer >= 0).any(1)
    assert offlayer_make_line.sum() >= 3

    # Record each row's update-time log-prob, keyed by its obs object.
    new_lp: dict[int, float] = {}
    evaluate = model.evaluate_actions_and_value

    def recording_evaluate(obs_list, actions, **kw):
        lp, ent, val = evaluate(obs_list, actions, **kw)
        for o, x in zip(obs_list, lp.detach().tolist()):
            new_lp[id(o)] = x
        return lp, ent, val

    model.evaluate_actions_and_value = recording_evaluate
    policy_update_loop(
        model, torch.optim.SGD(model.parameters(), lr=0.0), buffer,
        torch.device("cpu"), algo="ppo", n_epochs=1, batch_size=32,
    )

    old_lp = torch.as_tensor(tr.old_log_probs, dtype=torch.float64)
    got = torch.tensor([new_lp[id(o)] for o in buffer["obs_list"]], dtype=torch.float64)
    ratio = torch.exp(got - old_lp)
    bad = (ratio - 1).abs() > 1e-4
    assert not bad.any(), (
        f"{int(bad.sum())}/{len(ratio)} rows re-scored off ratio 1 "
        f"(min ratio {float(ratio.min()):.4f})"
    )
