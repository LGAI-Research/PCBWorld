"""Contract checks for the v51 reward-ablation launchers.

These tests intentionally avoid the KiCad C++ engine.  They pin the reward
math and CLI semantics that the v51 ablation launch scripts rely on before a
large ablation queue is launched.
"""

from __future__ import annotations

import pytest

from pcb_world.core.reward import PotentialReward, RewardState
from pcb_world.core.reward_config import get_reward_config
from methods.rl_agent.training.train_ppo import build_arg_parser


def _state(
    *,
    unconnected: int,
    wirelength: float = 0.0,
    via_count: int = 0,
    drc_violations: int = 0,
) -> RewardState:
    return RewardState(
        unconnected=unconnected,
        wirelength=wirelength,
        via_count=via_count,
        drc_violations=drc_violations,
        drc_errors=drc_violations,
        track_count=0,
    )


def test_wire_and_via_penalties_are_potential_terms() -> None:
    reward = PotentialReward(
        completion_bonus=0.0,
        unconnected_penalty=0.0,
        drc_penalty=0.0,
        wirelength_penalty=0.01,
        via_penalty=0.5,
        step_penalty=0.0,
    )
    before = _state(unconnected=1, wirelength=10.0, via_count=1)
    after = _state(unconnected=1, wirelength=12.0, via_count=2)

    assert reward._phi_wire_via(after) == pytest.approx(-(0.01 * 12.0 + 0.5 * 2))  # noqa: SLF001
    assert reward.compute_dense(before, after) == pytest.approx(-(0.01 * 2.0 + 0.5))


def test_dense_and_sparse_promoted_configs_share_potential_but_differ_mode() -> None:
    dense = get_reward_config("drc_dense_promoted")
    sparse = get_reward_config("drc_sparse_promoted")

    assert dense.mode == "per_step"
    assert sparse.mode == "terminal"

    dense_reward = dense.build_reward()
    sparse_reward = sparse.build_reward()

    assert dense_reward.wirelength_penalty == pytest.approx(0.0)
    assert sparse_reward.wirelength_penalty == pytest.approx(0.0)
    assert dense_reward.drc_shape == sparse_reward.drc_shape == "log_per_net"
    assert dense_reward.drc_severity_mode == sparse_reward.drc_severity_mode


def test_on_net_end_holds_only_wire_via_until_flush() -> None:
    reward = PotentialReward(
        completion_bonus=0.0,
        unconnected_penalty=1.0,
        drc_penalty=0.0,
        wirelength_penalty=0.02,
        via_penalty=1.0,
        step_penalty=0.0,
    )
    ref = before = _state(unconnected=2, wirelength=0.0, via_count=0)
    after_line = _state(unconnected=2, wirelength=20.0, via_count=1)
    after_net_end = _state(unconnected=1, wirelength=20.0, via_count=1)

    non_flush_reward, held_ref = reward.compute_dense_netend(
        before,
        after_line,
        ref,
        flush_wire_via=False,
    )
    flush_reward, next_ref = reward.compute_dense_netend(
        after_line,
        after_net_end,
        held_ref,
        flush_wire_via=True,
    )

    assert held_ref is ref
    assert next_ref is after_net_end
    assert non_flush_reward == pytest.approx(0.0)
    assert flush_reward == pytest.approx(1.0 - (0.02 * 20.0 + 1.0))


def test_drc_token_flags_only_control_observation_tokens() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--board",
            "tests/fixtures/simple_routing_board.kicad_pcb",
            "--reward-rule",
            "drc_dense_promoted",
            "--wirelength-penalty",
            "0.01",
            "--via-penalty",
            "0.5",
            "--wire-via-emission",
            "on_net_end",
            "--no-drc-tokens",
        ],
    )

    assert args.reward_rule == "drc_dense_promoted"
    assert args.wirelength_penalty == pytest.approx(0.01)
    assert args.via_penalty == pytest.approx(0.5)
    assert args.wire_via_emission == "on_net_end"
    assert args.no_drc_tokens is True

    override = parser.parse_args(
        [
            "--board",
            "tests/fixtures/simple_routing_board.kicad_pcb",
            "--no-drc-tokens",
            "--drc-tokens",
        ],
    )
    assert override.no_drc_tokens is False
