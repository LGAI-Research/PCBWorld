"""Contract tests for the v56 Jumanji Connector dense reward."""

from __future__ import annotations

import pytest

from pcb_world.core.reward import PotentialReward, RewardState
from pcb_world.core.reward_config import get_reward_config


def _state(unconnected: int, wirelength: float = 0.0, via_count: int = 0) -> RewardState:
    return RewardState(
        unconnected=unconnected,
        drc_violations=0,
        wirelength=wirelength,
        track_count=0,
        via_count=via_count,
    )


def test_jumanji_dense_rewards_connection_delta_minus_alive_net_cost() -> None:
    reward = PotentialReward(
        jumanji_connector_dense=True,
        step_penalty=0.03,
        wirelength_penalty=100.0,
        via_penalty=100.0,
        drc_penalty=100.0,
    )

    assert reward.compute_dense(_state(5), _state(4, wirelength=999.0, via_count=3)) == pytest.approx(0.85)
    assert reward.compute_dense(_state(4), _state(4, wirelength=1000.0, via_count=4)) == pytest.approx(-0.12)


def test_jumanji_dense_netend_is_same_as_dense() -> None:
    reward = PotentialReward(jumanji_connector_dense=True, step_penalty=0.003)
    before = _state(5)
    after = _state(3)
    value, next_ref = reward.compute_dense_netend(
        before,
        after,
        wire_via_ref_state=before,
        flush_wire_via=True,
    )

    assert value == pytest.approx(2.0 - 0.003 * 5)
    assert next_ref is before


def test_jumanji_reward_yaml_builds_contract_reward() -> None:
    cfg = get_reward_config("jumanji_connector_dense")
    reward = cfg.build_reward()

    assert cfg.mode == "per_step"
    assert reward.jumanji_connector_dense is True
    assert reward.wirelength_penalty == 0.0
    assert reward.via_penalty == 0.0
    assert reward.drc_penalty == 0.0
    assert reward.compute_dense(_state(1), _state(0)) == pytest.approx(0.97)
