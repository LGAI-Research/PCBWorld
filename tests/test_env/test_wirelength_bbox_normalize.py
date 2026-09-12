"""Contract tests for ``wirelength_bbox_normalize``.

When the flag is set, the env resolves the effective wirelength penalty at
init as ``wirelength_penalty / max(bbox_w, bbox_h)`` (same board-resolution
hook as ``completion_bonus_log_scale``), so the config knob prices wire in
board-long-edge units instead of absolute mm. CLI/ctor overrides compose:
``with_overrides`` applies first, then normalization divides the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcb_world.core.reward import PotentialReward
from pcb_world.core.reward_config import YamlRewardConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REWARD_RULES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "reward_rules"


def test_flag_defaults_off():
    assert PotentialReward().wirelength_bbox_normalize is False


def test_yaml_passes_flag_to_built_reward():
    cfg = YamlRewardConfig({
        "name": "test_wlnorm",
        "mode": "per_step",
        "potential": {
            "wirelength_penalty": 0.5,
            "wirelength_bbox_normalize": True,
        },
    })
    r = cfg.build_reward()
    assert r.wirelength_bbox_normalize is True
    assert r.wirelength_penalty == pytest.approx(0.5)


@pytest.mark.parametrize("yaml_name", [
    "reward_log_wlnorm.yaml",
    "reward_linear_wlnorm.yaml",
])
def test_d2b_wlnorm_rules_build(yaml_name):
    from pcb_world.core.reward_config import load_reward_config

    cfg = load_reward_config(REWARD_RULES_DIR / yaml_name)
    r = cfg.build_reward()
    assert r.wirelength_bbox_normalize is True
    assert r.wirelength_penalty == pytest.approx(0.5)
    assert r.via_penalty == pytest.approx(0.2)
    assert cfg.invalid_action_penalty == pytest.approx(0.01)


def _probe_env(board_path, **kwargs):
    """Build a PCBWorld, read (effective_wl_penalty, bbox long edge), tear down.

    KiCadEngine is a per-process singleton — fully release the native router
    (close + del + gc.collect) before the next construction, mirroring the
    test_reward_modes episode contract.
    """
    import gc

    from pcb_world.core.env import PCBWorld

    env = PCBWorld(board_path=str(board_path), max_steps=10, **kwargs)
    try:
        long_edge = max(env._meta.bbox_w, env._meta.bbox_h)
        effective = env._potential_reward.wirelength_penalty
    finally:
        env.close()
        del env
        gc.collect()
    return effective, long_edge


def test_env_divides_by_board_long_edge(board_path):
    effective, long_edge = _probe_env(
        board_path,
        reward_rule=str(REWARD_RULES_DIR / "reward_linear_wlnorm.yaml"),
    )
    assert long_edge > 0
    assert effective == pytest.approx(0.5 / long_edge)


def test_env_normalizes_ctor_override_too(board_path):
    # with_overrides applies the ctor/CLI value first, then normalization
    # divides — a campaign script passing --wirelength-penalty must land
    # normalized, not absolute.
    effective, long_edge = _probe_env(
        board_path,
        reward_rule=str(REWARD_RULES_DIR / "reward_linear_wlnorm.yaml"),
        wirelength_penalty=0.7,
    )
    assert effective == pytest.approx(0.7 / long_edge)
