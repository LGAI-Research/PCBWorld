"""Contract tests for config-level reward overrides (``with_overrides``).

Per-run reward overrides (``PCBWorld(via_penalty=...)`` & co.) are applied
to a COPY of the YAML reward config *before* ``build_reward()``
(pcb_world/core/reward_config.py). That pins two invariants:

1. Agreement — the env's reward config and its built PotentialReward report
   the same value for every overridden field (no split-brain reads between
   ``_reward_config`` and ``_potential_reward``).
2. Cache purity — ``get_reward_config()`` hands out cached, process-shared
   instances; overrides must never leak into them (train/eval in the same
   process would silently score with the overridden values otherwise).
"""

from __future__ import annotations

import pytest

from configs.loader.schema import EnvConfig, RewardOverrides
from pcb_world.core.reward_config import get_reward_config

# drc_only_dense.yaml: step_penalty=0.0, no via_penalty key (default 0.0).
RULE = "drc_only_dense"


def test_override_lands_in_copy_and_built_reward():
    base = get_reward_config(RULE)
    cfg = base.with_overrides(via_penalty=0.123, step_penalty=0.456)
    assert cfg is not base
    built = cfg.build_reward()
    assert built.via_penalty == pytest.approx(0.123)
    assert built.step_penalty == pytest.approx(0.456)
    # Config property and built object agree — the old post-build mutation
    # split exactly here (cfg.step_penalty kept the YAML value).
    assert cfg.step_penalty == pytest.approx(0.456)


def test_cached_instance_stays_pristine():
    base = get_reward_config(RULE)
    base.with_overrides(via_penalty=9.9, step_penalty=9.9)
    again = get_reward_config(RULE)
    assert again is base  # still the same cached instance
    built = again.build_reward()
    assert built.via_penalty == pytest.approx(0.0)   # PotentialReward default
    assert built.step_penalty == pytest.approx(0.0)  # YAML value


def test_none_overrides_are_noop():
    base = get_reward_config(RULE)
    assert base.with_overrides(via_penalty=None, wire_via_emission=None) is base


def test_legacy_mode_names_load_as_aliases():
    # Paper-alignment S2: sparse/dense keep loading (old checkpoints, var/
    # snapshots) but resolve to the paper names terminal/per_step.
    from pcb_world.core.reward_config import RewardRuleSchema
    from tests._rule_helpers import RULE_PENALTIES

    for legacy, current in [("sparse", "terminal"), ("dense", "per_step")]:
        with pytest.warns(DeprecationWarning, match=legacy):
            rule = RewardRuleSchema.from_dict({"name": "x", "mode": legacy, "potential": RULE_PENALTIES}, "<inline>")
        assert rule.mode == current


def test_invalid_override_values_fail_at_build():
    base = get_reward_config(RULE)
    with pytest.raises(ValueError, match="drc_log_offset"):
        base.with_overrides(drc_log_offset=0.0).build_reward()
    with pytest.raises(ValueError, match="wire_via_emission"):
        base.with_overrides(wire_via_emission="sometimes").build_reward()


def test_unknown_override_key_fails_at_build():
    base = get_reward_config(RULE)
    with pytest.raises(TypeError):
        base.with_overrides(via_pnalty=0.1).build_reward()  # typo key


def test_env_config_and_built_reward_agree(board_path):
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(
        board_path,
        EnvConfig.with_overrides(reward_rule=RULE, reward=RewardOverrides(via_penalty=0.123, reward_step_penalty=0.456)),
    )
    try:
        assert env._potential_reward.via_penalty == pytest.approx(0.123)
        assert env._potential_reward.step_penalty == pytest.approx(0.456)
        assert env._reward_config.step_penalty == pytest.approx(0.456)
        # Rebuilding from the env's config reproduces the overridden values
        # (completion_bonus differs by design: board-resolved at env init).
        rebuilt = env._reward_config.build_reward()
        assert rebuilt.via_penalty == pytest.approx(
            env._potential_reward.via_penalty
        )
        assert rebuilt.step_penalty == pytest.approx(
            env._potential_reward.step_penalty
        )
        # Process-wide cache still returns pristine YAML values.
        assert get_reward_config(RULE).build_reward().via_penalty == pytest.approx(0.0)
    finally:
        env.close()


def test_effective_reward_config_covers_every_reward_override():
    # One mapping from RewardOverrides to PotentialReward arguments: the env and the eval's
    # scoring-vs-training check both build the trained reward through it.
    from dataclasses import fields

    from pcb_world.core.reward_config import REWARD_OVERRIDE_ARGS, effective_reward_config

    assert set(REWARD_OVERRIDE_ARGS) == {f.name for f in fields(RewardOverrides)}
    cfg = effective_reward_config(RULE, {**vars(RewardOverrides()), "via_penalty": 0.3,
                                         "reward_step_penalty": 0.2})
    assert (cfg.build_reward().via_penalty, cfg.step_penalty) == (0.3, 0.2)


def test_phi_diff_compares_only_what_defines_phi():
    from pcb_world.core.reward_config import phi_diff

    base = get_reward_config("drc_dense_promoted")
    # step_penalty / truncation / emission timing are not Φ; via_penalty is
    assert phi_diff(base, base.with_overrides(step_penalty=0.7, wire_via_emission="on_net_end")) == {}
    assert phi_diff(base, base.with_overrides(via_penalty=0.1)) == {"via_penalty": (0.0, 0.1)}
    # an alias resolves to the same Φ
    assert phi_diff(get_reward_config("drc_log_ladder_linear_w7a"), get_reward_config("pcbworld_reward")) == {}
    # the jumanji connector switches change compute_dense only, not potential()
    import copy

    jumanji = copy.copy(base)
    jumanji._potential_cfg = {**base._potential_cfg, "jumanji_connector_dense": True}
    assert phi_diff(base, jumanji) == {}
