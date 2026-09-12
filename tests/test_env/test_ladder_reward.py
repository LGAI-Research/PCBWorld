"""Contract tests for the ladder reward bonuses.

Ladder terms added on top of the base potential:
    + net_completion_bonus  · (#fully-connected target nets)
    + net_clean_bonus       · (#target nets with zero violations, severity set)
    + clean_completion_bonus· I(total violation count == 0, orphans included)

Key invariants:
- All three default to 0.0 → existing configs see an unchanged potential.
- Per-net clean count excludes the phantom ``<orphan>`` key; the board-level
  clean indicator does NOT (orphan violations block it).
- Connectivity-dependent bonuses fail loudly when the snapshot lacks per-net
  connectivity (sentinel -1), never silently skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcb_world.core.reward import PotentialReward, RewardState
from pcb_world.engine.drc import ORPHAN_NET_KEY

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REWARD_RULES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "reward_rules"


def _state(unconnected=0, drc=0, per_net=None, connected=-1, targets=-1,
           uc=None):
    per_net = per_net or {}
    return RewardState(
        unconnected=unconnected,
        drc_violations=drc,
        wirelength=0.0,
        track_count=0,
        drc_violations_per_net=dict(per_net),
        drc_errors=drc,
        drc_errors_per_net=dict(per_net),
        connected_net_count=connected,
        target_net_count=targets,
        unconnected_net_codes=(frozenset(uc) if uc is not None else None),
    )


def _ladder(**overrides):
    kwargs = dict(
        completion_bonus=3.0,
        net_completion_bonus=1.0,
        net_clean_bonus=1.0,
        clean_completion_bonus=3.0,
        unconnected_penalty=1.0,
        drc_shape="linear",
        drc_penalty=1.0,
        step_penalty=0.0,
    )
    kwargs.update(overrides)
    return PotentialReward(**kwargs)


def test_defaults_off_leave_potential_unchanged():
    base = PotentialReward(completion_bonus=3.0, drc_shape="linear",
                           drc_penalty=1.0, step_penalty=0.0)
    st = _state(unconnected=3, drc=2, per_net={"A": 2})
    # Sentinel -1 connectivity must not matter when the knobs are off.
    assert base.potential(st) == pytest.approx(-3 - 2)


def test_ladder_reset_state():
    # Reset: 5 nets, 7 edges unconnected, no copper → all nets clean.
    r = _ladder()
    st = _state(unconnected=7, connected=0, targets=5)
    # -7 (edges) + 0 (connected) + 5 (clean nets) + 3 (board clean)
    assert r.potential(st) == pytest.approx(-7 + 5 + 3)


def test_ladder_full_clean_completion():
    r = _ladder()
    st = _state(unconnected=0, connected=5, targets=5)
    # +3 (completion) + 5 (net completion) + 5 (clean) + 3 (board clean)
    assert r.potential(st) == pytest.approx(3 + 5 + 5 + 3)


def test_net_completion_event_is_plus_two_and_board_final_plus_five():
    r = _ladder()
    # 2-pin net closes its last edge (not board-final): edge +1, net +1.
    before = _state(unconnected=2, connected=3, targets=5)
    after = _state(unconnected=1, connected=4, targets=5)
    assert r.potential(after) - r.potential(before) == pytest.approx(2.0)
    # Board-final completion stacks the global +3 → +5.
    final = _state(unconnected=0, connected=5, targets=5)
    assert r.potential(final) - r.potential(after) == pytest.approx(5.0)


def test_first_violation_on_clean_board_is_minus_five():
    r = _ladder()
    clean = _state(unconnected=0, connected=5, targets=5)
    dirty = _state(unconnected=0, drc=1, per_net={"A": 1},
                   connected=5, targets=5)
    # -1 linear, -1 net-clean lost, -3 board-clean lost.
    assert r.potential(dirty) - r.potential(clean) == pytest.approx(-5.0)
    # A further violation on the same net costs only the linear -1.
    dirtier = _state(unconnected=0, drc=2, per_net={"A": 2},
                     connected=5, targets=5)
    assert r.potential(dirtier) - r.potential(dirty) == pytest.approx(-1.0)


def test_orphan_violations_block_board_clean_but_not_net_clean():
    r = _ladder()
    st = _state(unconnected=0, drc=1, per_net={ORPHAN_NET_KEY: 1},
                connected=5, targets=5)
    # completion 3 + net completion 5 + all 5 nets still clean 5
    # + board clean 0 (orphan blocks) - 1 linear
    assert r.potential(st) == pytest.approx(3 + 5 + 5 - 1)


def test_missing_connectivity_fails_loudly():
    r = _ladder()
    with pytest.raises(ValueError, match="net-subset routing"):
        r.potential(_state(unconnected=1))


def test_yaml_rule_builds():
    from pcb_world.core.reward_config import load_reward_config

    cfg = load_reward_config(REWARD_RULES_DIR / "reward_ladder_wlnorm.yaml")
    r = cfg.build_reward()
    assert r.net_completion_bonus == pytest.approx(1.0)
    assert r.net_clean_bonus == pytest.approx(1.0)
    assert r.clean_completion_bonus == pytest.approx(3.0)
    assert r.drc_shape == "linear"
    assert r.wirelength_bbox_normalize is True



# --- Size-weighted ladder (net_bonus_size_log_scale) -----------------------

import math


def _sw_ladder(**overrides):
    """Weighted twin of _ladder: two nets A(code 1, w=ln4) / B(code 2, w=ln3)."""
    r = _ladder(net_bonus_size_log_scale=1.0, **overrides)
    r.set_net_size_weights({1: math.log(4), 2: math.log(3)},
                           {"A": math.log(4), "B": math.log(3)})
    return r


def test_weighted_reset_and_net_close_math():
    r = _sw_ladder()
    wA, wB = math.log(4), math.log(3)
    # Reset: A has 2 open edges, B has 1; open edges count as "unconnected"
    # violations → both nets dirty, no weighted bonus active.
    reset = _state(unconnected=3, drc=3, per_net={"A": 2, "B": 1},
                   connected=0, targets=2, uc={1, 2})
    assert r.potential(reset) == pytest.approx(-3 - 3)
    # B fully routed, clean: earns its completion weight AND clean weight.
    after = _state(unconnected=2, drc=2, per_net={"A": 2},
                   connected=1, targets=2, uc={1})
    assert r.potential(after) == pytest.approx(-2 - 2 + 2 * wB)
    # Board complete and clean: completion 3 + board clean 3 + both nets'
    # completion+clean weights.
    final = _state(unconnected=0, connected=2, targets=2, uc=set())
    assert r.potential(final) == pytest.approx(3 + 3 + 2 * (wA + wB))


def test_weighted_dirty_outside_universe_does_not_subtract():
    r = _sw_ladder()
    st = _state(unconnected=0, drc=2,
                per_net={ORPHAN_NET_KEY: 1, "X": 1},
                connected=2, targets=2, uc=set())
    # Orphan + non-universe dirt: linear -2, board clean lost (-3), but both
    # weighted nets keep completion+clean weights.
    wsum = math.log(4) + math.log(3)
    assert r.potential(st) == pytest.approx(3 + 2 * wsum - 2)


def test_weighted_unresolved_weights_fail_loudly():
    r = _ladder(net_bonus_size_log_scale=1.0)
    with pytest.raises(ValueError, match="unresolved"):
        r.potential(_state(unconnected=1, connected=0, targets=2, uc={1}))


def test_weighted_requires_unconnected_codes_in_state():
    r = _sw_ladder()
    with pytest.raises(ValueError, match="unconnected_net_codes"):
        r.potential(_state(unconnected=1, connected=0, targets=2))


def test_weight_map_mismatch_fails():
    r = _ladder(net_bonus_size_log_scale=1.0)
    with pytest.raises(ValueError, match="disagree"):
        r.set_net_size_weights({1: 1.0, 2: 1.0}, {"A": 1.0})
    with pytest.raises(ValueError, match="positive finite"):
        r.set_net_size_weights({1: 0.0}, {"A": 0.0})


def test_negative_size_log_scale_rejected():
    with pytest.raises(ValueError, match="net_bonus_size_log_scale"):
        _ladder(net_bonus_size_log_scale=-0.5)


def test_env_resolves_weights_and_clean_log_scale(board_path, tmp_path):
    # W6-style rule: size-weighted ladder + both board bonuses 2*ln(1+N).
    import gc

    from pcb_world.core.env import PCBWorld

    rule = tmp_path / "reward_sw.yaml"
    rule.write_text(
        "name: sw_test\nmode: per_step\npotential:\n"
        "  completion_bonus_log_scale: 2.0\n"
        "  clean_completion_bonus_log_scale: 2.0\n"
        "  net_completion_bonus: 1.0\n"
        "  net_clean_bonus: 1.0\n"
        "  net_bonus_size_log_scale: 1.0\n"
        "  unconnected_penalty: 1.0\n"
        "  drc_shape: log_per_net\n"
        "  drc_log_scale: 1.0\n"
        "  drc_log_agg_scale: 3.0\n"
        "  drc_log_offset: 2.0\n"
        "  step_penalty: 0.0\n"
        "  drc_severity_mode: errors_and_promoted\n"
    )
    env = PCBWorld(board_path=str(board_path), max_steps=10,
                   reward_rule=str(rule))
    try:
        env.reset()
        pot = env._potential_reward
        n = env._meta.net_count
        assert pot.completion_bonus == pytest.approx(2.0 * math.log1p(n))
        assert pot.clean_completion_bonus == pytest.approx(2.0 * math.log1p(n))
        groups = env._engine.get_pad_groups()
        assert set(pot.net_size_weights_by_code) == set(env._routable_nets)
        for c, w in pot.net_size_weights_by_code.items():
            assert w == pytest.approx(math.log1p(groups[c]))
        assert len(pot.net_size_weights_by_name) == len(
            pot.net_size_weights_by_code)
        st = env._reward.prev_state
        assert st.unconnected_net_codes is not None
        pot.potential(st)  # must not raise
        assert math.isfinite(env._initial_potential)
    finally:
        env.close()
        del env
        gc.collect()


def test_env_snapshot_populates_connectivity(board_path):
    # Env always routes a net subset → snapshot must carry real counts and
    # the ladder potential must evaluate at reset without raising.
    import gc

    from pcb_world.core.env import PCBWorld

    env = PCBWorld(
        board_path=str(board_path), max_steps=10,
        reward_rule=str(REWARD_RULES_DIR / "reward_ladder_wlnorm.yaml"),
    )
    try:
        env.reset()
        st = env._reward.prev_state
        assert st.target_net_count > 0
        assert 0 <= st.connected_net_count <= st.target_net_count
        env._potential_reward.potential(st)  # must not raise
    finally:
        env.close()
        del env
        gc.collect()
