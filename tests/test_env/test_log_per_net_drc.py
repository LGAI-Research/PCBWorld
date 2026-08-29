"""Unit tests for per-net logarithmic DRC potential in PotentialReward.

log_per_net shape:
    Φ_drc(s) = -Σ_i drc_log_scale · ln(1 + x_i / drc_log_offset)

Iterates over state.drc_violations_per_net (orphan violations are grouped
under the phantom "<orphan>" key; see DRCUtils).
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.core.reward import PotentialReward, RewardState  # noqa: E402
from pcb_world.core.reward_config import YamlRewardConfig  # noqa: E402


def _state(unconnected=0, drc=0, per_net=None):
    # Mirror legacy counts into the error-only fields so pre-severity-split
    # tests still hit the same branch (default drc_penalty_include_warning
    # is False → uses drc_errors_per_net).
    per_net = per_net or {}
    return RewardState(
        unconnected=unconnected,
        drc_violations=drc,
        wirelength=0.0,
        track_count=0,
        drc_violations_per_net=dict(per_net),
        drc_errors=drc,
        drc_errors_per_net=dict(per_net),
    )


def _reward(**overrides):
    kwargs = dict(
        completion_bonus=0.0,
        unconnected_penalty=0.0,
        wirelength_penalty=0.0,
        drc_shape="log_per_net",
        drc_log_scale=1.0,
        drc_log_offset=2.0,
        step_penalty=0.0,
    )
    kwargs.update(overrides)
    return PotentialReward(**kwargs)


# ---------------------------------------------------------------------------
# Shape semantics
# ---------------------------------------------------------------------------


def test_log_per_net_zero_violations_gives_zero_drc_penalty():
    r = _reward()
    assert r.potential(_state()) == pytest.approx(0.0)


def test_log_per_net_matches_formula_single_net():
    # x_i=2, offset=2, scale=1 -> pen = ln(1 + 2/2) = ln(2)
    r = _reward()
    phi = r.potential(_state(drc=2, per_net={"NET1": 2}))
    assert phi == pytest.approx(-math.log(2.0))


def test_log_per_net_additive_over_nets():
    # Two nets each with 1 violation -> 2 * ln(1 + 1/2) = 2 * ln(1.5)
    r = _reward()
    phi = r.potential(_state(drc=2, per_net={"A": 1, "B": 1}))
    assert phi == pytest.approx(-2.0 * math.log(1.5))


def test_log_per_net_sub_additive_vs_linear():
    # Log is concave: 1 net with 4 violations < 4 nets each with 1 violation
    r = _reward()
    concentrated = -r.potential(_state(drc=4, per_net={"A": 4}))
    spread = -r.potential(_state(drc=4, per_net={"A": 1, "B": 1, "C": 1, "D": 1}))
    assert concentrated < spread


def test_log_per_net_orphan_phantom_counted():
    r = _reward()
    phi = r.potential(_state(drc=3, per_net={"<orphan>": 3}))
    # 3 orphan violations grouped under single phantom key -> ln(1 + 3/2)
    assert phi == pytest.approx(-math.log(2.5))


def test_log_per_net_marginal_larger_at_low_count():
    # ΔΦ when x_i goes 1->0 should exceed ΔΦ when x_i goes 20->19 on same net.
    r = _reward()
    phi_1 = r.potential(_state(drc=1, per_net={"A": 1}))
    phi_0 = r.potential(_state(drc=0, per_net={}))
    phi_20 = r.potential(_state(drc=20, per_net={"A": 20}))
    phi_19 = r.potential(_state(drc=19, per_net={"A": 19}))
    late_marginal = phi_0 - phi_1  # positive
    early_marginal = phi_19 - phi_20  # positive
    assert late_marginal > early_marginal


def test_log_per_net_scale_scales_linearly():
    base = _reward(drc_log_scale=1.0)
    scaled = _reward(drc_log_scale=3.0)
    st = _state(drc=5, per_net={"A": 5})
    assert scaled.potential(st) == pytest.approx(3.0 * base.potential(st))


def test_log_per_net_offset_changes_curvature():
    small_o = _reward(drc_log_offset=0.5)
    large_o = _reward(drc_log_offset=10.0)
    st = _state(drc=1, per_net={"A": 1})
    # smaller offset -> steeper near zero -> bigger penalty magnitude at x=1
    assert -small_o.potential(st) > -large_o.potential(st)


def test_log_per_net_invalid_offset_raises():
    with pytest.raises(ValueError, match="drc_log_offset must be positive"):
        _reward(drc_log_offset=0.0)


def test_log_per_net_zero_log_scale_disables_term():
    # With both drc_log_scale and drc_log_agg_scale at 0, Φ_drc == 0
    # regardless of violation count.
    r = _reward(drc_log_scale=0.0, drc_log_agg_scale=0.0)
    phi = r.potential(_state(drc=100, per_net={"A": 100}))
    assert phi == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dynamic completion_bonus (resolved at env init)
# ---------------------------------------------------------------------------


def test_completion_bonus_log_scale_default_none():
    r = PotentialReward()
    assert r.completion_bonus_log_scale is None


def test_completion_bonus_static_by_default_in_saturating():
    cfg = YamlRewardConfig({
        "name": "test",
        "mode": "per_step",
        "potential": {
            "completion_bonus": 2.0,
            "unconnected_penalty": 1.0,
            "drc_shape": "saturating",
            "drc_aggregate_scale": 5.0,
            "drc_per_net_scale": 1.0,
            "drc_saturation_offset": 2.0,
            "step_penalty": 0.01,
        },
    })
    r = cfg.build_reward()
    assert r.completion_bonus_log_scale is None
    assert r.completion_bonus == pytest.approx(2.0)


def test_yaml_config_passes_log_per_net_fields():
    cfg = YamlRewardConfig({
        "name": "test_log",
        "mode": "per_step",
        "potential": {
            "completion_bonus_log_scale": 1.0,
            "unconnected_penalty": 1.0,
            "drc_shape": "log_per_net",
            "drc_log_scale": 1.0,
            "drc_log_offset": 2.0,
            "step_penalty": 0.01,
        },
    })
    r = cfg.build_reward()
    assert r.drc_shape == "log_per_net"
    assert r.drc_log_scale == pytest.approx(1.0)
    assert r.drc_log_offset == pytest.approx(2.0)
    assert r.completion_bonus_log_scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Aggregate-log term (all-log variant)
# ---------------------------------------------------------------------------


def test_log_agg_default_off_matches_pure_per_net():
    # drc_log_agg_scale=0.0 by default -> aggregate term contributes nothing.
    r = _reward()
    st = _state(drc=5, per_net={"A": 3, "B": 2})
    expected = -(math.log(1 + 3 / 2) + math.log(1 + 2 / 2))
    assert r.potential(st) == pytest.approx(expected)


def test_log_agg_adds_breadth_term():
    # With agg_scale=3, shared offset=2: x=2 -> breadth = 3·ln(1 + 2/2) = 3·ln(2)
    r = _reward(drc_log_agg_scale=3.0)
    st = _state(drc=2, per_net={"A": 1, "B": 1})
    expected = -(3.0 * math.log(2.0) + 2.0 * math.log(1.5))
    assert r.potential(st) == pytest.approx(expected)


def test_log_agg_is_discrete_in_x():
    # Same x but different depth distribution -> agg term unchanged.
    r = _reward(drc_log_agg_scale=3.0)
    state_a = _state(drc=4, per_net={"A": 4})  # x=1
    state_b = _state(drc=4, per_net={"A": 1, "B": 1, "C": 1, "D": 1})  # x=4
    # A has bigger per-net log (ln(3)) but smaller agg (ln(1.5))
    # B has smaller per-net logs (4·ln(1.5)) but bigger agg (ln(3))
    agg_a = 3.0 * math.log(1.5)
    agg_b = 3.0 * math.log(3.0)
    assert agg_b > agg_a  # breadth dominates when spread across nets


def test_log_agg_zero_when_no_violations():
    r = _reward(drc_log_agg_scale=5.0)
    assert r.potential(_state()) == pytest.approx(0.0)


def test_log_agg_cliff_strength_scales_with_agg_scale():
    # First violation (x=0 -> x=1) adds agg_scale · ln(1.5) of penalty.
    low = _reward(drc_log_agg_scale=1.0)
    high = _reward(drc_log_agg_scale=5.0)
    st = _state(drc=1, per_net={"A": 1})
    low_breadth = -(low.potential(st) - (-math.log(1.5)))  # strip per-net
    high_breadth = -(high.potential(st) - (-math.log(1.5)))
    assert high_breadth == pytest.approx(5.0 * low_breadth)


def test_yaml_config_shaped_log_all_loads():
    # Smoke: the shipped shaped_log_all.yaml loads and round-trips new fields.
    from pcb_world.core.reward_config import get_reward_config

    cfg = get_reward_config("shaped_log_all")
    r = cfg.build_reward()
    assert r.drc_shape == "log_per_net"
    assert r.drc_log_agg_scale == pytest.approx(3.0)
    assert r.drc_log_offset == pytest.approx(2.0)
    assert r.drc_log_scale == pytest.approx(1.0)
    assert r.completion_bonus_log_scale == pytest.approx(1.0)
