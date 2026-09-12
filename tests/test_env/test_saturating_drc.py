"""Unit tests for saturating DRC potential in PotentialReward.

Saturating aggregate uses x = number of offending nets (len of per-net dict).
Orphan violations (no associated net) are grouped under a single phantom
`"<orphan>"` key whose value is the total orphan violation count.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.engine.containers import RewardSnapshot  # noqa: E402
from pcb_world.engine.drc import DRCUtils  # noqa: E402
from pcb_world.core.reward import PotentialReward, RewardState  # noqa: E402


def _state(unconnected=0, drc=0, per_net=None):
    # Mirror the legacy sum counts into the error-only slots so tests that
    # predate the severity split (treating every violation as penalizable)
    # continue to exercise the same reward path under the new default
    # (drc_penalty_include_warning=False → error-only).
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
        drc_shape="saturating",
        drc_aggregate_scale=5.0,
        drc_per_net_scale=1.0,
        drc_saturation_offset=2.0,
        step_penalty=0.0,
    )
    kwargs.update(overrides)
    return PotentialReward(**kwargs)


# ---------------------------------------------------------------------------
# Saturating potential tests
# ---------------------------------------------------------------------------


def test_saturating_zero_violations_gives_zero_drc_penalty():
    r = _reward()
    assert r.potential(_state()) == pytest.approx(0.0)


def test_saturating_single_offending_net_single_violation():
    # x (offending nets) = 1, x_i = 1: Φ_drc = -(5·1/3 + 1/3) = -2.0
    r = _reward()
    phi = r.potential(_state(drc=1, per_net={"NET1": 1}))
    assert phi == pytest.approx(-(5 / 3 + 1 / 3))


def test_saturating_concentrated_on_one_net():
    # x = 1 (one offending net), x_i = 10 many violations on single net
    # Φ_drc = -(5·1/3 + 10/12) = -5/3 - 10/12
    r = _reward()
    phi = r.potential(_state(drc=10, per_net={"NET1": 10}))
    expected = -(5 * 1 / 3 + 10 / 12)
    assert phi == pytest.approx(expected)


def test_saturating_spread_across_nets():
    # x = 5 offending nets, x_i = 2 each
    # Φ_drc = -(5·5/7 + 5·(2/4)) = -25/7 - 2.5
    r = _reward()
    per_net = {f"NET{i}": 2 for i in range(5)}
    phi = r.potential(_state(drc=10, per_net=per_net))
    expected = -(5 * 5 / 7 + 5 * (2 / 4))
    assert phi == pytest.approx(expected)


def test_saturating_aggregate_bounded_by_net_count():
    # Even with huge per-net counts, aggregate x is just number of offending
    # nets — so penalty is bounded by drc_aggregate_scale + N_nets·per_net_scale.
    r = _reward()
    per_net = {f"NET{i}": 10_000 for i in range(3)}
    phi = r.potential(_state(drc=30_000, per_net=per_net))
    # x=3 offending nets → aggregate = -5·3/5 = -3.0
    # per-net each ~ -1.0 (saturated) → -3.0
    # total ≈ -6.0, never diverges
    assert phi == pytest.approx(-(5 * 3 / 5 + 3 * 10_000 / 10_002), rel=1e-6)
    assert phi > -7.0


def test_saturating_upper_bound_single_net():
    # x=1 net, x_i→∞: Φ_drc → -(5·1/3 + 1) = -8/3 ≈ -2.667
    r = _reward()
    phi = r.potential(_state(drc=10_000, per_net={"NET1": 10_000}))
    assert phi < -(5 / 3 + 0.99)
    assert phi > -(5 / 3 + 1.0)


def test_saturating_uses_len_per_net_not_drc_violations():
    """Aggregate x must come from len(per_net), not the scalar drc_violations field."""
    r = _reward()
    # drc_violations field says 100, but per_net has only 2 offending nets.
    # Saturating should use 2 (not 100) for aggregate.
    phi = r.potential(_state(drc=100, per_net={"A": 50, "B": 50}))
    expected = -(5 * 2 / 4 + 50 / 52 + 50 / 52)
    assert phi == pytest.approx(expected)


def test_saturating_orphan_phantom_is_single_net():
    # All orphan violations are grouped under one "<orphan>" phantom net.
    r = _reward()
    # 1 real offending net + 1 orphan phantom (with 2 orphan violations) → x=2.
    phi = r.potential(_state(
        drc=5,  # irrelevant for saturating
        per_net={"NET1": 3, "<orphan>": 2},
    ))
    expected = -(5 * 2 / 4 + 3 / 5 + 2 / 4)
    assert phi == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Linear shape backward compatibility
# ---------------------------------------------------------------------------


def test_linear_shape_still_default():
    r = PotentialReward(
        completion_bonus=0.0,
        unconnected_penalty=0.0,
        wirelength_penalty=0.0,
        drc_penalty=0.5,
        step_penalty=0.0,
    )
    assert r.drc_shape == "linear"
    phi = r.potential(_state(drc=4))
    assert phi == pytest.approx(-2.0)


def test_linear_shape_ignores_per_net():
    """Linear shape uses total drc count only, not per-net dict."""
    r = PotentialReward(
        completion_bonus=0.0, unconnected_penalty=0.0, wirelength_penalty=0.0,
        drc_penalty=0.5, step_penalty=0.0, drc_shape="linear",
    )
    phi_a = r.potential(_state(drc=4, per_net={"A": 4}))
    phi_b = r.potential(_state(drc=4, per_net={"A": 1, "B": 1, "C": 1, "D": 1}))
    assert phi_a == phi_b == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_saturating_invalid_shape_raises():
    with pytest.raises(ValueError):
        PotentialReward(drc_shape="nonsense")


def test_saturating_invalid_offset_raises():
    with pytest.raises(ValueError):
        PotentialReward(drc_shape="saturating", drc_saturation_offset=0.0)


# ---------------------------------------------------------------------------
# Dense / truncation integration
# ---------------------------------------------------------------------------


def test_dense_step_uses_saturating_shape():
    r = _reward()
    before = _state(unconnected=1, drc=2, per_net={"A": 2})
    after = _state(unconnected=0, drc=0, per_net={})
    expected = r.potential(after) - r.potential(before)
    assert r.compute_dense(before, after) == pytest.approx(expected)


def test_truncation_penalty_only_uses_saturating():
    r = _reward(truncation_mode="penalty_only", unconnected_penalty=1.0)
    state = _state(unconnected=3, drc=5, per_net={"A": 3, "B": 2})
    # x = 2 offending nets; Φ_drc = -(5·2/4 + 3/5 + 2/4) = -2.5 - 0.6 - 0.5
    expected = -3.0 - (5 * 2 / 4 + 3 / 5 + 2 / 4)
    assert r.compute_truncation(state) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RewardState plumbing
# ---------------------------------------------------------------------------


def test_rewardstate_from_snapshot_includes_per_net():
    snap = RewardSnapshot(
        unrouted_count=1,
        track_count=2,
        total_wirelength=3.0,
        drc_violation_count=4,
        drc_violations_per_net={"A": 3, "B": 1},
    )
    state = RewardState.from_snapshot(snap)
    assert state.drc_violations == 4
    assert state.drc_violations_per_net == {"A": 3, "B": 1}
    # Ensure it's a copy, not a reference.
    state.drc_violations_per_net["C"] = 99
    assert "C" not in snap.drc_violations_per_net


# ---------------------------------------------------------------------------
# DRCUtils orphan handling
# ---------------------------------------------------------------------------


class _FakeViolation:
    def __init__(self, net_names, error_type="clearance"):
        self.net_names = list(net_names)
        self.error_type = error_type
        self.error_code = 0
        self.message = ""
        self.x_mm = 0.0
        self.y_mm = 0.0
        self.layer = 0
        self.severity = 0x20


def test_drchelper_groups_orphans_into_single_phantom_net():
    helper = DRCUtils()
    helper.update([
        _FakeViolation(["NET1"]),
        _FakeViolation(["NET1", "NET2"]),  # multi-net: each gets +1
        _FakeViolation([]),                 # orphan
        _FakeViolation([]),                 # orphan
    ])
    counts = helper.get_violation_counts_by_net()
    assert counts == {
        "NET1": 2,
        "NET2": 1,
        "<orphan>": 2,
    }


def test_drchelper_all_orphans_single_phantom():
    helper = DRCUtils()
    helper.update([_FakeViolation([]) for _ in range(3)])
    counts = helper.get_violation_counts_by_net()
    assert counts == {"<orphan>": 3}
    assert len(counts) == 1  # aggregate x = 1 offending "net"
