"""W7a linear-ladder knobs: pad-group-proportional net weights + board bonuses.

The log-shaped twins (``net_bonus_size_log_scale`` /
``completion_bonus_log_scale``) are covered by ``test_ladder_reward.py`` and
``test_reward_parity.py``; this file pins the LINEAR shape added for W7a and
the mutual-exclusion guards, so a future refactor cannot silently collapse the
two shapes into one.
"""
import math

import pytest

from pcb_world.core.reward import PotentialReward

PAD_GROUPS = {1: 2, 2: 3, 3: 5}
NET_NAMES = {1: "a", 2: "b", 3: "c"}
ROUTABLE = frozenset(PAD_GROUPS)
TOTAL_PG = sum(PAD_GROUPS.values())  # 10


def _bind(**kwargs) -> PotentialReward:
    pr = PotentialReward(**kwargs)
    pr.bind_board(net_count=len(PAD_GROUPS), bbox_w=40.0, bbox_h=20.0)
    pr.bind_board(
        pad_groups=PAD_GROUPS, net_names=NET_NAMES, routable_nets=ROUTABLE
    )
    return pr


def test_linear_net_weights_are_proportional_to_pad_groups():
    pr = _bind(net_bonus_size_linear_scale=0.5)
    assert pr.net_size_weights_by_code == {c: 0.5 * pg for c, pg in PAD_GROUPS.items()}
    assert pr.net_size_weights_by_name == {
        NET_NAMES[c]: 0.5 * pg for c, pg in PAD_GROUPS.items()
    }


def test_log_shape_still_compresses():
    """The two shapes must stay distinguishable (linear > log for pg >= 2)."""
    lin = _bind(net_bonus_size_linear_scale=1.0).net_size_weights_by_code
    log = _bind(net_bonus_size_log_scale=1.0).net_size_weights_by_code
    assert log == {c: math.log1p(pg) for c, pg in PAD_GROUPS.items()}
    assert all(lin[c] > log[c] for c in PAD_GROUPS)


def test_board_bonuses_scale_with_total_pad_groups():
    pr = _bind(
        completion_bonus_padgroup_scale=0.5,
        clean_completion_bonus_padgroup_scale=0.25,
    )
    assert pr.completion_bonus == 0.5 * TOTAL_PG
    assert pr.clean_completion_bonus == 0.25 * TOTAL_PG


def test_board_bonus_rebinds_per_reset():
    """Keep-routing makes pad groups episode-dependent — a rebind must move
    the board bonus, not keep the first episode's value."""
    pr = _bind(completion_bonus_padgroup_scale=1.0)
    assert pr.completion_bonus == TOTAL_PG
    smaller = {1: 2, 2: 2, 3: 2}
    pr.bind_board(
        pad_groups=smaller, net_names=NET_NAMES, routable_nets=ROUTABLE
    )
    assert pr.completion_bonus == 6


def test_unresolved_pad_groups_fail_loudly():
    pr = PotentialReward(net_bonus_size_linear_scale=0.5)
    with pytest.raises(RuntimeError, match="no pad groups"):
        pr.bind_board(
            pad_groups={1: 2}, net_names=NET_NAMES, routable_nets=ROUTABLE
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(net_bonus_size_log_scale=1.0, net_bonus_size_linear_scale=0.5),
        dict(completion_bonus_log_scale=2.0, completion_bonus_padgroup_scale=0.5),
        dict(
            clean_completion_bonus_log_scale=2.0,
            clean_completion_bonus_padgroup_scale=0.5,
        ),
    ],
)
def test_shapes_are_mutually_exclusive(kwargs):
    with pytest.raises(ValueError, match="mutually exclusive"):
        PotentialReward(**kwargs)


def test_defaults_are_unchanged_without_the_knobs():
    """A config that never sets the linear knobs must behave exactly as before
    (the W-series campaigns depend on this)."""
    pr = _bind(net_bonus_size_log_scale=1.0, completion_bonus_log_scale=2.0)
    assert pr.completion_bonus == pytest.approx(2.0 * math.log1p(len(PAD_GROUPS)))
    assert pr.net_size_weights_by_code == {
        c: math.log1p(pg) for c, pg in PAD_GROUPS.items()
    }
