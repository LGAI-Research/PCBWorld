"""The action constraints every agent meets (pcb_world.core.constraints), as pure rules over an
observation: the same-point state, the forced routing mode and the per-action check under both
``action_target`` values. The env applies them in ``PCBWorld.step`` (tests/test_env/test_env_constraints.py)."""
from __future__ import annotations

import numpy as np
import pytest

from pcb_world.core import constraints as C
from pcb_world.core.action_schema import (ACT_FINISH, ACT_IDLE, ACT_MAKE_LINE, ACT_MAKE_VIA,
                                          ACT_NET_END, ACT_NET_SELECT, ACT_START_ROUTE)
from pcb_world.vec.candidate_pool import CTYPE_DIRECTIONAL
from tests._mock_obs import _make_pad, make_mock_obs

START = (110.0, 60.0, 1)


def _routing_obs():
    """Net 1 selected and routing, head at (130, 70) on layer 1, board bbox (100, 50, 60, 40);
    one same-net pad off the board (real geometry, never off-board-masked)."""
    obs = make_mock_obs(n_nets=1, pads_per_net=2, n_ratsnest_per_net=1, is_routing=True,
                        current_net_phase=2)
    obs["router_head"].update(current_xy=[130.0, 70.0], current_layer=1, current_net=1)
    obs["board_static"]["nets"]["net_1"]["pads"]["pad_out"] = _make_pad(95.0, 45.0)
    obs["closed_nets"] = [7]
    return obs


def _act(obs, *, start=None, offboard=True):
    return C.act_constraints(
        obs, action_mask=np.ones(7, dtype=bool),
        cand_aug={"directional_candidates": "multi_resolution", "connectivity_filter": False,
                  "route_start_xy": start, "cluster_keys": None, "pad_graze_margin_mm": 0.0},
        start_route_xy=start, mask_start_point=True, offboard_mask=offboard,
        force_walkaround=False, net_reselect=False)


def _check(action, obs, *, target, start=None, act=None, offboard=True, reselect=False):
    return C.violation(action, action_target=target, obs=obs, act=act, start_route_xy=start,
                       mask_start_point=True, offboard_mask=offboard, net_reselect=reselect)


@pytest.mark.parametrize("action,want", [
    ({"action_type": ACT_START_ROUTE, "x_mm": 1.0, "y_mm": 2.0, "layer": 2}, (1.0, 2.0, 2)),
    ({"action_type": ACT_MAKE_LINE, "x_mm": 5.0, "y_mm": 5.0, "routing_mode": 2}, START),
    ({"action_type": ACT_MAKE_VIA, "x_mm": 5.0, "y_mm": 5.0, "routing_mode": 2}, START),
    ({"action_type": ACT_FINISH, "routing_mode": 2}, START),
    ({"action_type": ACT_IDLE}, START),
    ({"action_type": ACT_NET_END}, None),
    ({"action_type": ACT_NET_SELECT, "net_id": 1}, None),
])
def test_start_point_rule(action, want):
    assert C.next_start_route_xy(START, action) == want


def test_forced_routing_mode():
    assert C.forced_routing_mode(False) is None
    assert C.forced_routing_mode(True) == 2


@pytest.mark.parametrize("target", C.ACTION_TARGETS)
def test_a_closed_net_is_not_selectable_unless_reselect(target):
    obs = _routing_obs()
    sel = {"action_type": ACT_NET_SELECT, "net_id": 7}
    act = _act(obs) if target == "candidates" else None
    assert _check(sel, obs, target=target, act=act) == "net_closed"
    assert _check(sel, obs, target=target, act=act, reselect=True) is None
    assert _check({**sel, "net_id": 1}, obs, target=target, act=act) is None


def test_free_targets_meet_the_rules_that_need_no_pool():
    obs = _routing_obs()
    line = {"action_type": ACT_MAKE_LINE, "routing_mode": 2}
    assert _check({**line, "x_mm": 123.4, "y_mm": 77.7}, obs, target="free") is None   # any point
    assert _check({**line, "x_mm": 170.0, "y_mm": 70.0}, obs, target="free") == "off_board"
    assert _check({**line, "x_mm": 170.0, "y_mm": 70.0}, obs, target="free", offboard=False) is None
    start = (130.0, 70.0, 1)
    assert _check({**line, "x_mm": 130.0, "y_mm": 70.0}, obs, target="free", start=start) == "same_point"
    sr = {"action_type": ACT_START_ROUTE, "x_mm": 130.0, "y_mm": 70.0}
    assert _check({**sr, "layer": 1}, obs, target="free", start=start) == "same_point"
    assert _check({**sr, "layer": 2}, obs, target="free", start=start) is None   # sibling layer


def test_candidate_targets_must_be_unblocked_pool_points():
    obs = _routing_obs()
    act = _act(obs)
    line = {"action_type": ACT_MAKE_LINE, "routing_mode": 2}
    assert _check({**line, "x_mm": 123.4, "y_mm": 77.7}, obs, target="candidates", act=act) \
        == "not_a_candidate"
    hx, hy = 130.0, 70.0
    i_in = next(i for i, ((x, y, _l), ct) in enumerate(zip(act.cand_mm, act.cand_ctype))
                if ct == CTYPE_DIRECTIONAL and (x, y) == (hx + 1.0, hy))
    x, y, _ = act.cand_mm[i_in]
    assert _check({**line, "x_mm": x, "y_mm": y}, obs, target="candidates", act=act) is None
    # the 25 mm north / south rungs leave the board (y 45 / 95) -> off-board-masked on every row
    off = [(x, y) for (x, y, _l), ct in zip(act.cand_mm, act.cand_ctype)
           if ct == CTYPE_DIRECTIONAL and not 50.0 <= y <= 90.0]
    assert off
    assert _check({**line, "x_mm": off[0][0], "y_mm": off[0][1]}, obs, target="candidates",
                  act=act) == "off_board"
    # a pad outside the bbox is real geometry: not off-board-masked
    assert _check({"action_type": ACT_START_ROUTE, "x_mm": 95.0, "y_mm": 45.0, "layer": 1}, obs,
                  target="candidates", act=act) is None


def test_candidate_start_point_and_off_layer():
    obs = _routing_obs()
    pad = obs["board_static"]["nets"]["net_1"]["pads"]["pad_0"]["center"]["xy"]
    act = _act(obs, start=(pad[0], pad[1], 1))
    sr = {"action_type": ACT_START_ROUTE, "x_mm": pad[0], "y_mm": pad[1], "layer": 1}
    assert _check(sr, obs, target="candidates", act=act, start=(pad[0], pad[1], 1)) == "same_point"
    # make_line onto a layer-2-only candidate: off_layer (the head is on layer 1)
    obs2 = _routing_obs()
    obs2["board_static"]["nets"]["net_1"]["pads"]["pad_b"] = _make_pad(140.0, 80.0, layer=2)
    act2 = _act(obs2)
    line = {"action_type": ACT_MAKE_LINE, "x_mm": 140.0, "y_mm": 80.0, "routing_mode": 2}
    assert _check(line, obs2, target="candidates", act=act2) == "off_layer"
    assert _check({**line, "action_type": ACT_MAKE_VIA}, obs2, target="candidates", act=act2) is None


def test_unknown_target_and_missing_pool_raise():
    obs = _routing_obs()
    line = {"action_type": ACT_MAKE_LINE, "x_mm": 1.0, "y_mm": 1.0, "routing_mode": 2}
    with pytest.raises(ValueError, match="action_target"):
        _check(line, obs, target="grid")
    with pytest.raises(ValueError, match="ActConstraints"):
        _check(line, obs, target="candidates", act=None)


def test_free_same_point_uses_the_rl_mask_tolerance():
    """Free mode refuses exactly the points the RL same-point mask blocks (one 0.01 mm
    tolerance): a start point 5 µm off is the same point, 20 µm off is not."""
    obs = _routing_obs()
    start = (130.0, 70.0, 1)
    sr = {"action_type": ACT_START_ROUTE, "layer": 1}
    near, far = (130.005, 70.0), (130.02, 70.0)
    for (x, y), want in ((near, "same_point"), (far, None)):
        assert _check({**sr, "x_mm": x, "y_mm": y}, obs, target="free", start=start) == want
        rl = C.start_route_pointer_indices(mask_start_point=True, start_route_xy=start,
                                           cand_mm=[(x, y, 1)])
        assert (rl.size == 1) == (want == "same_point")
