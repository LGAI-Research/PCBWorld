"""Under ``action_target: candidates`` the LLM sees the RL target set: the ``(candidates ...)``
block of its text obs lists exactly the pool points the RL masks leave open for each
coordinate action the action mask allows, and the env accepts every listed point."""
from __future__ import annotations

import os
import re

import numpy as np
import pytest

from configs.loader.schema import EnvConfig
from pcb_world.core import constraints as C
from pcb_world.core.action_schema import (ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_NET_SELECT,
                                          ACT_START_ROUTE, ACTION_NAMES)
from pcb_world.engine import engine_available
from tests._mock_obs import make_mock_obs

BOARD = os.path.join(os.path.dirname(__file__), "fixtures", "simple_routing_board.kicad_pcb")
COORD_ACTIONS = (ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA)
needs_engine = pytest.mark.skipif(not engine_available(), reason="kicad_rl_router not available")


def _mock_act():
    """Routing on net 1 from one of its pads (same-point active), a directional ladder
    that leaves the board (off-board), and pads on both layers (off-layer)."""
    obs = make_mock_obs(n_nets=1, pads_per_net=3, n_ratsnest_per_net=1, is_routing=True,
                        current_net_phase=2)
    obs["router_head"].update(current_xy=[130.0, 70.0], current_layer=1, current_net=1)
    pads = obs["board_static"]["nets"]["net_1"]["pads"]
    pads["pad_1"]["layer"] = 2
    pads["pad_2"]["center"]["xy"] = [145.9611, 71.3536]          # finer than 3 decimals
    x, y = pads["pad_0"]["center"]["xy"]
    start = (float(x), float(y), 1)
    act = C.act_constraints(
        obs, action_mask=np.ones(7, dtype=bool),
        cand_aug={"directional_candidates": "multi_resolution", "connectivity_filter": False,
                  "route_start_xy": start, "cluster_keys": None, "pad_graze_margin_mm": 0.0},
        start_route_xy=start, mask_start_point=True, offboard_mask=True,
        force_walkaround=False, net_reselect=False)
    return obs, act, start


@pytest.mark.parametrize("at", COORD_ACTIONS)
def test_allowed_candidates_are_what_violation_accepts(at):
    obs, act, start = _mock_act()
    allowed = C.allowed_candidates(act, at)
    blocked = set().union(*C.blocked_candidates(act, at).values())
    assert blocked and allowed                                   # the fixture exercises both
    assert set(allowed) | blocked == set(range(len(act.cand_raw)))
    assert not set(allowed) & blocked
    allowed_xy = {act.cand_mm[i][:2] for i in allowed}
    for i, (x, y, layer) in enumerate(act.cand_mm):
        got = C.violation({"action_type": at, "x_mm": x, "y_mm": y, "layer": layer,
                           "routing_mode": 2},
                          action_target="candidates", obs=obs, act=act, start_route_xy=start,
                          mask_start_point=True, offboard_mask=True, net_reselect=False)
        # make_line / make_via ignore the named layer, so a blocked layer variant of an
        # allowed point is accepted as that point.
        accepted = i in allowed or (at != ACT_START_ROUTE and (x, y) in allowed_xy)
        assert (got is None) == accepted, (i, got)


def test_rendered_candidates_are_accepted_as_written():
    """Each listed target, parsed back from the text, passes the env's check — also a
    coordinate 3 decimals would round (the env matches candidates exactly)."""
    from methods.llm_agent.wrappers.adapter import candidate_rows
    from methods.llm_agent.wrappers.state_converter import format_candidates

    obs, act, start = _mock_act()
    rendered = _rendered(format_candidates(candidate_rows(act)))
    assert set(rendered) == {"start_route", "make_line", "make_via"}
    assert (145.9611, 71.3536, 1) in rendered["start_route"]
    for name, targets in rendered.items():
        for x, y, layer in targets:
            assert C.violation({"action_type": ACTION_NAMES.index(name), "x_mm": x, "y_mm": y,
                                "layer": layer, "routing_mode": 2},
                               action_target="candidates", obs=obs, act=act, start_route_xy=start,
                               mask_start_point=True, offboard_mask=True,
                               net_reselect=False) is None, (name, x, y)


def _rendered(text: str) -> dict[str, set]:
    """The ``(candidates ...)`` block of ``text`` → {action: {(x, y, layer), ...}}."""
    block = text[text.index("(candidates"):]
    out, cur = {}, None
    for line in block.splitlines()[1:]:
        if m := re.fullmatch(r"\s*\((\w+)\)?", line):
            cur = m.group(1)
            out[cur] = set()
        elif m := re.fullmatch(r"\s*\(c (\S+) (\S+) (\d+) \w+\)", line):
            out[cur].add((round(float(m.group(1)), 6), round(float(m.group(2)), 6), int(m.group(3))))
    return out


def _expected(env) -> dict[str, set]:
    """From the masks alone: per allowed coordinate action, the unblocked pool points."""
    act = env.act_constraints()
    m = act.masks
    out = {}
    for at in COORD_ACTIONS:
        if not m["action"][at]:
            continue
        blocked = set(m["pointer"].tolist()) | set(m["offboard"].tolist())
        if at == ACT_MAKE_LINE:
            blocked |= set(m["offlayer"].tolist())
        out[ACTION_NAMES[at]] = {(round(x, 6), round(y, 6), layer)
                                 for i, (x, y, layer) in enumerate(act.cand_mm) if i not in blocked}
    return out


def _worker(target):
    from methods.llm_agent.wrappers.adapter import KiCadLLMWrapper

    return KiCadLLMWrapper(board_path=BOARD, seed=0,
                           env_config=EnvConfig.with_overrides(max_steps=50, action_target=target),
                           state_format="sexpr", prompt_version="latest")


@needs_engine
def test_rendered_candidates_are_the_rl_targets_and_are_accepted():
    worker = _worker("candidates")
    try:
        text, _ = worker.reset()
        assert _rendered(text) == _expected(worker.env) == {}      # net_select only
        net = next(int(k.split("_")[1]) for k, v in worker.get_last_obs_dict()["routing_geometry"].items()
                   if v.get("points"))
        text, _, _, info = worker.step({"action_type": ACT_NET_SELECT, "net_id": net})
        stepped = []
        for _ in range(4):
            got = _rendered(text)
            assert got == _expected(worker.env)
            name = next(n for n in ("make_line", "make_via", "start_route") if got.get(n))
            x, y, layer = sorted(got[name])[0]
            text, _, _, info = worker.step({"action_type": ACTION_NAMES.index(name), "x_mm": x,
                                            "y_mm": y, "layer": layer, "routing_mode": 2})
            assert info.get("constraint") is None and info["action_class"] != "mask_reject", info
            stepped.append(name)
        assert {"start_route", "make_line"} <= set(stepped), stepped
    finally:
        worker.env.close()


@needs_engine
@pytest.mark.parametrize("target", ["candidates", "free"])
def test_only_a_candidates_env_shows_the_block(target):
    worker = _worker(target)
    try:
        text, _ = worker.reset()
        net = next(int(k.split("_")[1]) for k, v in worker.get_last_obs_dict()["routing_geometry"].items()
                   if v.get("points"))
        text, *_ = worker.step({"action_type": ACT_NET_SELECT, "net_id": net})
        assert ("(candidates" in text) == (target == "candidates")
    finally:
        worker.env.close()


@pytest.mark.parametrize("x", [100 / 3, 100 / 30 * 7, 12.0000004, 145.9611, 25.0, -0.1 / 3])
def test_candidate_coordinates_print_back_to_the_env_float(x):
    """A listed coordinate parses back to the exact float the env matches (grid spacings
    like 100/N repeat; KiCad nm values need more than 3 decimals), with no trailing dot, and
    the guided-decoding grammar can emit it."""
    from methods.llm_agent.wrappers.grammar import _FLOAT
    from methods.llm_agent.wrappers.state_converter import format_candidates
    from pcb_world.vec.candidate_pool import CTYPE_PAD

    line = format_candidates({"make_line": [(x, x, 1, CTYPE_PAD)]}).splitlines()[2]
    sx, sy = re.fullmatch(r"\s*\(c (\S+) (\S+) 1 pad\)", line).groups()
    assert sx == sy and abs(float(sx) - x) < 1e-9, (sx, x)      # the env's match tolerance
    assert not sx.endswith(".") and re.fullmatch(_FLOAT, sx), sx


def test_candidates_mode_header_does_not_claim_every_value_has_3_decimals():
    from configs.loader.schema import LLMConfig
    from methods.llm_agent.wrappers.prompts import get_pcbworld_prompts

    def header(target):
        env = LLMConfig().env.with_flat({"action_target": target})
        return get_pcbworld_prompts("latest", env_config=env).get_state_format_desc(
            "sexpr").splitlines()[0]

    assert "All floating-point values are rounded to 3 decimal places." in header("free")
    assert "candidates" in header("candidates") and header("candidates") != header("free")
