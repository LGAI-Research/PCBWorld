"""``PCBWorld.step`` enforces the action constraints (pcb_world.core.constraints) for every
caller, as an RL policy meets them through its masks (requires C++ kicad_rl_router):
  - under ``free`` (the LLM env) a closed net is not selectable again (before: the env took it);
  - under ``candidates`` (the RL env) a target that is not a pool point is refused;
  - an RL policy that samples only unmasked pointers is never refused by a constraint;
  - an MCTS checkpoint carries the same-point state.
"""
import os

import numpy as np
import pytest

from configs.loader.schema import EnvConfig, RLEnvConfig
from pcb_world.core.action_schema import (ACT_MAKE_LINE, ACT_NET_END, ACT_NET_SELECT,
                                          ACT_START_ROUTE)
from pcb_world.engine import engine_available

BOARD = os.path.join(os.path.dirname(__file__), "..", "fixtures", "simple_routing_board.kicad_pcb")

pytestmark = pytest.mark.skipif(not engine_available(), reason="kicad_rl_router not available")


def _env(**kw):
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(BOARD, EnvConfig.with_overrides(max_steps=50, **kw))
    env.reset(seed=0)
    return env


def _first_open_net(env, obs):
    return next(int(k.split("_")[1]) for k, v in obs["routing_geometry"].items() if v.get("points"))


def test_free_env_refuses_reselecting_a_closed_net():
    env = _env(action_target="free")
    try:
        obs = env._last_obs
        net = _first_open_net(env, obs)
        env.step({"action_type": ACT_NET_SELECT, "net_id": net})
        obs, *_ = env.step({"action_type": ACT_NET_END})          # pcbworld_masking: give up
        assert net in obs["closed_nets"]
        _, _, _, _, info = env.step({"action_type": ACT_NET_SELECT, "net_id": net})
        assert (info["action_class"], info.get("constraint")) == ("mask_reject", "net_closed")
        assert env._dispatcher.current_net_id is None
    finally:
        env.close()


def test_candidates_env_refuses_a_point_off_the_pool():
    env = _env(action_target="candidates")
    try:
        net = _first_open_net(env, env._last_obs)
        env.step({"action_type": ACT_NET_SELECT, "net_id": net})
        x, y, layer = env.act_constraints().cand_mm[0]
        env.step({"action_type": ACT_START_ROUTE, "x_mm": x, "y_mm": y, "layer": layer})
        _, _, _, _, info = env.step({"action_type": ACT_MAKE_LINE, "x_mm": x + 0.123,
                                     "y_mm": y + 0.456, "routing_mode": 2})
        assert (info["action_class"], info.get("constraint")) == ("mask_reject", "not_a_candidate")
    finally:
        env.close()


def test_masked_rl_policy_is_never_refused():
    """Random actions drawn only from what the masks allow (as a policy samples) all pass the
    env's check — the env and the RL masks are one rule set."""
    from methods.rl_agent.wrappers.factory import make_decoder_env

    cfg = RLEnvConfig().with_flat({"max_steps": 80})
    env = make_decoder_env(BOARD, cfg, seed=3, policy_net_select=True)
    rng = np.random.default_rng(11)
    env.reset()
    refused = []
    try:
        for _ in range(200):
            m = env.mask_payload()
            at = int(rng.choice(np.flatnonzero(m["action"][:6])))
            if at == ACT_NET_SELECT:
                ptr = int(rng.choice(np.flatnonzero(m["net_valid"])))
            else:
                blocked = set(m["pointer"].tolist()) | set(m["offboard"].tolist())
                if at == ACT_MAKE_LINE:
                    blocked |= set(m["offlayer"].tolist())
                ok = [i for i in range(len(env._cand_mm)) if i not in blocked]
                ptr = int(rng.choice(ok)) if ok else 0
            _, _, term, trunc, info = env.step(np.array([at, ptr, 2], dtype=np.int64))
            if "constraint" in info:
                refused.append((at, ptr, info["constraint"]))
            if term or trunc:
                env.reset()
    finally:
        env.close()
    assert not refused


def test_checkpoint_carries_the_start_point():
    env = _env(action_target="free")
    try:
        net = _first_open_net(env, env._last_obs)
        env.step({"action_type": ACT_NET_SELECT, "net_id": net})
        x, y, layer = env.act_constraints().cand_mm[0]
        env.step({"action_type": ACT_START_ROUTE, "x_mm": x, "y_mm": y, "layer": layer})
        ckpt = env.checkpoint()
        assert env._start_route_xy == (x, y, layer)
        env.step({"action_type": ACT_NET_END})   # not legal while routing: still clears the state
        env._start_route_xy = None
        env.restore(ckpt)
        assert env._start_route_xy == (x, y, layer)
        env.release_checkpoint(ckpt)
    finally:
        env.close()


def test_a_refused_start_route_keeps_the_start_point():
    """A start_route the env refuses (not executed) does not move the same-point state."""
    env = _env(action_target="candidates")
    try:
        net = _first_open_net(env, env._last_obs)
        env.step({"action_type": ACT_NET_SELECT, "net_id": net})
        x, y, layer = env.act_constraints().cand_mm[0]
        env.step({"action_type": ACT_START_ROUTE, "x_mm": x, "y_mm": y, "layer": layer})
        assert env._start_route_xy == (x, y, layer)
        _, _, _, _, info = env.step({"action_type": ACT_START_ROUTE, "x_mm": x + 0.123,
                                     "y_mm": y + 0.456, "layer": layer})
        assert info["action_class"] == "mask_reject"
        assert env._start_route_xy == (x, y, layer)
    finally:
        env.close()


def test_reset_best_checkpoint_holds_no_previous_episode_obs():
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(BOARD, EnvConfig.with_overrides(max_steps=50), output_best_board=True)
    try:
        env.reset(seed=0)
        env.step({"action_type": ACT_NET_SELECT, "net_id": _first_open_net(env, env._last_obs)})
        stale = env._last_obs
        env.act_constraints()
        env.reset(seed=0)
        assert env._best_ckpt.last_obs is not stale
        assert env._best_ckpt.act is None or env._best_ckpt.last_obs is env._last_obs
    finally:
        env.close()


def test_checkpoint_without_obs_cache_rebuilds_on_restore():
    """``checkpoint(obs_cache=False)`` (MCTS_OBS_CACHE=0) keeps no obs; restore rebuilds it."""
    env = _env(action_target="candidates")
    try:
        env.step({"action_type": ACT_NET_SELECT, "net_id": _first_open_net(env, env._last_obs)})
        want = env.act_constraints().cand_mm
        ckpt = env.checkpoint(obs_cache=False)
        assert ckpt.last_obs is None and ckpt.act is None
        env.step({"action_type": ACT_NET_END})
        env.restore(ckpt)
        assert env.act_constraints().cand_mm == want
        env.release_checkpoint(ckpt)
    finally:
        env.close()
