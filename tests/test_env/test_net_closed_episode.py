"""net_end consumes its net for the episode (requires C++ kicad_rl_router).

Contract:
  - A successful ``net_end`` retires the current net for the rest of the
    episode: it lands in ``obs["closed_nets"]`` and drops out of the
    ``net_valid_mask`` ACT_NET_SELECT candidates.
  - When EVERY routable net (>=2 pads) has been opened and closed once, the
    episode ends with ``terminated=True`` (voluntary finish — a true terminal,
    not a truncation, so no GAE bootstrap) even when nets remain unrouted.
  - Under the default masking rule this changes nothing: ``net_end`` is only
    exposed after full connection, so ``unconnected == 0`` always fires first.

The give-up loop this kills (permissive rules): select -> net_end -> reselect
the same net forever (observed as a net-free collapse in an early run).
"""

import os

import pytest

from pcb_world.engine import engine_available

from pcb_world.core.action_schema import ACT_NET_END, ACT_NET_SELECT

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "simple_routing_board.kicad_pcb"
)

NETFREE_RULE = """\
name: netfree_test
actions:
  net_select:
    when: {has_net: false, is_routing: false}
  start_route:
    when: {has_net: true, is_routing: false}
  net_end:
    when: {has_net: true, is_routing: false}
  make_line:
    when: {is_routing: true}
  make_via:
    when: {is_routing: true}
  finish:
    when: {is_routing: true}
"""


def _skip_if_unavailable():
    if not os.path.exists(BOARD_PATH):
        pytest.skip(f"Board not found: {BOARD_PATH}")
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


def _select(env, net_id: int):
    return env.step({
        "action_type": ACT_NET_SELECT,
        "x_mm": 0.0, "y_mm": 0.0, "layer": 1,
        "net_id": int(net_id), "routing_mode": 2,
    })


def _net_end(env):
    return env.step({
        "action_type": ACT_NET_END,
        "x_mm": 0.0, "y_mm": 0.0, "layer": 1,
        "net_id": 0, "routing_mode": 2,
    })


@pytest.fixture
def netfree_env(tmp_path):
    _skip_if_unavailable()
    from pcb_world.core.env import PCBWorld

    rule = tmp_path / "netfree_test.yaml"
    rule.write_text(NETFREE_RULE)
    e = PCBWorld(board_path=BOARD_PATH, max_steps=200, masking_rule=str(rule))
    yield e
    e.close()


@pytest.fixture
def default_env():
    _skip_if_unavailable()
    from pcb_world.core.env import PCBWorld

    e = PCBWorld(board_path=BOARD_PATH, max_steps=50)
    yield e
    e.close()


class TestNetfreeGiveUpEpisode:
    def test_close_all_nets_terminates_with_unrouted(self, netfree_env):
        env = netfree_env
        obs, _ = env.reset()
        assert obs["closed_nets"] == []
        routable = sorted(env._routable_nets)
        assert routable, "fixture board must have routable nets"

        for i, nid in enumerate(routable):
            obs, _, terminated, truncated, info = _select(env, nid)
            assert info["action_success"], f"net_select({nid}) failed"
            assert not terminated
            obs, _, terminated, truncated, info = _net_end(env)
            assert info["action_success"], f"net_end({nid}) failed"
            assert nid in obs["closed_nets"]
            last = i == len(routable) - 1
            assert terminated == last, (
                f"terminated={terminated} after closing {i + 1}/{len(routable)}"
            )
            assert not truncated

        # Voluntary finish: terminal despite unrouted work remaining.
        assert info["unrouted_count"] > 0
        # Give-up termination is NOT success (success = fully routed,
        # matching the post-hoc scorer) — val/success_rate must not read
        # abandoned episodes as wins.
        assert info["success"] is False
        assert info["ratsnest_reduction"] < 1.0

    def test_closed_net_leaves_net_valid_mask(self, netfree_env):
        from methods.rl_agent.models.v1.encoding import (
            net_valid_mask, sorted_net_codes_from_obs,
        )

        env = netfree_env
        obs, _ = env.reset()
        codes = sorted_net_codes_from_obs(obs)
        routable = sorted(env._routable_nets)
        if len(routable) < 2:
            pytest.skip("need >=2 routable nets to check partial exclusion")
        target = routable[0]

        before = net_valid_mask(sorted_net_codes=codes, last_obs=obs)
        assert before[codes.index(target)]

        _select(env, target)
        obs, _, _, _, _ = _net_end(env)
        after = net_valid_mask(sorted_net_codes=codes, last_obs=obs)
        assert not after[codes.index(target)]
        # Other routable nets stay selectable.
        other = routable[1]
        assert after[codes.index(other)]

    def test_reset_clears_closed_nets(self, netfree_env):
        env = netfree_env
        env.reset()
        target = sorted(env._routable_nets)[0]
        _select(env, target)
        obs, _, _, _, _ = _net_end(env)
        assert target in obs["closed_nets"]
        obs, _ = env.reset()
        assert obs["closed_nets"] == []
        assert env._episode_closed_nets == set()


class TestDefaultRuleUnchanged:
    def test_net_end_still_masked_and_no_closed_termination(self, default_env):
        env = default_env
        obs, _ = env.reset()
        nid = sorted(env._routable_nets)[0]
        _, _, terminated, _, info = _select(env, nid)
        assert info["action_success"]
        assert not terminated
        # Default rule: net_end unavailable on an incomplete net -> the
        # closed set can never grow, so the new termination clause is inert.
        assert not env.action_masks()[ACT_NET_END]
        assert env._episode_closed_nets == set()
