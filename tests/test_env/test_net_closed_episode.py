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
  - ``net_reselect=True`` (env knob, default off): a closed net with ratsnest
    left stays in the net-valid mask and ``net_select`` reopens it; the
    all-nets-closed termination is unchanged.

The give-up loop this kills (permissive rules): select -> net_end -> reselect
the same net forever (observed as a net-free collapse in an early run).
"""

import os

import pytest

from configs.loader.schema import EnvConfig
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


def _net_closed_flags(obs, monkeypatch) -> dict[int, float]:
    """NET-token ``closed`` channel per net code, as the RL tokenizer emits it."""
    from methods.rl_agent.models.v1 import tokenizer as tk
    from methods.rl_agent.models.v1.encoding import sorted_net_codes_from_obs

    flags: dict[int, float] = {}
    orig = tk._SmallBlockBufs.emit_net_row

    def spy(self, b, pos, slot, constraints, is_closed, ctx, net_positions_b):
        flags[slot] = 1.0 if is_closed else 0.0
        return orig(self, b, pos, slot, constraints, is_closed, ctx, net_positions_b)

    monkeypatch.setattr(tk._SmallBlockBufs, "emit_net_row", spy)
    tk.BatchedStateTokenizer(disable_slot_emb=False, d_model=32, n_freq=4)._walk_obs([obs])
    monkeypatch.setattr(tk._SmallBlockBufs, "emit_net_row", orig)
    codes = sorted_net_codes_from_obs(obs)
    return {codes[slot]: v for slot, v in flags.items()}


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
    e = PCBWorld(BOARD_PATH, EnvConfig.with_overrides(max_steps=200, masking_rule=str(rule)))
    yield e
    e.close()


@pytest.fixture
def default_env():
    _skip_if_unavailable()
    from pcb_world.core.env import PCBWorld

    e = PCBWorld(BOARD_PATH, EnvConfig.with_overrides(max_steps=50, masking_rule="default"))  # the paper rule (net_end masked until complete)
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


@pytest.fixture
def reselect_env(tmp_path):
    _skip_if_unavailable()
    from pcb_world.core.env import PCBWorld

    rule = tmp_path / "netfree_test.yaml"
    rule.write_text(NETFREE_RULE)
    e = PCBWorld(
        BOARD_PATH,
        EnvConfig.with_overrides(max_steps=200, masking_rule=str(rule), net_reselect=True),
    )
    yield e
    e.close()


class TestNetReselect:
    """``net_reselect``: a net closed while unrouted stays selectable and
    selecting it reopens it; termination is the unchanged all-nets-closed rule."""

    def test_closed_unrouted_net_stays_in_net_valid_mask(self, reselect_env):
        from methods.rl_agent.models.v1.encoding import (
            net_valid_mask, sorted_net_codes_from_obs,
        )

        env = reselect_env
        obs, _ = env.reset()
        codes = sorted_net_codes_from_obs(obs)
        target = sorted(env._routable_nets)[0]
        _select(env, target)
        obs, _, _, _, _ = _net_end(env)
        assert target in obs["closed_nets"]
        mask = net_valid_mask(sorted_net_codes=codes, last_obs=obs,
                              net_reselect=env.net_reselect)
        assert mask[codes.index(target)]
        # The same observation under the default rule retires the net.
        assert not net_valid_mask(
            sorted_net_codes=codes, last_obs=obs)[codes.index(target)]

    def test_select_reopens_the_net_and_reclose_counts_for_termination(
            self, reselect_env, monkeypatch):
        """net_select is only legal with no net held, so a reopened net is the
        current net: reopening takes it out of ``closed_nets`` (its NET-token
        closed flag returns to 0) and bumps ``net_reopens``; its next net_end
        closes it again, and the episode terminates only once every routable
        net is closed at the same time."""
        env = reselect_env
        env.reset()
        routable = sorted(env._routable_nets)
        if len(routable) < 2:
            pytest.skip("need >=2 routable nets so the re-close does not terminate")
        first, rest = routable[0], routable[1:]

        _select(env, first)
        obs, _, terminated, _, info = _net_end(env)
        assert first in obs["closed_nets"] and not terminated
        assert info["net_reopens"] == 0
        assert _net_closed_flags(obs, monkeypatch)[first] == 1.0

        obs, _, terminated, _, info = _select(env, first)        # reopen
        assert info["action_success"] and not terminated
        assert obs["router_head"]["current_net"] == first
        assert first not in obs["closed_nets"]
        assert info["net_reopens"] == 1
        assert _net_closed_flags(obs, monkeypatch)[first] == 0.0

        obs, _, terminated, _, info = _net_end(env)              # re-close
        assert first in obs["closed_nets"] and not terminated
        assert info["net_reopens"] == 1

        for nid in rest:                                          # close the others
            _select(env, nid)
            obs, _, terminated, truncated, info = _net_end(env)
        assert set(routable) <= set(obs["closed_nets"])
        assert terminated and not truncated
        assert info["unrouted_count"] > 0 and info["success"] is False

    def test_checkpoint_restore_carries_reopen_state(self, reselect_env):
        env = reselect_env
        env.reset()
        first = sorted(env._routable_nets)[0]
        _select(env, first)
        _net_end(env)
        ck = env.checkpoint()
        try:
            _, _, _, _, info = _select(env, first)
            assert info["net_reopens"] == 1
            env.restore(ck)
            assert first in env._episode_closed_nets
            assert env._net_reopen_count == 0
        finally:
            env.release_checkpoint(ck)

    def test_factory_wires_the_knob_to_env_and_wrapper(self, tmp_path):
        """One knob: RLEnvConfig's env.net_reselect reaches PCBWorld, whose
        property the wrapper's net-valid mask reads."""
        _skip_if_unavailable()
        from configs.loader.schema import EnvConfig, RLEnvConfig
        from methods.rl_agent.wrappers.factory import make_decoder_env

        rule = tmp_path / "netfree_test.yaml"
        rule.write_text(NETFREE_RULE)
        for on in (True, False):
            cfg = RLEnvConfig(env=EnvConfig(masking_rule=str(rule), net_reselect=on))
            w = make_decoder_env(BOARD_PATH, cfg, seed=0, policy_net_select=True)
            try:
                assert w.env.net_reselect is on
            finally:
                w.close()

    def test_subproc_pool_workers_carry_the_knob(self, tmp_path):
        """make_decoder_env_pool: each subprocess worker's PCBWorld gets the
        value, and the env record of its config carries it."""
        _skip_if_unavailable()
        from methods.rl_agent.wrappers.factory import env_record, make_decoder_env_pool
        from tests.helpers.env_kwargs import full_env_kwargs

        rule = tmp_path / "netfree_test.yaml"
        rule.write_text(NETFREE_RULE)
        kwargs = full_env_kwargs(max_steps=20, masking_rule=str(rule), net_reselect=True)
        pool = make_decoder_env_pool(BOARD_PATH, n_envs=1, advance_rng_on_reload=False, **kwargs)
        try:
            assert env_record(**kwargs)["net_reselect"] is True
            assert pool.env_method("get_wrapper_attr", "net_reselect") == [True]
        finally:
            pool.close()

    def test_complete_net_is_not_reopened(self):
        """Only a net closed while it still has ratsnest reopens: net_select on a
        closed, fully routed net fails (no ratsnest edge) and the net stays
        closed, exactly as with the knob off."""
        board = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                             "two_net_multiterm_board.kicad_pcb")
        _skip_if_unavailable()
        from pcb_world.core.action_schema import ACT_MAKE_LINE, ACT_START_ROUTE
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(
            board,
            EnvConfig.with_overrides(max_steps=60, masking_rule="default", net_reselect=True),
        )
        try:
            env.reset()
            _select(env, 1)
            for at, xy in ((ACT_START_ROUTE, (10.0, 10.0)), (ACT_MAKE_LINE, (25.0, 10.0)),
                           (ACT_START_ROUTE, (25.0, 10.0)), (ACT_MAKE_LINE, (40.0, 10.0)),
                           (ACT_START_ROUTE, (25.0, 5.0)), (ACT_MAKE_LINE, (25.0, 10.0))):
                _, _, _, _, info = env.step({"action_type": at, "x_mm": xy[0], "y_mm": xy[1],
                                             "layer": 1, "net_id": 0, "routing_mode": 2})
                assert info["action_success"], info
            obs, _, terminated, _, _ = _net_end(env)      # net 1 complete, closed
            assert 1 in obs["closed_nets"] and not terminated
            obs, _, _, _, info = _select(env, 1)
            assert not info["action_success"]
            assert 1 in obs["closed_nets"]
            assert info["net_reopens"] == 0
        finally:
            env.close()

    def test_off_by_default_select_does_not_reopen(self, netfree_env):
        env = netfree_env
        env.reset()
        assert env.net_reselect is False
        target = sorted(env._routable_nets)[0]
        _select(env, target)
        _net_end(env)
        obs, _, _, _, _ = _select(env, target)
        assert target in obs["closed_nets"]


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
