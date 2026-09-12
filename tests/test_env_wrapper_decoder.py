"""Tests for methods.rl_agent.wrappers.adapter.

Test strategy
-------------
The wrapper has two responsibilities that must be tested independently:

1. **Helper edge cases** — pure Python utilities that operate on dict
   shapes the real env can't easily produce (empty nets, missing keys,
   string-vs-numeric net code ordering). These use literal dicts and
   have no C++ dependency.

2. **Real-board behaviour** — everything else (pointer ordering vs
   tokenizer, action decode, reset/step/action_masks). These run the
   real ``PCBWorld`` against ``simple_routing_board.kicad_pcb``,
   capture obs snapshots at three canonical states (idle / net_selected
   / routing), and replay those obs through the wrapper. Where the test
   needs to inspect the env action dict the wrapper produces, we feed
   the real obs into a tiny ``_StubEnv`` so ``step`` doesn't actually
   advance state — that lets us assert on ``last_action`` without
   committing to a real env transition.

Both groups are skipped only when the C++ ``kicad_rl_router`` module or
the fixture board is missing — and only the real-board tests are
affected by that skip.

Board layout (simple_routing_board.kicad_pcb, 2-layer, 3 nets):
  NET1: (10,10) <-> (40,10)  horizontal
  NET2: (10,20) <-> (40,20)  horizontal
  NET3: (25, 5) <-> (25,25)  vertical
"""

from __future__ import annotations

import copy
import os
import re
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from pcb_world.engine import engine_available
import torch
from gymnasium import spaces

from pcb_world.core.masking import (
    ACT_FINISH,
    ACT_IDLE,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)
from tests.helpers.reference_tokenizer import StateTokenizer
from methods.rl_agent.wrappers.adapter import (
    KiCadRLWrapper,
    _cand_mm_list_from_obs,
    _sorted_net_codes_from_obs,
)


# ===================================================================
# Real-board fixture (skip if C++ kicad_rl_router or board missing)
# ===================================================================
_FIXTURE_BOARD = os.path.join(
    os.path.dirname(__file__), "fixtures", "simple_routing_board.kicad_pcb",
)


def _skip_if_no_env() -> None:
    if not os.path.exists(_FIXTURE_BOARD):
        pytest.skip(f"Fixture board not found: {_FIXTURE_BOARD}")
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


@pytest.fixture(scope="module")
def real_obs_states() -> dict[str, dict]:
    """Capture three canonical obs snapshots from the real env.

    Returns a dict with keys:
        - ``idle``         : just after reset(), no net selected
        - ``net_selected`` : after net_select(NET1) — first net by net_code
        - ``routing``      : after start_route from NET1's first candidate

    Each value is a deep-copied JSON observation dict so subsequent test
    mutations (or env teardown) cannot affect the cached snapshot.
    """
    _skip_if_no_env()
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
    try:
        wrapper = KiCadRLWrapper(env)

        idle, _ = wrapper.reset()
        idle_snapshot = copy.deepcopy(idle)

        # net_select pointer 0 → first net in sorted order (NET1, net_code=1)
        net_selected, *_ = wrapper.step(
            np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64),
        )
        net_selected_snapshot = copy.deepcopy(net_selected)

        # start_route pointer 0 → first candidate (RATSNEST point on a pad)
        routing, *_ = wrapper.step(
            np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64),
        )
        routing_snapshot = copy.deepcopy(routing)

        return {
            "idle": idle_snapshot,
            "net_selected": net_selected_snapshot,
            "routing": routing_snapshot,
        }
    finally:
        env.close()


# ===================================================================
# Stub env: captures the action dict the wrapper produces
# ===================================================================
class _StubEnv(gym.Env):
    """Records ``step`` calls so tests can assert on the decoded action.

    Subclasses :class:`gym.Env` so :class:`gym.Wrapper` can wrap it
    without tripping its ``isinstance(env, Env)`` assertion. Implements
    only the bits of :class:`PCBWorld` the wrapper actually touches:
    ``reset`` / ``step`` / ``action_masks``. ``step`` does NOT advance
    state — it just stores the action dict and re-emits the same obs,
    which is what lets us run dozens of decode-table assertions against
    a single real-board obs snapshot.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, obs: dict, action_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._obs = obs
        self._action_mask = (
            action_mask
            if action_mask is not None
            else np.ones(6, dtype=bool)
        )
        self.last_action: dict | str | None = None

        # Derive board_info.nets from the obs dict so the wrapper's
        # __init__ can call ``sorted(env.board_info.nets.keys())``.
        nets_dict: dict[int, object] = {}
        board_static = obs.get("board_static", {})
        for key in board_static.get("nets", {}):
            m = re.match(r"net_(\d+)", key)
            if m:
                nets_dict[int(m.group(1))] = board_static["nets"][key]
        self.board_info = SimpleNamespace(nets=nets_dict)

        # Minimal placeholder spaces — wrapper doesn't read them but
        # gym.Env subclasses are expected to declare them.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)

    def reset(self, *, seed=None, options=None):
        return self._obs, {}

    def step(self, action_dict):
        self.last_action = action_dict
        return self._obs, 0.0, False, False, {}

    def action_masks(self) -> np.ndarray:
        return self._action_mask.copy()


def _wrap_stub(
    obs: dict, action_mask: np.ndarray | None = None,
) -> KiCadRLWrapper:
    """Build a wrapper around ``_StubEnv`` preloaded with ``obs``.

    The wrapper's ``__init__`` does not call ``reset()`` itself, so we
    invoke it once here to prime the cache before any test step.
    """
    env = _StubEnv(obs, action_mask)
    w = KiCadRLWrapper(env)
    w.reset()  # prime _last_obs / _sorted_net_codes / _cand_mm
    return w


# ===================================================================
# 1. Pure helper edge cases (no env, no fixture)
# ===================================================================
class TestSortedNetCodesHelper:
    """Edge cases for _sorted_net_codes_from_obs that real env won't hit."""

    def test_basic_numeric_ordering(self):
        # net_10 must come AFTER net_3 (numeric, not lexicographic)
        obs = {"board_static": {"nets": {"net_3": {}, "net_1": {}, "net_10": {}}}}
        assert _sorted_net_codes_from_obs(obs) == [1, 3, 10]

    def test_empty_nets(self):
        obs = {"board_static": {"nets": {}}}
        assert _sorted_net_codes_from_obs(obs) == []

    def test_missing_board_static(self):
        assert _sorted_net_codes_from_obs({}) == []


# ===================================================================
# 2. Pointer ordering vs tokenizer (real obs from real board)
# ===================================================================
class TestPointerOrderingVsTokenizer:
    """The wrapper's pointer→identity tables must match the tokenizer.

    These are the most important invariants: a silent ordering mismatch
    here would corrupt every net_select / cand-based action and produce
    near-random GRPO learning that's almost impossible to debug.
    """

    def test_net_codes_match_tokenizer_at_idle(self, real_obs_states):
        obs = real_obs_states["idle"]
        tokenizer = StateTokenizer(d_model=32)
        out = tokenizer([obs])

        wrapper_codes = _sorted_net_codes_from_obs(obs)

        # Tokenizer emits NET_END token positions in _sorted_net_keys order;
        # the count of valid (>=0) entries must equal the wrapper's count.
        num_tokenized_nets = int((out.net_indices[0] >= 0).sum().item())
        assert len(wrapper_codes) == num_tokenized_nets
        # Real board has 3 user nets (NET1/NET2/NET3) → codes [1, 2, 3]
        assert wrapper_codes == [1, 2, 3]

    def test_cand_list_matches_tokenizer_when_net_selected(
        self, real_obs_states,
    ):
        obs = real_obs_states["net_selected"]
        tokenizer = StateTokenizer(d_model=32)
        out = tokenizer([obs])

        wrapper_cands = _cand_mm_list_from_obs(obs)
        tokenizer_cands = out.cand_mm_list[0]

        # tuple-wise identity: a single off-by-one or reordered entry would
        # silently re-bind every pointer index to the wrong (x, y, layer).
        assert len(wrapper_cands) > 0, "expected non-empty cand pool"
        assert len(wrapper_cands) == len(tokenizer_cands)
        for w, t in zip(wrapper_cands, tokenizer_cands):
            assert w[0] == pytest.approx(t[0])
            assert w[1] == pytest.approx(t[1])
            assert w[2] == t[2]

    def test_cand_list_matches_tokenizer_during_routing(
        self, real_obs_states,
    ):
        """During routing the pool gains 8 directional candidates — the
        wrapper and tokenizer must add them in the same place."""
        obs = real_obs_states["routing"]
        assert obs["router_head"]["is_routing"] is True

        tokenizer = StateTokenizer(d_model=32)
        out = tokenizer([obs])

        wrapper_cands = _cand_mm_list_from_obs(obs)
        tokenizer_cands = out.cand_mm_list[0]

        assert len(wrapper_cands) == len(tokenizer_cands)
        for w, t in zip(wrapper_cands, tokenizer_cands):
            assert w[0] == pytest.approx(t[0])
            assert w[1] == pytest.approx(t[1])
            assert w[2] == t[2]

    def test_cand_list_empty_at_idle(self, real_obs_states):
        """No net selected → no candidates of any type."""
        obs = real_obs_states["idle"]
        assert _cand_mm_list_from_obs(obs) == []


# ===================================================================
# 3. Wrapper-level basics (stub env wrapping real obs)
# ===================================================================
class TestReset:

    def test_returns_raw_dict_obs(self, real_obs_states):
        w = _wrap_stub(real_obs_states["idle"])
        reset_obs, info = w.reset()
        # Critical: dict NOT numpy. The decoder-only policy needs the
        # full nested structure for tokenization.
        assert isinstance(reset_obs, dict)
        assert "board_static" in reset_obs
        assert "router_head" in reset_obs
        assert isinstance(info, dict)

    def test_cache_populated_after_reset(self, real_obs_states):
        w = _wrap_stub(real_obs_states["net_selected"])
        # NET1/NET2/NET3 → sorted [1, 2, 3]
        assert w.sorted_net_codes == [1, 2, 3]
        # net_selected state must have cands (the selected net's pads/ratsnest)
        assert len(w.cand_mm_list) > 0


class TestActionMasks:

    def test_passthrough_shape_and_values(self, real_obs_states):
        # Inject a custom mask through the stub env to verify the wrapper
        # forwards env.action_masks() byte-for-byte without modification.
        custom_mask = np.array(
            [True, False, True, True, False, True, False], dtype=bool,
        )
        w = _wrap_stub(real_obs_states["idle"], action_mask=custom_mask)
        out = w.action_masks()
        assert out.shape == (7,)
        assert out.dtype == np.bool_
        np.testing.assert_array_equal(out, custom_mask)


# ===================================================================
# 4. Action decode table (one test per action_type + edge cases)
# ===================================================================
class TestDecodeAction:

    def test_net_select_pointer_to_net_code(self, real_obs_states):
        w = _wrap_stub(real_obs_states["idle"])
        # ACT_NET_SELECT ignores the pointer and picks via _pick_net_id()
        # (prefers unrouted nets, random among them). Just verify a valid
        # net code is chosen.
        w.step(np.array([ACT_NET_SELECT, 2, -1], dtype=np.int64))
        assert w.env.last_action["action_type"] == ACT_NET_SELECT
        assert w.env.last_action["net_id"] in w.sorted_net_codes

    def test_net_select_pointer_zero_picks_first_net(self, real_obs_states):
        w = _wrap_stub(real_obs_states["idle"])
        # ACT_NET_SELECT ignores the pointer — verify a valid net is picked.
        w.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
        assert w.env.last_action["action_type"] == ACT_NET_SELECT
        assert w.env.last_action["net_id"] in w.sorted_net_codes

    # NB: out-of-range pointer cases are covered by TestInvalidPointerReject
    # below — they do not reach env.step at all.

    def test_start_route_includes_layer_from_cand(self, real_obs_states):
        w = _wrap_stub(real_obs_states["net_selected"])
        # First cand is whatever collect_raw_candidates yields first
        # (priority: RATSNEST > PAD > ...). Reference w.cand_mm_list[0]
        # so the test stays robust to fixture board changes.
        expected_x, expected_y, expected_layer = w.cand_mm_list[0]
        w.step(np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64))
        assert w.env.last_action == {
            "action_type": ACT_START_ROUTE,
            "x_mm": pytest.approx(expected_x),
            "y_mm": pytest.approx(expected_y),
            "layer": expected_layer,  # ← start_route IS the only cand action that uses layer
        }

    def test_net_end_has_no_extra_params(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        w.step(np.array([ACT_NET_END, -1, -1], dtype=np.int64))
        assert w.env.last_action == {"action_type": ACT_NET_END}

    def test_make_line_omits_layer_field(self, real_obs_states):
        """make_line MUST NOT include a 'layer' key — the env dispatcher
        doesn't accept one and the engine uses the router head's current
        layer instead. (See pcb_world/core/action.py.)"""
        w = _wrap_stub(real_obs_states["routing"])
        expected_x, expected_y, _layer = w.cand_mm_list[0]
        w.step(np.array([ACT_MAKE_LINE, 0, 1], dtype=np.int64))
        assert w.env.last_action == {
            "action_type": ACT_MAKE_LINE,
            "x_mm": pytest.approx(expected_x),
            "y_mm": pytest.approx(expected_y),
            "routing_mode": 1,
        }
        # Explicit absence assertion — silent inclusion would be hard to
        # spot since ACTION_REGISTRY would just filter the extra key.
        assert "layer" not in w.env.last_action

    def test_make_via_omits_layer_field(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        expected_x, expected_y, _layer = w.cand_mm_list[0]
        w.step(np.array([ACT_MAKE_VIA, 0, 0], dtype=np.int64))
        assert w.env.last_action == {
            "action_type": ACT_MAKE_VIA,
            "x_mm": pytest.approx(expected_x),
            "y_mm": pytest.approx(expected_y),
            "routing_mode": 0,
        }
        assert "layer" not in w.env.last_action

    def test_finish_uses_only_routing_mode(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        w.step(np.array([ACT_FINISH, -1, 2], dtype=np.int64))
        assert w.env.last_action == {
            "action_type": ACT_FINISH,
            "routing_mode": 2,
        }

    def test_routing_mode_minus_one_clamps_to_default(self, real_obs_states):
        """Defensive: if a slot says ``mode=-1`` (unused) but the action
        actually needs a mode, clamp to Walkaround (2) instead of letting
        a negative int reach the env dispatcher."""
        w = _wrap_stub(real_obs_states["routing"])
        w.step(np.array([ACT_MAKE_LINE, 0, -1], dtype=np.int64))
        assert w.env.last_action is not None
        assert w.env.last_action["routing_mode"] == 2


# ===================================================================
# 5. Wrapper-level invalid-pointer reject
# ===================================================================
class TestInvalidPointerReject:
    """Out-of-range pointer must not feed garbage coords to the engine.

    The wrapper decodes an OOR pointer to the idle fallback
    (``FALLBACK_ACTION`` = ``action_type=ACT_IDLE`` + ``_parse_invalid``), which
    the env penalises via its parse_fail path — the same invalid-input handling
    as the LLM branch. So env.step IS called, but with idle (no coords), never
    the garbage-coord action. (The stub returns reward 0.0; the real parse_fail
    penalty is exercised by env-level tests.)
    """

    _IDLE_FALLBACK = {"action_type": ACT_IDLE, "_parse_invalid": True}

    def test_make_line_with_oor_pointer_idle_fallback(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        _, reward, term, trunc, _ = w.step(
            np.array([ACT_MAKE_LINE, 99999, 1], dtype=np.int64),
        )
        assert term is False and trunc is False
        # env.step called with the idle fallback (no garbage coords)
        assert w.env.last_action == self._IDLE_FALLBACK

    def test_make_via_with_oor_pointer_idle_fallback(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        w.step(np.array([ACT_MAKE_VIA, 99999, 0], dtype=np.int64))
        assert w.env.last_action == self._IDLE_FALLBACK

    def test_start_route_with_oor_pointer_idle_fallback(self, real_obs_states):
        w = _wrap_stub(real_obs_states["net_selected"])
        w.step(np.array([ACT_START_ROUTE, 99999, -1], dtype=np.int64))
        assert w.env.last_action == self._IDLE_FALLBACK

    def test_net_select_with_oor_pointer_not_rejected(self, real_obs_states):
        """ACT_NET_SELECT ignores the pointer entirely (_pick_net_id),
        so an out-of-range pointer is harmless — the action proceeds."""
        w = _wrap_stub(real_obs_states["idle"])
        _, reward, _, _, info = w.step(
            np.array([ACT_NET_SELECT, 99, -1], dtype=np.int64),
        )
        assert reward == 0.0  # no penalty
        assert w.env.last_action["action_type"] == ACT_NET_SELECT
        assert w.env.last_action["net_id"] in w.sorted_net_codes

    def test_net_end_does_not_check_pointer(self, real_obs_states):
        """Pointer-less actions are unaffected by the reject check."""
        w = _wrap_stub(real_obs_states["routing"])
        w.step(np.array([ACT_NET_END, 99999, -1], dtype=np.int64))
        # net_end ignores the pointer entirely; env.step IS called
        assert w.env.last_action == {"action_type": ACT_NET_END}

    def test_finish_does_not_check_pointer(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        w.step(np.array([ACT_FINISH, 99999, 1], dtype=np.int64))
        assert w.env.last_action == {
            "action_type": ACT_FINISH,
            "routing_mode": 1,
        }


# ===================================================================
# 6. Action input format flexibility
# ===================================================================
class TestActionInputTypes:
    """The wrapper accepts numpy arrays, torch tensors, and Python lists."""

    def test_numpy_array_input(self, real_obs_states):
        w = _wrap_stub(real_obs_states["idle"])
        w.step(np.array([ACT_NET_END, -1, -1], dtype=np.int64))
        assert w.env.last_action == {"action_type": ACT_NET_END}

    def test_torch_tensor_input(self, real_obs_states):
        # This is the form policy.act() returns: actions[i] is a 1-D tensor.
        w = _wrap_stub(real_obs_states["idle"])
        w.step(torch.tensor([ACT_NET_END, -1, -1], dtype=torch.int64))
        assert w.env.last_action == {"action_type": ACT_NET_END}

    def test_python_list_input(self, real_obs_states):
        w = _wrap_stub(real_obs_states["idle"])
        w.step([ACT_NET_END, -1, -1])
        assert w.env.last_action == {"action_type": ACT_NET_END}

    def test_wrong_shape_raises_value_error(self, real_obs_states):
        # Length-2 input must NOT be silently truncated/extended — that
        # would mask off-by-one bugs in the caller.
        w = _wrap_stub(real_obs_states["idle"])
        with pytest.raises(ValueError):
            w.step(np.array([ACT_NET_END, -1], dtype=np.int64))


# ===================================================================
# 7. End-to-end with the live env (no stub)
# ===================================================================
class TestEndToEnd:
    """Drive the real env through the wrapper without any stubbing."""

    def test_real_env_reset_and_action_masks(self):
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env)
            obs, info = wrapper.reset()
            assert "board_static" in obs
            assert "router_head" in obs

            mask = wrapper.action_masks()
            assert mask.shape == (7,)
            assert mask.dtype == np.bool_

            assert wrapper.sorted_net_codes == [1, 2, 3]
        finally:
            env.close()

    def test_policy_to_env_rollout(self):
        """Full pipeline: tokenizer → policy → wrapper → env, run for 3 steps."""
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld
        from methods.rl_agent.models.v1.net import KiCadRLModel

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env)
            policy = KiCadRLModel(
                d_model=32, n_heads=2, n_layers=2, d_ff=64,
            )
            policy.eval()

            obs, _ = wrapper.reset()
            for _ in range(3):
                mask = torch.from_numpy(wrapper.action_masks()).unsqueeze(0)
                actions, _log_probs = policy.act([obs], action_masks=mask)
                assert actions.shape == (1, 3)
                obs, _reward, terminated, truncated, _info = wrapper.step(
                    actions[0],
                )
                if terminated or truncated:
                    break
        finally:
            env.close()


# ===================================================================
# 8. Same-point masking (MLP-equivalent)
# ===================================================================
class TestStartPointMaskingState:
    """Unit-level tracking of ``_start_route_xy`` via the stub env.

    These use the stub env so they can exercise every action_type in
    isolation without the real engine's state machine constraining the
    sequence. The real-env scenarios are covered in
    :class:`TestStartPointMaskingRealEnv` below.
    """

    def test_reset_clears_state(self, real_obs_states):
        w = _wrap_stub(real_obs_states["net_selected"])
        # Simulate a prior episode's start_route leaving state behind.
        w._start_route_xy = (1.23, 4.56, 1)
        w.reset()
        assert w._start_route_xy is None
        assert w.start_route_pointer_indices().shape == (0,)

    def test_start_route_sets_xy(self, real_obs_states):
        w = _wrap_stub(real_obs_states["net_selected"])
        expected_x, expected_y, expected_l = w.cand_mm_list[0]
        w.step(np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64))
        assert w._start_route_xy == pytest.approx(
            (expected_x, expected_y, expected_l))

    def test_make_line_preserves_xy(self, real_obs_states):
        # Re-use the routing snapshot (post start_route) so cand pool
        # is non-empty. Manually prime _start_route_xy to something the
        # stub won't overwrite (stub doesn't advance env state).
        w = _wrap_stub(real_obs_states["routing"])
        w._start_route_xy = (7.0, 8.0, 1)
        w.step(np.array([ACT_MAKE_LINE, 0, 2], dtype=np.int64))
        assert w._start_route_xy == (7.0, 8.0, 1)

    def test_make_via_preserves_xy(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        w._start_route_xy = (7.0, 8.0, 1)
        w.step(np.array([ACT_MAKE_VIA, 0, 2], dtype=np.int64))
        assert w._start_route_xy == (7.0, 8.0, 1)

    def test_net_end_clears_xy(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        w._start_route_xy = (7.0, 8.0, 1)
        w.step(np.array([ACT_NET_END, -1, -1], dtype=np.int64))
        assert w._start_route_xy is None

    def test_finish_preserves_xy(self, real_obs_states):
        # finish() preserves _start_route_xy so the next start_route cannot
        # restart from the same pad after a finish (success OR fail).
        w = _wrap_stub(real_obs_states["routing"])
        w._start_route_xy = (7.0, 8.0, 1)
        w.step(np.array([ACT_FINISH, -1, 2], dtype=np.int64))
        assert w._start_route_xy == (7.0, 8.0, 1)

    def test_net_select_clears_xy(self, real_obs_states):
        w = _wrap_stub(real_obs_states["idle"])
        w._start_route_xy = (7.0, 8.0, 1)
        w.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
        assert w._start_route_xy is None

    def test_start_route_overwrites_xy(self, real_obs_states):
        """New start_route overwrites any prior value — prior start
        point is no longer masked, the new one is."""
        w = _wrap_stub(real_obs_states["net_selected"])
        w._start_route_xy = (99.0, 99.0, 1)  # stale prior value
        expected_x, expected_y, expected_l = w.cand_mm_list[0]
        w.step(np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64))
        assert w._start_route_xy == pytest.approx(
            (expected_x, expected_y, expected_l))


class TestStartRoutePointerIndices:
    """``start_route_pointer_indices()`` must return the cand-pool
    indices matching ``_start_route_xy`` — same ``(x, y)`` AND same
    layer — so the policy masks exactly the started candidate while
    same-xy siblings on other layers survive."""

    def test_returns_empty_when_no_start_xy(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        assert w._start_route_xy is None
        assert w.start_route_pointer_indices().shape == (0,)

    def test_returns_indices_of_matching_cands(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        # Pick any cand from the pool and claim it's the start point.
        # Only cands with the same (x, y) AND layer must come back.
        target_x, target_y, target_l = w.cand_mm_list[0]
        w._start_route_xy = (target_x, target_y, target_l)
        expected = [
            i for i, (cx, cy, cl) in enumerate(w.cand_mm_list)
            if abs(cx - target_x) < 0.01 and abs(cy - target_y) < 0.01
            and cl == target_l
        ]
        idxs = w.start_route_pointer_indices().tolist()
        assert idxs == expected
        assert 0 in idxs

        if len(w.cand_mm_list) > 2:
            tx, ty, tl = w.cand_mm_list[2]
            w._start_route_xy = (tx, ty, tl)
            expected2 = [
                i for i, (cx, cy, cl) in enumerate(w.cand_mm_list)
                if abs(cx - tx) < 0.01 and abs(cy - ty) < 0.01
                and cl == tl
            ]
            assert w.start_route_pointer_indices().tolist() == expected2

    def test_returns_empty_when_xy_not_in_pool(self, real_obs_states):
        w = _wrap_stub(real_obs_states["routing"])
        w._start_route_xy = (99999.0, 99999.0, 1)
        assert w.start_route_pointer_indices().shape == (0,)

    def test_mask_start_point_false_disables(self, real_obs_states):
        """``mask_start_point=False`` turns the whole feature off —
        even with ``_start_route_xy`` set, the result is empty."""
        env = _StubEnv(real_obs_states["routing"])
        w = KiCadRLWrapper(env, mask_start_point=False)
        w.reset()
        target_x, target_y, target_l = w.cand_mm_list[0]
        w._start_route_xy = (target_x, target_y, target_l)
        assert w.start_route_pointer_indices().shape == (0,)

    def test_same_xy_other_layer_survives(self):
        """A stacked front/back pad pair puts the WHOLE pool at one xy.
        Excluding the started layer must keep the sibling-layer candidate
        selectable, so the pool never empties.
        """
        from methods.rl_agent.models.v1 import encoding as _mask
        cand = [(125.0, 49.53, 1), (125.0, 49.53, 2)]
        idxs = _mask.start_route_pointer_indices(
            mask_start_point=True,
            start_route_xy=(125.0, 49.53, 1),
            cand_mm=cand,
        )
        assert idxs.tolist() == [0]  # layer-2 twin stays selectable


class TestStartPointMaskingPolicyIntegration:
    """End-to-end: wrapper-reported index + policy's pointer_masks kwarg
    must actually zero out the logit for that cand index."""

    def test_policy_masks_blocked_cand(self, real_obs_states):
        """Deterministic argmax sampling with a pointer_mask that kills
        the argmax cand must fall back to a different index.

        Force action_type ∈ {make_line, make_via} so the pointer slot
        definitely comes from ``cand_indices`` (not ``net_indices`` —
        ``pointer_masks`` only blocks cand logits, by design).
        """
        _skip_if_no_env()
        from methods.rl_agent.models.v1.net import KiCadRLModel

        obs = real_obs_states["routing"]
        policy = KiCadRLModel(
            d_model=32, n_heads=2, n_layers=2, d_ff=64,
        )
        policy.eval()

        # Only make_line / make_via allowed — both use the cand pointer.
        at_mask = np.zeros(7, dtype=bool)
        at_mask[ACT_MAKE_LINE] = True
        at_mask[ACT_MAKE_VIA] = True
        action_mask = torch.from_numpy(at_mask).unsqueeze(0)  # (1, 7)

        # Baseline: no pointer masking — record the argmax cand.
        actions_unmasked, _ = policy.act(
            [obs], action_masks=action_mask, deterministic=True,
        )
        baseline_ptr = int(actions_unmasked[0, 1].item())
        assert baseline_ptr >= 0, (
            f"expected a valid cand index, got {baseline_ptr}"
        )

        # With pointer_masks: block that cand, the policy must pick another.
        ptr_masks = torch.tensor([baseline_ptr], dtype=torch.long)
        actions_masked, _ = policy.act(
            [obs],
            action_masks=action_mask,
            deterministic=True,
            pointer_masks=ptr_masks,
        )
        masked_ptr = int(actions_masked[0, 1].item())
        assert masked_ptr != baseline_ptr, (
            f"policy returned blocked cand idx {masked_ptr} even "
            f"though it was masked (baseline was {baseline_ptr})"
        )

    def test_evaluate_respects_pointer_masks(self, real_obs_states):
        """Re-scoring the same action with pointer_masks that block that
        very index must yield log_prob = -inf (the distribution assigns
        zero probability to a masked outcome)."""
        _skip_if_no_env()
        from methods.rl_agent.models.v1.net import KiCadRLModel

        obs = real_obs_states["routing"]
        policy = KiCadRLModel(
            d_model=32, n_heads=2, n_layers=2, d_ff=64,
        )
        policy.eval()

        # Force a make_line action at cand idx 0.
        action = torch.tensor(
            [[ACT_MAKE_LINE, 0, 2]], dtype=torch.long,
        )
        action_mask = torch.ones(1, 7, dtype=torch.bool)
        # Block cand idx 0.
        ptr_masks = torch.tensor([0], dtype=torch.long)
        log_prob, _entropy, _value = policy.evaluate_actions_and_value(
            [obs], action,
            action_masks=action_mask,
            pointer_masks=ptr_masks,
        )
        # log_prob(masked action) = -inf. Note: -inf * 0 → NaN is NOT
        # what we'd want for the teacher-forced loss, but the collector
        # never stores a blocked action in the first place — the policy
        # sampled under the same mask, so log_prob on re-scoring sees
        # the same mask. So this is a safety assertion, not an
        # operational path.
        assert torch.isinf(log_prob).item() and log_prob.item() < 0


# ===================================================================
# 11. Policy-driven net selection (--policy-net-select)
# ===================================================================
class TestPolicyNetSelect:
    """``policy_net_select=True`` routes the policy's pointer to env net_id."""

    def test_decode_net_select_uses_pointer(self, real_obs_states):
        obs = real_obs_states["idle"]
        env = _StubEnv(obs)
        w = KiCadRLWrapper(env, policy_net_select=True)
        w.reset()
        nets = w.sorted_net_codes
        assert len(nets) >= 2
        # Pointer → 2nd net in sorted pool.
        w.step(np.array([ACT_NET_SELECT, 1, -1], dtype=np.int64))
        assert env.last_action == {
            "action_type": ACT_NET_SELECT,
            "net_id": nets[1],
        }

    def test_net_select_out_of_range_idle_fallback(self, real_obs_states):
        obs = real_obs_states["idle"]
        env = _StubEnv(obs)
        w = KiCadRLWrapper(env, policy_net_select=True)
        w.reset()
        bad_ptr = len(w.sorted_net_codes) + 5
        _obs, reward, term, trunc, info = w.step(
            np.array([ACT_NET_SELECT, bad_ptr, -1], dtype=np.int64),
        )
        # OOR net pointer decodes to idle fallback; env.step sees idle, not a
        # garbage net_id.
        assert not term and not trunc
        assert env.last_action == {"action_type": ACT_IDLE, "_parse_invalid": True}

    def test_net_valid_mask_order_matches_sorted_nets(self, real_obs_states):
        obs = real_obs_states["idle"]
        env = _StubEnv(obs)
        w = KiCadRLWrapper(env, policy_net_select=True)
        w.reset()
        mask = w.net_valid_mask()
        assert mask.shape == (len(w.sorted_net_codes),)
        # Idle state → all nets still have ratsnest points → mask all True.
        assert mask.all()

    def test_legacy_path_when_flag_off(self, real_obs_states):
        """Default (policy_net_select=False) preserves legacy random pick."""
        obs = real_obs_states["idle"]
        env = _StubEnv(obs)
        w = KiCadRLWrapper(env, seed=0)  # flag defaults to False
        w.reset()
        # Any in-range pointer must be IGNORED — env still gets a valid net_id
        # chosen by _pick_net_id (not necessarily the one the pointer indexed).
        w.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
        assert env.last_action["action_type"] == ACT_NET_SELECT
        assert env.last_action["net_id"] in w.sorted_net_codes


class TestPolicyNetSelectLogProb:
    """Policy-driven path: net_valid_mask must force -inf on routed-net slots."""

    def test_net_select_pointer_masked(self, real_obs_states):
        from methods.rl_agent.models.v1.net import (
            KiCadRLModel,
        )
        obs = real_obs_states["idle"]
        n_nets = len(_sorted_net_codes_from_obs(obs))
        assert n_nets >= 2, "fixture must expose >=2 nets"
        policy = KiCadRLModel(
            d_model=32, n_heads=2, n_layers=2, d_ff=64,
            use_critic=True, policy_net_select=True,
        )
        policy.eval()
        am = torch.zeros(1, 7, dtype=torch.bool)
        am[0, ACT_NET_SELECT] = True
        mode_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        # Mask every net except the last one.
        nvm = torch.zeros(1, n_nets, dtype=torch.bool)
        nvm[0, -1] = True
        acts, lp, _v = policy.act_and_value(
            [obs], action_masks=am,
            mode_mask=mode_mask,
            net_valid_mask=nvm,
            allow_net_select_lp=True,
            deterministic=True,
        )
        assert acts[0, 0].item() == ACT_NET_SELECT
        assert acts[0, 1].item() == n_nets - 1
        assert torch.isfinite(lp).all()


# ===================================================================
# The wrapper's _cand_mm must reflect directional_candidates.
# ===================================================================
class TestDirectionalCandidatesRegression:
    """The wrapper builds its pointer-index → coord table from
    ``_cand_mm_list_from_obs``, which inspects ``obs["_aug"]["directional_candidates"]``.
    ``_refresh_cache`` must run on the aug'd obs so the wrapper's cand pool
    matches the configured grid the policy's tokenizer uses — keeping
    ``pointer_idx`` decoding to the coordinates the policy intended.
    """

    def test_wrapper_cand_pool_uses_directional_candidates(self) -> None:
        """After reset+net_select+start_route, the wrapper's _cand_mm
        directional segment must contain 4 axis-aligned candidates at
        the configured grid spacing — not the 8-dir × 0.5mm fallback.
        """
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env, directional_candidates="grid10")
            wrapper.reset()
            wrapper.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
            obs, *_ = wrapper.step(
                np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64),
            )
            # In grid-10 mode, directional candidates are exactly 4
            # axis-aligned points at ±10mm from the head. They must all
            # land on the env's 10mm grid (offset 7.5 for our synth boards;
            # for the simple_routing_board the head is at a pad whose
            # coordinates we recover from the obs).
            head = obs["router_head"]["current_xy"]
            hx, hy = float(head[0]), float(head[1])
            expected = {
                (hx + 10.0, hy),
                (hx - 10.0, hy),
                (hx, hy + 10.0),
                (hx, hy - 10.0),
            }
            # _cand_mm contains pads/vias/track-endpoints/directional, all
            # tuples of (x, y, layer). Filter to (x, y) and check the
            # expected 4-way directional set is a subset.
            xy_set = {(round(x, 6), round(y, 6)) for x, y, _l in wrapper._cand_mm}
            for ex, ey in expected:
                key = (round(ex, 6), round(ey, 6))
                assert key in xy_set, (
                    f"directional candidate {(ex, ey)} missing from wrapper "
                    f"_cand_mm — wrapper likely fell back to grid_size=None "
                    f"(8-dir × 0.5mm). Full pool: {sorted(xy_set)}"
                )
            # And the 0.5mm sub-grid candidates that the buggy fallback
            # would have produced must NOT appear (e.g. head + (0.5, 0)).
            sub_grid_ghost = (round(hx + 0.5, 6), round(hy, 6))
            assert sub_grid_ghost not in xy_set, (
                f"0.5mm sub-grid candidate {sub_grid_ghost} found in wrapper "
                f"_cand_mm — fallback path is still active"
            )
        finally:
            env.close()

    def test_wrapper_cand_pool_matches_tokenizer_when_routing(self) -> None:
        """The wrapper's cand pool (``_cand_mm_list_from_obs``) must be
        tuple-wise identical to the tokenizer's ``_build_candidate_pool``
        for the same aug'd obs — required for pointer decode to stay valid.
        """
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld
        from pcb_world.vec.candidate_pool import (
            build_directional_candidates,
            collect_raw_candidates,
        )

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env, directional_candidates="grid10")
            wrapper.reset()
            wrapper.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
            obs, *_ = wrapper.step(
                np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64),
            )
            # Reproduce the tokenizer's path exactly: collect_raw_candidates
            # + build_directional_candidates using aug.directional_candidates.
            rh = obs.get("router_head", {})
            current_net_id = rh.get("current_net", -1)
            if current_net_id is None or current_net_id <= 0:
                current_net_id = None
            aug = obs.get("_aug") or {}
            assert aug.get("directional_candidates") == "grid10", (
                f"aug missing or wrong: {aug}"
            )
            extra = build_directional_candidates(
                (rh["current_xy"][0], rh["current_xy"][1]),
                int(rh.get("current_layer", 1)),
                mode=aug.get("directional_candidates"),
            )
            tokenizer_pool = [
                (round(x, 6), round(y, 6), layer)
                for x, y, layer, _ctype in collect_raw_candidates(
                    obs, current_net_id, extra,
                )
            ]
            wrapper_pool = [
                (round(x, 6), round(y, 6), layer)
                for x, y, layer in wrapper._cand_mm
            ]
            assert wrapper_pool == tokenizer_pool, (
                "wrapper._cand_mm diverges from tokenizer-equivalent pool. "
                f"wrapper={wrapper_pool}\ntokenizer={tokenizer_pool}"
            )
        finally:
            env.close()

    def test_multi_resolution_mode_reaches_wrapper_pool(self) -> None:
        """directional_candidates="multi_resolution" must flow through _aug
        into the wrapper's _cand_mm: the 25mm ring is present and the
        default 0.5mm ring is absent while routing.
        """
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env, directional_candidates="multi_resolution")
            wrapper.reset()
            wrapper.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
            obs, *_ = wrapper.step(
                np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64),
            )
            assert obs.get("_aug", {}).get("directional_candidates") == (
                "multi_resolution"
            )
            head = obs["router_head"]["current_xy"]
            hx, hy = float(head[0]), float(head[1])
            xy_set = {(round(x, 6), round(y, 6)) for x, y, _l in wrapper._cand_mm}
            assert (round(hx + 25.0, 6), round(hy, 6)) in xy_set, (
                f"25mm ladder candidate missing — pool: {sorted(xy_set)}"
            )
            assert (round(hx + 0.5, 6), round(hy, 6)) not in xy_set, (
                "default 0.5mm ring leaked into multi_resolution mode"
            )
        finally:
            env.close()

    def test_mres8_offboard_mask_on_fixture_board(self) -> None:
        """directional_candidates="mres8" + offboard_mask=True: the 25 / 50 mm
        rungs leave the fixture board; the wrapper embeds exactly those
        directional indices under obs["_masks"]["offboard"], the pool itself
        is unchanged (a mask, not a filter), and the default knob masks nothing.
        """
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld
        from pcb_world.vec.candidate_pool import CTYPE_DIRECTIONAL

        for knob in (True, False):
            env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
            try:
                wrapper = KiCadRLWrapper(
                    env, directional_candidates="mres8", offboard_mask=knob,
                )
                wrapper.reset()
                wrapper.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
                obs, *_ = wrapper.step(
                    np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64),
                )
                bs = obs["board_static"]
                x0, y0 = bs["bbox_x"], bs["bbox_y"]
                x1, y1 = x0 + bs["bbox_w"], y0 + bs["bbox_h"]
                hx, hy = obs["router_head"]["current_xy"]
                pool = wrapper.cand_mm_list
                # the 50 mm rung is in the pool either way (mask != filter)
                assert any(abs(x - (hx + 50.0)) < 1e-6 and abs(y - hy) < 1e-6
                           for x, y, _l in pool)
                got = obs["_masks"]["offboard"]
                assert got.dtype == np.int64
                assert np.array_equal(got, wrapper.offboard_pointer_indices())
                if not knob:
                    assert got.shape == (0,)
                    continue
                want = {
                    i for i, ((x, y, _l), ct) in enumerate(zip(pool, wrapper._cand_ctype))
                    if ct == CTYPE_DIRECTIONAL
                    and not (x0 <= x <= x1 and y0 <= y <= y1)
                }
                assert want, "fixture board should not contain a 50 mm rung"
                assert set(got.tolist()) == want
                # existing copper (pads / track ends) is never masked
                assert all(wrapper._cand_ctype[i] == CTYPE_DIRECTIONAL for i in got)
            finally:
                env.close()

    def test_aug_present_immediately_after_reset(self) -> None:
        """obs returned from reset() must already have _aug attached
        and _refresh_cache must have used it.
        """
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env, directional_candidates="grid10")
            obs, _ = wrapper.reset()
            aug = obs.get("_aug")
            assert aug is not None, "obs missing _aug after reset"
            assert aug.get("directional_candidates") == "grid10"
        finally:
            env.close()


# ===================================================================
# Connectivity filter (_aug["cluster_keys"]) — wrapper ↔ engine wiring
# ===================================================================


class TestConnectivityFilterIntegration:
    """The wrapper resolves the route head's connectivity cluster through the
    engine and injects it as ``_aug["cluster_keys"]``; the candidate pool then
    drops exactly that copper. Guards the wiring end to end on a real board.
    """

    def test_cluster_keys_absent_until_routing(self) -> None:
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env)
            obs, _ = wrapper.reset()
            # No route → no head → nothing is "already connected to me".
            assert obs["_aug"]["cluster_keys"] is None
            obs, *_ = wrapper.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
            assert obs["_aug"]["cluster_keys"] is None
        finally:
            env.close()

    def test_start_origin_is_in_its_own_cluster_and_dropped(self) -> None:
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env)
            wrapper.reset()
            wrapper.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
            start_xy = wrapper._cand_mm[0]
            obs, *_ = wrapper.step(np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64))

            keys = obs["_aug"]["cluster_keys"]
            assert keys, "routing head must report a cluster"
            origin_key = (round(start_xy[0], 3), round(start_xy[1], 3), start_xy[2])
            assert origin_key in keys
            # ...and the pool no longer offers it (zero-length re-target).
            assert origin_key not in [
                (round(x, 3), round(y, 3), l) for x, y, l in wrapper._cand_mm
            ]
        finally:
            env.close()

    def test_filter_off_leaves_cluster_keys_unset(self) -> None:
        _skip_if_no_env()
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        try:
            wrapper = KiCadRLWrapper(env, connectivity_filter=False)
            wrapper.reset()
            wrapper.step(np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64))
            obs, *_ = wrapper.step(np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64))
            assert obs["_aug"]["cluster_keys"] is None
        finally:
            env.close()
