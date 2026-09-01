"""mask-in-obs equivalence tests — wrapper-embedded act-time masks.

``KiCadRLWrapper`` embeds the act-time mask arrays under ``obs["_masks"]``
on every step/reset (computed in-worker right after ``_refresh_cache``);
``gather_mask_arrays(obs_list=)`` stacks them locally instead of issuing 4
``env_method`` IPC round-trips per rollout step (query fallback for obs
without the payload — stubs, diagnostics). Semantics: the state right
after a step/reset return equals the state at the next act time (nothing
mutates in between), so the embedded arrays must be **bit-identical** to the
legacy queries — and the collected trajectories must be identical too (same
values, different source). Verified here:

1. gather glue parity — obs path == query path, bit-identical incl. padding.
2. real fixture-board parity over a canonical action sequence.
3. PPO collector end-to-end — mask-in-obs on vs off produces a bit-identical
   PPOCollectorOutput under a deterministic scripted policy.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pcb_world.engine import engine_available
import torch

from pcb_world.core.masking import NUM_ACTIONS
from methods.rl_agent.policy.agent import gather_mask_arrays
from methods.rl_agent.training.collect import collect_n_steps_ppo

from tests._mock_obs import make_mock_obs

_FIXTURE_BOARD = os.path.join(
    os.path.dirname(__file__), "fixtures", "simple_routing_board.kicad_pcb",
)


# ===================================================================
# Deterministic scripted env (no C++)
# ===================================================================
class _ScriptedEnv:
    """KiCadRLWrapper stand-in whose trajectory is a pure function of the
    action sequence (``_acc`` folds every action into the state), with
    state-dependent masks so parity is meaningful. ``embed_masks`` toggles
    the ``obs["_masks"]`` payload — everything else is identical."""

    def __init__(self, env_id: int, episode_len: int = 5,
                 embed_masks: bool = False):
        self._env_id = env_id
        self._episode_len = episode_len
        self._embed_masks = embed_masks
        self._episode = 0
        self._step = 0
        self._acc = env_id + 1

    def _make_obs(self) -> dict:
        obs = make_mock_obs(
            n_nets=2, pads_per_net=2, n_tracks=self._step % 3,
            n_ratsnest_per_net=1, is_routing=self._step > 0,
            current_net_phase=1 if self._step > 0 else 0,
        )
        obs["board_static"]["_env_id"] = self._env_id
        obs["_trace"] = {
            "env_id": self._env_id, "episode": self._episode,
            "step": self._step, "acc": self._acc,
        }
        if self._embed_masks:
            obs["_masks"] = {
                "action": self.action_masks(),
                "pointer": self.start_route_pointer_indices(),
                "mode": self.mode_mask(),
                "net_valid": self.net_valid_mask(),
                "offboard": self.offboard_pointer_indices(),
            }
        return obs

    def reset(self, **kwargs):
        self._episode += 1
        self._step = 0
        self._acc = self._env_id + 1 + 1000 * self._episode
        return self._make_obs(), {}

    def step(self, action):
        a = np.asarray(action).reshape(-1)
        self._acc = (self._acc * 31 + int(a[0]) + 7 * int(a[1])
                     + 13 * int(a[2])) % 1000003
        self._step += 1
        reward = float((self._acc % 199) - 99) / 100.0
        at_end = self._step >= self._episode_len
        terminated = at_end and (self._env_id % 2 == 0)
        truncated = at_end and (self._env_id % 2 == 1)
        info = {
            "action_success": (self._acc % 7) != 0,
            "drc_violations": self._acc % 5,
            "final_potential": float(self._acc % 11),
            "wirelength": float(self._acc % 17) + 1.0,
            "via_count": 1 + self._acc % 2,
            "track_count": 2 + self._acc % 4,
            "ratsnest_reduction": float(self._acc % 10) / 10.0,
            "TimeLimit.truncated": truncated,
        }
        return self._make_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[self._acc % NUM_ACTIONS] = True
        mask[(self._acc + 2) % NUM_ACTIONS] = True
        return mask

    def start_route_pointer_indices(self) -> np.ndarray:
        k = self._acc % 3  # variable K per env/step, incl. 0
        return np.arange(k, dtype=np.int64) + (self._acc % 4)

    def mode_mask(self) -> np.ndarray:
        return np.ones(3, dtype=bool)

    def net_valid_mask(self) -> np.ndarray:
        m = np.ones(2, dtype=bool)
        m[self._acc % 2] = (self._acc % 3) != 0
        return m

    def offboard_pointer_indices(self) -> np.ndarray:
        # variable K per env/step incl. 0 — appended to the pointer block
        k = self._acc % 2
        return np.arange(k, dtype=np.int64) + 10 + (self._acc % 3)

    def close(self):
        pass


class _ScriptedPolicy:
    """Deterministic policy — a pure function of each obs alone."""

    policy_net_select = False
    use_critic = True

    def eval(self):
        pass

    def act_and_value(self, obs_list, action_masks=None, pointer_masks=None,
                      mode_mask=None, deterministic=False, **_kw):
        B = len(obs_list)
        actions = torch.zeros(B, 3, dtype=torch.long)
        log_probs = torch.zeros(B)
        values = torch.zeros(B)
        for i, obs in enumerate(obs_list):
            tr = obs["_trace"]
            valid = action_masks[i].nonzero(as_tuple=False).flatten()
            actions[i, 0] = valid[tr["acc"] % len(valid)]
            actions[i, 1] = (tr["acc"] + tr["env_id"]) % 5
            actions[i, 2] = tr["step"] % 3
            log_probs[i] = float(tr["acc"] % 97) / 97.0
            values[i] = float((tr["acc"] + 3) % 89) / 89.0
        return actions, log_probs, values


# ===================================================================
# 1. Mask parity — shared gather glue over embedded masks
# ===================================================================
class TestGatherMaskParity:
    def test_obs_masks_match_query_path(self):
        """gather via obs['_masks'] == gather via per-env method queries."""
        envs = [_ScriptedEnv(i, embed_masks=True) for i in range(4)]
        for e in envs:
            e.reset()
        for s in range(3):
            for e in envs:
                e.step(np.array([s % NUM_ACTIONS, s, s % 3]))
        obs = [e._make_obs() for e in envs]

        for pns in (False, True):
            m_obs = gather_mask_arrays(envs, [0, 1, 2, 3],
                                       policy_net_select=pns, obs_list=obs)
            m_query = gather_mask_arrays(envs, [0, 1, 2, 3],
                                         policy_net_select=pns)
            for got, want, name in zip(
                m_obs, m_query, ("action", "pointer", "mode", "net_valid"),
            ):
                if want is None:
                    assert got is None, name
                    continue
                assert got.dtype == want.dtype, name
                assert np.array_equal(got, want), name

    def test_missing_masks_falls_through(self):
        """obs without '_masks' → legacy query path (no crash, same output)."""
        envs = [_ScriptedEnv(i, embed_masks=False) for i in range(2)]
        obs = [e.reset()[0] for e in envs]
        m_obs = gather_mask_arrays(envs, [0, 1], policy_net_select=False,
                                   obs_list=obs)
        m_query = gather_mask_arrays(envs, [0, 1], policy_net_select=False)
        for got, want in zip(m_obs[:3], m_query[:3]):
            assert np.array_equal(got, want)


    def test_offboard_rides_pointer_block(self):
        """Off-board rows are appended to the same-point rows per env, THEN
        padded — identically on the obs path, the list path and the
        env_method path (stack_cand_block_masks)."""
        envs = [_ScriptedEnv(i, embed_masks=True) for i in range(4)]
        obs = [e.reset()[0] for e in envs]
        for s in range(2):
            obs = [e.step(np.array([s % NUM_ACTIONS, s, s % 3]))[0] for e in envs]

        class _Pool:  # minimal VecBackend.env_method shim
            def env_method(self, name, *a, indices=None, **kw):
                return [getattr(envs[i], name)(*a, **kw) for i in indices]

        idx = [0, 1, 2, 3]
        ptr_obs = gather_mask_arrays(envs, idx, policy_net_select=False,
                                     obs_list=obs)[1]
        ptr_list = gather_mask_arrays(envs, idx, policy_net_select=False)[1]
        ptr_pool = gather_mask_arrays(_Pool(), idx, policy_net_select=False)[1]
        assert ptr_obs.dtype == np.int64
        assert np.array_equal(ptr_obs, ptr_list)
        assert np.array_equal(ptr_obs, ptr_pool)
        assert any(e.offboard_pointer_indices().size for e in envs)  # not vacuous
        for k, e in enumerate(envs):
            want = np.concatenate([e.start_route_pointer_indices(),
                                   e.offboard_pointer_indices()])
            assert np.array_equal(ptr_obs[k, :want.size], want)
            assert (ptr_obs[k, want.size:] == -1).all()


# ===================================================================
# 2. Mask parity — real env (fixture board, C++ router)
# ===================================================================
class TestRealEnvMaskParity:
    def _make_wrapper(self):
        if not os.path.exists(_FIXTURE_BOARD):
            pytest.skip(f"Fixture board not found: {_FIXTURE_BOARD}")
        if not engine_available():   # probe only — no GPL import (import-hygiene)
            pytest.skip("kicad_rl_router not available")
        from pcb_world.core.env import PCBWorld
        from methods.rl_agent.wrappers.adapter import KiCadRLWrapper

        env = PCBWorld(board_path=_FIXTURE_BOARD, max_steps=20)
        return KiCadRLWrapper(env)

    @staticmethod
    def _check_parity(wrapper, obs) -> None:
        m = obs["_masks"]
        for got, want, name in (
            (m["action"], wrapper.action_masks(), "action"),
            (m["pointer"], wrapper.start_route_pointer_indices(), "pointer"),
            (m["mode"], wrapper.mode_mask(), "mode"),
            (m["net_valid"], wrapper.net_valid_mask(), "net_valid"),
            (m["offboard"], wrapper.offboard_pointer_indices(), "offboard"),
        ):
            assert got.dtype == want.dtype, name
            assert np.array_equal(got, want), name

    def test_embedded_masks_bit_identical_over_steps(self):
        from pcb_world.core.masking import (
            ACT_NET_SELECT, ACT_START_ROUTE, ACT_MAKE_LINE, ACT_NET_END,
        )
        wrapper = self._make_wrapper()
        try:
            obs, _ = wrapper.reset()
            self._check_parity(wrapper, obs)
            # Canonical sequence (net_select → start_route → make_line →
            # net_end); parity must hold after every step, including the
            # same-point pointer masking during routing.
            for action in (
                np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64),
                np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64),
                np.array([ACT_MAKE_LINE, 1, 2], dtype=np.int64),
                np.array([ACT_NET_END, -1, -1], dtype=np.int64),
            ):
                obs, _r, terminated, truncated, _info = wrapper.step(action)
                self._check_parity(wrapper, obs)
                if terminated or truncated:
                    break
        finally:
            wrapper.env.close()


# ===================================================================
# 3. PPO collector end-to-end — mask-in-obs on vs off bit-identical
# ===================================================================
class TestPPOCollectorMaskInObs:
    @staticmethod
    def _collect(embed_masks: bool, n_envs: int = 4, n_steps: int = 12):
        envs = [_ScriptedEnv(i, embed_masks=embed_masks) for i in range(n_envs)]
        return collect_n_steps_ppo(
            envs, _ScriptedPolicy(), torch.device("cpu"), n_steps=n_steps,
        )

    def test_bit_identical_output(self):
        a = self._collect(False)
        b = self._collect(True)
        assert len(a.obs_list) == len(b.obs_list)
        for oa, ob in zip(a.obs_list, b.obs_list):
            # obs differ only by the embedded '_masks' payload
            assert oa["_trace"] == ob["_trace"]
        for name in ("actions", "log_probs", "action_masks", "pointer_masks",
                     "mode_masks", "rewards", "raw_rewards", "values",
                     "episode_starts", "terminated_mask", "final_values"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), name
        assert np.array_equal(a.terminal_values, b.terminal_values,
                              equal_nan=True)
        assert a.episode_rewards == b.episode_rewards
        assert a.episode_lengths == b.episode_lengths
        assert a.invalid_action_ratio == b.invalid_action_ratio


# ===================================================================
# 4. Off-board pointer mask — semantics (no C++)
# ===================================================================
class TestOffboardPointerMask:
    """``offboard_pointer_indices()``: exactly the DIRECTIONAL candidates
    outside the board bbox (edges inclusive), never real geometry, and empty
    with the knob off. Wrapper via ``__new__`` + ``_refresh_cache`` (as in
    ``test_refresh_cache_raises_on_empty_pool_with_net``), obs via
    ``make_mock_obs``."""

    @staticmethod
    def _wrapper(obs, *, offboard_mask):
        from types import SimpleNamespace
        from methods.rl_agent.wrappers.adapter import KiCadRLWrapper

        w = object.__new__(KiCadRLWrapper)
        w.env = SimpleNamespace(board_path="mock.kicad_pcb")
        w._offboard_mask = offboard_mask
        w._refresh_cache(obs)
        return w

    @staticmethod
    def _routing_obs():
        from tests._mock_obs import _make_pad

        # 30 x 20 mm board, head at its centre (15, 10). mres8 rungs 25 / 50 mm
        # leave the board in every direction, the 10 mm rung lands ON the
        # top / bottom edge (inclusive -> stays), 5 mm and below stay inside.
        obs = make_mock_obs(n_nets=1, pads_per_net=2, n_ratsnest_per_net=1,
                            is_routing=True, current_net_phase=1,
                            bbox=(0.0, 0.0, 30.0, 20.0))
        # A same-net PAD outside the bbox: real geometry, never masked.
        obs["board_static"]["nets"]["net_1"]["pads"]["pad_out"] = _make_pad(-5.0, -5.0)
        obs["_aug"] = {"directional_candidates": "mres8"}
        return obs

    def test_masks_exactly_the_offboard_directional_cands(self):
        from pcb_world.vec.candidate_pool import CTYPE_DIRECTIONAL

        obs = self._routing_obs()
        w = self._wrapper(obs, offboard_mask=True)
        hx, hy = obs["router_head"]["current_xy"]
        got = w.offboard_pointer_indices()
        assert got.dtype == np.int64
        want = {
            i for i, ((x, y, _l), ct) in enumerate(zip(w.cand_mm_list, w._cand_ctype))
            if ct == CTYPE_DIRECTIONAL and max(abs(x - hx), abs(y - hy)) >= 25.0
        }
        assert len(want) == 16                      # 8 dirs x {25, 50} mm
        assert set(got.tolist()) == want            # 10 mm edge rung NOT masked
        pad_out = [i for i, (x, y, _l) in enumerate(w.cand_mm_list)
                   if (x, y) == (-5.0, -5.0)]
        assert pad_out and pad_out[0] not in set(got.tolist())

    def test_off_by_default(self):
        w = self._wrapper(self._routing_obs(), offboard_mask=False)
        got = w.offboard_pointer_indices()
        assert got.shape == (0,) and got.dtype == np.int64


def test_refresh_cache_raises_on_empty_pool_with_net(tmp_path, monkeypatch):
    """If a net is selected but the candidate pool is empty, _refresh_cache
    must fail immediately with the env context attached, while obs is still
    live in the wrapper. Just before raising, dump_context saves the full
    raw_obs to a .pt file and appends its path as dump=<path> to the message."""
    import pytest
    from types import SimpleNamespace
    from methods.rl_agent.wrappers.adapter import KiCadRLWrapper

    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))
    w = object.__new__(KiCadRLWrapper)          # bypass __init__
    w.env = SimpleNamespace(board_path="dummy.kicad_pcb")
    # net_5 exists but has 0 pads and no geometry/ratsnest -> empty pool
    obs = {
        "router_head": {"current_net": 5, "is_routing": False, "step": 7},
        "board_static": {"copper_layers": 2, "nets": {"net_5": {"pads": {}}}},
        "routing_geometry": {},
        "closed_nets": [3],
    }
    with pytest.raises(RuntimeError, match="empty candidate pool") as exc_info:
        w._refresh_cache(obs)
    dump_path = str(exc_info.value).rsplit("dump=", 1)[1]
    payload = torch.load(dump_path, weights_only=False)
    assert payload["board_path"] == "dummy.kicad_pcb"
    assert payload["raw_obs"]["router_head"]["current_net"] == 5

    # With no net selected, an empty pool is normal (waiting for net_select).
    obs_idle = {
        "router_head": {"current_net": -1, "is_routing": False, "step": 0},
        "board_static": {"copper_layers": 2, "nets": {}},
        "routing_geometry": {},
    }
    w._refresh_cache(obs_idle)
    assert w.cand_mm_list == []

