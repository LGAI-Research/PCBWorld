"""Integration tests for PCBWorld (requires C++ kicad_rl_router).

Tests:
1. env_hl: reset, observation structure, action mask, step dispatch
2. end-to-end: net_select → start_route → make_line → finish → net_end cycle
"""

import os

import numpy as np
import pytest

from pcb_world.engine import engine_available

from pcb_world.core.masking import ACT_IDLE, NUM_ACTIONS

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "simple_obstacle_board.kicad_pcb"
)


def _skip_if_no_board():
    if not os.path.exists(BOARD_PATH):
        pytest.skip(f"Board not found: {BOARD_PATH}")


def _skip_if_no_kicad():
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


def _first_selectable_net_id(env):
    """Return a net that still has at least one ratsnest edge."""
    net_ids = sorted({int(edge.net_code) for edge in env._engine.get_ratsnest()})
    if not net_ids:
        pytest.skip("No selectable unrouted nets on board")
    return net_ids[0]


@pytest.fixture
def env():
    """Create PCBWorld instance."""
    _skip_if_no_board()
    _skip_if_no_kicad()
    from pcb_world.core.env import PCBWorld
    e = PCBWorld(board_path=BOARD_PATH, max_steps=50)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. Reset and observation structure
# ---------------------------------------------------------------------------

class TestHLEnvReset:
    """Test environment reset and initial observation."""

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_has_required_keys(self, env):
        obs, _ = env.reset()
        assert "board_static" in obs
        assert "routing_geometry" in obs
        assert "router_head" in obs
        # board_meta fields are now part of board_static
        assert "bbox_x" in obs["board_static"]
        assert "scale" in obs["board_static"]

    def test_board_static_populated(self, env):
        obs, _ = env.reset()
        ctx = obs["board_static"]
        assert "boardlines" in ctx
        assert "nets" in ctx
        assert "obstacles" in ctx
        assert "board_constraints" in ctx
        # Should have at least some nets with pads
        assert len(ctx["nets"]) > 0

    def test_router_head_initial_state(self, env):
        obs, _ = env.reset()
        mode = obs["router_head"]
        assert mode["current_net_phase"] == 0  # NET_SELECT
        assert mode["routing_mode"] == 2  # Walkaround
        assert mode["step"] == 0
        assert mode["is_routing"] == False

    def test_action_mask_initial(self, env):
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (NUM_ACTIONS,)
        # Only net_select should be allowed at start
        assert mask[0] == True   # ACT_NET_SELECT
        assert mask[1] == False  # ACT_START_ROUTE
        assert mask[3] == False  # ACT_MAKE_LINE

    def test_info_has_phase(self, env):
        _, info = env.reset()
        assert info["phase"] == "NET_SELECT"
        assert info["step"] == 0
        assert "track_count" in info
        assert "unrouted_count" in info


# ---------------------------------------------------------------------------
# 2. Action mask validation
# ---------------------------------------------------------------------------

class TestHLEnvActionMask:
    """Test action masking enforcement."""

    def test_invalid_action_returns_penalty(self, env):
        env.reset()
        # Try start_route without selecting net first
        action = {
            "action_type": 1,  # ACT_START_ROUTE — invalid in NET_SELECT
            "x_mm": 0.0, "y_mm": 0.0, "layer": 1, "net_id": 1, "routing_mode": 2,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        expected = -(env._reward_config.step_penalty + env._reward_config.mask_reject_penalty)
        assert reward == pytest.approx(expected)
        assert info["action_success"] == False
        assert info["error"] == "invalid_action"

    def test_action_masks_method(self, env):
        env.reset()
        mask = env.action_masks()
        assert mask.dtype == bool
        assert mask.shape == (NUM_ACTIONS,)
        assert mask[0] == True  # net_select


# ---------------------------------------------------------------------------
# 3. Step dispatch — net_select
# ---------------------------------------------------------------------------

class TestHLEnvNetSelect:
    """Test net_select action."""

    def test_net_select_changes_phase(self, env):
        env.reset()
        net_id = _first_selectable_net_id(env)
        action = {
            "action_type": 0,  # ACT_NET_SELECT
            "x_mm": 0.0, "y_mm": 0.0, "layer": 1,
            "net_id": net_id, "routing_mode": 2,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["action_success"] == True
        assert info["phase"] == "START_ROUTE"
        # Now mask should allow start_route
        mask = env.action_masks()
        assert mask[1] == True  # ACT_START_ROUTE


# ---------------------------------------------------------------------------
# 4. End-to-end routing cycle
# ---------------------------------------------------------------------------

class TestHLEnvE2E:
    """End-to-end test: select net → start route → routing → end."""

    def test_routing_cycle(self, env):
        """Attempt a full routing cycle (may fail at routing, that's ok)."""
        obs, info = env.reset()

        # Step 1: net_select
        net_id = _first_selectable_net_id(env)

        action = {
            "action_type": 0, "x_mm": 0.0, "y_mm": 0.0,
            "layer": 1, "net_id": net_id, "routing_mode": 2,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["phase"] == "START_ROUTE"

        # Step 2: start_route — pick first pad of this net
        pads = env._engine.get_pads()
        net_pads = [p for p in pads if p.net_code == net_id]
        if not net_pads:
            pytest.skip(f"No pads for net {net_id}")

        pad = net_pads[0]
        human_layer = env._engine.layer_map.board_to_human(pad.layer)
        action = {
            "action_type": 1, "x_mm": pad.x_mm, "y_mm": pad.y_mm,
            "layer": human_layer, "net_id": 0, "routing_mode": 2,
        }
        obs, reward, terminated, truncated, info = env.step(action)

        if info["action_success"]:
            assert info["phase"] == "ROUTING"
            # Mask should allow make_line, make_via, finish
            mask = env.action_masks()
            assert mask[3] == True  # ACT_MAKE_LINE
            assert mask[4] == True  # ACT_MAKE_VIA
            assert mask[5] == True  # ACT_FINISH

            # Step 3: try finish (auto-complete)
            action = {
                "action_type": 5, "x_mm": 0.0, "y_mm": 0.0,
                "layer": 1, "net_id": 0, "routing_mode": 2,
            }
            obs, reward, terminated, truncated, info = env.step(action)
            # Whether finish succeeds or fails, phase should be START_ROUTE
            # (either via on_routing_done or sync_with_router)
            assert info["phase"] == "START_ROUTE"
        else:
            # start_route failed (snap to pad failed) — still START_ROUTE
            assert info["phase"] == "START_ROUTE"

    def test_multiple_steps_no_crash(self, env):
        """Run multiple random valid actions without crashing."""
        obs, info = env.reset()

        for _ in range(10):
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                break

            action_type = int(valid_actions[0])
            selectable_nets = sorted(
                {int(edge.net_code) for edge in env._engine.get_ratsnest()}
            )
            net_id = selectable_nets[0] if selectable_nets else 1

            # Get a pad position for coordinate-based actions
            pads = env._engine.get_pads()
            x_mm = pads[0].x_mm if pads else 100.0
            y_mm = pads[0].y_mm if pads else 100.0

            action = {
                "action_type": action_type,
                "x_mm": x_mm, "y_mm": y_mm,
                "layer": 1, "net_id": net_id, "routing_mode": 2,
            }
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Should reach here without any exception
        assert True


# ---------------------------------------------------------------------------
# 5. Masking rule configuration
# ---------------------------------------------------------------------------

class TestHLEnvMaskingConfig:
    """Test that masking rule can be configured."""

    def test_relaxed_rule(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.core.env import PCBWorld
        e = PCBWorld(board_path=BOARD_PATH, max_steps=10, masking_rule="relaxed_phase")
        try:
            obs, _ = e.reset()
            # Select a net first
            selectable_nets = sorted(
                {int(edge.net_code) for edge in e._engine.get_ratsnest()}
            )
            if selectable_nets:
                action = {
                    "action_type": 0, "x_mm": 0.0, "y_mm": 0.0,
                    "layer": 1, "net_id": selectable_nets[0], "routing_mode": 2,
                }
                obs, _, _, _, _ = e.step(action)
                mask = e.action_masks()
                # Relaxed allows net_end even when not fully connected
                assert mask[2] == True  # ACT_NET_END
        finally:
            e.close()


# ---------------------------------------------------------------------------
# 6. Idle action
# ---------------------------------------------------------------------------

class TestHLEnvIdle:
    """Test idle action (no-op fallback for unparseable LLM outputs)."""

    def test_idle_never_masked_in(self, env):
        """Idle should never appear as a valid action in any state."""
        env.reset()
        mask = env.action_masks()
        assert mask[ACT_IDLE] == False

        # Also after net_select
        selectable_nets = sorted(
            {int(edge.net_code) for edge in env._engine.get_ratsnest()}
        )
        if selectable_nets:
            env.step({"action_type": 0, "net_id": selectable_nets[0]})
            mask = env.action_masks()
            assert mask[ACT_IDLE] == False

    def test_idle_does_not_change_obs(self, env):
        """Observation should be identical before and after idle."""
        obs_before, _ = env.reset()
        obs_after, _, _, _, _ = env.step({"action_type": ACT_IDLE})

        # routing_geometry and board_static should be unchanged
        assert obs_before["board_static"] == obs_after["board_static"]
        assert obs_before["routing_geometry"] == obs_after["routing_geometry"]

    def test_idle_increments_step(self, env):
        """Step counter should advance on idle."""
        env.reset()
        assert env._step_count == 0

        env.step({"action_type": ACT_IDLE})
        assert env._step_count == 1

        env.step({"action_type": ACT_IDLE})
        assert env._step_count == 2

    def test_idle_returns_step_penalty(self, env):
        """Idle reward should equal -step_penalty (not the -1.0 invalid penalty)."""
        env.reset()
        _, reward, _, _, _ = env.step({"action_type": ACT_IDLE})
        expected = -env._reward_config.step_penalty
        assert reward == pytest.approx(expected)

    def test_idle_action_success_false(self, env):
        """Idle should report action_success=False and idle=True."""
        env.reset()
        _, _, _, _, info = env.step({"action_type": ACT_IDLE})
        assert info["action_success"] == False
        assert info["idle"] == True

    def test_idle_triggers_truncation(self):
        """Repeated idle should hit max_steps and truncate."""
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.core.env import PCBWorld
        e = PCBWorld(board_path=BOARD_PATH, max_steps=3)
        try:
            e.reset()
            for i in range(2):
                _, _, terminated, truncated, _ = e.step({"action_type": ACT_IDLE})
                assert not truncated

            # 3rd step should truncate
            _, _, terminated, truncated, info = e.step({"action_type": ACT_IDLE})
            assert truncated
            assert not terminated
        finally:
            e.close()

    def test_idle_truncation_populates_terminal_info(self):
        """IDLE-truncated step must populate final_potential and
        TimeLimit.truncated so eval/telemetry don't fall back to NaN/0.

        Regression for the bug where the IDLE early-return path skipped
        the final_potential write at the end of env.step.
        """
        import math
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.core.env import PCBWorld
        e = PCBWorld(board_path=BOARD_PATH, max_steps=2)
        try:
            e.reset()
            e.step({"action_type": ACT_IDLE})
            _, _, terminated, truncated, info = e.step({"action_type": ACT_IDLE})
            assert truncated and not terminated
            assert "final_potential" in info, (
                "IDLE-truncate must set info['final_potential'] for eval/TB"
            )
            assert math.isfinite(float(info["final_potential"]))
            assert info.get("TimeLimit.truncated") is True
            assert "drc_violations" in info
        finally:
            e.close()

    def test_nonidle_truncation_sets_timelimit_flag(self):
        """Non-IDLE truncation must also set TimeLimit.truncated so the
        episode_terminated metric counts it as truncation, not termination.
        """
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.core.env import PCBWorld
        e = PCBWorld(board_path=BOARD_PATH, max_steps=1)
        try:
            e.reset()
            net_codes = list(e._board_info.nets.keys())
            assert net_codes, "fixture must have at least one net"
            # First (and only) non-IDLE step pushes step_count to 1 == max_steps.
            _, _, terminated, truncated, info = e.step(
                {"action_type": 0, "net_id": net_codes[0]}
            )
            assert truncated and not terminated
            assert info.get("TimeLimit.truncated") is True
            assert "final_potential" in info
        finally:
            e.close()
