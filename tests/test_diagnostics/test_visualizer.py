"""Tests for StepVisualizer: sidecar generation and episode summary.

Pure Python tests (no C++ dependency).
"""

import json
import os
import tempfile

import numpy as np
import pytest

from tools.diagnostics.visualizer.step_visualizer import StepVisualizer


class TestStepSidecar:
    """Test StepVisualizer.create_step_sidecar()."""

    def _make_info(self, *, track_count=3, unrouted_count=1):
        return {
            "action_name": "move",
            "action_success": True,
            "track_count": track_count,
            "unrouted_count": unrouted_count,
            "router_state": "ROUTING",
            "is_routing": True,
            "reward_info": {"reward": 1.23},
        }

    def _make_prev_info(self):
        return {
            "track_count": 2,
            "unrouted_count": 2,
            "router_state": "ROUTING",
            "is_routing": True,
        }

    def _make_action(self):
        return {"action_id": 1, "x_mm": 1.5, "y_mm": 2.5, "layer": 1, "flag": 0, "int_param": 0}

    def test_basic_structure(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=3,
            action=self._make_action(),
            reward=1.23,
            info=self._make_info(),
        )

        assert sidecar["step"] == 3
        assert "transition" in sidecar
        assert "policy_data" in sidecar
        assert "action_mask" in sidecar  # present but None

    def test_transition_tuple_fields(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=-0.5,
            info=self._make_info(),
            prev_info=self._make_prev_info(),
        )

        t = sidecar["transition"]
        assert t["r_t"] == -0.5
        assert "a_t" in t
        assert t["a_t"]["action_id"] == 1
        assert t["a_t"]["action_name"] == "move"
        assert t["done"] is False
        assert t["terminated"] is False
        assert t["truncated"] is False

    def test_s_t_from_prev_info(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=2,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(track_count=3, unrouted_count=1),
            prev_info=self._make_prev_info(),
            prev_image_path="frames/step_0001.png",
            image_path="frames/step_0002.png",
        )

        t = sidecar["transition"]
        # s_t comes from prev_info
        assert t["s_t"]["track_count"] == 2
        assert t["s_t"]["unrouted_count"] == 2
        assert t["s_t"]["board_image"] == "frames/step_0001.png"
        # s_{t+1} comes from info
        assert t["s_t_plus_1"]["track_count"] == 3
        assert t["s_t_plus_1"]["unrouted_count"] == 1
        assert t["s_t_plus_1"]["board_image"] == "frames/step_0002.png"

    def test_s_t_none_when_no_prev_info(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(),
        )

        assert sidecar["transition"]["s_t"] is None

    def test_s_t_plus_1_board_state(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(),
        )

        s_next = sidecar["transition"]["s_t_plus_1"]
        assert s_next["track_count"] == 3
        assert s_next["unrouted_count"] == 1
        assert s_next["router_state"] == "ROUTING"
        assert s_next["is_routing"] is True

    def test_done_flag(self):
        vis = StepVisualizer()
        info = self._make_info()
        info["terminated"] = True
        sidecar = vis.create_step_sidecar(
            step_num=5,
            action=self._make_action(),
            reward=10.0,
            info=info,
        )

        t = sidecar["transition"]
        assert t["done"] is True
        assert t["terminated"] is True
        assert t["truncated"] is False

    def test_policy_data_none(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(),
        )

        pd = sidecar["policy_data"]
        assert pd["V_s_t"] is None
        assert pd["log_prob"] is None

    def test_policy_data_provided(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(),
            value=0.42,
            log_prob=-1.5,
        )

        pd = sidecar["policy_data"]
        assert pd["V_s_t"] == pytest.approx(0.42)
        assert pd["log_prob"] == pytest.approx(-1.5)

    def test_action_mask_included(self):
        vis = StepVisualizer()
        mask = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(),
            action_mask=mask,
        )

        assert sidecar["action_mask"] == [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_action_mask_none_by_default(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(),
        )

        assert sidecar["action_mask"] is None

    def test_image_path_in_s_t_plus_1(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=0.0,
            info=self._make_info(),
            image_path="frames/step_0001.png",
        )

        assert sidecar["transition"]["s_t_plus_1"]["board_image"] == "frames/step_0001.png"

    def test_json_serializable(self):
        """Sidecar dict must be fully JSON-serializable."""
        vis = StepVisualizer()
        action = {"action_id": np.int64(1), "x_mm": np.float32(1.5), "y_mm": np.float32(2.5),
                  "layer": 1, "flag": 0, "int_param": 0}
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=action,
            reward=0.0,
            info=self._make_info(),
            prev_info=self._make_prev_info(),
            action_mask=np.zeros(18, dtype=np.float32),
        )
        # Must not raise
        result = json.dumps(sidecar)
        assert isinstance(result, str)

    def test_reward_breakdown(self):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action=self._make_action(),
            reward=1.23,
            info=self._make_info(),
        )

        assert sidecar["transition"]["reward_breakdown"] == {"reward": 1.23}


class TestSaveStepSidecar:
    def test_write_file(self, tmp_path):
        vis = StepVisualizer()
        sidecar = vis.create_step_sidecar(
            step_num=1,
            action={"action_id": 0, "x_mm": 0.0, "y_mm": 0.0, "layer": 1, "flag": 0, "int_param": 0},
            reward=1.0,
            info={"track_count": 0, "unrouted_count": 1},
        )
        out_path = str(tmp_path / "frames" / "step_0001.json")
        vis.save_step_sidecar(sidecar, out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["step"] == 1
        assert "transition" in loaded


class TestEpisodeSummary:
    def _make_trajectory(self, tmp_path):
        """Create a minimal .npz trajectory for summary testing."""
        filepath = str(tmp_path / "test_ep.npz")
        np.savez_compressed(
            filepath,
            episode_id=np.array("ep_test123"),
            rewards=np.array([0.1, -0.2, 0.5, 1.0, -0.1], dtype=np.float32),
            actions=np.array([[0, 0, 0, 0, 0, 0],
                              [1, 1.5, 2.5, 0, 0, 0],
                              [1, 2.0, 3.0, 0, 0, 0],
                              [2, 3.0, 5.0, 0, 1, 0],
                              [15, 0, 0, 0, 0, 0]], dtype=np.float32),
            episode_length=np.array(5),
            total_reward=np.array(1.3),
        )
        return filepath

    def test_creates_stats_json(self, tmp_path):
        traj = self._make_trajectory(tmp_path)
        summary_dir = str(tmp_path / "summary")

        vis = StepVisualizer()
        outputs = vis.create_episode_summary(traj, summary_dir)

        assert "stats" in outputs
        assert os.path.exists(outputs["stats"])

        with open(outputs["stats"]) as f:
            stats = json.load(f)
        assert stats["episode_length"] == 5
        assert stats["total_reward"] == pytest.approx(1.3)
        assert "mean_reward" in stats
        assert "min_reward" in stats
        assert "max_reward" in stats

    def test_creates_reward_curve(self, tmp_path):
        traj = self._make_trajectory(tmp_path)
        summary_dir = str(tmp_path / "summary")

        vis = StepVisualizer()
        outputs = vis.create_episode_summary(traj, summary_dir)

        # matplotlib should be available
        assert "reward_curve" in outputs
        assert os.path.exists(outputs["reward_curve"])

    def test_creates_action_distribution(self, tmp_path):
        traj = self._make_trajectory(tmp_path)
        summary_dir = str(tmp_path / "summary")

        vis = StepVisualizer()
        outputs = vis.create_episode_summary(traj, summary_dir)

        assert "action_distribution" in outputs
        assert os.path.exists(outputs["action_distribution"])
