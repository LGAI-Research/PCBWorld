"""Tests for TrajectoryLogger extensions: policy fields and JSON sidecar.

These tests are pure Python (no C++ dependency).
"""

import json
import os
import tempfile

import numpy as np
import pytest

from pcb_world.trajectory.trajectory_logger import StepRecord, TrajectoryLogger


class TestStepRecordExtended:
    """Test StepRecord with value/log_prob fields."""

    def test_default_none(self):
        rec = StepRecord(action=0, reward=1.0, terminated=False, truncated=False)
        assert rec.value is None
        assert rec.log_prob is None

    def test_with_policy_data(self):
        rec = StepRecord(
            action=0, reward=1.0, terminated=False, truncated=False,
            value=0.5, log_prob=-1.2,
        )
        assert rec.value == 0.5
        assert rec.log_prob == -1.2


class TestTrajectoryLoggerExtended:
    """Test TrajectoryLogger with policy data and JSON sidecar."""

    def _make_logger(self, tmp_path):
        return TrajectoryLogger(output_dir=str(tmp_path), prefix="test")

    def _make_obs(self):
        return {"scalars": np.zeros(7, dtype=np.float32)}

    def _make_action(self):
        return {"action_id": 1, "x_mm": 1.0, "y_mm": 2.0, "layer": 1, "flag": 0, "int_param": 0}

    def test_on_step_backward_compatible(self, tmp_path):
        """on_step works without value/log_prob (backward compatibility)."""
        logger = self._make_logger(tmp_path)
        logger.on_reset("test.kicad_pcb", self._make_obs())
        logger.on_step(self._make_action(), self._make_obs(), 1.0, False, False)
        assert len(logger._steps) == 1
        assert logger._steps[0].value is None
        assert logger._steps[0].log_prob is None

    def test_on_step_with_policy_data(self, tmp_path):
        """on_step correctly stores value/log_prob."""
        logger = self._make_logger(tmp_path)
        logger.on_reset("test.kicad_pcb", self._make_obs())
        logger.on_step(
            self._make_action(), self._make_obs(), 1.0, False, False,
            value=0.42, log_prob=-1.5,
        )
        assert logger._steps[0].value == 0.42
        assert logger._steps[0].log_prob == -1.5

    def test_npz_without_policy_data(self, tmp_path):
        """NPZ file does NOT contain values/log_probs when not provided."""
        logger = self._make_logger(tmp_path)
        logger.on_reset("test.kicad_pcb", self._make_obs())
        logger.on_step(self._make_action(), self._make_obs(), 1.0, True, False)
        filepath = logger.on_episode_end()

        data = TrajectoryLogger.load(filepath)
        assert "values" not in data
        assert "log_probs" not in data

    def test_npz_with_policy_data(self, tmp_path):
        """NPZ file contains values/log_probs when provided."""
        logger = self._make_logger(tmp_path)
        logger.on_reset("test.kicad_pcb", self._make_obs())
        logger.on_step(
            self._make_action(), self._make_obs(), 1.0, False, False,
            value=0.5, log_prob=-1.0,
        )
        logger.on_step(
            self._make_action(), self._make_obs(), 2.0, True, False,
            value=0.8, log_prob=-0.5,
        )
        filepath = logger.on_episode_end()

        data = TrajectoryLogger.load(filepath)
        assert "values" in data
        assert "log_probs" in data
        np.testing.assert_allclose(data["values"], [0.5, 0.8], atol=1e-6)
        np.testing.assert_allclose(data["log_probs"], [-1.0, -0.5], atol=1e-6)

    def test_npz_mixed_policy_data(self, tmp_path):
        """Steps with and without policy data produce NaN for missing values."""
        logger = self._make_logger(tmp_path)
        logger.on_reset("test.kicad_pcb", self._make_obs())
        logger.on_step(
            self._make_action(), self._make_obs(), 1.0, False, False,
            value=0.5,
        )
        logger.on_step(
            self._make_action(), self._make_obs(), 2.0, True, False,
        )
        filepath = logger.on_episode_end()

        data = TrajectoryLogger.load(filepath)
        assert "values" in data
        assert data["values"][0] == pytest.approx(0.5)
        assert np.isnan(data["values"][1])
        assert np.isnan(data["log_probs"][0])
        assert np.isnan(data["log_probs"][1])

    def test_json_meta_sidecar_created(self, tmp_path):
        """on_episode_end creates a _meta.json sidecar file."""
        logger = self._make_logger(tmp_path)
        logger.on_reset("test.kicad_pcb", self._make_obs())
        logger.on_step(self._make_action(), self._make_obs(), 1.5, True, False)
        filepath = logger.on_episode_end()

        meta_path = filepath.replace(".npz", "_meta.json")
        assert os.path.exists(meta_path)

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["board_path"] == "test.kicad_pcb"
        assert meta["episode_length"] == 1
        assert meta["total_reward"] == pytest.approx(1.5)
        assert "episode_id" in meta
        assert "timestamp" in meta
        assert isinstance(meta["terminated"], bool)
        assert isinstance(meta["truncated"], bool)

    def test_load_roundtrip(self, tmp_path):
        """Save and load trajectory preserves all fields."""
        logger = self._make_logger(tmp_path)
        logger.on_reset("board.kicad_pcb", self._make_obs())
        for i in range(5):
            logger.on_step(
                self._make_action(), self._make_obs(),
                float(i), i == 4, False,
            )
        filepath = logger.on_episode_end()

        data = TrajectoryLogger.load(filepath)
        assert data["episode_length"] == 5
        assert len(data["rewards"]) == 5
        np.testing.assert_allclose(data["rewards"], [0, 1, 2, 3, 4])
