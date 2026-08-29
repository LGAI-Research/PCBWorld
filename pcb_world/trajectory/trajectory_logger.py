"""Trajectory logger for episode recording.

Stores initial_state + action_history for ~100x storage efficiency
compared to full state snapshots at every step.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """Record of a single environment step."""
    action: Any  # Action taken (will be converted to numpy for storage)
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)
    # SB3 policy data (optional, populated by MDPLoggingCallback)
    value: float | None = None       # V(s_t)
    log_prob: float | None = None    # log pi(a|s_t)


class TrajectoryLogger:
    """Logs environment trajectories for offline replay and analysis.

    Usage:
        logger = TrajectoryLogger(output_dir="datasets/trajectories")
        logger.on_reset(board_path="test/fixtures/board.kicad_pcb", initial_obs=obs)
        for step in episode:
            logger.on_step(action, obs, reward, terminated, truncated, info)
        logger.on_episode_end(final_info={})

    Output format: {episode_id}.npz containing:
        - board_path: str
        - actions: np.ndarray of shape (T, action_dim)
        - rewards: np.ndarray of shape (T,)
        - terminated: np.ndarray of shape (T,) bool
        - truncated: np.ndarray of shape (T,) bool
        - initial_obs: dict of np.ndarrays (first observation)
        - episode_id: str
        - timestamp: float
        - total_reward: float
        - episode_length: int
    """

    def __init__(self, output_dir: str = "datasets/trajectories", prefix: str = "ep"):
        self.output_dir = output_dir
        self.prefix = prefix
        os.makedirs(output_dir, exist_ok=True)
        self._reset_buffer()

    def _reset_buffer(self):
        self._board_path = None
        self._initial_obs = None
        self._steps = []
        self._episode_id = None
        self._start_time = None

    def on_reset(self, board_path: str, initial_obs: dict | np.ndarray):
        """Called at environment reset. Starts a new episode recording."""
        self._reset_buffer()
        self._board_path = board_path
        self._initial_obs = initial_obs if isinstance(initial_obs, dict) else {"obs": initial_obs}
        self._episode_id = f"{self.prefix}_{uuid.uuid4().hex[:8]}"
        self._start_time = time.time()

    def on_step(self, action, obs, reward: float, terminated: bool, truncated: bool, info: dict = None,
                *, value: float | None = None, log_prob: float | None = None):
        """Called after each environment step.

        Args:
            action: Action taken.
            obs: Observation after step.
            reward: Reward received.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Info dict from env.step().
            value: V(s_t) from SB3 critic (optional).
            log_prob: log pi(a|s_t) from SB3 policy (optional).
        """
        self._steps.append(StepRecord(
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info or {},
            value=value,
            log_prob=log_prob,
        ))

    def on_episode_end(self, final_info: dict = None) -> str:
        """Called when episode ends. Saves trajectory to disk.

        Returns path to saved .npz file.
        """
        if not self._steps:
            return None

        # Convert to numpy arrays
        actions = np.array([self._to_array(s.action) for s in self._steps])
        rewards = np.array([s.reward for s in self._steps], dtype=np.float32)
        terminated = np.array([s.terminated for s in self._steps], dtype=bool)
        truncated = np.array([s.truncated for s in self._steps], dtype=bool)

        # Build save dict
        save_dict = {
            "board_path": np.array(self._board_path),
            "actions": actions,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "episode_id": np.array(self._episode_id),
            "timestamp": np.array(self._start_time),
            "total_reward": np.array(rewards.sum()),
            "episode_length": np.array(len(self._steps)),
        }

        # Save initial observation
        if isinstance(self._initial_obs, dict):
            for key, val in self._initial_obs.items():
                save_dict[f"initial_obs_{key}"] = np.asarray(val)
        else:
            save_dict["initial_obs"] = np.asarray(self._initial_obs)

        # Save policy data if any step has it (backward compatible)
        has_policy_data = any(s.value is not None for s in self._steps)
        if has_policy_data:
            save_dict["values"] = np.array(
                [s.value if s.value is not None else np.nan for s in self._steps],
                dtype=np.float32,
            )
            save_dict["log_probs"] = np.array(
                [s.log_prob if s.log_prob is not None else np.nan for s in self._steps],
                dtype=np.float32,
            )

        # Save metadata from final_info
        if final_info:
            for key, val in final_info.items():
                try:
                    save_dict[f"final_{key}"] = np.asarray(val)
                except (ValueError, TypeError):
                    pass  # Skip non-serializable info

        # Write file
        filepath = os.path.join(self.output_dir, f"{self._episode_id}.npz")
        np.savez_compressed(filepath, **save_dict)

        # Episode metadata JSON sidecar
        meta_path = filepath.replace(".npz", "_meta.json")
        meta = {
            "episode_id": self._episode_id,
            "board_path": self._board_path,
            "episode_length": len(self._steps),
            "total_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()) if len(rewards) > 0 else 0.0,
            "terminated": bool(terminated[-1]) if len(terminated) > 0 else False,
            "truncated": bool(truncated[-1]) if len(truncated) > 0 else False,
            "timestamp": self._start_time,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._reset_buffer()
        return filepath

    @staticmethod
    def _to_array(action) -> np.ndarray:
        """Convert action to numpy array."""
        if isinstance(action, np.ndarray):
            return action.flatten()
        if isinstance(action, dict):
            # For Dict action spaces, flatten all values
            parts = []
            for key in sorted(action.keys()):
                val = np.asarray(action[key]).flatten()
                parts.append(val)
            return np.concatenate(parts)
        return np.atleast_1d(np.asarray(action))

    @staticmethod
    def load(filepath: str) -> dict:
        """Load a saved trajectory file.

        Returns dict with all stored arrays.
        """
        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.files}

    @property
    def is_recording(self) -> bool:
        """True if currently recording an episode."""
        return self._episode_id is not None
