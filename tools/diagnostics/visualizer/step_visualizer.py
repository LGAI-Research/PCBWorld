"""Step-level MDP sidecar data generation and episode summary visualization."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def plot_reward_curve(ax, steps, rewards) -> None:
    """Draw the shared step-reward + cumulative-reward line pair on ``ax``.

    Used by both :meth:`StepVisualizer.create_episode_summary` and
    ``episode_dashboard.generate_episode_dashboard`` (Chart 1).
    """
    ax.plot(steps, rewards, "b-", linewidth=1.5, label="Step reward")
    ax.plot(steps, np.cumsum(rewards), "r--", linewidth=1.0, label="Cumulative")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)


class StepVisualizer:
    """Generates step-level MDP sidecar JSON and episode summary charts.

    The sidecar JSON captures all MDP information that cannot be shown
    in the KiCad board image: reward breakdown, action details, action
    mask, policy data, and board state metrics.
    """

    def create_step_sidecar(
        self,
        step_num: int,
        action: dict[str, Any],
        reward: float,
        info: dict[str, Any],
        *,
        prev_info: dict[str, Any] | None = None,
        prev_image_path: str | None = None,
        image_path: str | None = None,
        action_mask: np.ndarray | None = None,
        value: float | None = None,
        log_prob: float | None = None,
        board_bounds: dict[str, float] | None = None,
        obs_summary: dict[str, Any] | None = None,
    ) -> dict:
        """Create a JSON-serializable step sidecar dict.

        The sidecar encodes a full MDP transition tuple
        (s_t, a_t, r_t, s_{t+1}, done) so that every step file is
        self-contained.

        Args:
            step_num: Current step number.
            action: Action dict sent to env.step().
            reward: Reward received.
            info: Info dict from env.step() (used for s_{t+1}).
            prev_info: Info dict from the *previous* step or reset
                (used for s_t). When ``None`` the s_t block is omitted.
            prev_image_path: Relative path to the s_t frame PNG.
            image_path: Relative path to the s_{t+1} frame PNG.
            action_mask: Boolean action mask array.
            value: V(s_t) from critic (optional).
            log_prob: log pi(a|s_t) (optional).
            board_bounds: Board bounding box for coordinate mapping.
                ``{"bbox_x", "bbox_y", "bbox_w", "bbox_h", "scale"}``.
            obs_summary: Condensed observation data for visualization.
                See :func:`extract_obs_summary`.

        Returns:
            Dict ready for json.dump().
        """
        # Extract reward breakdown from info if available
        reward_info = info.get("reward_info", {})
        reward_breakdown = {}
        if isinstance(reward_info, dict):
            reward_breakdown = {
                k: float(v) for k, v in reward_info.items()
                if isinstance(v, (int, float))
            }

        # -- s_t (before action) ------------------------------------------
        s_t: dict[str, Any] | None = None
        if prev_info is not None:
            s_t = self._extract_board_state(prev_info)
            if prev_image_path is not None:
                s_t["board_image"] = prev_image_path

        # -- s_{t+1} (after action) ---------------------------------------
        s_t_plus_1 = self._extract_board_state(info)
        if image_path is not None:
            s_t_plus_1["board_image"] = image_path

        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))

        sidecar: dict[str, Any] = {
            "step": step_num,
            "transition": {
                "s_t": s_t,
                "a_t": self._serialize_action(action, info),
                "r_t": float(reward),
                "reward_breakdown": reward_breakdown,
                "s_t_plus_1": s_t_plus_1,
                "done": terminated or truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            "action_mask": (
                [int(x) for x in action_mask.flatten()]
                if action_mask is not None
                else None
            ),
            "policy_data": {
                "V_s_t": float(value) if value is not None else None,
                "log_prob": float(log_prob) if log_prob is not None else None,
            },
        }

        # Optional enrichment fields (omitted when None for backward compat)
        if board_bounds is not None:
            sidecar["board_bounds"] = board_bounds
        if obs_summary is not None:
            sidecar["obs_summary"] = obs_summary

        return sidecar

    def save_step_sidecar(self, sidecar: dict, output_path: str) -> str:
        """Write step sidecar dict to a JSON file.

        Args:
            sidecar: Dict from create_step_sidecar().
            output_path: File path to write.

        Returns:
            The output_path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        return output_path

    def create_episode_summary(
        self,
        trajectory_path: str,
        output_dir: str,
    ) -> dict[str, str]:
        """Generate episode summary charts and statistics.

        Creates:
        - reward_curve.png: Reward over steps (matplotlib)
        - action_distribution.png: Action ID histogram
        - episode_stats.json: Aggregate statistics

        Args:
            trajectory_path: Path to .npz trajectory file.
            output_dir: Directory to write summary files.

        Returns:
            Dict mapping file type to output path.
        """
        os.makedirs(output_dir, exist_ok=True)
        data = np.load(trajectory_path, allow_pickle=True)

        rewards = data["rewards"]
        actions = data["actions"]
        episode_length = int(data["episode_length"])
        total_reward = float(data["total_reward"])

        outputs = {}

        # Episode statistics JSON
        stats = {
            "episode_id": str(data["episode_id"]),
            "episode_length": episode_length,
            "total_reward": total_reward,
            "mean_reward": float(rewards.mean()) if len(rewards) > 0 else 0.0,
            "min_reward": float(rewards.min()) if len(rewards) > 0 else 0.0,
            "max_reward": float(rewards.max()) if len(rewards) > 0 else 0.0,
            "std_reward": float(rewards.std()) if len(rewards) > 0 else 0.0,
        }
        stats_path = os.path.join(output_dir, "episode_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        outputs["stats"] = stats_path

        # Generate matplotlib charts
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Reward curve
            fig, ax = plt.subplots(figsize=(10, 4))
            steps = np.arange(1, len(rewards) + 1)
            plot_reward_curve(ax, steps, rewards)
            ax.set_title(f"Reward Curve (total={total_reward:.2f})")
            ax.legend()
            reward_path = os.path.join(output_dir, "reward_curve.png")
            fig.savefig(reward_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            outputs["reward_curve"] = reward_path

            # Action distribution
            if actions.ndim >= 1:
                # First column is action_id for Dict action spaces
                action_ids = actions[:, 0].astype(int) if actions.ndim == 2 else actions.astype(int)
                fig, ax = plt.subplots(figsize=(10, 4))
                unique, counts = np.unique(action_ids, return_counts=True)
                ax.bar(unique, counts, color="steelblue")
                ax.set_xlabel("Action ID")
                ax.set_ylabel("Count")
                ax.set_title("Action Distribution")
                ax.set_xticks(unique)
                ax.grid(True, alpha=0.3, axis="y")
                action_path = os.path.join(output_dir, "action_distribution.png")
                fig.savefig(action_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                outputs["action_distribution"] = action_path

        except ImportError:
            pass  # matplotlib not available

        return outputs

    @staticmethod
    def _extract_board_state(info: dict[str, Any]) -> dict[str, Any]:
        """Extract board-state fields from an info dict."""
        return {
            "track_count": info.get("track_count", 0),
            "unrouted_count": info.get("unrouted_count", 0),
            "router_state": info.get("router_state", "UNKNOWN"),
            "is_routing": info.get("is_routing", False),
            "is_dragging": info.get("is_dragging", False),
        }

    @staticmethod
    def _serialize_action(action: dict[str, Any], info: dict[str, Any]) -> dict:
        """Make action dict JSON-serializable."""
        result = {}
        for k, v in action.items():
            if isinstance(v, (np.integer, np.floating)):
                result[k] = v.item()
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        # Add action name from info if available
        if "action_name" in info:
            result["action_name"] = info["action_name"]
        return result


def extract_obs_summary(obs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Extract a condensed observation summary for sidecar enrichment.

    Converts key observation tensor fields to JSON-serializable lists.
    This is a standalone function (not a method) so it can be used
    without instantiating StepVisualizer.

    Args:
        obs: Observation dict from env.step() or env.reset().
            Expected keys: ``route_head``, ``routing_target``,
            ``active_net``, ``router_meta``, ``drc_count``,
            ``drc_violations``.  Missing keys are silently skipped.

    Returns:
        Dict with list-valued fields ready for JSON serialization.
    """
    summary: dict[str, Any] = {}

    for key in ("route_head", "routing_target", "router_meta"):
        val = obs.get(key)
        if val is not None:
            summary[key] = val.tolist() if isinstance(val, np.ndarray) else list(val)

    for key in ("active_net", "drc_count"):
        val = obs.get(key)
        if val is not None:
            if isinstance(val, np.ndarray):
                summary[key] = float(val.item()) if val.size == 1 else val.tolist()
            else:
                summary[key] = float(val)

    drc = obs.get("drc_violations")
    if drc is not None:
        if isinstance(drc, np.ndarray):
            # Only include non-zero rows (active violations)
            mask = np.any(drc != 0, axis=-1) if drc.ndim == 2 else drc != 0
            summary["drc_violations"] = drc[mask].tolist() if mask.any() else []
        else:
            summary["drc_violations"] = list(drc)

    return summary
