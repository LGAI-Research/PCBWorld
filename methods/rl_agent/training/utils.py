"""RL training helpers — small support utilities for the decoder-policy trainers.

These are support utilities (reward-scale stabilizers, value-fit diagnostic,
device resolution), not the training skeleton (rollout collection lives in
:mod:`methods.rl_agent.training.collect`, advantage + update in
:mod:`methods.rl_agent.algorithms`).
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Running reward std (GRPO advantage scale stabilizer)
# ---------------------------------------------------------------------------
class RunningRewardStd:
    """Welford's online running variance tracker for episode returns.

    Provides a stable advantage scale across iterations. Only tracks
    variance/std — mean is NOT subtracted (GRPO uses ``group_mean`` as the
    per-iteration baseline).
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch_mean = float(np.mean(batch))
        batch_var = float(np.var(batch))
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))


# ---------------------------------------------------------------------------
# Reward normalizer (PPO — discounted-return scaling)
# ---------------------------------------------------------------------------
class RewardNormalizer:
    """Reward normalization for PPO, following the VecNormalize scheme.

    Implements the reward pipeline of Stable-Baselines3's ``VecNormalize``:

    * Maintains a per-env discounted return ``R_t = gamma * R_{t-1} + r_t``
      that resets to 0 after a terminal/truncated step; the reset happens
      AFTER the running stats have already absorbed the step that ended
      the episode.
    * Tracks running variance of those returns via Welford's algorithm
      (mean is NOT subtracted — rewards are divided by std only).
    * ``normalize(r) = clip(r / sqrt(var + eps), -clip, +clip)`` returns
      a per-step scaled reward.

    Designed to be called once per env step from the PPO collector.

    Args:
        n_envs: number of parallel envs whose returns are tracked.
        gamma: discount factor used to roll the returns forward (must
            match the trainer's ``gamma``).
        clip: symmetric clipping bound after dividing by std.
        epsilon: numerical floor inside the sqrt.
    """

    def __init__(
        self,
        n_envs: int,
        *,
        gamma: float = 0.99,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self._returns = np.zeros(n_envs, dtype=np.float64)
        # Welford running stats over the discounted-return signal.
        self._mean = 0.0
        self._var = 1.0
        self._count = 1e-4

    @property
    def std(self) -> float:
        return float(np.sqrt(self._var + self.epsilon))

    def state_dict(self) -> dict:
        """Running stats for checkpointing.

        Per-env ``_returns`` are transient (n_envs may differ on resume) and
        are NOT saved; they restart at 0 on load.
        """
        return {"mean": self._mean, "var": self._var, "count": self._count}

    def load_state_dict(self, state: dict) -> None:
        self._mean = float(state["mean"])
        self._var = float(state["var"])
        self._count = float(state["count"])

    def normalize_step(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Update stats and return the normalized rewards for this step.

        Args:
            rewards: ``(n_envs,)`` raw rewards from the env.
            dones: ``(n_envs,)`` bool — True if the episode ended at this
                step (terminated OR truncated).

        Returns:
            ``(n_envs,)`` float32 normalized rewards.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)

        # 1. Update returns (discounted sum so far, including this step).
        self._returns = self._returns * self.gamma + rewards

        # 2. Welford update over the just-updated returns vector.
        self._update_running(self._returns)

        # 3. Compute normalized rewards (BEFORE resetting returns — the
        #    order is stats first, normalize next, then reset).
        normalized = np.clip(
            rewards / self.std, -self.clip, self.clip,
        ).astype(np.float32)

        # 4. Reset per-env returns at episode boundaries.
        if dones.any():
            self._returns[dones] = 0.0

        return normalized

    def _update_running(self, batch: np.ndarray) -> None:
        batch_mean = float(np.mean(batch))
        batch_var = float(np.var(batch))
        batch_count = len(batch)

        delta = batch_mean - self._mean
        total = self._count + batch_count
        self._mean += delta * batch_count / total
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        self._var = (
            m_a + m_b + delta ** 2 * self._count * batch_count / total
        ) / total
        self._count = total


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance — diagnostic for value-function fit.

    Returns ``1 - Var(y_true - y_pred) / Var(y_true)`` (higher is better).
    Returns ``0.0`` if ``Var(y_true) == 0``.
    """
    var_y = float(np.var(y_true))
    if var_y == 0.0:
        return 0.0
    return 1.0 - float(np.var(y_true - y_pred)) / var_y


def auto_device(spec: str = "auto") -> torch.device:
    """Resolve a device string ('auto'/'cpu'/'cuda'/'mps')."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = [
    "RunningRewardStd",
    "RewardNormalizer",
    "explained_variance",
    "auto_device",
]
