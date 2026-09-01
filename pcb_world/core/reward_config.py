"""Reward configuration system with YAML-based config loading.

Mirrors the masking rule pattern: Protocol + YAML loader + registry + auto-discovery.

YAML format (unified potential):
    name: default
    mode: terminal     # "terminal" or "per_step"
    potential:
      completion_bonus: 5.0
      unconnected_penalty: 1.0
      wirelength_penalty: 0.0
      drc_penalty: 0.5
      step_penalty: 0.01
      truncation_mode: none

Legacy YAML format (still supported):
    name: default
    mode: terminal
    step_penalty: 0.01
    step_reward: { ... }
    final_reward: { ... }

Usage:
    cfg = get_reward_config("default")
    reward = cfg.build_reward()
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Protocol

from pcb_world.core.reward import FinalOnlyReward, PotentialReward, RewardFunction


# ---------------------------------------------------------------------------
# Reward config protocol
# ---------------------------------------------------------------------------

VALID_MODES = ("terminal", "per_step")


class RewardConfig(Protocol):
    """Interface for reward configurations."""

    @property
    def name(self) -> str: ...

    @property
    def mode(self) -> str:
        """Reward mode: 'terminal' or 'per_step'."""
        ...

    @property
    def step_penalty(self) -> float:
        """Fixed penalty applied every step (positive value, negated at use)."""
        ...

    @property
    def invalid_action_penalty(self) -> float:
        """Penalty applied when the engine rejects an action that passed the
        action-type mask (e.g. make_line to an unreachable point). Positive
        value, subtracted from step reward when ``info["action_success"]``
        is False."""
        ...

    @property
    def parse_fail_penalty(self) -> float:
        """Penalty applied when the agent's output is unparseable (LLM env
        only). The LLM projection layer flags the action dict with
        ``_parse_invalid=True`` so PCBWorld can apply this penalty as the
        single source of truth for invalid-action handling."""
        ...

    @property
    def mask_reject_penalty(self) -> float:
        """Penalty applied when the agent picks a state-illegal action
        (action-mask rejects before engine dispatch). Positive value,
        subtracted from step reward — cousin of ``invalid_action_penalty``
        (which fires post-dispatch) and ``parse_fail_penalty`` (output
        unparseable). Kept as a separate knob since mask-reject signals a
        higher-level policy error than empty/dispatch-fail."""
        ...

    def build_reward(self) -> PotentialReward:
        """Build unified reward function from config."""
        ...

    def with_overrides(self, **overrides: Any) -> "RewardConfig":
        """Return a copy with non-None potential-field overrides applied.

        ``None`` values mean "keep the config value". Implementations must
        NOT mutate ``self``: instances handed out by
        :func:`get_reward_config` are cached and shared process-wide.
        """
        ...

    # Deprecated — kept for backward compatibility
    def build_step_reward(self) -> RewardFunction:
        """Deprecated: use build_reward() instead."""
        ...

    def build_final_reward(self) -> FinalOnlyReward:
        """Deprecated: use build_reward() instead."""
        ...


# ---------------------------------------------------------------------------
# YAML-based reward config
# ---------------------------------------------------------------------------

class YamlRewardConfig:
    """Reward configuration loaded from a YAML config dict.

    Supports both the new unified ``potential:`` format and the legacy
    ``step_reward:`` / ``final_reward:`` format.  Legacy configs use
    ``final_reward`` weights as the ground truth for the unified potential.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._name: str = config.get("name", "yaml_custom")
        self._mode: str = config.get("mode", "terminal")

        # Migrate legacy mode names (paper terms: terminal / per_step).
        _MODE_MIGRATIONS = {
            "final_only": "terminal",
            "sparse": "terminal",
            "combined": "per_step",
            "step": "per_step",
            "dense": "per_step",
        }
        if self._mode in _MODE_MIGRATIONS:
            new_mode = _MODE_MIGRATIONS[self._mode]
            warnings.warn(
                f"Reward mode {self._mode!r} is deprecated, treating as {new_mode!r}.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._mode = new_mode

        if self._mode not in VALID_MODES:
            raise ValueError(
                f"Invalid reward mode={self._mode!r}. "
                f"Must be one of {VALID_MODES}."
            )

        if "potential" in config:
            # New unified format
            self._potential_cfg: dict[str, Any] = dict(config["potential"])
            self._potential_cfg.pop("use_drc", None)  # deprecated, ignored
            self._step_penalty: float = self._potential_cfg.pop(
                "step_penalty", 0.01,
            )
            self._invalid_action_penalty: float = self._potential_cfg.pop(
                "invalid_action_penalty", 0.01,
            )
            # Penalty for unparseable agent output (LLM env only). Default
            # matches the legacy env_manager parse_reward (-0.2) so existing
            # configs see no behavior change.
            self._parse_fail_penalty: float = self._potential_cfg.pop(
                "parse_fail_penalty", 0.2,
            )
            self._mask_reject_penalty: float = self._potential_cfg.pop(
                "mask_reject_penalty", 0.09,
            )
            self._potential_cfg.setdefault("step_penalty", self._step_penalty)
        else:
            # Legacy format: step_reward + final_reward
            self._step_penalty = config.get("step_penalty", 0.01)
            self._invalid_action_penalty = config.get(
                "invalid_action_penalty", 0.01,
            )
            self._parse_fail_penalty = config.get("parse_fail_penalty", 0.2)
            self._mask_reject_penalty = config.get("mask_reject_penalty", 0.09)
            final_cfg = dict(config.get("final_reward", {}))
            # Build potential config from final_reward weights (ground truth).
            # Legacy use_drc=False is preserved by forcing drc_penalty=0 so the
            # resulting potential has no DRC contribution.
            legacy_use_drc = final_cfg.get("use_drc", False)
            drc_pen = final_cfg.get("drc_penalty", 0.5) if legacy_use_drc else 0.0
            self._potential_cfg = {
                "completion_bonus": final_cfg.get("completion_bonus", 5.0),
                "unconnected_penalty": final_cfg.get("unconnected_penalty", 1.0),
                "wirelength_penalty": final_cfg.get("wirelength_penalty", 0.0),
                "via_penalty": final_cfg.get("via_penalty", 0.0),
                "drc_penalty": drc_pen,
                "step_penalty": self._step_penalty,
                "truncation_mode": final_cfg.get("truncation_mode", "none"),
                "wire_via_emission": final_cfg.get(
                    "wire_via_emission", "per_step",
                ),
            }

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def step_penalty(self) -> float:
        return self._step_penalty

    @property
    def invalid_action_penalty(self) -> float:
        return self._invalid_action_penalty

    @property
    def parse_fail_penalty(self) -> float:
        return self._parse_fail_penalty

    @property
    def mask_reject_penalty(self) -> float:
        return self._mask_reject_penalty

    def build_reward(self) -> PotentialReward:
        """Build a unified PotentialReward from config."""
        return PotentialReward(**self._potential_cfg)

    def with_overrides(self, **overrides: Any) -> "YamlRewardConfig":
        """Return a copy with non-None potential-field overrides applied.

        Keys must be :class:`PotentialReward` constructor arguments — unknown
        keys or invalid values fail loudly at the next :meth:`build_reward`.
        Always copies so the shared, cached instances handed out by
        :func:`get_reward_config` stay pristine.
        """
        applied = {k: v for k, v in overrides.items() if v is not None}
        if not applied:
            return self
        new = copy.copy(self)
        new._potential_cfg = {**self._potential_cfg, **applied}
        if "step_penalty" in applied:
            new._step_penalty = applied["step_penalty"]
        return new

    # -- Deprecated backward-compatible methods --

    def build_step_reward(self) -> RewardFunction:
        """Deprecated: use build_reward() instead."""
        warnings.warn(
            "build_step_reward() is deprecated, use build_reward().",
            DeprecationWarning,
            stacklevel=2,
        )
        return RewardFunction()  # type: ignore[return-value]

    def build_final_reward(self) -> FinalOnlyReward:
        """Deprecated: use build_reward() instead."""
        warnings.warn(
            "build_final_reward() is deprecated, use build_reward().",
            DeprecationWarning,
            stacklevel=2,
        )
        return FinalOnlyReward()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Loading & registry
# ---------------------------------------------------------------------------

_REWARD_CONFIG_CACHE: dict[str, RewardConfig] = {}

_CONFIGS_REWARD_DIR = Path(__file__).resolve().parents[2] / "configs" / "reward"

# Backward-compatible name mapping (old names → new canonical names).
_NAME_COMPAT: dict[str, str] = {
    "shaped_log_all": "drc_only_dense",
    "step_drc": "drc_linear",
    # drc_sparse_promoted was renamed to drc_sparse_promoted_grpo when the
    # PPO variant (drc_sparse_promoted_ppo, truncation_mode=none) was split
    # off — the truncation_mode=full version is GRPO-only because GRPO has
    # no critic bootstrap to absorb terminal Φ. Legacy v51/v56 scripts still
    # reference the bare name; alias keeps them resolving.
    "drc_sparse_promoted": "drc_sparse_promoted_grpo",
}


def register_reward_config(name: str, config: RewardConfig) -> None:
    """Register a reward config instance in the cache."""
    _REWARD_CONFIG_CACHE[name] = config


def load_reward_config(path: str | Path) -> YamlRewardConfig:
    """Load a YamlRewardConfig from a YAML config file.

    Args:
        path: Path to a reward config YAML file.

    Returns:
        Instantiated YamlRewardConfig.
    """
    from configs.loader import load_config

    return YamlRewardConfig(load_config(path))


def get_reward_config(name: str = "default") -> RewardConfig:
    """Get a reward config by name or YAML path.

    Resolution order:
        1. If *name* ends with .yaml/.yml -> load directly from that path.
        2. Apply backward-compatible name mapping.
        3. If *name* is already cached -> return cached instance.
        4. Look for configs/reward/{name}.yaml -> load and cache.
        5. Raise KeyError if not found.
    """
    # 1. Explicit YAML path
    if name.endswith((".yaml", ".yml")):
        return load_reward_config(name)

    # 2. Backward-compatible name mapping
    name = _NAME_COMPAT.get(name, name)

    # 3. Cache hit
    if name in _REWARD_CONFIG_CACHE:
        return _REWARD_CONFIG_CACHE[name]

    # 3. Auto-discover from configs/reward/{name}.yaml
    yaml_path = _CONFIGS_REWARD_DIR / f"{name}.yaml"
    if yaml_path.exists():
        config = load_reward_config(yaml_path)
        _REWARD_CONFIG_CACHE[name] = config
        return config

    raise KeyError(
        f"Unknown reward config '{name}'. "
        f"No cached config and no YAML file found at {yaml_path}."
    )
