"""Stateless action masking based on observable state conditions.

Determines valid actions from engine state (has_net, is_routing,
net_fully_connected) without a separate phase state machine.
Masking rules are loaded from condition-based YAML configs.

YAML format example (configs/masking/default.yaml):
    name: strict
    actions:
      net_select:
        when: {has_net: false, is_routing: false}
      start_route:
        when: {has_net: true, is_routing: false}
      ...

Usage:
    ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=True)
    mask = build_action_mask(ctx, rule_name="strict")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Action registry — single source of truth lives in action_schema.py.
# Re-imported here so existing `pcb_world.core.masking` consumers are unchanged.
# ---------------------------------------------------------------------------

from pcb_world.core.action_schema import (  # noqa: E402
    ActionDef,
    ACTION_REGISTRY,
    NUM_ACTIONS,
    ACTION_NAMES,
    _ACTION_NAME_TO_IDX,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACT_NET_END,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_FINISH,
    ACT_IDLE,
)


# ---------------------------------------------------------------------------
# Masking context
# ---------------------------------------------------------------------------

@dataclass
class MaskContext:
    """Observable state for masking decisions.

    All fields are derived from engine state — no separate phase tracking.
    """
    has_net: bool = False              # current_net_id is not None
    is_routing: bool = False           # engine.is_routing()
    net_fully_connected: bool = False  # no ratsnest edges for current net
    step_count: int = 0
    max_steps: int = 100
    unrouted_count: int = 0


# ---------------------------------------------------------------------------
# Masking rule protocol
# ---------------------------------------------------------------------------

class MaskingRule(Protocol):
    """Interface for masking rules."""

    def build_mask(self, ctx: MaskContext) -> np.ndarray:
        """Return a boolean array of shape (NUM_ACTIONS,) indicating valid actions."""
        ...


# ---------------------------------------------------------------------------
# YAML condition-based masking rule
# ---------------------------------------------------------------------------

class YamlConditionMask:
    """Masking rule loaded from a condition-based YAML config.

    Each action has a ``when`` dict mapping MaskContext field names
    to required values. An action is enabled when ALL conditions match.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.name: str = config.get("name", "yaml_custom")
        # _action_conditions: [(action_index, {field: required_value})]
        self._action_conditions: list[tuple[int, dict[str, Any]]] = []
        self._parse_actions(config.get("actions", {}))

    def _parse_actions(self, actions_cfg: dict[str, Any]) -> None:
        for action_name, action_def in actions_cfg.items():
            idx = _ACTION_NAME_TO_IDX[action_name]
            conditions = action_def.get("when", {}) if isinstance(action_def, dict) else {}
            self._action_conditions.append((idx, conditions))

    def build_mask(self, ctx: MaskContext) -> np.ndarray:
        """Return a boolean action mask based on condition matching."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for action_idx, conditions in self._action_conditions:
            if all(
                getattr(ctx, field, None) == value
                for field, value in conditions.items()
            ):
                mask[action_idx] = True
        return mask


# ---------------------------------------------------------------------------
# Loading & registry
# ---------------------------------------------------------------------------

_MASKING_RULE_CACHE: dict[str, MaskingRule] = {}

_CONFIGS_MASKING_DIR = Path(__file__).resolve().parents[2] / "configs" / "masking"

# Backward-compatible name mapping (old names → new canonical names).
# All legacy variants (strict*, phase-based) resolve to the current defaults,
# which preserve the previous "strict" semantics.
_NAME_COMPAT: dict[str, str] = {
    "strict": "default",
    "strict_phase": "default",
    "strict_no_finish": "default_no_finish",
    "relaxed_phase": "relaxed",
}


def register_masking_rule(name: str, rule: MaskingRule) -> None:
    """Register a masking rule instance in the cache."""
    _MASKING_RULE_CACHE[name] = rule


def load_masking_rule(path: str | Path) -> YamlConditionMask:
    """Load a YamlConditionMask from a YAML config file.

    Args:
        path: Path to a masking rule YAML file.

    Returns:
        Instantiated YamlConditionMask.
    """
    from configs.loader import load_config

    return YamlConditionMask(load_config(path))


def get_masking_rule(name: str = "strict") -> MaskingRule:
    """Get a masking rule by name or YAML path.

    Resolution order:
        1. If *name* ends with .yaml/.yml -> load directly from that path.
        2. Apply backward-compatible name mapping (e.g. "strict_phase" -> "default").
        3. If *name* is already cached -> return cached instance.
        4. Look for configs/masking/{name}.yaml -> load and cache.
        5. Raise KeyError if not found.
    """
    # 1. Explicit YAML path
    if name.endswith((".yaml", ".yml")):
        return load_masking_rule(name)

    # 2. Backward-compatible name mapping
    name = _NAME_COMPAT.get(name, name)

    # 3. Cache hit
    if name in _MASKING_RULE_CACHE:
        return _MASKING_RULE_CACHE[name]

    # 4. Auto-discover from configs/masking/{name}.yaml
    yaml_path = _CONFIGS_MASKING_DIR / f"{name}.yaml"
    if yaml_path.exists():
        rule = load_masking_rule(yaml_path)
        _MASKING_RULE_CACHE[name] = rule
        return rule

    raise KeyError(
        f"Unknown masking rule '{name}'. "
        f"No cached rule and no YAML file found at {yaml_path}."
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_action_mask(
    ctx: MaskContext,
    rule_name: str = "strict",
) -> np.ndarray:
    """Build a boolean action mask using the specified rule.

    Args:
        ctx: Current masking context (observable state).
        rule_name: Name of the masking rule.

    Returns:
        np.ndarray of shape (NUM_ACTIONS,) with dtype=bool.
    """
    return get_masking_rule(rule_name).build_mask(ctx)
