"""Stateless action masking based on observable state conditions.

Determines valid actions from engine state (has_net, is_routing,
net_fully_connected) without a separate phase state machine.
Masking rules are loaded from condition-based YAML configs.

YAML format example (configs/masking/pcbworld_masking.yaml; validated by
:class:`MaskingRuleSchema` — every listed action states ``when:``, ``{}`` = always
enabled; unlisted actions are always masked):
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

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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

    #: Every action this rule can ever grant (an LLM prompt must describe exactly these).
    actions: frozenset[str]

    def build_mask(self, ctx: MaskContext) -> np.ndarray:
        """Return a boolean array of shape (NUM_ACTIONS,) indicating valid actions."""
        ...


# ---------------------------------------------------------------------------
# Masking rule schema + YAML condition-based masking rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskingRuleSchema:
    """Schema of one masking rule YAML: action name → its ``when`` conditions
    (:class:`MaskContext` field → required value). Build it with :meth:`from_dict`."""

    name: str
    actions: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_dict(cls, raw: Any, source: str | Path) -> "MaskingRuleSchema":
        """Validate a parsed rule YAML; an unknown, missing or mistyped key — an unknown
        action, a key other than ``when``, or a ``when`` field that is not a
        :class:`MaskContext` field — raises ``ValueError`` naming *source*."""
        from configs.loader import check_mapping

        where = f"masking rule {source}"
        top = check_mapping(raw, _SCHEMA_TYPES, where=where)
        check_mapping(top["actions"], dict.fromkeys(ACTION_NAMES, dict), required=(), where=f"{where} actions")
        for action, action_def in top["actions"].items():
            check_mapping(action_def, {"when": dict}, where=f"{where} actions.{action}")
            check_mapping(action_def["when"], _MASK_CONTEXT_TYPES, required=(), where=f"{where} actions.{action}.when")
        return cls(name=top["name"], actions=MappingProxyType(
            {action: MappingProxyType(dict(d["when"])) for action, d in top["actions"].items()}))


_SCHEMA_TYPES = typing.get_type_hints(MaskingRuleSchema)
_MASK_CONTEXT_TYPES = typing.get_type_hints(MaskContext)


class YamlConditionMask:
    """Masking rule built from a validated :class:`MaskingRuleSchema`. An action is
    enabled when ALL of its ``when`` conditions match the context."""

    def __init__(self, rule: MaskingRuleSchema) -> None:
        self.name: str = rule.name
        self.actions: frozenset[str] = frozenset(rule.actions)
        # _action_conditions: [(action_index, {field: required_value})]
        self._action_conditions: list[tuple[int, dict[str, Any]]] = [
            (_ACTION_NAME_TO_IDX[action], dict(when)) for action, when in rule.actions.items()
        ]

    def build_mask(self, ctx: MaskContext) -> np.ndarray:
        """Return a boolean action mask based on condition matching."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for action_idx, conditions in self._action_conditions:
            if all(
                getattr(ctx, field) == value
                for field, value in conditions.items()
            ):
                mask[action_idx] = True
        return mask


# ---------------------------------------------------------------------------
# Loading & registry
# ---------------------------------------------------------------------------

_MASKING_RULE_CACHE: dict[str, MaskingRule] = {}

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Name lookup order: the live rules in configs/masking first, then the KDD-paper
# rules kept with their reproduction recipes under experiments/kdd/configs/masking
# (paper checkpoints store rule NAMES, not paths). New rules go to configs/masking.
_MASKING_SEARCH_DIRS = (
    _REPO_ROOT / "configs" / "masking",
    _REPO_ROOT / "experiments" / "kdd" / "configs" / "masking",
)

# Backward-compatible name mapping (old names → new canonical names).
# All legacy variants (strict*, phase-based) resolve to the current defaults,
# which preserve the previous "strict" semantics.
_NAME_COMPAT: dict[str, str] = {
    "strict": "default",
    "strict_phase": "default",
    "strict_no_finish": "default_no_finish",
    "relaxed_phase": "relaxed",
    # The shipped rule was renamed when it became the one public masking rule; checkpoints store the name.
    "netend_free_no_finish": "pcbworld_masking",
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
    from configs.loader import load_rule_file

    return YamlConditionMask(MaskingRuleSchema.from_dict(load_rule_file(path, "masking"), path))


def masking_rule_path(name: str) -> Path | None:
    """The YAML file the masking rule *name* resolves to (an explicit path, else ``{name}.yaml``
    under the search dirs after the name mapping); None when no file has that name (a rule
    registered in-process only)."""
    if name.endswith((".yaml", ".yml")):
        return Path(name)
    name = _NAME_COMPAT.get(name, name)
    return next((d / f"{name}.yaml" for d in _MASKING_SEARCH_DIRS if (d / f"{name}.yaml").exists()), None)


def get_masking_rule(name: str = "strict") -> MaskingRule:
    """Get a masking rule by name or YAML path.

    Resolution order:
        1. If *name* ends with .yaml/.yml -> load directly from that path.
        2. Apply backward-compatible name mapping (e.g. "strict_phase" -> "default").
        3. If *name* is already cached -> return cached instance.
        4. Look for {name}.yaml in configs/masking/, then in
           experiments/kdd/configs/masking/ (paper-era rules) -> load and cache.
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

    # 4. Auto-discover {name}.yaml from the search dirs
    yaml_path = masking_rule_path(name)
    if yaml_path is not None:
        rule = load_masking_rule(yaml_path)
        _MASKING_RULE_CACHE[name] = rule
        return rule

    raise KeyError(
        f"Unknown masking rule '{name}'. No cached rule and no {name}.yaml under "
        + " or ".join(str(d) for d in _MASKING_SEARCH_DIRS) + "."
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
