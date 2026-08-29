"""Single source of truth for the routing action set.

This is the ONE file that defines *what actions exist*, their parameter
signatures, and the routing-mode encoding. Everything else derives from here:

- ``masking.py`` (env-side availability) imports the registry / ``ACT_*`` /
  ``NUM_ACTIONS``. Masking stays **default-deny**: a mask starts all-False and
  the masking rule grants each action — so an action defined here is *not*
  selectable until masking permits it.
- ``projection`` (LLM parse/format) imports ``PARSE_SCHEMA`` / mode tables /
  ``FALLBACK_ACTION``.

To add/change an action or its params/mode encoding, edit **only this file**.

Placement note: this lives in ``pcb_world/core`` (not ``pcb_world/vec``) because
``masking`` is an env-layer concern and the wrappers depend *downward* on the
env — putting it beside the wrappers would invert that dependency.

idle (index 6): an LLM-only fallback action emitted when projection fails to
parse a response (see ``FALLBACK_ACTION``). RL never selects it — the masking
rule does not grant idle, so it stays masked. It remains in the registry only
so ``NUM_ACTIONS`` / the action index space are stable across both branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Routing-mode encoding (single table)
# ---------------------------------------------------------------------------

MODE_LETTER_TO_INT: dict[str, int] = {"m": 0, "p": 1, "w": 2}
MODE_INT_TO_LETTER: dict[int, str] = {v: k for k, v in MODE_LETTER_TO_INT.items()}

# Named routing-mode indices — mirror the engine binding's ``krl.MODE_*``
# constants (kicad_rl_router: MODE_MARK_OBSTACLES/MODE_SHOVE/MODE_WALKAROUND),
# derived here from the letter encoding so the pure-Python layer has them without
# importing the C++ module. Walkaround is the safe default fallback for an
# unused/invalid routing_mode slot.
MODE_MARK_OBSTACLES = MODE_LETTER_TO_INT["m"]
MODE_SHOVE = MODE_LETTER_TO_INT["p"]
MODE_WALKAROUND = MODE_LETTER_TO_INT["w"]


def parse_mode(value: str) -> int:
    """Convert a routing mode letter (m/p/w) or numeric string to int."""
    if value in MODE_LETTER_TO_INT:
        return MODE_LETTER_TO_INT[value]
    return int(value)


# ---------------------------------------------------------------------------
# Action registry (canonical order; idle last)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionDef:
    """Definition of a single routing action.

    ``params`` is the list of parameter *names* (kept ``list[str]`` so existing
    consumers like ``action.py`` are unaffected). Per-param type converters for
    LLM parsing live in :data:`PARSE_SCHEMA`, not here.
    """
    name: str
    params: list[str]
    description: str = ""


ACTION_REGISTRY: list[ActionDef] = [
    ActionDef("net_select",  ["net_id"],                        "Select a net for routing"),
    ActionDef("start_route", ["x_mm", "y_mm", "layer"],         "Start routing from a pad point"),
    ActionDef("net_end",     [],                                "Finalize the current net"),
    ActionDef("make_line",   ["x_mm", "y_mm", "routing_mode"],  "Route a line segment to a point"),
    ActionDef("make_via",    ["x_mm", "y_mm", "routing_mode"],  "Route to a point and place a via there"),
    ActionDef("finish",      ["routing_mode"],                  "Finish routing to closest unconnected pad"),
    ActionDef("idle",        [],                                "No-op fallback (LLM parse failure); never granted in RL"),
]

# Derived constants
NUM_ACTIONS = len(ACTION_REGISTRY)
ACTION_NAMES = [a.name for a in ACTION_REGISTRY]
_ACTION_NAME_TO_IDX: dict[str, int] = {a.name: i for i, a in enumerate(ACTION_REGISTRY)}

ACT_NET_SELECT  = _ACTION_NAME_TO_IDX["net_select"]
ACT_START_ROUTE = _ACTION_NAME_TO_IDX["start_route"]
ACT_NET_END     = _ACTION_NAME_TO_IDX["net_end"]
ACT_MAKE_LINE   = _ACTION_NAME_TO_IDX["make_line"]
ACT_MAKE_VIA    = _ACTION_NAME_TO_IDX["make_via"]
ACT_FINISH      = _ACTION_NAME_TO_IDX["finish"]
ACT_IDLE        = _ACTION_NAME_TO_IDX["idle"]


# ---------------------------------------------------------------------------
# LLM parse schema (selectable actions only; idle excluded — it's the fallback)
# ---------------------------------------------------------------------------

PARSE_SCHEMA: dict[str, list[tuple[str, Callable[[str], Any]]]] = {
    "net_select":  [("net_id", int)],
    "start_route": [("x_mm", float), ("y_mm", float), ("layer", int)],
    "net_end":     [],
    "make_line":   [("x_mm", float), ("y_mm", float), ("routing_mode", parse_mode)],
    "make_via":    [("x_mm", float), ("y_mm", float), ("routing_mode", parse_mode)],
    "finish":      [("routing_mode", parse_mode)],
}

# name -> action_type for parsed (selectable) actions. Matches ACT_* for these.
ACTION_TYPE_MAP: dict[str, int] = {name: i for i, name in enumerate(PARSE_SCHEMA)}

# Default values per param name (used when a param is missing during parse).
PARAM_DEFAULTS: dict[str, Any] = {
    "x_mm":         0.0,
    "y_mm":         0.0,
    "layer":        1,      # human layer (1=Top)
    "net_id":       0,
    "routing_mode": 2,      # Walkaround
}

# Fallback action dict used when parsing fails completely. Carries the
# `_parse_invalid` flag so PCBWorld applies its `parse_fail_penalty` directly.
FALLBACK_ACTION: dict[str, Any] = {
    "action_type":    ACT_IDLE,   # idle: no state change
    "_parse_invalid": True,       # signal to gym env
}
