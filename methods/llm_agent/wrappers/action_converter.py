"""LLM action converter — LLM text → PCBWorld action dict.

One half of ``projection.py`` (with ``state_converter``), mirroring
the RL codec ``methods.rl_agent.models.v1.encoding``. Parses ``<action>...</action>``
language into an action dict, with idle fallback for malformed / ambiguous
output. Action schema / type map live in the single source of truth
(``pcb_world.core.action_schema``).

Public: cadagent_projection / extract_think_action / render_action_history /
ACTION_TYPE_MAP.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pcb_world.core.action_schema import (
    PARSE_SCHEMA as _ACTION_SCHEMA,
    ACTION_TYPE_MAP,
    PARAM_DEFAULTS as _PARAM_DEFAULTS,
    FALLBACK_ACTION as _FALLBACK_ACTION,
)

def _parse_action_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse a language-format action from <action>...</action> tags.

    Format: "<action>action_name param1 param2 ...</action>"
    Parameters are positional, matched to _ACTION_SCHEMA in order.

    Returns:
        Parsed action dict with all required keys, or None on failure.
    """
    start_idx = text.find("<action>")
    end_idx = text.find("</action>")
    if start_idx == -1 or end_idx == -1:
        return None

    raw = text[start_idx + len("<action>"):end_idx].strip()
    tokens = raw.split()
    if not tokens:
        return None

    action_name = tokens[0]
    if action_name not in _ACTION_SCHEMA:
        return None

    schema = _ACTION_SCHEMA[action_name]
    params: Dict[str, Any] = {}
    value_tokens = tokens[1:]

    for i, (param_name, converter) in enumerate(schema):
        if i < len(value_tokens):
            try:
                params[param_name] = converter(value_tokens[i])
            except (ValueError, TypeError):
                params[param_name] = _PARAM_DEFAULTS[param_name]
        else:
            params[param_name] = _PARAM_DEFAULTS[param_name]

    return {
        "action_type": ACTION_TYPE_MAP[action_name],
        **params,
    }


def extract_think_action(text: str) -> str:
    """Concatenate <think>...</think><action>...</action> blocks from raw LLM text.

    Returns the concatenation when both blocks are present; otherwise returns
    the original text unchanged.
    """
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    act_start = text.find("<action>")
    act_end = text.find("</action>")
    if -1 in (think_start, think_end, act_start, act_end):
        return text
    think_block = text[think_start:think_end + len("</think>")]
    action_block = text[act_start:act_end + len("</action>")]
    return think_block + action_block


def render_action_history(text: str, valid: bool, action_success: bool) -> str:
    """Render one history entry tagged by validity.

    - valid projection + env dispatched : "[Valid] <think>...</think><action>...</action>"
    - projection failure                : "[Invalid] Idle fallback (Parsing error)"
    - env rejected (mask / dispatch)    : "[Invalid] Idle fallback (Unavailable action)"
    """
    if valid and action_success:
        return f"[Valid] {extract_think_action(text)}"
    if not valid:
        return "[Invalid] Idle fallback (Parsing error)"
    return "[Invalid] Idle fallback (Unavailable action)"


def _has_valid_think(text: str) -> bool:
    """Check that <think>...</think> is well-formed with non-empty content."""
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    if think_start == -1 or think_end == -1:
        return False
    if think_end <= think_start:
        return False
    if not text[think_start + len("<think>"):think_end].strip():
        return False
    return True


def cadagent_projection(
    actions: List[str],
    action_pools: Optional[List[Dict[str, bool]]] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Map raw LLM text actions to PCBWorld action dicts.

    Args:
        actions:      List of raw LLM output strings (one per environment).
        action_pools: Optional per-env action masks {action_name: bool}.
                      Not used for hard filtering here; kept for API parity.

    Returns:
        parsed_actions: List of action dicts compatible with PCBWorld.step().
        valids:         List of int flags (1 = structurally valid, 0 = fallback).
    """
    parsed_actions: List[Dict[str, Any]] = []
    valids: List[int] = [0] * len(actions)

    for i, text in enumerate(actions):
        action_dict = _parse_action_text(text)

        if action_dict is None:
            parsed_actions.append(dict(_FALLBACK_ACTION))
            continue

        # Require well-formed <think>...</think>; fall back to idle when ambiguous
        if not _has_valid_think(text):
            parsed_actions.append(dict(_FALLBACK_ACTION))
            continue

        # Reject outputs containing Chinese characters — fall back to idle
        if re.search(r"[\u4e00-\u9fff]", text):
            parsed_actions.append(dict(_FALLBACK_ACTION))
            continue

        valids[i] = 1
        parsed_actions.append(action_dict)

    return parsed_actions, valids
