"""Shared feedback helpers consumed by both the training-side env_manager
(``KiCadLLMVerlManager``) and the eval driver (``rollout/cadagent.py``).

Two complementary signals are surfaced into the v5 prompt:

  1. **Rejection streak** — a per-env triple (body, first_step, last_step,
     count) tracking the most recent invalid-attempt run (parse fail OR
     action-mask reject). Rendered into the v5 ``{rejected_attempts}`` slot
     when non-empty; cleared on any successful step.

  2. **No-effect marker** — when a step is env-accepted but placed no new
     geometry (``empty_action`` covers make_line / make_via / finish that
     drew nothing) we prefix its history entry with ``[no effect] `` and
     retroactively tag the immediately preceding ``start_route`` (the env
     state machine allows at most one ``start_route`` between two
     successful actions).
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Constants & low-level parsers
# ---------------------------------------------------------------------------

NO_EFFECT_PREFIX = "[no effect] "

_ACTION_BODY_RE = re.compile(r"<action>(.*?)</action>", re.S)
_ACTION_NAME_RE = re.compile(r"<action>\s*(\w+)")


def extract_action_body(text: str) -> str:
    """Return the inner text of the LLM's ``<action>...</action>`` block,
    whitespace-collapsed.

    Caller invariant: invoked only when ``cadagent_projection`` returned
    ``valid=1``, which guarantees the tag exists.
    """
    return " ".join(_ACTION_BODY_RE.search(text).group(1).split())


# ---------------------------------------------------------------------------
# No-effect marker
# ---------------------------------------------------------------------------

def is_no_progress(info: dict) -> bool:
    """True when the env reported an action that was accepted but produced
    no routing progress (``empty_action`` — covers make_line / make_via /
    finish that placed no geometry).
    """
    return bool(info.get("empty_action", False))


def mark_preceding_start_route_no_effect(history: list[dict]) -> None:
    """If the entry immediately before the just-appended one is a
    ``start_route``, prefix it with the no-effect marker. The env state
    machine allows at most one ``start_route`` between two successful
    actions, so a single look-back is sufficient. No-op when the previous
    entry is not a ``start_route`` or is already marked.
    """
    j = len(history) - 2  # the just-appended entry sits at -1
    if j < 0:
        return
    prev = history[j]["action"]
    body = prev[len(NO_EFFECT_PREFIX):] if prev.startswith(NO_EFFECT_PREFIX) else prev
    m = _ACTION_NAME_RE.search(body)
    if m and m.group(1) == "start_route" and not prev.startswith(NO_EFFECT_PREFIX):
        history[j]["action"] = NO_EFFECT_PREFIX + prev


# ---------------------------------------------------------------------------
# Rejection streak (parse fail or mask reject)
# ---------------------------------------------------------------------------
#
# Streak shape: {"body": str, "first_step": int, "last_step": int, "count": int}
# A new body resets the streak; an identical body extends the count and the
# last_step. ``None`` means "no active streak".

def update_rejection_streak(
    streak: Optional[dict], body: str, step_no: int,
) -> dict:
    """Return the updated streak after observing one invalid attempt."""
    if streak is not None and streak["body"] == body:
        streak["last_step"] = step_no
        streak["count"] += 1
        return streak
    return {
        "body": body,
        "first_step": step_no,
        "last_step": step_no,
        "count": 1,
    }


def render_rejection_streak(streak: Optional[dict]) -> str:
    """Render the streak into the v5 prompt slot.

    Returns ``""`` when there is no active streak so the prompt is
    byte-identical to v4 (and v1–v4 templates that simply ignore the kwarg).
    """
    if streak is None:
        return ""
    if streak["count"] == 1:
        line = f"Step {streak['first_step']}: {streak['body']}"
    else:
        line = (
            f"Step {streak['first_step']}-{streak['last_step']}: "
            f"{streak['body']} (rejected ×{streak['count']})"
        )
    return (
        "\n\n# Last rejected attempt (since last valid action — do NOT repeat)\n"
        + line
    )


__all__ = [
    "NO_EFFECT_PREFIX",
    "extract_action_body",
    "is_no_progress",
    "mark_preceding_start_route_no_effect",
    "update_rejection_streak",
    "render_rejection_streak",
]
