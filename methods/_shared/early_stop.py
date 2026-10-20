"""Rollout-side early-stop guards (shared by the RL and LLM drivers).

Extracted from ``methods/rl_agent/rollout/transformer.py`` so the LLM rollout
(``methods/llm_agent/rollout/pcbworld.py``) can reuse the same guards without
importing the RL driver — that module pulls in torch and belongs to a different
method family. The definitions are unchanged; ``transformer.py`` re-imports them
from here, so the RL path keeps the identical behaviour.

``_geometry_signature`` reads the board-state measures PCBWorld stamps on
``valid_*`` steps only (``track_count`` / ``via_count`` / ``wirelength`` /
``ratsnest_reduction``). On a parse-error / mask-rejected step those keys are
absent, the signature is ``None``, and the guards leave the streak untouched
rather than counting the missing measurement either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FinishNoProgressGuard",
    "NoGeometryProgressGuard",
]


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _geometry_signature(info: dict[str, Any]) -> tuple[float, float, float, float] | None:
    vals = (
        _float_or_nan(info.get("track_count")),
        _float_or_nan(info.get("via_count")),
        _float_or_nan(info.get("wirelength")),
        _float_or_nan(info.get("ratsnest_reduction")),
    )
    if all(math.isnan(v) for v in vals):
        return None
    return vals


def _geometry_changed(
    prev: tuple[float, float, float, float] | None,
    cur: tuple[float, float, float, float] | None,
) -> bool:
    if prev is None or cur is None:
        return False
    for a, b in zip(prev, cur):
        if math.isnan(a) or math.isnan(b):
            continue
        if abs(a - b) > 1e-9:
            return True
    return False


@dataclass
class FinishNoProgressGuard:
    """Track consecutive ``finish`` attempts that failed without geometry change."""
    threshold: int = 0
    count: int = 0
    last_geometry: tuple[float, float, float, float] | None = None

    def update(
        self, *, action_name: str, action_class: str, info: dict[str, Any],
    ) -> bool:
        cur = _geometry_signature(info)
        if _geometry_changed(self.last_geometry, cur):
            self.count = 0
        if cur is not None:
            self.last_geometry = cur
        if action_name == "finish" and action_class in {
            "valid_dispatch_fail", "valid_empty",
        }:
            self.count += 1
        return self.threshold > 0 and self.count >= self.threshold


@dataclass
class NoGeometryProgressGuard:
    """Track consecutive steps with no observed geometry metric change."""
    threshold: int = 0
    count: int = 0
    last_geometry: tuple[float, float, float, float] | None = None

    def update(self, *, info: dict[str, Any]) -> bool:
        cur = _geometry_signature(info)
        if cur is None:
            return self.threshold > 0 and self.count >= self.threshold
        if self.last_geometry is None:
            self.last_geometry = cur
            return False
        if _geometry_changed(self.last_geometry, cur):
            self.count = 0
        else:
            self.count += 1
        self.last_geometry = cur
        return self.threshold > 0 and self.count >= self.threshold
