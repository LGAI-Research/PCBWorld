"""PCBWorld prompt dispatcher.

``latest.py`` is the prompt that is edited in place. The frozen versions live in
``legacy/v<N>.py`` (``v1`` … ``v7``) and are never edited, so their output stays
byte-identical (golden test); they are auto-discovered at import time — freezing
``latest`` as ``legacy/v8.py`` makes ``get_pcbworld_prompts("v8")`` selectable.

Public API:
    PCBWorldPromptBundle      — dataclass with the prompt symbols for one version
    get_pcbworld_prompts(v)   — load a bundle for version v (``"latest"``, ``"v1"`` …)
    PCBWORLD_PROMPT_VERSIONS  — discovered legacy versions (numerically sorted), then ``latest``
    LATEST                    — the name of the editable version
    check_prompt_masking(v, m)— raise when prompt v and masking rule m disagree on the actions

Each version module exposes the same six symbols:
    get_state_format_desc(state_format: str) -> str
    PCBWORLD_SYSTEM_PROMPT          (str, format-template)
    PCBWORLD_USER_PROMPT_NO_HIS     (str, format-template)
    PCBWORLD_USER_PROMPT            (str, format-template)
    PCBWORLD_TEMPLATE_NO_HIS        (str, format-template)
    PCBWORLD_TEMPLATE               (str, format-template)
plus ``ACTIONS`` (the routing actions it is written for) — except v1–v7, which
predate it and are listed in ``_PRE_ACTIONS_VERSIONS``.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, FrozenSet

from pcb_world.core.action_schema import ACT_FINISH, ACT_IDLE, ACTION_NAMES

LATEST = "latest"

_ROUTING_ACTIONS = tuple(ACTION_NAMES[:ACT_IDLE])   # every action but the LLM-only idle
# Routing actions of the versions written before modules declared ACTIONS (their files
# stay byte-identical, so the sets live here): v1–v6 ran with finish-granting masking
# rules (strict / relaxed); v7 was written for rules without finish (its header).
_PRE_ACTIONS_VERSIONS: Dict[str, tuple] = {
    **{f"v{n}": _ROUTING_ACTIONS for n in range(1, 7)},
    "v7": tuple(ACTION_NAMES[:ACT_FINISH]),
}


@dataclass(frozen=True)
class PCBWorldPromptBundle:
    """Snapshot of the prompt symbols for one prompt version."""

    version: str
    actions: FrozenSet[str]
    get_state_format_desc: Callable[[str], str]
    PCBWORLD_SYSTEM_PROMPT: str
    PCBWORLD_USER_PROMPT_NO_HIS: str
    PCBWORLD_USER_PROMPT: str
    PCBWORLD_TEMPLATE_NO_HIS: str
    PCBWORLD_TEMPLATE: str

    @property
    def content_hash(self) -> str:
        """sha256 over everything that shapes the prompt text: the five templates,
        both state-format descriptions and the elements this version may render.
        Recorded with each LLM eval run (``manifest.json``)."""
        from methods.llm_agent.wrappers.state_converter import obs_elements

        parts = [
            self.PCBWORLD_SYSTEM_PROMPT, self.PCBWORLD_USER_PROMPT_NO_HIS,
            self.PCBWORLD_USER_PROMPT, self.PCBWORLD_TEMPLATE_NO_HIS, self.PCBWORLD_TEMPLATE,
            self.get_state_format_desc("sexpr"), self.get_state_format_desc("xml"),
            *sorted(obs_elements(self.version)),
        ]
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()


_VERSION_STEM = re.compile(r"^v(\d+)$")


def _discover_version_modules() -> Dict[str, ModuleType]:
    """Import every ``legacy/v<N>.py`` (numerically sorted), then ``latest.py``."""
    here = Path(__file__).parent
    found: list[tuple[int, str]] = []
    for path in (here / "legacy").glob("v*.py"):
        m = _VERSION_STEM.match(path.stem)
        if m is not None:
            found.append((int(m.group(1)), path.stem))
    found.sort()
    modules = {stem: importlib.import_module(f".legacy.{stem}", package=__package__)
               for _, stem in found}
    modules[LATEST] = importlib.import_module(f".{LATEST}", package=__package__)
    return modules


_VERSION_MODULES: Dict[str, ModuleType] = _discover_version_modules()
PCBWORLD_PROMPT_VERSIONS = tuple(_VERSION_MODULES.keys())


def get_pcbworld_prompts(version: str) -> PCBWorldPromptBundle:
    """Return the prompt bundle for the given version (case-insensitive).

    Raises:
        ValueError: if ``version`` is not a known prompt version.
    """
    key = version.lower()
    if key not in _VERSION_MODULES:
        known = ", ".join(PCBWORLD_PROMPT_VERSIONS)
        raise ValueError(
            f"Unknown pcbworld prompt version {version!r}; choose from: {known}"
        )
    mod = _VERSION_MODULES[key]
    actions = _PRE_ACTIONS_VERSIONS[key] if key in _PRE_ACTIONS_VERSIONS else mod.ACTIONS
    return PCBWorldPromptBundle(
        version=key,
        actions=frozenset(actions),
        get_state_format_desc=mod.get_state_format_desc,
        PCBWORLD_SYSTEM_PROMPT=mod.PCBWORLD_SYSTEM_PROMPT,
        PCBWORLD_USER_PROMPT_NO_HIS=mod.PCBWORLD_USER_PROMPT_NO_HIS,
        PCBWORLD_USER_PROMPT=mod.PCBWORLD_USER_PROMPT,
        PCBWORLD_TEMPLATE_NO_HIS=mod.PCBWORLD_TEMPLATE_NO_HIS,
        PCBWORLD_TEMPLATE=mod.PCBWORLD_TEMPLATE,
    )


def check_prompt_masking(prompt_version: str, masking_rule: str) -> None:
    """Raise ``ValueError`` when the masking rule can grant a different set of
    actions than the prompt version is written for (e.g. ``latest`` describes no
    ``finish`` while the ``default`` / ``strict`` rule grants it)."""
    from pcb_world.core.masking import get_masking_rule

    prompt_actions = get_pcbworld_prompts(prompt_version).actions
    rule_actions = get_masking_rule(masking_rule).actions
    if prompt_actions != rule_actions:
        raise ValueError(
            f"prompt_version {prompt_version!r} is written for the actions "
            f"{sorted(prompt_actions)}, but masking_rule {masking_rule!r} can grant "
            f"{sorted(rule_actions)} (only in prompt: "
            f"{sorted(prompt_actions - rule_actions)}, only in rule: "
            f"{sorted(rule_actions - prompt_actions)}) — pick a matching pair "
            "(e.g. latest / v7 with pcbworld_masking or default_no_finish; "
            "v1–v6 with strict / relaxed)"
        )


__all__ = [
    "LATEST",
    "PCBWorldPromptBundle",
    "PCBWORLD_PROMPT_VERSIONS",
    "check_prompt_masking",
    "get_pcbworld_prompts",
]
