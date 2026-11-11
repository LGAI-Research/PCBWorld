"""PCBWorld prompt dispatcher.

``latest`` is generated per run from ``glossary.yaml`` and the run's EnvConfig
(``latest.build``). The frozen versions live in ``legacy/v<N>.py`` (``v1`` … ``v8``) and
are never edited, so their output stays byte-identical (golden test); they are
auto-discovered at import time — freezing a generated ``latest`` as ``legacy/v9.py``
makes ``get_pcbworld_prompts("v9")`` selectable.

Public API:
    PCBWorldPromptBundle      — dataclass with the prompt symbols for one version
    get_pcbworld_prompts(v, env_config=, guidelines=) — the bundle for version v
                              (``"latest"`` needs the run's EnvConfig)
    PCBWORLD_PROMPT_VERSIONS  — discovered legacy versions (numerically sorted), then ``latest``
    LATEST                    — the name of the editable version
    check_prompt_masking(v, m)— raise when prompt v and masking rule m disagree on the actions

Each legacy module exposes the same six symbols:
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
from typing import Any, Callable, Dict, FrozenSet, List, Tuple

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
    state_formats: Tuple[str, ...]
    PCBWORLD_SYSTEM_PROMPT: str
    PCBWORLD_USER_PROMPT_NO_HIS: str
    PCBWORLD_USER_PROMPT: str
    PCBWORLD_TEMPLATE_NO_HIS: str
    PCBWORLD_TEMPLATE: str
    format_valid_actions: Callable[[List[bool]], str]

    @property
    def content_hash(self) -> str:
        """sha256 over everything that shapes the prompt text: the five templates, the
        state-format descriptions, the elements this version may render and (``latest``)
        the generated action lines.
        Recorded with each LLM eval run (``manifest.json``)."""
        from methods.llm_agent.wrappers.state_converter import obs_elements

        parts = [
            self.PCBWORLD_SYSTEM_PROMPT, self.PCBWORLD_USER_PROMPT_NO_HIS,
            self.PCBWORLD_USER_PROMPT, self.PCBWORLD_TEMPLATE_NO_HIS, self.PCBWORLD_TEMPLATE,
            *(self.get_state_format_desc(f) for f in self.state_formats),
            *sorted(obs_elements(self.version)),
        ]
        if self.version == LATEST:   # its action lines are generated (legacy: one fixed table)
            parts.append(self.format_valid_actions([n in self.actions for n in _ROUTING_ACTIONS]))
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
    return modules


_VERSION_MODULES: Dict[str, ModuleType] = _discover_version_modules()
PCBWORLD_PROMPT_VERSIONS = tuple(_VERSION_MODULES.keys()) + (LATEST,)


def get_pcbworld_prompts(version: str, *, env_config: Any = None,
                         guidelines: bool = True) -> PCBWorldPromptBundle:
    """Return the prompt bundle for the given version (case-insensitive).

    ``latest`` is built from the glossary for ``env_config`` (required) and carries the
    strategy hints when ``guidelines``; a legacy version is fixed text (``env_config``
    unused, ``guidelines`` must stay on — its hints are part of the frozen text).

    Raises:
        ValueError: if ``version`` is not a known prompt version, or the arguments do not
            fit it.
    """
    key = version.lower()
    if key == LATEST:
        from methods.llm_agent.wrappers.prompts import latest

        if env_config is None:
            raise ValueError("prompt_version latest is built from the run's EnvConfig — "
                             "pass env_config")
        p = latest.build(env_config, guidelines=guidelines)

        def desc(state_format: str) -> str:
            if state_format != "sexpr":
                raise ValueError("prompt_version latest describes the sexpr state format only")
            return p.state_format_desc

        return PCBWorldPromptBundle(
            version=key, actions=p.actions, get_state_format_desc=desc,
            state_formats=("sexpr",),
            PCBWORLD_SYSTEM_PROMPT=p.system, PCBWORLD_USER_PROMPT_NO_HIS=p.user_no_his,
            PCBWORLD_USER_PROMPT=p.user, PCBWORLD_TEMPLATE_NO_HIS=p.template_no_his,
            PCBWORLD_TEMPLATE=p.template, format_valid_actions=p.format_valid_actions)
    if key not in _VERSION_MODULES:
        known = ", ".join(PCBWORLD_PROMPT_VERSIONS)
        raise ValueError(
            f"Unknown pcbworld prompt version {version!r}; choose from: {known}"
        )
    if not guidelines:
        raise ValueError(f"prompt_version {key}: the routing guidelines are part of its frozen "
                         "text — guidelines off needs prompt_version latest")
    from methods.llm_agent.wrappers.state_converter import format_valid_actions

    mod = _VERSION_MODULES[key]
    actions = _PRE_ACTIONS_VERSIONS[key] if key in _PRE_ACTIONS_VERSIONS else mod.ACTIONS
    return PCBWorldPromptBundle(
        version=key,
        actions=frozenset(actions),
        get_state_format_desc=mod.get_state_format_desc,
        state_formats=("sexpr", "xml"),
        PCBWORLD_SYSTEM_PROMPT=mod.PCBWORLD_SYSTEM_PROMPT,
        PCBWORLD_USER_PROMPT_NO_HIS=mod.PCBWORLD_USER_PROMPT_NO_HIS,
        PCBWORLD_USER_PROMPT=mod.PCBWORLD_USER_PROMPT,
        PCBWORLD_TEMPLATE_NO_HIS=mod.PCBWORLD_TEMPLATE_NO_HIS,
        PCBWORLD_TEMPLATE=mod.PCBWORLD_TEMPLATE,
        format_valid_actions=format_valid_actions,
    )


def check_prompt_masking(prompt_version: str, masking_rule: str, net_reselect: bool = False) -> None:
    """Raise ``ValueError`` when prompt ``prompt_version`` cannot describe the actions the
    masking rule can grant: a legacy version is written for a fixed set (e.g. v7 has no
    ``finish``); ``latest`` describes any rule the glossary has sentences for."""
    from pcb_world.core.masking import get_masking_rule

    if prompt_version.lower() == LATEST:
        from methods.llm_agent.wrappers.prompts import latest

        latest.action_lines(masking_rule, net_reselect)
        return
    prompt_actions = get_pcbworld_prompts(prompt_version).actions
    rule_actions = get_masking_rule(masking_rule).actions
    if prompt_actions != rule_actions:
        raise ValueError(
            f"prompt_version {prompt_version!r} is written for the actions "
            f"{sorted(prompt_actions)}, but masking_rule {masking_rule!r} can grant "
            f"{sorted(rule_actions)} (only in prompt: "
            f"{sorted(prompt_actions - rule_actions)}, only in rule: "
            f"{sorted(rule_actions - prompt_actions)}) — pick a matching pair "
            "(e.g. v7 / v8 with pcbworld_masking or default_no_finish; "
            "v1–v6 with strict / relaxed; latest with any rule)"
        )


__all__ = [
    "LATEST",
    "PCBWorldPromptBundle",
    "PCBWORLD_PROMPT_VERSIONS",
    "check_prompt_masking",
    "get_pcbworld_prompts",
]
