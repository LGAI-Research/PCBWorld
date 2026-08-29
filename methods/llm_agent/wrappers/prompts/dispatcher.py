"""Cadagent prompt dispatcher.

Each prompt version lives in its own sibling module named ``v<N>.py``
(``v1.py``, ``v2.py``, ...). Versions are auto-discovered at import time —
dropping a new ``v5.py`` next to this file is enough to make it selectable
via ``get_cadagent_prompts("v5")``.

Public API:
    CadagentPromptBundle      — dataclass with the prompt symbols for one version
    get_cadagent_prompts(v)   — load a bundle for version v (e.g. "v1", "v2")
    CADAGENT_PROMPT_VERSIONS  — tuple of discovered version strings, numerically sorted

Each versioned module must expose the same six symbols:
    get_state_format_desc(state_format: str) -> str
    CADAGENT_SYSTEM_PROMPT          (str, format-template)
    CADAGENT_USER_PROMPT_NO_HIS     (str, format-template)
    CADAGENT_USER_PROMPT            (str, format-template)
    CADAGENT_TEMPLATE_NO_HIS        (str, format-template)
    CADAGENT_TEMPLATE               (str, format-template)
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict


@dataclass(frozen=True)
class CadagentPromptBundle:
    """Snapshot of the prompt symbols for one prompt version."""

    version: str
    get_state_format_desc: Callable[[str], str]
    CADAGENT_SYSTEM_PROMPT: str
    CADAGENT_USER_PROMPT_NO_HIS: str
    CADAGENT_USER_PROMPT: str
    CADAGENT_TEMPLATE_NO_HIS: str
    CADAGENT_TEMPLATE: str


_VERSION_STEM = re.compile(r"^v(\d+)$")


def _discover_version_modules() -> Dict[str, ModuleType]:
    """Import every sibling ``v<N>.py`` and return them keyed by stem."""
    here = Path(__file__).parent
    found: list[tuple[int, str]] = []
    for path in here.glob("v*.py"):
        m = _VERSION_STEM.match(path.stem)
        if m is not None:
            found.append((int(m.group(1)), path.stem))
    found.sort()
    return {stem: importlib.import_module(f".{stem}", package=__package__)
            for _, stem in found}


_VERSION_MODULES: Dict[str, ModuleType] = _discover_version_modules()
CADAGENT_PROMPT_VERSIONS = tuple(_VERSION_MODULES.keys())


def get_cadagent_prompts(version: str = "v1") -> CadagentPromptBundle:
    """Return the prompt bundle for the given version (case-insensitive).

    Raises:
        ValueError: if ``version`` is not a known prompt version.
    """
    key = (version or "v1").lower()
    if key not in _VERSION_MODULES:
        known = ", ".join(CADAGENT_PROMPT_VERSIONS)
        raise ValueError(
            f"Unknown cadagent prompt version {version!r}; choose from: {known}"
        )
    mod = _VERSION_MODULES[key]
    return CadagentPromptBundle(
        version=key,
        get_state_format_desc=mod.get_state_format_desc,
        CADAGENT_SYSTEM_PROMPT=mod.CADAGENT_SYSTEM_PROMPT,
        CADAGENT_USER_PROMPT_NO_HIS=mod.CADAGENT_USER_PROMPT_NO_HIS,
        CADAGENT_USER_PROMPT=mod.CADAGENT_USER_PROMPT,
        CADAGENT_TEMPLATE_NO_HIS=mod.CADAGENT_TEMPLATE_NO_HIS,
        CADAGENT_TEMPLATE=mod.CADAGENT_TEMPLATE,
    )


__all__ = [
    "CadagentPromptBundle",
    "CADAGENT_PROMPT_VERSIONS",
    "get_cadagent_prompts",
]
