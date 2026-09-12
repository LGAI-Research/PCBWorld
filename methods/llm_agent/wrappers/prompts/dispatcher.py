"""PCBWorld prompt dispatcher.

Each prompt version lives in its own sibling module named ``v<N>.py``
(``v1.py``, ``v2.py``, ...). Versions are auto-discovered at import time —
dropping a new ``v5.py`` next to this file is enough to make it selectable
via ``get_pcbworld_prompts("v5")``.

Public API:
    PCBWorldPromptBundle      — dataclass with the prompt symbols for one version
    get_pcbworld_prompts(v)   — load a bundle for version v (e.g. "v1", "v2")
    PCBWORLD_PROMPT_VERSIONS  — tuple of discovered version strings, numerically sorted

Each versioned module must expose the same six symbols:
    get_state_format_desc(state_format: str) -> str
    PCBWORLD_SYSTEM_PROMPT          (str, format-template)
    PCBWORLD_USER_PROMPT_NO_HIS     (str, format-template)
    PCBWORLD_USER_PROMPT            (str, format-template)
    PCBWORLD_TEMPLATE_NO_HIS        (str, format-template)
    PCBWORLD_TEMPLATE               (str, format-template)
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict


@dataclass(frozen=True)
class PCBWorldPromptBundle:
    """Snapshot of the prompt symbols for one prompt version."""

    version: str
    get_state_format_desc: Callable[[str], str]
    PCBWORLD_SYSTEM_PROMPT: str
    PCBWORLD_USER_PROMPT_NO_HIS: str
    PCBWORLD_USER_PROMPT: str
    PCBWORLD_TEMPLATE_NO_HIS: str
    PCBWORLD_TEMPLATE: str


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
PCBWORLD_PROMPT_VERSIONS = tuple(_VERSION_MODULES.keys())


def get_pcbworld_prompts(version: str = "v1") -> PCBWorldPromptBundle:
    """Return the prompt bundle for the given version (case-insensitive).

    Raises:
        ValueError: if ``version`` is not a known prompt version.
    """
    key = (version or "v1").lower()
    if key not in _VERSION_MODULES:
        known = ", ".join(PCBWORLD_PROMPT_VERSIONS)
        raise ValueError(
            f"Unknown pcbworld prompt version {version!r}; choose from: {known}"
        )
    mod = _VERSION_MODULES[key]
    return PCBWorldPromptBundle(
        version=key,
        get_state_format_desc=mod.get_state_format_desc,
        PCBWORLD_SYSTEM_PROMPT=mod.PCBWORLD_SYSTEM_PROMPT,
        PCBWORLD_USER_PROMPT_NO_HIS=mod.PCBWORLD_USER_PROMPT_NO_HIS,
        PCBWORLD_USER_PROMPT=mod.PCBWORLD_USER_PROMPT,
        PCBWORLD_TEMPLATE_NO_HIS=mod.PCBWORLD_TEMPLATE_NO_HIS,
        PCBWORLD_TEMPLATE=mod.PCBWORLD_TEMPLATE,
    )


__all__ = [
    "PCBWorldPromptBundle",
    "PCBWORLD_PROMPT_VERSIONS",
    "get_pcbworld_prompts",
]
