"""Config-loading code for the KiCad RL environment (the code half of ``configs/``).

The sibling directories under ``configs/`` are pure data (YAML/JSON); this
subpackage holds everything importable: YAML IO (``load_config`` /
``merge_configs``), the typed schema (``schema``), the dataset/output path
resolver (``paths``) and the CLI (``cli``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_CONFIGS_DIR = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Parsed configuration as a nested dict.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override config into base config.

    Args:
        base: Base configuration dict.
        override: Override values (takes precedence).

    Returns:
        Merged configuration dict.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
