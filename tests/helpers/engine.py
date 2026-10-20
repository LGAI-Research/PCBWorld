"""``KiCadEngine`` for tests.

``KiCadEngine`` holds no default for its engine knobs; tests build them from the
YAML config (``configs/env.yaml`` ``env:``) and override only what they
exercise.
"""

from __future__ import annotations

from typing import Any


def open_engine(board_path: str, project_path: str | None = None, *,
                allow_default_rules: bool = False, **env_overrides: Any):
    """``KiCadEngine`` built from ``EnvConfig.with_overrides(**env_overrides).engine_kwargs()``."""
    from configs.loader.schema import EnvConfig
    from pcb_world.engine.kicad_engine import KiCadEngine

    return KiCadEngine(board_path, project_path, allow_default_rules=allow_default_rules,
                       **EnvConfig.with_overrides(**env_overrides).engine_kwargs())
