"""Import-path selection for the KiCad RL pybind module."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _engine_api_path(env: dict[str, str | None]) -> str:
    code = "import sys, pcb_world.engine; print(sys.path[0])"
    run_env = os.environ.copy()
    for key, value in env.items():
        if value is None:
            run_env.pop(key, None)
        else:
            run_env[key] = value
    run_env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=run_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_engine_api_uses_default_build_rl_path_without_override() -> None:
    path = _engine_api_path(
        {
            "CADAGENT_KICAD_RL_BUILD_DIR": None,
            "CADAGENT_KICAD_RL_MODULE_DIR": None,
        }
    )

    assert path == str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl")


def test_engine_api_uses_build_dir_override(tmp_path: Path) -> None:
    build_dir = tmp_path / "build_rl_custom"
    module_dir = build_dir / "pcbnew" / "python" / "rl"
    module_dir.mkdir(parents=True)

    path = _engine_api_path(
        {
            "CADAGENT_KICAD_RL_BUILD_DIR": str(build_dir),
            "CADAGENT_KICAD_RL_MODULE_DIR": None,
        }
    )

    assert path == str(module_dir)
