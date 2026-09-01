#!/usr/bin/env python3
"""Which KiCad tools the PCBench -> D3 chain runs as child processes.

The chain shells out to ``kicad-cli pcb drc`` and runs the ``pcbnew`` Python module in
a child interpreter (nothing GPL is imported in-process). Both should be the engine's
own build of the pinned, patched KiCad source — ``BUILD_CLI=1 BUILD_PCBNEW=1 bash
engine/build_rl_router.sh`` — so the data set is made with exactly the KiCad the
environment routes with (whose DRC reports every violation; a stock build caps them).

Resolution, first hit wins. The choice is written back into the environment, so the
worker processes the scripts fork inherit it:

  kicad-cli   $KICAD_CLI  ->  <build_rl>/kicad/kicad-cli  ->  ``kicad-cli`` on PATH
  pcbnew      $PCBNEW_PYTHON  ->  this interpreter, when ``import pcbnew`` works with
              <build_rl>/pcbnew on PYTHONPATH  ->  /usr/bin/python3 (the interpreter
              an apt-installed KiCad extends)

``<build_rl>`` is ``$CADAGENT_KICAD_RL_BUILD_DIR`` or ``<repo>/build_rl`` (the variable
``pcb_world.engine`` reads). A ``kicad-cli`` that sits in a KiCad build tree finds its
``_pcbnew.kiface`` only with ``KICAD_RUN_FROM_BUILD_DIR=1``; it is exported here
whenever the resolved binary has ``../pcbnew/_pcbnew.kiface`` beside it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = Path(os.environ.get("CADAGENT_KICAD_RL_BUILD_DIR", REPO_ROOT / "build_rl"))
STOCK_PYTHON = "/usr/bin/python3"
BUILD_HINT = "BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh"
_PROBE = "import pcbnew; print(pcbnew.GetBuildVersion())"


def kicad_cli() -> str:
    """Path of the kicad-cli to run; exits when there is none."""
    cli = os.environ.get("KICAD_CLI")
    if not cli:
        built = BUILD_DIR / "kicad" / "kicad-cli"
        cli = str(built) if os.access(built, os.X_OK) else shutil.which("kicad-cli")
    if not cli:
        raise SystemExit(f"kicad-cli not found: build it ({BUILD_HINT}), set KICAD_CLI, "
                         "or put a KiCad 9 kicad-cli on PATH")
    if (Path(cli).resolve().parent.parent / "pcbnew" / "_pcbnew.kiface").exists():
        os.environ.setdefault("KICAD_RUN_FROM_BUILD_DIR", "1")
    os.environ["KICAD_CLI"] = cli
    return cli


def kicad_cli_uncapped() -> bool:
    """True when the resolved kicad-cli reports complete DRC violation lists.

    The engine's own build lifts the per-type report caps (its drc_engine.cpp
    patch); a stock KiCad caps at 199 (extended types: 499) per type. The
    discriminator is the engine's build-provenance stamp
    (``../pcbnew/python/rl/ENGINE_VERSION`` relative to the binary) — a vanilla
    KiCad source build has the same tree layout but no stamp and stays "capped".
    ``KICAD_CLI_UNCAPPED=1/0`` overrides the detection; the result is written
    back so worker processes inherit it.
    """
    env = os.environ.get("KICAD_CLI_UNCAPPED")
    if env is not None:
        return env not in ("0", "", "false")
    build_root = Path(kicad_cli()).resolve().parent.parent
    uncapped = (build_root / "pcbnew" / "python" / "rl" / "ENGINE_VERSION").is_file()
    os.environ["KICAD_CLI_UNCAPPED"] = "1" if uncapped else "0"
    return uncapped


def pcbnew_python() -> str:
    """Interpreter that imports pcbnew (PYTHONPATH gains the engine build when it is used)."""
    py = os.environ.get("PCBNEW_PYTHON")
    if py:
        return py
    mod_dir = BUILD_DIR / "pcbnew"
    if (mod_dir / "pcbnew.py").is_file() and (mod_dir / "_pcbnew.so").exists():
        old = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(mod_dir) + (os.pathsep + old if old else "")
        if pcbnew_version(sys.executable) is not None:
            py = sys.executable
        elif old is None:
            del os.environ["PYTHONPATH"]
        else:
            os.environ["PYTHONPATH"] = old
    os.environ["PCBNEW_PYTHON"] = py or STOCK_PYTHON
    return os.environ["PCBNEW_PYTHON"]


def kicad_cli_version(cli: str) -> str | None:
    """First line of ``kicad-cli --version``, None when it cannot run."""
    try:
        out = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = out.stdout.strip()
    return text.splitlines()[0] if out.returncode == 0 and text else None


def pcbnew_version(py: str) -> str | None:
    """``pcbnew.GetBuildVersion()`` as printed by ``py``, None when the import fails."""
    try:
        out = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = out.stdout.strip()
    return text if out.returncode == 0 and text else None


def announce(need_pcbnew: bool = True) -> None:
    """Resolve the tools, print one line naming them + their versions, exit when unusable."""
    cli = kicad_cli()
    cli_ver = kicad_cli_version(cli)
    if cli_ver is None:
        raise SystemExit(f"{cli} --version failed")
    line = f"tools: kicad-cli={cli} ({cli_ver}{', uncapped DRC' if kicad_cli_uncapped() else ''})"
    if need_pcbnew:
        py = pcbnew_python()
        ver = pcbnew_version(py)
        if ver is None:
            raise SystemExit(f"{py} cannot import pcbnew: build it ({BUILD_HINT}) or set "
                             "PCBNEW_PYTHON to an interpreter that can")
        line += f"  pcbnew={py} ({ver})"
    print(line, flush=True)


if __name__ == "__main__":
    announce()
