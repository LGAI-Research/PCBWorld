"""KiCad toolchain resolution for the KiCad 10.0.7 PCBench -> D3 chain.

The prep chain uses the Engine's uncapped KiCad 10.0.7 build for conversion
and DRC. The separately installed stock KiCad CLI/Python are checked too, so
the independent Step10 save/refill/reopen path cannot silently fall back to 9.x
or 10.0.6.

Explicit overrides:
  KICAD_CLI          Engine CLI override (must carry the Engine build stamp)
  PCBNEW_PYTHON      interpreter that loads the Engine build's pcbnew module
  KICAD_STOCK_CLI    independent stock CLI (default: host's pinned install)
  KICAD_STOCK_PYTHON independent stock Python (or KICAD_PYTHON)
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = Path(os.environ.get("PCBWORLD_KICAD_RL_BUILD_DIR", REPO_ROOT / "build_rl"))
EXPECTED_VERSION = "10.0.7"
BUILD_HINT = "BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh"
_PROBE = (
    "import json, pcbnew, sys; "
    "print(json.dumps({'version': pcbnew.GetBuildVersion(), "
    "'module': pcbnew.__file__, 'python': list(sys.version_info[:3])}))"
)
_STOCK_CLEAN_ENV = (
    "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "KICAD_RUN_FROM_BUILD_DIR",
    "KICAD_CLI_UNCAPPED",
)


def _default_stock_paths() -> tuple[str, str]:
    if platform.system() == "Darwin":
        app = Path.home() / "Applications" / "KiCad-10.0.7" / "KiCad.app"
        return (
            str(app / "Contents" / "MacOS" / "kicad-cli"),
            str(app / "Contents" / "Frameworks" / "Python.framework" /
                "Versions" / "Current" / "bin" / "python3"),
        )
    if platform.system() == "Linux":
        return "/usr/bin/kicad-cli", "/usr/bin/python3"
    raise SystemExit(f"unsupported host for the pinned KiCad toolchain: {platform.system()}")


def _version_is_pinned(version: str | None) -> bool:
    return bool(version and re.match(r"^10\.0\.7(?:$|[-+])", version.strip()))


def _build_root_from_cli(cli: str) -> Path:
    """Resolve build_rl from either the Linux binary or macOS app-bundle CLI."""
    path = Path(cli).resolve()
    if (path.parent.name == "MacOS" and path.parent.parent.name == "Contents"
            and path.parent.parent.parent.suffix == ".app" and len(path.parents) > 4):
        return path.parents[4]
    return path.parent.parent


def _bundle_frameworks(*paths: str) -> Path | None:
    for value in paths:
        for parent in Path(value).resolve().parents:
            if parent.suffix == ".app":
                return parent / "Contents" / "Frameworks"
    return None


def _engine_build_root() -> Path:
    return _build_root_from_cli(kicad_cli())


def kicad_cli() -> str:
    """Resolve the Engine CLI only; never fall back to a host/PATH KiCad."""
    cli = os.environ.get("KICAD_CLI")
    if not cli:
        candidates = []
        if platform.system() == "Darwin":
            candidates.append(
                BUILD_DIR / "kicad" / "KiCad.app" / "Contents" / "MacOS" / "kicad-cli")
        candidates.append(BUILD_DIR / "kicad" / "kicad-cli")
        cli = next((str(candidate) for candidate in candidates
                    if os.access(candidate, os.X_OK)), None)
    if not cli or not os.access(cli, os.X_OK):
        raise SystemExit(
            f"Engine kicad-cli not found: build it ({BUILD_HINT}) or set "
            "KICAD_CLI to a KiCad 10.0.7 Engine build"
        )
    build_root = _build_root_from_cli(cli)
    if (build_root / "pcbnew" / "_pcbnew.kiface").exists():
        os.environ.setdefault("KICAD_RUN_FROM_BUILD_DIR", "1")
    os.environ["KICAD_CLI"] = cli
    return cli


def kicad_cli_uncapped() -> bool:
    """True only for a CLI next to the Engine build-provenance stamp."""
    build_root = _build_root_from_cli(kicad_cli())
    return (build_root / "pcbnew" / "python" / "rl" / "ENGINE_VERSION").is_file()


def pcbnew_python() -> str:
    """Resolve an interpreter importing the Engine's own ``pcbnew`` build."""
    mod_dir = _engine_build_root() / "pcbnew"
    if not (mod_dir / "pcbnew.py").is_file() or not list(mod_dir.glob("_pcbnew.*")):
        raise SystemExit(f"Engine pcbnew build missing under {mod_dir}; build it ({BUILD_HINT})")
    old = os.environ.get("PYTHONPATH")
    entries = old.split(os.pathsep) if old else []
    if str(mod_dir) not in entries:
        os.environ["PYTHONPATH"] = str(mod_dir) + (os.pathsep + old if old else "")
    py = os.environ.get("PCBNEW_PYTHON") or sys.executable
    os.environ["PCBNEW_PYTHON"] = py
    return py


def kicad_cli_version(cli: str) -> str | None:
    """First line of ``kicad-cli --version``, None when it cannot run."""
    try:
        out = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = out.stdout.strip()
    return text.splitlines()[0] if out.returncode == 0 and text else None


def pcbnew_info(py: str) -> dict | None:
    """Return build version, module path, and Python version for ``py``."""
    try:
        out = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None


def pcbnew_version(py: str) -> str | None:
    """``pcbnew.GetBuildVersion()`` as printed by ``py``, None when import fails."""
    info = pcbnew_info(py)
    return info.get("version") if info else None


def announce(need_pcbnew: bool = True) -> None:
    """Assert and print the Engine CLI/Python versions before processing."""
    cli = kicad_cli()
    cli_ver = kicad_cli_version(cli)
    if not _version_is_pinned(cli_ver):
        raise SystemExit(f"Engine CLI must be KiCad {EXPECTED_VERSION}: {cli} reports {cli_ver!r}")
    if not kicad_cli_uncapped():
        raise SystemExit(f"{cli} is not the uncapped Engine build (missing ENGINE_VERSION stamp)")
    line = f"engine tools: kicad-cli={cli} ({cli_ver}, uncapped DRC)"
    if need_pcbnew:
        py = pcbnew_python()
        info = pcbnew_info(py)
        if not info:
            raise SystemExit(f"{py} cannot import the Engine pcbnew build under {_engine_build_root() / 'pcbnew'}")
        if not _version_is_pinned(info.get("version")):
            raise SystemExit(
                f"Engine pcbnew must be KiCad {EXPECTED_VERSION}: {py} reports "
                f"{info.get('version')!r}"
            )
        module = Path(info.get("module", "")).resolve()
        engine_module = (_engine_build_root() / "pcbnew").resolve()
        if module.parent != engine_module:
            raise SystemExit(
                f"Engine pcbnew module resolved outside its build: {module} "
                f"(expected under {engine_module})"
            )
        line += f"  pcbnew={py} ({info['version']}, {module})"
    print(line, flush=True)


def assert_stock_tools() -> tuple[str, str]:
    """Assert the independent stock KiCad CLI/Python are exactly 10.0.7."""
    default_cli, default_python = _default_stock_paths()
    cli = os.environ.get("KICAD_STOCK_CLI", default_cli)
    py = os.environ.get("KICAD_STOCK_PYTHON", os.environ.get("KICAD_PYTHON", default_python))
    inherited = {key: os.environ.pop(key) for key in _STOCK_CLEAN_ENV if key in os.environ}
    try:
        if platform.system() == "Darwin":
            frameworks = _bundle_frameworks(cli, py)
            if frameworks is not None:
                os.environ["DYLD_LIBRARY_PATH"] = str(frameworks)
                os.environ["DYLD_FRAMEWORK_PATH"] = str(frameworks)
        cli_ver = kicad_cli_version(cli)
        info = pcbnew_info(py)
    finally:
        for key in _STOCK_CLEAN_ENV:
            os.environ.pop(key, None)
        os.environ.update(inherited)

    if not _version_is_pinned(cli_ver):
        raise SystemExit(f"stock CLI must be KiCad {EXPECTED_VERSION}: {cli} reports {cli_ver!r}")
    if not info or not _version_is_pinned(info.get("version")):
        version = info.get("version") if info else None
        raise SystemExit(f"stock pcbnew must be KiCad {EXPECTED_VERSION}: {py} reports {version!r}")

    engine_cli = Path(kicad_cli()).resolve()
    stock_cli = Path(cli).resolve()
    stock_build = _build_root_from_cli(cli)
    if stock_cli == engine_cli or (stock_build / "pcbnew" / "python" / "rl" / "ENGINE_VERSION").is_file():
        raise SystemExit(f"stock CLI must be independent of the Engine build: {cli}")
    engine_module = (_engine_build_root() / "pcbnew").resolve()
    stock_module = Path(info.get("module", "")).resolve()
    if engine_module == stock_module or engine_module in stock_module.parents:
        raise SystemExit(f"stock Python imported the Engine pcbnew module: {stock_module}")
    if platform.system() == "Darwin" and info.get("python") != [3, 9, 13]:
        raise SystemExit(
            f"macOS stock KiCad Python must be 3.9.13: {py} reports "
            f"{info.get('python')!r}"
        )
    print(f"stock tools: kicad-cli={cli} ({cli_ver})  pcbnew={py} ({info['version']})", flush=True)
    return cli, py


def assert_all() -> None:
    """Preflight Engine and independent stock toolchains before the pipeline."""
    announce()
    assert_stock_tools()


if __name__ == "__main__":
    if sys.argv[1:] == ["--assert-all"]:
        assert_all()
    else:
        announce()
