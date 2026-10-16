"""The Engine CLI resolver finds the macOS app bundle under build_rl."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import kicad_tools  # noqa: E402


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho 10.0.7\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_macos_bundle_cli_wins_and_uses_build_root(tmp_path, monkeypatch):
    build = tmp_path / "build_rl"
    bundle_cli = _executable(
        build / "kicad" / "KiCad.app" / "Contents" / "MacOS" / "kicad-cli")
    _executable(build / "kicad" / "kicad-cli")
    kiface = build / "pcbnew" / "_pcbnew.kiface"
    kiface.parent.mkdir(parents=True)
    kiface.write_bytes(b"")
    stamp = build / "pcbnew" / "python" / "rl" / "ENGINE_VERSION"
    stamp.parent.mkdir(parents=True)
    stamp.write_text("1.5\n")

    monkeypatch.delenv("KICAD_CLI", raising=False)
    monkeypatch.delenv("KICAD_RUN_FROM_BUILD_DIR", raising=False)
    monkeypatch.setattr(kicad_tools, "BUILD_DIR", build)
    monkeypatch.setattr(kicad_tools.platform, "system", lambda: "Darwin")

    assert kicad_tools.kicad_cli() == str(bundle_cli)
    assert os.environ["KICAD_RUN_FROM_BUILD_DIR"] == "1"
    assert kicad_tools.kicad_cli_uncapped()
