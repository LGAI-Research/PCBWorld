"""Tool resolution of the PCBench -> D3 chain (tools/datagen/pcbench_prep/kicad_tools.py).

Pins the order the chain picks its child-process KiCad tools in, on synthetic
layouts (no engine build, no KiCad install needed):
  kicad-cli   $KICAD_CLI -> <build_rl>/kicad/kicad-cli -> PATH; a build-tree binary
              (../pcbnew/_pcbnew.kiface beside it) exports KICAD_RUN_FROM_BUILD_DIR=1
  pcbnew      $PCBNEW_PYTHON -> this interpreter with <build_rl>/pcbnew on PYTHONPATH
              when the import probe passes -> /usr/bin/python3
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import kicad_tools  # noqa: E402

_ENV = ("KICAD_CLI", "PCBNEW_PYTHON", "PYTHONPATH", "KICAD_RUN_FROM_BUILD_DIR", "KICAD_CLI_UNCAPPED")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(kicad_tools, "BUILD_DIR", tmp_path / "build_rl")
    # the stock fallback must not resolve on a host that happens to have apt KiCad
    monkeypatch.setattr(kicad_tools, "STOCK_PYTHON", str(tmp_path / "no-python3"))
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    return tmp_path


def _executable(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\necho 9.0.8\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_kicad_cli_env_override_wins(clean_env, monkeypatch):
    _executable(clean_env / "build_rl" / "kicad" / "kicad-cli")
    monkeypatch.setenv("KICAD_CLI", "/opt/somewhere/kicad-cli")
    assert kicad_tools.kicad_cli() == "/opt/somewhere/kicad-cli"


def test_kicad_cli_prefers_engine_build_and_marks_build_tree(clean_env):
    built = _executable(clean_env / "build_rl" / "kicad" / "kicad-cli")
    (clean_env / "build_rl" / "pcbnew").mkdir()
    (clean_env / "build_rl" / "pcbnew" / "_pcbnew.kiface").write_bytes(b"")
    assert kicad_tools.kicad_cli() == str(built)
    assert os.environ["KICAD_RUN_FROM_BUILD_DIR"] == "1"   # inherited by the DRC children
    assert os.environ["KICAD_CLI"] == str(built)            # inherited by the worker processes


def test_kicad_cli_falls_back_to_path(clean_env, monkeypatch):
    on_path = _executable(clean_env / "bin" / "kicad-cli")
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert kicad_tools.kicad_cli() == str(on_path)
    assert "KICAD_RUN_FROM_BUILD_DIR" not in os.environ


def test_kicad_cli_missing_exits(clean_env):
    with pytest.raises(SystemExit, match="kicad-cli not found"):
        kicad_tools.kicad_cli()


def test_pcbnew_python_env_override_wins(clean_env, monkeypatch):
    monkeypatch.setenv("PCBNEW_PYTHON", "/opt/kicad/python3")
    assert kicad_tools.pcbnew_python() == "/opt/kicad/python3"


def test_pcbnew_python_uses_engine_build_when_import_works(clean_env, monkeypatch):
    mod = clean_env / "build_rl" / "pcbnew"
    mod.mkdir(parents=True)
    (mod / "pcbnew.py").write_text("def GetBuildVersion(): return '9.0.8-test'\n")
    (mod / "_pcbnew.so").write_bytes(b"")
    monkeypatch.setenv("PYTHONPATH", "/existing")
    assert kicad_tools.pcbnew_python() == sys.executable      # the real import probe ran
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(mod), "/existing"]
    assert os.environ["PCBNEW_PYTHON"] == sys.executable
    assert kicad_tools.pcbnew_version(sys.executable) == "9.0.8-test"


def test_pcbnew_python_stock_fallback_restores_pythonpath(clean_env, monkeypatch):
    mod = clean_env / "build_rl" / "pcbnew"
    mod.mkdir(parents=True)
    (mod / "pcbnew.py").write_text("raise ImportError('broken build')\n")
    (mod / "_pcbnew.so").write_bytes(b"")
    assert kicad_tools.pcbnew_python() == kicad_tools.STOCK_PYTHON
    assert "PYTHONPATH" not in os.environ


def test_announce_prints_one_line_and_exits_without_pcbnew(clean_env, capsys):
    _executable(clean_env / "build_rl" / "kicad" / "kicad-cli")
    kicad_tools.announce(need_pcbnew=False)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("tools: kicad-cli=") and "(9.0.8)" in out[0]
    with pytest.raises(SystemExit, match="cannot import pcbnew"):
        kicad_tools.announce()   # /usr/bin/python3 fallback has no pcbnew here


def test_uncapped_detection_and_override(clean_env, monkeypatch):
    built = _executable(clean_env / "build_rl" / "kicad" / "kicad-cli")
    (clean_env / "build_rl" / "pcbnew" / "python" / "rl").mkdir(parents=True)
    (clean_env / "build_rl" / "pcbnew" / "_pcbnew.kiface").write_bytes(b"")
    (clean_env / "build_rl" / "pcbnew" / "python" / "rl" / "ENGINE_VERSION").write_text("1.3\n")
    assert kicad_tools.kicad_cli_uncapped() is True        # engine build (stamped)
    assert os.environ["KICAD_CLI_UNCAPPED"] == "1"          # inherited by workers
    monkeypatch.setenv("KICAD_CLI_UNCAPPED", "0")           # explicit override wins
    assert kicad_tools.kicad_cli_uncapped() is False


def test_vanilla_kicad_build_tree_is_capped(clean_env, monkeypatch):
    # same layout as any KiCad source build, but no ENGINE_VERSION stamp:
    # run-from-build-dir applies, the uncapped guard does not
    built = _executable(clean_env / "build_rl" / "kicad" / "kicad-cli")
    (clean_env / "build_rl" / "pcbnew").mkdir()
    (clean_env / "build_rl" / "pcbnew" / "_pcbnew.kiface").write_bytes(b"")
    monkeypatch.setenv("KICAD_CLI", str(built))
    assert kicad_tools.kicad_cli() == str(built)
    assert os.environ["KICAD_RUN_FROM_BUILD_DIR"] == "1"
    assert kicad_tools.kicad_cli_uncapped() is False
    assert os.environ["KICAD_CLI_UNCAPPED"] == "0"


def test_path_cli_is_treated_as_capped(clean_env, monkeypatch):
    on_path = _executable(clean_env / "bin" / "kicad-cli")
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert kicad_tools.kicad_cli_uncapped() is False
    assert os.environ["KICAD_CLI_UNCAPPED"] == "0"


def test_make_guide_truncation_gate_follows_cli(clean_env, monkeypatch):
    import make_guide
    counts = {"clearance": 700, "unconnected_items": 12}
    monkeypatch.setenv("KICAD_CLI_UNCAPPED", "0")
    assert make_guide.drc_truncated(counts) is True         # stock CLI: capped list
    monkeypatch.setenv("KICAD_CLI_UNCAPPED", "1")
    assert make_guide.drc_truncated(counts) is False        # engine CLI: counts are real
