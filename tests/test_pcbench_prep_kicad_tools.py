"""KiCad 10.0.7 tool resolution for the PCBench -> D3 chain."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import kicad_tools  # noqa: E402

_ENV = (
    "KICAD_CLI", "PCBNEW_PYTHON", "PYTHONPATH", "KICAD_RUN_FROM_BUILD_DIR",
    "KICAD_CLI_UNCAPPED", "KICAD_STOCK_CLI", "KICAD_STOCK_PYTHON", "KICAD_PYTHON",
    "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "LD_LIBRARY_PATH",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(kicad_tools, "BUILD_DIR", tmp_path / "build_rl")
    monkeypatch.setattr(
        kicad_tools, "_default_stock_paths",
        lambda: (str(tmp_path / "stock" / "kicad-cli"), str(tmp_path / "stock" / "python3")),
    )
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    return tmp_path


def _executable(path: Path, output: str = "10.0.7") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {output}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _engine_tree(root: Path, *, pcbnew_version: str = "10.0.7") -> tuple[Path, Path]:
    cli = _executable(root / "kicad" / "kicad-cli")
    module = root / "pcbnew"
    module.mkdir(parents=True)
    (module / "pcbnew.py").write_text(
        f"def GetBuildVersion(): return '{pcbnew_version}'\n")
    (module / "_pcbnew.so").write_bytes(b"")
    (module / "_pcbnew.kiface").write_bytes(b"")
    stamp = module / "python" / "rl" / "ENGINE_VERSION"
    stamp.parent.mkdir(parents=True)
    stamp.write_text("1.5\n")
    return cli, module


def test_kicad_cli_env_override_wins(clean_env, monkeypatch):
    override = _executable(clean_env / "override" / "kicad-cli")
    _engine_tree(clean_env / "build_rl")
    monkeypatch.setenv("KICAD_CLI", str(override))
    assert kicad_tools.kicad_cli() == str(override)


def test_kicad_cli_prefers_engine_build_and_marks_build_tree(clean_env):
    built, module = _engine_tree(clean_env / "build_rl")
    assert kicad_tools.kicad_cli() == str(built)
    assert os.environ["KICAD_RUN_FROM_BUILD_DIR"] == "1"
    assert os.environ["KICAD_CLI"] == str(built)
    assert kicad_tools.kicad_cli_uncapped() is True


def test_macos_engine_app_bundle_resolves_build_root(clean_env, monkeypatch):
    root = clean_env / "build_rl"
    cli = _executable(root / "kicad" / "KiCad.app" / "Contents" / "MacOS" / "kicad-cli")
    module = root / "pcbnew"
    module.mkdir(parents=True)
    (module / "_pcbnew.kiface").write_bytes(b"")
    monkeypatch.setenv("KICAD_CLI", str(cli))
    assert kicad_tools._engine_build_root() == root
    assert os.environ["KICAD_RUN_FROM_BUILD_DIR"] == "1"


def test_kicad_cli_never_falls_back_to_path(clean_env, monkeypatch):
    on_path = _executable(clean_env / "bin" / "kicad-cli")
    monkeypatch.setenv("PATH", str(on_path.parent))
    with pytest.raises(SystemExit, match="Engine kicad-cli not found"):
        kicad_tools.kicad_cli()


def test_kicad_cli_missing_exits(clean_env):
    with pytest.raises(SystemExit, match="Engine kicad-cli not found"):
        kicad_tools.kicad_cli()


def test_pcbnew_python_uses_engine_build_and_version(clean_env, monkeypatch):
    built, module = _engine_tree(clean_env / "build_rl")
    monkeypatch.setenv("PYTHONPATH", "/existing")
    assert kicad_tools.pcbnew_python() == sys.executable
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(module), "/existing"]
    info = kicad_tools.pcbnew_info(sys.executable)
    assert info["version"] == "10.0.7"
    assert Path(info["module"]).resolve().parent == module.resolve()


def test_pcbnew_python_override_still_loads_engine_module(clean_env, monkeypatch):
    _engine_tree(clean_env / "build_rl")
    monkeypatch.setenv("PCBNEW_PYTHON", "/opt/engine-python")
    assert kicad_tools.pcbnew_python() == "/opt/engine-python"
    assert str(clean_env / "build_rl" / "pcbnew") in os.environ["PYTHONPATH"]


def test_pcbnew_requires_engine_build(clean_env):
    _executable(clean_env / "build_rl" / "kicad" / "kicad-cli")
    with pytest.raises(SystemExit, match="Engine pcbnew build missing"):
        kicad_tools.pcbnew_python()


def test_announce_requires_versioned_uncapped_engine_pair(clean_env, capsys):
    _engine_tree(clean_env / "build_rl")
    kicad_tools.announce()
    assert "10.0.7" in capsys.readouterr().out


def test_announce_rejects_engine_cli_version_mismatch(clean_env):
    cli, _ = _engine_tree(clean_env / "build_rl")
    _executable(cli, output="10.0.6")
    with pytest.raises(SystemExit, match="Engine CLI must be KiCad 10.0.7"):
        kicad_tools.announce()


def test_announce_rejects_engine_pcbnew_version_mismatch(clean_env):
    _engine_tree(clean_env / "build_rl", pcbnew_version="10.0.6")
    with pytest.raises(SystemExit, match="Engine pcbnew must be KiCad 10.0.7"):
        kicad_tools.announce()


def test_announce_rejects_stock_cli_as_engine(clean_env, monkeypatch):
    stock = _executable(clean_env / "stock" / "kicad-cli")
    _engine_tree(clean_env / "build_rl")
    monkeypatch.setenv("KICAD_CLI", str(stock))
    with pytest.raises(SystemExit, match="not the uncapped Engine build"):
        kicad_tools.announce()


def test_stock_preflight_checks_cli_and_pcbnew_versions(clean_env, monkeypatch, capsys):
    _engine_tree(clean_env / "build_rl")
    stock_cli = _executable(clean_env / "stock" / "kicad-cli")
    stock_python = str(clean_env / "stock" / "python3")
    monkeypatch.setenv("KICAD_STOCK_CLI", str(stock_cli))
    monkeypatch.setenv("KICAD_STOCK_PYTHON", stock_python)
    monkeypatch.setattr(kicad_tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(kicad_tools, "pcbnew_info", lambda py: {
        "version": "10.0.7", "module": "/usr/lib/python3/dist-packages/pcbnew.py",
        "python": [3, 12, 4],
    })
    assert kicad_tools.assert_stock_tools() == (str(stock_cli), stock_python)
    assert "stock tools:" in capsys.readouterr().out


def test_stock_probes_ignore_and_restore_engine_runtime_paths(clean_env, monkeypatch):
    _engine_tree(clean_env / "build_rl")
    stock_cli = _executable(clean_env / "stock" / "kicad-cli")
    monkeypatch.setenv("KICAD_STOCK_CLI", str(stock_cli))
    monkeypatch.setenv("KICAD_STOCK_PYTHON", "/stock/python3")
    monkeypatch.setenv("PYTHONPATH", "/engine/pcbnew")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/engine/frameworks")
    monkeypatch.setattr(kicad_tools.platform, "system", lambda: "Linux")

    def cli_version(path):
        assert "PYTHONPATH" not in os.environ
        assert "DYLD_LIBRARY_PATH" not in os.environ
        return "10.0.7"

    def pcbnew_info(path):
        assert "PYTHONPATH" not in os.environ
        assert "DYLD_LIBRARY_PATH" not in os.environ
        return {"version": "10.0.7", "module": "/usr/lib/python3/dist-packages/pcbnew.py",
                "python": [3, 12, 4]}

    monkeypatch.setattr(kicad_tools, "kicad_cli_version", cli_version)
    monkeypatch.setattr(kicad_tools, "pcbnew_info", pcbnew_info)
    kicad_tools.assert_stock_tools()
    assert os.environ["PYTHONPATH"] == "/engine/pcbnew"
    assert os.environ["DYLD_LIBRARY_PATH"] == "/engine/frameworks"


def test_macos_stock_probes_use_stock_bundle_frameworks(clean_env, monkeypatch):
    _engine_tree(clean_env / "build_rl")
    app = clean_env / "stock" / "KiCad-10.0.7.app"
    frameworks = app / "Contents" / "Frameworks"
    cli = _executable(app / "Contents" / "MacOS" / "kicad-cli")
    py = _executable(frameworks / "Python.framework" / "Versions" / "Current" / "bin" / "python3")
    monkeypatch.setenv("KICAD_STOCK_CLI", str(cli))
    monkeypatch.setenv("KICAD_STOCK_PYTHON", str(py))
    monkeypatch.setenv("PYTHONPATH", "/engine/pcbnew")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/engine/frameworks")
    monkeypatch.setenv("KICAD_RUN_FROM_BUILD_DIR", "1")
    monkeypatch.setattr(kicad_tools.platform, "system", lambda: "Darwin")

    def cli_version(path):
        assert os.environ["DYLD_LIBRARY_PATH"] == str(frameworks)
        assert os.environ["DYLD_FRAMEWORK_PATH"] == str(frameworks)
        assert "PYTHONPATH" not in os.environ
        assert "KICAD_RUN_FROM_BUILD_DIR" not in os.environ
        return "10.0.7"

    def pcbnew_info(path):
        assert os.environ["DYLD_LIBRARY_PATH"] == str(frameworks)
        return {"version": "10.0.7", "module": str(frameworks / "pcbnew.py"),
                "python": [3, 9, 13]}

    monkeypatch.setattr(kicad_tools, "kicad_cli_version", cli_version)
    monkeypatch.setattr(kicad_tools, "pcbnew_info", pcbnew_info)
    kicad_tools.assert_stock_tools()
    assert os.environ["DYLD_LIBRARY_PATH"] == "/engine/frameworks"
    assert os.environ["KICAD_RUN_FROM_BUILD_DIR"] == "1"


def test_stock_preflight_requires_macos_python_3913(clean_env, monkeypatch):
    _engine_tree(clean_env / "build_rl")
    cli = _executable(clean_env / "stock" / "kicad-cli")
    monkeypatch.setenv("KICAD_STOCK_CLI", str(cli))
    monkeypatch.setenv("KICAD_STOCK_PYTHON", "/Applications/KiCad/Python3")
    monkeypatch.setattr(kicad_tools.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(kicad_tools, "pcbnew_info", lambda py: {
        "version": "10.0.7", "module": "/Applications/KiCad/pcbnew.py", "python": [3, 11, 0],
    })
    with pytest.raises(SystemExit, match="must be 3.9.13"):
        kicad_tools.assert_stock_tools()


def test_stock_preflight_rejects_wrong_cli_version(clean_env, monkeypatch):
    _engine_tree(clean_env / "build_rl")
    cli = _executable(clean_env / "stock" / "kicad-cli", output="9.0.9")
    monkeypatch.setenv("KICAD_STOCK_CLI", str(cli))
    monkeypatch.setenv("KICAD_STOCK_PYTHON", "/stock/python3")
    with pytest.raises(SystemExit, match="stock CLI must be KiCad 10.0.7"):
        kicad_tools.assert_stock_tools()
