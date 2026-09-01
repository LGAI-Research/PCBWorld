"""Contract tests for data-root resolution in configs/loader/paths.py.

Pins the single-default-line contract:
  1. $CADAGENT_DATA_ROOT set   -> it wins (over the paths.yaml default);
  2. unset                     -> the paths.yaml default falls back;
  3. both empty (the public-release configuration, or the var forced to "")
                               -> resolution fails loudly naming the variable.

The yaml files here are synthetic so the tests pin the MECHANISM, not the
internal default value. One smoke test loads the real configs/paths.yaml and
only asserts that a data root resolves.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from configs.loader import paths as paths_mod

_YAML_TEMPLATE = """\
campaign: kdd
roots:
  data:   ${{CADAGENT_DATA_ROOT:-{default}}}
  staged: {staged}
  out:    ./var/results
  ckpt:   ./var/checkpoints
datasets:
  demo: {{sub: some/sub}}
"""


def _write_yaml(tmp_path: Path, default: str) -> Path:
    p = tmp_path / "paths.yaml"
    p.write_text(_YAML_TEMPLATE.format(default=default, staged=tmp_path / "staged"))
    return p


def test_env_var_wins_over_yaml_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CADAGENT_DATA_ROOT", "/tmp/env_root")
    cfg = paths_mod.load_paths(_write_yaml(tmp_path, "/tmp/yaml_default_root"))
    assert cfg.data == Path("/tmp/env_root")
    assert paths_mod.resolve_dataset("demo", cfg) == Path("/tmp/env_root/some/sub")


def test_unset_var_falls_back_to_yaml_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CADAGENT_DATA_ROOT", raising=False)
    cfg = paths_mod.load_paths(_write_yaml(tmp_path, "/tmp/yaml_default_root"))
    assert cfg.data == Path("/tmp/yaml_default_root")
    assert paths_mod.resolve_dataset("demo", cfg) == Path(
        "/tmp/yaml_default_root/some/sub"
    )


@pytest.mark.parametrize("scenario", ["yaml_default_empty", "var_forced_empty"])
def test_both_empty_fails_loudly(tmp_path, monkeypatch, scenario):
    if scenario == "yaml_default_empty":
        # The release configuration: the yaml default is flipped to empty.
        monkeypatch.delenv("CADAGENT_DATA_ROOT", raising=False)
        cfg = paths_mod.load_paths(_write_yaml(tmp_path, ""))
    else:
        # The var explicitly exported empty overrides the internal default.
        monkeypatch.setenv("CADAGENT_DATA_ROOT", "")
        cfg = paths_mod.load_paths(_write_yaml(tmp_path, "/tmp/yaml_default_root"))
    assert cfg.data is None
    with pytest.raises(RuntimeError) as exc:
        paths_mod.resolve_dataset("demo", cfg)
    assert str(exc.value) == (
        "dataset 'demo' has no staged copy and CADAGENT_DATA_ROOT is "
        "not set — export CADAGENT_DATA_ROOT pointing at your dataset "
        "root (expected layout: configs/paths.yaml)."
    )


def test_expand_env_var_wins(monkeypatch):
    monkeypatch.setenv("CADAGENT_DATA_ROOT", "/tmp/env_root")
    assert (
        paths_mod.expand_data_path("${CADAGENT_DATA_ROOT}/x/y.kicad_pcb")
        == "/tmp/env_root/x/y.kicad_pcb"
    )


def test_expand_unset_var_falls_back_to_yaml_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CADAGENT_DATA_ROOT", raising=False)
    cfg = paths_mod.load_paths(_write_yaml(tmp_path, "/tmp/yaml_default_root"))
    assert (
        paths_mod.expand_data_path("${CADAGENT_DATA_ROOT}/x.kicad_pcb", cfg)
        == "/tmp/yaml_default_root/x.kicad_pcb"
    )
    # A path with no ${...} passes through unchanged (plain entries stay valid).
    assert paths_mod.expand_data_path("relative/board.kicad_pcb", cfg) == (
        "relative/board.kicad_pcb"
    )


def test_expand_both_empty_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("CADAGENT_DATA_ROOT", raising=False)
    cfg = paths_mod.load_paths(_write_yaml(tmp_path, ""))
    with pytest.raises(KeyError) as exc:
        paths_mod.expand_data_path("${CADAGENT_DATA_ROOT}/x.kicad_pcb", cfg)
    assert str(exc.value.args[0]) == (
        "dataset path '${CADAGENT_DATA_ROOT}/x.kicad_pcb' references unset "
        "environment variable(s): CADAGENT_DATA_ROOT. Point "
        "CADAGENT_DATA_ROOT at your dataset root (expected layout: "
        "configs/paths.yaml)."
    )


def test_resolve_dataset_or_empty_returns_empty_not_raise(tmp_path, monkeypatch):
    # Module-level/argparse defaults use this variant: empty root -> "" (the
    # caller checks before use), never an import-time exception.
    monkeypatch.delenv("CADAGENT_DATA_ROOT", raising=False)
    cfg = paths_mod.load_paths(_write_yaml(tmp_path, ""))
    assert paths_mod.resolve_dataset_or_empty("demo", cfg) == ""
    cfg = paths_mod.load_paths(_write_yaml(tmp_path, "/tmp/yaml_default_root"))
    assert paths_mod.resolve_dataset_or_empty("demo", cfg) == (
        "/tmp/yaml_default_root/some/sub"
    )


def test_real_paths_yaml_has_no_baked_in_data_root(monkeypatch, tmp_path):
    # The tracked configs/paths.yaml carries NO default data root: the datasets are not
    # distributed, so with the env var unset the root is None and a dataset that needs it
    # fails loudly naming the variable; with it set, every registered sub resolves under it.
    monkeypatch.delenv("CADAGENT_DATA_ROOT", raising=False)
    # A staged local copy (var/datasets/<sub>) is served without the data root — the
    # README's trial set lives exactly there — so point the staged root at an empty dir.
    monkeypatch.setenv("CADAGENT_STAGED_ROOT", str(tmp_path / "staged"))
    cfg = paths_mod.load_paths()
    assert cfg.data is None
    with pytest.raises(RuntimeError, match="CADAGENT_DATA_ROOT"):
        paths_mod.resolve_dataset("synth_2L_v2", cfg)
    monkeypatch.setenv("CADAGENT_DATA_ROOT", str(tmp_path))
    cfg = paths_mod.load_paths()
    assert cfg.data == tmp_path
    assert paths_mod.resolve_dataset("synth_2L_v2", cfg) == tmp_path / "synthetic" / "synth_2L_v2"
