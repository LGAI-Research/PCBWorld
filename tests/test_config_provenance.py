"""Config provenance: resolved-config dump + batch preflight diff.

Three test groups:

1. **Dump helper unit** (`methods._shared.config_dump`) — dict→yaml
   round-trip, meta keys, resume filename branch (original never overwritten),
   ``resume_start_iter`` mirroring ``RLTrainer._resume``.
2. **``train_ppo --dump-config-only`` subprocess** — writes the file and exits 0
   WITHOUT the engine: PYTHONPATH/LD_LIBRARY_PATH are stripped so any attempt
   to import ``kicad_rl_router`` (or reach build_rl) would fail the run.
3. **``tools/experiments/preflight_diff.py`` exit-code contract** — identical /
   expected-diff → 0, unexpected diff (incl. absent key) → 1, meta keys ignored.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from methods._shared.config_dump import (
    META_KEYS,
    dump_resolved_config,
    resolved_config_filename,
    resume_start_iter,
)

REPO = Path(__file__).resolve().parent.parent
PREFLIGHT = REPO / "tools" / "experiments" / "preflight_diff.py"


# ===================================================================
# 1. dump helper unit
# ===================================================================
class TestDumpHelper:
    def test_roundtrip_and_meta_keys(self, tmp_path, capsys):
        args = {"gamma": 0.99, "board": "x.kicad_pcb",
                "no_drc_tokens": False, "resume": None}
        path = dump_resolved_config(args, str(tmp_path))
        assert path == tmp_path / "config_resolved.yaml"

        loaded = yaml.safe_load(path.read_text())
        for key in META_KEYS:
            assert key in loaded, f"meta key {key} missing"
        # round-trip: non-meta content is exactly the input dict
        assert {k: v for k, v in loaded.items() if not k.startswith("_")} == args

        out = capsys.readouterr().out
        assert "[config] gamma = 0.99" in out
        assert "[config] _version = " in out

    def test_resume_writes_separate_file_keeps_original(self, tmp_path):
        dump_resolved_config({"iterations": 600}, str(tmp_path))
        original = (tmp_path / "config_resolved.yaml").read_text()

        path = dump_resolved_config(
            {"iterations": 300}, str(tmp_path), start_iter=301,
        )
        assert path == tmp_path / "config_resolved.resume_iter301.yaml"
        assert yaml.safe_load(path.read_text())["iterations"] == 300
        # the original run's dump is untouched (side-by-side audit trail)
        assert (tmp_path / "config_resolved.yaml").read_text() == original

    def test_resolved_config_filename(self):
        assert resolved_config_filename() == "config_resolved.yaml"
        assert resolved_config_filename(43) == "config_resolved.resume_iter43.yaml"

    def test_ckpt_args_compat_with_new_flag(self):
        """`--dump-config-only` adds a `dump_config_only` key inside new ckpts'
        args dict. The eval-path readers must ignore it (and old ckpts lacking
        it must keep falling back to defaults) — no warning, no config change."""
        import warnings

        from configs.loader.schema import RLEnvConfig, RLPolicyConfig

        old_args: dict = {}                          # pre-flag checkpoint
        new_args = {"dump_config_only": True}        # post-flag checkpoint
        with warnings.catch_warnings():
            warnings.simplefilter("error")           # any drift warning -> fail
            assert (RLEnvConfig.from_checkpoint(new_args, max_steps=100)
                    == RLEnvConfig.from_checkpoint(old_args, max_steps=100))
            assert (RLPolicyConfig.from_checkpoint(new_args)
                    == RLPolicyConfig.from_checkpoint(old_args))

    def test_resume_start_iter_mirrors_trainer(self, tmp_path):
        torch = pytest.importorskip("torch")
        ckpt = tmp_path / "policy_iter_42.pt"
        torch.save({"iteration": 42}, ckpt)
        assert resume_start_iter(str(ckpt)) == 43
        # missing key falls back like RLTrainer._resume: get("iteration", 0) + 1
        empty = tmp_path / "empty.pt"
        torch.save({}, empty)
        assert resume_start_iter(str(empty)) == 1


# ===================================================================
# 2. --dump-config-only subprocess (no GPU / no engine)
# ===================================================================
def _engine_free_env() -> dict:
    """Strip build_rl paths: importing kicad_rl_router must be impossible."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    return env


def test_dump_config_only_writes_file_and_exits_zero(tmp_path):
    board = "tests/fixtures/simple_routing_board.kicad_pcb"
    res = subprocess.run(
        [sys.executable, "-m", "methods.rl_agent.training.train_ppo",
         "--board", board, "--save-dir", str(tmp_path), "--dump-config-only"],
        cwd=REPO, env=_engine_free_env(),
        capture_output=True, text=True, timeout=180,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"

    cfg = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())
    assert cfg["board"] == board
    assert cfg["dump_config_only"] is True
    for key in META_KEYS:
        assert key in cfg
    # sorted stdout echo — destined for the head of a nohup launch.log
    assert "[config] board = " in res.stdout
    assert "[config] _git_rev = " in res.stdout


# ===================================================================
# 3. preflight_diff exit-code contract
# ===================================================================
_BASE_CFG = {
    "gamma": 0.99, "masking_rule": "hybrid", "n_envs": 8,
    "_version": "v0.11.32", "_git_rev": "abc1234", "_created": "2026-07-06",
}


def _write_cfg(tmp_path: Path, case: str, **overrides) -> Path:
    cfg = dict(_BASE_CFG, **overrides)
    path = tmp_path / case / "config_resolved.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run_preflight(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PREFLIGHT), *argv],
        capture_output=True, text=True, timeout=60,
    )


class TestPreflightDiff:
    def test_identical_configs_exit_0_meta_ignored(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB",
                       _git_rev="fff0000", _created="2026-07-07")
        res = _run_preflight(str(a), str(b))
        assert res.returncode == 0, res.stdout + res.stderr
        assert "identical" in res.stdout

    def test_unexpected_diff_exit_1(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB", gamma=1.0)
        res = _run_preflight(str(a), str(b))
        assert res.returncode == 1
        assert "gamma" in res.stdout
        assert "FAIL" in res.stdout

    def test_expected_diff_exit_0_with_matrix(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB", gamma=1.0)
        res = _run_preflight(str(a), str(b), "--expect", "gamma")
        assert res.returncode == 0, res.stdout + res.stderr
        assert "gamma" in res.stdout          # matrix row
        assert "0.99" in res.stdout and "1.0" in res.stdout

    def test_absent_key_is_unexpected_diff(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB", time_feature="sin_remaining")
        res = _run_preflight(str(a), str(b))
        assert res.returncode == 1
        assert "time_feature" in res.stdout

        res = _run_preflight(str(a), str(b), "--expect", "time_feature")
        assert res.returncode == 0
        assert "<absent>" in res.stdout

    def test_glob_input_and_expected_axis_warning(self, tmp_path):
        _write_cfg(tmp_path, "caseA")
        _write_cfg(tmp_path, "caseB")
        res = _run_preflight(
            str(tmp_path / "*" / "config_resolved.yaml"),
            "--expect", "gamma,masking_rule",
        )
        # nothing actually differs -> exit 0 but warn the axes are silent
        assert res.returncode == 0
        assert "WARNING" in res.stdout
