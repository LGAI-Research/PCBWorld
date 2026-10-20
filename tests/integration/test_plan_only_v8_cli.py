"""KDD plan-only evaluator (``eval_plan_only_llm_v8_standalone.py``) end to end.

The script once passed a kwarg ``PCBWorld`` rejects: every board ended ``init_failed``
while the run exited 0. A ``--dry-run`` (the auto-router's sequence stands in for the
LLM) must evaluate the board; a board that errors must stay in every rate as a failure,
make the run exit non-zero, and be counted the same way by ``--reaggregate``. The
launcher keeps running the other scenarios when one fails, and the sweep aggregator
shows each rate's denominator.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pcb_world.engine import engine_available

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments/kdd/llm_eval/eval_plan_only_llm_v8_standalone.py"
LAUNCHER = REPO / "experiments/kdd/table1_llm/baselines/run_plan_only_llm_v8_standalone.sh"
AGGREGATOR = REPO / "experiments/kdd/llm_eval/aggregate_qwen_together_sweep.py"
BOARD = REPO / "tests/fixtures/crossover_board.kicad_pcb"
_RATES = ("pass_at_k", "mean_success_rate", "routability_at_k_best_mean",
          "routability_at_k_mean_mean")


@pytest.fixture
def v8(monkeypatch):
    # The script inserts into sys.path at import and in main(); restore it afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))
    from experiments.kdd.llm_eval import eval_plan_only_llm_v8_standalone as mod
    return mod


@pytest.mark.skipif(not engine_available(), reason="needs the PCBWorld engine build")
def test_dry_run_evaluates_the_board(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(SCRIPT), str(BOARD), "-o", str(out),
                           "--dry-run", "-k", "1"], capture_output=True, text=True)
    overall = json.loads((out / "overall.json").read_text())
    assert (overall["boards_evaluated"], overall["boards_failed"]) == (1, 0), proc.stderr[-2000:]
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_every_board_failing_exits_nonzero(tmp_path, monkeypatch, v8):
    def _broken_env(*_a, **_kw):
        raise TypeError("PCBWorld got unknown kwarg(s)")

    monkeypatch.setattr(v8, "_build_env", _broken_env)
    monkeypatch.setattr(sys, "argv", ["v8", str(BOARD), "-o", str(tmp_path / "out"), "--dry-run"])
    assert v8.main() == 1
    overall = json.loads((tmp_path / "out/overall.json").read_text())
    assert (overall["boards_total"], overall["boards_failed"], overall["pass_at_k"]) == (1, 1, 0.0)


class _FakeEnv:
    def close(self):
        pass


@pytest.fixture
def fake_engine(monkeypatch, v8):
    """No engine: the board renders as a stub and every dry-run replay succeeds."""
    monkeypatch.setattr(v8, "_build_env", lambda *_a, **_kw: _FakeEnv())
    monkeypatch.setattr(v8, "initial_board_state_text", lambda *_a: "static")
    monkeypatch.setattr(v8, "dry_run_response_for", lambda *_a: "<actions>\nnet_end\n</actions>\n")
    monkeypatch.setattr(v8, "replay_actions_and_eval", lambda *_a: {
        "success": True, "routability": 1.0, "drv_count": 0, "wirelength_mm": 1.0})


def _run(monkeypatch, v8, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["v8", *map(str, argv)])
    return v8.main()


def _summary(out: Path) -> dict[str, dict]:
    with (out / "summary.csv").open() as f:
        return {row["board_id"]: row for row in csv.DictReader(f)}


@pytest.fixture
def good_and_bad_run(tmp_path, monkeypatch, capsys, v8, fake_engine):
    """A live ``-k 2 --ks 1,2`` run of two boards; ``bad`` raises a bug in its second replay."""
    boards = tmp_path / "boards"
    boards.mkdir()
    for name in ("good", "bad"):
        (boards / f"{name}.kicad_pcb").write_text("")
    replays: dict[str, int] = {}
    ok = v8.replay_actions_and_eval

    def _replay(src, actions, pcb):
        replays[src.stem] = replays.get(src.stem, 0) + 1
        if src.stem == "bad" and replays["bad"] == 2:   # after sample_00.json is on disk
            raise RuntimeError("replay bug")
        return ok(src, actions, pcb)

    monkeypatch.setattr(v8, "replay_actions_and_eval", _replay)
    out = tmp_path / "out"
    capsys.readouterr()
    code = _run(monkeypatch, v8, boards / "good.kicad_pcb", boards / "bad.kicad_pcb",
                "-o", out, "--dry-run", "-k", 2, "--ks", "1,2")
    return out, code, capsys.readouterr().err


def test_errored_board_counts_as_failure_and_reaggregate_agrees(good_and_bad_run, monkeypatch, v8):
    out, code, err = good_and_bad_run
    assert code == 1
    assert "1 of 2 boards errored" in err
    live = json.loads((out / "overall.json").read_text())
    assert (live["boards_total"], live["boards_evaluated"], live["boards_failed"]) == (2, 1, 1)
    assert live["pass_at_k"] == 0.5
    assert _summary(out).keys() == {p.name for p in (out / "per_board").iterdir()} \
        == {"boards__good", "boards__bad"}
    # The errored board's own aggregate.json carries the error; there is no other marker.
    bad = json.loads((out / "per_board/boards__bad/aggregate.json").read_text())
    assert bad["error"].startswith("unhandled: RuntimeError: replay bug")
    assert sorted(p.name for p in (out / "per_board/boards__bad").iterdir()) == [
        "aggregate.json", "prompt.txt", "sample_00.json", "sample_00.response.txt",
        "sample_01.response.txt"]

    assert _run(monkeypatch, v8, "--reaggregate", "-o", out, "-k", 2) == 1
    again = json.loads((out / "overall.json").read_text())
    keys = ("boards_total", "boards_evaluated", "boards_failed", *_RATES)
    assert {k: again[k] for k in keys} == {k: live[k] for k in keys}


def _summary_k(out: Path, k: int) -> dict[str, dict]:
    with (out / f"summary_k{k}.csv").open() as f:
        return {row["board_id"]: row for row in csv.DictReader(f)}


def test_errored_board_is_in_the_denominator_only_up_to_its_requested_samples(
        good_and_bad_run, monkeypatch, v8):
    """Reaggregating without ``-k`` (default 5) still uses the k each board was asked for."""
    out, _code, _err = good_and_bad_run
    live = json.loads((out / "overall_multi_k.json").read_text())["per_k"]
    live_rows = {k: _summary_k(out, k) for k in (1, 2)}
    assert _run(monkeypatch, v8, "--reaggregate", "-o", out, "--ks", "1,2,5") == 1
    per_k = json.loads((out / "overall_multi_k.json").read_text())["per_k"]
    assert {k: per_k[k] for k in ("1", "2")} == live
    assert (per_k["1"]["boards_total"], per_k["1"]["boards_failed"], per_k["1"]["pass_at_k"]) \
        == (2, 1, 0.5)
    assert {k: _summary_k(out, k) for k in (1, 2)} == live_rows
    # No board reached 5 samples and the errored one was asked for only 2.
    assert per_k["5"] == {"k": 5, "boards_total": 0, "boards_evaluated": 0, "boards_failed": 0}
    # summary_k{k}.csv lists exactly the boards in that k's denominator.
    for k in (1, 2, 5):
        assert len(_summary_k(out, k)) == per_k[str(k)]["boards_total"]


def test_board_without_samples_errors_live_and_on_reaggregate(tmp_path, monkeypatch, v8, fake_engine):
    monkeypatch.setattr(v8, "call_llm", lambda **_kw: ([], {"total_tokens": 0}))
    out = tmp_path / "out"
    assert _run(monkeypatch, v8, BOARD, "-o", out, "-k", 2) == 1
    assert _summary(out)["fixtures__crossover_board"]["error"] == "no_samples"
    assert _run(monkeypatch, v8, "--reaggregate", "-o", out, "-k", 2) == 1
    assert _summary(out)["fixtures__crossover_board"]["error"] == "no_samples"


def test_rerun_drops_the_previous_runs_samples(tmp_path, monkeypatch, v8, fake_engine):
    out = tmp_path / "out"
    stale = out / "per_board/fixtures__crossover_board/sample_07.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"success": False, "routability": 0.0}))
    assert _run(monkeypatch, v8, BOARD, "-o", out, "--dry-run", "-k", 1) == 0
    assert not stale.exists()
    assert _run(monkeypatch, v8, "--reaggregate", "-o", out, "-k", 1) == 0
    row = _summary(out)["fixtures__crossover_board"]
    assert (row["num_samples"], row["mean_success_rate"]) == ("1", "1.0")


def test_failed_api_call_keeps_the_previous_runs_responses(tmp_path, monkeypatch, v8, fake_engine):
    paid = tmp_path / "out/per_board/fixtures__crossover_board/sample_00.response.txt"
    paid.parent.mkdir(parents=True)
    paid.write_text("<actions>\nnet_end\n</actions>\n")

    def _api_down(**_kw):
        raise ConnectionError("api down")

    monkeypatch.setattr(v8, "call_llm", _api_down)
    assert _run(monkeypatch, v8, BOARD, "-o", tmp_path / "out", "-k", 1) == 1
    assert paid.exists()
    assert _summary(tmp_path / "out")["fixtures__crossover_board"]["error"].startswith(
        "api_failed: ConnectionError")


_FAKE_PYTHON = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
case "$*" in
    *d3a_boards.py*) [ -n "${FAIL_D3A:-}" ] && exit 1; echo /fake/board.kicad_pcb ;;
    *synth_2L_few_shot*) exit 3 ;;
esac
"""


def _launch_all(tmp_path, **env_overrides) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the launcher's ``all`` against a fake python; returns it and the eval calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "python3"):
        (bin_dir / name).write_text(_FAKE_PYTHON)
        (bin_dir / name).chmod(0o755)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ex.json").write_text("{}")
    log = tmp_path / "calls.log"
    log.touch()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_LOG": str(log),
           "CONDA_SETUP": str(tmp_path / "no_conda.sh"), "OUT_ROOT": str(tmp_path / "out"),
           "SYNTH_2L_TEST_DIR": str(tmp_path), "SYNTH_2L_VAL_DIR": str(tmp_path),
           "REAL_BOARD_DIR": str(tmp_path), "SYNTH_APISEQ_FEWSHOT_CACHE": str(cache),
           "REAL_APISEQ_FEWSHOT_CACHE": str(cache), **env_overrides}
    proc = subprocess.run(["bash", str(LAUNCHER), "all"], env=env, capture_output=True,
                          text=True, timeout=60)
    evals = [line for line in log.read_text().splitlines() if "eval_plan_only_llm_v8" in line]
    return proc, evals


def test_launcher_runs_every_scenario_and_fails_naming_the_failed_one(tmp_path):
    proc, evals = _launch_all(tmp_path)
    assert [e.split("--mode ")[1].split()[0] for e in evals] == [
        "zero_shot", "few_shot", "zero_shot", "few_shot"], proc.stderr
    assert proc.returncode == 1
    assert "1 scenario(s) failed: synth_2L/few_shot" in proc.stderr


def test_sweep_aggregator_shows_the_rate_denominator(tmp_path):
    run = tmp_path / "sweep/model/apiseq/real_zero_shot"
    run.mkdir(parents=True)
    stats = {"k": 1, "boards_total": 3, "boards_evaluated": 2, "boards_failed": 1,
             "pass_at_k": 1 / 3}
    (run / "overall_multi_k.json").write_text(json.dumps({"ks": [1], "per_k": {"1": stats}}))
    subprocess.run([sys.executable, str(AGGREGATOR), str(tmp_path / "sweep")], check=True,
                   capture_output=True)
    with (tmp_path / "sweep/comparison.csv").open() as f:
        (row,) = csv.DictReader(f)
    assert (row["boards_total"], row["boards_evaluated"], row["boards_failed"]) == ("3", "2", "1")


def test_launcher_lists_scenarios_failing_before_the_eval(tmp_path):
    """A missing few-shot cache or an unresolvable D3-A list fails only that scenario."""
    proc, evals = _launch_all(tmp_path, SYNTH_APISEQ_FEWSHOT_CACHE=str(tmp_path / "missing"),
                              FAIL_D3A="1")
    assert [e.split("--mode ")[1].split()[0] for e in evals] == ["zero_shot"], proc.stderr
    assert proc.returncode == 1
    assert "3 scenario(s) failed: synth_2L/few_shot real/zero_shot real/few_shot" in proc.stderr
