"""The KDD LLM-row DRV counts DRC errors only, like the RL and rule-based rows.

Each scorer the LLM rows go through (the shared rescorer, the engine-free and plan-only
evaluators, the Table 2 level plot) is fed a sample whose errors, promoted warnings and
warnings differ, and must report the errors; the level plot's CP@5 likewise needs zero
errors, and the Table 2 aggregator refuses summaries of different DRV definitions.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval.metrics


def _evaluated(*_a, **_kw):
    return {"success": True, "routability": 1.0, "track_count": 2, "via_count": 0,
            "wirelength_mm": 1.0, "drv_errors_only_count": 1,
            "drv_errors_and_promoted_count": 4, "clean_pass": False, "final_potential": 0.0}


def test_rescorer_drv_count_is_errors_only(monkeypatch, tmp_path):
    from experiments._lib.metrics import score_rollouts

    monkeypatch.setattr(eval.metrics, "evaluate_one", _evaluated)
    sample = score_rollouts._evaluate_sample(tmp_path / "s.kicad_pcb", None, "drc_dense_promoted", 45)
    assert sample["drv_count"] == 1


@pytest.fixture
def no_sys_path_leak(monkeypatch):
    # The standalone evaluators insert into sys.path at import; restore it afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))


def test_engine_free_drv_count_is_errors_only(monkeypatch, tmp_path, no_sys_path_leak):
    from experiments.kdd.llm_eval import eval_engine_free_llm_v3_standalone as v3

    monkeypatch.setattr(eval.metrics, "evaluate_one", _evaluated)
    board = tmp_path / "b.kicad_pcb"
    board.write_text("(kicad_pcb)\n")
    m = v3.evaluate_candidate(board, "", tmp_path / "out.kicad_pcb")
    assert m["drv_count"] == 1


def test_plan_only_drv_count_is_errors_only(monkeypatch, no_sys_path_leak):
    from experiments.kdd.llm_eval import eval_plan_only_llm_v8_standalone as v8

    snap = SimpleNamespace(unrouted_count=0, track_count=2, via_count=0, total_wirelength=1.0,
                           drc_violation_count=6, drc_error_count=1, drc_warning_count=5,
                           drc_promoted_count=3)
    env = SimpleNamespace(reset=lambda: None, close=lambda: None, _initial_unconnected=2,
                          _engine=SimpleNamespace(get_reward_snapshot=lambda run_drc: snap))
    monkeypatch.setattr(v8, "_build_env", lambda *_a, **_kw: env)
    assert v8.replay_actions_and_eval(Path("b.kicad_pcb"), [], None)["drv_count"] == 1


_REEVAL_HEADER = ["file", "success", "drv_count", "drc_error_count"]


def _write(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows([_REEVAL_HEADER, *rows])


def test_gpt_levels_drv_and_cp_are_errors_only(tmp_path):
    from experiments.kdd.table2 import plot_gpt_levels as plot

    # Interactive: board a has won rollouts with 1 and 0 errors (both with warnings), board b
    # wins once with 3 errors; c never wins.
    _write(tmp_path / "per_run/gpt-5.4_PCBench/eval_out_summary.csv",
           [["a_episode_00_env_00.kicad_pcb", 1, 3, 1], ["a_episode_01_env_00.kicad_pcb", 1, 1, 0],
            ["b_episode_00_env_00.kicad_pcb", 1, 3, 3], ["b_episode_01_env_00.kicad_pcb", 1, 4, 3],
            ["c_episode_00_env_00.kicad_pcb", 0, 9, 9]])
    # Open-loop: the input.kicad_pcb row is the unrouted board, not a sample.
    _write(tmp_path / "baseline_reeval/cadgen_real_zs_gpt-5.4/summary.csv",
           [["a/input.kicad_pcb", 1, 0, 0], ["a/sample_00.kicad_pcb", 1, 5, 4],
            ["a/sample_01.kicad_pcb", 1, 3, 2], ["b/sample_00.kicad_pcb", 0, 0, 0]])
    idx = {("gpt-5.4", "PCBench"): {"n_boards": "3", "drv_won_mean": "2.75", "succ@5": "0.6667",
                                    "succ&drv0@5": "0.0"},
           ("cadgen_gpt-5.4", "PCBench"): {"n_boards": "2", "drv_won_mean": "4.0", "succ@5": "0.5",
                                           "succ&drv0@5": "0.0"},
           # no per-rollout csv: zero DRV and CP@5 == P@5 under errors + warnings carry over exactly
           ("gpt-5.4", "synth2L"): {"n_boards": "4", "drv_won_mean": "0.0000", "succ@5": "0.75",
                                    "succ&drv0@5": "0.75"}}
    plot.add_errors_only_columns(idx, tmp_path)
    inter = idx[("gpt-5.4", "PCBench")]
    assert inter["cp_errors_only@5"] == pytest.approx(1 / 3)            # board a (0 errors)
    assert inter["drv_errors_only_won_mean"] == pytest.approx(7 / 4)
    assert inter["drv_errors_only_won_std"] == pytest.approx(statistics.stdev([1, 0]) / 2)  # mean of per-board stds (a: 0.707, b: 0)
    open_loop = idx[("cadgen_gpt-5.4", "PCBench")]
    assert open_loop["cp_errors_only@5"] == 0.0
    assert open_loop["drv_errors_only_won_mean"] == 3.0
    assert open_loop["drv_errors_only_won_std"] == pytest.approx(statistics.stdev([4, 2]))  # pooled
    assert (idx[("gpt-5.4", "synth2L")]["drv_errors_only_won_mean"],
            idx[("gpt-5.4", "synth2L")]["cp_errors_only@5"]) == (0.0, 0.75)

    idx[("apiseq_gpt-5.4", "synth2L")] = {"n_boards": "4", "drv_won_mean": "0.0", "succ@5": "0.5",
                                          "succ&drv0@5": "0.25"}
    with pytest.raises(SystemExit, match="apiseq_synth_zs_gpt-5.4"):
        plot.add_errors_only_columns(idx, tmp_path)


def _overall(root: Path, model: str, **extra) -> None:
    d = root / model / "d2a" / "_eval" / "interactive"
    d.mkdir(parents=True)
    (d / "overall.json").write_text(json.dumps({"boards_evaluated": 1, "pass_at_k": 1.0, **extra}))


def test_table2_aggregator_refuses_mixed_drv_definitions(tmp_path):
    from experiments._lib.metrics.score_rollouts import DRV_DEFINITION
    from experiments.kdd.llm_eval import aggregate_p_cp_at_k as agg

    _overall(tmp_path, "new", drv_definition=DRV_DEFINITION)
    assert [r["drv_definition"] for r in agg.collect(tmp_path, 5, False)] == ["errors_only"]
    _overall(tmp_path, "old")    # written before the key existed
    with pytest.raises(SystemExit, match="different DRV definitions"):
        agg.collect(tmp_path, 5, False)
    assert agg.collect(tmp_path / "old", 5, False)[0]["drv_definition"] == agg.LEGACY_DRV_DEFINITION
