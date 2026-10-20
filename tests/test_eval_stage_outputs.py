import math
from pathlib import Path
import sys

import pytest

from configs.loader.schema import EnvConfig
from eval import pipeline as core
from eval import eval_utils as u
from eval.aggregation import aggregate_boards, aggregate_inline
from methods._shared.board_loader import BoardSpec
from eval.metrics import EvalResult
from tests.helpers.ckpt_payload import ckpt_config


def _policy():
    """A loaded policy stand-in: what the pipeline reads off it is the loader's sources."""
    from types import SimpleNamespace

    return SimpleNamespace(ckpt_sources={})


@pytest.mark.parametrize(
    ("rollout_mode", "n_envs", "inline_drc"),
    [
        ("serial", 1, "on"),
        ("serial", 1, "off"),
        ("parallel", 2, "off"),
    ],
)
def test_stage3_per_board_metric_outputs_are_populated_for_eval_modes(
    tmp_path,
    monkeypatch,
    rollout_mode,
    n_envs,
    inline_drc,
):
    output_dir = tmp_path / f"stage3_{rollout_mode}_{inline_drc}"
    ckpt = tmp_path / "policy.pt"
    ckpt.write_bytes(b"unused")
    boards_dir = tmp_path / "boards"
    boards_dir.mkdir()
    board = BoardSpec(0, "board_a", str(boards_dir / "board_a.kicad_pcb"))

    monkeypatch.setattr(core, "_resolve_device", lambda spec: "cpu")
    monkeypatch.setattr(
        core,
        "load_boards_from_dir_or_list",
        lambda *, boards_dir=None, boards_list=None: [board],
    )
    monkeypatch.setattr(core, "load_policy_from_ckpt",
                        lambda ckpt, device, *, strict: (_policy(), ckpt_config(seed=0), 0))
    monkeypatch.setattr(core, "apply_n_max_slots_override", lambda policy, slots, explicit: None)

    def _row(rollout_idx, *, with_drc):
        row = {field: "" for field in u.PER_ROLLOUT_FIELDS}
        row.update(
            {
                "board_index": 0,
                "board_id": "board_a",
                "board_path": board.path,
                "rollout_idx": rollout_idx,
                "seed": 123 + rollout_idx,
                "status": "completed",
                "steps": 10 + rollout_idx,
                "episode_return": 0.1 * rollout_idx,
                "final_potential": 1.0 + 2.0 * rollout_idx,
                "routability": 0.5 + 0.5 * rollout_idx,
                "wirelength_mm": 10.0 + 10.0 * rollout_idx,
                "via_count": rollout_idx,
                "track_count": 2 + rollout_idx,
                "success": bool(rollout_idx),
                # uniform cell layout: <board_id>_<cell>_s<SS>_r<RR>.kicad_pcb
                "artifact_path": f"board_a_{output_dir.name}_s00_r{rollout_idx:02d}.kicad_pcb",
            }
        )
        if with_drc:
            row.update(
                {
                    "eval_status": "ok",
                    "eval_time_sec": 0.2 + rollout_idx,
                    "clean_pass": bool(rollout_idx),
                    "drv_errors_only_count": 1 - rollout_idx,
                    "drv_errors_and_promoted_count": 2 - rollout_idx,
                    "track_angle_drv_count": 0,
                    "total_drv_count": 2 - rollout_idx,
                    "routed_pcb": str(output_dir / row["artifact_path"]),
                }
            )
        return row

    def fake_eval_transformer(policy, device, boards, **kwargs):
        rows = [_row(0, with_drc=inline_drc == "on"), _row(1, with_drc=inline_drc == "on")]
        u.write_csv(kwargs["per_rollout_flush_path"], rows, u.PER_ROLLOUT_FIELDS)
        return EvalResult(per_rollout=rows)

    def fake_eval_kicad_pcb(rollout_dir, **kwargs):
        rows = u.read_csv(Path(rollout_dir) / "per_rollout.csv")
        for row in rows:
            rollout_idx = int(row["rollout_idx"])
            row.update(_row(rollout_idx, with_drc=True))
        u.write_csv(Path(rollout_dir) / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
        return rows

    monkeypatch.setattr(core, "eval_transformer", fake_eval_transformer)
    monkeypatch.setattr(core, "eval_kicad_pcb", fake_eval_kicad_pcb)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--ckpt",
            str(ckpt),
            "--boards-dir",
            str(boards_dir),
            "--seed",
            "123",
            "--n-rollouts",
            "2",
            "--n-envs",
            str(n_envs),
            "--rollout-mode",
            rollout_mode,
            "--inline-drc",
            inline_drc,
            "--output-dir",
            str(output_dir),
        ],
    )

    assert core.main() == 0

    # The run record names the scoring a later re-scoring of this cell reuses.
    import json
    import yaml
    scoring = yaml.safe_load((output_dir / "run_record.yaml").read_text())["extra"]["scoring"]
    corner = json.loads((output_dir / "manifest.json").read_text())["env_kwargs"]["corner_mode"]
    assert scoring == {"reward_config": core.DEFAULTS.reward_config, "check_angle": corner,
                       "selection_method": core.DEFAULTS.selection_method,
                       "training_reward_rule": "pcbworld_reward", "training_reward_overrides": {},
                       "matches_training_rule": True}

    # Stage 3 = aggregate_boards: per_boards_{ckpts,overall,summary}.csv
    overall = u.read_csv(output_dir / "per_boards_overall.csv")[0]
    ckpts = u.read_csv(output_dir / "per_boards_ckpts.csv")
    assert (output_dir / "per_boards_summary.csv").is_file()

    # one board, one ckpt (s00), two rollouts (r00/r01)
    # avg = mean over rollouts; best = winner (max final_potential = r01)
    assert overall["final_potential_avg"] == "2.0"
    assert overall["routability_avg"] == "0.75"
    assert overall["wirelength_mm_avg"] == "15.0"
    assert overall["success_avg"] == "0.5"
    assert overall["eval_time_sec_avg"] == "0.7"           # time: mean over rollouts
    assert overall["drv_errors_only_count_avg"] == "0.5"
    assert overall["final_potential_best"] == "3.0"
    assert overall["routability_best"] == "1.0"
    assert ckpts[0]["selected_rollout_idx"] == "1"


def test_flatten_rollout_artifacts_renames_to_unified_grammar(tmp_path):
    # Stage-1 layout: artifacts/board_<idx>_rollout_<r>.kicad_pcb (+ sidecar)
    cell_dir = tmp_path / "mycell"
    artifacts = cell_dir / "artifacts"
    artifacts.mkdir(parents=True)
    rows = []
    for rollout_idx in (0, 1):
        pcb = artifacts / f"board_00000_rollout_{rollout_idx}.kicad_pcb"
        pcb.write_text("pcb")
        pcb.with_suffix(".kicad_pro").write_text("pro")
        row = {field: "" for field in u.PER_ROLLOUT_FIELDS}
        row.update(
            {
                "board_index": 0,
                "board_id": "board_00000",
                "rollout_idx": rollout_idx,
                "status": "completed",
                "artifact_path": str(pcb),
            }
        )
        rows.append(row)
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    moved = core.flatten_rollout_artifacts(cell_dir, ckpt_seed=42)

    assert moved == 2
    for rollout_idx in (0, 1):
        dst = cell_dir / "boards" / f"board_00000_mycell_s42_r{rollout_idx:02d}.kicad_pcb"
        assert dst.is_file()
        assert dst.with_suffix(".kicad_pro").is_file()
        # the new name parses back through the single grammar source
        assert u.parse_cell_artifact(dst.name, "mycell") == ("board_00000", 42, rollout_idx)
    assert not artifacts.exists()  # staging dir emptied and removed
    new_rows = u.read_csv(cell_dir / "per_rollout.csv")
    assert [r["artifact_path"] for r in new_rows] == [
        str(cell_dir / "boards" / f"board_00000_mycell_s42_r{i:02d}.kicad_pcb")
        for i in (0, 1)
    ]
    # already-unified rows are untouched: a second run is a no-op
    assert core.flatten_rollout_artifacts(cell_dir, ckpt_seed=42) == 0


def test_aggregate_boards_fails_loudly_without_grammar_matches(tmp_path):
    # per_rollout.csv rows whose artifact_path never matches the cell grammar
    # (the pre-flatten Stage-1 naming) must abort with a message naming the
    # directory and the expected grammar — not exit 0 with nothing written.
    cell_dir = tmp_path / "somecell"
    cell_dir.mkdir()
    row = {field: "" for field in u.PER_ROLLOUT_FIELDS}
    row.update(
        {
            "board_index": 0,
            "board_id": "board_00000",
            "rollout_idx": 0,
            "status": "completed",
            "artifact_path": str(cell_dir / "artifacts" / "board_00000_rollout_0.kicad_pcb"),
        }
    )
    u.write_csv(cell_dir / "per_rollout.csv", [row], u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit) as excinfo:
        aggregate_boards(cell_dir)

    msg = str(excinfo.value)
    assert "_somecell_s<SS>_r<RR>.kicad_pcb" in msg
    assert str(cell_dir / "boards") in msg
    assert not (cell_dir / "per_boards_overall.csv").exists()


def test_eval_cli_defaults_to_final_potential_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--ckpt",
            str(tmp_path / "policy.pt"),
            "--boards-dir",
            str(tmp_path),
            "--seed",
            "1",
            "--n-rollouts",
            "1",
        ],
    )

    _, args = core._parse_args()

    assert args.selection_method == "final_potential"


def test_skip_rollout_accepts_existing_output_dir_only(monkeypatch, tmp_path):
    output_dir = tmp_path / "existing_rollout"
    output_dir.mkdir()
    row = {field: "" for field in u.PER_ROLLOUT_FIELDS}
    row.update(
        {
            "board_index": 0,
            "board_id": "board_a",
            "board_path": "/tmp/board_a.kicad_pcb",
            "rollout_idx": 0,
            "status": "completed",
            "final_potential": 2.0,
            "routability": 1.0,
            "wirelength_mm": 10.0,
            "success": True,
            "artifact_path": str(output_dir / f"board_a_{output_dir.name}_s00_r00.kicad_pcb"),
        }
    )
    u.write_csv(output_dir / "per_rollout.csv", [row], u.PER_ROLLOUT_FIELDS)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--skip-rollout",
            "--skip-drc",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert core.main() == 0

    assert (output_dir / "per_boards_overall.csv").is_file()
    assert (output_dir / "per_boards_summary.csv").is_file()
    summary = (output_dir / "eval_overall_summary.json").read_text()
    assert '"ckpt": null' in summary
    assert '"seed": null' in summary


def test_skip_rollout_defaults_inline_drc_off(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--skip-rollout",
            "--output-dir",
            str(tmp_path),
        ],
    )

    _, args = core._parse_args()
    _mode, final_drc = core._resolve_modes(args)

    assert final_drc is False


def test_best_per_board_filename_uses_selection_method(monkeypatch, tmp_path):
    output_dir = tmp_path / "existing_rollout"
    output_dir.mkdir()
    rows = []
    for rollout_idx, drv_count in [(0, 3), (1, 0)]:
        row = {field: "" for field in u.PER_ROLLOUT_FIELDS}
        row.update(
            {
                "board_index": 0,
                "board_id": "board_a",
                "board_path": "/tmp/board_a.kicad_pcb",
                "rollout_idx": rollout_idx,
                "status": "completed",
                "final_potential": 2.0 - rollout_idx,
                "routability": 1.0,
                "wirelength_mm": 10.0,
                "via_count": 0,
                "success": True,
                "drv_errors_only_count": drv_count,
                "drv_errors_and_promoted_count": drv_count,
                "artifact_path": str(
                    output_dir / f"board_a_{output_dir.name}_s00_r{rollout_idx:02d}.kicad_pcb"
                ),
            }
        )
        rows.append(row)
    u.write_csv(output_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--skip-rollout",
            "--skip-drc",
            "--output-dir",
            str(output_dir),
            "--selection-mode",
            "posthoc_drc_aware",
        ],
    )

    assert core.main() == 0

    # posthoc_drc_aware: routability tie (1.0) -> fewer drv wins -> r01 (drv=0)
    ckpts = u.read_csv(output_dir / "per_boards_ckpts.csv")
    assert ckpts[0]["selected_rollout_idx"] == "1"


def test_selection_mode_alias_maps_to_selection_method(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--ckpt",
            str(tmp_path / "policy.pt"),
            "--boards-dir",
            str(tmp_path),
            "--seed",
            "1",
            "--n-rollouts",
            "1",
            "--selection-mode",
            "posthoc_drc_aware",
        ],
    )

    _, args = core._parse_args()

    assert args.selection_method == "posthoc_drc_aware"


def test_aggregation_mode_arg_is_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--ckpt",
            str(tmp_path / "policy.pt"),
            "--boards-dir",
            str(tmp_path),
            "--seed",
            "1",
            "--n-rollouts",
            "1",
            "--aggregation-mode",
            "best",
        ],
    )

    with pytest.raises(SystemExit):
        core._parse_args()


def test_per_board_aggregation_is_direct_average_not_best():
    rows = [
        {
            "board_index": 0,
            "rollout_idx": 0,
            "status": "completed",
            "success": True,
            "final_potential": 1.0,
            "routability": 0.5,
            "wirelength_mm": 10.0,
        },
        {
            "board_index": 0,
            "rollout_idx": 1,
            "status": "completed",
            "success": False,
            "final_potential": 3.0,
            "routability": 1.0,
            "wirelength_mm": 20.0,
        },
    ]

    per_board, overall = aggregate_inline(
        rows,
        [BoardSpec(0, "board_a", "/tmp/board_a.kicad_pcb")],
        aggregation_mode="mean",
        selection_mode="final_potential",
    )

    assert per_board[0]["final_potential"] == 2.0
    assert per_board[0]["routability"] == 0.75
    assert per_board[0]["wirelength_mm"] == 15.0
    assert per_board[0]["success"] == 0.5
    assert overall["final_potential"] == 2.0
    assert per_board[0]["selected_by_mode_rollout_idx"] == 1


# ---------------------------------------------------------------------------
# Stage 3: an attempt without a routed artifact is a failure, never dropped
# ---------------------------------------------------------------------------


def _cell_row(cell, board_id, board_index, rollout_idx, *, ckpt=0, artifact=True, **values):
    row = {field: "" for field in u.PER_ROLLOUT_FIELDS}
    row.update(
        {
            "board_index": board_index,
            "board_id": board_id,
            "rollout_idx": rollout_idx,
            "status": "completed" if artifact else "crashed",
            "artifact_path": (
                f"{board_id}_{cell}_s{ckpt:02d}_r{rollout_idx:02d}.kicad_pcb" if artifact else ""
            ),
        }
    )
    row.update(values)
    return row


def _scored(clean_pass, success, routability, wirelength_mm, final_potential):
    return dict(
        eval_status="ok", clean_pass=clean_pass, success=success, routability=routability,
        wirelength_mm=wirelength_mm, final_potential=final_potential, per_board_rollout_time=2.0,
    )


def _three_board_cell(tmp_path):
    cell_dir = tmp_path / "rbcell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "b", 1, 0, **_scored(False, True, 1.0, 30.0, 3.0)),
        # RL-style crash: no artifact, but env-side values that must neither win the
        # selection nor enter the board-quality means.
        _cell_row(c, "b", 1, 1, artifact=False, crashed=True, final_potential=99.0,
                  wirelength_mm=500.0, per_board_rollout_time=4.0),
        # Rule-based-style crash: no artifact, metric columns blank.
        _cell_row(c, "c", 2, 0, artifact=False, crashed=True, per_board_rollout_time=1.0,
                  error="krt rc=0"),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return cell_dir


def _by_board(rows):
    return {r["board_id"]: r for r in rows}


def test_aggregate_boards_counts_no_artifact_rows_as_failures(tmp_path):
    cell_dir = _three_board_cell(tmp_path)

    _ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    # the board whose only attempt left no artifact stays in the denominator
    assert summary["n_boards"] == 3
    assert summary["clean_pass_best"] == pytest.approx(1 / 3)
    assert summary["success_best"] == pytest.approx(2 / 3)
    assert summary["routability_best"] == pytest.approx(2 / 3)
    ov = _by_board(overall_rows)
    assert ov["c"]["clean_pass_best"] == 0.0
    assert ov["c"]["success_avg"] == 0.0
    assert ov["c"]["routability_best"] == 0.0
    # a failed attempt counts in the per-attempt average too
    assert ov["b"]["success_avg"] == pytest.approx(0.5)
    assert ov["b"]["routability_avg"] == pytest.approx(0.5)


def test_aggregate_boards_keeps_board_quality_metrics_over_routed_attempts(tmp_path):
    cell_dir = _three_board_cell(tmp_path)

    ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    ov = _by_board(overall_rows)
    # the crashed rollout's env values neither win nor enter the means
    assert _by_board(ckpt_rows)["b"]["selected_rollout_idx"] == "0"
    assert ov["b"]["final_potential_best"] == 3.0
    assert ov["b"]["wirelength_mm_avg"] == 30.0
    assert ov["c"]["wirelength_mm_best"] is None
    assert ov["c"]["final_potential_best"] is None
    assert summary["wirelength_mm_best"] == pytest.approx(20.0)
    assert summary["final_potential_best"] == pytest.approx(4.0)
    # time is charged for every attempt, failed ones included
    assert ov["b"]["per_board_rollout_time_avg"] == pytest.approx(3.0)
    assert ov["c"]["per_board_rollout_time_avg"] == pytest.approx(1.0)


def test_aggregate_boards_sums_llm_api_failures_and_warns(tmp_path, capsys):
    """LLM rows record the steps that fell back to idle after a failed API call:
    the summary sums them and warns, while the scoring stays as it was."""
    (tmp_path / "rl").mkdir()
    (tmp_path / "llm").mkdir()
    rl_ckpts, rl_overall, (before,) = aggregate_boards(_three_board_cell(tmp_path / "rl"))
    out = capsys.readouterr().out
    assert "api_failed_calls" not in before and "fell back to idle" not in out  # RL: blank
    assert {r["api_failed_calls"] for r in rl_ckpts + rl_overall} == {""}

    cell_dir = _three_board_cell(tmp_path / "llm")
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    for row, n in zip(rows, ("3", "0", "", "2")):    # "" = a crashed episode, not recorded
        row["api_failed_calls"] = n
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    assert (summary["api_failed_calls"], summary["api_failed_boards"]) == (5, 2)
    # per board: a = 3; b = 0 + its unrecorded crash; c = 2
    assert [_by_board(overall_rows)[b]["api_failed_calls"] for b in "abc"] == [3, 0, 2]
    assert [_by_board(ckpt_rows)[b]["api_failed_calls"] for b in "abc"] == [3, 0, 2]
    on_disk_overall = _by_board(u.read_csv(cell_dir / "per_boards_overall.csv"))
    assert on_disk_overall["a"]["api_failed_calls"] == "3"
    (on_disk,) = u.read_csv(cell_dir / "per_boards_summary.csv")
    assert (on_disk["api_failed_calls"], on_disk["api_failed_boards"]) == ("5", "2")
    assert ("WARNING: rbcell: 5 LLM step(s) on 2 board(s) fell back to idle after the "
            "provider call failed") in capsys.readouterr().out
    assert {k: v for k, v in summary.items() if not k.startswith("api_")} == before


def test_aggregate_boards_records_zero_api_failures_without_warning(tmp_path, capsys):
    cell_dir = _three_board_cell(tmp_path)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    for row in rows:
        row["api_failed_calls"] = "0"
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    _, _, (summary,) = aggregate_boards(cell_dir)

    assert (summary["api_failed_calls"], summary["api_failed_boards"]) == (0, 0)
    assert "fell back to idle" not in capsys.readouterr().out


def test_aggregate_boards_reports_failed_and_crashed_counts(tmp_path, capsys):
    cell_dir = _three_board_cell(tmp_path)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    rows.append(_cell_row(cell_dir.name, "c", 2, 1, artifact=False, status="completed"))
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    aggregate_boards(cell_dir)

    ckpts = _by_board(u.read_csv(cell_dir / "per_boards_ckpts.csv"))
    overall = _by_board(u.read_csv(cell_dir / "per_boards_overall.csv"))
    (summary,) = u.read_csv(cell_dir / "per_boards_summary.csv")
    assert [ckpts[b]["n_failed"] for b in "abc"] == ["0", "1", "2"]
    assert [overall[b]["n_failed"] for b in "abc"] == ["0", "1", "2"]
    # n_crashed = the failed attempts whose status is crashed (a subset)
    assert [ckpts[b]["n_crashed"] for b in "abc"] == ["0", "1", "1"]
    assert [overall[b]["n_crashed"] for b in "abc"] == ["0", "1", "1"]
    assert (summary["n_failed"], summary["n_crashed"]) == ("3", "2")
    assert summary["expected_boards"] == "per_rollout_rows"
    out = capsys.readouterr().out
    assert "failed=3 crashed=2" in out
    assert "WARNING" in out  # no expected board set: a board with no row is undetectable


def test_aggregate_boards_counts_expected_board_without_any_row(tmp_path):
    cell_dir = _three_board_cell(tmp_path)
    expected = [BoardSpec(i, bid, "") for i, bid in enumerate("abcd")]

    _ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir, expected_boards=expected)

    assert summary["n_boards"] == 4
    assert summary["clean_pass_best"] == pytest.approx(1 / 4)
    assert summary["n_failed"] == 3
    assert summary["n_crashed"] == 2  # the missing row is a failure, not a crash
    assert summary["expected_boards"] == "given"
    d = _by_board(overall_rows)["d"]
    assert d["board_index"] == "3"
    assert d["n_failed"] == 1
    assert d["clean_pass_best"] == 0.0


def test_aggregate_boards_rejects_rows_outside_the_expected_set(tmp_path):
    cell_dir = _three_board_cell(tmp_path)

    with pytest.raises(SystemExit, match="not in the expected board set"):
        aggregate_boards(cell_dir, expected_boards=[BoardSpec(0, "a", ""), BoardSpec(1, "b", "")])
    assert not (cell_dir / "per_boards_summary.csv").exists()


def _merged_cell(tmp_path, failed_ckpt):
    cell_dir = tmp_path / "mergedcell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, ckpt=42, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "a", 0, 0, ckpt=43, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "a", 0, 1, artifact=False, crashed=True),
    ]
    rows[2]["ckpt"] = failed_ckpt
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return cell_dir


def test_aggregate_boards_charges_a_failure_to_its_ckpt_column(tmp_path):
    cell_dir = _merged_cell(tmp_path, failed_ckpt=43)

    ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    by_ckpt = {r["ckpt"]: r for r in ckpt_rows}
    assert (by_ckpt[42]["n_failed"], by_ckpt[43]["n_failed"]) == (0, 1)
    assert by_ckpt[43]["success_avg"] == pytest.approx(0.5)
    a = _by_board(overall_rows)["a"]
    assert a["success_best"] == 1.0  # both ckpts routed the board once
    assert a["success_avg"] == pytest.approx(2 / 3)
    assert summary["n_unattributed_failed"] == 0


def test_aggregate_boards_keeps_a_failure_without_ckpt_in_avg_only(tmp_path, capsys):
    # A merged cell from before the ckpt column: the failure cannot be charged
    # to a ckpt, so it enters the pooled _avg, not _best, and is reported.
    cell_dir = _merged_cell(tmp_path, failed_ckpt="")

    ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    assert [r["n_failed"] for r in ckpt_rows] == [0, 0]
    a = _by_board(overall_rows)["a"]
    assert a["success_avg"] == pytest.approx(2 / 3)
    assert a["success_best"] == 1.0
    assert a["n_failed"] == 1 and a["n_unattributed_failed"] == 1
    assert summary["n_unattributed_failed"] == 1
    out = capsys.readouterr().out
    assert "1 failed row(s) carry no ckpt" in out
    # board a routed under both ckpts: the failure is an extra attempt, _best is exact
    assert "on 0 board(s) with a routed attempt that lack a ckpt" in out


def test_unattributed_failure_warning_counts_only_optimistic_boards(tmp_path, capsys):
    # optimistic = a routed board missing a ckpt (y: routed only under 42); not a: routed
    # under both ckpts, nor x: never routed (one failure carries its ckpt, one does not)
    cell_dir = _merged_cell(tmp_path, failed_ckpt="")
    c = cell_dir.name
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    rows += [_cell_row(c, "x", 1, 0, ckpt=42, artifact=False, crashed=True),
             _cell_row(c, "x", 1, 1, artifact=False, crashed=True),
             _cell_row(c, "y", 2, 0, ckpt=42, **_scored(True, True, 1.0, 10.0, 5.0)),
             _cell_row(c, "y", 2, 1, artifact=False, crashed=True)]
    rows[-4]["ckpt"], rows[-3]["ckpt"], rows[-1]["ckpt"] = 42, "", ""
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    aggregate_boards(cell_dir)

    out = capsys.readouterr().out
    assert "3 failed row(s) carry no ckpt" in out
    assert "on 1 board(s) with a routed attempt that lack a ckpt" in out


def test_pipeline_stage3_counts_a_stage1_board_without_rows(monkeypatch, tmp_path):
    # Stage 1 knows the board set: a board the rollout never wrote a row for is a
    # failure in the Stage-3 summary.
    output_dir = tmp_path / "cell"
    ckpt = tmp_path / "policy.pt"
    ckpt.write_bytes(b"unused")
    boards = [BoardSpec(0, "board_a", "a.kicad_pcb"), BoardSpec(1, "board_b", "b.kicad_pcb")]
    monkeypatch.setattr(core, "_resolve_device", lambda spec: "cpu")
    monkeypatch.setattr(core, "load_boards_from_dir_or_list",
                        lambda *, boards_dir=None, boards_list=None: boards)
    monkeypatch.setattr(core, "load_policy_from_ckpt",
                        lambda ckpt, device, *, strict: (_policy(), ckpt_config(seed=0), 0))
    monkeypatch.setattr(core, "apply_n_max_slots_override", lambda policy, slots, explicit: None)

    def fake_eval_transformer(policy, device, boards, **kwargs):
        rows = [_cell_row(output_dir.name, "board_a", 0, 0, **_scored(True, True, 1.0, 10.0, 5.0))]
        u.write_csv(kwargs["per_rollout_flush_path"], rows, u.PER_ROLLOUT_FIELDS)
        return EvalResult(per_rollout=rows)

    monkeypatch.setattr(core, "eval_transformer", fake_eval_transformer)
    monkeypatch.setattr(sys, "argv", [
        "eval", "--ckpt", str(ckpt), "--boards-dir", str(tmp_path), "--seed", "0",
        "--n-rollouts", "1", "--n-envs", "1", "--rollout-mode", "serial",
        "--inline-drc", "on", "--output-dir", str(output_dir),
    ])

    assert core.main() == 0

    (summary,) = u.read_csv(output_dir / "per_boards_summary.csv")
    assert summary["n_boards"] == "2"
    assert summary["clean_pass_best"] == "0.5"
    assert summary["n_failed"] == "1"
    assert summary["expected_boards"] == "given"


@pytest.mark.parametrize("selection_mode", ["final_potential", "posthoc_drc_aware"])
def test_aggregate_boards_never_selects_a_failed_attempt_over_a_routed_one(tmp_path, selection_mode):
    # A routed rollout whose DRC columns are blank (Stage 2 never ran on it) must
    # still beat a sibling attempt that left no artifact.
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, success=True, wirelength_mm=10.0),
        _cell_row(c, "a", 0, 1, artifact=False, crashed=True),
        _cell_row(c, "b", 1, 0, **_scored(True, True, 1.0, 20.0, 5.0)),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    ckpt_rows, overall_rows, _ = aggregate_boards(cell_dir, selection_mode=selection_mode)

    assert _by_board(ckpt_rows)["a"]["selected_rollout_idx"] == "0"
    assert _by_board(overall_rows)["a"]["wirelength_mm_best"] == 10.0


def test_aggregate_boards_leaves_unmeasured_metrics_blank_for_failed_attempts(tmp_path):
    # Stage 2 never ran: routed rows carry no routability / clean_pass, so a failed
    # attempt must not turn the board's value into a measured 0.
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, success=True, routability="nan", final_potential=4.0),
        _cell_row(c, "a", 0, 1, artifact=False, crashed=True),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    _, overall_rows, (summary,) = aggregate_boards(cell_dir)

    a = _by_board(overall_rows)["a"]
    assert a["routability_avg"] is None
    assert a["clean_pass_avg"] is None
    assert a["success_avg"] == pytest.approx(0.5)  # measured on the routed rows
    assert summary["n_failed"] == 1


def test_aggregate_boards_refuses_a_routed_artifact_outside_the_grammar(tmp_path):
    # An existing board file whose name does not follow the cell grammar is a layout
    # error, not a failed attempt.
    cell_dir = tmp_path / "cell"
    (cell_dir / "boards").mkdir(parents=True)
    c = cell_dir.name
    stray = cell_dir / "boards" / "a_othercell_s00_r01.kicad_pcb"
    stray.write_text("pcb")
    rows = [
        _cell_row(c, "a", 0, 0, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "a", 0, 1, **_scored(True, True, 1.0, 10.0, 5.0)),
    ]
    rows[1]["artifact_path"] = str(stray)
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit, match="not in the cell grammar"):
        aggregate_boards(cell_dir)


def test_aggregate_boards_refuses_a_failed_row_without_board_id(tmp_path):
    cell_dir = _three_board_cell(tmp_path)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    rows.append(_cell_row(cell_dir.name, "", 3, 0, artifact=False, crashed=True))
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit, match="no board_id"):
        aggregate_boards(cell_dir)


def test_aggregate_boards_keeps_input_board_counts_of_failed_attempts(tmp_path):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, initial_unconnected=7, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "a", 0, 1, artifact=False, crashed=True, initial_unconnected=9,
                  unrouted_count=3),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    _, overall_rows, _ = aggregate_boards(cell_dir)

    a = _by_board(overall_rows)["a"]
    assert a["initial_unconnected_avg"] == 8.0  # the failed row's count is kept
    assert a["unrouted_count_avg"] is None  # a measurement of a board that does not exist


# ---------------------------------------------------------------------------
# Failed attempts are priced at the bare board's Φ₀ (Stage 2's record)
# ---------------------------------------------------------------------------


def _write_bare_potentials(cell_dir, phi0, reward_config="pcbworld_reward"):
    u.write_csv(cell_dir / u.BARE_BOARD_POTENTIALS_CSV, [
        {"board_id": bid, "initial_potential": v, "reward_config": reward_config,
         "eval_status": "ok"} for bid, v in phi0.items()
    ], ["board_id", "initial_potential", "reward_config", "eval_status"])


def test_aggregate_boards_prices_failed_attempts_at_the_bare_board_potential(tmp_path, capsys):
    cell_dir = _three_board_cell(tmp_path)
    _write_bare_potentials(cell_dir, {"b": -20.0, "c": -10.0})

    ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    ov = _by_board(overall_rows)
    # b: routed fp 3.0 + failed attempt at Φ₀ -20.0; the routed attempt still wins
    assert ov["b"]["final_potential_avg"] == pytest.approx(-8.5)
    assert ov["b"]["final_potential_best"] == 3.0
    assert _by_board(ckpt_rows)["b"]["selected_rollout_idx"] == "0"
    # c: only a failed attempt — the board as if left untouched, no gain
    assert ov["c"]["final_potential_best"] == -10.0
    assert ov["c"]["initial_potential_best"] == -10.0
    assert ov["c"]["potential_gain_best"] == 0.0
    assert ov["c"]["wirelength_mm_best"] is None  # still no board to measure
    assert summary["final_potential_best"] == pytest.approx((5.0 + 3.0 - 10.0) / 3)
    assert summary["n_failed_boards_without_phi0"] == 0
    assert "have no Φ₀" not in capsys.readouterr().out


def test_aggregate_boards_warns_when_a_failed_board_has_no_bare_potential(tmp_path, capsys):
    cell_dir = _three_board_cell(tmp_path)
    _write_bare_potentials(cell_dir, {"b": -20.0})

    _, overall_rows, (summary,) = aggregate_boards(cell_dir)

    assert _by_board(overall_rows)["c"]["final_potential_best"] is None
    assert summary["n_failed_boards_without_phi0"] == 1
    assert "1 board(s) with a failed or unscored attempt have no Φ₀" in capsys.readouterr().out


def test_aggregate_boards_refuses_bare_potentials_from_another_reward_config(tmp_path):
    cell_dir = _three_board_cell(tmp_path)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    for r in rows:
        r["reward_config"] = "pcbworld_reward"
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"b": -20.0, "c": -10.0}, reward_config="other_rule")

    with pytest.raises(SystemExit, match="prices Φ₀ with"):
        aggregate_boards(cell_dir)


@pytest.mark.parametrize("col,values", [("reward_config", ("pcbworld_reward", "other_rule")),
                                        ("check_angle", (45, 90))])
def test_aggregate_boards_refuses_rows_scored_under_two_settings(tmp_path, col, values):
    # a Stage 2 re-scoring interrupted halfway leaves rows of two scorings: no mixed metric
    cell_dir = _three_board_cell(tmp_path)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    for i, r in enumerate(rows):
        r[col] = values[i % 2]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit, match=rf"several {col} values .*rerun Stage 2 \(--stages eval\) with one {col}"):
        aggregate_boards(cell_dir)


def _inline_cell_without_phi0(tmp_path):
    # Inline-scored cell (Stage 2 artifact scoring skipped) of an env whose reset ran no
    # DRC: the scored rows carry the scorer's final Φ but no Φ₀ / gain; board b also crashed.
    cell_dir = tmp_path / "inlinecell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, board_path="/in/a.kicad_pcb", reward_config="pcbworld_reward",
                  initial_potential="nan", potential_gain="nan",
                  **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "b", 1, 0, board_path="/in/b.kicad_pcb", reward_config="pcbworld_reward",
                  initial_potential="nan", potential_gain="nan",
                  **_scored(True, True, 1.0, 10.0, 3.0)),
        _cell_row(c, "b", 1, 1, artifact=False, crashed=True, board_path="/in/b.kicad_pcb"),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return cell_dir


def test_failed_board_inputs_cover_scored_rows_without_phi0(tmp_path):
    cell_dir = _inline_cell_without_phi0(tmp_path)

    need = core._failed_board_inputs(u.read_csv(cell_dir / "per_rollout.csv"), cell_dir, None)

    assert need == {"a": "/in/a.kicad_pcb", "b": "/in/b.kicad_pcb"}


def test_aggregate_boards_fills_scored_rows_phi0_from_bare_potentials(tmp_path, capsys):
    """potential_gain was averaged over the failed attempts' 0.0 only (scored rows NaN)."""
    cell_dir = _inline_cell_without_phi0(tmp_path)
    _write_bare_potentials(cell_dir, {"a": -5.0, "b": -7.0})

    _, overall_rows, (summary,) = aggregate_boards(cell_dir)

    ov = _by_board(overall_rows)
    assert ov["a"]["initial_potential_avg"] == -5.0
    assert ov["a"]["potential_gain_avg"] == pytest.approx(10.0)
    assert ov["b"]["potential_gain_avg"] == pytest.approx((10.0 + 0.0) / 2)
    assert ov["b"]["potential_gain_best"] == pytest.approx(10.0)
    assert summary["n_scored_without_potential_gain"] == 0
    assert "have no potential_gain" not in capsys.readouterr().out


def test_aggregate_boards_counts_scored_rows_left_without_gain(tmp_path, capsys):
    cell_dir = _inline_cell_without_phi0(tmp_path)
    _write_bare_potentials(cell_dir, {"b": -7.0})

    _, overall_rows, (summary,) = aggregate_boards(cell_dir)

    assert _by_board(overall_rows)["a"]["potential_gain_avg"] is None
    assert summary["n_scored_without_potential_gain"] == 1
    assert "1 scored row(s) have no potential_gain" in capsys.readouterr().out


@pytest.mark.parametrize("env_phi0_matches", [False, True])
def test_aggregate_inline_prices_failed_attempts_at_the_scored_rows_phi0(env_phi0_matches, capsys):
    """A crash row carries the env's Φ₀, a scored row the scorer's: the board's Φ₀ was whichever
    row came first. The env's Φ₀ prices a board without a scored row only when the scored rows
    record that it is the scorer's (same Φ); otherwise that board has no Φ₀, with a warning."""
    rows = [
        {"board_index": 0, "board_id": "a", "rollout_idx": 0, "status": "crashed",
         "initial_potential": -50.0, "final_potential": -50.0},
        {"board_index": 0, "board_id": "a", "rollout_idx": 1, "status": "completed",
         "eval_status": "ok", "initial_potential": -10.0, "final_potential": 2.0,
         "potential_gain": 12.0, "success": True, "env_phi0_matches": str(env_phi0_matches)},
        # scored run: board b's crash row carries only the env's Φ₀ -> no Φ₀
        {"board_index": 1, "board_id": "b", "rollout_idx": 0, "status": "crashed",
         "initial_potential": -60.0, "final_potential": -60.0},
    ]
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", "")]

    per_board, overall = aggregate_inline(rows, boards, aggregation_mode="mean",
                                          selection_mode=None)

    assert per_board[0]["final_potential"] == pytest.approx((-10.0 + 2.0) / 2)
    assert per_board[0]["initial_potential"] == pytest.approx(-10.0)
    if env_phi0_matches:
        assert per_board[1]["final_potential"] == pytest.approx(-60.0)
        assert overall["n_failed_boards_without_phi0"] == 0
    else:
        assert math.isnan(per_board[1]["final_potential"])
        assert overall["n_failed_boards_without_phi0"] == 1
        assert "1 board(s) with a failed or unscored attempt have no Φ₀" in capsys.readouterr().out


def test_aggregate_boards_refuses_filling_phi0_scored_by_another_config(tmp_path):
    """A row without Φ₀ scored with another config than Stage 2 priced with (a mixed cell)."""
    cell_dir = _inline_cell_without_phi0(tmp_path)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    rows[1]["reward_config"] = "other_rule"
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"a": -5.0, "b": -7.0})

    with pytest.raises(SystemExit, match="several reward_config values"):   # refused as mixed rows first
        aggregate_boards(cell_dir)


def test_flatten_rollout_artifacts_stamps_the_ckpt_on_every_row(tmp_path):
    cell_dir = tmp_path / "mycell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, ckpt=7),
        _cell_row(c, "a", 0, 1, artifact=False, crashed=True),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    assert core.flatten_rollout_artifacts(cell_dir, ckpt_seed=7) == 0

    assert [r["ckpt"] for r in u.read_csv(cell_dir / "per_rollout.csv")] == ["7", "7"]


def test_failed_board_inputs_cover_failed_rows_and_boards_without_rows():
    c = "cell"
    rows = [
        _cell_row(c, "a", 0, 0),
        _cell_row(c, "b", 1, 0, artifact=False, crashed=True, board_path="/in/b_row.kicad_pcb"),
        _cell_row(c, "c", 2, 0, artifact=False, crashed=True),
    ]
    boards = [BoardSpec(i, bid, f"/in/{bid}.kicad_pcb") for i, bid in enumerate("abcd")]

    need = core._failed_board_inputs(rows, Path(c), boards)

    # a routed everywhere: no Φ₀ needed; the row's own input path wins
    assert need == {"b": "/in/b_row.kicad_pcb", "c": "/in/c.kicad_pcb",
                    "d": "/in/d.kicad_pcb"}


def test_score_bare_boards_records_phi0_and_reruns_incrementally(monkeypatch, tmp_path):
    import eval.metrics as metrics

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    inp = tmp_path / "b.kicad_pcb"
    inp.write_text("pcb")
    rows = [_cell_row(cell_dir.name, "a", 0, 0),
            _cell_row(cell_dir.name, "b", 1, 0, artifact=False, crashed=True)]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", str(inp)),
              BoardSpec(2, "c", str(tmp_path / "missing.kicad_pcb"))]
    scored = []

    def fake_evaluate_one(pcb, pro, reward_config_name, check_angle):
        scored.append(pcb)
        return {"initial_potential": -12.5, "final_potential": -12.5}

    monkeypatch.setattr(metrics, "evaluate_one", fake_evaluate_one)

    out = core.score_bare_boards(cell_dir, boards, reward_config="pcbworld_reward")

    assert scored == [str(inp)]
    rec = {r["board_id"]: r for r in u.read_csv(cell_dir / u.BARE_BOARD_POTENTIALS_CSV)}
    assert rec["b"]["initial_potential"] == "-12.5" and rec["b"]["eval_status"] == "ok"
    assert rec["c"]["eval_status"] == "no_input_board"
    assert {r["board_id"] for r in out} == {"b", "c"}
    # rerun: the ok board is not scored again
    core.score_bare_boards(cell_dir, boards, reward_config="pcbworld_reward")
    assert scored == [str(inp)]


def test_score_bare_boards_takes_the_listed_boards_project_file(monkeypatch, tmp_path):
    # PCBench: the row's input is <stem>_unrouted.kicad_pcb with no project file of
    # its own; the design rules are the board-list entry's <stem>.kicad_pro.
    import eval.metrics as metrics

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    bdir = tmp_path / "b"
    bdir.mkdir()
    (bdir / "v3_unrouted.kicad_pcb").write_text("pcb")
    (bdir / "v3.kicad_pcb").write_text("pcb")
    (bdir / "v3.kicad_pro").write_text("pro")
    rows = [_cell_row(cell_dir.name, "a", 0, 0),
            _cell_row(cell_dir.name, "b", 1, 0, artifact=False, crashed=True,
                      board_path=str(bdir / "v3_unrouted.kicad_pcb"))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    calls = []
    monkeypatch.setattr(metrics, "evaluate_one", lambda pcb, pro, **kw: (
        calls.append((pcb, pro)) or {"initial_potential": 1.0}))

    core.score_bare_boards(cell_dir, [BoardSpec(0, "a", ""), BoardSpec(1, "b", str(bdir / "v3.kicad_pcb"))])

    assert calls == [(str(bdir / "v3_unrouted.kicad_pcb"), str(bdir / "v3.kicad_pro"))]


def test_skip_rollout_boards_list_is_the_expected_board_set(monkeypatch, tmp_path):
    output_dir = tmp_path / "cell"
    output_dir.mkdir()
    rows = [_cell_row(output_dir.name, "a", 0, 0, **_scored(True, True, 1.0, 10.0, 5.0))]
    u.write_csv(output_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    boards_list = tmp_path / "boards.txt"
    boards_list.write_text("unused")
    boards = [BoardSpec(0, "a", "a.kicad_pcb"), BoardSpec(1, "b", "b.kicad_pcb")]
    seen = {}
    monkeypatch.setattr(core, "load_boards_from_dir_or_list",
                        lambda *, boards_dir=None, boards_list=None: boards)
    monkeypatch.setattr(core, "eval_kicad_pcb", lambda *a, **k: [])
    monkeypatch.setattr(core, "score_bare_boards",
                        lambda cell, b, **k: seen.setdefault("boards", b))
    monkeypatch.setattr(sys, "argv", [
        "eval", "--stages", "eval,aggregate", "--boards-list", str(boards_list),
        "--output-dir", str(output_dir), "--reward-config", "pcbworld_reward", "--check-angle", "45",
    ])

    assert core.main() == 0

    assert seen["boards"] == boards  # Stage 2 prices the failed boards' Φ₀
    (summary,) = u.read_csv(output_dir / "per_boards_summary.csv")
    assert summary["n_boards"] == "2"
    assert summary["n_failed"] == "1"
    assert summary["expected_boards"] == "given"


def test_aggregate_cli_records_its_run_beside_the_cell_outputs(monkeypatch, tmp_path):
    import yaml

    from configs.loader.record import _sha256
    from eval import aggregation

    cell = tmp_path / "cell"
    cell.mkdir()
    rows = [_cell_row(cell.name, "a", 0, 0, **_scored(True, True, 1.0, 10.0, 5.0))]
    u.write_csv(cell / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    rollout_record = cell / "run_record.yaml"
    rollout_record.write_text("schema: 2\n")
    monkeypatch.setattr(sys, "argv", ["aggregation", "--cell", str(cell),
                                      "--selection-mode", "posthoc_drc_aware"])

    aggregation._aggregate_boards_cli()

    rec = yaml.safe_load((cell / u.AGGREGATE_RUN_RECORD).read_text())
    assert rec["config"]["selection_mode"]["value"] == "posthoc_drc_aware"
    assert rec["extra"]["inputs"][0]["records"] == {"run_record.yaml": _sha256(rollout_record)}
    assert (cell / "per_boards_summary.csv").is_file()


def _rescore(monkeypatch, cell_dir, *flags):
    """Run Stage 2 alone on ``cell_dir``; return the scoring kwargs it ran with."""
    seen = {}
    monkeypatch.setattr(core, "eval_kicad_pcb", lambda cell, **k: seen.update(k) or [])
    monkeypatch.setattr(core, "score_bare_boards", lambda cell, boards, **k: None)
    monkeypatch.setattr(sys, "argv", ["eval", "--stages", "eval", "--output-dir", str(cell_dir), *flags])
    assert core.main() == 0
    return {k: seen[k] for k in core.DRC_SCORING_KEYS}


def _scored_cell(tmp_path, *scorings):
    """A cell whose rows were scored with ``scorings`` (``(reward_config, check_angle)`` each;
    none = an unscored rule-based cell)."""
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    rows = [_cell_row(cell_dir.name, "a", 0, i, eval_status="ok" if sc else "",
                      reward_config=sc[0] if sc else "", check_angle=sc[1] if sc else "")
            for i, sc in enumerate(scorings or [()])]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return cell_dir


def test_rescoring_keeps_the_cells_recorded_scoring(monkeypatch, tmp_path):
    # A 90-degree D1 cell re-scored without flags keeps 90 (it was the 45 default), and the
    # re-scoring run records where its settings came from.
    import yaml

    cell_dir = _scored_cell(tmp_path, ("drc_dense_errors_only_eval", 90))

    assert _rescore(monkeypatch, cell_dir) == {"reward_config": "drc_dense_errors_only_eval",
                                              "check_angle": 90}
    record = yaml.safe_load((cell_dir / core.RESCORE_RUN_RECORD).read_text())
    assert record["extra"]["scoring"] == {"reward_config": "drc_dense_errors_only_eval", "check_angle": 90}
    assert record["config"]["check_angle"]["source"] == "cell"
    assert "per_rollout.csv column check_angle" in record["config"]["check_angle"]["where"]


def test_rescoring_takes_an_explicit_flag_over_the_cells_record_and_records_both(monkeypatch, tmp_path, capsys):
    # The user re-scores a cell under another setting on purpose (user 2026-10-19): the flag wins,
    # with one warning naming both values, and the re-scoring record keeps the cell's value too.
    import yaml

    cell_dir = _scored_cell(tmp_path, ("drc_dense_errors_only_eval", 90))

    assert _rescore(monkeypatch, cell_dir, "--check-angle", "45")["check_angle"] == 45
    warnings = [ln for ln in capsys.readouterr().out.splitlines() if "WARNING" in ln]
    assert len(warnings) == 1
    assert "--check-angle 45 differs from the check_angle the cell records, 90" in warnings[0]
    entry = yaml.safe_load((cell_dir / core.RESCORE_RUN_RECORD).read_text())["config"]["check_angle"]
    assert (entry["value"], entry["source"]) == (45, "cli")
    assert [(c["source"], c["value"]) for c in entry["chain"]] == [("cell", 90), ("cli", 45)]
    # the re-scored cell now records 45: the next flagless re-scoring keeps it
    assert _rescore(monkeypatch, cell_dir)["check_angle"] == 45
    # restating the record is fine, and warns about nothing
    capsys.readouterr()
    assert _rescore(monkeypatch, cell_dir, "--check-angle", "45")["check_angle"] == 45
    assert "WARNING" not in capsys.readouterr().out


def test_rescoring_a_cell_that_records_no_scoring_needs_the_flags(monkeypatch, tmp_path):
    from configs.loader.resolve import ConfigError

    cell_dir = _scored_cell(tmp_path)  # rule-based rollout: rows without scoring columns

    with pytest.raises(ConfigError, match="records no reward_config / check_angle .*"
                                          "pass --reward-config --check-angle"):
        _rescore(monkeypatch, cell_dir)
    with pytest.raises(ConfigError, match="records no reward_config.*pass --reward-config"):
        _rescore(monkeypatch, cell_dir, "--check-angle", "90")
    with pytest.raises(ConfigError, match="records no check_angle.*pass --check-angle"):
        _rescore(monkeypatch, cell_dir, "--reward-config", "pcbworld_reward")
    assert _rescore(monkeypatch, cell_dir, "--reward-config", "pcbworld_reward", "--check-angle", "90") == {
        "reward_config": "pcbworld_reward", "check_angle": 90}
    # ... and the cell now records them: the next re-scoring needs no flag
    assert _rescore(monkeypatch, cell_dir)["check_angle"] == 90


def test_rescoring_names_every_scoring_problem_in_one_error(monkeypatch, tmp_path):
    from configs.loader.resolve import ConfigError

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    rows = [_cell_row(cell_dir.name, "a", 0, i, eval_status="", reward_config="", check_angle=a)
            for i, a in enumerate((45, 90))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(ConfigError, match=r"several check_angle values \[45, 90\].*; .*records no "
                                          r"reward_config \(its rollout scored nothing"):
        _rescore(monkeypatch, cell_dir)


def test_rescoring_a_directory_with_no_record_says_so(monkeypatch, tmp_path):
    from configs.loader.resolve import ConfigError

    (tmp_path / "not_the_cell").mkdir()
    with pytest.raises(ConfigError, match="holds no run_record.yaml.*is it the cell"):
        _rescore(monkeypatch, tmp_path / "not_the_cell")
    # a rule-based cell is only its routed boards before its first scoring: that is a cell
    (tmp_path / "rb" / "boards").mkdir(parents=True)
    (tmp_path / "rb" / "boards" / "board_a_rb_s00_r00.kicad_pcb").write_text("")
    with pytest.raises(ConfigError, match="its rollout scored nothing"):
        _rescore(monkeypatch, tmp_path / "rb")


def test_rescoring_rows_scored_two_ways_need_the_flag_to_choose(monkeypatch, tmp_path):
    from configs.loader.resolve import ConfigError

    cell_dir = _scored_cell(tmp_path, ("pcbworld_reward", 45), ("pcbworld_reward", 90))

    with pytest.raises(ConfigError, match=r"several check_angle values \[45, 90\]"):
        _rescore(monkeypatch, cell_dir)
    assert _rescore(monkeypatch, cell_dir, "--check-angle", "90")["check_angle"] == 90


def _reaggregate(monkeypatch, cell_dir, *flags):
    """Run Stage 3 alone on ``cell_dir``; return the selection it aggregated with."""
    from eval import aggregation

    seen = {}
    monkeypatch.setattr(aggregation, "aggregate_boards", lambda cell, **k: seen.update(k))
    monkeypatch.setattr(sys, "argv", ["eval", "--stages", "aggregate", "--output-dir", str(cell_dir), *flags])
    assert core.main() == 0
    return seen["selection_mode"]


def test_rescoring_keeps_the_cells_recorded_selection(monkeypatch, tmp_path, capsys):
    """Stage 3 of a re-scoring run selects per board as the cell's rollout did (it fell back to
    the default final_potential): from the run record, else the rollout's manifest when that run
    aggregated (cells from before the record carried it); another --selection-method wins, warned."""
    import json

    import yaml

    cell_dir = _scored_cell(tmp_path, ("pcbworld_reward", 45))
    manifest = cell_dir / "manifest.json"
    manifest.write_text(json.dumps({"args": {"selection_method": "posthoc_drc_aware", "skip_aggregate": False}}))
    assert _reaggregate(monkeypatch, cell_dir) == "posthoc_drc_aware"
    (cell_dir / "run_record.yaml").write_text(yaml.safe_dump(
        {"extra": {"scoring": {"selection_method": "posthoc_drc_aware"}}}))
    manifest.unlink()
    assert _reaggregate(monkeypatch, cell_dir, "--selection-method", "posthoc_drc_aware") == "posthoc_drc_aware"
    capsys.readouterr()
    assert _reaggregate(monkeypatch, cell_dir, "--selection-method", "final_potential") == "final_potential"
    assert ("--selection-method 'final_potential' differs from the selection_method the cell records, "
            "'posthoc_drc_aware'") in capsys.readouterr().out


@pytest.mark.parametrize("manifest", [{"args": {"selection_method": "posthoc_drc_aware", "skip_aggregate": True}},
                                      {"env_kwargs": {}}])
def test_a_manifest_records_no_selection_unless_its_run_aggregated(tmp_path, manifest):
    """A rollout that skipped Stage 3 wrote only the option's default into its manifest; a
    manifest without ``args`` (older pipelines) records nothing either."""
    import json

    cell_dir = _scored_cell(tmp_path, ("pcbworld_reward", 45))
    (cell_dir / "manifest.json").write_text(json.dumps(manifest))
    assert "selection_method" not in core.recorded_scoring(cell_dir)


def test_an_aggregate_only_rescoring_records_its_selection(monkeypatch, tmp_path):
    """A ``--stages aggregate`` run records the selection it chose (and keeps the cell's DRC
    scoring in that record), so the next flagless run selects the same way."""
    import yaml

    cell_dir = _scored_cell(tmp_path, ("drc_dense_errors_only_eval", 90))
    assert _reaggregate(monkeypatch, cell_dir, "--selection-method", "posthoc_drc_aware") == "posthoc_drc_aware"
    record = yaml.safe_load((cell_dir / core.RESCORE_RUN_RECORD).read_text())
    assert record["extra"]["scoring"] == {"reward_config": "drc_dense_errors_only_eval", "check_angle": 90,
                                          "selection_method": "posthoc_drc_aware"}
    assert _reaggregate(monkeypatch, cell_dir) == "posthoc_drc_aware"


def test_recorded_scoring_reads_the_run_record_before_the_rows(tmp_path):
    import yaml

    # rows half-rescored by an old default run; the rollout's record names the cell's scoring
    cell_dir = _scored_cell(tmp_path, ("shaped", 90), ("pcbworld_reward", 45))
    (cell_dir / "run_record.yaml").write_text(yaml.safe_dump(
        {"extra": {"scoring": {"reward_config": "shaped", "check_angle": 90}}}))

    assert core.recorded_scoring(cell_dir) == {
        "reward_config": (["shaped"], f"{cell_dir / 'run_record.yaml'} extra.scoring.reward_config"),
        "check_angle": ([90], f"{cell_dir / 'run_record.yaml'} extra.scoring.check_angle")}


def test_recorded_scoring_skips_the_rows_when_the_records_name_both_settings(monkeypatch, tmp_path):
    import yaml

    cell_dir = _scored_cell(tmp_path, ("shaped", 90))
    (cell_dir / "run_record.yaml").write_text(yaml.safe_dump(
        {"extra": {"scoring": {"reward_config": "shaped", "check_angle": 90}}}))
    monkeypatch.setattr(u, "read_csv", lambda path: pytest.fail(f"read {path}"))

    assert core.recorded_scoring(cell_dir)["check_angle"][0] == [90]


def _rollout_cell(monkeypatch, tmp_path, *flags, eval_transformer=None, ckpt_args=None):
    """A Stage 1 rollout (``--stages rollout``) of one board from a stub ckpt; returns the cell.
    ``eval_transformer`` replaces the stub rollout; ``ckpt_args`` are the ckpt's training args."""
    tmp_path.mkdir(exist_ok=True)
    output_dir = tmp_path / "cell"
    ckpt = tmp_path / "policy.pt"
    ckpt.write_bytes(b"unused")
    board = BoardSpec(0, "board_a", str(tmp_path / "board_a.kicad_pcb"))
    monkeypatch.setattr(core, "_resolve_device", lambda spec: "cpu")
    monkeypatch.setattr(core, "load_boards_from_dir_or_list", lambda **k: [board])
    monkeypatch.setattr(core, "load_policy_from_ckpt",
                        lambda ckpt, device, *, strict: (_policy(), ckpt_config(
                            seed=0, **(ckpt_args or {})), 0))
    monkeypatch.setattr(core, "apply_n_max_slots_override", lambda policy, slots, explicit: None)

    def fake_eval_transformer(policy, device, boards, **kwargs):
        rows = [_cell_row(output_dir.name, "board_a", 0, 0)]
        u.write_csv(kwargs["per_rollout_flush_path"], rows, u.PER_ROLLOUT_FIELDS)
        return EvalResult(per_rollout=rows)

    monkeypatch.setattr(core, "eval_transformer", eval_transformer or fake_eval_transformer)
    monkeypatch.setattr(sys, "argv", ["eval", "--ckpt", str(ckpt), "--boards-dir", str(tmp_path),
                                      "--seed", "0", "--n-rollouts", "1", "--n-envs", "1", "--rollout-mode", "serial",
                                      "--output-dir", str(output_dir), *flags])
    assert core.main() == 0
    return output_dir


def test_a_skip_drc_rollout_leaves_the_rule_to_its_first_scoring(monkeypatch, tmp_path):
    # A rollout that scores nothing records the env's angle but no rule (the CLI default would
    # pin a rule nothing used); the first re-scoring names the rule, the angle stays the env's.
    import yaml

    cell_dir = _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout")
    scoring = yaml.safe_load((cell_dir / "run_record.yaml").read_text())["extra"]["scoring"]
    corner = ckpt_config(seed=0)["corner_mode"]
    assert scoring == {"check_angle": corner}

    from configs.loader.resolve import ConfigError
    with pytest.raises(ConfigError, match=r"records no reward_config \(its rollout scored nothing, e\.g\. "
                                          r"--skip-drc.*pass --reward-config"):
        _rescore(monkeypatch, cell_dir)
    assert _rescore(monkeypatch, cell_dir, "--reward-config", "drc_dense_errors_only_eval") == {
        "reward_config": "drc_dense_errors_only_eval", "check_angle": corner}


def test_a_rollout_that_names_its_rule_records_it_even_unscored(monkeypatch, tmp_path):
    import yaml

    cell_dir = _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout", "--reward-config", "shaped")
    scoring = yaml.safe_load((cell_dir / "run_record.yaml").read_text())["extra"]["scoring"]
    assert scoring["reward_config"] == "shaped"


def _failing_rescore(monkeypatch, cell_dir, failing, *flags):
    """Re-score ``cell_dir`` (Stage 2 + 3) with the ``failing`` stage raising."""
    from eval import aggregation

    def boom(*a, **k):
        raise RuntimeError("stage gone")

    monkeypatch.setattr(core, "eval_kicad_pcb", boom if failing == "eval" else lambda cell, **k: [])
    monkeypatch.setattr(core, "score_bare_boards", lambda cell, boards, **k: None)
    monkeypatch.setattr(aggregation, "aggregate_boards", boom)
    monkeypatch.setattr(sys, "argv", ["eval", "--stages", "eval,aggregate", "--output-dir", str(cell_dir), *flags])
    with pytest.raises(RuntimeError, match="stage gone"):
        core.main()


def test_a_failed_stage2_leaves_the_rescore_record_as_it_was(monkeypatch, tmp_path):
    cell_dir = _scored_cell(tmp_path, ("drc_dense_errors_only_eval", 90))
    _failing_rescore(monkeypatch, cell_dir, "eval")
    assert not (cell_dir / core.RESCORE_RUN_RECORD).exists()

    _rescore(monkeypatch, cell_dir)
    before = (cell_dir / core.RESCORE_RUN_RECORD).read_bytes()
    _failing_rescore(monkeypatch, cell_dir, "eval", "--check-angle", "45")
    assert (cell_dir / core.RESCORE_RUN_RECORD).read_bytes() == before


def test_a_failed_stage3_keeps_the_scoring_stage2_wrote(monkeypatch, tmp_path):
    # Stage 2 re-scored the rows under 45: the record must say 45 even though Stage 3 failed,
    # with the selection still the cell's (not the run's --selection-method).
    import json

    import yaml

    cell_dir = _scored_cell(tmp_path, ("drc_dense_errors_only_eval", 90))
    (cell_dir / "manifest.json").write_text(json.dumps(
        {"args": {"selection_method": "posthoc_drc_aware", "skip_aggregate": False}}))
    _failing_rescore(monkeypatch, cell_dir, "aggregate", "--check-angle", "45",
                     "--selection-method", "final_potential")
    record = yaml.safe_load((cell_dir / core.RESCORE_RUN_RECORD).read_text())
    assert record["extra"]["scoring"] == {"reward_config": "drc_dense_errors_only_eval", "check_angle": 45,
                                          "selection_method": "posthoc_drc_aware"}
    assert record["config"]["selection_method"]["source"] == "cell"


def test_a_dry_run_rescoring_writes_no_record(monkeypatch, tmp_path, capsys):
    cell_dir = _scored_cell(tmp_path, ("drc_dense_errors_only_eval", 90))
    monkeypatch.setattr(core, "eval_kicad_pcb", lambda cell, **k: [])
    monkeypatch.setattr(sys, "argv", ["eval", "--stages", "eval,aggregate", "--dry-run",
                                      "--check-angle", "45", "--output-dir", str(cell_dir)])
    assert core.main() == 0
    assert not (cell_dir / core.RESCORE_RUN_RECORD).exists()
    warning = next(ln for ln in capsys.readouterr().out.splitlines() if "differs from the check_angle" in ln)
    assert "records both" not in warning


def test_the_last_rescoring_record_supersedes_the_rollouts(monkeypatch, tmp_path, capsys):
    # A cell re-scored under 45 after a 90 rollout records 45 (its rows were scored last): a
    # flagless re-scoring keeps 45, and a differing flag names that record.
    import yaml

    cell_dir = _scored_cell(tmp_path, ("shaped", 45))
    for name, angle in (("run_record.yaml", 90), (core.RESCORE_RUN_RECORD, 45)):
        (cell_dir / name).write_text(yaml.safe_dump(
            {"extra": {"scoring": {"reward_config": "shaped", "check_angle": angle}}}))
    rescore_record = cell_dir / core.RESCORE_RUN_RECORD

    assert core.recorded_scoring(cell_dir)["check_angle"] == (
        [45], f"{rescore_record} extra.scoring.check_angle")
    assert _rescore(monkeypatch, cell_dir, "--check-angle", "90")["check_angle"] == 90
    assert f"the cell records, 45 ({rescore_record} extra.scoring.check_angle)" in capsys.readouterr().out


def test_rescoring_takes_a_rule_path_for_the_rule_name_the_cell_records(monkeypatch, tmp_path):
    cell_dir = _scored_cell(tmp_path, ("drc_dense_errors_only_eval", 90))
    path = "experiments/kdd/configs/reward/drc_dense_errors_only_eval.yaml"

    assert _rescore(monkeypatch, cell_dir, "--reward-config", path)["reward_config"] == path


# ---------------------------------------------------------------------------
# Inline (training val) aggregation: same failed-attempt rule
# ---------------------------------------------------------------------------


def _inline_row(board_index, rollout_idx, status="completed", **values):
    return {"board_index": board_index, "rollout_idx": rollout_idx, "status": status,
            **values}


def test_aggregate_inline_scores_failed_attempts_like_stage3():
    rows = [
        _inline_row(0, 0, success=True, final_potential=4.0, initial_potential=-6.0,
                    wirelength_mm=10.0),
        # crashed: priced at the board's Φ₀ from its sibling row, never selected
        _inline_row(0, 1, "crashed", success=False, final_potential=50.0,
                    wirelength_mm=99.0),
        # incompatible board: a failure, no longer dropped from the denominator
        _inline_row(1, 0, "skipped_incompatible_board", initial_potential=-3.0),
    ]
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", ""), BoardSpec(2, "c", "")]

    per_board, overall = aggregate_inline(rows, boards, selection_mode="final_potential")

    a, b, c = per_board
    assert a["final_potential"] == pytest.approx(-1.0)  # mean(4.0, Φ₀ -6.0)
    assert a["wirelength_mm"] == 10.0
    assert a["success"] == 0.5
    assert a["selected_by_mode_rollout_idx"] == 0
    assert b["final_potential"] == -3.0 and b["success"] == 0.0
    assert c["success"] == 0.0 and math.isnan(c["final_potential"])  # no Φ₀ known
    assert overall["n_failed"] == 3
    assert overall["n_crashed"] == 1
    assert overall["success"] == pytest.approx(0.5 / 3)
    assert overall["selected_success_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Review round 1: Φ₀ reaches the inline rows; edge cases of the failure rule
# ---------------------------------------------------------------------------


class _Phi0Pool:
    """Fake pool for ``_run_one_batch``: every reset reports Φ₀ in its info."""

    def __init__(self, phi0, n_nets=3):
        self.phi0 = phi0
        self._obs = {"board_static": {"nets": {i: {} for i in range(n_nets)}}}

    def reload_boards(self, board_paths):
        self.boards = list(board_paths)

    def reset_batch(self, indices, seeds=None):
        return [(self._obs, {"initial_potential": self.phi0}) for _ in indices]


def _run_one_batch(pool, *, max_net_slots):
    from methods.rl_agent.rollout.transformer import _run_one_batch as run

    return run(
        pool=pool, policy=None, device=None,
        jobs=[(BoardSpec(0, "a", "/tmp/a.kicad_pcb"), 0)],
        n_envs=1, max_steps=5, base_seed=0, deterministic=False,
        skip_incompatible=True, max_net_slots=max_net_slots,
        early_stop_finish_no_progress=0, early_stop_no_geometry_progress=0,
        save_artifacts=False, artifacts_dir=None, final_drc=False,
        reward_config="", check_angle=0,
    )


def test_incompatible_rollout_row_carries_the_reset_phi0():
    (row,) = _run_one_batch(_Phi0Pool(-7.5), max_net_slots=0)
    assert row["status"] == "skipped_incompatible_board"
    assert row["initial_potential"] == -7.5


def test_engine_crash_rollout_row_carries_the_reset_phi0(monkeypatch):
    import types

    import numpy as np

    from methods.rl_agent.rollout import transformer

    def crash_once(pool, policy, device, obs_by_slot, active_slots, done, **kw):
        # the pool's synthesized step after a worker death: no env state
        yield types.SimpleNamespace(
            actions=np.zeros((1, 1), dtype=np.int64), live=[0], rewards=[-1.0],
            infos=[{"engine_crash": True, "action_success": False}],
            terminateds=[True], truncateds=[False])

    monkeypatch.setattr(transformer, "iter_rollout", crash_once)
    (row,) = _run_one_batch(_Phi0Pool(-7.5), max_net_slots=99)
    assert row["status"] == "crashed"
    assert row["initial_potential"] == -7.5


def test_env_reset_info_reports_the_bare_board_phi0():
    from pcb_world.engine import engine_available

    if not engine_available():
        pytest.skip("kicad_rl_router not available")
    from pcb_world.core.env import PCBWorld

    board = Path(__file__).resolve().parent / "fixtures" / "simple_routing_board.kicad_pcb"
    env = PCBWorld(str(board), EnvConfig.with_overrides(max_steps=10))
    try:
        _obs, info = env.reset()
        assert info["initial_potential"] == env._initial_potential
        assert math.isfinite(info["initial_potential"])
    finally:
        env.close()


def test_aggregate_inline_counts_and_warns_failed_boards_without_phi0(capsys):
    rows = [
        _inline_row(0, 0, success=True, final_potential=4.0, initial_potential=-6.0),
        _inline_row(1, 0, "crashed"),  # no Φ₀ anywhere for board b
    ]
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", "")]

    _, overall = aggregate_inline(rows, boards, selection_mode=None)

    assert overall["n_failed_boards_without_phi0"] == 1
    assert "1 board(s) with a failed or unscored attempt have no Φ₀" in capsys.readouterr().out


def test_scoring_errors_fail_only_a_missing_board(tmp_path):
    # Evaluator.score_boards decides at the source: a scoring error on an existing
    # board file leaves it completed but unscored (eval_status=error; a board with no
    # scored attempt is then priced like a failed attempt, counted in n_eval_error, as
    # in Stage 3); a missing board file is a failed attempt.
    from eval.evaluator import _rows_from_scored

    present = tmp_path / "b.kicad_pcb"
    present.write_text("pcb")
    nested = [{"success": True, "final_potential": 4.0, "routability": 1.0,
               "initial_potential": 1.0, "potential_gain": 3.0},
              {"error": "boom"}, {"error": "no such file"}]
    pairs = [(str(tmp_path / "a.kicad_pcb"), None), (str(present), None),
             (str(tmp_path / "missing.kicad_pcb"), None)]
    (tmp_path / "a.kicad_pcb").write_text("pcb")
    rows = _rows_from_scored(nested, pairs, reward_config="rc", check_angle=45)
    assert [r["status"] for r in rows] == ["completed", "completed", "crashed"]

    boards = [BoardSpec(i, r["board_id"], "") for i, r in enumerate(rows)]
    per_board, overall = aggregate_inline(rows, boards, selection_mode=None)

    assert overall["n_failed"] == 1 and overall["n_eval_error"] == 1
    assert per_board[1]["success"] == 0.0 and per_board[2]["success"] == 0.0


def test_aggregate_boards_scores_a_board_failed_in_every_ckpt_as_a_failure_best(tmp_path):
    # A pre-ckpt-column merged cell where board b failed in every ckpt: its
    # failures are unattributed, but every ckpt's best on b is a failure.
    cell_dir = tmp_path / "mergedcell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, ckpt=42, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "a", 0, 0, ckpt=43, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "b", 1, 0, artifact=False, crashed=True),
        _cell_row(c, "b", 1, 0, artifact=False, crashed=True),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    _, overall_rows, (summary,) = aggregate_boards(cell_dir)

    b = _by_board(overall_rows)["b"]
    assert b["success_best"] == 0.0 and b["clean_pass_best"] == 0.0
    assert summary["success_best"] == pytest.approx(0.5)


def test_score_bare_boards_caps_the_pool_at_the_job_count(monkeypatch, tmp_path):
    import pcb_world.vec.subproc_pool as sp

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    inp = tmp_path / "b.kicad_pcb"
    inp.write_text("pcb")
    rows = [_cell_row(cell_dir.name, "a", 0, 0),
            _cell_row(cell_dir.name, "b", 1, 0, artifact=False, crashed=True,
                      board_path=str(inp))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    seen = {}

    class FakePool:
        def __init__(self, n_workers, **kw):
            seen["n_workers"] = n_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def evaluate_many(self, jobs, on_result):
            for k, _ in enumerate(jobs):
                on_result(k, {"initial_potential": -1.0})

    monkeypatch.setattr(sp, "SubprocEvalPool", FakePool)
    core.score_bare_boards(cell_dir, None, n_workers=64)
    assert seen["n_workers"] == 1


def test_score_bare_boards_refuses_an_unflattened_cell_like_stage3(tmp_path):
    # Stage 2 and Stage 3 share one failed-attempt rule: an existing routed board
    # outside the cell grammar is a layout error in both, never a failed attempt.
    cell_dir = tmp_path / "cell"
    (cell_dir / "artifacts").mkdir(parents=True)
    stray = cell_dir / "artifacts" / "board_00000_rollout_0.kicad_pcb"
    stray.write_text("pcb")
    rows = [_cell_row(cell_dir.name, "a", 0, 0)]
    rows[0]["artifact_path"] = str(stray)
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit, match=r"^\[score_bare_boards\] .*not in the cell grammar"):
        core.score_bare_boards(cell_dir, None)


def test_aggregate_boards_refuses_a_ckpt_column_that_contradicts_the_artifact(tmp_path):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    rows = [_cell_row(cell_dir.name, "a", 0, 0, ckpt=42, **_scored(True, True, 1.0, 10.0, 5.0))]
    rows[0]["ckpt"] = 0
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit, match="ckpt column"):
        aggregate_boards(cell_dir)


def test_aggregate_boards_blanks_a_failed_rows_own_phi0_without_stage2s(tmp_path):
    # The crash row carries the env's Φ₀ (possibly another reward config than the
    # cell's scorer); with no Stage-2 Φ₀ for the board it must not enter the means.
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [
        _cell_row(c, "a", 0, 0, initial_potential=-5.0, **_scored(True, True, 1.0, 10.0, 5.0)),
        _cell_row(c, "a", 0, 1, artifact=False, crashed=True, initial_potential=-99.0),
    ]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    _, overall_rows, _ = aggregate_boards(cell_dir)

    assert _by_board(overall_rows)["a"]["initial_potential_avg"] == -5.0


def test_aggregate_boards_fails_a_board_missing_under_one_ckpt(tmp_path, capsys):
    # Board a routed under ckpt 0 only, b under 0 and 1: a@1 has no row at all
    # (interrupted / partly merged ckpt) and is one failed attempt, so the cell's
    # CP is (0.5 + 1.0) / 2, not 1.0.
    cell_dir = tmp_path / "gridcell"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [_cell_row(c, b, i, 0, ckpt=s, **_scored(True, True, 1.0, 10.0, 5.0))
            for b, i, s in (("a", 0, 0), ("b", 1, 0), ("b", 1, 1))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    expected = [BoardSpec(0, "a", ""), BoardSpec(1, "b", "")]

    ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir, expected_boards=expected)

    assert len(ckpt_rows) == 4
    a = _by_board(overall_rows)["a"]
    assert (a["n_ckpts"], a["n_failed"], a["clean_pass_best"]) == (2, 1, 0.5)
    assert summary["clean_pass_best"] == pytest.approx(0.75)
    assert summary["n_failed"] == 1
    assert "1 (board, ckpt) pair(s) have no row" in capsys.readouterr().out


def test_aggregate_boards_scores_a_cell_where_every_attempt_failed(tmp_path, capsys):
    cell_dir = tmp_path / "allfailed"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [_cell_row(c, b, i, 0, artifact=False, crashed=True) for b, i in (("a", 0), ("b", 1))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"a": -10.0, "b": -20.0})

    _ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    assert (summary["n_boards"], summary["n_failed"]) == (2, 2)
    assert summary["clean_pass_best"] == 0.0
    assert summary["success_best"] == 0.0
    assert summary["final_potential_best"] == pytest.approx(-15.0)
    assert (cell_dir / "per_boards_summary.csv").is_file()
    assert "every attempt failed" in capsys.readouterr().out


def test_aggregate_boards_leaves_no_previous_outputs_when_it_stops(tmp_path):
    # A stale summary from an earlier run must not survive a run that errors out.
    cell_dir = _three_board_cell(tmp_path)
    aggregate_boards(cell_dir)
    assert (cell_dir / "per_boards_summary.csv").is_file()

    with pytest.raises(SystemExit, match="not in the expected board set"):
        aggregate_boards(cell_dir, expected_boards=[BoardSpec(0, "a", "")])
    for name in ("per_boards_ckpts.csv", "per_boards_overall.csv", "per_boards_summary.csv"):
        assert not (cell_dir / name).exists()


def test_aggregate_boards_counts_an_expected_board_without_rows_in_a_ckpt_less_cell(tmp_path):
    # Every attempt failed and no row names a ckpt: an expected board with no row is
    # still one failed attempt, priced at its Φ₀.
    cell_dir = tmp_path / "allfailed_nockpt"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [_cell_row(c, b, i, 0, artifact=False, crashed=True) for b, i in (("a", 0), ("b", 1))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"a": -1.0, "b": -2.0, "c": -3.0})
    expected = [BoardSpec(0, "a", ""), BoardSpec(1, "b", ""), BoardSpec(2, "c", "")]

    _ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir, expected_boards=expected)

    c_row = _by_board(overall_rows)["c"]
    assert (c_row["n_runs"], c_row["n_failed"], c_row["clean_pass_best"]) == (1, 1, 0.0)
    assert (summary["n_boards"], summary["n_failed"]) == (3, 3)
    assert summary["final_potential_best"] == pytest.approx(-2.0)


def test_aggregate_boards_removes_previous_outputs_before_reading_the_rows(tmp_path):
    cell_dir = _three_board_cell(tmp_path)
    aggregate_boards(cell_dir)
    (cell_dir / "per_rollout.csv").write_text(",".join(u.PER_ROLLOUT_FIELDS) + "\n")

    with pytest.raises(ValueError, match="empty per_rollout.csv"):
        aggregate_boards(cell_dir)
    assert not (cell_dir / "per_boards_summary.csv").exists()


def test_aggregate_boards_refuses_a_cell_of_completed_rows_without_artifacts(tmp_path):
    # --save-artifacts off: every row completed, none saved a board — nothing to score,
    # not a cell where every attempt failed.
    cell_dir = tmp_path / "noartifacts"
    cell_dir.mkdir()
    rows = [_cell_row(cell_dir.name, b, i, 0, artifact=False, status="completed",
                      success=True, clean_pass=True) for b, i in (("a", 0), ("b", 1))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit, match="completed row.*saved no board"):
        aggregate_boards(cell_dir)
    assert not (cell_dir / "per_boards_summary.csv").exists()


def test_aggregate_inline_counts_api_failures_of_its_boards_only():
    """A row of a board outside ``boards`` stays out of overall, as it does of
    n_failed and the per-board rows."""
    rows = [{"board_index": i, "board_id": b, "rollout_idx": 0, "status": "completed",
             "final_potential": 1.0, "success": True, "api_failed_calls": n}
            for i, b, n in ((0, "a", "2"), (1, "b", "3"))]
    per_board, overall = aggregate_inline(rows, [BoardSpec(0, "a", "")],
                                          selection_mode="final_potential")
    assert [r["api_failed_calls"] for r in per_board] == [2]
    assert (overall["api_failed_calls"], overall["api_failed_boards"]) == (2, 1)


# ---------------------------------------------------------------------------
# Bool metrics read from CSV strings; unmeasured bools are not failures
# ---------------------------------------------------------------------------


def test_selection_overlay_reads_csv_string_bools():
    # rows read back from per_rollout.csv carry "False"/"True" strings, not bools
    rows = [_inline_row(0, 0, success="False", clean_pass="False", final_potential="1.0"),
            _inline_row(1, 0, success="True", clean_pass="True", final_potential="2.0")]
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", "")]

    _, overall = aggregate_inline(rows, boards, selection_mode="final_potential")

    assert overall["selected_success_rate"] == pytest.approx(0.5)
    assert overall["selected_clean_pass_rate"] == pytest.approx(0.5)


def _unscored_cell(tmp_path):
    # board a scored; board b: rollout 0 scored, rollout 1's Stage-2 scoring failed
    # (eval_status=error leaves clean_pass / routability blank; success stays from
    # the rollout)
    cell_dir = tmp_path / "unscored"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [_cell_row(c, "a", 0, 0, **_scored(True, True, 1.0, 10.0, 5.0)),
            _cell_row(c, "b", 1, 0, **_scored(True, True, 1.0, 10.0, 2.0)),
            _cell_row(c, "b", 1, 1, eval_status="error", eval_error="boom",
                      success=True, final_potential=3.0)]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return cell_dir


def test_aggregate_boards_skips_unscored_bools_in_best_like_avg(tmp_path, capsys):
    cell_dir = _unscored_cell(tmp_path)

    ckpt_rows, overall_rows, (summary,) = aggregate_boards(cell_dir)

    b = _by_board(overall_rows)["b"]
    # the unscored row's blank clean_pass is skipped (1.0, not 0.5); best = scored row
    assert b["clean_pass_avg"] == 1.0 and b["clean_pass_best"] == 1.0
    assert _by_board(ckpt_rows)["b"]["selected_rollout_idx"] == "0"
    assert b["success_avg"] == 1.0  # the env value the row still carries is read
    assert (summary["n_eval_error"], summary["n_failed"]) == (1, 0)
    out = capsys.readouterr().out
    assert "1 routed row(s) have eval_status=error" in out
    # the warning names both rules: env values in _avg, unscored groups as failures
    assert "enter _avg unrescored; a (board, ckpt) with no scored attempt" in out


def test_aggregate_boards_leaves_bools_blank_before_stage2(tmp_path):
    cell_dir = tmp_path / "prestage2"
    cell_dir.mkdir()
    rows = [_cell_row(cell_dir.name, b, i, 0, final_potential=1.0) for b, i in (("a", 0), ("b", 1))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    ckpt_rows, _, (summary,) = aggregate_boards(cell_dir)

    assert "clean_pass_best" not in summary and "clean_pass_avg" not in summary
    assert all(r["clean_pass_best"] is None for r in ckpt_rows)


def test_aggregate_inline_skips_unscored_bools_in_best_and_selection():
    rows = [_inline_row(0, 0, success=True, clean_pass=True, eval_status="ok",
                        final_potential=5.0),
            _inline_row(1, 0, success=True, clean_pass=True, eval_status="ok",
                        final_potential=1.0),
            _inline_row(1, 1, success=True, clean_pass="", eval_status="error",
                        final_potential=3.0)]
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", "")]

    per_board, overall = aggregate_inline(
        rows, boards, aggregation_mode="best", selection_mode="final_potential")
    avg_board, _ = aggregate_inline(rows, boards, aggregation_mode="mean", selection_mode=None)

    assert per_board[1]["clean_pass"] == 1.0 and per_board[1]["selected_rollout_idx"] == 0
    assert avg_board[1]["clean_pass"] == 1.0  # blank skipped, not read as 0
    assert overall["clean_pass"] == pytest.approx(1.0)
    assert overall["selected_clean_pass_rate"] == pytest.approx(1.0)
    assert overall["selected_success_rate"] == pytest.approx(1.0)
    assert overall["n_eval_error"] == 1 and overall["n_failed"] == 0


# ---------------------------------------------------------------------------
# Selection: a scored attempt beats one whose scoring failed (scored-first)
# ---------------------------------------------------------------------------


def _scoring_error_cell(tmp_path, *, all_errors=False):
    # board a: rollout 0 scored (fp 3); rollout 1's scoring failed, so it keeps the
    # env's fp 9 and wirelength — a value from another scorer, which must not win
    cell_dir = tmp_path / "scorederr"
    cell_dir.mkdir()
    c = cell_dir.name
    first = (dict(eval_status="error", success=True, final_potential=3.0, wirelength_mm=40.0)
             if all_errors else _scored(True, True, 1.0, 10.0, 3.0))
    rows = [_cell_row(c, "a", 0, 0, **first),
            _cell_row(c, "a", 0, 1, eval_status="error", success=True,
                      final_potential=9.0, wirelength_mm=99.0)]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return cell_dir


def test_aggregate_boards_selects_a_scored_attempt_over_a_scoring_error(tmp_path):
    ckpt_rows, overall_rows, _ = aggregate_boards(_scoring_error_cell(tmp_path))

    a = _by_board(overall_rows)["a"]
    assert ckpt_rows[0]["selected_rollout_idx"] == "0"
    assert a["final_potential_best"] == 3.0 and a["wirelength_mm_best"] == 10.0
    assert a["clean_pass_best"] == 1.0


def test_aggregate_boards_prices_a_ckpt_without_a_scored_attempt_as_failures(tmp_path):
    # every routed attempt of (a, s00) failed scoring: priced like a board left
    # untouched (Stage 2's Φ₀, pass-type 0, no WL), never by the env values
    cell_dir = _scoring_error_cell(tmp_path, all_errors=True)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    rows.append(_cell_row(cell_dir.name, "b", 1, 0, **_scored(True, True, 1.0, 10.0, 5.0)))
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"a": -7.0})

    _, overall_rows, (summary,) = aggregate_boards(cell_dir)

    a = _by_board(overall_rows)["a"]
    assert a["final_potential_best"] == -7.0 and a["final_potential_avg"] == -7.0
    assert a["success_best"] == 0.0 and a["clean_pass_best"] == 0.0
    assert a["wirelength_mm_best"] is None and a["wirelength_mm_avg"] is None
    assert (a["n_eval_error"], a["n_failed"]) == (2, 0)
    assert summary["n_boards"] == 2 and summary["success_best"] == pytest.approx(0.5)


def test_aggregate_inline_prices_a_board_without_a_scored_attempt_as_failures():
    rows = [_inline_row(0, 0, eval_status="error", success=True, final_potential=9.0,
                        initial_potential=-6.0, wirelength_mm=99.0),
            _inline_row(1, 0, eval_status="ok", success=True, clean_pass=True,
                        final_potential=2.0, wirelength_mm=10.0)]
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", "")]

    per_board, overall = aggregate_inline(
        rows, boards, aggregation_mode="best", selection_mode="final_potential")

    # a scored run: the unscored row's env Φ₀ (training rule) does not price board a
    assert math.isnan(per_board[0]["final_potential"]) and per_board[0]["success"] == 0.0
    assert per_board[0]["clean_pass"] == 0.0 and math.isnan(per_board[0]["wirelength_mm"])
    assert overall["n_eval_error"] == 1 and overall["n_failed"] == 0
    assert overall["n_failed_boards_without_phi0"] == 1
    assert overall["selected_success_rate"] == pytest.approx(0.5)


def test_failed_board_inputs_cover_a_board_without_a_scored_attempt():
    # Stage 3 prices (b, s00) at b's Φ₀, so Stage 2 must record it; a, with one
    # scored attempt beside an unscored one, needs none
    c = "cell"
    rows = [_cell_row(c, "a", 0, 0, eval_status="ok", initial_potential="-1.0"),
            _cell_row(c, "a", 0, 1, eval_status="error"),
            _cell_row(c, "b", 1, 0, eval_status="error", board_path="/in/b_row.kicad_pcb")]
    boards = [BoardSpec(i, bid, f"/in/{bid}.kicad_pcb") for i, bid in enumerate("ab")]

    assert core._failed_board_inputs(rows, Path(c), boards) == {"b": "/in/b_row.kicad_pcb"}


def test_aggregate_inline_selects_a_scored_attempt_over_a_scoring_error():
    rows = [_inline_row(0, 0, eval_status="ok", success=True, clean_pass=True, final_potential=3.0),
            _inline_row(0, 1, eval_status="error", success=False, final_potential=9.0)]

    per_board, overall = aggregate_inline(
        rows, [BoardSpec(0, "a", "")], aggregation_mode="best", selection_mode="final_potential")

    assert per_board[0]["selected_rollout_idx"] == 0
    assert per_board[0]["selected_by_mode_rollout_idx"] == 0
    assert overall["selected_final_potential_mean"] == 3.0
    assert overall["selected_success_rate"] == 1.0


@pytest.mark.parametrize("eval_status", ["ok", ""])
def test_scored_first_leaves_cells_without_scoring_errors_unchanged(tmp_path, eval_status):
    # every row scored (ok), or a cell where DRC never ran (blank): fp max still wins
    from eval.aggregation import _select_routed_first
    rows = [{"rollout_idx": i, "eval_status": eval_status, "final_potential": fp}
            for i, fp in enumerate((1.0, 7.0, 4.0))]
    for mode in ("final_potential", "posthoc_drc_aware"):
        assert _select_routed_first(rows, mode) == u.select_best_in_board(rows, mode)
    assert _select_routed_first(rows, "final_potential")[0] == 1


def test_scored_first_leaves_a_failed_only_board_to_routed_first():
    # failed attempts are all priced at Φ₀: eval_status must not change which one wins
    from eval.aggregation import _FAILED, _select_routed_first
    rows = [{"rollout_idx": 0, "eval_status": "error", "final_potential": -5.0, _FAILED: True},
            {"rollout_idx": 1, "eval_status": "", "final_potential": -5.0, _FAILED: True}]
    for mode in ("final_potential", "posthoc_drc_aware"):
        assert _select_routed_first(rows, mode) == u.select_best_in_board(rows, mode)


def test_aggregate_boards_counts_an_unscored_and_a_crashed_attempt_apart(tmp_path, capsys):
    # (a, s00): one routed attempt whose scoring failed and one crash — both priced as
    # failures, counted once each; no Φ₀ recorded, so the warning names unscored too
    cell_dir = tmp_path / "mixed"
    cell_dir.mkdir()
    c = cell_dir.name
    rows = [_cell_row(c, "a", 0, 0, eval_status="error", success=True, final_potential=9.0,
                      wirelength_mm=99.0),
            _cell_row(c, "a", 0, 1, artifact=False, crashed=True),
            _cell_row(c, "b", 1, 0, **_scored(True, True, 1.0, 10.0, 5.0))]
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    ckpt_rows, overall_rows, _ = aggregate_boards(cell_dir)

    a = _by_board(overall_rows)["a"]
    assert (a["n_failed"], a["n_crashed"], a["n_eval_error"]) == (1, 1, 1)
    assert a["success_best"] == 0.0 and a["final_potential_best"] is None
    assert a["wirelength_mm_avg"] is None
    assert "1 board(s) with a failed or unscored attempt have no Φ₀" in capsys.readouterr().out


def test_aggregate_boards_refuses_a_cell_with_no_scored_routed_row(tmp_path):
    # every routed row failed scoring (a crash beside them does not count as scored):
    # a scorer failure, not a 0% result — stop with the cause, leave no outputs
    cell_dir = _scoring_error_cell(tmp_path, all_errors=True)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    rows[0]["eval_error"] = "RuntimeError: DRC engine gone\nTraceback (most recent call last):"
    for r in rows:
        r.update(reward_config="pcbworld_reward", check_angle=90)
    rows.append(_cell_row(cell_dir.name, "b", 1, 0, artifact=False, crashed=True))
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"a": -7.0, "b": -9.0})
    (cell_dir / "per_boards_summary.csv").write_text("stale\n")

    with pytest.raises(SystemExit) as exc:
        aggregate_boards(cell_dir)

    msg = str(exc.value)
    assert "none of the 2 routed row(s) was scored" in msg
    assert "RuntimeError: DRC engine gone" in msg and "Traceback" not in msg
    assert "--stages eval,aggregate" in msg
    # the message names the scoring the rerun reuses
    assert "the rows record: reward_config=pcbworld_reward ×2; check_angle=90 ×2)" in msg
    assert "--boards-dir" in msg
    assert not (cell_dir / "per_boards_summary.csv").exists()


def test_aggregate_boards_scorer_failure_names_disagreeing_scoring_flags(tmp_path):
    # rows scored under two check angles, a config recorded nowhere: name what is there
    cell_dir = _scoring_error_cell(tmp_path, all_errors=True)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    for r, angle in zip(rows, (45, 90)):
        r.update(check_angle=angle)
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    with pytest.raises(SystemExit) as exc:
        aggregate_boards(cell_dir)

    msg = str(exc.value)
    assert "check_angle=45 ×1, 90 ×1 (the rows disagree)" in msg
    assert "reward_config=<blank> ×2;" in msg  # recorded nowhere


def test_aggregate_boards_stops_on_a_scorer_failure_before_reading_bare_potentials(
        tmp_path, capsys, monkeypatch):
    # the scorer failure is the cause: no Φ₀ config clash and no later warning hides it
    import eval.aggregation as agg
    monkeypatch.setattr(agg, "_bare_board_potentials",
                        lambda *a, **k: pytest.fail("Φ₀ read before the scorer-failure stop"))
    cell_dir = _scoring_error_cell(tmp_path, all_errors=True)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    for r in rows:
        r.update(reward_config="pcbworld_reward")
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"a": -7.0}, reward_config="other_reward")

    with pytest.raises(SystemExit, match="a scorer failure"):
        aggregate_boards(cell_dir)
    assert "WARNING" not in capsys.readouterr().out


def test_aggregate_boards_reports_a_wrong_board_set_before_a_scorer_failure(tmp_path):
    # both wrong: the board-set error comes first, so the Stage 2 rerun is not wasted
    cell_dir = _scoring_error_cell(tmp_path, all_errors=True)

    with pytest.raises(SystemExit, match="not in the expected board set"):
        aggregate_boards(cell_dir, expected_boards=[BoardSpec(0, "z", "")])


def test_aggregate_boards_keeps_a_cell_with_one_scored_routed_row(tmp_path):
    # one scored routed row anywhere in the cell: the per-group rule prices the
    # unscored groups as failures and the cell is a result
    cell_dir = _scoring_error_cell(tmp_path, all_errors=True)
    rows = u.read_csv(cell_dir / "per_rollout.csv")
    rows.append(_cell_row(cell_dir.name, "a", 0, 0, ckpt=1, **_scored(True, True, 1.0, 10.0, 5.0)))
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    _write_bare_potentials(cell_dir, {"a": -7.0})

    ckpt_rows, _, (summary,) = aggregate_boards(cell_dir)

    by_ck = {r["ckpt"]: r for r in ckpt_rows}
    assert by_ck[0]["final_potential_best"] == -7.0 and by_ck[0]["success_best"] == 0.0
    assert by_ck[1]["final_potential_best"] == 5.0
    assert summary["n_eval_error"] == 2 and summary["success_best"] == pytest.approx(0.5)


def test_aggregate_inline_keeps_a_run_with_no_scored_attempt_and_warns(capsys):
    # inline has no Stage 2 to rerun; the run's rows still go out (Stage 3 is the gate)
    rows = [_inline_row(0, 0, eval_status="error", success=True, final_potential=9.0,
                        initial_potential=-6.0)]

    per_board, overall = aggregate_inline(
        rows, [BoardSpec(0, "a", "")], aggregation_mode="best", selection_mode=None)

    assert per_board[0]["final_potential"] == -6.0 and per_board[0]["success"] == 0.0
    assert overall["n_eval_error"] == 1
    out = capsys.readouterr().out
    assert "no routed attempt was scored" in out and "not a result" in out


@pytest.mark.parametrize("argv,angle", [
    ([], 45),                                          # the checkpoint's corner mode
    (["--corner-mode", "90"], 90),                     # overridden: scored at the routed angle
    (["--corner-mode", "90", "--check-angle", "90"], 90),   # an explicit value restating it
])
def test_check_angle_follows_the_final_env_corner_mode(tmp_path, monkeypatch, argv, angle):
    seen = _run_main_until_rollout(tmp_path, monkeypatch, argv)
    assert seen["check_angle"] == angle
    assert seen["env_config"].env.corner_mode == (90 if "--corner-mode" in argv else 45)


def test_explicit_check_angle_differing_from_the_env_corner_mode_raises(tmp_path, monkeypatch):
    """Scoring a 90-degree-routed board by the 45-degree rule only warned before."""
    with pytest.raises(ValueError, match="--check-angle 45 differs from the env corner_mode 90"):
        _run_main_until_rollout(tmp_path, monkeypatch,
                                ["--corner-mode", "90", "--check-angle", "45"])


@pytest.mark.parametrize("argv,explicit", [([], False), (["--override-n-max-slots", "1280"], True)])
def test_slot_cap_override_is_explicit_only_when_the_command_line_gives_it(
        tmp_path, monkeypatch, argv, explicit):
    """The default override is skipped for a checkpoint that cannot take it; a given one raises
    (``apply_n_max_slots_override``) — so the pipeline tells the two apart."""
    assert _run_main_until_rollout(tmp_path, monkeypatch, argv)["override_explicit"] is explicit


def test_default_slot_cap_skipped_for_a_slot_embedding_ckpt_under_a_recipe(tmp_path, monkeypatch):
    """A recipe setting an env key does not make the default override explicit: on a
    checkpoint with the slot embedding it is skipped, and the run record shows the cap in
    effect (None = the checkpoint's), not the default that was not applied."""
    from types import SimpleNamespace

    import yaml

    vocab = SimpleNamespace(n_max_slots=512, disable_slot_emb=False, slot_perm=False)
    policy = SimpleNamespace(ckpt_sources={}, tokenizer=SimpleNamespace(vocab=vocab))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("max_steps: 64\n")
    seen = _run_main_until_rollout(tmp_path, monkeypatch, ["--recipe", str(recipe)], policy=policy)
    assert vocab.n_max_slots == 512 and seen["max_steps"] == 64
    record = yaml.safe_load((tmp_path / "cell" / "run_record.yaml").read_text())
    assert record["extra"]["invocation"]["override_n_max_slots"] is None


def _run_main_until_rollout(tmp_path, monkeypatch, argv, policy=None):
    """``core.main`` on a corner_mode-45 stub checkpoint, stopped at Stage 1's
    ``eval_transformer`` call; returns that call's kwargs (+ ``override_explicit``: the
    ``explicit`` the slot-cap override got). With a ``policy`` (one carrying a tokenizer
    vocab) the real slot-cap override runs on it."""
    class _Stop(Exception):
        pass

    seen = {}

    def fake_eval_transformer(policy, device, boards, **kwargs):
        seen.update(kwargs)
        raise _Stop

    ckpt = tmp_path / "policy.pt"
    ckpt.write_bytes(b"unused")
    monkeypatch.setattr(core, "_resolve_device", lambda spec: "cpu")
    monkeypatch.setattr(core, "load_boards_from_dir_or_list",
                        lambda **_kw: [BoardSpec(0, "board_a", "a.kicad_pcb")])
    monkeypatch.setattr(core, "load_policy_from_ckpt",
                        lambda ckpt, device, **_kw: (policy or _policy(), ckpt_config(corner_mode=45), 0))
    if policy is None:
        monkeypatch.setattr(core, "apply_n_max_slots_override",
                            lambda policy, slots, explicit: seen.update(override_explicit=explicit))
    monkeypatch.setattr(core, "eval_transformer", fake_eval_transformer)
    monkeypatch.setattr(sys, "argv", ["eval", "--ckpt", str(ckpt), "--boards-dir", str(tmp_path),
                                      "--seed", "1", "--n-rollouts", "1",
                                      "--output-dir", str(tmp_path / "cell"), *argv])
    with pytest.raises(_Stop):
        core.main()
    return seen


def test_checkpoint_without_a_seed_fails_before_the_rollout(tmp_path, monkeypatch):
    """The training seed stamps the rollout artifacts: a checkpoint without one is
    refused before Stage 1 runs, not after it."""
    ran = []
    ckpt = tmp_path / "policy.pt"
    ckpt.write_bytes(b"unused")
    monkeypatch.setattr(core, "_resolve_device", lambda spec: "cpu")
    monkeypatch.setattr(core, "load_boards_from_dir_or_list",
                        lambda **_kw: [BoardSpec(0, "board_a", "a.kicad_pcb")])
    monkeypatch.setattr(core, "load_policy_from_ckpt",
                        lambda ckpt, device, **_kw: (object(), ckpt_config(drop=("seed",)), 0))
    monkeypatch.setattr(core, "apply_n_max_slots_override", lambda policy, slots, explicit: None)
    monkeypatch.setattr(core, "eval_transformer", lambda *a, **kw: ran.append(1))
    monkeypatch.setattr(sys, "argv", ["eval", "--ckpt", str(ckpt), "--boards-dir", str(tmp_path),
                                      "--seed", "1", "--n-rollouts", "1",
                                      "--output-dir", str(tmp_path / "cell")])
    with pytest.raises(SystemExit, match="stores no training seed"):
        core.main()
    assert ran == []


def test_a_stochastic_rollout_is_reproduced_by_its_seed(monkeypatch, tmp_path):
    """Stage 1 seeds torch's global RNG (policy sampling, in this process) from --seed: two runs
    of one command give the same rows, whatever the process drew before (it never seeded it)."""
    import torch

    drawn = []

    def sampling_eval_transformer(policy, device, boards, **kwargs):
        drawn.append(torch.multinomial(torch.ones(1000), 8, replacement=True).tolist())
        rows = [_cell_row("cell", "board_a", 0, 0)]
        u.write_csv(kwargs["per_rollout_flush_path"], rows, u.PER_ROLLOUT_FIELDS)
        return EvalResult(per_rollout=rows)

    for run in range(2):
        torch.rand(run + 1)   # the process RNG state differs between the two runs
        _rollout_cell(monkeypatch, tmp_path / str(run), "--stages", "rollout",
                      eval_transformer=sampling_eval_transformer)
    assert drawn[0] == drawn[1]


# ---------------------------------------------------------------------------
# DRC scoring rule vs the ckpt's training reward_rule
# ---------------------------------------------------------------------------


#: The KDD 2L training reward (tests/fixtures/ckpt_real/kdd_2l_default_s42.json).
KDD_2L_REWARD = {"reward_rule": "drc_dense_promoted", "via_penalty": 0.1, "wirelength_penalty": 0.002,
                 "wire_via_emission": "per_step"}


def _phi_warning(out):
    """The differing-argument list of the scoring-vs-training warning in ``out`` (None: no warning)."""
    lines = [ln for ln in out.splitlines() if "defines another Φ than the ckpt's training reward" in ln]
    return lines[0].split("): ", 1)[1].split(" — ")[0] if lines else None


def test_scoring_a_ckpt_with_another_phi_than_its_training_reward_warns_and_is_recorded(
        monkeypatch, tmp_path, capsys):
    # A ckpt trained with drc_dense_promoted + via/wirelength overrides, scored with the
    # errors-only rule that bakes those overrides in — only the severity gate differs.
    # Warned at the rollout and at a re-scoring (from the run record's ckpt layer), recorded by both.
    import yaml

    cell = _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout",
                         "--reward-config", "drc_dense_errors_only_eval", ckpt_args=KDD_2L_REWARD)
    assert _phi_warning(capsys.readouterr().out) == "drc_severity_mode: 'errors_and_promoted' -> 'errors_only'"
    expected = {"training_reward_rule": "drc_dense_promoted", "matches_training_rule": False,
                "training_reward_overrides": {"via_penalty": 0.1, "wirelength_penalty": 0.002,
                                              "wire_via_emission": "per_step"}}
    scoring = yaml.safe_load((cell / core.RUN_RECORD).read_text())["extra"]["scoring"]
    assert {k: scoring[k] for k in expected} == expected

    assert _rescore(monkeypatch, cell)["reward_config"] == "drc_dense_errors_only_eval"
    assert _phi_warning(capsys.readouterr().out) == "drc_severity_mode: 'errors_and_promoted' -> 'errors_only'"
    rescored = yaml.safe_load((cell / core.RESCORE_RUN_RECORD).read_text())["extra"]["scoring"]
    assert {k: rescored[k] for k in expected} == expected


def test_a_pooled_kdd_cell_keeps_its_training_reward_for_the_phi_check(monkeypatch, tmp_path, capsys):
    # The KDD recipes roll out each seed's ckpt alone and pool the parts into one cell
    # (experiments/_lib/pool_seed_cells.py); the cell's run record is a part's, so scoring the
    # pooled cell with the plain drc_dense_promoted (the paper cells' rule) still names what
    # differs from the ckpts' training reward (its wirelength/via overrides).
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "_lib"))
    from pool_seed_cells import pool

    parts = []
    for seed in (42, 43):
        def rollout(policy, device, boards, *, _seed=seed, **kwargs):
            rows = [_cell_row("cell", "board_a", 0, 0, ckpt=_seed)]
            u.write_csv(kwargs["per_rollout_flush_path"], rows, u.PER_ROLLOUT_FIELDS)
            return EvalResult(per_rollout=rows)
        part = _rollout_cell(monkeypatch, tmp_path / str(seed), "--stages", "rollout",
                             eval_transformer=rollout, ckpt_args=KDD_2L_REWARD)
        (part / f"board_a_cell_s{seed}_r00.kicad_pcb").write_text("routed")
        parts.append(part)
    cell = tmp_path / "results" / "cell"
    assert pool(cell, parts) == 2
    assert sorted(p.name for p in (cell / "boards").iterdir()) == ["board_a_cell_s42_r00.kicad_pcb",
                                                                   "board_a_cell_s43_r00.kicad_pcb"]
    capsys.readouterr()
    assert _rescore(monkeypatch, cell, "--reward-config", "drc_dense_promoted")["reward_config"] == "drc_dense_promoted"
    warning = _phi_warning(capsys.readouterr().out)
    assert warning is not None and "via_penalty" in warning and "wirelength_penalty" in warning


def test_a_training_rule_differing_only_outside_phi_does_not_warn(monkeypatch, tmp_path, capsys):
    # d2b's drc_dense_promoted_invpen01 = drc_dense_promoted with another invalid_action_penalty
    # (an env penalty, not Φ): scoring it with drc_dense_promoted measures the trained Φ.
    import yaml

    src = yaml.safe_load(Path("experiments/kdd/configs/reward/drc_dense_promoted.yaml").read_text())
    src["name"] = "drc_dense_promoted_invpen01"
    src["potential"]["invalid_action_penalty"] = 0.1
    rule = tmp_path / "drc_dense_promoted_invpen01.yaml"
    rule.write_text(yaml.safe_dump(src))

    cell = _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout", "--reward-config", "drc_dense_promoted",
                         ckpt_args={"reward_rule": str(rule)})
    assert _phi_warning(capsys.readouterr().out) is None
    assert yaml.safe_load((cell / core.RUN_RECORD).read_text())["extra"]["scoring"]["matches_training_rule"] is True


def test_a_training_rule_that_no_longer_resolves_names_the_comparison(monkeypatch, tmp_path):
    from configs.loader.resolve import ConfigError

    with pytest.raises(ConfigError, match=r"comparing the DRC scoring rule 'pcbworld_reward' with the ckpt's "
                                          r"training reward_rule 'gone_rule' failed.*Restore its YAML"):
        _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout", "--reward-config", "pcbworld_reward",
                      ckpt_args={"reward_rule": "gone_rule"})


@pytest.mark.parametrize(("training", "scoring"), [
    ("pcbworld_reward", "pcbworld_reward"),
    ("drc_log_ladder_linear_w7a", "pcbworld_reward"),                  # an alias of the rule
    ("pcbworld_reward", str(Path("configs/reward/pcbworld_reward.yaml").resolve())),   # a path to it
])
def test_scoring_with_the_training_rule_does_not_warn(monkeypatch, tmp_path, capsys, training, scoring):
    # CADET: trained and scored with pcbworld_reward — the same rule by content, however named.
    import yaml

    cell = _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout", "--reward-config", scoring,
                         ckpt_args={"reward_rule": training})
    assert _phi_warning(capsys.readouterr().out) is None
    recorded = yaml.safe_load((cell / core.RUN_RECORD).read_text())["extra"]["scoring"]
    assert recorded["training_reward_rule"] == training
    assert recorded["matches_training_rule"] is True


def test_rescoring_under_another_rule_checks_the_new_rule_against_training(monkeypatch, tmp_path, capsys):
    # A cell scored with its training rule, re-scored under another rule by flag: the Φ check
    # runs on the new rule and the re-scoring record says it no longer matches training.
    import yaml

    cell = _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout", "--reward-config", "drc_dense_promoted",
                         ckpt_args={"reward_rule": "drc_dense_promoted"})
    assert _phi_warning(capsys.readouterr().out) is None

    assert _rescore(monkeypatch, cell, "--reward-config", "drc_dense_errors_only_eval")["reward_config"] == \
        "drc_dense_errors_only_eval"
    out = capsys.readouterr().out
    assert "--reward-config 'drc_dense_errors_only_eval' differs from the reward_config the cell records" in out
    assert _phi_warning(out) is not None
    rescored = yaml.safe_load((cell / core.RESCORE_RUN_RECORD).read_text())["extra"]["scoring"]
    assert (rescored["reward_config"], rescored["matches_training_rule"]) == ("drc_dense_errors_only_eval", False)


def test_an_aggregate_only_rescoring_records_the_cells_rule_against_training(monkeypatch, tmp_path, capsys):
    # Rewriting the re-scoring record without Stage 2 keeps the training comparison of the
    # cell's rule (recorded, not warned again).
    import yaml

    cell = _rollout_cell(monkeypatch, tmp_path, "--stages", "rollout",
                         "--reward-config", "drc_dense_errors_only_eval", ckpt_args=KDD_2L_REWARD)
    capsys.readouterr()
    _reaggregate(monkeypatch, cell)
    assert _phi_warning(capsys.readouterr().out) is None
    scoring = yaml.safe_load((cell / core.RESCORE_RUN_RECORD).read_text())["extra"]["scoring"]
    assert (scoring["training_reward_rule"], scoring["matches_training_rule"]) == ("drc_dense_promoted", False)
