from pathlib import Path
import sys

import pytest

from eval import pipeline as core
from eval import eval_utils as u
from eval.aggregation import aggregate_boards, aggregate_inline
from methods._shared.board_loader import BoardSpec
from eval.metrics import EvalResult


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
    monkeypatch.setattr(core, "load_policy_from_ckpt", lambda ckpt, device: (object(), {}, 0))
    monkeypatch.setattr(core, "apply_n_max_slots_override", lambda policy, slots: None)
    monkeypatch.setattr(core, "env_kwargs_from_checkpoint", lambda args, max_steps: {})

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

    args = core._parse_args()

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

    args = core._parse_args()
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

    args = core._parse_args()

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
