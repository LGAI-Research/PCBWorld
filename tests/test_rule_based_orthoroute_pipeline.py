"""OrthoRoute Stage 1 must validate and account for every ORP."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.baselines.rule_based import run_orthoroute_pipeline as pipeline  # noqa: E402


def _board(root: Path, board_id: str = "board-a"):
    folder = root / board_id
    folder.mkdir(parents=True)
    pcb = folder / f"{pipeline.DEFAULT_STEM}_unrouted.kicad_pcb"
    pcb.touch()
    return board_id, pcb, None


def test_stage1_preflights_and_reports_reused_orp(tmp_path, monkeypatch):
    board = _board(tmp_path / "input")
    board_id = board[0]
    out_root = tmp_path / "out"
    orp_root = tmp_path / "prepared"
    reused = orp_root / board_id / f"{pipeline.DEFAULT_STEM}.orp"
    reused.parent.mkdir(parents=True)
    reused.write_bytes(b"valid ORP")
    calls = []
    monkeypatch.setattr(pipeline.kicad_tools, "assert_all", lambda: calls.append(True))
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: pytest.fail("converter should not run"))

    paths = pipeline.stage1_orp([board], out_root, orp_root, pipeline.DEFAULT_STEM)

    assert calls == [True]
    assert paths == {board_id: reused}
    report = json.loads((out_root / "_orp_stage1_report.json").read_text())
    assert report["converter"]["status"] == "not_run"
    assert report["boards"] == [{
        "board_id": board_id,
        "input_pcb": str(board[1]),
        "orp_path": str(reused),
        "status": "reused",
        "source": "orp_root",
        "converter_status": "not_run",
        "converter_msg": None,
        "error": None,
    }]


def test_stage1_accepts_reported_nonempty_converter_output(tmp_path, monkeypatch):
    board = _board(tmp_path / "input")
    board_id = board[0]
    out_root = tmp_path / "out"
    monkeypatch.setattr(pipeline.kicad_tools, "assert_all", lambda: None)
    monkeypatch.setattr(pipeline.kicad_tools, "pcbnew_python", lambda: "engine-python")

    def run(cmd, env):
        report_path = Path(cmd[cmd.index("--report") + 1])
        report_path.write_text(json.dumps([{"folder": board_id, "status": "ok", "msg": "done"}]))
        (out_root / board_id / f"{pipeline.DEFAULT_STEM}.orp").write_bytes(b"valid ORP")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    paths = pipeline.stage1_orp([board], out_root, None, pipeline.DEFAULT_STEM)

    assert paths[board_id] == out_root / board_id / f"{pipeline.DEFAULT_STEM}.orp"
    report = json.loads((out_root / "_orp_stage1_report.json").read_text())
    assert report["converter"]["status"] == "ok"
    assert report["converter"]["returncode"] == 0
    assert report["boards"][0]["status"] == "generated"
    assert report["boards"][0]["converter_status"] == "ok"


@pytest.mark.parametrize(
    "returncode, report_status, write_orp, report_folder, expected_error",
    [
        (0, "fail", True, "board-a", "converter status"),
        (1, "ok", True, "board-a", "converter exited with rc=1"),
        (0, "ok", False, "board-a", "expected ORP is missing or empty"),
        (0, "ok", True, "other-board", "converter report board mismatch"),
    ],
)
def test_stage1_fails_and_reports_invalid_converter_outcomes(
    tmp_path, monkeypatch, returncode, report_status, write_orp, report_folder, expected_error,
):
    board = _board(tmp_path / "input")
    board_id = board[0]
    out_root = tmp_path / "out"
    monkeypatch.setattr(pipeline.kicad_tools, "assert_all", lambda: None)
    monkeypatch.setattr(pipeline.kicad_tools, "pcbnew_python", lambda: "engine-python")

    def run(cmd, env):
        report_path = Path(cmd[cmd.index("--report") + 1])
        report_path.write_text(json.dumps([{
            "folder": report_folder, "status": report_status, "msg": "converter detail",
        }]))
        if write_orp:
            (out_root / board_id / f"{pipeline.DEFAULT_STEM}.orp").write_bytes(b"valid ORP")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises(SystemExit, match="Stage 1 failed to produce valid ORPs"):
        pipeline.stage1_orp([board], out_root, None, pipeline.DEFAULT_STEM)

    report = json.loads((out_root / "_orp_stage1_report.json").read_text())
    assert report["converter"]["status"] == "failed"
    assert report["boards"][0]["status"] == "failed"
    assert expected_error in report["boards"][0]["error"]


def test_stage1_reports_preflight_failure(tmp_path, monkeypatch):
    board = _board(tmp_path / "input")
    out_root = tmp_path / "out"

    def reject_preflight():
        raise SystemExit("stock KiCad is not pinned")

    monkeypatch.setattr(pipeline.kicad_tools, "assert_all", reject_preflight)
    with pytest.raises(SystemExit, match="stock KiCad is not pinned"):
        pipeline.stage1_orp([board], out_root, None, pipeline.DEFAULT_STEM)

    report = json.loads((out_root / "_orp_stage1_report.json").read_text())
    assert report["converter"]["status"] == "blocked_by_preflight"
    assert report["boards"][0]["status"] == "preflight_failed"
    assert report["boards"][0]["error"] == "stock KiCad is not pinned"
