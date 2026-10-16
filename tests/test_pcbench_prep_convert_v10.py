"""Per-board pcbnew crashes retain an actionable conversion outcome."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import convert_v10  # noqa: E402


def test_silent_nonzero_child_exit_records_returncode_and_missing_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "PCBs" / "source-board"
    source.mkdir(parents=True)
    (source / "processed.kicad_pcb").write_text("(kicad_pcb)\n")
    output = tmp_path / "v10"
    helper = tmp_path / "convert_helper.py"
    helper.write_text("# test child helper\n")
    monkeypatch.setattr(convert_v10.kicad_tools, "pcbnew_python", lambda: "engine-python")
    monkeypatch.setattr(convert_v10.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=-11, stdout="", stderr=""))

    result = convert_v10.convert_sample("source-board", source.parent, output, helper)

    assert result["status"] == "failed"
    assert result["returncode"] == -11
    assert "returncode -11 without diagnostics" in result["error"]
    assert "processed_v10.kicad_pcb" in result["error"]
    assert "processed_v10.kicad_pro" in result["error"]
    assert not (output / "source-board").exists()
