"""sort_prefix.py v10 ordering, source identity, report accounting, and ownership."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import sort_prefix  # noqa: E402


def _board(root: Path, name: str, pins: int, nets: int, footprints: int) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "final.json").write_text(json.dumps(
        {"nets": {f"n{i}": [f"p{j}" for j in range(pins // nets)] for i in range(nets)},
         "layers": ["F", "B"]}))
    (d / "processed_v10_guide_v3.kicad_pcb").write_text(
        "(kicad_pcb\n" + "\t(footprint x)\n" * footprints + ")\n")


def _run(monkeypatch, src: Path, out: Path, guide_status: dict[str, str] | None = None) -> list[str]:
    guide_status = guide_status or {}
    (src / "_guide_results.json").write_text(json.dumps([
        {"folder": p.name, "status": guide_status.get(p.name, "pass")}
        for p in sorted(src.iterdir()) if p.is_dir()
    ]))
    monkeypatch.setenv("PCBENCH_NEWDRC_OUT", str(src))
    monkeypatch.setenv("PCBENCH_SORTED_OUT", str(out))
    monkeypatch.setattr(sys, "argv", ["sort_prefix.py"])
    sort_prefix.main()
    return sorted(p.name for p in out.iterdir() if p.is_dir())


def test_order_layout_source_id_and_rerun_replaces_stale_entries(tmp_path, monkeypatch):
    src, out = tmp_path / "newdrc_v10", tmp_path / "sorted_v10"
    _board(src, "source-big", pins=40, nets=4, footprints=6)
    _board(src, "source-small", pins=8, nets=2, footprints=2)
    assert _run(monkeypatch, src, out) == ["0001_source-small", "0002_source-big"]
    rows = list(csv.DictReader(open(out / sort_prefix.CSV_NAME)))
    assert [(r["sample"], r["sorted_id"], r["pins"], r["nets"], r["components"], r["layers"])
            for r in rows] == [
        ("source-small", "0001_source-small", "8", "2", "2", "2"),
        ("source-big", "0002_source-big", "40", "4", "6", "2"),
    ]

    _board(src, "source-mid", pins=20, nets=2, footprints=3)
    (out / "notes").mkdir()
    assert _run(monkeypatch, src, out) == [
        "0001_source-small", "0002_source-mid", "0003_source-big", "notes"]
    assert (out / "0003_source-big" / "processed_v10_guide_v3.kicad_pcb").exists()
    rows = {r["sample"]: r["sorted_id"] for r in csv.DictReader(open(out / sort_prefix.CSV_NAME))}
    assert rows["source-big"] == "0003_source-big"
    assert rows["source-small"] == "0001_source-small"


def test_guide_rejections_are_reported_but_not_sorted(tmp_path, monkeypatch):
    src, out = tmp_path / "newdrc_v10", tmp_path / "sorted_v10"
    _board(src, "good", pins=4, nets=1, footprints=1)
    _board(src, "rejected", pins=8, nets=2, footprints=2)
    assert _run(monkeypatch, src, out, {"rejected": "fail"}) == ["0001_good"]
    results = json.loads((src / "_sort_results.json").read_text())
    assert {r["sample"]: r["status"] for r in results} == {
        "good": "ok", "rejected": "guide_rejected"}


def test_malformed_nets_are_reported_and_excluded(tmp_path, monkeypatch):
    src, out = tmp_path / "newdrc_v10", tmp_path / "sorted_v10"
    _board(src, "good", pins=4, nets=1, footprints=1)
    _board(src, "malformed", pins=4, nets=1, footprints=1)
    (src / "malformed" / "final.json").write_text(json.dumps({
        "nets": [], "layers": ["F", "B"]
    }))
    with pytest.raises(SystemExit) as raised:
        _run(monkeypatch, src, out)
    assert raised.value.code == 1
    report = json.loads((src / "_sort_results.json").read_text())
    by_sample = {row["sample"]: row for row in report}
    assert {name: row["status"] for name, row in by_sample.items()} == {
        "good": "ok", "malformed": "stage_error"}
    assert by_sample["malformed"]["error"] == "missing or malformed final.json/guide board"
    assert sorted(path.name for path in out.iterdir() if path.is_dir()) == ["0001_good"]


def test_zero_usable_boards_refuses_and_leaves_output(tmp_path, monkeypatch):
    src, out = tmp_path / "newdrc_v10", tmp_path / "sorted_v10"
    (src / "wrong_shape").mkdir(parents=True)
    (src / "_guide_results.json").write_text(json.dumps([
        {"folder": "wrong_shape", "status": "pass"}]))
    (out / "0001_keep").mkdir(parents=True)
    (out / "0001_keep" / "processed_v10_guide_v3.kicad_pcb").write_text("(kicad_pcb)")
    (out / sort_prefix.CSV_NAME).write_text(
        "sample,sorted_id,nets,components,pins,layers\nkeep,0001_keep,1,1,4,2\n")
    monkeypatch.setenv("PCBENCH_NEWDRC_OUT", str(src))
    monkeypatch.setenv("PCBENCH_SORTED_OUT", str(out))
    monkeypatch.setattr(sys, "argv", ["sort_prefix.py"])
    with pytest.raises(SystemExit, match="no successful guide boards"):
        sort_prefix.main()
    assert (out / "0001_keep" / "processed_v10_guide_v3.kicad_pcb").exists()
    assert "keep,0001_keep,1,1,4,2" in (out / sort_prefix.CSV_NAME).read_text()
