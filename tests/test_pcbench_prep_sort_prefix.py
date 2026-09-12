"""sort_prefix.py (tools/datagen/pcbench_prep): difficulty order, <NNNN>_<name>/ layout, and
ownership of the output directory — a re-run replaces its entries and removes the ones an
earlier run left, so a full run after a `--limit` trial leaves exactly the full set."""
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
        {"nets": {f"n{i}": [f"p{j}" for j in range(pins // nets)] for i in range(nets)}, "layers": ["F", "B"]}))
    (d / "processed_v9.kicad_pcb").write_text("(kicad_pcb\n" + "\t(footprint x)\n" * footprints + ")\n")


def _run(monkeypatch, src: Path, out: Path) -> list[str]:
    monkeypatch.setenv("PCBENCH_NEWDRC_OUT", str(src))
    monkeypatch.setenv("PCBENCH_SORTED_OUT", str(out))
    monkeypatch.setattr(sys, "argv", ["sort_prefix.py"])
    sort_prefix.main()
    return sorted(p.name for p in out.iterdir() if p.is_dir())


def test_order_layout_and_rerun_replaces_stale_entries(tmp_path, monkeypatch):
    src, out = tmp_path / "newdrc", tmp_path / "sorted"
    _board(src, "big", pins=40, nets=4, footprints=6)
    _board(src, "small", pins=8, nets=2, footprints=2)
    assert _run(monkeypatch, src, out) == ["0001_small", "0002_big"]
    rows = list(csv.DictReader(open(out / sort_prefix.CSV_NAME)))
    assert [(r["sample"], r["pins"], r["nets"], r["components"], r["layers"]) for r in rows] == \
        [("small", "8", "2", "2", "2"), ("big", "40", "4", "6", "2")]

    # a fuller input re-sorts: the earlier 0002_big is stale (big is now 0003_) and goes away
    _board(src, "mid", pins=20, nets=2, footprints=3)
    (out / "notes").mkdir()                       # not a <NNNN>_ entry: left alone
    assert _run(monkeypatch, src, out) == ["0001_small", "0002_mid", "0003_big", "notes"]
    assert (out / "0003_big" / "processed_v9.kicad_pcb").exists()


def test_folders_without_inputs_are_skipped(tmp_path, monkeypatch):
    src, out = tmp_path / "newdrc", tmp_path / "sorted"
    _board(src, "ok", pins=4, nets=1, footprints=1)
    (src / "no_final").mkdir()
    assert _run(monkeypatch, src, out) == ["0001_ok"]


def test_zero_usable_boards_refuses_and_leaves_output(tmp_path, monkeypatch):
    # a misconfigured input (nothing usable) must not sweep the existing output set
    src, out = tmp_path / "newdrc", tmp_path / "sorted"
    (src / "wrong_shape").mkdir(parents=True)
    _board(out.parent / "seed", "keep", pins=4, nets=1, footprints=1)  # build a fake staged entry
    (out / "0001_keep").mkdir(parents=True)
    (out / "0001_keep" / "processed_v9.kicad_pcb").write_text("(kicad_pcb)")
    (out / sort_prefix.CSV_NAME).write_text("sample,nets,components,pins,layers\nkeep,1,1,4,2\n")
    monkeypatch.setenv("PCBENCH_NEWDRC_OUT", str(src))
    monkeypatch.setenv("PCBENCH_SORTED_OUT", str(out))
    monkeypatch.setattr(sys, "argv", ["sort_prefix.py"])
    with pytest.raises(SystemExit, match="no usable board folders"):
        sort_prefix.main()
    assert (out / "0001_keep" / "processed_v9.kicad_pcb").exists()
    assert "keep,1,1,4,2" in (out / sort_prefix.CSV_NAME).read_text()
