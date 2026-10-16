"""Contract tests for source-ID-based v10 D3 manifest generation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments" / "kdd" / "d3_dataset"))
import build as d3_build  # noqa: E402


def _rows() -> list[dict[str, str]]:
    return [
        {"sample": "other-easy", "sorted_id": "0001_other-easy", "nets": "1", "components": "1", "pins": "1", "layers": "2"},
        {"sample": "karabas-nano_wifi_revA", "sorted_id": "0002_karabas-nano_wifi_revA", "nets": "1", "components": "1", "pins": "2", "layers": "2"},
        {"sample": "badge2016_Badge_init", "sorted_id": "0003_badge2016_Badge_init", "nets": "2", "components": "2", "pins": "10", "layers": "2"},
        {"sample": "cat-trainer_teensy_base_pcb", "sorted_id": "0004_cat-trainer_teensy_base_pcb", "nets": "2", "components": "2", "pins": "10", "layers": "2"},
        {"sample": "hard-board", "sorted_id": "0005_hard-board", "nets": "2", "components": "2", "pins": "101", "layers": "2"},
    ]


def _inputs(
    tmp_path: Path,
    rows: list[dict[str, str]],
    monkeypatch,
    relative_dir: Path = Path("pcbench/exacad_sorted_v10"),
) -> tuple[Path, Path]:
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    monkeypatch.setenv("PCBWORLD_DATA_ROOT", str(data_root))
    sorted_dir = data_root / relative_dir
    sorted_dir.mkdir(parents=True)
    for row in rows:
        (sorted_dir / row["sorted_id"]).mkdir()
    csv_path = sorted_dir / "pcb_characteristics_exacad_sorted.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, sorted_dir


def _args(csv_path: Path, sorted_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        csv=csv_path,
        sorted_dir=sorted_dir,
        out=Path("configs/datasets/d3_v2.json"),
        easy_rows=2,
        medium_pin_threshold=100,
        medium_n=1,
        hard_n=1,
        hard_trim_frac=0,
        easy_train_only_source="karabas-nano_wifi_revA",
    )


def test_source_ids_keep_identity_when_sorted_ranks_change(tmp_path, monkeypatch):
    csv_path, sorted_dir = _inputs(tmp_path, _rows(), monkeypatch)

    payload = d3_build.build(_args(csv_path, sorted_dir))

    assert payload["easy"]["train"] == ["0001_other-easy", "0002_karabas-nano_wifi_revA"]
    assert payload["easy"]["test"] == ["0001_other-easy"]
    assert payload["medium"]["test"] == ["0004_cat-trainer_teensy_base_pcb"]
    assert payload["_test_overrides"] == {
        "medium": {"badge2016_Badge_init": "cat-trainer_teensy_base_pcb"}
    }
    assert payload["board_filename"] == "processed_v10_guide_v3.kicad_pcb"
    portable_root = "${PCBWORLD_DATA_ROOT}/pcbench/exacad_sorted_v10"
    assert payload["dataset_dirs"] == {"train": portable_root, "test": portable_root}


@pytest.mark.parametrize(
    "source_id",
    ["karabas-nano_wifi_revA", "badge2016_Badge_init", "cat-trainer_teensy_base_pcb"],
)
def test_required_source_ids_fail_when_missing(tmp_path, monkeypatch, source_id):
    rows = [row for row in _rows() if row["sample"] != source_id]
    csv_path, sorted_dir = _inputs(tmp_path, rows, monkeypatch)

    with pytest.raises(SystemExit, match="matched 0"):
        d3_build.build(_args(csv_path, sorted_dir))


def test_source_id_requires_exactly_one_prefixed_folder(tmp_path, monkeypatch):
    rows = _rows()
    csv_path, sorted_dir = _inputs(tmp_path, rows, monkeypatch)
    (sorted_dir / "9999_karabas-nano_wifi_revA").mkdir()

    with pytest.raises(SystemExit, match="matched 2 folders"):
        d3_build.build(_args(csv_path, sorted_dir))


def test_custom_sorted_dir_serializes_relative_to_data_root(tmp_path, monkeypatch):
    csv_path, sorted_dir = _inputs(
        tmp_path, _rows(), monkeypatch, relative_dir=Path("trial copy/sorted boards")
    )

    payload = d3_build.build(_args(csv_path, sorted_dir))

    portable_root = "${PCBWORLD_DATA_ROOT}/trial copy/sorted boards"
    assert payload["dataset_dirs"] == {"train": portable_root, "test": portable_root}


def test_sorted_dir_outside_data_root_fails(tmp_path, monkeypatch):
    csv_path, sorted_dir = _inputs(tmp_path, _rows(), monkeypatch)
    monkeypatch.setenv("PCBWORLD_DATA_ROOT", str(tmp_path / "different-root"))

    with pytest.raises(SystemExit, match="not beneath PCBWORLD_DATA_ROOT"):
        d3_build.build(_args(csv_path, sorted_dir))


def test_cli_defaults_to_v10_manifest_and_protects_frozen_v1(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["build.py", "--csv", "missing.csv", "--sorted-dir", "missing-dir"],
    )
    args = d3_build.parse_args()
    assert args.out == Path("configs/datasets/d3_v2.json")
    assert args.easy_train_only_source == "karabas-nano_wifi_revA"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build.py",
            "--csv",
            "missing.csv",
            "--sorted-dir",
            "missing-dir",
            "--out",
            str(d3_build._V1_MANIFEST),
        ],
    )
    assert d3_build.main() == 2
    assert "refusing to overwrite frozen historical v1 manifest" in capsys.readouterr().err


def test_live_aliases_and_paths_point_to_v10():
    splits = json.loads((REPO / "experiments/kdd/configs/quickstart/splits.json").read_text())
    assert [splits[f"d3{suffix}"]["boards_json"] for suffix in "abc"] == [
        "configs/datasets/d3_v2.json"
    ] * 3

    paths = yaml.safe_load((REPO / "configs/paths.yaml").read_text())
    assert paths["datasets"]["pcbench_exacad"]["sub"] == "pcbench/exacad_sorted_v10"
    assert paths["datasets"]["pcbench_exacad_dsn"]["sub"] == "pcbench/exacad_sorted_dsn_v10"

    run_script = (REPO / "experiments/kdd/d3_dataset/run.sh").read_text()
    assert 'OUT="${OUT:-configs/datasets/d3_v2.json}"' in run_script
    assert "PCBENCH_SORTED_OUT" in run_script
