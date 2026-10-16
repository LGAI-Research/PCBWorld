#!/usr/bin/env python3
"""Build the v10 D3 (real-board) split JSON used by the LLM quickstart wrappers.

Source data
-----------
The PCB-bench ``exacad_sorted_v10`` distribution: board folders named
``NNNN_<source-id>`` and a companion CSV ``pcb_characteristics_exacad_sorted.csv``
with one row per board carrying ``sample`` (the raw source folder ID),
``sorted_id`` (the v10 output folder), and ``nets,components,pins,layers``. The
CSV is monotone non-decreasing in ``pins``, so its row order is the difficulty
order.

Difficulty classification
-------------------------
* rows ``[0, --easy-rows)``                       → **easy**
* rows ``[--easy-rows, ...)`` with pins ≤ N       → **medium**
* rows ``[--easy-rows, ...)`` with pins > N       → **hard**
  where N = --medium-pin-threshold (default 100).

Train sets
----------
Every board in each difficulty enters ``<diff>.train`` (this generator never
down-samples train).

Test sets (held-out for LLM eval)
---------------------------------
* **easy.test** — all easy boards except ``karabas-nano_wifi_revA``, which stays
  train-only so a single-board sanity command can target a known sample.
* **medium.test / hard.test** — 10 boards each, picked by:
    1. restrict to ``layer == 2``,
    2. (hard only) drop the top ``--hard-trim-frac`` of boards by ``pins``
       as outliers,
    3. sort by ``(pins, nets, sample)`` for deterministic ordering,
    4. split into 10 equal-sized quantile bins,
    5. take the lower-median board of each bin.

Output
------
The generated ``configs/datasets/d3_v2.json`` uses
``processed_v10_guide_v3.kicad_pcb`` and records the medium-test source-ID swap
in the top-level ``_test_overrides`` block. ``configs/datasets/d3.json`` is the
frozen historical v1 manifest and is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_PREFIX_RE = re.compile(r"^\d{4,5}_(.+)$")
_BOARD_FILENAME = "processed_v10_guide_v3.kicad_pcb"
_V1_MANIFEST = Path(__file__).resolve().parents[3] / "configs/datasets/d3.json"
_EASY_TRAIN_ONLY_SOURCE = "karabas-nano_wifi_revA"
_MEDIUM_TEST_SWAP = ("badge2016_Badge_init", "cat-trainer_teensy_base_pcb")


def _source_id_from_folder(name: str) -> str | None:
    match = _PREFIX_RE.match(name)
    return match.group(1) if match else None


def _index_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        sample = row.get("sample", "")
        if not sample:
            raise SystemExit("[error] CSV row is missing its source-ID 'sample'.")
        by_source.setdefault(sample, []).append(row)
    return by_source


def _require_one(
    rows_by_source: dict[str, list[dict[str, str]]], source_id: str, context: str
) -> dict[str, str]:
    matches = rows_by_source.get(source_id, [])
    if len(matches) != 1:
        raise SystemExit(
            f"[error] {context} source ID {source_id!r} matched {len(matches)} rows "
            "(need exactly 1)."
        )
    return matches[0]


def _load_sorted_map(
    sorted_dir: Path, rows_by_source: dict[str, list[dict[str, str]]]
) -> dict[str, str]:
    folders_by_source: dict[str, list[str]] = {}
    for entry in sorted_dir.iterdir():
        if not entry.is_dir():
            continue
        source_id = _source_id_from_folder(entry.name)
        if source_id is not None:
            folders_by_source.setdefault(source_id, []).append(entry.name)

    sorted_by_source: dict[str, str] = {}
    for source_id, rows in rows_by_source.items():
        if len(rows) != 1:
            raise SystemExit(
                f"[error] CSV source ID {source_id!r} matched {len(rows)} rows "
                "(need exactly 1)."
            )
        sorted_id = rows[0].get("sorted_id", "")
        if _source_id_from_folder(sorted_id) != source_id:
            raise SystemExit(
                f"[error] CSV sorted_id {sorted_id!r} does not identify source "
                f"folder {source_id!r}."
            )
        matches = folders_by_source.get(source_id, [])
        if len(matches) != 1:
            raise SystemExit(
                f"[error] source ID {source_id!r} matched {len(matches)} folders "
                f"under {sorted_dir} (need exactly 1)."
            )
        if matches[0] != sorted_id:
            raise SystemExit(
                f"[error] CSV sorted_id {sorted_id!r} does not match on-disk folder "
                f"{matches[0]!r} for source ID {source_id!r}."
            )
        sorted_by_source[source_id] = sorted_id
    return sorted_by_source


def _portable_dataset_dir(sorted_dir: Path) -> str:
    raw_data_root = os.environ.get("PCBWORLD_DATA_ROOT")
    if not raw_data_root:
        raise SystemExit(
            "[error] PCBWORLD_DATA_ROOT must be set to serialize a portable dataset path."
        )
    data_root = Path(raw_data_root).expanduser().resolve()
    resolved_sorted_dir = sorted_dir.resolve()
    try:
        relative_dir = resolved_sorted_dir.relative_to(data_root)
    except ValueError as exc:
        raise SystemExit(
            f"[error] sorted directory {resolved_sorted_dir} is not beneath "
            f"PCBWORLD_DATA_ROOT {data_root}; cannot serialize a portable dataset path."
        ) from exc
    suffix = relative_dir.as_posix()
    return "${PCBWORLD_DATA_ROOT}" + (f"/{suffix}" if suffix != "." else "")


def _classify(
    rows: list[dict[str, str]],
    *,
    easy_rows: int,
    medium_pin_threshold: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    easy = rows[:easy_rows]
    rest = rows[easy_rows:]
    medium = [r for r in rest if int(r["pins"]) <= medium_pin_threshold]
    hard = [r for r in rest if int(r["pins"]) > medium_pin_threshold]
    return easy, medium, hard


def _decile_median_pick(
    items: list[dict[str, str]], n: int
) -> list[dict[str, str]]:
    """n equal-size quantile bins by pins; pick lower-median of each bin."""
    ordered = sorted(items, key=lambda r: (int(r["pins"]), int(r["nets"]), r["sample"]))
    picks: list[dict[str, str]] = []
    for chunk in np.array_split(np.arange(len(ordered)), n):
        bucket = [ordered[i] for i in chunk]
        mid = (len(bucket) - 1) // 2
        picks.append(bucket[mid])
    return picks


def build(args: argparse.Namespace) -> dict:
    with args.csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise SystemExit(f"[error] empty CSV: {args.csv}")

    rows_by_source = _index_rows(rows)
    sorted_by_source = _load_sorted_map(args.sorted_dir, rows_by_source)
    easy_rows, medium_rows, hard_rows = _classify(
        rows,
        easy_rows=args.easy_rows,
        medium_pin_threshold=args.medium_pin_threshold,
    )

    easy_train_only = _require_one(
        _index_rows(easy_rows), args.easy_train_only_source, "easy train-only"
    )
    easy_train = [sorted_by_source[r["sample"]] for r in easy_rows]
    easy_test = [
        sorted_by_source[r["sample"]]
        for r in easy_rows
        if r["sample"] != easy_train_only["sample"]
    ]
    medium_train = [sorted_by_source[r["sample"]] for r in medium_rows]
    hard_train = [sorted_by_source[r["sample"]] for r in hard_rows]

    # Layer-2 quantile-median pick for medium / hard.
    medium_l2 = [r for r in medium_rows if int(r["layers"]) == 2]
    hard_l2 = [r for r in hard_rows if int(r["layers"]) == 2]

    # Hard: drop the top trim_frac by pins before picking.
    hard_l2_sorted = sorted(hard_l2, key=lambda r: int(r["pins"]))
    n_drop = int(round(len(hard_l2_sorted) * args.hard_trim_frac))
    hard_l2_kept = hard_l2_sorted[:-n_drop] if n_drop else hard_l2_sorted

    medium_source, medium_replacement = _MEDIUM_TEST_SWAP
    medium_l2_by_source = _index_rows(medium_l2)
    _require_one(medium_l2_by_source, medium_source, "medium-test override source")
    replacement_row = _require_one(
        medium_l2_by_source, medium_replacement, "medium-test override replacement"
    )
    medium_test_rows = _decile_median_pick(medium_l2, args.medium_n)
    source_picks = [r for r in medium_test_rows if r["sample"] == medium_source]
    if len(source_picks) != 1:
        raise SystemExit(
            f"[error] medium-test override source ID {medium_source!r} matched "
            f"{len(source_picks)} selected boards (need exactly 1)."
        )
    if any(r["sample"] == medium_replacement for r in medium_test_rows):
        raise SystemExit(
            f"[error] medium-test override replacement ID {medium_replacement!r} "
            "is already selected."
        )
    medium_test_rows = [
        r for r in medium_test_rows if r["sample"] != medium_source
    ] + [replacement_row]
    medium_test = sorted(sorted_by_source[r["sample"]] for r in medium_test_rows)

    hard_test = sorted(
        sorted_by_source[r["sample"]]
        for r in _decile_median_pick(hard_l2_kept, args.hard_n)
    )

    dataset_dir = _portable_dataset_dir(args.sorted_dir)
    return {
        "board_filename": _BOARD_FILENAME,
        "_test_overrides": {"medium": {medium_source: medium_replacement}},
        "easy": {"train": easy_train, "test": easy_test},
        "medium": {"train": medium_train, "test": medium_test},
        "hard": {"train": hard_train, "test": hard_test},
        "dataset_dirs": {"train": dataset_dir, "test": dataset_dir},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True,
                   help="pcb_characteristics_exacad_sorted.csv")
    p.add_argument("--sorted-dir", type=Path, required=True,
                   help="Directory containing NNNN_<source-id>/ board folders.")
    p.add_argument("--out", type=Path, default=Path("configs/datasets/d3_v2.json"),
                   help="Output v10 boards-json (default: configs/datasets/d3_v2.json).")
    p.add_argument("--easy-rows", type=int, default=100,
                   help="First N CSV rows = easy difficulty (default 100).")
    p.add_argument("--medium-pin-threshold", type=int, default=100,
                   help="Boards beyond --easy-rows with pins ≤ T are medium; "
                        "the rest are hard (default 100).")
    p.add_argument("--medium-n", type=int, default=10,
                   help="Number of medium-test boards to pick (default 10).")
    p.add_argument("--hard-n", type=int, default=10,
                   help="Number of hard-test boards to pick (default 10).")
    p.add_argument("--hard-trim-frac", type=float, default=0.10,
                   help="Drop the top fraction of hard L2 boards by pins before "
                        "the decile pick (default 0.10).")
    p.add_argument("--easy-train-only-source", default=_EASY_TRAIN_ONLY_SOURCE,
                   help="Source folder ID kept out of easy.test "
                        f"(default: {_EASY_TRAIN_ONLY_SOURCE}).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.resolve() == _V1_MANIFEST.resolve():
        print(
            "[error] refusing to overwrite frozen historical v1 manifest "
            f"{_V1_MANIFEST}; use configs/datasets/d3_v2.json.",
            file=sys.stderr,
        )
        return 2
    if not args.csv.is_file():
        print(f"[error] --csv not found: {args.csv}", file=sys.stderr)
        return 2
    if not args.sorted_dir.is_dir():
        print(f"[error] --sorted-dir not a directory: {args.sorted_dir}", file=sys.stderr)
        return 2

    payload = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    n_easy_tr, n_easy_te = len(payload["easy"]["train"]), len(payload["easy"]["test"])
    n_med_tr, n_med_te = len(payload["medium"]["train"]), len(payload["medium"]["test"])
    n_hard_tr, n_hard_te = len(payload["hard"]["train"]), len(payload["hard"]["test"])
    print(f"[d3-build] wrote {args.out}")
    print(f"  easy   train={n_easy_tr:3d}  test={n_easy_te:3d}")
    print(f"  medium train={n_med_tr:3d}  test={n_med_te:3d}")
    print(f"  hard   train={n_hard_tr:3d}  test={n_hard_te:3d}")
    print(f"  dataset_dirs.train = {payload['dataset_dirs']['train']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
