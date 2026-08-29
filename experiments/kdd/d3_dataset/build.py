#!/usr/bin/env python3
"""Build the D3 (real-board) split JSON used by the LLM quickstart wrappers.

Source data
-----------
The PCB-bench `exacad_sorted` distribution: a directory of board folders
prefixed with a 4-digit index (``0001_<name>``) and a companion CSV
``pcb_characteristics_exacad_sorted.csv`` with one row per board carrying
``sample,nets,components,pins,layers``. The CSV is monotone non-decreasing
in ``pins``, so its row order *is* the difficulty order.

Difficulty classification
-------------------------
* rows ``[0, --easy-rows)``                       → **easy**
* rows ``[--easy-rows, ...)`` with pins ≤ N       → **medium**
* rows ``[--easy-rows, ...)`` with pins > N       → **hard**
  where N = --medium-pin-threshold (default 100).

Train sets
----------
Every board in each difficulty enters ``<diff>.train`` (this generator
never down-samples train).

Test sets (held-out for LLM eval)
---------------------------------
* **easy.test** — all 99 easy boards *except* the one matching
  ``--easy-train-only-glob`` (default ``0096_*``), which stays as
  train-only so any single-board sanity command can target a known sample.
* **medium.test / hard.test** — 10 boards each, picked by:
    1. restrict to ``layer == 2``,
    2. (hard only) drop the top ``--hard-trim-frac`` of boards by ``pins``
       as outliers,
    3. sort by ``(pins, nets, sample)`` for deterministic ordering,
    4. split into 10 equal-sized quantile bins,
    5. take the lower-median board of each bin.

Output
------
A boards-json (the schema consumed by
``configs/quickstart/kdd/splits.json`` via the d3a/d3b/d3c aliases):

    {
      "easy":   {"train": [...100 boards...], "test": [...99 boards...]},
      "medium": {"train": [...287...],        "test": [...10...]},
      "hard":   {"train": [...292...],        "test": [...10...]},
      "dataset_dirs": {"train": "<--sorted-dir>", "test": "<--sorted-dir>"}
    }

Board names are written with their numeric prefix (e.g.
``0113_maytal_Maytal``) so they match the on-disk folder names directly.

Post-build overrides
--------------------
Manual swaps applied to the generator's output (and not yet wired as a CLI
flag) are recorded in the JSON's top-level ``_test_overrides`` block.
medium.test carries one such swap (``0373_badge2016_Badge_init`` →
``0376_cat-trainer_teensy_base_pcb``) — same quantile bin and CSV pins, but
without the file-pad outlier (650 components / 668 pad objects).

One board is also dropped from the pool outright (reason + name in
``configs/datasets/README.md``); it falls in the medium tier, so the shipped
``configs/datasets/d3.json`` has medium.train = 286 rather than the 287 above.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from pathlib import Path

import numpy as np

_PREFIX_RE = re.compile(r"^(\d{4,5})_(.+)$")


def _load_prefix_map(sorted_dir: Path) -> dict[str, str]:
    """Map bare ``sample`` → ``NNNN_<sample>`` from on-disk folder names."""
    out: dict[str, str] = {}
    for entry in sorted_dir.iterdir():
        m = _PREFIX_RE.match(entry.name)
        if m:
            out[m.group(2)] = entry.name
    return out


def _classify(
    rows: list[dict[str, str]],
    *,
    easy_rows: int,
    medium_pin_threshold: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    easy = rows[:easy_rows]
    rest = rows[easy_rows:]
    medium = [r for r in rest if int(r["pins"]) <= medium_pin_threshold]
    hard   = [r for r in rest if int(r["pins"]) >  medium_pin_threshold]
    return easy, medium, hard


def _decile_median_pick(
    items: list[dict[str, str]], n: int
) -> list[dict[str, str]]:
    """n equal-size quantile bins by pins; pick lower-median of each bin."""
    s = sorted(items, key=lambda r: (int(r["pins"]), int(r["nets"]), r["sample"]))
    picks: list[dict[str, str]] = []
    for chunk in np.array_split(np.arange(len(s)), n):
        bucket = [s[i] for i in chunk]
        mid = (len(bucket) - 1) // 2
        picks.append(bucket[mid])
    return picks


def build(args: argparse.Namespace) -> dict:
    rows = list(csv.DictReader(args.csv.open()))
    if not rows:
        raise SystemExit(f"[error] empty CSV: {args.csv}")

    prefix_by_sample = _load_prefix_map(args.sorted_dir)
    missing = [r["sample"] for r in rows if r["sample"] not in prefix_by_sample]
    if missing:
        raise SystemExit(
            f"[error] {len(missing)} CSV samples have no matching folder under "
            f"{args.sorted_dir}: first 3 = {missing[:3]}"
        )

    easy_rows, medium_rows, hard_rows = _classify(
        rows,
        easy_rows=args.easy_rows,
        medium_pin_threshold=args.medium_pin_threshold,
    )

    # Train = every board in that difficulty, ordered by the CSV (≡ on-disk
    # sort order). Use prefixed names so they map to folders 1:1.
    easy_train   = [prefix_by_sample[r["sample"]] for r in easy_rows]
    medium_train = [prefix_by_sample[r["sample"]] for r in medium_rows]
    hard_train   = [prefix_by_sample[r["sample"]] for r in hard_rows]

    # easy.test = all easy boards except the train-only one.
    easy_train_only = [
        n for n in easy_train if fnmatch.fnmatch(n, args.easy_train_only_glob)
    ]
    if len(easy_train_only) != 1:
        raise SystemExit(
            f"[error] --easy-train-only-glob {args.easy_train_only_glob!r} "
            f"matched {len(easy_train_only)} boards (need exactly 1)."
        )
    easy_test = [n for n in easy_train if n != easy_train_only[0]]

    # Layer-2 quantile-median pick for medium / hard.
    medium_l2 = [r for r in medium_rows if int(r["layers"]) == 2]
    hard_l2   = [r for r in hard_rows   if int(r["layers"]) == 2]

    # Hard: drop the top trim_frac by pins before picking.
    hard_l2_sorted = sorted(hard_l2, key=lambda r: int(r["pins"]))
    n_drop = int(round(len(hard_l2_sorted) * args.hard_trim_frac))
    hard_l2_kept = hard_l2_sorted[:-n_drop] if n_drop else hard_l2_sorted

    medium_test = sorted(
        prefix_by_sample[r["sample"]]
        for r in _decile_median_pick(medium_l2, args.medium_n)
    )
    hard_test = sorted(
        prefix_by_sample[r["sample"]]
        for r in _decile_median_pick(hard_l2_kept, args.hard_n)
    )

    sorted_dir_str = str(args.sorted_dir.resolve())
    return {
        "easy":   {"train": easy_train,   "test": easy_test},
        "medium": {"train": medium_train, "test": medium_test},
        "hard":   {"train": hard_train,   "test": hard_test},
        "dataset_dirs": {"train": sorted_dir_str, "test": sorted_dir_str},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True,
                   help="pcb_characteristics_exacad_sorted.csv")
    p.add_argument("--sorted-dir", type=Path, required=True,
                   help="Directory containing NNNN_<name>/ board folders.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output boards-json (e.g. configs/datasets/d3.json).")
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
    p.add_argument("--easy-train-only-glob", default="0096_*",
                   help="Glob matching the single easy board that stays in "
                        "train but is excluded from easy.test (default 0096_*).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.is_file():
        print(f"[error] --csv not found: {args.csv}", file=sys.stderr)
        return 2
    if not args.sorted_dir.is_dir():
        print(f"[error] --sorted-dir not a directory: {args.sorted_dir}", file=sys.stderr)
        return 2

    payload = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    n_easy_tr, n_easy_te = len(payload["easy"]["train"]),   len(payload["easy"]["test"])
    n_med_tr,  n_med_te  = len(payload["medium"]["train"]), len(payload["medium"]["test"])
    n_hard_tr, n_hard_te = len(payload["hard"]["train"]),   len(payload["hard"]["test"])
    print(f"[d3-build] wrote {args.out}")
    print(f"  easy   train={n_easy_tr:3d}  test={n_easy_te:3d}")
    print(f"  medium train={n_med_tr:3d}  test={n_med_te:3d}")
    print(f"  hard   train={n_hard_tr:3d}  test={n_hard_te:3d}")
    print(f"  dataset_dirs.train = {payload['dataset_dirs']['train']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
