#!/usr/bin/env python3
"""Sort the v10 PCBench boards and retain source-folder identity.

Reads v10 guide boards and PCBench ``final.json`` statistics, sorts by
``(pins, nets, components)``, and copies successful guides to
``<out>/<NNNN>_<source-folder>/``. The CSV carries both the raw source folder
(``sample``) and its v10 rank-based folder name (``sorted_id``), so consumers
can join on stable source identity after sort ranks change.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

CSV_NAME = "pcb_characteristics_exacad_sorted.csv"
FOOTPRINT_RE = re.compile(r"^\t\(footprint ", re.MULTILINE)
PREFIXED_RE = re.compile(r"^\d{4}_")
GUIDE_STEM = "processed_v10_guide_v3"


def _env_dir(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} is not set. Point it at the matching directory "
            "(see tools/datagen/pcbench_prep/README.md)."
        )
    return Path(val)


def board_stats(folder: Path) -> dict | None:
    final = folder / "final.json"
    pcb = folder / f"{GUIDE_STEM}.kicad_pcb"
    if not final.is_file() or not pcb.is_file():
        return None
    try:
        data = json.loads(final.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        nets = data.get("nets")
        layers = data.get("layers")
        if (not isinstance(nets, dict) or not isinstance(layers, list)
                or any(not isinstance(pads, list) for pads in nets.values())):
            return None
        return {
            "sample": folder.name,
            "nets": len(nets),
            "components": len(FOOTPRINT_RE.findall(pcb.read_text(encoding="utf-8"))),
            "pins": sum(len(pads) for pads in nets.values()),
            "layers": len(layers),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _guide_statuses(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"guide result report missing: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid guide result report {path}: {exc}") from exc
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("folder"), str)
        or not isinstance(row.get("status"), str) for row in rows
    ):
        raise SystemExit(f"malformed guide result report: {path}")
    statuses = {row["folder"]: row["status"] for row in rows}
    if len(statuses) != len(rows):
        raise SystemExit(f"duplicate board folders in guide result report: {path}")
    return statuses


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--guide-report", type=Path, default=None,
                   help="make_guide.py JSON log (default: <newdrc>/_guide_results.json)")
    p.add_argument("--report", type=Path, default=None,
                   help="sort outcomes JSON (default: <newdrc>/_sort_results.json)")
    args = p.parse_args()

    src_dir = _env_dir("PCBENCH_NEWDRC_OUT")
    out_dir = _env_dir("PCBENCH_SORTED_OUT")
    guide_report = args.guide_report or src_dir / "_guide_results.json"
    result_report = args.report or src_dir / "_sort_results.json"
    folders = sorted(f for f in src_dir.iterdir() if f.is_dir())
    statuses = _guide_statuses(guide_report)
    folder_names = {f.name for f in folders}
    report_names = set(statuses)
    if report_names != folder_names:
        missing = sorted(folder_names - report_names)
        unexpected = sorted(report_names - folder_names)
        raise SystemExit(
            f"guide report does not account for newdrc inputs: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    rows, outcomes, stage_errors = [], [], []
    for folder in folders:
        guide_status = statuses[folder.name]
        if guide_status == "fail":
            outcomes.append({"sample": folder.name, "status": "guide_rejected"})
            continue
        if guide_status != "pass":
            outcomes.append({"sample": folder.name, "status": "stage_error",
                             "guide_status": guide_status})
            stage_errors.append(folder.name)
            continue
        stats = board_stats(folder)
        if stats is None:
            outcomes.append({"sample": folder.name, "status": "stage_error",
                             "guide_status": guide_status,
                             "error": "missing or malformed final.json/guide board"})
            stage_errors.append(folder.name)
            continue
        rows.append(stats)

    rows.sort(key=lambda row: (row["pins"], row["nets"], row["components"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = set()
    for i, row in enumerate(rows, start=1):
        sorted_id = f"{i:04d}_{row['sample']}"
        row["sorted_id"] = sorted_id
        dst = out_dir / sorted_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir / row["sample"], dst)
        written.add(dst.name)
        outcomes.append({"sample": row["sample"], "sorted_id": sorted_id,
                         "status": "ok"})

    result_report.parent.mkdir(parents=True, exist_ok=True)
    outcomes.sort(key=lambda result: result["sample"])
    result_report.write_text(json.dumps(outcomes, indent=2, ensure_ascii=False), encoding="utf-8")
    if not rows:
        raise SystemExit(f"no successful guide boards under {src_dir} — output left untouched")

    stale = [d for d in out_dir.iterdir()
             if d.is_dir() and PREFIXED_RE.match(d.name) and d.name not in written]
    for d in stale:
        shutil.rmtree(d)
    if stale:
        print(f"Removed {len(stale)} stale <NNNN>_<source-folder>/ entries from v10 output")

    with open(out_dir / CSV_NAME, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["sample", "sorted_id", "nets", "components", "pins", "layers"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} boards to {out_dir} (+ {CSV_NAME})")
    print(f"Guide rejections: {sum(r['status'] == 'guide_rejected' for r in outcomes)}")
    if stage_errors:
        print(f"Sort stage errors: {len(stage_errors)}; see {result_report}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
