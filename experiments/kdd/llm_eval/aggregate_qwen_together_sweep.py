"""Aggregate Qwen-on-Together sweep results into a comparison table.

Walks the per-(model, task, scenario) ``overall_multi_k.json`` files emitted
by ``scripts/run_qwen_together_sweep.sh`` and produces:

    <sweep_root>/comparison.csv         long-format, one row per (model, task,
                                        scenario, k) with all metrics.
    <sweep_root>/comparison.md          model x k pivot tables (one per
                                        task+scenario), pass@k / rb_best /
                                        rb_mean.

Usage:
    python scripts/aggregate_qwen_together_sweep.py <sweep_root>
    python scripts/aggregate_qwen_together_sweep.py <sweep_root> --ks 1,5,10,25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


_METRIC_KEYS = (
    "boards_evaluated",
    "boards_failed",
    "pass_at_k",
    "pass_at_k_unbiased",
    "success_rate_at_k",
    "mean_success_rate",
    "routability_at_k_best_mean",
    "routability_at_k_best_std",
    "routability_at_k_mean_mean",
    "routability_at_k_mean_std",
)


def discover_runs(root: Path) -> list[tuple[str, str, str, Path]]:
    """Return (model, task, scenario, overall_multi_k.json path) tuples.

    Layout written by run_qwen_together_sweep.sh:
        <root>/<model_tag>/<task>/<set>_<mode>/overall_multi_k.json
    """
    out = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for scen_dir in sorted(task_dir.iterdir()):
                if not scen_dir.is_dir():
                    continue
                multi = scen_dir / "overall_multi_k.json"
                if not multi.is_file():
                    # Fall back to overall.json if multi-k wasn't written.
                    fallback = scen_dir / "overall.json"
                    if fallback.is_file():
                        multi = fallback
                    else:
                        continue
                out.append((model_dir.name, task, scen_dir.name, multi))
    return out


def load_per_k(path: Path) -> tuple[list[int], dict[int, dict], dict]:
    """Return (ks, per_k metrics dict, top-level meta)."""
    blob = json.loads(path.read_text())
    meta = {
        "api_model": blob.get("api_model", ""),
        "api_provider": blob.get("api_provider", ""),
        "mode": blob.get("mode", ""),
        "num_samples": blob.get("num_samples", blob.get("k", 0)),
        "wall_time_sec": blob.get("wall_time_sec", 0.0),
    }
    if "per_k" in blob:
        ks = sorted(int(k) for k in blob["per_k"].keys())
        per_k = {int(k): blob["per_k"][str(k)] for k in ks}
    else:
        # Single-k overall.json fallback — wrap as one-entry per_k.
        k = int(blob.get("k", 0))
        ks = [k]
        per_k = {k: blob}
    return ks, per_k, meta


def build_long_rows(runs: list[tuple[str, str, str, Path]],
                    ks_filter: list[int] | None) -> list[dict]:
    rows: list[dict] = []
    for model, task, scenario, path in runs:
        try:
            ks, per_k, meta = load_per_k(path)
        except Exception as exc:
            print(f"[WARN] failed to load {path}: {exc}", file=sys.stderr)
            continue
        for k in ks:
            if ks_filter is not None and k not in ks_filter:
                continue
            row = {
                "model": model,
                "api_model": meta["api_model"],
                "task": task,
                "scenario": scenario,
                "k": k,
                "num_samples": meta["num_samples"],
                "mode": meta["mode"],
            }
            stats = per_k.get(k, {}) or {}
            for key in _METRIC_KEYS:
                row[key] = stats.get(key, "")
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = ["model", "api_model", "task", "scenario", "k",
                  "num_samples", "mode", *_METRIC_KEYS]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_markdown(rows: list[dict], path: Path) -> None:
    """Pivot rows into one table per (task, scenario), models x k.

    For each cell we report ``pass@k_unb / rb_best / rb_mean`` so the table
    captures the three headline metrics at a glance.
    """
    by_ts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_ts[(r["task"], r["scenario"])].append(r)

    out: list[str] = ["# Qwen × Together sweep results\n"]
    for (task, scenario) in sorted(by_ts.keys()):
        block = by_ts[(task, scenario)]
        models = sorted({r["model"] for r in block})
        ks = sorted({int(r["k"]) for r in block})
        out.append(f"\n## {task} / {scenario}\n")
        out.append("Cell = `pass@k_unb / rb_best / rb_mean`. "
                   "Sample budget = `num_samples` per board.\n\n")
        # Header
        header = ["model", "n"] + [f"k={k}" for k in ks]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for m in models:
            mrows = [r for r in block if r["model"] == m]
            if not mrows:
                continue
            n = mrows[0].get("num_samples", "")
            cells = [m, str(n)]
            by_k = {int(r["k"]): r for r in mrows}
            for k in ks:
                r = by_k.get(k)
                if r is None:
                    cells.append("—")
                    continue
                pk = _fmt(r.get("pass_at_k_unbiased", r.get("pass_at_k", "")))
                rb = _fmt(r.get("routability_at_k_best_mean", ""))
                rm = _fmt(r.get("routability_at_k_mean_mean", ""))
                cells.append(f"{pk} / {rb} / {rm}")
            out.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("sweep_root", type=Path,
                   help="Path like eval_out/qwen_together_sweep/<DATE_TAG>/")
    p.add_argument("--ks", default="",
                   help="Optional comma-separated subset of k to include "
                        "(default: all ks present in the run).")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Override CSV output path (default <sweep_root>/comparison.csv).")
    p.add_argument("--out-md", type=Path, default=None,
                   help="Override Markdown output path (default <sweep_root>/comparison.md).")
    args = p.parse_args()

    root = args.sweep_root.resolve()
    if not root.is_dir():
        print(f"[ERROR] not a directory: {root}", file=sys.stderr)
        return 2

    ks_filter = None
    if args.ks:
        try:
            ks_filter = sorted({int(x) for x in args.ks.split(",") if x.strip()})
        except ValueError:
            print(f"[ERROR] --ks must be ints: {args.ks!r}", file=sys.stderr)
            return 2

    runs = discover_runs(root)
    if not runs:
        print(f"[ERROR] no runs discovered under {root}", file=sys.stderr)
        return 2

    rows = build_long_rows(runs, ks_filter)
    csv_path = args.out_csv or (root / "comparison.csv")
    md_path = args.out_md or (root / "comparison.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"  runs found  : {len(runs)}")
    print(f"  rows        : {len(rows)}")
    print(f"  csv         : {csv_path}")
    print(f"  markdown    : {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
