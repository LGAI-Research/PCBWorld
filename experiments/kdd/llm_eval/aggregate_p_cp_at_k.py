#!/usr/bin/env python3
"""Table 2 — P@K / CP@K aggregator.

Walks a Table 2 (or Table 1 (b)) experiment tree for the per-level
``overall.json`` artefacts produced by
``experiments/_lib/metrics/score_rollouts.py``, then collapses them into
the P@K / CP@K cells that fill the paper table.

Expected layout under ``--root`` (auto-discovered via ``rglob``)::

    <root>/<model>/<split>/_eval/<level>/overall.json   # Table 2 (3 levels)
    <root>/<model>/<split>/_eval/pcbworld/overall.json  # Table 1 (b)

``<level>`` ∈ ``interactive`` / ``engine_free`` / ``plan_only`` (legacy disk
trees use ``pcbworld`` / ``codelevel`` / ``apilevel`` — accepted and
normalized). ``<model>`` and
``<split>`` are recovered from the path relative to ``--root``; passing a
single ``<model>/<split>`` dir directly works too.

Outputs (under ``--output-dir`` with stem ``--output-prefix``):

* ``<prefix>_long.csv``    — one row per (model, split, level)
* ``<prefix>_pivot.csv``   — one row per (model, split); levels as columns
* ``<prefix>_summary.md``  — human-readable markdown of the pivot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

KNOWN_LEVELS = ("interactive", "engine_free", "plan_only")
# legacy on-disk level dir names -> canonical (paper) names
_LEVEL_ALIASES = {"pcbworld": "interactive", "codelevel": "engine_free",
                  "apilevel": "plan_only"}
_LEVEL_ORDER = {lvl: i for i, lvl in enumerate(KNOWN_LEVELS)}


def safe_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 3) -> str:
    v = safe_float(value)
    if not math.isfinite(v):
        return "-"
    return f"{v:.{digits}f}"


def _board_k_mode(summary_csv: Path) -> int | None:
    """Mode of ``num_samples`` across boards in a scenario's summary.csv."""
    if not summary_csv.is_file():
        return None
    counts: dict[int, int] = defaultdict(int)
    with summary_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            k = safe_int(r.get("num_samples"), 0)
            if k > 0:
                counts[k] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _experiment_id(experiment_dir: Path, root: Path) -> tuple[str, str, str]:
    """Return (model, split, label).

    ``model``/``split`` come from the last two relative path components.
    ``label`` is the joined relative path (for traceability).
    """
    try:
        rel = experiment_dir.resolve().relative_to(root.resolve())
        parts = rel.parts
    except ValueError:
        parts = (experiment_dir.name,)

    if len(parts) >= 2:
        model, split = parts[-2], parts[-1]
    elif len(parts) == 1:
        model, split = "", parts[0]
    else:
        model, split = "", experiment_dir.name
    label = "/".join(parts) if parts else experiment_dir.name
    return model, split, label


def collect(root: Path, expected_k: int, strict: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for overall in sorted(root.rglob("_eval/*/overall.json")):
        level = overall.parent.name
        level = _LEVEL_ALIASES.get(level, level)
        if level not in KNOWN_LEVELS:
            continue
        experiment_dir = overall.parent.parent.parent   # .../<model>/<split>
        model, split, label = _experiment_id(experiment_dir, root)

        try:
            data = json.loads(overall.read_text())
        except Exception as exc:
            print(f"[warn] cannot parse {overall}: {exc}", file=sys.stderr)
            continue

        k_found = _board_k_mode(overall.parent / "summary.csv")
        if strict and k_found is not None and k_found != expected_k:
            raise SystemExit(
                f"[error] expected k={expected_k} but boards in {overall} "
                f"have num_samples={k_found}"
            )

        rows.append({
            "experiment": label,
            "model": model,
            "split": split,
            "level": level,
            "k": k_found if k_found is not None else expected_k,
            "boards_evaluated": safe_int(data.get("boards_evaluated"), 0),
            "samples_evaluated": safe_int(data.get("samples_evaluated"), 0),
            f"p_at_{expected_k}":  safe_float(data.get("pass_at_k"), 0.0),
            f"cp_at_{expected_k}": safe_float(data.get("clean_pass_at_k"), 0.0),
            "mean_success_rate":            safe_float(data.get("mean_success_rate"), 0.0),
            "clean_mean_success_rate":      safe_float(data.get("clean_mean_success_rate"), 0.0),
            "routability_at_k_best_mean":   safe_float(data.get("routability_at_k_best_mean"), 0.0),
            "routability_at_k_mean_mean":   safe_float(data.get("routability_at_k_mean_mean"), 0.0),
            "final_potential_at_k_best_mean": safe_float(data.get("final_potential_at_k_best_mean"), 0.0),
            "drv_at_k_min_mean":            safe_float(data.get("drv_at_k_min_mean"), 0.0),
            "track_angle_drv_at_k_min_mean": safe_float(data.get("track_angle_drv_at_k_min_mean"), 0.0),
            "reward_config":  str(data.get("reward_config", "")),
            "check_angle":    safe_int(data.get("check_angle"), 0),
            "wall_time_sec":  safe_float(data.get("wall_time_sec"), 0.0),
            "source_overall_json": str(overall),
        })

    rows.sort(key=lambda r: (
        str(r["model"]),
        str(r["split"]),
        _LEVEL_ORDER.get(str(r["level"]), 99),
    ))
    return rows


def to_pivot(rows: list[dict[str, object]], k: int) -> list[dict[str, object]]:
    """One row per (model, split); each level contributes its own columns."""
    pivots: dict[tuple[str, str], dict[str, object]] = {}
    for r in rows:
        key = (str(r["model"]), str(r["split"]))
        pv = pivots.setdefault(key, {
            "model": r["model"],
            "split": r["split"],
            "experiment": r["experiment"],
        })
        lvl = str(r["level"])
        pv[f"{lvl}_boards"]      = r["boards_evaluated"]
        pv[f"{lvl}_p_at_{k}"]    = r[f"p_at_{k}"]
        pv[f"{lvl}_cp_at_{k}"]   = r[f"cp_at_{k}"]
    out = list(pivots.values())
    out.sort(key=lambda r: (str(r["model"]), str(r["split"])))
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_markdown(path: Path, pivot_rows: list[dict[str, object]], k: int) -> None:
    """Wide markdown: row per (model, split), columns per level."""
    header = ["model", "split"]
    for lvl in KNOWN_LEVELS:
        header += [f"{lvl} boards", f"{lvl} P@{k}", f"{lvl} CP@{k}"]
    align = ["---", "---"] + ["---:", "---:", "---:"] * len(KNOWN_LEVELS)

    lines = [
        f"# Table 2 — P@{k} / CP@{k}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
    ]
    for r in pivot_rows:
        row = [str(r.get("model", "")), str(r.get("split", ""))]
        for lvl in KNOWN_LEVELS:
            row += [
                str(r.get(f"{lvl}_boards", "")),
                fmt(r.get(f"{lvl}_p_at_{k}"),  3),
                fmt(r.get(f"{lvl}_cp_at_{k}"), 3),
            ]
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def print_table(rows: list[dict[str, object]], k: int) -> None:
    print()
    print(f"Table 2 — P@{k} / CP@{k}")
    print("=" * 96)
    hdr = (f"  {'model':<16} {'split':<8} {'level':<10} "
           f"{'boards':>6}  {'P@'+str(k):>7}  {'CP@'+str(k):>7}  "
           f"{'rout_best':>9}  {'mean_succ':>9}  {'clean_succ':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {str(r['model']):<16} {str(r['split']):<8} {str(r['level']):<10} "
            f"{safe_int(r['boards_evaluated']):>6}  "
            f"{fmt(r.get(f'p_at_{k}'), 3):>7}  "
            f"{fmt(r.get(f'cp_at_{k}'), 3):>7}  "
            f"{fmt(r.get('routability_at_k_best_mean'), 3):>9}  "
            f"{fmt(r.get('mean_success_rate'), 3):>9}  "
            f"{fmt(r.get('clean_mean_success_rate'), 3):>10}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", type=Path, required=True,
                   help="Table 2 / Table 1 (b) experiment root.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write the aggregated artefacts into.")
    p.add_argument("--output-prefix", default="table2",
                   help="File stem prefix (default: table2).")
    p.add_argument("--k", type=int, default=5,
                   help="K for P@K / CP@K labels (default 5).")
    p.add_argument("--strict", action="store_true",
                   help="Error out if any scenario's per-board num_samples != --k.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the stdout summary table.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"[error] not a directory: {root}", file=sys.stderr)
        return 2

    rows = collect(root, expected_k=args.k, strict=args.strict)
    if not rows:
        print(
            f"[error] no _eval/<level>/overall.json under {root}\n"
            f"        (run experiments/_lib/metrics/score_rollouts.py first)",
            file=sys.stderr,
        )
        return 2

    long_fields = [
        "model", "split", "level", "experiment",
        "k", "boards_evaluated", "samples_evaluated",
        f"p_at_{args.k}", f"cp_at_{args.k}",
        "mean_success_rate", "clean_mean_success_rate",
        "routability_at_k_best_mean", "routability_at_k_mean_mean",
        "final_potential_at_k_best_mean",
        "drv_at_k_min_mean", "track_angle_drv_at_k_min_mean",
        "reward_config", "check_angle", "wall_time_sec",
        "source_overall_json",
    ]
    pivot_fields = ["model", "split", "experiment"]
    for lvl in KNOWN_LEVELS:
        pivot_fields += [f"{lvl}_boards", f"{lvl}_p_at_{args.k}", f"{lvl}_cp_at_{args.k}"]

    out_dir = args.output_dir
    long_csv  = out_dir / f"{args.output_prefix}_long.csv"
    pivot_csv = out_dir / f"{args.output_prefix}_pivot.csv"
    md_path   = out_dir / f"{args.output_prefix}_summary.md"

    write_csv(long_csv, rows, long_fields)
    pivot_rows = to_pivot(rows, args.k)
    write_csv(pivot_csv, pivot_rows, pivot_fields)
    write_markdown(md_path, pivot_rows, args.k)

    print(f"[table2-report] long_rows={len(rows)} pivot_rows={len(pivot_rows)}")
    print(f"[table2-report] long  = {long_csv}")
    print(f"[table2-report] pivot = {pivot_csv}")
    print(f"[table2-report] md    = {md_path}")

    if not args.quiet:
        print_table(rows, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
