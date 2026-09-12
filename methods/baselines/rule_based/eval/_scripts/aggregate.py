#!/usr/bin/env python3
"""Reproduce the RQ2 baseline columns of `tab:rq2` (mean) and `tab:rq2-std`
(mean +/- std) from the per-board eval logs under `baselines/eval/`.

Rule (set by the paper):
  * Routability is the full-set mean.
  * DRV / WL / Via are restricted to boards with routability == 1.0
    ("routable-only" mean).
  * For stochastic methods with multiple seeds (Freerouting), we report
    seed-mean and sample stdev (n-1) over per-seed routable-only means.
  * PCBench (d3a) restricts to the fair-95 board subset.

Run from anywhere:
    python aggregate.py
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

EPS = 1e-9
EVAL_ROOT = Path(__file__).resolve().parent.parent
FAIR95_PATH = EVAL_ROOT / "_scripts" / "pcbench_fair95.txt"

DATASETS = ("d2a", "d3a")
ALGORITHMS = ("Freerouting", "OrthoRoute", "KiCadRoutingTools")


def load_fair95():
    return {l.strip() for l in open(FAIR95_PATH) if l.strip() and not l.startswith("#")}


def board_id(stem: str, algo: str) -> str:
    suf = f"_{algo}"
    return stem[: -len(suf)] if stem.endswith(suf) else stem


def load_seed(seed_dir: Path, algo: str, sample_set):
    out = {}
    logs = seed_dir / "logs"
    if not logs.is_dir():
        return out
    for j in sorted(logs.glob("*.json")):
        sid = board_id(j.stem, algo)
        if sample_set is not None and sid not in sample_set:
            continue
        d = json.loads(j.read_text())
        out[sid] = {
            "rout": d.get("routability", 0.0),
            "drv": d.get("drv_errors_only_count", 0),
            "wl":  d.get("wirelength_mm", 0) or 0,
            "via": d.get("via_count", 0),
        }
    return out


def seed_summary(rows_by_sid):
    """Per-seed routable-only summary."""
    rows = list(rows_by_sid.values())
    n = len(rows)
    if n == 0:
        return None
    rout = sum(r["rout"] for r in rows) / n
    rt = [r for r in rows if r["rout"] >= 1.0 - EPS]
    nr = len(rt)
    if nr == 0:
        drv = wl = via = None
    else:
        drv = sum(r["drv"] for r in rt) / nr
        wl  = sum(r["wl"]  for r in rt) / nr
        via = sum(r["via"] for r in rt) / nr
    return {"n": n, "n_routable": nr, "rout": rout, "drv": drv, "wl": wl, "via": via}


def aggregate(seed_summaries):
    """Across-seed mean and sample stdev (n-1)."""
    keys = ("rout", "drv", "wl", "via")
    out = {}
    for k in keys:
        vals = [s[k] for s in seed_summaries if s[k] is not None]
        if not vals:
            out[k] = {"mean": None, "std": None, "n_seeds": 0}
            continue
        m = sum(vals) / len(vals)
        sd = statistics.stdev(vals) if len(vals) >= 2 else None
        out[k] = {"mean": m, "std": sd, "n_seeds": len(vals)}
    out["n_boards"] = seed_summaries[0]["n"]
    return out


def fmt(metric, ndigits):
    m = metric["mean"]
    sd = metric["std"]
    if m is None:
        return "—"
    if sd is None:
        return f"{m:.{ndigits}f}"
    return f"{m:.{ndigits}f} +/- {sd:.{ndigits}f}"


def main():
    fair95 = load_fair95()

    print(f"{'dataset':6}  {'method':18}  {'seeds':>5}  {'n':>4}  "
          f"{'Rout.':>15}  {'DRV':>15}  {'WL':>17}  {'Via':>15}")
    print("-" * 110)

    for ds in DATASETS:
        ss = fair95 if ds == "d3a" else None
        for algo in ALGORITHMS:
            algo_dir = EVAL_ROOT / ds / algo
            if not algo_dir.is_dir():
                continue
            seed_dirs = sorted(algo_dir.glob("seed*"))
            seed_summaries = []
            for sd in seed_dirs:
                rows = load_seed(sd, algo, ss)
                s = seed_summary(rows)
                if s is not None:
                    seed_summaries.append(s)
            if not seed_summaries:
                continue
            agg = aggregate(seed_summaries)
            print(f"{ds:6}  {algo:18}  {len(seed_summaries):>5}  {agg['n_boards']:>4}  "
                  f"{fmt(agg['rout'], 2):>15}  {fmt(agg['drv'], 2):>15}  "
                  f"{fmt(agg['wl'], 2):>17}  {fmt(agg['via'], 2):>15}")


if __name__ == "__main__":
    main()
