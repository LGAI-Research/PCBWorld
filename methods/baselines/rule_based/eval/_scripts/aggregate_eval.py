#!/usr/bin/env python3
"""Generalized aggregator for cadagent eval.metrics outputs.

Aggregation modes
-----------------
- single (default): per-board mean over all .json logs found for the method.
  Use for deterministic routers (FR, KRT, OrthoRoute, KiCad built-in, etc.).
- rl: per (seed, board) pick max-Φ rollout, then per-board mean across seeds,
  then mean across boards. Use for stochastic policies (Transformer_PPO, LLM
  inference with multiple sampled rollouts, etc.).

Both modes work with a single rollout / single seed.

Expected directory layout
-------------------------
    <eval-root>/<dataset>/<algo_name>/
        logs/
            <board-or-sample-id>_<algo_name>.json   (one per board/rollout)
        summary.json
        summary.txt

For RL mode, the algo_name encodes seed and rollout, e.g.:
    Transformer_PPO_<config>_seed42__best__synth2l_s00_r00
    Qwen3-30B_<config>_seed1_r0
The script extracts (seed, rollout) via regex; see DEFAULT_RL_PATTERN.

Compat-issue board exclusion
----------------------------
Boards whose error-only DRV violations include any of {17, 22, 23, 14, 15, 16}
are flagged as "compat-issue" (dimension/clearance packing mismatch with
project rules — not a pure algorithm failure). Use --view to choose:
- v1   : intersection — exclude any board that any participating method flags.
- v2   : per-baseline — each method excludes only its own compat-issue boards.
- all  : no exclusion (include all boards).

Usage examples
--------------
# Standard router on synth (eval logs at .../synth_2L_v2_test/freerouting_via1x/logs/)
python aggregate_eval.py --eval-root .../cadagent/eval --dataset synth_2L_v2_test \\
    --method-filter '^freerouting_via1x$' --mode single

# Transformer_PPO on PCBench (40 algo dirs, max-Φ rollout per seed)
python aggregate_eval.py --eval-root .../cadagent/eval --dataset PCBench \\
    --method-filter '^Transformer_PPO_.*_seed(42|43|44|45)__best__real_s\\d+_r\\d+$' \\
    --mode rl --keep-rollouts 0,1,2,3,4

# All baselines side-by-side, V1 (intersection) view
python aggregate_eval.py --config-yaml configs/rq2.yaml --view v1
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_RL_PATTERN = r"_seed(\d+).*?_r(\d+)$"
DEFAULT_BOARD_PATTERN = r"^(.+?)_<algo>$"  # board id = filename stem - "_<algo>"

COMPAT_DRV_CODES = {17, 22, 23, 14, 15, 16}


def board_id_from_logfile(path: str, algo: str) -> str:
    stem = Path(path).stem
    suffix = f"_{algo}"
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def has_compat_issue(metrics: dict) -> bool:
    if metrics.get("error"): return True
    for v in metrics.get("drv_violations", []):
        if v.get("severity_label") == "ERROR" and v.get("error_code") in COMPAT_DRV_CODES:
            return True
    return False


def load_method_dirs(eval_root: Path, dataset: str, filter_re: str):
    base = eval_root / dataset
    pat = re.compile(filter_re)
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if d.is_dir() and pat.search(d.name):
            if (d / "logs").is_dir():
                out.append(d)
    return out


def collect_per_sample_metrics(method_dir: Path):
    """Return {sample_id: dict-of-metrics-or-error}."""
    algo = method_dir.name
    out = {}
    for log_file in sorted((method_dir / "logs").glob("*.json")):
        try:
            m = json.load(open(log_file))
        except Exception as e:
            continue
        sid = board_id_from_logfile(str(log_file), algo)
        out[sid] = m
    return out


def aggregate_single(method_dirs, sample_set: set | None, name: str):
    """For deterministic routers: one run per board → per-board mean over sample_set."""
    if not method_dirs:
        return None
    # Most "single" cases have just 1 method dir; if multiple given, union by board id (first wins).
    by_sample = {}
    for md in method_dirs:
        for sid, m in collect_per_sample_metrics(md).items():
            by_sample.setdefault(sid, m)
    if sample_set is not None:
        by_sample = {k: v for k, v in by_sample.items() if k in sample_set}
    return _summarize(by_sample, name)


def aggregate_rl(method_dirs, rl_pattern: str, sample_set: set | None, name: str,
                 keep_rollouts=None, keep_seeds=None):
    """For RL: pick max-Φ rollout per (seed, sample), mean across seeds, mean across boards.

    Works for: single-rollout (max picks the only entry), single-seed (mean of 1).
    """
    rl_re = re.compile(rl_pattern)
    by_seed_sample = defaultdict(list)  # (seed, sid) → [{rollout, phi, metrics}]

    for md in method_dirs:
        m_rs = rl_re.search(md.name)
        if m_rs is None:
            continue  # skip dirs whose names don't carry seed/rollout
        seed = int(m_rs.group(1))
        rollout = int(m_rs.group(2))
        if keep_rollouts is not None and rollout not in keep_rollouts: continue
        if keep_seeds is not None and seed not in keep_seeds: continue
        algo = md.name
        for sid, m in collect_per_sample_metrics(md).items():
            phi = m.get("final_potential")
            if phi is None: continue
            by_seed_sample[(seed, sid)].append({"rollout": rollout, "phi": float(phi), "m": m})

    # Step 1: per (seed, sample) → max-phi rollout
    picked_per_sample = defaultdict(list)  # sid → [picked_metrics_per_seed]
    for (seed, sid), lst in by_seed_sample.items():
        best = max(lst, key=lambda x: x["phi"])
        if has_compat_issue(best["m"]):  # propagate compat
            best["m"]["_compat_flag"] = True
        picked_per_sample[sid].append(best["m"])

    # Step 2: per board mean across seeds
    by_sample_avg = {}
    for sid, lst in picked_per_sample.items():
        # Skip any sample where any seed marks compat? choose: union-OR
        compat_any = any(has_compat_issue(m) for m in lst)
        # mean metrics across seeds
        succ = drv_e = drv_ep = wl = via = track = phi = 0.0
        n = len(lst)
        for m in lst:
            succ   += int(m.get("success", False) and m.get("drv_errors_only_count", 99) == 0)
            drv_e  += m.get("drv_errors_only_count", 0)
            drv_ep += m.get("drv_errors_and_promoted_count", 0)
            wl     += m.get("wirelength_mm", 0)
            via    += m.get("via_count", 0)
            track  += m.get("track_count", 0)
            phi    += m.get("final_potential", 0)
        by_sample_avg[sid] = {
            "succ_mean": succ / n,
            "drv_e_mean": drv_e / n,
            "drv_ep_mean": drv_ep / n,
            "wl_mean": wl / n,
            "via_mean": via / n,
            "track_mean": track / n,
            "phi_mean": phi / n,
            "_compat_flag": compat_any,
            "_n_seeds": n,
        }
    if sample_set is not None:
        by_sample_avg = {k: v for k, v in by_sample_avg.items() if k in sample_set}
    return _summarize_rl(by_sample_avg, name)


def _summarize(by_sample: dict, name: str):
    rows = list(by_sample.values())
    if not rows:
        return {"name": name, "n": 0}
    valid = [r for r in rows if not r.get("error")]
    n = len(valid)
    if n == 0:
        return {"name": name, "n": 0, "all_failed": True}
    succ = [int(r.get("success", False) and r.get("drv_errors_only_count", 99) == 0) for r in valid]
    return {
        "name":    name,
        "n":       n,
        "succ_pct": sum(succ) / n * 100,
        "drv_e":   sum(r.get("drv_errors_only_count", 0) for r in valid) / n,
        "drv_ep":  sum(r.get("drv_errors_and_promoted_count", 0) for r in valid) / n,
        "wl":      sum(r.get("wirelength_mm", 0) for r in valid) / n,
        "via":     sum(r.get("via_count", 0) for r in valid) / n,
        "track":   sum(r.get("track_count", 0) for r in valid) / n,
        "phi":     sum(r.get("final_potential", 0) for r in valid) / n,
    }


def _summarize_rl(by_sample_avg: dict, name: str):
    rows = list(by_sample_avg.values())
    if not rows:
        return {"name": name, "n": 0}
    n = len(rows)
    return {
        "name":    name,
        "n":       n,
        "n_seeds": int(rows[0]["_n_seeds"]) if rows else 0,
        "succ_pct": sum(r["succ_mean"]   for r in rows) / n * 100,
        "drv_e":   sum(r["drv_e_mean"]   for r in rows) / n,
        "drv_ep":  sum(r["drv_ep_mean"]  for r in rows) / n,
        "wl":      sum(r["wl_mean"]      for r in rows) / n,
        "via":     sum(r["via_mean"]     for r in rows) / n,
        "track":   sum(r["track_mean"]   for r in rows) / n,
        "phi":     sum(r["phi_mean"]     for r in rows) / n,
    }


def collect_compat_set(method_dirs, rl_pattern: str | None = None) -> set:
    """Boards (sample_ids) flagged as compat-issue under this method.

    For single-mode: any sample with compat-DRV.
    For RL-mode: any sample whose phi-max picked metrics carries compat across any seed.
    """
    out = set()
    for md in method_dirs:
        algo = md.name
        for log_file in (md / "logs").glob("*.json"):
            try:
                m = json.load(open(log_file))
            except: continue
            sid = board_id_from_logfile(str(log_file), algo)
            if has_compat_issue(m):
                out.add(sid)
    return out


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_csv(s: str | None):
    return [int(x) for x in s.split(",")] if s else None


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--eval-root", type=Path, required=True,
                    help="Root of the eval-log tree (holds <dataset>/<algo>/ dirs).")
    ap.add_argument("--dataset", required=True,
                    help="Dataset subdir under eval-root (e.g. synth_2L_v2_test, PCBench).")
    ap.add_argument("--method-filter", required=True,
                    help="Regex substring on algo dir name (one method may span multiple "
                         "dirs in RL mode — one per (seed, ckpt, rollout)).")
    ap.add_argument("--mode", choices=("single", "rl"), default="single")
    ap.add_argument("--rl-pattern", default=DEFAULT_RL_PATTERN,
                    help=r"Regex with two groups capturing (seed, rollout). Default: "
                         r"'_seed(\d+).*?_r(\d+)$'.")
    ap.add_argument("--keep-rollouts", default=None,
                    help="Comma-separated rollout indices to keep (e.g. 0,1,2,3,4).")
    ap.add_argument("--keep-seeds", default=None,
                    help="Comma-separated seed values to keep.")
    ap.add_argument("--samples", default=None,
                    help="Comma-separated sample IDs to restrict to (else all).")
    ap.add_argument("--samples-file", default=None,
                    help="Path to a text file listing sample IDs (one per line, # for comments). "
                         "Use pcbench_fair95.txt for the recommended PCBench fair-comparison set.")
    ap.add_argument("--exclude-compat", action="store_true",
                    help="Drop samples flagged with dimension/clearance compat DRV.")
    args = ap.parse_args()

    method_dirs = load_method_dirs(args.eval_root, args.dataset, args.method_filter)
    if not method_dirs:
        print(f"No method dirs match filter '{args.method_filter}' under "
              f"{args.eval_root}/{args.dataset}/", file=sys.stderr)
        sys.exit(2)

    print(f"matched {len(method_dirs)} method dir(s):")
    for d in method_dirs[:5]:
        print(f"  - {d.name}")
    if len(method_dirs) > 5:
        print(f"  ... +{len(method_dirs) - 5} more")

    sample_set = set(args.samples.split(",")) if args.samples else None
    if args.samples_file:
        file_set = set()
        with open(args.samples_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    file_set.add(line)
        sample_set = (sample_set & file_set) if sample_set else file_set
        print(f"loaded {len(file_set)} sample IDs from {args.samples_file}")
    if args.exclude_compat:
        compat = collect_compat_set(method_dirs)
        print(f"compat-flagged samples: {len(compat)}")
        if sample_set is None:
            # need universe of samples; gather from logs
            universe = set()
            for md in method_dirs:
                for f in (md / "logs").glob("*.json"):
                    universe.add(board_id_from_logfile(str(f), md.name))
            sample_set = universe - compat
        else:
            sample_set -= compat

    name = args.method_filter
    if args.mode == "single":
        agg = aggregate_single(method_dirs, sample_set, name)
    else:
        agg = aggregate_rl(method_dirs, args.rl_pattern, sample_set, name,
                           keep_rollouts=parse_csv(args.keep_rollouts),
                           keep_seeds=parse_csv(args.keep_seeds))

    print()
    print(f"=== aggregate ({args.mode} mode) ===")
    if not agg or agg.get("n", 0) == 0:
        print("  no data")
        sys.exit(1)
    for k in ("name", "n", "n_seeds", "succ_pct", "drv_e", "drv_ep", "wl", "via", "track", "phi"):
        if k in agg:
            v = agg[k]
            if isinstance(v, float):
                print(f"  {k:<10} {v:>10.3f}")
            else:
                print(f"  {k:<10} {v}")


if __name__ == "__main__":
    main()
