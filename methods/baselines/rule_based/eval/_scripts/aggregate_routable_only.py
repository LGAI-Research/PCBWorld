#!/usr/bin/env python3
"""Re-aggregate eval logs for the 4 RQ2 baselines, where DRV / via / WL / track
are averaged only over boards with routability == 1.0.

- Routability itself: full-set mean (unchanged from original).
- Success% (clean): full-set mean (unchanged).
- DRV / via / WL / track: success-only mean (NEW).
- Boards with no log treated as routability=0 (only KRT 0080/0099 for fair-95
  but those are now real eval logs after the unrouted-source replace).

Original logs are untouched. Output goes to:
    bench_results_0507_rtrenewal/<dataset>/<method>/{summary.json, summary.txt}

Sym (2L)  uses all 128 boards.
Real      uses fair-95 filter (pcbench_fair95.txt).

Transformer_PPO is RL: rollout-max(phi) -> seed-mean(per-board) -> board-mean.
The success-only filter is applied AFTER rollout-pick + seed-mean, on the
per-board averaged routability >= 1.0 - eps.
"""
from __future__ import annotations
import json, os, re, sys
import os
from collections import defaultdict
from pathlib import Path

EPS = 1e-9
# Roots of the eval-log trees. No baked-in defaults: the historical output
# tree sits on read-only shared storage, so point the env vars at your local
# copies of the logs (see baselines/eval/README.md).
def _require_env_path(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"{name} is not set — point it at the eval-log tree "
                         "(see methods/baselines/rule_based/eval/README.md).")
    return Path(val)


EVAL = _require_env_path("RQ2_EVAL_ROOT")
OUT = _require_env_path("RQ2_OUT_ROOT")
SAMPLES_FAIR95 = Path(os.environ.get(
    "RQ2_FAIR95",
    str(Path(__file__).resolve().parent / "pcbench_fair95.txt"),
))
RL_RE = re.compile(r"_seed(\d+).*?_r(\d+)$")

def load_fair95():
    s = set()
    for line in open(SAMPLES_FAIR95):
        x = line.strip()
        if x and not x.startswith("#"):
            s.add(x)
    return s

def board_id(logfile_name: str, algo: str) -> str:
    stem = logfile_name[:-5]  # strip .json
    suf = f"_{algo}"
    if stem.endswith(suf):
        return stem[:-len(suf)]
    return stem

def collect_logs(method_dir: Path) -> dict:
    """sid -> metrics dict"""
    algo = method_dir.name
    out = {}
    log_dir = method_dir / "logs"
    if not log_dir.is_dir(): return out
    for f in sorted(log_dir.iterdir()):
        if not f.name.endswith(".json"): continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        sid = board_id(f.name, algo)
        out[sid] = d
    return out

def aggregate_single(method_dir: Path, sample_set: set | None) -> dict:
    """Deterministic baseline: one log per board."""
    per_sample = collect_logs(method_dir)
    if sample_set is not None:
        per_sample = {k: v for k, v in per_sample.items() if k in sample_set}
    rows = []
    for sid, d in per_sample.items():
        rows.append({
            "sid": sid,
            "routability": d.get("routability", 0.0),
            "success_clean": int(bool(d.get("clean_pass", d.get("clean_success", False)))),
            "drv_e": d.get("drv_errors_only_count", 0),
            "drv_ep": d.get("drv_errors_and_promoted_count", 0),
            "wl": d.get("wirelength_mm", 0),
            "via": d.get("via_count", 0),
            "track": d.get("track_count", 0),
            "phi": d.get("final_potential", 0),
        })
    return summarize(rows)

def aggregate_rl(parent_dir: Path, name_filter: str, sample_set: set | None) -> dict:
    """Transformer_PPO: rollout-max(phi) -> seed-mean -> board-mean.
    name_filter is a substring; we only consume rollout dirs whose name contains it.
    """
    method_dirs = []
    for d in sorted(parent_dir.iterdir()):
        if not d.is_dir(): continue
        if name_filter not in d.name: continue
        if not RL_RE.search(d.name): continue       # need _seedN_rN encoding
        if not (d / "logs").is_dir(): continue
        method_dirs.append(d)

    by_seed_sample = defaultdict(list)
    for md in method_dirs:
        m = RL_RE.search(md.name)
        seed, rollout = int(m.group(1)), int(m.group(2))
        for sid, d in collect_logs(md).items():
            phi = d.get("final_potential")
            if phi is None: continue
            by_seed_sample[(seed, sid)].append({"rollout": rollout, "phi": float(phi), "m": d})

    # rollout-max per (seed, sid)
    picked_per_sid = defaultdict(list)
    for (seed, sid), lst in by_seed_sample.items():
        best = max(lst, key=lambda x: x["phi"])
        picked_per_sid[sid].append(best["m"])

    # per-board seed-mean
    rows = []
    for sid, ms in picked_per_sid.items():
        if sample_set is not None and sid not in sample_set: continue
        n = len(ms)
        rout = sum(m.get("routability", 0.0) for m in ms) / n
        # success_clean = fraction of seeds with clean_pass (legacy logs: clean_success)
        sc = sum(int(bool(m.get("clean_pass", m.get("clean_success", False)))) for m in ms) / n
        drv_e = sum(m.get("drv_errors_only_count", 0) for m in ms) / n
        drv_ep = sum(m.get("drv_errors_and_promoted_count", 0) for m in ms) / n
        wl = sum(m.get("wirelength_mm", 0) for m in ms) / n
        via = sum(m.get("via_count", 0) for m in ms) / n
        track = sum(m.get("track_count", 0) for m in ms) / n
        phi = sum(m.get("final_potential", 0) for m in ms) / n
        rows.append({
            "sid": sid, "routability": rout, "success_clean": sc,
            "drv_e": drv_e, "drv_ep": drv_ep, "wl": wl, "via": via,
            "track": track, "phi": phi, "_n_seeds": n,
        })

    s = summarize(rows)
    if rows:
        s["n_seeds"] = int(round(sum(r["_n_seeds"] for r in rows) / len(rows)))
    return s

def summarize(rows: list) -> dict:
    n_total = len(rows)
    if n_total == 0:
        return {"n_total": 0, "n_routable": 0}

    # full-set mean (routability + success%)
    routability_full = sum(r["routability"] for r in rows) / n_total
    succ_clean_full = sum(r["success_clean"] for r in rows) / n_total

    # success-only subset for DRV/via/WL/track/phi
    routable = [r for r in rows if r["routability"] >= 1.0 - EPS]
    n_routable = len(routable)

    if n_routable > 0:
        drv_e_so = sum(r["drv_e"] for r in routable) / n_routable
        drv_ep_so = sum(r["drv_ep"] for r in routable) / n_routable
        wl_so = sum(r["wl"] for r in routable) / n_routable
        via_so = sum(r["via"] for r in routable) / n_routable
        track_so = sum(r["track"] for r in routable) / n_routable
        phi_so = sum(r["phi"] for r in routable) / n_routable
    else:
        drv_e_so = drv_ep_so = wl_so = via_so = track_so = phi_so = None

    return {
        "n_total": n_total,
        "n_routable": n_routable,
        "routability_full_mean": round(routability_full, 6),
        "success_clean_full_mean": round(succ_clean_full, 6),
        "drv_e_routable_only": round(drv_e_so, 6) if drv_e_so is not None else None,
        "drv_ep_routable_only": round(drv_ep_so, 6) if drv_ep_so is not None else None,
        "wl_routable_only": round(wl_so, 6) if wl_so is not None else None,
        "via_routable_only": round(via_so, 6) if via_so is not None else None,
        "track_routable_only": round(track_so, 6) if track_so is not None else None,
        "phi_routable_only": round(phi_so, 6) if phi_so is not None else None,
    }

def write_outputs(out_dir: Path, summary: dict, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    full = {**meta, **summary}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(full, f, indent=2)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"dataset:        {meta['dataset']}\n")
        f.write(f"method:         {meta['method']}\n")
        f.write(f"source_logs:    {meta['source_dir']}\n")
        f.write(f"sample_filter:  {meta.get('sample_filter','(none)')}\n")
        f.write(f"n_total boards: {summary['n_total']}\n")
        f.write(f"n_routable:     {summary['n_routable']}  (routability == 1.0)\n")
        if "n_seeds" in summary:
            f.write(f"n_seeds (RL):   {summary['n_seeds']}\n")
        f.write("\n[full-set mean]\n")
        f.write(f"  routability:            {summary['routability_full_mean']}\n")
        f.write(f"  success_clean (frac):   {summary['success_clean_full_mean']}\n")
        f.write("\n[routable-only mean (routability == 1.0)]\n")
        for k in ("drv_e_routable_only","drv_ep_routable_only","wl_routable_only",
                  "via_routable_only","track_routable_only","phi_routable_only"):
            f.write(f"  {k:24} {summary.get(k)}\n")

def main():
    fair95 = load_fair95()

    JOBS = [
        # Sym (2L) — n=128, full set
        {"dataset": "synth_2L_v2_test", "method_label": "freerouting",        "kind": "single",
         "source": "freerouting_via1x", "sample_set": None},
        {"dataset": "synth_2L_v2_test", "method_label": "kicadroutingtools",  "kind": "single",
         "source": "kicadroutingtools_via1x", "sample_set": None},
        {"dataset": "synth_2L_v2_test", "method_label": "orthoroute",         "kind": "single",
         "source": "orthoroute_gpu_20260506", "sample_set": None},
        {"dataset": "synth_2L_v2_test", "method_label": "Transformer_PPO",    "kind": "rl",
         "source": "Transformer_PPO_v56_2L_dense_default", "sample_set": None},
        # Real (Small) — fair-95
        {"dataset": "PCBench", "method_label": "freerouting",        "kind": "single",
         "source": "freerouting_via1x_s00_r00", "sample_set": fair95},
        {"dataset": "PCBench", "method_label": "kicadroutingtools",  "kind": "single",
         "source": "kicadroutingtools_via1x_s00_r00", "sample_set": fair95},
        {"dataset": "PCBench", "method_label": "orthoroute",         "kind": "single",
         "source": "orthoroute_gpu_20260506", "sample_set": fair95},
        {"dataset": "PCBench", "method_label": "Transformer_PPO",    "kind": "rl",
         "source": "Transformer_PPO_v56_2L_dense_default", "sample_set": fair95},
    ]

    print(f"{'dataset':18} {'method':20} {'n':>4} {'n_rt':>5} {'rout':>7} {'DRV(e)':>9} {'WL':>9} {'Via':>8}")
    print("-" * 95)

    for job in JOBS:
        ds = job["dataset"]
        if job["kind"] == "single":
            md = EVAL / ds / job["source"]
            if not md.is_dir():
                print(f"  SKIP missing: {md}"); continue
            summary = aggregate_single(md, job["sample_set"])
            source_str = str(md)
        else:
            summary = aggregate_rl(EVAL / ds, job["source"], job["sample_set"])
            source_str = f"{EVAL / ds}/<rollout dirs containing '{job['source']}'>"

        meta = {
            "dataset": ds,
            "method":  job["method_label"],
            "source_dir": source_str,
            "sample_filter": ("pcbench_fair95.txt (n=95)"
                              if job["sample_set"] is not None else None),
            "aggregation_rule": ("DRV/via/WL/track/phi: mean over boards with routability==1.0; "
                                 "routability/success_clean: full-set mean."),
        }
        out_dir = OUT / ds / job["method_label"]
        write_outputs(out_dir, summary, meta)

        print(f"  {ds:18} {job['method_label']:20} "
              f"{summary['n_total']:>4} {summary['n_routable']:>5} "
              f"{summary['routability_full_mean']:>7.4f} "
              f"{summary['drv_e_routable_only'] if summary['drv_e_routable_only'] is not None else 'N/A':>9} "
              f"{summary['wl_routable_only'] if summary['wl_routable_only'] is not None else 'N/A':>9} "
              f"{summary['via_routable_only'] if summary['via_routable_only'] is not None else 'N/A':>8}")

    print(f"\nWrote summaries under: {OUT}")

if __name__ == "__main__":
    main()
