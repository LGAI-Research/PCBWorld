#!/usr/bin/env python3
"""Tables 24 & 25 -- PPO (per-step) / GRPO / PPO (terminal) on D2, D3-A and D3-B (D3-B new).

Table 24 (quality, mean ± std over seeds; best = final_potential winner):
  Routability | Success(clean) | DRV | WL Ratio (vs Freerouting) | Vias | Time(s)
  WL Ratio is computed per seed as (method WL / Freerouting WL on that split),
  then mean ± std over seeds.

Table 25 (serial timing distribution, from per_board_rollout_time):
  Boards | Mean | Median | P95 | Max  (board-level mean over its rollouts).
  NOTE: this is the parallel-rollout per-board time available in per_rollout.csv,
  not the paper's seed42 DRC-disabled serial pass; flagged in the caption.

Read-only: consumes per_rollout.csv; writes only under paper_outputs/.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

DATASETS = {"D2": "d2a", "D3-A": "d3/d3a", "D3-B": "d3/d3b"}
CONFIGS = [("PPO (per-step)", "transformer_pcbworld"),
           ("GRPO", "transformer_pcbworld_grpo"),
           ("PPO (terminal)", "transformer_pcbworld_episodic")]


def fr_wl(pre: str) -> float | None:
    rel = f"{pre}/freerouting"
    if not C.cell_exists(rel):
        return None
    return C.reduce_cell(rel, ["wl"])["grand"]["wl"]


def board_times(rel: str) -> list[float]:
    """Board-level mean of per_board_rollout_time over each board's rollouts."""
    rows = C.load_rollouts(rel)
    cell = C.disk_cell_name(rel)
    byb: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        p = C.u.parse_cell_artifact(r.get("artifact_path", "") or "", cell)
        if p is None:
            continue
        t = C.u.parse_metric(r.get("per_board_rollout_time"))
        if t is not None:
            byb[p[0]].append(t)
    return [sum(v) / len(v) for v in byb.values() if v]


def main(argv=None) -> None:  # argv unused; uniform signature for draw_figure dispatch
    import numpy as np

    # ---------------- Table 24: quality ----------------
    t24_header = ["Dataset", "Config", "Routability", "Success(clean)", "DRV",
                  "WL Ratio", "Vias", "Time(s)"]
    t24_rows: list[list[str]] = []
    # ---------------- Table 25: timing -----------------
    t25_header = ["Dataset", "Config", "Boards", "Mean", "Median", "P95", "Max"]
    t25_rows: list[list[str]] = []

    for ds, pre in DATASETS.items():
        wl_fr = fr_wl(pre)
        for disp, leaf in CONFIGS:
            rel = f"{pre}/{leaf}"
            if not C.cell_exists(rel):
                t24_rows.append([ds, disp] + ["--"] * 6)
                t25_rows.append([ds, disp] + ["--"] * 5)
                continue
            C.warn_if_sparse(rel, ["clean_pass", "final_potential",
                                   "routability", "wirelength_mm"])
            r = C.reduce_cell(rel, ["routability", "clean_pass", "drv", "wl", "via", "time"])
            a = r["agg"]
            # WL ratio per seed -> mean ± std
            if wl_fr:
                ratios = [(v / wl_fr if v is not None else None)
                          for v in r["per_seed"]["wl"].values()]
                wlr = C.fmt_pm(C._mean(ratios), C.u.sample_std(ratios), nd=2)
            else:
                wlr = "--"
            t24_rows.append([
                ds, disp,
                C.fmt_pm(*a["routability"][:2], nd=2),
                C.fmt_pm(*a["clean_pass"][:2], nd=2),
                C.fmt_pm(*a["drv"][:2], nd=2),
                wlr,
                C.fmt_pm(*a["via"][:2], nd=2),
                C.fmt(a["time"][0]),
            ])

            bt = board_times(rel)
            if bt:
                arr = np.array(bt)
                t25_rows.append([ds, disp, str(len(bt)),
                                 f"{arr.mean():.2f}", f"{np.median(arr):.2f}",
                                 f"{np.percentile(arr, 95):.2f}", f"{arr.max():.2f}"])
            else:
                t25_rows.append([ds, disp, "0", "--", "--", "--", "--"])

    C.write_table("table24_pg_quality", header=t24_header, rows=t24_rows,
                  title="Table 24 -- PPO (per-step)/GRPO/PPO (terminal) quality (mean ± std; D2/D3-A/D3-B). "
                        "† = single seed.")
    C.write_table("table25_pg_timing", header=t25_header, rows=t25_rows,
                  title="Table 25 -- per-board timing distribution from "
                        "per_board_rollout_time (parallel-rollout, not seed42 serial).")
    print("[table24] rows:");  [print("   ", " | ".join(x)) for x in t24_rows]
    print("[table25] rows:");  [print("   ", " | ".join(x)) for x in t25_rows]


if __name__ == "__main__":
    main()
