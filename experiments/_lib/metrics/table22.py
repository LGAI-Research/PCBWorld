#!/usr/bin/env python3
"""Table 22 -- std breakdown companion to Table 3.

Same dataset blocks/rows as Table 3, but reports every metric as ``mean ± std``
(sample std across seeds; ``†`` marks single-seed cells where std is undefined),
and adds DRV, WL and Via.

best = final_potential winner; WL/Via/DRV averaged over all boards.

Read-only: consumes per_rollout.csv; writes only under paper_outputs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

METRICS = ["clean_pass", "potential_gain", "routability", "drv", "wl", "via", "time"]
# per-metric decimal places
ND = {"clean_pass": 2, "potential_gain": 2, "routability": 2,
      "drv": 2, "wl": 1, "via": 2, "time": 2}
HEADER = ["Dataset", "Method", "CleanPass@5", "Pot.gain", "Rout.",
          "DRV", "WL", "Via", "Time(s)"]


def cellval(agg, m):
    mean, std, _ = agg[m]
    return C.fmt_pm(mean, std, nd=ND[m])


def main(argv=None) -> None:  # argv unused; uniform signature for draw_figure dispatch
    rows: list[list[str]] = []
    for ds in ("D2", "D3-A", "D3-B"):
        method_rows = C.rows_for_dataset(ds)
        if not method_rows:
            print(f"[table22] {ds}: no cells on disk yet -- skipped", file=sys.stderr)
            continue
        for disp, rel in method_rows:
            C.warn_if_sparse(rel, ["clean_pass", "potential_gain",
                                   "wirelength_mm", "drv_errors_only_count"])
            r = C.reduce_cell(rel, METRICS)
            a = r["agg"]
            rows.append([ds, disp] + [cellval(a, m) for m in METRICS])
    C.write_table("table22_std", header=HEADER, rows=rows,
                  title="Table 22 -- mean ± std (best@5; D2 / D3-A / D3-B). "
                        "† = single-seed (std undefined).")
    print("[table22] rows:")
    for r in rows:
        print("   ", " | ".join(r))


if __name__ == "__main__":
    main()
