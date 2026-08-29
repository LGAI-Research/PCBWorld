#!/usr/bin/env python3
"""Table 23 -- D1 grid-size sweep, per-seed std breakdown.

For each (grid, method in {Ours/PPO, Jumanji A2C, SABLE}) reports
  clean_pass(@5) | potential | routability | WL
as ``mean ± std`` across seeds (``†`` = single seed).

NOTE: there is no D1 ``freerouting`` cell on disk, so the Freerouting column from
the paper draft is omitted here (and flagged in the output).

Read-only: consumes per_rollout.csv; writes only under paper_outputs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

GRIDS = [10, 50, 100, 200, 500]
METHODS = [("Ours (PPO)", "transformer_pcbworld"),
           ("Jumanji A2C", "jumanji"),
           ("SABLE", "sable")]
METRICS = ["clean_pass", "potential_gain", "routability", "wl"]
ND = {"clean_pass": 2, "potential_gain": 2, "routability": 2, "wl": 1}
HEADER = ["Grid", "Method", "CleanPass@5", "Pot.gain", "Rout.", "WL"]


def main(argv=None) -> None:  # argv unused; uniform signature for draw_figure dispatch
    rows: list[list[str]] = []
    for g in GRIDS:
        for disp, leaf in METHODS:
            rel = f"d1/d1_grid{g}/{leaf}"
            if not C.cell_exists(rel):
                rows.append([str(g), disp, "--", "--", "--", "--"])
                continue
            C.warn_if_sparse(rel, ["clean_pass", "potential_gain",
                                   "routability", "wirelength_mm"])
            r = C.reduce_cell(rel, METRICS)
            a = r["agg"]
            rows.append([str(g), disp] +
                        [C.fmt_pm(a[m][0], a[m][1], nd=ND[m]) for m in METRICS])
    C.write_table("table23_d1_gridsweep", header=HEADER, rows=rows,
                  title="Table 23 -- D1 grid sweep (best@5, mean ± std over seeds). "
                        "Freerouting column omitted: no D1 freerouting cell on disk.")
    print("[table23] rows:")
    for r in rows:
        print("   ", " | ".join(r))


if __name__ == "__main__":
    main()
