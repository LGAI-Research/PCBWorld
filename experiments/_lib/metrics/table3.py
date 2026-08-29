#!/usr/bin/env python3
"""Table 3 -- routing quality on D2, D3-A and D3-B (D3-B is new).

Columns (best = final_potential winner, mean over seeds):
  clean_pass (CP@5) | potential | routability | time(s)

Rows per dataset: Reference (eval-only boards; D3 only), Freerouting, OrthoRoute,
KiCadRoutingTools, PCBWorld LLM backbones, PPO (per-step), PPO (terminal), GRPO --
only those present on disk (see common.rows_for_dataset).

Read-only: consumes per_rollout.csv; writes only under paper_outputs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

METRICS = ["clean_pass", "potential_gain", "routability", "time"]
HEADER = ["Dataset", "Method", "CleanPass@5", "Pot.gain", "Rout.", "Time(s)"]


def main(argv=None) -> None:  # argv unused; uniform signature for draw_figure dispatch
    rows: list[list[str]] = []
    for ds in ("D2", "D3-A", "D3-B"):
        method_rows = C.rows_for_dataset(ds)
        if not method_rows:
            print(f"[table3] {ds}: no cells on disk yet -- skipped", file=sys.stderr)
            continue
        for disp, rel in method_rows:
            C.warn_if_sparse(rel, ["clean_pass", "potential_gain", "routability"])
            r = C.reduce_cell(rel, METRICS)
            a = r["agg"]
            rows.append([
                ds, disp,
                C.fmt(a["clean_pass"][0]),
                C.fmt(a["potential_gain"][0]),
                C.fmt(a["routability"][0]),
                C.fmt(a["time"][0]),
            ])
    C.write_table("table3_quality", header=HEADER, rows=rows,
                  title="Table 3 -- routing quality (best@5; D2 / D3-A / D3-B)")
    print("[table3] rows:")
    for r in rows:
        print("   ", " | ".join(r))


if __name__ == "__main__":
    main()
