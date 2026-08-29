#!/usr/bin/env python3
"""Figure 6c -- D1 grid-size scalability, plotted as (best) clean-pass@5.

Replaces the old rollout-average routability axis with CP@5 (the
``final_potential``-winner's ``clean_pass``, grand-meaned over boards and
seeds; see common.reduce_cell). One line per method across grids.

Methods on disk: transformer_pcbworld (=PCBWorld/PPO), jumanji (A2C), sable.
jumanji@grid500 is OOM (absent) and simply omitted from its line.

Read-only: consumes per_rollout.csv; writes only under paper_outputs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

GRIDS = [10, 50, 100, 200, 500]
METHODS = [  # (display, leaf, color, marker)
    ("PCBWorld (PPO)", "transformer_pcbworld", C.DRAFT_BLUE,   "o"),
    ("Jumanji A2C",    "jumanji",               C.DRAFT_ORANGE, "s"),
    ("SABLE",          "sable",                  C.DRAFT_GREEN,  "^"),
]
METRIC = "clean_pass"
# logit y-axis spreads the near-0 / near-1 region (the draft's "log scale" look).
# clean_pass hits exactly 0 and 1, which logit can't render, so clamp to [EPS,1-EPS].
EPS = 3e-3


def main(argv=None) -> None:  # argv unused; uniform signature for draw_figure dispatch
    series: dict[str, list[tuple[int, float]]] = {}
    table_rows: list[list[str]] = []
    for disp, leaf, _c, _m in METHODS:
        pts = []
        for g in GRIDS:
            rel = f"d1/d1_grid{g}/{leaf}"
            if not C.cell_exists(rel):
                table_rows.append([disp, str(g), "--", "(absent/OOM)"])
                continue
            C.warn_if_sparse(rel, ["clean_pass", "final_potential", "artifact_path"])
            r = C.reduce_cell(rel, [METRIC])
            cp = r["grand"][METRIC]
            table_rows.append([disp, str(g), C.fmt_raw(cp, pct=True),
                               f"n_boards={r['n_boards']} n_seeds={r['n_seeds']}"])
            if cp is not None:
                pts.append((g, cp))
        series[disp] = pts

    # CSV + Markdown
    C.write_table("fig6c_d1_cleanpass",
                  header=["method", "grid", "clean_pass@5 (%)", "note"],
                  rows=table_rows, title="Figure 6c -- D1 clean-pass@5 across grid sizes")

    # Plot
    plt = C.setup_mpl()
    fig, ax = plt.subplots(figsize=(5.2, 3.24))  # original 3.6 trimmed 10%
    for disp, leaf, color, marker in METHODS:
        pts = series[disp]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [min(max(p[1], EPS), 1 - EPS) for p in pts]  # clamp for logit
        ax.plot(xs, ys, marker=marker, color=color, label=disp, lw=2, ms=6)
    ax.set_xscale("log")
    ax.set_xticks(GRIDS)
    ax.set_xticklabels([str(g) for g in GRIDS])
    ax.set_xlabel("Grid size", fontweight="bold")
    ax.set_ylabel("CP", fontweight="bold")
    # logit y-axis with draft-style ticks 0.0 / 0.1 / 0.5 / 0.9 / 1.0
    ax.set_yscale("logit")
    ax.set_ylim(EPS, 1 - EPS)
    ax.set_yticks([EPS, 0.1, 0.5, 0.9, 1 - EPS])
    ax.set_yticklabels(["0.0", "0.1", "0.5", "0.9", "1.0"])
    ax.get_yaxis().set_minor_locator(plt.NullLocator())
    C.clean_axes(ax)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    outs = C.savefig(fig, "fig6c_d1_cleanpass")
    print("[fig6c] wrote", *[str(o) for o in outs])
    for row in table_rows:
        print("   ", row)


if __name__ == "__main__":
    main()
