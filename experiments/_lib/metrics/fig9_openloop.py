#!/usr/bin/env python3
"""Figure 9 -- interactive agent vs open-loop baselines.

Interfaces:  Interactive (interactive_*)  |  Plan-only (plan_only_*)  |
             Engine-free (engine_free_*). Existing disk cells keep the legacy
             prefixes (pcbworld_/apiseq_/cadgen_) — common.cell_dir read-aliases them.
Models:      gpt-5.4, gpt-5.4-mini, gpt-5.4-nano.
Metrics:     CP, Potential, Routability, DRV (errors-only), Parse-fail.
Splits:      D2 and D3-A  -> one figure per split (5 panels each).

`Parse-fail` = fraction of generations whose .kicad_pcb could not be
parsed/evaluated (eval_status == 'error' / worker_exception). These rollouts
carry blank metrics and are otherwise dropped from the other panels, so they are
surfaced as their own panel + CSV rows (rate over rollouts; plus a count of
boards whose *every* rollout failed).

best = final_potential winner; the four quality panels are grand means pooled
over (board, seed). DRV uses errors-only. Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPLITS = {"D2": "d2a", "D3-A": "d3/d3a"}
INTERFACES = [("Interactive", "interactive"), ("Plan-only", "plan_only"),
              ("Engine-free", "engine_free")]
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
# Match the reward-sweep figure's palette (orange + teal), extended with a muted
# purple for the third model so the two figures share a visual theme.
OPENLOOP_ORANGE, OPENLOOP_TEAL, OPENLOOP_PURPLE = "#e8853a", "#1f9e9e", "#8c6bb1"
MODEL_C = {"gpt-5.4": OPENLOOP_ORANGE, "gpt-5.4-mini": OPENLOOP_TEAL,
           "gpt-5.4-nano": OPENLOOP_PURPLE}
# (panel title, metric key). "parse_fail" is special -> from parse_fail_stats.
PANELS = [("CP", "clean_pass"), ("Pot.gain", "potential_gain"),
          ("Routability", "routability"), ("DRV (↓)", "drv"),
          ("Parse-fail (↓)", "parse_fail")]


def value(rel: str, metric: str) -> float | None:
    if not C.cell_exists(rel):
        return None
    if metric == "parse_fail":
        return C.parse_fail_stats(rel)["fail_rate"]
    C.warn_if_sparse(rel, ["clean_pass", "final_potential", "routability"])
    return C.reduce_cell(rel, [metric])["grand"][metric]


def main(argv=None) -> None:  # argv unused; uniform signature for draw_figure dispatch
    plt = C.setup_mpl()
    import numpy as np
    table_rows: list[list[str]] = []

    for split_disp, pre in SPLITS.items():
        data = {m: {ifc: {} for ifc, _ in INTERFACES} for _, m in PANELS}
        for ifc_disp, ifc_leaf in INTERFACES:
            for mdl in MODELS:
                rel = f"{pre}/{ifc_leaf}_{mdl}"
                for _panel_disp, metric in PANELS:
                    data[metric][ifc_disp][mdl] = value(rel, metric)
                # quality metric rows
                for panel_disp, metric in PANELS:
                    table_rows.append([split_disp, ifc_disp, mdl, panel_disp,
                                       C.fmt_raw(data[metric][ifc_disp][mdl],
                                                 pct=(metric == "parse_fail"))])
                # explicit parse-fail counts row
                if C.cell_exists(rel):
                    pf = C.parse_fail_stats(rel)
                    table_rows.append([split_disp, ifc_disp, mdl, "Parse-fail (count)",
                                       f"{pf['n_fail_rollouts']}/{pf['n_rollouts']} rollouts; "
                                       f"{pf['n_fail_boards_all']}/{pf['n_boards']} boards all-fail"])

        # height = original 3.4 trimmed 10% (-> 3.06); suptitle removed.
        fig, axes = plt.subplots(1, len(PANELS), figsize=(16.5, 3.06))
        x = np.arange(len(INTERFACES)); width = 0.26
        for ax, (panel_disp, metric) in zip(axes, PANELS):
            for j, mdl in enumerate(MODELS):
                ys = [data[metric][ifc][mdl] if data[metric][ifc].get(mdl) is not None
                      else float("nan") for ifc, _ in INTERFACES]
                bars = ax.bar(x + (j - 1) * width, ys, width, color=MODEL_C[mdl],
                              label=mdl if ax is axes[0] else None)
                # small diamond marker at the top-centre of each (non-NaN) bar
                for rect, y in zip(bars, ys):
                    if y == y:  # skip NaN (missing cell)
                        ax.plot(rect.get_x() + rect.get_width() / 2, y, marker="D",
                                color="#222222", markersize=4, markeredgecolor="white",
                                markeredgewidth=0.5, zorder=5, clip_on=False)
            ax.set_xticks(x)
            ax.set_xticklabels([d for d, _ in INTERFACES], rotation=15, ha="right")
            ax.set_title(panel_disp, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            if metric in ("clean_pass", "routability", "parse_fail"):
                ax.set_ylim(0, 1.05)
        axes[0].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        stem = f"fig9_openloop_{pre.replace('/', '_')}"
        outs = C.savefig(fig, stem)
        print(f"[fig9 {split_disp}] wrote", *[str(o) for o in outs])

    C.write_table("fig9_openloop", header=["Split", "Interface", "Model", "Metric", "Value"],
                  rows=table_rows,
                  title="Figure 9 — open-loop vs interactive "
                        "(CP / Potential / Routability / DRV[errors-only] / Parse-fail)")
    print("[fig9] parse-fail summary:")
    for r in table_rows:
        if r[3] == "Parse-fail (count)" and not r[4].startswith("0/"):
            print("   ", " | ".join(r))


if __name__ == "__main__":
    main()
