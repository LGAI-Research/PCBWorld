#!/usr/bin/env python3
"""GPT family × eval-level (interactive / plan-only / engine-free) metric comparison.

Reads bench_results_official_kicadpcb/_kicad_gym_summaries/eval_metrics.csv and
plots, for d2a (synth2L) and d3a (PCBench), four metrics each:
  P@5, CP@5, Routability(mean), DRV_min@5
as 8 subplots in one wide row (d2a block then d3a block).

Each subplot is a grouped bar chart: x = the three eval levels, bars = GPT
family (gpt-5.4 / -mini / -nano), each methodology a distinct hue from the
rq4_factorial palette (orange / teal) plus a complementary purple.
"""
import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from configs.loader.paths import data_root_path

# Input metrics CSV (env-overridable, mirrors rq4's EXPR_ROOT convention).
# The benchmark results tree is not shipped with the repo: EVAL_METRICS_CSV, or
# the same file under $CADAGENT_DATA_ROOT, or "" (checked before use).
DEFAULT_METRICS_CSV = os.environ.get("EVAL_METRICS_CSV") or data_root_path(
    "KDD_benchmark", "bench_results", "bench_results_260501",
    "bench_results_official_kicadpcb", "_kicad_gym_summaries", "eval_metrics.csv",
)
# Output lives under <overleaf-root>/figs, same as plot_reward_factorial.py.
DEFAULT_OVERLEAF_ROOT = os.environ.get("OVERLEAF_ROOT", "var/results/kdd/paper_outputs")
OUTPUT_STEM = "gpt_levels_t2_t3a"

# GPT family — distinct hue per methodology, reusing the rq4_factorial
# palette (orange / teal) plus a complementary purple for the third model.
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
COLOR = {"gpt-5.4": "#e68632", "gpt-5.4-mini": "#1b9aaa", "gpt-5.4-nano": "#8e6cab"}

# Eval levels → model-key prefix in the CSV.
# display label -> model-name prefix in the (legacy) metrics csv
LEVELS = [("Interactive", ""), ("Plan-only", "apiseq_"), ("Engine-free", "cadgen_")]

# Metric order requested: P@5, CP@5, Routability, DRV_min.
# (col, std_col, label, lower_better, integer_axis, show_err)
METRICS = [
    ("succ@5",           "succ@5_std",        "P@5",                False, False, False),
    ("succ&drv0@5",      "succ&drv0@5_std",   "CP@5",               False, False, False),
    ("routability_best@5", "routability_best@5_std", "Routability (best@5)", False, False, True),
    ("drv_won_mean",     "drv_won_std",       "DRV (won mean) (↓)", True,  True,  True),
]

DATASETS = [("synth2L", "d2a (synth_2L)"), ("PCBench", "d3a (PCBench)")]


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    idx = {(r["model"], r["dataset"]): r for r in rows}
    return idx


def val(idx, ds, level_prefix, model, col):
    r = idx.get((level_prefix + model, ds))
    if r is None:
        return math.nan
    try:
        return float(r[col])
    except (KeyError, ValueError, TypeError):
        return math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overleaf-root", type=Path, default=Path(DEFAULT_OVERLEAF_ROOT))
    parser.add_argument("--metrics-csv", type=Path, default=Path(DEFAULT_METRICS_CSV))
    args = parser.parse_args()

    fig_dir = args.overleaf_root / "figs"
    out_stem = fig_dir / OUTPUT_STEM

    idx = load(args.metrics_csv)
    # Block titles for each group of 4 subplots (one per dataset).
    BLOCK_TITLE = {"synth2L": "Synthetic 2-layer (D2)", "PCBench": "Real board (D3-A)"}
    # Global font bump.
    plt.rcParams.update({"font.size": 15})

    # Layout: each task is its own 2x2 block —
    #   top row    : P@5, CP@5
    #   bottom row : Routability, DRV
    # The two task-blocks sit side by side → overall 2 rows x 4 columns
    # (a thin spacer column separates them).
    # METRICS order is [P@5, CP@5, Routability, DRV]; metric i → (i//2, i%2).
    fig = plt.figure(figsize=(18.0, 8.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.18, 1, 1],
                          wspace=0.32, hspace=0.55, top=0.88, bottom=0.13)
    BLOCK_COLS = [(0, 1), (3, 4)]  # grid columns for task 0 (D2) / task 1 (D3-A)

    level_x = list(range(len(LEVELS)))
    bw = 0.25  # bar width per model within a level group

    ax_by = {}  # (dataset_idx, metric_idx) -> Axes
    for b, (ds_key, _ds_label) in enumerate(DATASETS):
        for i, (col, std_col, mlabel, lower_better, integer_axis, show_err) in enumerate(METRICS):
            gr, gc = i // 2, BLOCK_COLS[b][i % 2]
            ax = fig.add_subplot(gs[gr, gc])
            ax_by[(b, i)] = ax
            for mi, model in enumerate(MODELS):
                offs = (mi - (len(MODELS) - 1) / 2) * bw
                xs = [x + offs for x in level_x]
                ys = [val(idx, ds_key, pref, model, col) for _, pref in LEVELS]
                kw = {}
                if show_err:
                    # std error bars where available (NaN std → 0 for that one).
                    es = [val(idx, ds_key, pref, model, std_col) for _, pref in LEVELS]
                    kw = dict(yerr=[e if e == e else 0.0 for e in es], capsize=3,
                              error_kw=dict(ecolor="#444444", elinewidth=1.0, capthick=1.0))
                ax.bar(xs, ys, width=bw, color=COLOR[model],
                       edgecolor="white", linewidth=0.6, **kw)
            ax.set_xticks(level_x)
            ax.set_xticklabels([lvl for lvl, _ in LEVELS], rotation=20, fontsize=15)
            ax.tick_params(axis="y", labelsize=15)
            ax.set_title(mlabel, fontsize=18, fontweight="bold", pad=8)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            # Rate-like metrics (P@5 / CP@5 / Routability) share a fixed
            # [0, 1.1] range so heights compare directly; DRV (lower-better,
            # unbounded) keeps its autoscaled top with integer ticks.
            if integer_axis:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if lower_better:
                ax.set_ylim(0, ax.get_ylim()[1])
            else:
                ax.set_ylim(0, 1.1)

    handles = [Patch(facecolor=COLOR[m], edgecolor="white", label=m) for m in MODELS]

    # Per task-block: centered title (top) + legend (bottom), horizontally
    # centered on the block's two columns (use the top-row axes' span).
    for b, (ds_key, _ds_label) in enumerate(DATASETS):
        x0 = ax_by[(b, 0)].get_position().x0     # top-left panel of the block
        x1 = ax_by[(b, 1)].get_position().x1     # top-right panel of the block
        xc = (x0 + x1) / 2
        fig.text(xc, 0.945, BLOCK_TITLE[ds_key], ha="center", va="center",
                 fontsize=22, fontweight="bold")
        fig.legend(handles=handles, loc="center", ncol=len(MODELS),
                   frameon=False, fontsize=16, bbox_to_anchor=(xc, 0.035))

    fig_dir.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved -> {png}")
    print(f"saved -> {pdf}")

    # Also echo the underlying numbers as text.
    for ds_key, ds_label in DATASETS:
        print(f"\n=== {ds_label} ===")
        print(f"{'level':10s} {'model':14s} " + " ".join(f"{m[2]:>10}" for m in METRICS))
        for lvl, pref in LEVELS:
            for model in MODELS:
                vals = [val(idx, ds_key, pref, model, m[0]) for m in METRICS]
                print(f"{lvl:10s} {model:14s} " +
                      " ".join(f"{v:>10.3f}" if v == v else f"{'—':>10}" for v in vals))


if __name__ == "__main__":
    main()
