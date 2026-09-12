#!/usr/bin/env python3
"""Figure 8 -- reward-weight sweep (3x3 factorial) on D2.

Marginal means of WL and Via:
  (a) over vp  -> bars at wp in {0, 0.001, 0.002}
  (b) over wp  -> bars at vp in {0, 0.05, 0.1}
WL on the left axis (orange), Via on the right axis (teal), as in the draft.

WL/Via are the plain mean over *all* rollouts in each cell (128 boards x 4
rollouts = 512 routed boards) -- NOT the per-(board,seed) final_potential
winner used elsewhere in the paper. Error bars are the standard error of that
mean (the raw rollout std is ~180 mm of board-size spread and would swamp the
zoomed axes).
Read-only: consumes per_rollout.csv; writes only under paper_outputs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

WP = ["0", "0.001", "0.002"]
VP = ["0", "0.05", "0.1"]
WL_C, VIA_C = "#e8853a", "#1f9e9e"  # orange / teal (match draft)


def _pooled(rows, col):
    """(mean, standard_error) over every rollout's ``col`` value (blanks skipped)."""
    xs = [C.u.parse_metric(r.get(col)) for r in rows]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, 0.0
    m = sum(xs) / len(xs)
    s = C.u.sample_std(xs) or 0.0
    return m, s / (len(xs) ** 0.5)


def cell_wl_via(rel: str):
    """(wl_mean, via_mean, wl_sem, via_sem) averaged over *all* rollouts in a cell.

    Every rollout row counts equally (no final_potential winner selection); the
    error component is the standard error of the mean."""
    if not C.cell_exists(rel):
        return None, None, None, None
    C.warn_if_sparse(rel, ["wirelength_mm", "via_count"])
    rows = C.load_rollouts(rel)
    wl_m, wl_e = _pooled(rows, "wirelength_mm")
    via_m, via_e = _pooled(rows, "via_count")
    return wl_m, via_m, wl_e, via_e


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main(argv=None) -> None:  # argv unused; uniform signature for draw_figure dispatch
    grid: dict[tuple[str, str], tuple] = {}
    for w in WP:
        for v in VP:
            grid[(w, v)] = cell_wl_via(f"d2a/reward_w{w}_v{v}")

    # marginal means; error bars = mean of the folded cells' standard errors
    # (sampling noise of each all-rollout cell mean, not the swept-variable effect).
    wl_over_vp = {w: _mean([grid[(w, v)][0] for v in VP]) for w in WP}   # x=wp
    via_over_vp = {w: _mean([grid[(w, v)][1] for v in VP]) for w in WP}
    wl_over_wp = {v: _mean([grid[(w, v)][0] for w in WP]) for v in VP}   # x=vp
    via_over_wp = {v: _mean([grid[(w, v)][1] for w in WP]) for v in VP}
    wl_over_vp_e = {w: (_mean([grid[(w, v)][2] for v in VP]) or 0.0) for w in WP}
    via_over_vp_e = {w: (_mean([grid[(w, v)][3] for v in VP]) or 0.0) for w in WP}
    wl_over_wp_e = {v: (_mean([grid[(w, v)][2] for w in WP]) or 0.0) for v in VP}
    via_over_wp_e = {v: (_mean([grid[(w, v)][3] for w in WP]) or 0.0) for v in VP}

    # ---- CSV / Markdown (the raw 3x3 + marginals) ----
    rows = []
    for w in WP:
        for v in VP:
            wl, via = grid[(w, v)][0], grid[(w, v)][1]
            rows.append([f"wp={w}", f"vp={v}", C.fmt_raw(wl), C.fmt_raw(via)])
    for w in WP:
        rows.append([f"marg wp={w} (over vp)", "--",
                     C.fmt_raw(wl_over_vp[w]), C.fmt_raw(via_over_vp[w])])
    for v in VP:
        rows.append([f"marg vp={v} (over wp)", "--",
                     C.fmt_raw(wl_over_wp[v]), C.fmt_raw(via_over_wp[v])])
    C.write_table("fig8_reward_sweep", header=["cell", "_", "WL", "Via"], rows=rows,
                  title="Figure 8 -- reward-weight sweep (WL/Via marginals)")

    # ---- plot ----
    plt = C.setup_mpl()
    # height = original 3.8 trimmed 10% (-> 3.42); suptitle removed.
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 3.42))

    def _zoom(ax, vals, errs):
        # zoom the axis to the data range (don't anchor at 0), like the draft
        finite = [(v, e) for v, e in zip(vals, errs) if v == v]
        if not finite:  # no reward-sweep cells on disk: keep default limits
            return      # (absent-data degrade, like the other figures)
        lo = min(v - e for v, e in finite)
        hi = max(v + e for v, e in finite)
        pad = (hi - lo) * 0.30 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)

    def panel(ax, xs, wl_map, via_map, wl_emap, via_emap, xlabel):
        import numpy as np
        x = np.arange(len(xs)); width = 0.38
        ax2 = ax.twinx()
        wl_vals = [wl_map[k] if wl_map[k] is not None else float("nan") for k in xs]
        via_vals = [via_map[k] if via_map[k] is not None else float("nan") for k in xs]
        wl_err = [wl_emap[k] for k in xs]
        via_err = [via_emap[k] for k in xs]
        b1 = ax.bar(x - width / 2, wl_vals, width, color=WL_C, label="Wirelength",
                    yerr=wl_err, capsize=3, ecolor="#333333")
        b2 = ax2.bar(x + width / 2, via_vals, width, color=VIA_C, label="Via count",
                     yerr=via_err, capsize=3, ecolor="#333333")
        for rect, val, e in zip(b1, wl_vals, wl_err):
            if val == val:
                ax.annotate(f"{val:.1f}", (rect.get_x() + rect.get_width() / 2, val + e),
                            ha="center", va="bottom", fontsize=8)
        for rect, val, e in zip(b2, via_vals, via_err):
            if val == val:
                ax2.annotate(f"{val:.2f}", (rect.get_x() + rect.get_width() / 2, val + e),
                             ha="center", va="bottom", fontsize=8, color=VIA_C)
        _zoom(ax, wl_vals, wl_err)
        _zoom(ax2, via_vals, via_err)
        ax.set_xticks(x); ax.set_xticklabels([f"{xlabel}={k}" for k in xs])
        ax.set_ylabel("Wirelength (mean)", color=WL_C, fontweight="bold")
        ax2.set_ylabel("Via count (mean)", color=VIA_C, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    panel(axa, WP, wl_over_vp, via_over_vp, wl_over_vp_e, via_over_vp_e, "wp")
    axa.set_title("over via penalty", fontweight="bold")
    panel(axb, VP, wl_over_wp, via_over_wp, wl_over_wp_e, via_over_wp_e, "vp")
    axb.set_title("over wirelength penalty", fontweight="bold")
    fig.tight_layout()
    outs = C.savefig(fig, "fig8_reward_sweep")
    print("[fig8] wrote", *[str(o) for o in outs])


if __name__ == "__main__":
    main()
