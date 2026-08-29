"""Per-board and overall aggregation for eval results.

Two aggregation entry points share this module and the low-level primitives in
:mod:`eval.eval_utils` (``parse_metric``, ``mean``, ``sample_std``,
``select_best_in_board``):

  * :func:`aggregate_inline` — in-memory, grouped by ``board_index``, returns
    (per_board, overall) dicts. Used by the training/validation path
    (``eval_transformer`` -> training/RL W&B / checkpoint selection).
  * :func:`aggregate_boards` — post-hoc, reads a cell's ``per_rollout.csv``,
    groups by (board, ckpt) from the boards/ filenames, writes
    ``per_boards_{ckpts,overall,summary}.csv``. The canonical Stage-3 aggregate.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eval import eval_utils as u
from configs.loader.schema import DEFAULTS
from methods._shared.board_loader import BoardSpec


# ============================================================================
# Per-board summary
# ============================================================================


def _aggregate_value_metric(
    rows: list[dict],
    key: str,
) -> float:
    if key in u.CADAGENT_BOOL_VALUE_METRIC_FIELDS:
        vals = [float(u.bool_value(r[key])) for r in rows if key in r and r.get(key) != ""]
        return float(np.mean(vals)) if vals else math.nan
    return u.nanmean_field(rows, key)


def _selected_value_metric(row: dict | None, key: str) -> float:
    if row is None:
        return math.nan
    if key in u.CADAGENT_BOOL_VALUE_METRIC_FIELDS:
        return float(u.bool_value(row.get(key)))
    return u.safe_float(row.get(key))


def _per_board_summary(
    rollouts: list[dict],
    *,
    board: BoardSpec,
    aggregation_mode: str,
    selection_mode: str = DEFAULTS.selection_method,
) -> dict[str, Any]:
    """Build one row of per-board statistics from this board's rollouts."""
    fps = np.array(
        [u.safe_float(r.get("final_potential")) for r in rollouts], dtype=np.float64,
    )
    valid = ~np.isnan(fps)
    if valid.any():
        fp_max = float(fps[valid].max())
        fp_mean = float(fps[valid].mean())
        fp_std = float(fps[valid].std())
    else:
        fp_max = fp_mean = fp_std = math.nan
    n_term = sum(1 for r in rollouts if u.bool_value(r.get("terminated")))

    selected_idx, selected = u.select_best_in_board(rollouts, selection_mode)

    if aggregation_mode == "best":
        value_metrics = {
            key: _selected_value_metric(selected, key)
            for key in u.CADAGENT_VALUE_METRIC_FIELDS
        }
    else:
        value_metrics = {
            key: _aggregate_value_metric(rollouts, key)
            for key in u.CADAGENT_VALUE_METRIC_FIELDS
        }

    row = {field: "" for field in u.PER_BOARD_FIELDS}
    row.update({
        "board_id": board.board_id,
        "board_index": int(board.index),
        "board_path": board.path,
        **value_metrics,
        "fp_max": fp_max, "fp_mean": fp_mean, "fp_std": fp_std,
        "n_terminated": n_term,
        "n_rollouts": len(rollouts),
        "selected_rollout_idx": selected_idx,
        "aggregation_mode": aggregation_mode,
        "drc_mean": value_metrics["drc_violations"],
        "unrouted_mean": value_metrics["unrouted_count"],
        "routability_mean": value_metrics["routability"],
        "success_rate": value_metrics["success"],
        "episode_return_mean": value_metrics["episode_return"],
        "episode_length_mean": value_metrics["steps"],
        "wirelength_mean": value_metrics["wirelength_mm"],
        "via_count_mean": value_metrics["via_count"],
        "track_count_mean": value_metrics["track_count"],
    })
    return row


# ============================================================================
# Overall stats
# ============================================================================


def _nanmean(arr: np.ndarray) -> float:
    return float(arr.mean()) if len(arr) else math.nan


def _overall_from_per_board(per_board: list[dict]) -> dict[str, float]:
    """Held-style overall stats: averages over board summaries."""
    def col(key: str) -> np.ndarray:
        arr = np.array(
            [u.safe_float(b.get(key)) for b in per_board], dtype=np.float64,
        )
        return arr[~np.isnan(arr)]

    fp_means = col("fp_mean")
    fp_maxes = col("fp_max")
    value_overall = {
        key: _nanmean(col(key)) for key in u.CADAGENT_VALUE_METRIC_FIELDS
    }
    # success is the exact integer condition (every net's pads in one group);
    # routability == 1.0 is its float shadow, so count success directly.
    succ = col("success")
    n_passed = int(np.sum(succ >= 1.0 - 1e-9)) if len(succ) else 0
    return {
        **value_overall,
        "n_boards": len(per_board),
        "n_passed": n_passed,
        "fp_mean_of_means": _nanmean(fp_means),
        "fp_mean_of_maxes": _nanmean(fp_maxes),
        "fp_overall_max": float(fp_maxes.max()) if len(fp_maxes) else math.nan,
        "unrouted_count_mean": _nanmean(col("unrouted_mean")),
        "routability_mean": _nanmean(col("routability_mean")),
        "success_rate": _nanmean(col("success_rate")),
        "episode_return_mean": _nanmean(col("episode_return_mean")),
        "episode_length_mean": _nanmean(col("episode_length_mean")),
        "wirelength_mean": _nanmean(col("wirelength_mean")),
        "via_count_mean": _nanmean(col("via_count_mean")),
        "track_count_mean": _nanmean(col("track_count_mean")),
        "drc_mean": _nanmean(col("drc_mean")),
    }


# ============================================================================
# Table1-style selection overlay
# ============================================================================


def _selection_overlay(
    rollouts_by_board: dict[int, list[dict]],
    *,
    selection_mode: str,
) -> tuple[list[dict], dict[str, float]]:
    """For each board pick one rollout under ``selection_mode``, then aggregate.

    Rows already carry the merged DRC-scoring columns (filled inline in serial
    mode or by the post-hoc DRC pass), so no separate join is needed.

    Returns held-style selected_* stats AND table1-style passed-board-only
    stats + time/step statistics in a single overall dict.
    """
    candidates: list[dict] = [
        r for rollouts in rollouts_by_board.values() for r in rollouts
    ]

    winners = u.select_best_per_board(
        candidates, selection_mode, board_key_fn=lambda r: r.get("board_index"),
    )

    def sel_col(key: str) -> list[float]:
        return [u.safe_float(w.get(key)) for w in winners]

    # ---- held-style selected stats (all winners) ----
    overall: dict[str, Any] = {
        "selected_routability_mean": u.mean(sel_col("routability")),
        "selected_success_rate": float(
            np.mean([1.0 if w.get("success") else 0.0 for w in winners])
        ) if winners else math.nan,
        "selected_clean_pass_rate": float(
            np.mean([1.0 if u.bool_value(w.get("clean_pass")) else 0.0 for w in winners])
        ) if winners else math.nan,
        "selected_drv_errors_only_mean": u.mean(sel_col("drv_errors_only_count")),
        "selected_drv_errors_and_promoted_mean": u.mean(
            sel_col("drv_errors_and_promoted_count"),
        ),
        "selected_total_drv_mean": u.mean(sel_col("total_drv_count")),
        "selected_wirelength_mean": u.mean(sel_col("wirelength_mm")),
        "selected_via_count_mean": u.mean(sel_col("via_count")),
        "selected_final_potential_mean": u.mean(sel_col("final_potential")),
        "selection_mode": selection_mode,
    }

    # ---- table1-style: passed-board-only metrics ----
    passed = [w for w in winners if u.bool_value(w.get("success"))]

    def passed_col(key: str) -> list[float]:
        return [u.safe_float(w.get(key)) for w in passed]

    overall["n_selected"] = len(winners)
    overall["n_passed"] = len(passed)
    overall["selected_drv_errors_only_mean_passed"] = u.mean(
        passed_col("drv_errors_only_count"),
    )
    overall["selected_drv_errors_and_promoted_mean_passed"] = u.mean(
        passed_col("drv_errors_and_promoted_count"),
    )
    overall["selected_total_drv_mean_passed"] = u.mean(
        passed_col("total_drv_count"),
    )
    overall["selected_wirelength_mm_mean_passed"] = u.mean(
        passed_col("wirelength_mm"),
    )
    overall["selected_via_count_mean_passed"] = u.mean(passed_col("via_count"))

    # ---- time / step / early-stop stats ----
    times = sel_col("rollout_finish_time")
    steps = sel_col("steps")
    early = [u.safe_float(w.get("early_stop")) for w in winners]
    early_finite = [v for v in early if math.isfinite(v)]

    overall["selected_steps_mean"] = u.mean(steps)
    overall["selected_steps_median"] = u.quantile(steps, 0.50)
    overall["selected_time_mean"] = u.mean(times)
    overall["selected_time_median"] = u.quantile(times, 0.50)
    overall["selected_time_p95"] = u.quantile(times, 0.95)
    overall["selected_early_stop_count"] = (
        int(sum(early_finite)) if early_finite else 0
    )
    overall["selected_early_stop_rate"] = u.mean(early)

    return winners, overall


# ============================================================================
# Top-level aggregate entry point
# ============================================================================


def aggregate_inline(
    per_rollout: list[dict],
    boards: list[BoardSpec],
    *,
    aggregation_mode: str = "mean",
    selection_mode: str | None,
) -> tuple[list[dict], dict[str, Any]]:
    """In-memory per-board aggregation for the training/validation path.

    Called inside ``eval_transformer`` (RL training validation) to turn
    one eval's rollout rows into (per_board, overall) dicts for W&B logging /
    checkpoint selection — grouped by ``board_index``, returned in memory, no
    file I/O. This is distinct from the post-hoc, cell-level, file-writing
    :func:`aggregate_boards` (Stage 3).

    ``per_rollout`` rows carry the merged DRC-scoring columns (when DRC ran),
    so aggregation reads engine + DRC metrics from the same rows.

    Returns ``(per_board, overall)`` where ``per_board`` is a list of
    per-board summary dicts and ``overall`` is a flat dict of scalar metrics.
    """
    by_board: dict[int, list[dict]] = {}
    for row in per_rollout:
        if row.get("status") == "skipped_incompatible_board":
            continue
        try:
            bi = int(row["board_index"])
        except (KeyError, ValueError, TypeError):
            continue
        by_board.setdefault(bi, []).append(row)

    per_board: list[dict] = []
    per_board_selection_mode = selection_mode or "final_potential"
    for b in boards:
        rollouts = by_board.get(int(b.index), [])
        per_board.append(_per_board_summary(
            rollouts,
            board=b,
            aggregation_mode=aggregation_mode,
            selection_mode=per_board_selection_mode,
        ))

    overall = _overall_from_per_board(per_board)
    if selection_mode is not None:
        winners, sel_overall = _selection_overlay(
            by_board, selection_mode=selection_mode,
        )
        winner_by_board = {int(w["board_index"]): w for w in winners}
        for row in per_board:
            w = winner_by_board.get(int(row["board_index"]))
            if w is None:
                continue
            row["selected_by_mode_rollout_idx"] = int(w.get("rollout_idx", -1))
            row["selection_mode_table1"] = selection_mode
        overall.update(sel_overall)
    return per_board, overall


def aggregate_per_board_outputs(
    per_rollout: list[dict],
    boards: list[BoardSpec],
    *,
    selection_mode: str = DEFAULTS.selection_method,
) -> tuple[list[dict], dict[str, Any], list[dict], dict[str, Any]]:
    """Return the two canonical per-board outputs.

    ``avg`` is the direct average of each shared cadagent metric over rollouts
    for that board. ``best`` selects one rollout per board by ``selection_mode``
    and projects the same metric vocabulary.
    """
    per_board_avg, overall_avg = aggregate_inline(
        per_rollout,
        boards,
        aggregation_mode="mean",
        selection_mode=None,
    )
    per_board_best, overall_best = aggregate_inline(
        per_rollout,
        boards,
        aggregation_mode="best",
        selection_mode=selection_mode,
    )
    return per_board_avg, overall_avg, per_board_best, overall_best


# ============================================================================
# Post-hoc cell-level aggregation: per_boards_{ckpts,overall,summary}.csv
# ============================================================================

# Metric vocab reuses the canonical CADAGENT_VALUE_METRIC_FIELDS. Time metrics are
# split out because the "best" view SUMS them (total compute for best@k) while
# everything else is averaged/selected by _per_board_summary.
_TIME_METRICS = ("per_board_rollout_time", "eval_time_sec")
_BOARD_METRICS = tuple(m for m in u.CADAGENT_VALUE_METRIC_FIELDS if m not in _TIME_METRICS)


def _bmean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _bsum(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) if xs else None


def _fmt(v) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else repr(v)
    return v


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Write rows, formatting at the boundary: rows hold raw values (float/None/
    int/str); ``_fmt`` (None/nan -> "", float -> repr) is applied here once."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows({k: _fmt(v) for k, v in r.items()} for r in rows)


def aggregate_boards(cell_dir, selection_mode: str = "final_potential"):
    """Post-hoc, method-agnostic per-board aggregation over a cell's unified
    ``per_rollout.csv``. Writes ``per_boards_ckpts.csv`` (one row per board x
    ckpt), ``per_boards_overall.csv`` (one row per board), and
    ``per_boards_summary.csv`` (one model-level row).

      ckpt   = seed in the boards/ filename ``..._<cell>_s<SS>_r<RR>`` (NOT the
               csv ``seed`` column, which is the rollout RNG seed for RL).
      <m>_avg  = mean over rollouts ; <m>_best = value at the winner rollout
               (``selection_mode`` via :func:`eval.eval_utils.select_best_in_board`)
      overall: <m>_avg/_avg_std over all runs ; <m>_best/_best_std over ckpts
      time:    _avg = mean ; _best = SUM over rollouts (total compute for best@k)
    """
    cell_dir = Path(cell_dir).resolve()
    cell = cell_dir.name
    pr = cell_dir / "per_rollout.csv"
    if not pr.is_file():
        raise FileNotFoundError(f"per_rollout.csv not found under {cell_dir}")
    with open(pr) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty per_rollout.csv under {cell_dir}")
    # Column alias: a per_rollout.csv carrying ``clean_success`` is exposed
    # under the paper name ``clean_pass`` (var/ is read-only, never rewritten).
    for row in rows:
        if "clean_pass" not in row and "clean_success" in row:
            row["clean_pass"] = row["clean_success"]

    times = [t for t in _TIME_METRICS if t in rows[0]]

    boards: dict[str, dict[int, list[dict]]] = {}
    board_index: dict[str, str] = {}
    for row in rows:
        p = u.parse_cell_artifact(row.get("artifact_path", "") or "", cell)
        if p is None:
            continue  # crashed / no-artifact rollout
        bid = row.get("board_id", "")
        boards.setdefault(bid, {}).setdefault(p[1], []).append(row)  # p[1] = seed/ckpt
        board_index.setdefault(bid, row.get("board_index", ""))

    if not boards:
        # Fail loudly: exiting 0 with no per_boards_*.csv written silently
        # truncates the pipeline (the very failure this message names).
        raise SystemExit(
            f"[aggregate_boards] no per-board output: no artifact_path row in "
            f"{pr} matches the unified cell grammar "
            f"'<board_id>_{cell}_s<SS>_r<RR>.kicad_pcb' (files expected under "
            f"{cell_dir / 'boards'}). A pipeline rollout run flattens its "
            f"artifacts into this layout automatically; for hand-built cells, "
            f"rename the files to the grammar above."
        )

    def _bspec(bid: str) -> BoardSpec:
        return BoardSpec(int(board_index[bid] or 0), bid, "")

    # Per (board, ckpt): reuse _per_board_summary for the avg/best metric values
    # (same CADAGENT vocab, bool->rate, and DRC-aware selection as the inline
    # path); only the time avg/SUM is added here.
    ckpt_rows: list[dict] = []
    per_bc: dict[str, dict[int, dict]] = {}
    for bid in sorted(boards):
        bspec = _bspec(bid)
        for s in sorted(boards[bid]):
            rr = boards[bid][s]
            _, winner = u.select_best_in_board(rr, selection_mode)
            winner = winner or rr[0]
            avg_row = _per_board_summary(
                rr, board=bspec, aggregation_mode="mean", selection_mode=selection_mode)
            best_row = _per_board_summary(
                rr, board=bspec, aggregation_mode="best", selection_mode=selection_mode)
            out = {"board_id": bid, "board_index": board_index[bid], "ckpt": s,
                   "n_rollouts": len(rr),
                   "selected_rollout_idx": winner.get("rollout_idx", "")}
            agg: dict[str, tuple] = {}
            for mcol in _BOARD_METRICS:
                a, b = u.parse_metric(avg_row.get(mcol)), u.parse_metric(best_row.get(mcol))
                out[f"{mcol}_avg"] = a
                out[f"{mcol}_best"] = b
                agg[mcol] = (a, b)
            for t in times:
                tvals = [u.parse_metric(r.get(t)) for r in rr]
                tavg, tbest = _bmean(tvals), _bsum(tvals)
                out[f"{t}_avg"] = tavg
                out[f"{t}_best"] = tbest
                agg[t] = (tavg, tbest)
            ckpt_rows.append(out)
            per_bc.setdefault(bid, {})[s] = agg

    overall_rows: list[dict] = []
    for bid in sorted(boards):
        ckpts = sorted(boards[bid])
        all_rows = [r for s in ckpts for r in boards[bid][s]]
        avg_all = _per_board_summary(
            all_rows, board=_bspec(bid), aggregation_mode="mean", selection_mode=selection_mode)
        out = {"board_id": bid, "board_index": board_index[bid],
               "n_ckpts": len(ckpts), "n_runs": len(all_rows)}
        for mcol in _BOARD_METRICS:
            run_vals = [u.parse_metric(r.get(mcol)) for r in all_rows]   # all individual runs
            ckpt_bests = [per_bc[bid][s][mcol][1] for s in ckpts]  # per-ckpt best
            out[f"{mcol}_avg"] = u.parse_metric(avg_all.get(mcol))
            out[f"{mcol}_avg_std"] = u.sample_std(run_vals)
            out[f"{mcol}_best"] = _bmean(ckpt_bests)
            out[f"{mcol}_best_std"] = u.sample_std(ckpt_bests)
        for t in times:
            out[f"{t}_avg"] = _bmean([per_bc[bid][s][t][0] for s in ckpts])
            out[f"{t}_best"] = _bmean([per_bc[bid][s][t][1] for s in ckpts])
        overall_rows.append(out)

    # model-level summary: column-wise mean over boards of per_boards_overall
    summary = {"n_boards": len(overall_rows), "selection_mode": selection_mode}
    for c in [k for k in overall_rows[0] if k not in ("board_id", "board_index")]:
        vals = [u.parse_metric(r.get(c)) for r in overall_rows]
        if any(v is not None for v in vals):
            summary[c] = _bmean(vals)

    _write_rows(cell_dir / "per_boards_ckpts.csv", ckpt_rows)
    _write_rows(cell_dir / "per_boards_overall.csv", overall_rows)
    _write_rows(cell_dir / "per_boards_summary.csv", [summary])
    print(f"[aggregate_boards] {cell}: boards={len(boards)} ckpt_rows={len(ckpt_rows)} "
          f"overall_rows={len(overall_rows)} metrics={len(_BOARD_METRICS)} times={len(times)} "
          f"selection={selection_mode!r}")
    return ckpt_rows, overall_rows, [summary]


def _aggregate_boards_cli() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc per-board aggregation "
                                 "(per_boards_ckpts/overall/summary.csv) for a cell.")
    ap.add_argument("--cell", type=Path, required=True, help="cell dir with per_rollout.csv")
    ap.add_argument("--selection-mode", default="final_potential",
                    choices=("final_potential", "posthoc_drc_aware"))
    args = ap.parse_args()
    aggregate_boards(args.cell, selection_mode=args.selection_mode)


if __name__ == "__main__":
    _aggregate_boards_cli()


def _safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(statistics.fmean(xs)) if xs else float("nan")


def _safe_stdev(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0


def aggregate(per_sample: list[dict]) -> dict:
    """Mean/stdev/sums + DRV type counters across :func:`evaluate_one` results.

    Nested-dict aggregator: operates on the rich ``evaluate_one`` output, not
    the flat canonical rows that :func:`eval.aggregation.aggregate_inline`
    consumes. Used by the rule-based-router summary path.
    """
    ok = [s for s in per_sample if "error" not in s]
    n_total = len(per_sample)
    n_ok = len(ok)
    n_fail = n_total - n_ok

    def col(key: str) -> list[float]:
        return [float(s[key]) for s in ok]

    def angle_col() -> list[float]:
        return [float(s.get("track_angle_drv", {}).get("count", 0)) for s in ok]

    def total_drv_col() -> list[float]:
        out: list[float] = []
        for s in ok:
            if "total_drv_count" in s:
                out.append(float(s["total_drv_count"]))
            else:
                out.append(
                    float(s.get("drv_errors_and_promoted_count", 0))
                    + float(s.get("track_angle_drv", {}).get("count", 0))
                )
        return out

    success_rate = (
        sum(1 for s in ok if s["success"]) / n_ok if n_ok else float("nan")
    )
    clean_pass_rate = (
        sum(1 for s in ok if s.get("clean_pass")) / n_ok
        if n_ok else float("nan")
    )

    type_counter_eo: Counter = Counter()
    type_counter_ep: Counter = Counter()
    for s in ok:
        for r in s["drv_breakdown"]["errors_only_by_type"]:
            key = (r["severity"], r["error_code"], r["error_type"])
            type_counter_eo[key] += r["count"]
        for r in s["drv_breakdown"]["errors_and_promoted_by_type"]:
            key = (r["severity"], r["error_code"], r["error_type"])
            type_counter_ep[key] += r["count"]

    def _flat_counter(c: Counter) -> list[dict]:
        return [
            {"severity": s, "error_code": cd, "error_type": et, "count": n}
            for (s, cd, et), n in c.most_common()
        ]

    angle_counts = angle_col()
    angle_modes = {
        s.get("track_angle_drv", {}).get("mode") for s in ok
    } - {None}

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "success_rate": float(success_rate),
        "clean_pass_rate": float(clean_pass_rate),
        "mean": {
            "routability": _safe_mean(col("routability")),
            "track_count": _safe_mean(col("track_count")),
            "via_count": _safe_mean(col("via_count")),
            "wirelength_mm": _safe_mean(col("wirelength_mm")),
            "drv_errors_only_count": _safe_mean(col("drv_errors_only_count")),
            "drv_errors_and_promoted_count": _safe_mean(col("drv_errors_and_promoted_count")),
            "track_angle_drv_count": _safe_mean(angle_counts),
            "total_drv_count": _safe_mean(total_drv_col()),
            "final_potential": _safe_mean(col("final_potential")),
        },
        "stdev": {
            "routability": _safe_stdev(col("routability")),
            "track_count": _safe_stdev(col("track_count")),
            "via_count": _safe_stdev(col("via_count")),
            "wirelength_mm": _safe_stdev(col("wirelength_mm")),
            "drv_errors_only_count": _safe_stdev(col("drv_errors_only_count")),
            "drv_errors_and_promoted_count": _safe_stdev(col("drv_errors_and_promoted_count")),
            "track_angle_drv_count": _safe_stdev(angle_counts),
            "total_drv_count": _safe_stdev(total_drv_col()),
            "final_potential": _safe_stdev(col("final_potential")),
        },
        "track_angle_drv": {
            "modes_seen": sorted(int(m) for m in angle_modes),
            "samples_with_violations": sum(1 for v in angle_counts if v > 0),
            "total_violations": int(sum(angle_counts)),
        },
        "drv_breakdown": {
            "errors_only_by_type": _flat_counter(type_counter_eo),
            "errors_and_promoted_by_type": _flat_counter(type_counter_ep),
        },
    }
