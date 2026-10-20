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

import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eval import eval_utils as u
from configs.loader.schema import DEFAULTS
from methods._shared.board_loader import BoardSpec, load_boards_from_dir_or_list


# ============================================================================
# Failed attempts — one rule for the inline (training val) and Stage-3 paths
# ============================================================================

# A failed attempt = an attempt that produced no board to score. Stage 3 sees a
# row without a routed artifact in the cell grammar (router crash, nothing
# routed, unconvertible output, timeout, incompatible board); the inline path a
# crashed / skipped-incompatible row; on both, an expected board with no row at
# all is one failed attempt. It stays in the denominator: the pass-type metrics
# score it as a failure and its potential is the bare board's Φ₀ — the board as
# if left untouched (potential_gain 0), a crash's partial routing included —
# while the measurements of a routed board are blanked, so wirelength / vias /
# DRV average only attempts that produced a board. Rollout bookkeeping (time,
# steps, flags, tokens) and the input-board counts are kept. A pass-type metric
# the routed rows never measured (e.g. routability before Stage 2) stays blank
# instead of turning 0.
_FAILED_ATTEMPT_SCORE = {"success": False, "clean_pass": False, "routability": 0.0}
_ROUTED_BOARD_MEASUREMENTS = (
    "final_potential", "potential_gain", "ratsnest_reduction", "unrouted_count",
    "wirelength_mm", "via_count", "track_count", "drc_violations",
    "drv_errors_only_count", "drv_errors_and_promoted_count",
    "track_angle_drv_count", "total_drv_count", "unrouted_edges_remaining",
)
_INLINE_FAILED_STATUSES = ("crashed", "skipped_incompatible_board")
_MISSING_ROW = "missing_row"  # status of the placeholder for a board with no row
_FAILED = "failed_attempt"
# A routed attempt priced as a failure because every routed attempt of its
# selection group failed scoring (``_unscored_as_failed``); counted in
# ``n_eval_error``, not ``n_failed``.
_UNSCORED = "unscored_attempt"


def _failure_score(routed_rows: Iterable[dict]) -> dict:
    """The pass-type failure values for the metrics ``routed_rows`` measured —
    all of them when every routed row failed scoring (nothing measured them, but
    an attempt priced as a failure passes nothing, as in a cell with no routed row;
    Stage 3 stops on such a cell before using it, inline aggregates with it)."""
    rows = list(routed_rows)
    if rows and all(r.get("eval_status") == "error" for r in rows):
        return dict(_FAILED_ATTEMPT_SCORE)
    return {k: v for k, v in _FAILED_ATTEMPT_SCORE.items()
            if any(u.parse_metric(r.get(k)) is not None for r in rows)}


def _failed_attempt(row: dict, score: dict, phi0: float | None) -> dict:
    """``row`` re-scored as a failed attempt; ``phi0`` = the board's Φ₀
    (``None`` leaves the potential blank)."""
    out = dict(row)
    out.update({k: "" for k in (*_ROUTED_BOARD_MEASUREMENTS, *_FAILED_ATTEMPT_SCORE)})
    out.update(score)
    # Φ₀ comes from the caller only: a row's own initial_potential may be priced
    # by another reward config than the cell's scorer.
    out.update(initial_potential="" if phi0 is None else phi0)
    if phi0 is not None:
        out.update(final_potential=phi0, potential_gain=0.0)
    out[_FAILED] = True
    return out


def all_routed_unscored(rows: Iterable[dict]) -> bool:
    """True when ``rows`` has a routed attempt and every routed attempt failed
    scoring (``eval_status=error``). For one selection group: the group Stage 3
    prices as failed attempts and Stage 2 records Φ₀ for; for a whole cell: the
    scorer failure Stage 3 stops on."""
    routed = [r for r in rows if not r.get(_FAILED)]
    return bool(routed) and all(r.get("eval_status") == "error" for r in routed)


def scored_without_phi0(row: dict) -> bool:
    """A scored routed attempt (``eval_status=ok``) without Φ₀: inline scoring of
    an env whose reset ran no DRC (:func:`eval.metrics.compute_metrics_inline`).
    Stage 2 records its board's Φ₀ and Stage 3 fills the row's Φ₀ / gain from it."""
    return (not row.get(_FAILED) and row.get("eval_status") == "ok"
            and u.parse_metric(row.get("initial_potential")) is None)


def _unscored_as_failed(rows: list[dict], score: dict, phi0: float | None) -> list[dict]:
    """``rows`` (one selection group: an inline board, a Stage-3 (board, ckpt))
    with its routed attempts re-scored as failed attempts when every one of
    them failed scoring (``eval_status=error``): the env values they still carry
    are not ranked or averaged; the group is priced like a board left untouched
    (Φ₀ ``phi0``, pass-type metrics 0). Otherwise ``rows`` is returned as is."""
    if not all_routed_unscored(rows):
        return rows
    return [r if r.get(_FAILED) else {**_failed_attempt(r, score, phi0), _UNSCORED: True}
            for r in rows]


def _missing_row(board: BoardSpec) -> dict:
    row = {f: "" for f in u.PER_ROLLOUT_FIELDS}
    row.update(board_id=board.board_id, board_index=board.index, board_path=board.path,
               status=_MISSING_ROW)
    return row


def _select_routed_first(rows: list[dict], mode: str) -> tuple[int, dict | None]:
    """:func:`eval.eval_utils.select_best_in_board` with two preferences, in order
    (index into ``rows``): a failed attempt wins only when the board has no routed
    attempt (routed-first); among the routed attempts, a row whose scoring failed
    (``eval_status=error``) wins only when none was scored (scored-first) — it
    still carries the env's rollout values (Stage 2 did not overwrite them), so
    ranking it against scorer values would mix two scorers. A row with a blank
    ``eval_status`` (DRC not run) counts as scored, so unscored cells are unchanged;
    a board with failed attempts only selects among all of them as before. (The
    aggregations re-price a group with no scored attempt first —
    ``_unscored_as_failed`` — so the all-unscored fallback is a guard.)"""
    routed = [i for i, r in enumerate(rows) if not r.get(_FAILED)]
    pool = ([i for i in routed if rows[i].get("eval_status") != "error"] or routed
            or list(range(len(rows))))
    if len(pool) == len(rows):
        return u.select_best_in_board(rows, mode)
    j, winner = u.select_best_in_board([rows[i] for i in pool], mode)
    return pool[j], winner


def _inline_failed(row: dict) -> bool:
    """The inline rule: a crashed / skipped-incompatible rollout or a placeholder
    for a board without rows. The producer decides the status (a scoring error
    on an existing board stays ``completed`` with ``eval_status=error``)."""
    return row.get("status") in (*_INLINE_FAILED_STATUSES, _MISSING_ROW)


def classify_cell_rows(
    rows: list[dict], cell_dir: Path, source: object, *, caller: str = "aggregate_boards",
) -> tuple[dict[str, dict[int, list[dict]]], dict[str, list[dict]], dict[str, str]]:
    """Stage 3's rule, shared with Stage 2's Φ₀ step: routed attempts by
    ``(board_id, ckpt)`` from the artifact name in the cell grammar, failed
    attempts (no artifact in the grammar) by ``board_id``, and each board's
    ``board_index``. An existing board file outside the grammar, a failed row
    without ``board_id``, or a ``ckpt`` column that contradicts the artifact
    name is an error, not a failure (``source`` names the rows, ``caller`` the
    stage in messages)."""
    cell = cell_dir.name
    routed: dict[str, dict[int, list[dict]]] = {}
    failed: dict[str, list[dict]] = {}
    board_index: dict[str, str] = {}
    for row in rows:
        bid = row.get("board_id", "")
        ap = row.get("artifact_path", "") or ""
        p = u.parse_cell_artifact(ap, cell)
        if p is None:
            if not bid:
                raise SystemExit(f"[{caller}] a row in {source} has no board_id")
            if ap and (Path(ap) if Path(ap).is_absolute() else cell_dir / ap).is_file():
                raise SystemExit(
                    f"[{caller}] {source}: the routed board {ap} exists but its name "
                    f"is not in the cell grammar '<board_id>_{cell}_s<SS>_r<RR>.kicad_pcb'"
                    f" — rename it (a Stage-1 artifacts/ path: eval.pipeline."
                    f"flatten_rollout_artifacts); it is not a failed attempt."
                )
            failed.setdefault(bid, []).append(row)
        else:
            ck = u.parse_metric(row.get("ckpt"))
            if ck is not None and int(ck) != p[1]:
                raise SystemExit(
                    f"[{caller}] {source}: {ap} names ckpt s{p[1]:02d} but the row's "
                    f"ckpt column says {int(ck)} — one ckpt per row; fix the column or the name."
                )
            routed.setdefault(bid, {}).setdefault(p[1], []).append(row)  # p[1] = seed/ckpt
        board_index.setdefault(bid, row.get("board_index", ""))
    return routed, failed, board_index


def _n_failed(rows: Iterable[dict]) -> int:
    return sum(1 for r in rows if r.get(_FAILED) and not r.get(_UNSCORED))


def _n_crashed(rows: Iterable[dict]) -> int:
    return sum(1 for r in rows if r.get(_FAILED) and r.get("status") == "crashed")


def _n_eval_error(rows: Iterable[dict]) -> int:
    """Attempts with a board whose scoring failed (``eval_status=error``): not a
    failure; the metrics it left blank are skipped, while the env's rollout values
    it still carries (e.g. ``success``, ``final_potential``) are read as they are —
    unless every routed attempt of its group failed scoring, which prices them as
    failed attempts (``_unscored_as_failed``)."""
    return sum(1 for r in rows if r.get(_UNSCORED)
               or (not r.get(_FAILED) and r.get("eval_status") == "error"))


def _bool_metric(row: dict | None, key: str) -> float:
    """A bool metric (``PCBWORLD_BOOL_VALUE_METRIC_FIELDS``: the pass-type metrics
    and the rollout flags) as 0/1, ``nan`` when the row did not measure it
    (blank or absent — e.g. before Stage 2, or ``eval_status=error``). One rule
    for the ``avg``, ``best`` and selection views: unmeasured is skipped, not 0."""
    if row is None or row.get(key) in ("", None):
        return math.nan
    return float(u.bool_value(row.get(key)))


# ============================================================================
# Per-board summary
# ============================================================================


def _aggregate_value_metric(
    rows: list[dict],
    key: str,
) -> float:
    if key in u.PCBWORLD_BOOL_VALUE_METRIC_FIELDS:
        return u.mean(_bool_metric(r, key) for r in rows)
    return u.nanmean_field(rows, key)


def _selected_value_metric(row: dict | None, key: str) -> float:
    if row is None:
        return math.nan
    if key in u.PCBWORLD_BOOL_VALUE_METRIC_FIELDS:
        return _bool_metric(row, key)
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

    selected_idx, selected = _select_routed_first(rollouts, selection_mode)

    if aggregation_mode == "best":
        value_metrics = {
            key: _selected_value_metric(selected, key)
            for key in u.PCBWORLD_VALUE_METRIC_FIELDS
        }
    else:
        value_metrics = {
            key: _aggregate_value_metric(rollouts, key)
            for key in u.PCBWORLD_VALUE_METRIC_FIELDS
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
        "api_failed_calls": _api_failed_sum(rollouts),
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
        key: _nanmean(col(key)) for key in u.PCBWORLD_VALUE_METRIC_FIELDS
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
    winners: list[dict] = []
    for key in sorted(rollouts_by_board, key=str):
        rollouts = rollouts_by_board[key]
        if rollouts:
            _, w = _select_routed_first(rollouts, selection_mode)
            winners.append({**w, "candidate_count": len(rollouts),
                            "selection_mode": selection_mode})

    def sel_col(key: str) -> list[float]:
        return [u.safe_float(w.get(key)) for w in winners]

    # ---- held-style selected stats (all winners) ----
    overall: dict[str, Any] = {
        "selected_routability_mean": u.mean(sel_col("routability")),
        # over the winners that measured the metric (see _bool_metric)
        "selected_success_rate": u.mean(_bool_metric(w, "success") for w in winners),
        "selected_clean_pass_rate": u.mean(_bool_metric(w, "clean_pass") for w in winners),
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
# Give-up quality (diagnostic) — one rule for the inline and Stage-3 paths
# ============================================================================


def _giveup_records(rows: Iterable[dict]) -> list[dict]:
    """Rows with a recorded give-up log (:data:`eval.eval_utils.GIVEUP_FIELDS`),
    parsed: board, rollout, events, end-state unrouted nets (``None`` when the
    end state was not read), final Φ."""
    out = []
    for r in rows:
        events = u.parse_json_list(r.get("giveup_events"))
        if events is None:
            continue
        unrouted = u.parse_json_list(r.get("unrouted_nets"))
        out.append({
            "board_id": str(r.get("board_id", r.get("board_index", ""))),
            "board_index": r.get("board_index", ""),
            "rollout_idx": r.get("rollout_idx", ""),
            "seed": r.get("seed", ""),
            "ckpt": r.get("ckpt", ""),
            "events": [[int(x) for x in e] for e in events],
            "unrouted_nets": None if unrouted is None else [int(n) for n in unrouted],
            "final_potential": u.safe_float(r.get("final_potential")),
        })
    return out


def giveup_log(rows: Iterable[dict]) -> list[dict]:
    """Compact per-rollout give-up log (JSON-safe), for result files that keep
    no per-rollout rows (the async-val watchers' ``iter_*.json``) — enough to
    recompute :func:`giveup_quality` across runs, e.g. pooled with a paired cell
    (one watcher run's log: ``board_key="board_index"``, as inline)."""
    return [
        {**rec, "final_potential": (rec["final_potential"]
                                    if math.isfinite(rec["final_potential"]) else None)}
        for rec in _giveup_records(rows)
    ]


def giveup_quality(rows: Iterable[dict], *, board_key: str) -> dict[str, Any]:
    """Give-up quality over one eval's rollout rows (diagnostic, not a headline).

    A give-up = a successful ``net_end`` on a net with ratsnest edges left that
    the rollout left standing (``giveup_events``, :func:`eval.eval_utils.giveup_columns`).
    Every key but ``giveup_unread_rollouts`` is computed over the rollouts with a
    read end state. Keys (``{}`` when no row recorded a give-up log, e.g.
    rule-based or pre-diagnostic rows, so older outputs stay as they were):

    * ``giveup_read_rollouts`` / ``giveup_unread_rollouts`` — recorded rollouts
      with / without a read end state (unread = crashed, or the read failed;
      read 0 = nothing was read, not "no give-up").
    * ``giveup_count`` — give-up events over the read rollouts (= the sum of
      their rows' ``giveup_count``).
    * ``giveup_precision`` — over (rollout, net) give-ups whose board has at
      least one other rollout with a read end state: the share of those nets
      that NO other rollout of the same board finished (net in every other
      rollout's ``unrouted_nets``). 1.0 = it only gave up nets nobody routed.
      NaN when nothing could be judged (one rollout per board, no give-ups).
    * ``giveup_judged`` — the denominator of ``giveup_precision``.
    * ``giveup_fp_delta`` — per board with both kinds of rollouts: mean final Φ
      of rollouts with a give-up minus mean final Φ of rollouts without, then
      the mean over those boards (``giveup_fp_delta_boards``); NaN when none.
      Φ is each row's own ``final_potential`` (the end state the give-ups led
      to), not the failed-attempt Φ₀ the value metrics substitute.

    Boards are keyed by ``board_key`` — ``board_id`` for a Stage-3 cell (every
    rollout of every ckpt of a board counts as "another eval rollout"),
    ``board_index`` inline, where a ``board_id`` (a file stem) need not be unique.
    """
    records = _giveup_records(rows)
    if not records:
        return {}
    read = [rec for rec in records if rec["unrouted_nets"] is not None]
    by_board: dict[str, list[dict]] = {}
    for rec in read:
        by_board.setdefault(str(rec[board_key]), []).append(rec)
    judged = precise = 0
    deltas: list[float] = []
    for rr in by_board.values():
        unrouted = [set(rec["unrouted_nets"]) for rec in rr]
        for i, rec in enumerate(rr):
            others = unrouted[:i] + unrouted[i + 1:]
            if not others:
                continue
            for net in {e[0] for e in rec["events"]}:
                judged += 1
                precise += all(net in un for un in others)
        with_g = [rec["final_potential"] for rec in rr if rec["events"]]
        without = [rec["final_potential"] for rec in rr if not rec["events"]]
        mw, mo = u.mean(with_g), u.mean(without)
        if math.isfinite(mw) and math.isfinite(mo):
            deltas.append(mw - mo)
    return {
        "giveup_read_rollouts": len(read),
        "giveup_unread_rollouts": len(records) - len(read),
        "giveup_count": sum(len(rec["events"]) for rec in read),
        "giveup_judged": judged,
        "giveup_precision": precise / judged if judged else math.nan,
        "giveup_fp_delta": u.mean(deltas),
        "giveup_fp_delta_boards": len(deltas),
    }


def api_failure_summary(rows: Iterable[dict], *, board_key: str) -> dict[str, Any]:
    """Sum the LLM provider-call failures of rollout rows (``api_failed_calls``,
    :data:`eval.eval_utils.LLM_CALL_FIELDS`): steps that fell back to idle after
    the provider call failed. Those steps stay scored (a hidden exclusion would
    flatter the method); this only surfaces them. Returns ``{}`` when no row
    recorded the column (RL / rule-based cells), else ``api_failed_calls`` (the
    sum) and ``api_failed_boards`` (boards, keyed by ``board_key``, with at
    least one). Callers warn with :func:`api_failure_warning`."""
    total = 0
    boards: set[str] = set()
    recorded = False
    for r in rows:
        n = u.parse_metric(r.get("api_failed_calls"))
        if n is None:
            continue
        recorded = True
        if n > 0:
            total += int(n)
            boards.add(str(r.get(board_key, "")))
    if not recorded:
        return {}
    return {"api_failed_calls": total, "api_failed_boards": len(boards)}


def _api_failed_sum(rows: Iterable[dict]) -> int | str:
    """``api_failed_calls`` summed over ``rows``; ``""`` when none recorded it."""
    vals = [v for v in (u.parse_metric(r.get("api_failed_calls")) for r in rows)
            if v is not None]
    return int(sum(vals)) if vals else ""


def api_failure_warning(summary: dict[str, Any]) -> str | None:
    """One warning line for an :func:`api_failure_summary` with failures, else None."""
    n = summary.get("api_failed_calls") or 0
    if not n:
        return None
    return (f"{n} LLM step(s) on {summary['api_failed_boards']} board(s) fell back to "
            f"idle after the provider call failed (api_failed_calls) — they are "
            f"scored as idle steps; see the rollout log's API errors.")


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

    A crashed / skipped-incompatible row and a board in ``boards`` with no row
    are failed attempts (see ``_FAILED_ATTEMPT_SCORE``), priced at the board's
    Φ₀ from the ``initial_potential`` of its scored rows (``eval_status=ok``: the
    scorer's Φ₀), else of any of its rows (the rollout copies the env's reset-time
    Φ₀ onto crash and skip rows) when that is the scorer's Φ₀ too: a run with no
    scored row, or one whose scored rows all record ``env_phi0_matches`` (the env's
    reward and the scoring config define the same Φ); ``overall`` counts them in
    ``n_failed`` / ``n_crashed``, and the boards whose failed (or unscored —
    below) attempts found no Φ₀ in ``n_failed_boards_without_phi0``, with a warning. A row whose scoring
    failed (``eval_status=error``) is not a failure: every view skips the metrics
    it left blank and reads the env values it still carries, except on a board
    with no scored attempt, whose routed attempts are priced as failed attempts
    (``_unscored_as_failed``); both count in ``n_eval_error``, not ``n_failed``.
    A run with routed rows none of which was scored is aggregated the same way,
    with a scorer-failure warning instead of Stage 3's stop: the caller still
    gets its rows back (the LLM rollout exports them for Stage 2/3).

    Returns ``(per_board, overall)`` where ``per_board`` is a list of
    per-board summary dicts and ``overall`` is a flat dict of scalar metrics.
    """
    by_board: dict[int, list[dict]] = {}
    for row in per_rollout:
        try:
            bi = int(row["board_index"])
        except (KeyError, ValueError, TypeError):
            continue
        by_board.setdefault(bi, []).append(row)
    score = _failure_score(
        r for rr in by_board.values() for r in rr if not _inline_failed(r))
    scorer_failed = all_routed_unscored(
        r for b in boards for r in by_board.get(int(b.index), []) if not _inline_failed(r))
    scored = [r for rr in by_board.values() for r in rr if r.get("eval_status") == "ok"]
    env_phi0_scores = all(u.bool_value(r.get("env_phi0_matches")) for r in scored)
    n_without_phi0 = 0
    for b in boards:
        rr = by_board.get(int(b.index)) or [_missing_row(b)]
        ok_first = sorted(rr, key=lambda r: r.get("eval_status") != "ok")
        phi0s = [u.parse_metric(r.get("initial_potential")) for r in ok_first
                 if env_phi0_scores or r.get("eval_status") == "ok"]
        phi0 = next((v for v in phi0s if v is not None), None)
        rr = _unscored_as_failed(
            [_failed_attempt(r, score, phi0) if _inline_failed(r) else r for r in rr],
            score, phi0)
        if phi0 is None and any(r.get(_FAILED) for r in rr):
            n_without_phi0 += 1
        by_board[int(b.index)] = rr
    if n_without_phi0:
        print(
            f"[aggregate_inline] WARNING: {n_without_phi0} board(s) with a failed or unscored attempt "
            f"have no Φ₀ (none of their rows carries an initial_potential — in a scored run whose "
            f"env defines another Φ than the scoring config, none of their scored rows) — their failed "
            f"attempts leave final_potential blank, so the potential means skip them.",
            flush=True,
        )

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
    board_rows = [r for b in boards for r in by_board[int(b.index)]]
    overall["n_failed"] = _n_failed(board_rows)
    overall["n_crashed"] = _n_crashed(board_rows)
    overall["n_eval_error"] = _n_eval_error(board_rows)
    if overall["n_eval_error"]:
        print(f"[aggregate_inline] WARNING: {overall['n_eval_error']} rollout(s) have "
              f"eval_status=error — their blank metrics are skipped, not counted as 0, and the "
              f"env rollout values they still carry are not rescored; a board with no "
              f"scored attempt is priced as failed attempts.",
              flush=True)
    if scorer_failed:
        # Not a stop: inline callers return the rows with this summary (an LLM run
        # exports them for Stage 2/3), and Stage 3 refuses such a cell.
        print("[aggregate_inline] WARNING: no routed attempt was scored (all eval_status=error) "
              "— a scorer failure: this overall is not a result.", flush=True)
    overall["n_failed_boards_without_phi0"] = n_without_phi0
    overall.update(giveup_quality(per_rollout, board_key="board_index"))
    overall.update(api_failure_summary(board_rows, board_key="board_index"))
    if selection_mode is not None:
        winners, sel_overall = _selection_overlay(
            by_board, selection_mode=selection_mode,
        )
        winner_by_board = {int(w["board_index"]): w for w in winners}
        for row in per_board:
            w = winner_by_board.get(int(row["board_index"]))
            if w is None:
                continue
            ridx = w.get("rollout_idx", "")  # "" = a board-without-rows placeholder
            row["selected_by_mode_rollout_idx"] = int(ridx) if ridx != "" else -1
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

    ``avg`` is the direct average of each shared pcbworld metric over rollouts
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

# Metric vocab reuses the canonical PCBWORLD_VALUE_METRIC_FIELDS. Time metrics are
# split out because the "best" view SUMS them (total compute for best@k) while
# everything else is averaged/selected by _per_board_summary.
_TIME_METRICS = ("per_board_rollout_time", "eval_time_sec")
_BOARD_METRICS = tuple(m for m in u.PCBWORLD_VALUE_METRIC_FIELDS if m not in _TIME_METRICS)

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


def _scored_with(routed_rows: list[dict], col: str) -> set[str]:
    """The scoring values (``reward_config`` / ``check_angle``) the routed rows record."""
    return {str(r[col]) for r in routed_rows if r.get(col)}


def _bare_board_potentials(cell_dir: Path, routed_rows: list[dict]) -> dict[str, float]:
    """board_id -> Φ₀ from Stage 2's ``initial_potentials.csv`` (empty when absent).

    Φ₀ must be priced by the reward config that scored the routed rows; a
    mismatch is an error, not a silent mix of two potentials.
    """
    path = cell_dir / u.BARE_BOARD_POTENTIALS_CSV
    if not path.is_file():
        return {}
    ok = [r for r in u.read_csv(path) if r.get("eval_status") == "ok"]
    scored_by = _scored_with(routed_rows, "reward_config")
    priced_by = {r.get("reward_config") for r in ok}
    # The rows whose Φ₀ / gain Stage 3 fills from this file must share its config too.
    filled_by = {r.get("reward_config") for r in routed_rows
                 if scored_without_phi0(r) and r.get("reward_config")}
    if (scored_by and priced_by - scored_by) or (ok and filled_by - priced_by):
        raise SystemExit(
            f"[aggregate_boards] {path} prices Φ₀ with {sorted(priced_by)} but the routed "
            f"rows were scored with {sorted(scored_by)} (those without Φ₀ with "
            f"{sorted(filled_by)}) — rerun Stage 2 (--stages eval) with one reward config."
        )
    return {r["board_id"]: v for r in ok
            if (v := u.parse_metric(r.get("initial_potential"))) is not None}


def _scorer_failure_message(cell_dir: Path, routed_rows: list[dict]) -> str:
    """Stage 3's stop for a cell whose routed rows were all left unscored: the
    first scorer errors, the scoring the rows record (the rerun re-scores with
    the scoring the cell records), and the Stage 2 + 3 rerun."""
    first = list(dict.fromkeys(
        ((r.get("eval_error") or "").strip().splitlines() or ["<no eval_error>"])[0]
        for r in routed_rows))
    def _recorded(col: str) -> str:
        seen = sorted(Counter(str(r.get(col) or "") for r in routed_rows).items())
        return (f"{col}=" + ", ".join(f"{v or '<blank>'} ×{n}" for v, n in seen)
                + (" (the rows disagree)" if len(seen) > 1 else ""))
    recorded = "; ".join(_recorded(col) for col in ("reward_config", "check_angle"))
    return (
        f"[aggregate_boards] {cell_dir.name}: none of the {len(routed_rows)} routed row(s) "
        f"was scored (all eval_status=error) — a scorer failure, not a result; no "
        f"per_boards_*.csv is written. First errors: {first[:3]}. Fix the cause and rerun "
        f"Stage 2 + 3: scripts/eval.py --stages eval,aggregate --output-dir {cell_dir} — it "
        f"re-scores with the scoring the cell records (the rows record: {recorded}) — with the "
        f"run's --boards-dir / --boards-list when it had them."
    )


def aggregate_boards(
    cell_dir,
    selection_mode: str = "final_potential",
    expected_boards: Iterable[BoardSpec] | None = None,
):
    """Post-hoc, method-agnostic per-board aggregation over a cell's unified
    ``per_rollout.csv``. Writes ``per_boards_ckpts.csv`` (one row per board x
    ckpt), ``per_boards_overall.csv`` (one row per board), and
    ``per_boards_summary.csv`` (one model-level row).

      ckpt   = seed in the boards/ filename ``..._<cell>_s<SS>_r<RR>`` (NOT the
               csv ``seed`` column, which is the rollout RNG seed for RL); a
               row without an artifact uses its ``ckpt`` column.
      <m>_avg  = mean over rollouts ; <m>_best = value at the winner rollout
               (``selection_mode`` via :func:`eval.eval_utils.select_best_in_board`)
      overall: <m>_avg/_avg_std over all runs ; <m>_best/_best_std over ckpts
      time:    _avg = mean ; _best = SUM over rollouts (total compute for best@k)

    A row without a routed artifact is a failed attempt (see
    ``_FAILED_ATTEMPT_SCORE``), counted in ``n_failed`` (``n_crashed`` = those
    with status ``crashed``) and priced at the board's Φ₀ from Stage 2's
    ``initial_potentials.csv``; so is a scored row without Φ₀ (inline scoring of an
    env whose reset ran no DRC): its ``initial_potential`` / ``potential_gain`` are
    filled from there, and a row left without ``potential_gain`` is counted in
    ``n_scored_without_potential_gain``, with a warning. A routed row whose scoring failed
    (``eval_status=error``) is not a failure: ``_avg``, ``_best`` and the
    selection skip the metrics it left blank and read the env values it still
    carries, except in a (board, ckpt) with no scored attempt, whose routed
    attempts are priced as failed attempts at Stage 2's Φ₀ (``_unscored_as_failed``;
    ``n_eval_error``, not ``n_failed``, with a warning). The board set is ``expected_boards`` when given
    (a row outside the set is an error), otherwise the boards present in
    ``per_rollout.csv`` (a board with no row is undetectable, so a warning is
    printed). Every board owes one attempt per ckpt the cell ran: a missing
    (board, ckpt) is one failed attempt (status ``missing_row``). A failed row
    with no ``ckpt`` in a multi-ckpt cell enters only the board's ``_avg``
    (``n_unattributed_failed``), with a warning, and its board's missing ckpts
    are not filled. A cell where every attempt failed is aggregated (CP 0); a
    cell with routed rows none of which was scored is a scorer failure and
    stops (``SystemExit``). The previous run's outputs are removed first, so a
    run that stops on an error leaves none behind.
    """
    cell_dir = Path(cell_dir).resolve()
    cell = cell_dir.name
    # A run that stops below must not leave the previous run's outputs looking current.
    for name in ("per_boards_ckpts.csv", "per_boards_overall.csv", "per_boards_summary.csv"):
        (cell_dir / name).unlink(missing_ok=True)
    pr = cell_dir / "per_rollout.csv"
    if not pr.is_file():
        raise FileNotFoundError(f"per_rollout.csv not found under {cell_dir}")
    with open(pr) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty per_rollout.csv under {cell_dir}")
    # Legacy column alias: pre-rename per_rollout.csv (var/ is read-only, never
    # rewritten) carries ``clean_success``; expose it under the paper name.
    for row in rows:
        if "clean_pass" not in row and "clean_success" in row:
            row["clean_pass"] = row["clean_success"]

    times = [t for t in _TIME_METRICS if t in rows[0]]

    boards, failed, board_index = classify_cell_rows(rows, cell_dir, pr)

    # No routed board: rows that all failed (no artifact, a failure status) are a
    # result. A row naming an artifact that is not there (layout) or a completed row
    # without one (--save-artifacts off, a failed save) is an error here, since no
    # routed row proves the cell layout (in a mixed cell such a row is a failure).
    if not boards and any(r.get("artifact_path") for rr in failed.values() for r in rr):
        # Fail loudly: exiting 0 with no per_boards_*.csv written silently
        # truncates the pipeline (the very failure this message names).
        raise SystemExit(
            f"[aggregate_boards] no per-board output: no artifact_path row in "
            f"{pr} matches the unified cell grammar "
            f"'<board_id>_{cell}_s<SS>_r<RR>.kicad_pcb' (files expected under "
            f"{cell_dir / 'boards'}). A pipeline rollout run flattens its "
            f"artifacts into this layout automatically; for hand-built cells, "
            f"rename the files to the grammar above. If those attempts really failed, "
            f"clear their artifact_path."
        )
    no_board = [r for rr in failed.values() for r in rr
                if r.get("status") == "completed" and not r.get("artifact_path")]
    if not boards and no_board:
        raise SystemExit(
            f"[aggregate_boards] no per-board output: {len(no_board)} completed row(s) in {pr} "
            f"saved no board (a run with --save-artifacts off, or a failed save) and no row "
            f"was routed — nothing to aggregate."
        )
    if not boards:
        print(f"[aggregate_boards] WARNING: {cell}: every attempt failed "
              f"({sum(map(len, failed.values()))} rows, none with an artifact).", flush=True)

    routed_rows = [r for per in boards.values() for rr in per.values() for r in rr]
    if expected_boards is not None:
        expected = {b.board_id: b for b in expected_boards}
        unexpected = sorted(set(board_index) - set(expected))
        if unexpected:
            raise SystemExit(
                f"[aggregate_boards] {len(unexpected)} board(s) in {pr} are not in the "
                f"expected board set ({len(expected)} boards), e.g. {unexpected[:3]}"
            )
    if all_routed_unscored(routed_rows):
        # Routed boards exist and the scorer scored none of them: a scorer failure,
        # not a 0% result (a cell of failed attempts only is a result, above).
        raise SystemExit(_scorer_failure_message(cell_dir, routed_rows))
    for col in ("reward_config", "check_angle"):
        if len(seen := _scored_with(routed_rows, col)) > 1:
            # e.g. a Stage 2 re-scoring under another setting interrupted halfway
            raise SystemExit(
                f"[aggregate_boards] the routed rows in {pr} were scored with several {col} "
                f"values {sorted(seen)} — rerun Stage 2 (--stages eval) with one {col}.")
    # Nothing routed: no metric was measured, but a failed attempt passes nothing.
    score = _failure_score(routed_rows) if routed_rows else dict(_FAILED_ATTEMPT_SCORE)
    phi0 = _bare_board_potentials(cell_dir, routed_rows)
    # A scored row without Φ₀ (inline scoring, reset without DRC): Stage 2 priced
    # its input board with the same scorer (checked above).
    for r in routed_rows:
        if scored_without_phi0(r) and (v := phi0.get(r["board_id"])) is not None:
            fp = u.parse_metric(r.get("final_potential"))
            r.update(initial_potential=v, potential_gain="" if fp is None else fp - v)
    n_scored_without_gain = sum(1 for r in routed_rows if r.get("eval_status") == "ok"
                                and u.parse_metric(r.get("potential_gain")) is None)

    # Charge each failed attempt to its ckpt: the row's ``ckpt`` column, or the
    # cell's only ckpt. Without either (a merged multi-ckpt cell from before the
    # column) it is unattributed and enters only the board's pooled _avg.
    routed_ckpts = sorted({s for per in boards.values() for s in per})
    unattributed: dict[str, list[dict]] = {}
    for bid, rr in failed.items():
        for r in rr:
            ck = u.parse_metric(r.get("ckpt"))
            if ck is None and len(routed_ckpts) == 1:
                ck = routed_ckpts[0]
            row = _failed_attempt(r, score, phi0.get(bid))
            if ck is None:
                unattributed.setdefault(bid, []).append(row)
            else:
                boards.setdefault(bid, {}).setdefault(int(ck), []).append(row)
    # A (board, ckpt) whose routed attempts all failed scoring is priced as failed
    # attempts (Φ₀ from Stage 2 — score_bare_boards records these boards too).
    for bid, per in boards.items():
        for ck in per:
            per[ck] = _unscored_as_failed(per[ck], score, phi0.get(bid))
    ckpts_all = sorted({s for per in boards.values() for s in per})
    n_unattributed = sum(map(len, unattributed.values()))
    if n_unattributed:
        # Optimistic = a board with a routed attempt that lacks a ckpt: the fill
        # below skips it, so its _best averages only the ckpts it has.
        n_optimistic = sum(1 for bid in unattributed
                           if any(not r.get(_FAILED) for rr in boards.get(bid, {}).values()
                                  for r in rr)
                           and len(boards[bid]) < len(ckpts_all))
        print(
            f"[aggregate_boards] WARNING: {cell}: {n_unattributed} failed row(s) carry no "
            f"ckpt in a {len(ckpts_all)}-ckpt cell — counted in <m>_avg only, not in "
            f"<m>_best or per_boards_ckpts.csv (n_unattributed_failed); on {n_optimistic} "
            f"board(s) with a routed attempt that lack a ckpt, <m>_best averages only the "
            f"ckpts present (optimistic).",
            flush=True,
        )
    if expected_boards is not None:
        for bid, spec in expected.items():
            if bid not in board_index:       # no row at all: filled below like a missing ckpt
                board_index[bid] = str(spec.index)
                boards[bid] = {}
    else:
        print(
            f"[aggregate_boards] WARNING: {cell}: no expected board set given — the board "
            f"set is the boards in per_rollout.csv, so a board with no row at all is not "
            f"counted as a failure.",
            flush=True,
        )

    def _bspec(bid: str) -> BoardSpec:
        return BoardSpec(int(board_index[bid] or 0), bid, "")

    # A (board, ckpt) with no row (an interrupted or partly merged ckpt, or an expected
    # board with no row at all) failed there; without this the board's _best would
    # average only the ckpts it has. In a cell with no known ckpt a board with no row
    # is one unattributed failure. A board with unattributed failures is left as is:
    # those rows may be exactly its missing ckpts.
    n_missing = 0
    for bid, per in boards.items():
        if bid in unattributed:
            continue
        for s in (ckpts_all or ([None] if not per else [])):
            if s in per:
                continue
            row = _failed_attempt(_missing_row(_bspec(bid)), score, phi0.get(bid))
            if s is None:
                unattributed[bid] = [row]
            else:
                per[s] = [row]
            n_missing += 1
    if n_missing:
        print(f"[aggregate_boards] WARNING: {cell}: {n_missing} (board, ckpt) pair(s) have no "
              f"row — each counted as one failed attempt.", flush=True)

    # Per (board, ckpt): reuse _per_board_summary for the avg/best metric values
    # (same PCBWORLD vocab, bool->rate, and DRC-aware selection as the inline
    # path); only the time avg/SUM is added here.
    ckpt_rows: list[dict] = []
    per_bc: dict[str, dict[int, dict]] = {}
    for bid in sorted(boards):
        bspec = _bspec(bid)
        for s in sorted(boards[bid]):
            rr = boards[bid][s]
            _, winner = _select_routed_first(rr, selection_mode)
            winner = winner or rr[0]
            avg_row = _per_board_summary(
                rr, board=bspec, aggregation_mode="mean", selection_mode=selection_mode)
            best_row = _per_board_summary(
                rr, board=bspec, aggregation_mode="best", selection_mode=selection_mode)
            out = {"board_id": bid, "board_index": board_index[bid], "ckpt": s,
                   "n_rollouts": len(rr), "n_failed": _n_failed(rr),
                   "n_crashed": _n_crashed(rr), "n_eval_error": _n_eval_error(rr),
                   "api_failed_calls": _api_failed_sum(rr),
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
    for bid in sorted(set(boards) | set(unattributed)):
        ckpts = sorted(boards.get(bid, {}))
        loose = unattributed.get(bid, [])
        all_rows = [r for s in ckpts for r in boards[bid][s]] + loose
        avg_all = _per_board_summary(
            all_rows, board=_bspec(bid), aggregation_mode="mean", selection_mode=selection_mode)
        out = {"board_id": bid, "board_index": board_index[bid],
               "n_ckpts": len(ckpts), "n_runs": len(all_rows),
               "n_failed": _n_failed(all_rows), "n_crashed": _n_crashed(all_rows),
               "n_eval_error": _n_eval_error(all_rows),
               "n_unattributed_failed": len(loose),
               "api_failed_calls": _api_failed_sum(all_rows)}
        # Every attempt failed and none names its ckpt: each ckpt's best on this
        # board is a failure, so _best is the failed attempt's value.
        fail_best = (_per_board_summary(loose, board=_bspec(bid), aggregation_mode="best",
                                        selection_mode=selection_mode)
                     if loose and not ckpts else None)
        for mcol in _BOARD_METRICS:
            run_vals = [u.parse_metric(r.get(mcol)) for r in all_rows]   # all individual runs
            ckpt_bests = [per_bc[bid][s][mcol][1] for s in ckpts]  # per-ckpt best
            out[f"{mcol}_avg"] = u.parse_metric(avg_all.get(mcol))
            out[f"{mcol}_avg_std"] = u.sample_std(run_vals)
            out[f"{mcol}_best"] = (_bmean(ckpt_bests) if fail_best is None
                                   else u.parse_metric(fail_best.get(mcol)))
            out[f"{mcol}_best_std"] = u.sample_std(ckpt_bests)
        for t in times:
            out[f"{t}_avg"] = _bmean([per_bc[bid][s][t][0] for s in ckpts])
            out[f"{t}_best"] = _bmean([per_bc[bid][s][t][1] for s in ckpts])
        overall_rows.append(out)

    # model-level summary: column-wise mean over boards of per_boards_overall;
    # the failure counts are summed instead.
    counts = ("n_failed", "n_crashed", "n_eval_error", "n_unattributed_failed")
    summary = {"n_boards": len(overall_rows), "selection_mode": selection_mode}
    for c in [k for k in overall_rows[0]
              if k not in ("board_id", "board_index", *counts)]:
        vals = [u.parse_metric(r.get(c)) for r in overall_rows]
        if any(v is not None for v in vals):
            summary[c] = _bmean(vals)
    for c in counts:
        summary[c] = sum(r[c] for r in overall_rows)
    no_phi0 = sorted({bid for bid, per in boards.items() for rr in per.values() for r in rr
                      if r.get(_FAILED) and bid not in phi0}
                     | {bid for bid in unattributed if bid not in phi0})
    if summary["n_eval_error"]:
        print(f"[aggregate_boards] WARNING: {cell}: {summary['n_eval_error']} routed row(s) "
              f"have eval_status=error — their blank metrics are skipped in _avg/_best, not "
              f"counted as 0, and the env rollout values they still carry enter _avg unrescored; "
              f"a (board, ckpt) with no scored attempt is priced as failed attempts; rerun Stage 2 (--stages eval) to score them.", flush=True)
    summary["n_failed_boards_without_phi0"] = len(no_phi0)
    summary["n_scored_without_potential_gain"] = n_scored_without_gain
    summary["expected_boards"] = "per_rollout_rows" if expected_boards is None else "given"
    summary.update(giveup_quality(rows, board_key="board_id"))
    # Replaces the per-board mean the loop above took of api_failed_calls with the sum.
    summary.update(api_failure_summary(rows, board_key="board_id"))
    if (api_warn := api_failure_warning(summary)):
        print(f"[aggregate_boards] WARNING: {cell}: {api_warn}", flush=True)
    if no_phi0:
        print(
            f"[aggregate_boards] WARNING: {cell}: {len(no_phi0)} board(s) with a failed or unscored "
            f"attempt have no Φ₀ in {u.BARE_BOARD_POTENTIALS_CSV} (e.g. {no_phi0[:3]}) — "
            f"their failed or unscored attempts leave final_potential blank, so the potential means "
            f"skip them. Run Stage 2 (--stages eval) with --boards-list.",
            flush=True,
        )

    if n_scored_without_gain:
        print(
            f"[aggregate_boards] WARNING: {cell}: {n_scored_without_gain} scored row(s) have no "
            f"potential_gain (inline scoring without reset DRC, and no Φ₀ for their board in "
            f"{u.BARE_BOARD_POTENTIALS_CSV}) — potential_gain / initial_potential average only "
            f"the other rows, so they are biased. Run Stage 2 (--stages eval).",
            flush=True,
        )

    _write_rows(cell_dir / "per_boards_ckpts.csv", ckpt_rows)
    _write_rows(cell_dir / "per_boards_overall.csv", overall_rows)
    _write_rows(cell_dir / "per_boards_summary.csv", [summary])
    print(f"[aggregate_boards] {cell}: boards={len(overall_rows)} ckpt_rows={len(ckpt_rows)} "
          f"overall_rows={len(overall_rows)} metrics={len(_BOARD_METRICS)} times={len(times)} "
          f"selection={selection_mode!r} failed={summary['n_failed']} "
          f"crashed={summary['n_crashed']}")
    return ckpt_rows, overall_rows, [summary]


def _aggregate_boards_cli() -> None:
    from configs.loader.cli import StrictArgumentParser
    ap = StrictArgumentParser(description="Post-hoc per-board aggregation "
                              "(per_boards_ckpts/overall/summary.csv) for a cell.")
    ap.add_argument("--cell", type=Path, required=True, help="cell dir with per_rollout.csv")
    ap.add_argument("--selection-mode", default="final_potential",
                    choices=("final_potential", "posthoc_drc_aware"))
    ap.add_argument("--boards-list", type=Path, default=None,
                    help="the cell's board list = the expected board set (a board "
                         "with no row counts as a failure)")
    args = ap.parse_args()
    from configs.loader.record import record_tool_run
    record_tool_run(ap, args, out_dir=args.cell, name=u.AGGREGATE_RUN_RECORD,
                    inputs=[args.cell, args.boards_list])
    expected = (load_boards_from_dir_or_list(boards_list=args.boards_list)
                if args.boards_list else None)
    aggregate_boards(args.cell, selection_mode=args.selection_mode, expected_boards=expected)


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

    Legacy nested-dict aggregator (operates on the rich ``evaluate_one`` output,
    not the flat canonical rows that :func:`eval.aggregation.aggregate_inline`
    consumes). Used by the rule-based-router summary path. A sample with an
    ``error`` key is a failed board: it counts in ``n_fail``, in the rates'
    denominator and as routability 0 (it passes nothing, as in Stage 3); the
    other means cover the scored boards only (this path has no Φ₀ for a failure).
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
        sum(1 for s in ok if s["success"]) / n_total if n_total else float("nan")
    )
    clean_pass_rate = (
        sum(1 for s in ok if s.get("clean_pass")) / n_total
        if n_total else float("nan")
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
            "routability": _safe_mean(col("routability") + [0.0] * n_fail),
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
            "routability": _safe_stdev(col("routability") + [0.0] * n_fail),
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
