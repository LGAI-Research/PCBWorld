"""Shared vocabulary for the rollout -> eval -> aggregation pipeline.

Used by:
  - :mod:`eval.rollout.rl` / :mod:`methods.rl_agent.rollout.transformer` /
    :mod:`eval.pipeline` — rollout, post-hoc DRC, aggregation
    (``eval_transformer``, ``eval_kicad_pcb``, ``emit_csv_artifacts``)
  - :mod:`experiments._lib.metrics` extractors (multi-run paper aggregation)

Stdlib-only on purpose: these helpers must be importable without torch,
CUDA, or KiCad shared libraries, so the lightweight CSV-consumer stages
do not pay the heavy import cost of the core.

Three sections, each documented at its banner:

  Section A — Rollout / DRC-scoring CSV schemas + I/O + winner selection
  Section B — DRC-scoring record + KiCad PCB sidecar handling
  Section C — Generic numeric helpers (no rollout-specific knowledge)
"""
from __future__ import annotations

import csv
import math
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from configs.loader.schema import DEFAULTS


# ============================================================================
# Section A — Rollout / DRC-scoring CSV schemas + I/O + winner selection
# ============================================================================
#
# These shared field lists are the cadagent eval wire protocol. Value metrics
# are declared once and reused by per-rollout rows, per-board rows, overall
# aggregation, and TensorBoard/W&B emission. Adding a cadagent metric means
# updating CADAGENT_VALUE_METRIC_FIELDS first.


ROLLOUT_ID_FIELDS: tuple[str, ...] = (
    "board_index", "board_id", "board_path",
    "rollout_idx", "seed", "deterministic",
)


ROLLOUT_STATUS_FIELDS: tuple[str, ...] = (
    "status",                # completed | crashed | skipped_incompatible_board | truncated
    "skip_reason",
    "early_stop_reason",
)


CADAGENT_VALUE_METRIC_FIELDS: tuple[str, ...] = (
    "terminated", "truncated", "crashed", "engine_crash",
    "success",
    "steps", "episode_return",
    "final_potential",
    "initial_potential",   # Phi of the bare board (baseline; NaN on inline path)
    "potential_gain",      # final_potential - initial_potential (NaN if no baseline)
    # routability = pad-group metric, filled by the DRC eval stage only.
    # ratsnest_reduction = env-side signed ratsnest shrinkage, always filled.
    "routability", "ratsnest_reduction",
    "unrouted_count", "initial_unconnected",
    "wirelength_mm", "via_count", "track_count",
    "drc_violations",        # NaN when env-DRC is off
    "early_stop", "early_stop_threshold",
    "finish_no_progress_since_progress",
    "no_geometry_progress_since_progress",
    # Wall-clock routing time for THIS single rollout (reset->terminal, excludes
    # post-hoc DRC + PCB save). Only attributable when n_envs==1 (serial); NaN
    # in parallel waves where many rollouts share the step loop.
    "per_board_rollout_time",
    "eval_time_sec",
    "clean_pass",
    "drv_errors_only_count", "drv_errors_and_promoted_count",
    "track_angle_drv_count", "total_drv_count",
    "initial_unrouted_edges", "unrouted_edges_remaining",
    # LLM-rollout token accounting (NaN/blank for RL + rule-based rollouts).
    # Part of the single merged value-metric vocab so the RL and LLM eval
    # summaries share one schema, one aggregation, one logger projection.
    "system_tokens", "user_tokens", "response_tokens", "call_count",
)


CADAGENT_BOOL_VALUE_METRIC_FIELDS: frozenset[str] = frozenset({
    "terminated", "truncated", "crashed", "engine_crash",
    "success", "early_stop", "clean_pass",
})


ROLLOUT_ARTIFACT_FIELDS: tuple[str, ...] = (
    # Epoch seconds stamped at end-of-rollout BEFORE PCB serialization. Mirrors
    # the LLM-eval convention so downstream `span = max - min` works directly.
    "rollout_finish_time",
    "artifact_path",
    "error",
)


DRC_METADATA_FIELDS: tuple[str, ...] = (
    # ---- DRC-scoring enrichment ----
    # Filled inline during serial Stage 1, or in-place by the parallel Stage 2
    # post-hoc DRC pass. Blank until DRC runs (and stays blank for crashed rows).
    # The shared metric columns above (success, routability, wirelength_mm,
    # via_count, track_count, final_potential) are overwritten with the
    # KiCad-measured value when DRC runs.
    "routed_pcb", "pro_path",
    "eval_status",           # ok | error  (blank when DRC not run)
    "eval_error",
    "reward_config", "check_angle",
)


PER_ROLLOUT_FIELDS: list[str] = [
    *ROLLOUT_ID_FIELDS,
    *ROLLOUT_STATUS_FIELDS,
    *CADAGENT_VALUE_METRIC_FIELDS,
    *ROLLOUT_ARTIFACT_FIELDS,
    *DRC_METADATA_FIELDS,
]


PER_BOARD_FIELDS: list[str] = [
    "board_id", "board_index", "board_path",
    *CADAGENT_VALUE_METRIC_FIELDS,
    "fp_max", "fp_mean", "fp_std",
    "n_terminated", "n_rollouts",
    "selected_rollout_idx",
    "aggregation_mode",
    "drc_mean",
    "unrouted_mean",
    "routability_mean",
    "success_rate",
    "episode_return_mean",
    "episode_length_mean",
    "wirelength_mean",
    "via_count_mean",
    "track_count_mean",
    "selected_by_mode_rollout_idx",
    "selection_mode_table1",
]


PER_BOARD_METRIC_FIELDS: tuple[str, ...] = CADAGENT_VALUE_METRIC_FIELDS


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read every row of ``path`` into a list of dicts. UTF-8 with newline=''."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write ``rows`` to ``path`` as CSV with the given header. Extras are
    silently dropped via ``extrasaction='ignore'``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Winner selection — pick one rollout per board.
# ---------------------------------------------------------------------------
#
# Three modes, each producing a tuple suitable for ``max(...)``:
#
#   "final_potential_finite_first" — held-out heuristic. Prefer rows whose
#       final_potential is finite (i.e. the engine produced a usable phi);
#       fall back to ratsnest_reduction for crashed/truncated rows. Tie-breaks
#       on routability, then episode_return, then -wirelength.
#
#   "final_potential" — pick by phi alone. Simple, table1-compatible.
#       Tie-breaks on routability, -drv_errors_only, -wirelength, -via_count.
#
#   "posthoc_drc_aware" — table1 canonical. Prioritize routability, then
#       fewer DRVs (errors-only, then errors+promoted), then phi, then route
#       compactness. Requires post-hoc DRC-scoring fields.


SELECTION_MODES = (
    "final_potential_finite_first",
    "final_potential",
    "posthoc_drc_aware",
)


def _first_present(row: dict, *keys: str) -> Any:
    """Return the first key whose value is neither None nor ''."""
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


def selection_key(row: dict, mode: str) -> tuple[float, ...]:
    """Return a tuple usable with ``max(..., key=...)`` to pick a winner.

    Accepts both rollout-stage and scoring-stage field naming (e.g.
    ``wirelength_mm`` vs ``wirelength``, ``drv_errors_only`` vs
    ``drv_errors_only_count``) so the same key works on per_rollout rows
    pre- or post- DRC-scoring enrichment.
    """
    fp = safe_float(row.get("final_potential"), -math.inf)
    rout = safe_float(row.get("routability"), -math.inf)
    wl = safe_float(_first_present(row, "wirelength_mm", "wirelength"), math.inf)
    via = safe_float(row.get("via_count"), math.inf)
    drv_only = safe_float(
        _first_present(row, "drv_errors_only", "drv_errors_only_count"), math.inf,
    )
    drv_promoted = safe_float(
        _first_present(row, "drv_errors_and_promoted", "drv_errors_and_promoted_count"),
        math.inf,
    )
    rollout_idx = safe_int(row.get("rollout_idx"), 10**9)

    if mode == "posthoc_drc_aware":
        return (
            rout, -drv_only, -drv_promoted, fp, -wl, -via, -float(rollout_idx),
        )
    if mode == "final_potential":
        return (
            fp, rout, -drv_only, -wl, -via, -float(rollout_idx),
        )
    if mode == "final_potential_finite_first":
        is_finite = 1.0 if math.isfinite(fp) else 0.0
        ratsnest_red = safe_float(row.get("ratsnest_reduction"), -math.inf)
        # When fp is finite, primary signal is fp; otherwise how far the
        # ratsnest got knocked down (signed — a board that grew islands ranks
        # below one that stood still).
        primary = fp if is_finite else ratsnest_red
        episode_return = safe_float(row.get("episode_return"), -math.inf)
        return (is_finite, primary, rout, episode_return, -wl)
    raise ValueError(f"unknown selection mode: {mode!r} (valid: {SELECTION_MODES})")


def select_best_in_board(
    rollouts: Sequence[dict],
    mode: str,
) -> tuple[int, dict | None]:
    """Pick a single winner from one board's rollouts. Returns ``(idx, row)``
    where ``idx`` is the position in ``rollouts`` and ``row`` is the winner.
    ``(-1, None)`` when the input is empty."""
    if not rollouts:
        return -1, None
    idx, row = max(enumerate(rollouts), key=lambda item: selection_key(item[1], mode))
    return int(idx), row


def select_best_per_board(
    rows: Iterable[dict],
    mode: str,
    *,
    board_key_fn: Callable[[dict], Any] | None = None,
) -> list[dict]:
    """Group ``rows`` by board, pick one winner per board with ``selection_key``.

    Each winner gets ``candidate_count`` and ``selection_mode`` annotations
    appended. Output is sorted by stringified board key for determinism.
    """
    if board_key_fn is None:
        def board_key_fn(r: dict) -> Any:  # type: ignore[misc]
            return _first_present(r, "board_index", "board_key", "board_id")
    by_board: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        by_board[board_key_fn(row)].append(row)
    winners: list[dict] = []
    for key in sorted(by_board.keys(), key=lambda k: (str(k))):
        candidates = by_board[key]
        winner = max(candidates, key=lambda r: selection_key(r, mode))
        winners.append({
            **winner,
            "candidate_count": len(candidates),
            "selection_mode": mode,
        })
    return winners


# ============================================================================
# Section B — DRC-scoring record + KiCad PCB sidecar handling
# ============================================================================


# ---------------------------------------------------------------------------
# Runtime (live-env) metric semantics — the single derivation point
# ---------------------------------------------------------------------------


def _float_or_nan(value: object) -> float:
    """``float(value)`` or NaN. Unlike :func:`safe_float`, keeps ±inf."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def success_from(
    *,
    terminated: bool,
    unrouted_count: object = None,
    crashed: bool = False,
) -> bool:
    """Canonical episode success: terminated AND fully routed (unrouted == 0).

    Matches the post-hoc scorer (``eval.metrics.compute_metrics``:
    ``success = u_t == 0``) and the env's ``info["success"]``: the voluntary
    all-nets-closed finish (give-up) terminates with unrouted > 0 and is NOT
    a success. When ``unrouted_count`` is unknown (None/NaN), falls back to
    ``terminated`` alone (producers that emit no routed-state count).
    """
    if crashed:
        return False
    u_t = _float_or_nan(unrouted_count) if unrouted_count is not None else math.nan
    if math.isnan(u_t):
        return bool(terminated)
    return bool(terminated) and u_t == 0.0


def clean_pass_from(success: bool, drv_errors_only: object) -> bool | str:
    """clean_pass = success AND zero errors-only DRVs (promoted excluded).

    Returns blank ``""`` when the errors-only count is unknown (env-DRC off):
    blank cells are skipped by aggregation instead of counting as False.
    """
    if drv_errors_only is None or drv_errors_only == "":
        return ""
    return bool(success) and float(drv_errors_only) == 0.0


def runtime_metrics_from_info(
    info: dict[str, Any],
    *,
    terminated: bool,
    truncated: bool,
    crashed: bool = False,
) -> dict[str, Any]:
    """Derive the canonical runtime metric columns from a live-env ``info``.

    The single place where per-rollout metric semantics live for every
    live-env row producer (RL rollout, LLM plan-only replay, LLM live
    episodes): adding or changing a derived-metric definition here reflects
    consistently across all producers. Definitions match the post-hoc
    canonical scorer (``eval.metrics.compute_metrics``); the DRC-scoring
    pass (Stage 2) later overwrites the measured columns with the
    KiCad-measured values.

    ``info`` follows the ``PCBWorld`` step-info vocabulary
    (``unrouted_count``, ``initial_unconnected``, ``wirelength``,
    ``drc_errors``, ...). Producers without a live info dict (e.g. plan-only
    replay's end-of-replay snapshot) pass a synthesized dict with whatever
    measurements they have; every derived field degrades gracefully
    (NaN / blank) on missing keys.

    ``crashed`` marks producer-detected crashes (worker death, exception);
    ``info["engine_crash"]`` is folded in on top.
    """
    engine_crash = bool(info.get("engine_crash", False))
    crashed = bool(crashed) or engine_crash

    initial_unconnected = _float_or_nan(info.get("initial_unconnected"))
    unrouted_count = _float_or_nan(info.get("unrouted_count"))

    # Success: trust the env's episode-end verdict when present (the env owns
    # the exact unconnected count at terminal time); otherwise derive it.
    if crashed:
        success = False
    elif "success" in info:
        success = bool(info["success"])
    else:
        success = success_from(terminated=terminated, unrouted_count=unrouted_count)

    # Ratsnest reduction: env-computed when present; otherwise the signed
    # ratio. This is the per-step progress proxy — routability itself is a
    # pad-group metric that only the DRC eval stage computes, so it stays NaN
    # here and a run without eval cannot pass off the proxy as routability.
    ratsnest_reduction = _float_or_nan(info.get("ratsnest_reduction"))
    if math.isnan(ratsnest_reduction) and not math.isnan(initial_unconnected) \
            and not math.isnan(unrouted_count):
        ratsnest_reduction = (
            1.0 if initial_unconnected == 0
            else (initial_unconnected - unrouted_count) / initial_unconnected
        )

    return {
        "status": "crashed" if crashed else "completed",
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "crashed": crashed,
        "engine_crash": engine_crash,
        "success": success,
        # Inline clean_pass mirrors the post-hoc definition (fully routed AND
        # zero errors-only DRVs) using the env's episode-end DRC count. Blank
        # (-> skipped in aggregation) when env-DRC was off for this rollout.
        "clean_pass": clean_pass_from(success, info.get("drc_errors")),
        "drv_errors_only_count": _float_or_nan(info.get("drc_errors")),
        "final_potential": _float_or_nan(info.get("final_potential")),
        "initial_potential": _float_or_nan(info.get("initial_potential")),
        "potential_gain": _float_or_nan(info.get("potential_gain")),
        "routability": math.nan,
        "ratsnest_reduction": ratsnest_reduction,
        "unrouted_count": unrouted_count,
        "initial_unconnected": initial_unconnected,
        "wirelength_mm": _float_or_nan(info.get("wirelength")),
        "via_count": _float_or_nan(info.get("via_count")),
        "track_count": _float_or_nan(info.get("track_count")),
        "drc_violations": _float_or_nan(info.get("drc_violations")),
        "early_stop": bool(info.get("early_stop", False)),
        "early_stop_reason": str(info.get("early_stop_reason", "")),
        "early_stop_threshold": info.get("early_stop_threshold", ""),
        "finish_no_progress_since_progress": info.get(
            "finish_no_progress_since_progress", "",
        ),
        "no_geometry_progress_since_progress": info.get(
            "no_geometry_progress_since_progress", "",
        ),
        "error": str(info.get("error", "")),
    }


def flatten_scored_result(
    result: dict | None,
    *,
    reward_config: str = "",
    check_angle: int = DEFAULTS.check_angle,
    error: str = "",
) -> dict[str, Any]:
    """Project ``eval.metrics.compute_metrics``/``evaluate_one``'s nested dict
    into the flat DRC-scoring metric columns of :data:`PER_ROLLOUT_FIELDS`.

    Does NOT populate identity fields (``board_index``/``board_id``/
    ``rollout_idx``/``routed_pcb``/``pro_path``) or ``eval_time_sec``.
    The caller adds those.
    """
    base = {"reward_config": str(reward_config), "check_angle": int(check_angle)}
    if error or result is None or "error" in (result or {}):
        return {
            **base,
            "eval_status": "error",
            "eval_error": (
                error or (result.get("error", "") if result else "missing_result")
            ),
        }
    track_angle = result.get("track_angle_drv") or {}
    extras = result.get("extras") or {}
    out = {
        **base,
        "eval_status": "ok",
        "eval_error": "",
        "success": bool(result.get("success", False)),
        "clean_pass": bool(result.get("clean_pass", False)),
        "routability": float(result.get("routability", math.nan)),
        "track_count": int(result.get("track_count", 0)),
        "via_count": int(result.get("via_count", 0)),
        "wirelength_mm": float(result.get("wirelength_mm", math.nan)),
        "drv_errors_only_count": int(result.get("drv_errors_only_count", 0)),
        "drv_errors_and_promoted_count": int(
            result.get("drv_errors_and_promoted_count", 0),
        ),
        "track_angle_drv_count": int(track_angle.get("count", 0)),
        "total_drv_count": int(result.get("total_drv_count", 0)),
        "final_potential": float(result.get("final_potential", math.nan)),
        "initial_unrouted_edges": int(extras.get("initial_unrouted_edges", 0)),
        "unrouted_edges_remaining": int(extras.get("unrouted_edges_remaining", 0)),
        # Source result may override reward_config; otherwise stick with the
        # caller-supplied value.
        "reward_config": str(result.get("reward_config", reward_config)),
    }
    # Only emit the bare-board baseline when this eval actually computed it
    # (post-hoc path resets the engine; the inline path supplies u_0 and does
    # not, leaving it None). Omitting the keys avoids clobbering the value the
    # rollout already wrote from the env's reset-time potential.
    if result.get("initial_potential") is not None:
        out["initial_potential"] = float(result["initial_potential"])
    if result.get("potential_gain") is not None:
        out["potential_gain"] = float(result["potential_gain"])
    return out


def assemble_rollout_row(
    scored: dict | None = None,
    *,
    identity: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    reward_config: str = "",
    check_angle: int = DEFAULTS.check_angle,
    error: str = "",
) -> dict[str, Any]:
    """The single assembly point for a canonical per-rollout row.

    Layered, later layers winning on key overlap:

      1. :data:`PER_ROLLOUT_FIELDS` skeleton (all columns blank)
      2. ``runtime``  — env-info / episode bookkeeping (steps, episode_return,
         status, env-measured metrics, token counts, ...)
      3. ``scored``   — :func:`flatten_scored_result` of a
         ``compute_metrics``/``evaluate_one`` dict. Applied only when a scored
         result (or scoring ``error``) is present, so DRC columns stay blank
         on unscored rows; KiCad-measured values overwrite the runtime ones.
      4. ``identity`` — board_index/board_id/board_path/rollout_idx/seed/...
         (omit when ``runtime`` already carries the identity columns)

    Shared by every row producer (RL rollout, ``Evaluator.score_boards``, the
    LLM manager/eval paths) so RL and LLM emit one schema from one place.
    """
    row = {f: "" for f in PER_ROLLOUT_FIELDS}
    if runtime:
        row.update(runtime)
    if scored is not None or error:
        row.update(flatten_scored_result(
            scored, reward_config=reward_config,
            check_angle=check_angle, error=error,
        ))
    if identity:
        row.update(identity)
    return row


def resolve_pro_path(
    routed_pcb: Path,
    boards_dir: Path | None = None,
) -> Path | None:
    """Find the ``.kicad_pro`` accompanying a routed ``.kicad_pcb``.

    Order:
      1. Co-located ``<stem>.kicad_pro`` next to the routed PCB.
      2. ``<boards-dir>/<stem>.kicad_pro``.
      3. ``<boards-dir>/<source_stem>.kicad_pro`` where ``source_stem`` is the
         routed file's stem with a trailing ``_rollout_<idx>`` stripped — the
         convention written by the rollout entrypoint
         (``board_00000_rollout_3`` -> ``board_00000``).
      4. Flat ``boards/`` layout: ``<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb`` with
         a shared ``<board_id>.kicad_pro`` co-located in the same dir (D3
         rule-based/LLM cells keep one project file per board, not per rollout) —
         match the longest co-located ``.kicad_pro`` whose stem prefixes this one.
    """
    routed_pcb = Path(routed_pcb)
    co = routed_pcb.with_suffix(".kicad_pro")
    if co.is_file():
        return co
    # Rule 4 — uses the routed PCB's own dir, so it works even without boards_dir.
    if re.search(r"_s\d+_r\d+$", routed_pcb.stem):
        prefix_pros = [
            p for p in routed_pcb.parent.glob("*.kicad_pro")
            if routed_pcb.stem.startswith(p.stem + "_")
        ]
        if prefix_pros:
            return max(prefix_pros, key=lambda p: len(p.stem))
    if boards_dir is None:
        return None
    boards_dir = Path(boards_dir)
    direct = boards_dir / f"{routed_pcb.stem}.kicad_pro"
    if direct.is_file():
        return direct
    stem = routed_pcb.stem
    if "_rollout_" in stem:
        source_stem = stem.rsplit("_rollout_", 1)[0]
        candidate = boards_dir / f"{source_stem}.kicad_pro"
        if candidate.is_file():
            return candidate
    return None


def copy_kicad_sidecars(
    output_path: Path | str,
    source_board_path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Copy ``.kicad_pro`` and ``.kicad_prl`` siblings of ``source_board_path``
    next to ``output_path``. Missing source sidecars are skipped silently.

    When ``overwrite=False`` (default), existing destinations are left as-is.
    """
    output_path = Path(output_path)
    source_board_path = Path(source_board_path)
    for suffix in (".kicad_pro", ".kicad_prl"):
        src = source_board_path.with_suffix(suffix)
        dst = output_path.with_suffix(suffix)
        if not src.is_file():
            continue
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)


# ============================================================================
# Section C — Generic numeric helpers (no rollout-specific knowledge)
# ============================================================================


def safe_float(value: object, default: float = math.nan) -> float:
    """Best-effort float coercion. Returns ``default`` on None, parse failure,
    or non-finite input."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: object, default: int = 0) -> int:
    """Best-effort int coercion (via float to tolerate ``'1.0'``)."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def bool_value(value: object) -> bool:
    """Loose bool parse. Treats ``'1'``, ``'true'``, ``'yes'``, ``'y'``
    (case-insensitive) as True; everything else as False."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def mean(values: Iterable[float]) -> float:
    """Mean of the finite values. ``nan`` when empty."""
    finite = [v for v in values if math.isfinite(v)]
    return statistics.fmean(finite) if finite else math.nan


def quantile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated quantile of the finite values. ``nan`` when empty."""
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[lo]
    return finite[lo] + (finite[hi] - finite[lo]) * (pos - lo)


def nanmean_field(rows: Sequence[dict], key: str) -> float:
    """Mean of ``row[key]`` across ``rows``, ignoring rows whose value parses
    as NaN. ``nan`` when no finite values."""
    vals = [safe_float(r.get(key)) for r in rows]
    return mean(vals)


def mean_bool_field(rows: Sequence[dict], key: str) -> float:
    """Mean of ``row[key]`` interpreted as 0/1. Skips rows where ``key`` is
    absent (vs ``False`` which counts as 0)."""
    vals = [float(bool(r[key])) for r in rows if key in r]
    return float(statistics.fmean(vals)) if vals else math.nan


def parse_metric(value: object) -> float | None:
    """Parse a CSV metric cell to float for aggregation.

    ``"True"/"False"`` -> 1.0/0.0 (bool metrics), numbers -> float; blank,
    non-numeric, or non-finite (nan/inf) -> ``None`` (treated as missing, i.e.
    excluded from means/std rather than poisoning them). Shared by the boards
    aggregation so every method parses metrics identically.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    if s in ("True", "true", "TRUE"):
        return 1.0
    if s in ("False", "false", "FALSE"):
        return 0.0
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def sample_std(values: Iterable[float | None]) -> float | None:
    """Sample standard deviation (ddof=1) of the non-None values; ``None`` when
    fewer than two samples (std undefined)."""
    xs = [v for v in values if v is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def parse_cell_artifact(filename: str, cell: str) -> tuple[str, int, int] | None:
    """Parse the cell-artifact filename contract
    ``<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb`` -> ``(board_id, seed, rollout)``,
    or ``None`` if it does not match. Single source for the flat boards/ naming
    used by rollout-row synthesis and per-board aggregation, so the contract is
    defined in exactly one place."""
    m = re.match(rf"^(?P<bid>.+)_{re.escape(cell)}_s(\d+)_r(\d+)\.kicad_pcb$",
                 Path(filename).name)
    return (m["bid"], int(m.group(2)), int(m.group(3))) if m else None
