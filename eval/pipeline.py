"""Eval CLI orchestration + post-hoc DRC stage.

The pipeline's command-line entry point (:func:`main`) and the post-hoc DRC
stage (:func:`eval_kicad_pcb`). Rollout producers live under ``eval.rollout``;
the sinks (``emit_tensorboard`` / ``emit_csv_artifacts``) live in their homes
(:mod:`methods._shared.logger` / :mod:`eval.evaluator`) and are re-imported here for the
CLI.

  - :func:`eval_kicad_pcb`  — post-hoc DRC on saved ``.kicad_pcb`` artifacts
  - :func:`eval_transformer` — (re-export) rollout a live policy + aggregate
  - :func:`emit_tensorboard` / :func:`emit_csv_artifacts` — (re-export) sinks

CLI usage::

    python -m eval.pipeline --ckpt <path> --boards-dir <path> --seed 42 --n-rollouts 5 --n-envs 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from eval import eval_utils as u
from configs.loader.schema import (
    CHECK_ANGLES,
    DEFAULTS,
    DRC_SWITCHES,
    ROLLOUT_MODES,
    SCHEMA_VERSION,
    SELECTION_METHODS,
)
from methods._shared.board_loader import (
    BoardSpec,
    load_boards_from_dir_or_list,
    load_boards_from_split_json,
)
from eval.metrics import EvalResult
from configs.loader.ckpt import add_strict_ckpt_arg
from configs.loader.expect import check_expect
from configs.loader.record import RUN_RECORD, board_record, write_run_record
from configs.loader.resolve import ConfigError, Entry, Layer, RunConfig, same_value, user_layers
from configs.loader.schema import RLEnvConfig
from methods.rl_agent.rollout.transformer import (
    apply_n_max_slots_override,
    load_policy_from_ckpt,
)
from eval.rollout.rl import eval_env_config, eval_transformer, resolve_eval_env

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# Post-hoc DRC on saved artifacts
# ============================================================================


def _collect_drc_jobs(
    rollout_dir: Path,
    per_rollout: list[dict[str, Any]],
    boards_dir: Path | None,
    already_done: set[str] | None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in per_rollout:
        try:
            board_index = int(row["board_index"])
            rollout_idx = int(row["rollout_idx"])
        except (KeyError, ValueError):
            continue
        artifact_path = row.get("artifact_path", "")
        if not artifact_path:
            continue
        # Unique per-rollout key = the saved board filename. (board_index,
        # rollout_idx) is NOT unique across ckpt seeds (s42..s45 etc.): those
        # rows would collide and all but one seed would be left unscored.
        key = os.path.basename(artifact_path)
        if already_done is not None and key in already_done:
            continue
        if row.get("status", "") != "completed":
            continue
        routed_pcb = Path(artifact_path)
        if not routed_pcb.is_absolute():
            routed_pcb = (rollout_dir / routed_pcb).resolve()
        if not routed_pcb.is_file():
            continue
        pro = u.resolve_pro_path(routed_pcb, str(boards_dir) if boards_dir else None)
        jobs.append({
            "key": key,
            "board_index": board_index,
            "board_id": row.get("board_id", ""),
            "rollout_idx": rollout_idx,
            "routed_pcb": str(routed_pcb),
            "pro_path": str(pro) if pro else "",
        })
    return jobs


def _rows_from_boards_dir(cell_dir: Path) -> list[dict[str, Any]]:
    """Synthesize ``per_rollout`` rows from a flat ``<cell>/boards/`` directory.

    For cells that only emit routed boards and carry no ``per_rollout.csv``
    (LLM / rule-based generators), recover the per-rollout records from the
    uniform filename convention
    ``<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb`` so Stage 2 DRC can score them
    through the same path as policy rollouts. ``board_index`` is assigned by
    sorted unique ``board_id`` (one stable index per board); rows are marked
    ``completed`` (the board exists) with ``artifact_path`` pointing at the file.
    """
    cell = cell_dir.name
    boards = cell_dir / "boards"
    if not boards.is_dir():
        return []
    parsed: list[tuple[str, int, int, Path]] = []
    for fn in sorted(os.listdir(boards)):
        p = u.parse_cell_artifact(fn, cell)
        if p:
            parsed.append((p[0], p[1], p[2], boards / fn))
    if not parsed:
        return []
    idx_of = {bid: i for i, bid in enumerate(sorted({p[0] for p in parsed}))}
    rows: list[dict[str, Any]] = []
    for bid, s, r, pcb in parsed:
        row = {f: "" for f in u.PER_ROLLOUT_FIELDS}
        row.update({
            "board_index": idx_of[bid],
            "board_id": bid,
            "rollout_idx": r,
            "seed": s,
            "ckpt": s,
            "status": "completed",
            "artifact_path": str(pcb.resolve()),
        })
        rows.append(row)
    rows.sort(key=lambda x: (x["board_index"], x["rollout_idx"]))
    return rows


def flatten_rollout_artifacts(cell_dir: Path, *, ckpt_seed: int = 0) -> int:
    """Move Stage-1 ``artifacts/`` rollouts into the unified ``boards/`` layout.

    Stage 1 saves each rollout as ``artifacts/board_<idx>_rollout_<r>.kicad_pcb``,
    while Stage 3 (:func:`eval.aggregation.aggregate_boards`) consumes the cell
    grammar ``boards/<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb`` parsed from the
    ``artifact_path`` column of ``per_rollout.csv``. This step bridges the two:
    each saved rollout board (with its ``.kicad_pro``/``.kicad_prl`` sidecars)
    is renamed into ``boards/`` and its ``artifact_path`` row is rewritten in
    place. ``s<SS>`` is the checkpoint's training seed (one ckpt per rollout
    run); it is also written to the ``ckpt`` column of every row, so a failed
    attempt (no artifact) stays attributable after cells are merged. Rows whose
    filename already matches the grammar are left untouched, so re-runs are
    no-ops. Returns the number of files moved.
    """
    cell_dir = Path(cell_dir).resolve()
    cell = cell_dir.name
    per_rollout_path = cell_dir / "per_rollout.csv"
    if not per_rollout_path.is_file():
        return 0
    rows = u.read_csv(per_rollout_path)
    boards_dir = cell_dir / "boards"
    moved = 0
    stamped = 0
    for row in rows:
        if row.get("ckpt", "") == "":
            row["ckpt"] = int(ckpt_seed)
            stamped += 1
        ap = row.get("artifact_path", "") or ""
        if not ap or u.parse_cell_artifact(ap, cell) is not None:
            continue  # no artifact, or already in the unified grammar
        src = Path(ap)
        if not src.is_absolute():
            src = (cell_dir / src).resolve()
        board_id = row.get("board_id", "")
        try:
            rollout_idx = int(row["rollout_idx"])
        except (KeyError, ValueError, TypeError):
            continue
        if not board_id or not src.is_file():
            continue
        dst = boards_dir / (
            f"{board_id}_{cell}_s{int(ckpt_seed):02d}_r{rollout_idx:02d}.kicad_pcb"
        )
        boards_dir.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        for suffix in (".kicad_pro", ".kicad_prl"):
            sidecar = src.with_suffix(suffix)
            if sidecar.is_file():
                sidecar.replace(dst.with_suffix(suffix))
        if row.get("routed_pcb", "") == ap:  # inline-DRC runs score pre-move
            row["routed_pcb"] = str(dst)
        row["artifact_path"] = str(dst)
        moved += 1
    if moved or stamped:
        u.write_csv(per_rollout_path, rows, u.PER_ROLLOUT_FIELDS)
    if moved:
        try:  # drop the staging dir when the move emptied it
            (cell_dir / "artifacts").rmdir()
        except OSError:
            pass
        print(
            f"[flatten] {moved} artifacts -> {boards_dir} "
            f"(<board_id>_{cell}_s<SS>_r<RR>.kicad_pcb)",
            flush=True,
        )
    return moved


def eval_kicad_pcb(
    rollout_dir: Path,
    *,
    n_workers: int = 1,
    reward_config: str = DEFAULTS.reward_config,
    check_angle: int = DEFAULTS.check_angle,
    boards_dir: Path | None = None,
    output_csv: Path | None = None,
    resume: bool = False,
    start_method: str = "forkserver",
) -> list[dict[str, Any]]:
    """Run post-hoc DRC scoring on saved rollout artifacts.

    No live policy or env pool is needed — works purely on saved PCB files.
    DRC metrics are merged in place into the matching ``per_rollout.csv`` rows
    (keyed by board_index/rollout_idx); the file is rewritten after each result.
    """
    rollout_dir = Path(rollout_dir).resolve()
    per_rollout_path = rollout_dir / "per_rollout.csv"
    if not per_rollout_path.is_file():
        # No rollout CSV (LLM / rule-based cell that only emitted boards/): rebuild
        # the per-rollout records from the flat boards/ filenames so every method
        # is scored through this one Stage 2 path.
        synth = _rows_from_boards_dir(rollout_dir)
        if not synth:
            raise FileNotFoundError(
                f"neither per_rollout.csv nor boards/*.kicad_pcb found under {rollout_dir}"
            )
        u.write_csv(per_rollout_path, synth, u.PER_ROLLOUT_FIELDS)
        print(
            f"[posthoc_drc] no per_rollout.csv; synthesized {len(synth)} rows "
            f"from {rollout_dir / 'boards'}",
            flush=True,
        )
    if output_csv is None:
        output_csv = per_rollout_path

    rows = u.read_csv(per_rollout_path)
    # Key rows by their saved board filename — unique per rollout. Keying on
    # (board_index, rollout_idx) collides across ckpt seeds (s42..s45 etc.),
    # leaving all but one seed's rows unscored.
    rows_by_key: dict[str, dict] = {}
    for row in rows:
        ap = row.get("artifact_path", "") or ""
        if ap:
            rows_by_key[os.path.basename(ap)] = row

    done_keys: set[str] = set()
    if resume:
        done_keys = {
            key for key, row in rows_by_key.items()
            if row.get("eval_status") == "ok"
        }

    jobs = _collect_drc_jobs(
        rollout_dir, rows, boards_dir,
        already_done=done_keys if resume else None,
    )
    print(
        f"[posthoc_drc] rollout_dir={rollout_dir} "
        f"jobs={len(jobs)} n_workers={n_workers} "
        f"resume={resume} already_done={len(done_keys)}",
        flush=True,
    )
    if not jobs:
        return rows

    def _record(job: dict, result: dict, eval_time: float) -> str:
        flat = u.flatten_scored_result(
            result, reward_config=reward_config, check_angle=check_angle,
        )
        row = rows_by_key.get(job["key"])
        if row is None:
            return "missing_row"
        row["routed_pcb"] = job["routed_pcb"]
        row["pro_path"] = job["pro_path"]
        row["eval_time_sec"] = round(eval_time, 4)
        row.update(flat)
        return str(row.get("eval_status", "?"))

    def _flush() -> None:
        u.write_csv(output_csv, rows, u.PER_ROLLOUT_FIELDS)

    t_start = time.monotonic()

    if n_workers <= 1:
        from eval.metrics import evaluate_one

        for k, job in enumerate(jobs):
            t0 = time.monotonic()
            try:
                result = evaluate_one(
                    job["routed_pcb"],
                    job["pro_path"] or None,
                    reward_config_name=reward_config,
                    check_angle=check_angle,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                result = {"error": f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"}
            status = _record(job, result, time.monotonic() - t0)
            print(
                f"[posthoc_drc {k+1}/{len(jobs)}] "
                f"board={job['board_index']:05d} rollout={job['rollout_idx']} "
                f"status={status} "
                f"t={time.monotonic() - t0:.1f}s",
                flush=True,
            )
            _flush()
    else:
        from pcb_world.vec.subproc_pool import SubprocEvalPool

        pool_jobs = [(j["routed_pcb"], j["pro_path"] or None) for j in jobs]
        send_times: dict[int, float] = {k: time.monotonic() for k in range(len(jobs))}

        def _on_result(job_idx: int, result: dict[str, Any]) -> None:
            dt = time.monotonic() - send_times.get(job_idx, time.monotonic())
            job = jobs[job_idx]
            status = _record(job, result, dt)
            print(
                f"[posthoc_drc {job_idx+1}/{len(jobs)}] "
                f"board={job['board_index']:05d} rollout={job['rollout_idx']} "
                f"status={status} t={dt:.1f}s",
                flush=True,
            )
            _flush()

        with SubprocEvalPool(
            n_workers=n_workers,
            reward_config=reward_config,
            check_angle=check_angle,
            start_method=start_method,
        ) as pool:
            pool.evaluate_many(pool_jobs, on_result=_on_result)

    _flush()
    total_dt = time.monotonic() - t_start
    n_ok = sum(1 for r in rows if r.get("eval_status") == "ok")
    print(
        f"[posthoc_drc] done in {total_dt:.1f}s: "
        f"jobs={len(jobs)} ok={n_ok} -> {output_csv}",
        flush=True,
    )
    return rows


_BARE_BOARD_FIELDS = (
    "board_id", "board_path", "pro_path", "initial_potential", "reward_config",
    "eval_status", "error",
)


def _failed_board_inputs(
    rows: list[dict[str, Any]], cell_dir: Path, boards: list[BoardSpec] | None,
) -> dict[str, str]:
    """board_id -> input ``.kicad_pcb`` of every board with a failed attempt
    (Stage 3's rule, :func:`eval.aggregation.classify_cell_rows`, plus an
    expected board without a row, plus a board with a ckpt whose routed attempts
    all failed scoring — Stage 3 prices those as failed attempts too — plus a
    board with a scored attempt that carries no Φ₀: inline scoring of an env whose
    reset ran no DRC, :func:`eval.metrics.compute_metrics_inline`; Stage 3 fills its
    Φ₀ and gain); ``""`` when no path is known."""
    from eval.aggregation import all_routed_unscored, classify_cell_rows, scored_without_phi0

    routed, failed, board_index = classify_cell_rows(
        rows, cell_dir, cell_dir / "per_rollout.csv", caller="score_bare_boards")
    expected = {b.board_id: b.path for b in boards or ()}
    priced = dict(failed)
    for bid, per in routed.items():
        unscored = [rr for rr in per.values() if all_routed_unscored(rr)]
        if unscored and bid not in priced:
            priced[bid] = unscored[0]
        if bid not in priced and any(scored_without_phi0(r) for rr in per.values() for r in rr):
            priced[bid] = [r for rr in per.values() for r in rr]
    need = {
        bid: next((r["board_path"] for r in rr if r.get("board_path")), "")
        or expected.get(bid, "")
        for bid, rr in priced.items()
    }
    for bid, path in expected.items():
        if bid not in board_index:
            need[bid] = path
    return need


def score_bare_boards(
    cell_dir: Path,
    boards: list[BoardSpec] | None,
    *,
    n_workers: int = 1,
    reward_config: str = DEFAULTS.reward_config,
    check_angle: int = DEFAULTS.check_angle,
    boards_dir: Path | None = None,
    start_method: str = "forkserver",
) -> list[dict[str, Any]]:
    """Stage 2 for failed attempts: record the bare-board Φ₀ of every board
    with one (or with a scored attempt that carries no Φ₀) in
    ``<cell>/initial_potentials.csv``.

    A failed attempt left no board, so Stage 3 prices it as the input board
    left untouched: :func:`eval.metrics.evaluate_one` on the input board, whose
    ``initial_potential`` (tracks/vias stripped, DRC included) is the same Φ₀ a
    routed attempt's scoring reports. Boards already recorded under the same
    ``reward_config`` are skipped, so reruns are incremental.
    """
    cell_dir = Path(cell_dir).resolve()
    per_rollout_path = cell_dir / "per_rollout.csv"
    rows = u.read_csv(per_rollout_path) if per_rollout_path.is_file() else []
    need = _failed_board_inputs(rows, cell_dir, boards)
    out_csv = cell_dir / u.BARE_BOARD_POTENTIALS_CSV
    records: dict[str, dict[str, Any]] = {}
    if out_csv.is_file():
        records = {
            r["board_id"]: r for r in u.read_csv(out_csv)
            if r.get("eval_status") == "ok" and r.get("reward_config") == reward_config
        }
    # An input board without its own project file (e.g. PCBench's
    # ``processed_v9_guide_v3_unrouted.kicad_pcb``) takes the one next to the
    # board-list entry (``processed_v9_guide_v3.kicad_pro``).
    listed_pro = {b.board_id: Path(b.path).with_suffix(".kicad_pro")
                  for b in boards or () if b.path}
    jobs: list[dict[str, Any]] = []
    for bid, path in sorted(need.items()):
        if bid in records:
            continue
        rec = {"board_id": bid, "board_path": path, "reward_config": reward_config}
        if not path or not Path(path).is_file():
            records[bid] = {**rec, "eval_status": "no_input_board",
                            "error": f"input board not found: {path!r}"}
            continue
        pro = u.resolve_pro_path(Path(path), str(boards_dir) if boards_dir else None)
        if pro is None and bid in listed_pro and listed_pro[bid].is_file():
            pro = listed_pro[bid]
        jobs.append({**rec, "pro_path": str(pro) if pro else ""})
    print(f"[bare_boards] {cell_dir.name}: boards_with_failed_or_unscored={len(need)} "
          f"jobs={len(jobs)} already_done={len(need) - len(jobs)}", flush=True)

    def _record(job: dict[str, Any], result: dict[str, Any]) -> None:
        phi0 = result.get("initial_potential")
        ok = "error" not in result and phi0 is not None and math.isfinite(phi0)
        records[job["board_id"]] = {
            **job, "initial_potential": phi0 if ok else "",
            "eval_status": "ok" if ok else "error",
            "error": "" if ok else str(result.get("error", "no initial_potential"))[:500],
        }

    if jobs and n_workers <= 1:
        from eval.metrics import evaluate_one

        for job in jobs:
            try:
                result = evaluate_one(job["board_path"], job["pro_path"] or None,
                                      reward_config_name=reward_config,
                                      check_angle=check_angle)
            except Exception as exc:  # noqa: BLE001
                result = {"error": f"{type(exc).__name__}: {exc}"}
            _record(job, result)
    elif jobs:
        from pcb_world.vec.subproc_pool import SubprocEvalPool

        with SubprocEvalPool(n_workers=min(n_workers, len(jobs)), reward_config=reward_config,
                             check_angle=check_angle, start_method=start_method) as pool:
            pool.evaluate_many(
                [(j["board_path"], j["pro_path"] or None) for j in jobs],
                on_result=lambda k, result: _record(jobs[k], result),
            )
    out = [records[b] for b in sorted(records)]
    if out:
        u.write_csv(out_csv, out, _BARE_BOARD_FIELDS)
    bad = [r for r in out if r.get("eval_status") != "ok"]
    if bad:
        print(f"[bare_boards] WARNING: {len(bad)} board(s) have no Φ₀ "
              f"(e.g. {bad[0]['board_id']}: {bad[0]['error']}) — Stage 3 leaves "
              f"their failed attempts' potential blank.", flush=True)
    return out


# ============================================================================
# Emission helpers — moved to the sink homes (logger / evaluator); imported
# here for the CLI main(). emit_tensorboard lives next to log_metrics;
# emit_csv_artifacts next to export_csv/export_json.
# ============================================================================
from methods._shared.logger import emit_tensorboard  # noqa: E402,F401
from eval.evaluator import emit_csv_artifacts  # noqa: E402,F401


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _selection_output_name(selection_method: str) -> str:
    if selection_method == "none":
        return "final_potential"
    return selection_method


# ============================================================================
# CLI orchestration
# ============================================================================


def _resolve_device(spec: str) -> "torch.device":
    import torch
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _parse_args() -> tuple[Any, argparse.Namespace]:
    """The pipeline's parser and its parse (the parser keeps the dests the command line gave)."""
    from configs.loader.cli import StrictArgumentParser
    from configs.loader.resolve import add_run_args
    p = StrictArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", type=Path, default=None)
    add_strict_ckpt_arg(p)
    boards = p.add_mutually_exclusive_group(required=False)
    boards.add_argument("--boards-dir", type=Path)
    boards.add_argument("--boards-list", type=Path)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-rollouts", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=DEFAULTS.n_envs)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default=DEFAULTS.device)
    p.add_argument("--rollout-mode", choices=ROLLOUT_MODES,
                   default=DEFAULTS.rollout_mode)
    p.add_argument("--boards-per-batch", type=int, default=None)
    p.add_argument(
        "--early-stop-finish-no-progress",
        "--early-stop-finish-no-progress-since-progress",
        type=int,
        default=DEFAULTS.early_stop_finish_no_progress,
        dest="early_stop_finish_no_progress",
    )
    p.add_argument(
        "--early-stop-no-geometry-progress",
        type=int,
        default=DEFAULTS.early_stop_no_geometry_progress,
        help=(
            "When >0, truncate a rollout after this many consecutive steps "
            "without changes to track_count, via_count, wirelength, or "
            "ratsnest_reduction."
        ),
    )
    p.add_argument("--override-n-max-slots", type=int, default=DEFAULTS.override_n_max_slots)
    p.add_argument(
        "--env-drc", choices=DRC_SWITCHES, default=DEFAULTS.env_drc,
        help=(
            "Env DRC reward/tokens during rollout. Omitted (default): follow the "
            "ckpt's training emit_drc_tokens. 'on': force on. 'off': strip env DRC "
            "reward/tokens (faster, but mismatches a DRC-trained ckpt)."
        ),
    )
    p.add_argument(
        "--inline-drc", choices=DRC_SWITCHES, default=DEFAULTS.inline_drc,
        help=(
            "on (serial only): score DRC inline on the live engine and skip "
            "Stage 2. off (default): score saved .kicad_pcb post-hoc in Stage 2."
        ),
    )
    p.add_argument(
        "--reward-config", default=DEFAULTS.reward_config,
        help=(
            "DRC scoring config (configs/reward/*.yaml; same namespace as the "
            f"training reward_rule). Default: {DEFAULTS.reward_config!r} "
            "(the shipped training rule). Override to score with a different DRV config; "
            "warns when it differs from the ckpt's training reward_rule. With "
            "--skip-rollout (re-scoring) the cell's recorded rule is used unless "
            "this flag names another (warned; both recorded) or the cell records none."
        ),
    )
    p.add_argument(
        "--check-angle", type=int, choices=CHECK_ANGLES, default=None,
        help=(
            "DRC track-angle check (45/90). Omitted: the env's final "
            "corner_mode (the ckpt's, or --corner-mode). With --skip-rollout "
            "(re-scoring) the cell's recorded angle is used unless this flag names "
            "another (warned; both recorded) or the cell records none."
        ),
    )
    # ---- env-kwargs overrides (None = inherit the ckpt's training args) ----
    p.add_argument(
        "--reward-rule", default=None,
        help=(
            "Override the ckpt's training reward rule. WARNING: penalties that "
            "relied on the training rule's defaults shift to the new rule's "
            "defaults unless also overridden."
        ),
    )
    p.add_argument("--via-penalty", type=float, default=None)
    p.add_argument("--wirelength-penalty", type=float, default=None)
    p.add_argument("--drc-penalty", type=float, default=None)
    p.add_argument("--reward-step-penalty", type=float, default=None)
    p.add_argument(
        "--corner-mode", type=int, choices=CHECK_ANGLES, default=None,
        help="Override routing corner mode (45/90); inherits ckpt when omitted.",
    )
    p.add_argument(
        "--selection-method",
        "--selection-mode",
        choices=SELECTION_METHODS,
        default=DEFAULTS.selection_method,
        dest="selection_method",
        help=("Stage 3 per-board rollout selection. With --skip-rollout (re-scoring) the "
              "cell's recorded selection is used unless this flag names another (warned; "
              "both recorded) or the cell records none."),
    )
    p.add_argument("--dump-per-board-csv", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--save-artifacts", choices=("on", "off"), default=DEFAULTS.save_artifacts)
    p.add_argument("--skip-rollout", action="store_true")
    p.add_argument("--skip-drc", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true")
    p.add_argument(
        "--stages", default=None,
        help="Positive stage selector: comma list of {rollout,eval,aggregate} "
             "(aliases drc=eval, agg=aggregate). Default = all. When given, "
             "overrides the --skip-* flags. e.g. --stages eval,aggregate.",
    )
    p.add_argument("--dry-run", action="store_true")
    add_run_args(p)
    args = p.parse_args()
    # --stages (positive) -> the three skip booleans (overrides --skip-*).
    if args.stages is not None:
        _alias = {"drc": "eval", "agg": "aggregate", "aggregation": "aggregate",
                  "roll": "rollout"}
        sel: set[str] = set()
        for tok in args.stages.split(","):
            t = tok.strip().lower()
            if not t:
                continue
            t = _alias.get(t, t)
            if t not in ("rollout", "eval", "aggregate"):
                p.error(f"--stages: unknown stage {tok!r} (choose rollout,eval,aggregate)")
            sel.add(t)
        if not sel:
            p.error("--stages must name at least one of rollout,eval,aggregate")
        args.skip_rollout = "rollout" not in sel
        args.skip_drc = "eval" not in sel
        args.skip_aggregate = "aggregate" not in sel
    if args.early_stop_finish_no_progress < 0:
        p.error("--early-stop-finish-no-progress must be >= 0")
    if args.early_stop_no_geometry_progress < 0:
        p.error("--early-stop-no-geometry-progress must be >= 0")
    if args.skip_rollout:
        if args.output_dir is None:
            p.error("--skip-rollout requires --output-dir containing per_rollout.csv")
        if args.inline_drc == "on":
            p.error("--inline-drc on is incompatible with --skip-rollout")
        if args.recipe is not None or args.expect is not None:
            p.error("--recipe / --expect apply to the rollout's env config: not with --skip-rollout")
    else:
        if args.ckpt is None:
            p.error("--ckpt is required unless --skip-rollout is set")
        if args.boards_dir is None and args.boards_list is None:
            p.error("--boards-dir or --boards-list is required unless --skip-rollout is set")
        if args.seed is None:
            p.error("--seed is required unless --skip-rollout is set")
        if args.n_rollouts is None:
            p.error("--n-rollouts is required unless --skip-rollout is set")
    return p, args


def _resolve_modes(args: argparse.Namespace) -> tuple[str, bool]:
    """Resolve rollout mode and where DRC *scoring* runs (inline vs Stage 2).

    The env DRC *reward/token* switch (``step_drc``) is resolved later in
    :func:`main`, once the ckpt is loaded, so it can default to the ckpt's
    training ``emit_drc_tokens`` setting.
    """
    mode = args.rollout_mode
    if mode == "serial" and args.n_envs != 1:
        raise SystemExit("--rollout-mode serial requires --n-envs 1")
    inline = args.inline_drc
    if inline == "on" and mode != "serial":
        raise SystemExit("--inline-drc on requires serial mode")
    final_drc = (inline == "on")
    return mode, final_drc


#: The eval CLI's env overrides: dest = the env record key (None = inherit the checkpoint).
ENV_OVERRIDE_DESTS = ("max_steps", "reward_rule", "via_penalty", "wirelength_penalty", "drc_penalty",
                      "reward_step_penalty", "corner_mode")


def env_overrides_vs_ckpt(env_run: Any, ckpt_where: str) -> dict[str, tuple[Any, Any]]:
    """``{key: (ckpt value, eval value)}`` of the env keys the eval CLI or recipe set to another
    value than the checkpoint's — compared with the checkpoint layer's entry (``where`` =
    ``ckpt_where``), not the layer just below (a recipe or the DRC_OFF ``mode-default`` layer);
    a key that restates the checkpoint value is not an override."""
    out = {}
    for k, e in env_run.provenance.items():
        ckpt = _ckpt_entry(e.chain, ckpt_where)
        if (e.source == "cli" or e.source.startswith("recipe:")) and ckpt and not same_value(ckpt["value"], e.value):
            out[k] = (ckpt["value"], e.value)
    return out


#: The scoring settings a cell records and a re-scoring run reuses: Stage 2's DRC scoring (the
#: scored rows carry it too) and Stage 3's per-board selection.
SCORING_KEYS = ("reward_config", "check_angle", "selection_method")
DRC_SCORING_KEYS = SCORING_KEYS[:2]
#: The run record of a re-scoring run (``--skip-rollout``), beside the rollout's ``run_record.yaml``.
RESCORE_RUN_RECORD = "run_record.rescore.yaml"


def _same_scoring(key: str, a: Any, b: Any) -> bool:
    """Two scoring values name the same setting: a rule file path ``…/foo.yaml`` is the rule
    ``foo`` (the name the scored rows record), and ``90`` is ``"90"``."""
    def name(v: Any) -> Any:
        return Path(v).stem if key == "reward_config" and str(v).endswith((".yaml", ".yml")) else v
    return same_value(name(a), name(b))


def scoring_vs_training(training: Mapping[str, Any], scoring_rule: str, *,
                        warn: bool = True) -> dict[str, Any]:
    """The record of how an RL cell's DRC scoring rule relates to the reward its checkpoint
    trained with — ``training`` holds the ckpt's ``reward_rule`` and its per-run overrides
    (:data:`~pcb_world.core.reward_config.REWARD_OVERRIDE_ARGS`) — for ``extra.scoring``. Compared
    by the Φ they define (``phi_diff``: an alias or a path to the same rule matches); a different
    Φ is warned about (``warn``), naming both and the arguments that differ."""
    from pcb_world.core.reward_config import (REWARD_OVERRIDE_ARGS, effective_reward_config,
                                              get_reward_config, phi_diff)

    rule = str(training["reward_rule"])
    try:
        trained = effective_reward_config(rule, training)
    except (KeyError, FileNotFoundError) as exc:
        raise ConfigError(
            f"comparing the DRC scoring rule {scoring_rule!r} with the ckpt's training reward_rule "
            f"{rule!r} failed: the training rule does not resolve ({exc}). Restore its YAML under "
            "configs/reward/ or experiments/kdd/configs/reward/ (`git log --all -- '*/<name>.yaml'` "
            "finds it) and rerun") from exc
    diff = phi_diff(trained, get_reward_config(scoring_rule))
    overrides = {k: training[k] for k in REWARD_OVERRIDE_ARGS if training[k] is not None}
    if diff and warn:
        fields = ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in diff.items())
        print(f"[eval] WARNING: DRC scoring rule {scoring_rule!r} defines another Φ than the ckpt's "
              f"training reward ({rule!r}, overrides {overrides}): {fields} — Φ is not the "
              "potential the policy was trained to raise.", flush=True)
    return {"training_reward_rule": rule, "training_reward_overrides": overrides,
            "matches_training_rule": not diff}


def _ckpt_entry(chain: Sequence[Mapping[str, Any]], ckpt_where: str) -> Mapping[str, Any] | None:
    """The checkpoint layer's entry (``where`` = ``ckpt_where``) of a provenance chain; None
    when the checkpoint did not set the key."""
    return next((c for c in chain if c["where"] == ckpt_where), None)


def cell_training_reward(cell_dir: Path) -> dict[str, Any] | None:
    """The training ``reward_rule`` and reward overrides of the checkpoint an RL cell rolled out:
    the checkpoint layer of each key in its run record (an eval override is not training's).
    None for a cell no checkpoint rolled out (rule-based, LLM) or without a run record."""
    import yaml

    from pcb_world.core.reward_config import REWARD_OVERRIDE_ARGS

    path = cell_dir / RUN_RECORD
    if not path.is_file():
        return None
    record = yaml.safe_load(path.read_text())
    ckpt = (record.get("extra") or {}).get("ckpt")
    if ckpt is None:
        return None
    out = {}
    for key in ("reward_rule", *REWARD_OVERRIDE_ARGS):
        entry = _ckpt_entry(record["config"][key]["chain"] if key in record["config"] else (), ckpt)
        if entry is None:
            raise ConfigError(f"{path} records no checkpoint value of {key} (config.{key}.chain has "
                              f"no entry where {ckpt!r}): the cell's training reward is unknown")
        out[key] = entry["value"]
    return out


def recorded_scoring(cell_dir: Path) -> dict[str, tuple[list[Any], str]]:
    """The scoring settings a cell records: ``{key: (values, where)}`` for each
    :data:`SCORING_KEYS` member it records — from the run records' ``extra.scoring`` (the last
    re-scoring run's, which scored the rows last, else the rollout's), else the DRC scoring
    from the ``per_rollout.csv`` rows (every value the rows carry: rows scored by two runs list
    two) and the selection from the rollout's ``manifest.json`` ``args`` when that
    run aggregated (cells rolled out before the records carried it; a run that skipped Stage 3
    wrote only the option's default). The manifest's DRC scoring is not read: its
    ``check_angle: null`` meant the ckpt's ``corner_mode`` in older pipelines and the final
    env's later, so a cell no scoring has touched names its settings on the command line."""
    import yaml

    found: dict[str, tuple[list[Any], list[str]]] = {}

    def put(key: str, values: list[Any], where: str) -> None:
        values = [int(v) if key == "check_angle" else str(v) for v in values if v not in (None, "")]
        if values:
            have, wheres = found.setdefault(key, ([], []))
            for v in values:
                if not any(_same_scoring(key, v, h) for h in have):
                    have.append(v)
            wheres.append(where)

    for name in (RESCORE_RUN_RECORD, RUN_RECORD):
        path = cell_dir / name
        if path.is_file():
            scoring = (yaml.safe_load(path.read_text()).get("extra") or {}).get("scoring") or {}
            for key in SCORING_KEYS:
                if key in scoring and key not in found:
                    put(key, [scoring[key]], f"{path} extra.scoring.{key}")
    rows_path = cell_dir / "per_rollout.csv"
    missing = [key for key in DRC_SCORING_KEYS if key not in found]
    if missing and rows_path.is_file():
        rows = u.read_csv(rows_path)
        for key in missing:
            put(key, sorted({r[key] for r in rows if r.get(key)}), f"{rows_path} column {key}")
    manifest = cell_dir / "manifest.json"   # the rollout's eval command line, not a checkpoint's
    if "selection_method" not in found and manifest.is_file():
        stage1 = json.loads(manifest.read_text())
        stage1 = stage1["args"] if "args" in stage1 else {}   # older manifests: env kwargs only
        if "selection_method" in stage1 and stage1.get("skip_aggregate") is False:
            put("selection_method", [stage1["selection_method"]], f"{manifest} args.selection_method")
    return {key: (values, "; ".join(wheres)) for key, (values, wheres) in found.items()}


def resolve_rescoring(cell_dir: Path, parser: Any, args: argparse.Namespace,
                      needed: tuple[str, ...]) -> RunConfig:
    """The scoring settings of a re-scoring run of ``cell_dir``: the ones the cell records
    (:func:`recorded_scoring`), carried whole so the run's record keeps them, and the
    ``needed`` ones (of :data:`SCORING_KEYS`: those of the stages this run runs) decided. An
    explicit flag (the ``parser`` parse ``args``, its ``needed`` options) wins, also over another
    recorded value — warned about, naming both, with the cell's value kept in the provenance
    chain under the flag's — and chooses among several recorded values; without the flag the
    cell's single recorded value is used, never the option's default. A needed DRC scoring key
    the cell does not record must be on the command line (no default); an unrecorded selection
    is the command line's, else the option's default (the one a rollout starts from)."""
    recorded = recorded_scoring(cell_dir)
    cli = user_layers(parser, args, keys=needed)[-1]
    flags = {a.dest: a.option_strings[0] for a in parser._actions if a.dest in SCORING_KEYS}
    provenance: dict[str, Entry] = {}
    unrecorded: list[str] = []
    problems: list[str] = []
    for key in SCORING_KEYS:
        flag = flags[key]
        values, where = recorded.get(key, ([], ""))
        shown = values[0] if len(values) == 1 else values
        chain = [{"source": "cell", "value": shown, "where": where}] if values else []
        if key not in needed:
            if len(values) == 1:
                provenance[key] = Entry(shown, "cell", where, tuple(chain))
            continue
        if key in cli.values:
            given = cli.entry(key)
            if values and not any(_same_scoring(key, given["value"], v) for v in values):
                print(f"[eval] WARNING: {given['where']} differs from the {key} the cell records, "
                      f"{shown!r} ({where}): re-scoring under {given['value']!r} changes the cell's "
                      "metrics" + ("" if args.dry_run else f"; {RESCORE_RUN_RECORD} records both"),
                      flush=True)
            chain.append(given)
        elif len(values) > 1:
            problems.append(f"the cell records several {key} values {values} ({where}): pass "
                           f"{flag} with the one to re-score with")
            continue
        elif not values and key == "selection_method":
            chain.append({"source": "defaults", "value": parser.get_default(key), "where": flag})
        elif not values:
            unrecorded.append(key)
            continue
        last = chain[-1]
        provenance[key] = Entry(last["value"], last["source"], last["where"], tuple(chain))
    if unrecorded:
        has_record = (any((cell_dir / n).is_file() for n in (RESCORE_RUN_RECORD, RUN_RECORD, "per_rollout.csv",
                                                              "manifest.json"))
                      or any((cell_dir / "boards").glob("*.kicad_pcb")))
        why = ("its rollout scored nothing, e.g. --skip-drc or a rule-based rollout, and no row was "
               "scored since" if has_record else
               f"it holds no {RUN_RECORD}, {RESCORE_RUN_RECORD}, per_rollout.csv, manifest.json or boards/ — "
               "is it the cell?")
        problems.append(f"{cell_dir} records no {' / '.join(unrecorded)} ({why}): pass "
                        f"{' '.join(flags[k] for k in unrecorded)} — re-scoring does not fall back to a default")
    if problems:
        raise ConfigError("; ".join(problems))
    return RunConfig("eval", provenance)


def _rescore_drc_only(rescoring: RunConfig) -> RunConfig:
    """``rescoring`` with its selection back at the cell's recorded entry (dropped when the cell
    records none): the record of a re-scoring whose Stage 3 has not run yet."""
    provenance = dict(rescoring.provenance)
    entry = provenance.pop("selection_method", None)
    cell = next((c for c in entry.chain if c["source"] == "cell"), None) if entry else None
    if cell is not None:
        provenance["selection_method"] = Entry(cell["value"], "cell", cell["where"], (cell,))
    return RunConfig("eval", provenance)


def _default_output_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.ckpt is None or args.seed is None:
        raise SystemExit("--output-dir is required when --ckpt or --seed is omitted")
    return Path(DEFAULTS.output_root) / f"{ts}_{Path(args.ckpt).stem}_seed{args.seed}"


def _boards_from_per_rollout(per_rollout_rows: list[dict]) -> list[BoardSpec]:
    """Reconstruct BoardSpec list from per_rollout.csv rows (for --skip-rollout)."""
    seen: dict[int, BoardSpec] = {}
    for row in per_rollout_rows:
        try:
            bi = int(row["board_index"])
        except (KeyError, ValueError, TypeError):
            continue
        if bi not in seen:
            seen[bi] = BoardSpec(
                index=bi,
                board_id=str(row.get("board_id", "")),
                path=str(row.get("board_path", "")),
            )
    return [seen[k] for k in sorted(seen)]


def main() -> int:
    parser, args = _parse_args()
    if args.output_dir is None:
        args.output_dir = _default_output_dir(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode, final_drc = _resolve_modes(args)
    inline_drc = "on" if final_drc else "off"
    print(f"[eval] output_dir={output_dir}")
    print(f"[eval] mode={mode} n_envs={args.n_envs} final_drc={final_drc} "
          f"env_drc={args.env_drc!r} selection_method={args.selection_method}")
    # Post-hoc Stage 2 DRC scores saved .kicad_pcb artifacts. Without inline DRC
    # and without saved artifacts (and not resuming a prior run), there is
    # nothing to score — warn so the run doesn't silently report 0 DRC rows.
    if (
        not final_drc
        and not args.skip_drc
        and not args.skip_rollout
        and args.save_artifacts == "off"
    ):
        print(
            "[eval] WARNING: --save-artifacts off with post-hoc DRC "
            "(inline-drc off) — no .kicad_pcb will be saved, so Stage 2 DRC "
            "will score 0 rollouts. Pass --inline-drc on or --save-artifacts on.",
            flush=True,
        )

    started_at = datetime.now(timezone.utc).isoformat()
    stage_wall_times: dict[str, float] = {}
    boards: list[BoardSpec] | None = None
    overall: dict[str, Any] | None = None
    overall_best: dict[str, Any] | None = None
    # Env DRC reward/token switch — resolved in Stage 1 from the ckpt (or the
    # explicit --env-drc override); stays None on the --skip-rollout path.
    step_drc: bool | None = None
    env_overrides_changed: dict[str, tuple[Any, Any]] = {}
    # The Stage 2 scoring settings: a rollout's from the CLI and its env (Stage 1), a
    # re-scoring run's from the cell's record (resolve_rescoring); None = not scored here.
    reward_config: str | None = None
    check_angle: int | None = None
    rescoring: RunConfig | None = None
    rescoring_vs_training: dict[str, Any] = {}   # a re-scoring run's scoring_vs_training
    selection_method = args.selection_method   # a re-scoring run's: the cell's (resolve_rescoring)

    def write_rescore_record(run: RunConfig) -> None:
        """Record a re-scoring stage that succeeded (Stage 2, then Stage 3): a failed stage or a
        --dry-run leaves the cell's records as they were; settings not decided are carried over."""
        write_run_record(output_dir / RESCORE_RUN_RECORD, run, boards=None if boards is None else
                         board_record([b.path for b in boards], boards_list=args.boards_list), extra={
            "scoring": {**run.values, **rescoring_vs_training},
            "explicit": sorted(parser.explicit_dests)})

    # ---- Stage 1 — rollout ----
    if args.skip_rollout:
        print("[eval] --skip-rollout; expecting existing per_rollout.csv")
        # The cell's board list is its expected board set (Stage 2 prices the
        # failed boards' Φ₀, Stage 3 counts a board without rows). --boards-dir
        # here only locates .kicad_pro files and may be a whole dataset root.
        if args.boards_list is not None:
            boards = load_boards_from_dir_or_list(boards_list=args.boards_list)
        # Re-scoring uses the cell's scoring (Stage 2) and selection (Stage 3) — the settings
        # its records name — unless an explicit flag gives another (warned, both recorded).
        needed = (() if args.skip_drc else DRC_SCORING_KEYS) + (() if args.skip_aggregate else ("selection_method",))
        if needed:
            rescoring = resolve_rescoring(output_dir, parser, args, needed)
            if not args.skip_drc:
                reward_config, check_angle = rescoring.reward_config, rescoring.check_angle
            training = cell_training_reward(output_dir)
            if training is not None and "reward_config" in rescoring.values:
                # an aggregate-only run records the cell's rule against training, unwarned
                rescoring_vs_training = scoring_vs_training(training, rescoring.reward_config,
                                                            warn=not args.skip_drc)
            if not args.skip_aggregate:
                selection_method = rescoring.selection_method
    else:
        device = _resolve_device(args.device)
        boards = load_boards_from_dir_or_list(
            boards_dir=args.boards_dir, boards_list=args.boards_list,
        )
        if not boards:
            raise SystemExit("no boards found")
        # The eval CLI's env overrides (recipe < cli), the USER layers of the env config.
        user = user_layers(parser, args, keys=ENV_OVERRIDE_DESTS)

        if args.dry_run:
            print(json.dumps({
                "ckpt": str(args.ckpt),
                "boards": [b.path for b in boards],
                "board_count": len(boards),
                "n_rollouts": args.n_rollouts,
                "n_envs": args.n_envs,
                "rollout_mode": mode,
                "env_drc": args.env_drc, "final_drc": final_drc,
                "env_overrides": _jsonable({k: v for layer in user for k, v in layer.values.items()}),
                "seed": args.seed,
                "output_dir": str(output_dir),
                "save_artifacts": args.save_artifacts,
                "early_stop_finish_no_progress": args.early_stop_finish_no_progress,
                "early_stop_no_geometry_progress": args.early_stop_no_geometry_progress,
            }, indent=2))
            return 0

        print(f"[eval] device={device} boards={len(boards)} "
              f"n_rollouts={args.n_rollouts} seed={args.seed}", flush=True)
        policy, ckpt_args, _it = load_policy_from_ckpt(
            args.ckpt, device, strict=args.strict_ckpt,
        )
        # The training seed stamps every artifact row (flatten below): checked
        # here, before the rollout, not after it.
        if "seed" not in ckpt_args:
            raise SystemExit(f"[eval] checkpoint {args.ckpt} stores no training seed — "
                             "its rollout artifacts cannot be stamped")
        ckpt_seed = int(ckpt_args["seed"])
        # The cap in effect (None = the ckpt's): what the run record and manifest show.
        slot_cap = apply_n_max_slots_override(
            policy, args.override_n_max_slots, explicit="override_n_max_slots" in parser.explicit_dests)

        # The env config: the ckpt's training env, the eval CLI's overrides (recipe < cli) on
        # top — resolved once in mode eval (only eval_override fields may differ from the
        # ckpt). Env DRC reward/tokens follow the ckpt's emit_drc_tokens unless --env-drc
        # forces them; step_drc=False is the DRC_OFF layer.
        emit_drc_tokens = bool(ckpt_args["emit_drc_tokens"])
        step_drc = emit_drc_tokens if args.env_drc is None else (args.env_drc == "on")
        if not step_drc and emit_drc_tokens:
            print(
                f"[eval] WARNING: env DRC tokens/reward will be stripped "
                f"(--env-drc={args.env_drc!r}) but the ckpt was trained with "
                f"emit_drc_tokens=True -> train/eval mismatch.",
                flush=True,
            )
        elif args.env_drc == "on" and not emit_drc_tokens:
            print(
                "[eval] NOTE: --env-drc on but the ckpt was trained with "
                "no_drc_tokens; env keeps the ckpt's no-token setting.",
                flush=True,
            )
        env_run = resolve_eval_env(
            RLEnvConfig.from_checkpoint(ckpt_args, int(ckpt_args["max_steps"])),
            step_drc=step_drc, where=str(args.ckpt), sources=policy.ckpt_sources, user=user)
        if args.expect is not None:
            check_expect(args.expect, env_run)
        env_config = eval_env_config(env_run)
        max_steps = env_config.env.max_steps
        env_overrides_changed = env_overrides_vs_ckpt(env_run, str(args.ckpt))
        for key, (old, new) in env_overrides_changed.items():
            print(f"[eval] WARNING: env override {key}: ckpt={old!r} -> {new!r} "
                  f"(differs from training)", flush=True)
        if "reward_rule" in env_overrides_changed:
            print(
                "[eval] WARNING: reward_rule overridden — via/wirelength/drc "
                "penalties that relied on the training rule's defaults now follow "
                "the NEW rule's defaults unless explicitly overridden.",
                flush=True,
            )
        print(f"[eval] step_drc={step_drc} (env-drc={args.env_drc!r}, "
              f"ckpt emit_drc_tokens={emit_drc_tokens})", flush=True)
        reward_config = args.reward_config
        # DRC track-angle: the angle the env routes with (the ckpt's, or the
        # --corner-mode override); an explicit --check-angle must equal it.
        check_angle = u.resolve_check_angle(env_config.env.corner_mode, args.check_angle)
        print(f"[eval] check_angle follows the env corner_mode: {check_angle}",
              flush=True)
        # Recorded before any env exists, so a crash still leaves it; the scoring settings
        # are what a later re-scoring of this cell reuses (resolve_rescoring). check_angle is
        # the env's; reward_config only when this run scores (Stage 2 or inline) or names it —
        # an unscored cell's first scoring chooses its rule —, selection_method likewise only
        # when this run aggregates (Stage 3) or names it.
        # A scored rule is checked against the ckpt's training reward_rule (scoring_vs_training).
        scoring = ({"reward_config": reward_config,
                    **scoring_vs_training(ckpt_args, reward_config)}
                   if final_drc or not args.skip_drc or "reward_config" in parser.explicit_options else {})
        scoring["check_angle"] = check_angle
        if not args.skip_aggregate or "selection_method" in parser.explicit_options:
            scoring["selection_method"] = selection_method
        write_run_record(output_dir, env_run, boards=board_record(
            [b.path for b in boards], boards_dir=args.boards_dir, boards_list=args.boards_list), extra={
            "ckpt": str(args.ckpt), "scoring": scoring,
            "invocation": _jsonable({**vars(args), "override_n_max_slots": slot_cap}), "explicit": sorted(parser.explicit_dests)})

        artifacts_dir = output_dir / "artifacts" if args.save_artifacts == "on" else None
        per_rollout_csv = output_dir / "per_rollout.csv"
        sel_mode_for_overlay = (
            None if args.selection_method == "none" else args.selection_method
        )

        # Stochastic rollouts sample actions from torch's global RNG in this process (the env
        # workers seed themselves from --seed per slot): seeded here, a --seed reproduces them.
        import torch
        torch.manual_seed(args.seed)
        t0 = time.monotonic()
        result = eval_transformer(
            policy, device, boards,
            env_config=env_config,
            n_rollouts=args.n_rollouts, n_envs=args.n_envs,
            base_seed=args.seed, max_steps=max_steps,
            deterministic=args.deterministic,
            boards_per_batch=args.boards_per_batch,
            final_drc=final_drc,
            reward_config=reward_config, check_angle=check_angle,
            selection_mode=sel_mode_for_overlay,
            save_artifacts=(args.save_artifacts == "on"),
            artifacts_dir=artifacts_dir,
            early_stop_finish_no_progress=args.early_stop_finish_no_progress,
            early_stop_no_geometry_progress=args.early_stop_no_geometry_progress,
            per_rollout_flush_path=per_rollout_csv,
            skip_aggregation=True,
        )
        stage_wall_times["rollout_wall_sec"] = time.monotonic() - t0
        # per_rollout.csv already flushed incrementally. Write manifest only.
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "env_kwargs": _jsonable(env_config.to_record()),
            "args": _jsonable({**vars(args), "override_n_max_slots": slot_cap}),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        # Bridge Stage 1 -> Stage 3: rename the saved artifacts into the
        # unified boards/ cell layout that aggregation parses, and stamp the
        # ckpt on every row. No manual flattening step is needed between the
        # stages.
        flatten_rollout_artifacts(output_dir, ckpt_seed=ckpt_seed)

    # ---- Stage 2 — post-hoc DRC ----
    if args.skip_drc:
        print("[eval] --skip-drc set")
    else:
        t_drc = time.monotonic()
        if final_drc:
            print("[eval] inline DRC ran during rollout; skipping Stage 2 artifact scoring")
        else:
            eval_kicad_pcb(
                output_dir,
                n_workers=args.n_envs,
                reward_config=reward_config,
                check_angle=check_angle,
                boards_dir=args.boards_dir,
            )
        if not args.dry_run:
            score_bare_boards(
                output_dir, boards,
                n_workers=args.n_envs,
                reward_config=reward_config,
                check_angle=check_angle,
                boards_dir=args.boards_dir,
            )
        stage_wall_times["drc_wall_sec"] = time.monotonic() - t_drc
        if rescoring is not None and not args.dry_run:
            # The rows now carry the new DRC scoring: recorded before Stage 3 (whose failure must
            # not leave the old record), with the selection still the cell's.
            write_rescore_record(_rescore_drc_only(rescoring))

    # ---- Stage 3 — standard per-board aggregation (method-agnostic) ----
    # Reads the unified per_rollout.csv (DRC-filled by Stage 1 inline / Stage 2)
    # and writes per_boards_ckpts.csv + per_boards_overall.csv. Operates at the
    # CELL level (output_dir holds boards/<id>_<cell>_s<SS>_r<RR> + per_rollout.csv),
    # so the typical call is `--stages eval,aggregate --output-dir <cell>` after
    # rollouts have been unified. Paper-table-specific selection/metrics (Table 1
    # DRC-aware pick, Table 2 P@k/CP@k) live in the separate Stage-4 tooling.
    if args.skip_aggregate:
        print("[eval] --skip-aggregate set")
    else:
        print("\n[eval] >>> Stage 3 aggregate_boards", flush=True)
        if not args.dry_run:
            t_agg = time.monotonic()
            from eval.aggregation import aggregate_boards
            sel_mode = (selection_method
                        if selection_method in ("final_potential", "posthoc_drc_aware")
                        else "final_potential")
            # Stage 1's board set, or --boards-list on --skip-rollout; else the rows'.
            aggregate_boards(output_dir, selection_mode=sel_mode, expected_boards=boards)
            stage_wall_times["aggregate_wall_sec"] = time.monotonic() - t_agg
            print(
                f"[eval] <<< Stage 3 aggregate done in "
                f"{stage_wall_times['aggregate_wall_sec']:.1f}s",
                flush=True,
            )

    if rescoring is not None and not args.dry_run and not args.skip_aggregate:
        write_rescore_record(rescoring)   # the selection Stage 3 aggregated with

    finished_at = datetime.now(timezone.utc).isoformat()
    summary_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "rollout_mode": mode,
        "step_drc": step_drc, "final_drc": final_drc, "inline_drc": inline_drc,
        "env_drc": args.env_drc,
        "env_overrides": {
            k: {"ckpt": _jsonable(old), "override": _jsonable(new)}
            for k, (old, new) in env_overrides_changed.items()
        },
        "reward_config": reward_config,
        "check_angle": check_angle,
        "n_envs": int(args.n_envs),
        "n_rollouts": int(args.n_rollouts) if args.n_rollouts is not None else None,
        "seed": int(args.seed) if args.seed is not None else None,
        "selection_method": selection_method,
        "selection_mode": selection_method,
        "skipped": {
            "rollout": bool(args.skip_rollout),
            "drc": bool(args.skip_drc),
            "aggregate": bool(args.skip_aggregate),
        },
        "stage_wall_times_sec": stage_wall_times,
        "total_rollout_time": stage_wall_times.get("rollout_wall_sec"),
        "total_wall_sec": sum(stage_wall_times.values()),
        "output_dir": str(output_dir),
        "ckpt": str(args.ckpt) if args.ckpt is not None else None,
        "boards_dir": str(args.boards_dir) if args.boards_dir else None,
        "boards_list": str(args.boards_list) if args.boards_list else None,
    }
    if overall is not None:
        summary_data["overall"] = _jsonable(overall)
    if overall_best is not None:
        best_selection_name = _selection_output_name(selection_method)
        summary_data[f"overall_best_{best_selection_name}"] = _jsonable(overall_best)
    (output_dir / "eval_overall_summary.json").write_text(
        json.dumps(summary_data, indent=2, sort_keys=True),
    )

    print(f"\n[eval] all stages done")
    print(f"[eval] stage wall times: {stage_wall_times}")
    print(f"[eval] total wall: {sum(stage_wall_times.values()):.1f}s")
    print(f"[eval] summary: {output_dir / 'eval_overall_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
