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
from methods.rl_agent.models.loader import (
    _corner_mode_code,
    apply_drc_off,
    env_kwargs_from_checkpoint,
    env_kwargs_from_training_args,
)
from methods.rl_agent.rollout.transformer import (
    apply_n_max_slots_override,
    load_policy_from_ckpt,
)
from eval.rollout.rl import eval_transformer

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
    run). Rows whose filename already matches the grammar are left untouched,
    so re-runs are no-ops. Returns the number of files moved.
    """
    cell_dir = Path(cell_dir).resolve()
    cell = cell_dir.name
    per_rollout_path = cell_dir / "per_rollout.csv"
    if not per_rollout_path.is_file():
        return 0
    rows = u.read_csv(per_rollout_path)
    boards_dir = cell_dir / "boards"
    moved = 0
    for row in rows:
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
    if moved:
        u.write_csv(per_rollout_path, rows, u.PER_ROLLOUT_FIELDS)
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
    return sorted(
        output_rows.values(),
        key=lambda r: (int(r.get("board_index", 0)), int(r.get("rollout_idx", 0))),
    )


# ============================================================================
# Emission helpers — each lives with its sink (emit_tensorboard next to
# log_metrics, emit_csv_artifacts next to export_csv/export_json) and is
# re-imported here for the CLI main().
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", type=Path, default=None)
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
            "(errors-only). Override to score with a different DRV config; "
            "warns when it differs from the ckpt's training reward_rule."
        ),
    )
    p.add_argument(
        "--check-angle", type=int, choices=CHECK_ANGLES, default=None,
        help=(
            "DRC track-angle check (45/90). Omitted: inherit the ckpt's "
            "corner_mode."
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
    else:
        if args.ckpt is None:
            p.error("--ckpt is required unless --skip-rollout is set")
        if args.boards_dir is None and args.boards_list is None:
            p.error("--boards-dir or --boards-list is required unless --skip-rollout is set")
        if args.seed is None:
            p.error("--seed is required unless --skip-rollout is set")
        if args.n_rollouts is None:
            p.error("--n-rollouts is required unless --skip-rollout is set")
    return args


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


def _env_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Collect explicitly-set eval-time env overrides (None values = inherit).

    Returned keys match :func:`methods.rl_agent.models.loader.env_kwargs_from_checkpoint` so they
    can be overlaid directly onto the ckpt-derived env_kwargs.
    """
    raw = {
        "reward_rule": args.reward_rule,
        "via_penalty": args.via_penalty,
        "wirelength_penalty": args.wirelength_penalty,
        "drc_penalty": args.drc_penalty,
        "reward_step_penalty": args.reward_step_penalty,
        "corner_mode": (
            None if args.corner_mode is None else _corner_mode_code(args.corner_mode)
        ),
        # store_true flag: absent → None (inherit ckpt/default False)
        "simplify_outline": True if getattr(args, "simplify_outline", False) else None,
    }
    return {k: v for k, v in raw.items() if v is not None}


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
    args = _parse_args()
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
    # reward_config: fixed canonical eval default (overridable; warns on mismatch
    # with the ckpt's reward_rule). check_angle: None = inherit ckpt corner_mode
    # (resolved in Stage 1), falling back to the EvalDefaults constant w/o a ckpt.
    reward_config: str = args.reward_config
    check_angle: int | None = args.check_angle

    # ---- Stage 1 — rollout ----
    if args.skip_rollout:
        print("[eval] --skip-rollout; expecting existing per_rollout.csv")
    else:
        device = _resolve_device(args.device)
        boards = load_boards_from_dir_or_list(
            boards_dir=args.boards_dir, boards_list=args.boards_list,
        )
        if not boards:
            raise SystemExit("no boards found")

        if args.dry_run:
            print(json.dumps({
                "ckpt": str(args.ckpt),
                "boards": [b.path for b in boards],
                "board_count": len(boards),
                "n_rollouts": args.n_rollouts,
                "n_envs": args.n_envs,
                "rollout_mode": mode,
                "env_drc": args.env_drc, "final_drc": final_drc,
                "env_overrides": _jsonable(_env_overrides_from_args(args)),
                "seed": args.seed,
                "output_dir": str(output_dir),
                "save_artifacts": args.save_artifacts,
                "early_stop_finish_no_progress": args.early_stop_finish_no_progress,
                "early_stop_no_geometry_progress": args.early_stop_no_geometry_progress,
            }, indent=2))
            return 0

        print(f"[eval] device={device} boards={len(boards)} "
              f"n_rollouts={args.n_rollouts} seed={args.seed}", flush=True)
        policy, ckpt_args, _it = load_policy_from_ckpt(args.ckpt, device)
        apply_n_max_slots_override(policy, args.override_n_max_slots)
        max_steps = int(args.max_steps) if args.max_steps is not None else int(
            ckpt_args.get("max_steps", 256),
        )
        # Base env config follows the ckpt's training args; eval CLI overrides
        # (non-None only) are overlaid on top. None = inherit.
        env_kwargs = env_kwargs_from_checkpoint(ckpt_args, max_steps)
        overrides = _env_overrides_from_args(args)
        for key, val in overrides.items():
            ckpt_val = env_kwargs.get(key)
            env_kwargs[key] = val
            if ckpt_val != val:
                env_overrides_changed[key] = (ckpt_val, val)
        if overrides:
            print(f"[eval] env overrides (eval CLI): {overrides}", flush=True)
            for key, (old, new) in env_overrides_changed.items():
                print(
                    f"[eval] WARNING: env override {key}: ckpt={old!r} -> {new!r} "
                    f"(differs from training)",
                    flush=True,
                )
            if "reward_rule" in env_overrides_changed:
                print(
                    "[eval] WARNING: reward_rule overridden — via/wirelength/drc "
                    "penalties that relied on the training rule's defaults now follow "
                    "the NEW rule's defaults unless explicitly overridden.",
                    flush=True,
                )
        else:
            print("[eval] env_kwargs fully inherited from ckpt (no eval overrides)",
                  flush=True)

        # Env DRC reward/tokens during rollout: follow the ckpt's emit_drc_tokens
        # unless --env-drc forces it. step_drc=False makes eval_transformer strip
        # the env DRC reward (apply_drc_off).
        emit_drc_tokens = bool(env_kwargs.get("emit_drc_tokens", True))
        if args.env_drc is None:
            step_drc = emit_drc_tokens
        else:
            step_drc = (args.env_drc == "on")
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
        print(
            f"[eval] step_drc={step_drc} (env-drc={args.env_drc!r}, "
            f"ckpt emit_drc_tokens={emit_drc_tokens})",
            flush=True,
        )

        # DRC scoring config: fixed canonical eval default (errors-only); warn
        # when it differs from the ckpt's training reward_rule.
        reward_config = args.reward_config
        ckpt_rule = str(ckpt_args.get("reward_rule", "") or "")
        if ckpt_rule and ckpt_rule != reward_config:
            print(f"[eval] WARNING: DRC scoring reward_config {reward_config!r} "
                  f"differs from ckpt reward_rule {ckpt_rule!r} — scoring with a "
                  f"different DRV config than training.", flush=True)
        # DRC track-angle: inherit the ckpt's corner_mode unless overridden.
        ckpt_corner = int(ckpt_args.get("corner_mode", DEFAULTS.check_angle))
        if ckpt_corner not in CHECK_ANGLES:
            ckpt_corner = DEFAULTS.check_angle
        if args.check_angle is None:
            check_angle = ckpt_corner
            print(f"[eval] check_angle inherited from ckpt corner_mode: "
                  f"{check_angle}", flush=True)
        else:
            check_angle = args.check_angle
            if ckpt_corner != check_angle:
                print(f"[eval] WARNING: --check-angle {check_angle} differs from "
                      f"ckpt corner_mode {ckpt_corner}.", flush=True)

        artifacts_dir = output_dir / "artifacts" if args.save_artifacts == "on" else None
        per_rollout_csv = output_dir / "per_rollout.csv"
        sel_mode_for_overlay = (
            None if args.selection_method == "none" else args.selection_method
        )

        t0 = time.monotonic()
        result = eval_transformer(
            policy, device, boards,
            env_kwargs=env_kwargs,
            n_rollouts=args.n_rollouts, n_envs=args.n_envs,
            base_seed=args.seed, max_steps=max_steps,
            deterministic=args.deterministic,
            boards_per_batch=args.boards_per_batch,
            step_drc=step_drc, final_drc=final_drc,
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
            "env_kwargs": _jsonable(env_kwargs),
            "args": _jsonable(vars(args)),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        # Bridge Stage 1 -> Stage 3: rename the saved artifacts into the
        # unified boards/ cell layout that aggregation parses. No manual
        # flattening step is needed between the stages.
        if args.save_artifacts == "on":
            flatten_rollout_artifacts(
                output_dir, ckpt_seed=int(ckpt_args.get("seed", 0) or 0),
            )

    # check_angle inherits the ckpt's corner_mode; without a ckpt (e.g.
    # --skip-rollout) it stays None, so fall back to the constant here.
    if check_angle is None:
        check_angle = DEFAULTS.check_angle

    # ---- Stage 2 — post-hoc DRC ----
    if args.skip_drc:
        print("[eval] --skip-drc set")
    elif final_drc:
        print("[eval] inline DRC ran during rollout; skipping Stage 2")
    else:
        t_drc = time.monotonic()
        eval_kicad_pcb(
            output_dir,
            n_workers=args.n_envs,
            reward_config=reward_config,
            check_angle=check_angle,
            boards_dir=args.boards_dir,
        )
        stage_wall_times["drc_wall_sec"] = time.monotonic() - t_drc

    # ---- Stage 3 — standard per-board aggregation (method-agnostic) ----
    # Reads the unified per_rollout.csv (DRC-filled by Stage 1 inline / Stage 2)
    # and writes per_boards_ckpts.csv + per_boards_overall.csv. Operates at the
    # CELL level (output_dir holds boards/<id>_<cell>_s<SS>_r<RR> + per_rollout.csv),
    # so the typical call is `--stages eval,aggregate --output-dir <cell>` after
    # rollouts have been unified. Paper-table-specific selection/metrics (Table 1
    # DRC-aware pick, Table 2 P@k/CP@k) live in the separate paper-extraction
    # tooling under experiments/_lib/metrics/.
    if args.skip_aggregate:
        print("[eval] --skip-aggregate set")
    else:
        print("\n[eval] >>> Stage 3 aggregate_boards", flush=True)
        if not args.dry_run:
            t_agg = time.monotonic()
            from eval.aggregation import aggregate_boards
            sel_mode = (args.selection_method
                        if args.selection_method in ("final_potential", "posthoc_drc_aware")
                        else "final_potential")
            aggregate_boards(output_dir, selection_mode=sel_mode)
            stage_wall_times["aggregate_wall_sec"] = time.monotonic() - t_agg
            print(
                f"[eval] <<< Stage 3 aggregate done in "
                f"{stage_wall_times['aggregate_wall_sec']:.1f}s",
                flush=True,
            )

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
        "selection_method": args.selection_method,
        "selection_mode": args.selection_method,
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
        best_selection_name = _selection_output_name(args.selection_method)
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
