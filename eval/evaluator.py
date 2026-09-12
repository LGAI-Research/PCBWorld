"""Central evaluator + sinks.

There is one evaluator. "Validation" is just this evaluator run on a held-out
set during training and logged under ``val/*``; the standalone CLI is the same
evaluator run over a full board set and exported to CSV. Both produce the same
:class:`eval.metrics.EvalSummary`, which is then handed to a sink.

  rollout_fn ─▶ Evaluator.run() ─▶ EvalSummary ─┬─▶ log_metrics()  (wandb / TensorBoard)
                                                       ├─▶ export_csv()   (per_rollout / per_board / summary)
                                                       └─▶ export_json()  (overall / per_board[/ per_rollout])

The rollout strategy is injected (``rollout_fn``), so the evaluator never
depends on a particular policy/runtime: RL passes ``eval.rollout.rl.
eval_transformer``; the LLM path passes an episode-rollout that returns
per-rollout rows. The metric *data* (:class:`EvalSummary`) is kept separate
from the *sink* (:class:`MetricLogger` / export functions) so wandb/TensorBoard
is not baked into the metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence, Union

from methods._shared.board_loader import BoardSpec
from configs.loader.schema import DEFAULTS, SCHEMA_VERSION
from eval.metrics import EvalResult, EvalSummary, _jsonable

__all__ = [
    "Evaluator",
    "export_csv",
    "export_json",
    "emit_csv_artifacts",
]


# A rollout function maps a board list to either an aggregated EvalResult (RL
# ``eval_transformer``) or raw per-rollout rows (the LLM episode path); the
# evaluator normalizes both into EvalSummary.
RolloutResult = Union[EvalResult, list]
RolloutFn = Callable[[Sequence[BoardSpec]], RolloutResult]


class Evaluator:
    """Run an injected rollout, aggregate, and return EvalSummary.

    Stateless aside from its configuration; call :meth:`run` as often as needed
    (e.g. once per validation cadence during training).
    """

    def __init__(
        self,
        rollout_fn: RolloutFn,
        boards: Sequence[BoardSpec],
        *,
        selection_mode: str | None = None,
    ) -> None:
        self.rollout_fn = rollout_fn
        self.boards = list(boards)
        self.selection_mode = selection_mode

    def run(self) -> EvalSummary:
        """Execute one evaluation pass and return its canonical summary."""
        result = self.rollout_fn(self.boards)
        if isinstance(result, EvalResult):
            return EvalSummary.from_eval_result(result)
        # Raw per-rollout rows (e.g. the LLM episode path): aggregate here.
        return EvalSummary.from_rollouts(
            list(result), self.boards, selection_mode=self.selection_mode,
        )

    @staticmethod
    def score_boards(
        jobs: Sequence[tuple[str, str | None]],
        *,
        reward_config: str = DEFAULTS.reward_config,
        check_angle: int = DEFAULTS.check_angle,
        parallel: int = 1,
        on_result: Callable[[int, dict], None] | None = None,
    ) -> EvalSummary:
        """Score saved ``.kicad_pcb`` files post-hoc (mode B) -> EvalSummary.

        The second evaluation mode, alongside :meth:`run` (rollout, mode A): no
        policy, just design-rule scoring of finished boards. Each ``job`` is a
        ``(routed_pcb, pro_path)`` pair treated as one board (``board_index`` =
        position). ``parallel > 1`` runs the KiCad-fork-safe
        :class:`pcb_world.vec.subproc_pool.SubprocEvalPool`; otherwise scoring is serial
        via :func:`eval.metrics.evaluate_one`. Results flow through the same
        flatten -> :class:`EvalSummary` path as the rollout modes, so every
        sink (logger / CSV / JSON) is shared.
        """
        pairs = [(j[0], j[1]) for j in jobs]
        if parallel and parallel > 1:
            from pcb_world.vec.subproc_pool import SubprocEvalPool

            with SubprocEvalPool(
                parallel, reward_config=reward_config, check_angle=check_angle,
            ) as pool:
                nested = pool.evaluate_many(pairs, on_result=on_result)
        else:
            import traceback

            from eval.metrics import evaluate_one

            nested = []
            for i, (pcb, pro) in enumerate(pairs):
                try:
                    res = evaluate_one(
                        pcb, pro or None,
                        reward_config_name=reward_config, check_angle=check_angle,
                    )
                except Exception as exc:  # noqa: BLE001 — one bad board != abort
                    res = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                nested.append(res)
                if on_result is not None:
                    on_result(i, res)

        rows = _rows_from_scored(
            nested, pairs, reward_config=reward_config, check_angle=check_angle,
        )
        boards = [
            BoardSpec(i, Path(pcb).stem, str(pcb)) for i, (pcb, _pro) in enumerate(pairs)
        ]
        return EvalSummary.from_rollouts(rows, boards)


def _rows_from_scored(
    nested: Sequence[dict],
    pairs: Sequence[tuple[str, str | None]],
    *,
    reward_config: str,
    check_angle: int,
) -> list[dict]:
    """Flatten ``evaluate_one`` nested results into canonical per_rollout rows.

    Shared by :meth:`Evaluator.score_boards`; factored out so the
    flatten + identity assembly is unit-testable without invoking KiCad.
    """
    from eval import eval_utils as u

    rows: list[dict] = []
    for i, (res, (pcb, pro)) in enumerate(zip(nested, pairs)):
        row = u.assemble_rollout_row(
            res,
            identity={
                "board_index": i,
                "board_id": Path(pcb).stem,
                "rollout_idx": 0,
                "routed_pcb": str(pcb),
                "pro_path": str(pro or ""),
            },
            reward_config=reward_config,
            check_angle=check_angle,
        )
        row["status"] = "completed" if row.get("eval_status") == "ok" else "crashed"
        rows.append(row)
    return rows


# ============================================================================
# Sinks
# ============================================================================
#
# Scalar logging (MetricLogger / log_metrics / TensorBoardLogger / WandbLogger)
# lives in :mod:`methods._shared.logger`; CSV/JSON export below.


def export_csv(
    metrics: EvalSummary,
    output_dir: Path | str,
    **kwargs: Any,
) -> None:
    """Write per_rollout / per_board / summary CSVs (+ optional manifest).

    Accepts the keyword arguments of :func:`emit_csv_artifacts`
    (``per_board_best``, ``overall_best``, ``best_selection_name``,
    ``env_kwargs``, ``args_meta``, ``write_manifest``, ``write_summary``).
    """
    emit_csv_artifacts(metrics.as_eval_result(), Path(output_dir), **kwargs)


def emit_csv_artifacts(
    result: EvalResult,
    output_dir: Path,
    *,
    per_board_best: list[dict[str, Any]] | None = None,
    overall_best: dict[str, Any] | None = None,
    best_selection_name: str = "final_potential",
    env_kwargs: dict[str, Any] | None = None,
    args_meta: dict[str, Any] | None = None,
    write_manifest: bool = True,
    write_summary: bool = True,
) -> None:
    """Persist an :class:`~eval.metrics.EvalResult` to ``output_dir``.

    The CSV/JSON sink for the eval pipeline — per_rollout / per_board(_avg) CSVs
    plus an optional manifest + overall-summary JSON. Lives here in the sink
    home (next to :func:`export_csv` / :func:`export_json`); the CLI
    (:mod:`eval.pipeline`) calls it for its staged outputs.
    """
    from eval import eval_utils as u

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    u.write_csv(output_dir / "per_rollout.csv", result.per_rollout, u.PER_ROLLOUT_FIELDS)
    u.write_csv(output_dir / "per_board_avg.csv", result.per_board, u.PER_BOARD_FIELDS)
    # Backward-compatible alias for older readers.
    u.write_csv(output_dir / "per_board.csv", result.per_board, u.PER_BOARD_FIELDS)
    if per_board_best is not None:
        best_path = output_dir / f"per_board_best_{best_selection_name}.csv"
        u.write_csv(best_path, per_board_best, u.PER_BOARD_FIELDS)
    if write_manifest and (env_kwargs is not None or args_meta is not None):
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "env_kwargs": _jsonable(env_kwargs or {}),
            "args": _jsonable(args_meta or {}),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
        )
    if write_summary:
        payload = {"overall": _jsonable(result.overall)}
        if overall_best is not None:
            payload[f"overall_best_{best_selection_name}"] = _jsonable(overall_best)
        (output_dir / "eval_overall_summary.json").write_text(json.dumps(payload, indent=2))


def export_json(
    metrics: EvalSummary,
    path: Path | str,
    *,
    include_per_rollout: bool = False,
    indent: int = 2,
) -> None:
    """Write the JSON-safe ``{overall, per_board[, per_rollout]}`` summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            metrics.to_json_dict(include_per_rollout=include_per_rollout),
            indent=indent,
        )
    )
