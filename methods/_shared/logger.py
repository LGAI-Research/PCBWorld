"""Metric-logging (sink) layer for the eval pipeline.

The half of the metric/data split that knows about *where scalars go*:

  * :class:`MetricLogger` — the minimal sink contract (``add_scalar``).
  * :func:`log_metrics`  — project an :class:`eval.metrics.EvalSummary`
    to ``<prefix>/<key>`` scalars and push them to a ``MetricLogger``.
  * :class:`TensorBoardLogger` / :class:`WandbLogger` — single-backend sinks.
  * :class:`MultiLogger` — fan a scalar out to several sinks.
  * :func:`build_logger` — the common case: TensorBoard always, plus W&B when
    enabled, composed into one ``MultiLogger``.

Kept separate from :mod:`eval.metrics` (the sink-agnostic *data* kernel) so the
data does not depend on torch/wandb. ``SummaryWriter``/``wandb`` are imported
lazily inside the backend loggers so importing this module stays torch-free.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Protocol, runtime_checkable

from eval.metrics import EvalResult, EvalSummary

# W&B x-axis field for late-arriving (async validation) scalars — see
# WandbLogger.add_scalars_async.
_ASYNC_STEP_KEY = "async_val/iteration"


@runtime_checkable
class MetricLogger(Protocol):
    """Minimal scalar-logging sink.

    Any object exposing ``add_scalar(tag, value, step)`` conforms — the backend
    loggers below, a :class:`MultiLogger`, and a bare
    ``torch.utils.tensorboard.SummaryWriter``.
    """

    def add_scalar(self, tag: str, value: float, step: int) -> None: ...


def log_metrics(
    logger: MetricLogger,
    metrics: EvalSummary,
    step: int,
    *,
    prefix: str = "val",
    include_per_board: bool = True,
) -> None:
    """Send a metrics summary to a scalar logger under ``<prefix>/<key>`` tags.

    Validation uses ``prefix="val"``; the standalone eval path uses
    ``prefix="eval"`` (see :func:`emit_tensorboard` below). Keeping this
    free function separate from :class:`EvalSummary` is the metric/logger
    split: the data does not know about wandb/TensorBoard.
    """
    for tag, value in metrics.to_logger_dict(
        prefix=prefix, include_per_board=include_per_board,
    ).items():
        logger.add_scalar(tag, value, step)


def emit_tensorboard(
    result: EvalResult,
    writer: MetricLogger,
    *,
    step: int,
    prefix: str = "eval",
) -> None:
    """Push an :class:`~eval.metrics.EvalResult`'s scalars to a metric logger.

    The ``EvalResult``-shaped counterpart of :func:`log_metrics`: projects via
    :class:`~eval.metrics.EvalSummary` and logs under ``<prefix>/<key>``.
    The standalone eval CLI passes ``prefix="eval"``; in-training validation
    uses :func:`log_metrics` directly with ``prefix="val"``.
    """
    log_metrics(
        writer, EvalSummary.from_eval_result(result), step, prefix=prefix,
    )


# ============================================================================
# Single-backend sinks
# ============================================================================


class TensorBoardLogger:
    """Scalar sink backed by a TensorBoard ``SummaryWriter``."""

    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir)

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        self._writer.add_histogram(tag, values, step)

    def close(self) -> None:
        self._writer.close()


class WandbLogger:
    """Scalar sink backed by Weights & Biases.

    Construct via :meth:`maybe`, which honours the opt-in rules and returns
    ``None`` when W&B is not requested or fails to initialise (so training
    never crashes on a W&B issue).
    """

    def __init__(self, wandb_module: Any) -> None:
        self._wandb = wandb_module
        # add_scalars_async bookkeeping: highest step logged so far (late
        # results must not rewind W&B's monotonic step) + prefixes already
        # routed to the async step metric via define_metric.
        self._last_step = 0
        self._async_prefixes: set[str] = set()

    @classmethod
    def maybe(
        cls,
        log_dir: str,
        *,
        args: Any | None = None,
        config_extras: dict[str, Any] | None = None,
    ) -> "WandbLogger | None":
        """Build a W&B logger, or ``None`` if disabled/unavailable.

        W&B is **opt-in**: skipped unless ``--wandb`` is set. When it is set, a
        W&B credential must be resolvable (``WANDB_API_KEY`` env OR a
        ``~/.netrc`` ``api.wandb.ai`` entry — the latter is what an interactive
        ``wandb login`` writes and what ``wandb.init`` itself reads). When W&B
        is requested but no credential is found it **warns loudly** before
        returning ``None`` so a run that silently logs to TensorBoard only is
        never mistaken for a W&B run. On a ``wandb.init`` exception (offline
        cluster, network issue) it warns once and returns ``None``.
        """
        if args is None or not getattr(args, "wandb", False):
            return None
        # Accept env-var OR netrc auth (wandb.init resolves the key from either).
        have_cred = bool(os.environ.get("WANDB_API_KEY"))
        if not have_cred:
            try:
                import netrc as _netrc
                have_cred = bool(_netrc.netrc().authenticators("api.wandb.ai"))
            except Exception:  # noqa: BLE001  (missing/unhealthy ~/.netrc)
                have_cred = False
        if not have_cred:
            warnings.warn(
                "--wandb was requested but no credential found (set WANDB_API_KEY "
                "or run `wandb login` to populate ~/.netrc); logging to "
                "TensorBoard only. Drop --wandb to silence this.",
                stacklevel=2,
            )
            return None
        try:
            import wandb  # type: ignore

            config: dict[str, Any] = {
                k: v for k, v in vars(args).items() if _is_jsonable(v)
            }
            if config_extras:
                config.update(config_extras)

            tags = getattr(args, "wandb_tags", None)
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            run_name = getattr(args, "wandb_run_name", None) or os.path.basename(
                os.path.normpath(log_dir)
            )
            wandb.init(
                project=getattr(args, "wandb_project", None)
                or os.environ.get("WANDB_PROJECT", "pcbworld"),
                entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
                group=getattr(args, "wandb_group", None) or os.environ.get("WANDB_RUN_GROUP"),
                id=os.environ.get("WANDB_RUN_ID") or None,
                name=run_name,
                tags=tags,
                config=config,
                dir=log_dir,
                resume="allow",
            )
            return cls(wandb)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"wandb.init failed ({exc!r}); continuing without W&B.",
                stacklevel=2,
            )
            return None

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        try:
            self._last_step = max(self._last_step, int(step))
            self._wandb.log({tag: float(value)}, step=int(step))
        except Exception:  # noqa: BLE001 — never crash training
            pass

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        try:
            self._last_step = max(self._last_step, int(step))
            self._wandb.log(
                {tag: self._wandb.Histogram(values)}, step=int(step),
            )
        except Exception:  # noqa: BLE001 — never crash training
            pass

    def add_scalars_async(self, scalars: dict, step: int) -> None:
        """Log scalars computed for a PAST step (async validation results).

        W&B silently drops writes whose ``step`` is behind the run's current
        step, so late results are merged into the CURRENT row instead, carrying
        their true x-position in the ``async_val/iteration`` field;
        ``define_metric`` points every affected ``<prefix>/*`` panel at that
        field as its x-axis (same values the inline path would have used).
        """
        try:
            for pfx in {tag.split("/", 1)[0] for tag in scalars}:
                if pfx not in self._async_prefixes:
                    self._wandb.define_metric(
                        f"{pfx}/*", step_metric=_ASYNC_STEP_KEY,
                    )
                    self._async_prefixes.add(pfx)
            row = {tag: float(v) for tag, v in scalars.items()}
            row[_ASYNC_STEP_KEY] = int(step)
            self._wandb.log(row, step=max(self._last_step, int(step)))
        except Exception:  # noqa: BLE001 — never crash training
            pass

    def close(self) -> None:
        try:
            self._wandb.finish()
        except Exception:  # noqa: BLE001
            pass


# ============================================================================
# Composition
# ============================================================================


class MultiLogger:
    """Fan ``add_scalar`` / ``close`` out to several :class:`MetricLogger` sinks."""

    def __init__(self, loggers: list[MetricLogger]) -> None:
        self._loggers = [lg for lg in loggers if lg is not None]

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        for lg in self._loggers:
            lg.add_scalar(tag, value, step)

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        for lg in self._loggers:
            fn = getattr(lg, "add_histogram", None)
            if fn is not None:
                fn(tag, values, step)

    def add_scalars_async(self, scalars: dict, step: int) -> None:
        """Fan out a batch of scalars for a PAST ``step`` (async validation).

        Sinks that can handle out-of-order steps directly (TensorBoard) get
        plain per-scalar ``add_scalar`` calls at the true step; sinks with a
        monotonic-step constraint (W&B) implement ``add_scalars_async``.
        """
        for lg in self._loggers:
            fn = getattr(lg, "add_scalars_async", None)
            if fn is not None:
                fn(scalars, step)
            else:
                for tag, value in scalars.items():
                    lg.add_scalar(tag, value, step)

    def close(self) -> None:
        for lg in self._loggers:
            close = getattr(lg, "close", None)
            if close is not None:
                close()


def build_logger(
    log_dir: str,
    *,
    args: Any | None = None,
    config_extras: dict[str, Any] | None = None,
) -> MultiLogger:
    """The common training sink: TensorBoard always, plus W&B when enabled.

    The caller passes the parsed argparse ``args`` plus an optional
    ``config_extras`` dict; W&B is added only if :meth:`WandbLogger.maybe`
    returns a logger.
    """
    sinks: list[MetricLogger] = [TensorBoardLogger(log_dir)]
    wb = WandbLogger.maybe(log_dir, args=args, config_extras=config_extras)
    if wb is not None:
        sinks.append(wb)
    return MultiLogger(sinks)


def _is_jsonable(v: Any) -> bool:
    """Guard against non-serialisable argparse values (e.g. file handles)."""
    return isinstance(v, (str, int, float, bool, type(None), list, tuple, dict))
