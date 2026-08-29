"""Generic training-loop skeleton shared by RL and (later) LLM trainers.

`Trainer(ABC)` owns only the algorithm-agnostic scaffolding:

  setup() -> for iteration in [start, iterations]: train_iteration(it);
             on_iteration_end(it, metrics) -> teardown()

plus a `validate()` hook that drives the central :class:`eval.evaluator.Evaluator`
and logs under ``val/*`` (the single RL/LLM validation surface), and thin
logging plumbing. Everything algorithm-specific (rollout, update, metric
sources, checkpoint policy) lives in subclasses.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    """Algorithm-agnostic training loop.

    Subclasses fill :meth:`setup`, :meth:`train_iteration`, and optionally
    :meth:`on_iteration_end` / :meth:`teardown`. ``writer`` (a
    :class:`~methods._shared.logger.MetricLogger`) and ``evaluator`` (an
    :class:`~eval.evaluator.Evaluator`, or ``None`` to skip validation) are set
    during :meth:`setup`.
    """

    writer: Any = None
    evaluator: Any = None

    def __init__(self, *, iterations: int, start_iteration: int = 1,
                 max_wallclock_sec: float | None = None) -> None:
        self.iterations = iterations
        self.start_iteration = start_iteration
        # Optional wallclock budget: fit() stops after the iteration that first
        # crosses this many seconds (policy_last.pt is saved every iteration, so
        # the latest policy is always preserved). None = run all iterations.
        self.max_wallclock_sec = max_wallclock_sec

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """Build models/optimizers/envs/logger/evaluator. Called once by fit()."""

    @abstractmethod
    def train_iteration(self, iteration: int) -> dict:
        """Run one training iteration and return its scalar metrics dict."""

    def on_iteration_end(self, iteration: int, metrics: dict) -> None:
        """Post-iteration bookkeeping (checkpoints, validation cadence)."""

    def teardown(self) -> None:
        """Release resources after the final iteration."""

    # ------------------------------------------------------------------
    # Validation (central Evaluator -> val/* logging)
    # ------------------------------------------------------------------

    def validate(self, iteration: int, *, prefix: str = "val") -> Any:
        """Run the injected Evaluator and log its summary under ``<prefix>/*``.

        Returns the :class:`~eval.metrics.EvalSummary` (or ``None`` when no
        evaluator is configured). Wall-clock is folded into the canonical
        ``eval_time_sec`` field so it logs as ``<prefix>/eval_time_sec``.
        """
        if self.evaluator is None:
            return None
        from methods._shared.logger import log_metrics

        t0 = time.time()
        metrics = self.evaluator.run()
        metrics.overall["eval_time_sec"] = time.time() - t0
        # Log only the overall averages (<prefix>/<key>), NOT the per-board
        # <prefix>/per_board/<board_id>/<key> scalars — those explode the
        # dashboard (128 synth + 10 d3b boards × many metrics ≈ thousands of
        # tags). Per-board detail lives in the eval CSVs, not TB/W&B.
        log_metrics(self.writer, metrics, step=iteration, prefix=prefix,
                    include_per_board=False)
        return metrics

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        # Fatal-signal crash log for the main process (pcb_world.diag);
        # fail-soft + idempotent, atexit removes the empty log on clean exit.
        from pcb_world import diag

        diag.install_crash_handler("trainer")
        self.setup()
        t0 = time.perf_counter()
        # True only when the loop below exits normally (incl. wallclock stop).
        # teardown() reads it: an aborted run must NOT signal "train_done" to
        # an --async-val watcher — the watcher would exit and the relaunch
        # would then run with no watcher at all.
        self.fit_completed = False
        try:
            for iteration in range(self.start_iteration, self.iterations + 1):
                metrics = self.train_iteration(iteration)
                self.on_iteration_end(iteration, metrics)
                if (self.max_wallclock_sec is not None
                        and time.perf_counter() - t0 >= self.max_wallclock_sec):
                    print(f"  [wallclock stop] {time.perf_counter() - t0:.0f}s >= "
                          f"{self.max_wallclock_sec:.0f}s budget after iteration "
                          f"{iteration}; stopping.", flush=True)
                    break
            self.fit_completed = True
        except Exception:
            # Observability only — dump context and re-raise unchanged
            # (fail-fast stays; no exception is swallowed here).
            import traceback

            diag.dump_context("trainer_exception", traceback=traceback.format_exc())
            raise
        finally:
            self.teardown()
