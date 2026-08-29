"""KiCad PCB evaluation pipeline.

Naming policy: inside this package the redundant ``eval_`` prefix is dropped, and
the Stage-1 rollout producers live under the ``rollout`` subpackage.

Canonical eval defaults + shared constants (SCHEMA_VERSION, RLEvalConfig /
DEFAULTS) live in ``configs.loader.schema``.

Modules
-------
args            argparse builders for the LLM-env eval.
eval_utils      Stdlib-only CSV / schema / metric helpers.
metrics         Scoring kernel: compute_metrics / evaluate_one / aggregate
                + EvalResult + EvalSummary (sink-agnostic summary).
evaluator       Central Evaluator (run / score_boards) + CSV/JSON export sinks
                (export_csv / export_json / emit_csv_artifacts).
aggregation     Per-board / overall aggregation.
pipeline        Eval CLI orchestration + post-hoc DRC (eval_kicad_pcb, main);
                re-imports the emit_* sinks.
rollout/        Stage-1 rollout producers — one module per method family;
                LLM producers live in methods/llm_agent/rollout:
  rl            Stage-1 RL rollout driver (eval_transformer): batches (board,
                rollout) jobs into waves, drives the loop in
                methods/rl_agent/rollout/transformer, flushes per-rollout rows.
  rule-based     Rule-based-router rollout.

Moved out of this package: board loaders (BoardSpec) -> methods/_shared/
board_loader.py, metric-logging sinks -> methods/_shared/logger.py,
checkpoint/policy loaders -> methods/rl_agent/models/loader.py, the parallel
DRC worker pool -> pcb_world/vec/subproc_pool.py.
"""
