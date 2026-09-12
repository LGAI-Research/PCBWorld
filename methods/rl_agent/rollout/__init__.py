"""RL rollout layer (transformer decoder policy).

Modules
-------
primitive    Shared CORE ``iter_rollout`` (act+step→``StepBatch``) — the one
             per-step loop under eval rollout AND the training collectors
             (``methods.rl_agent.training.collect``). ≈ SB3 collect_rollouts.
transformer  Eval rollout loop on top of it: ``_run_one_batch`` + early-stop
             guards + per-rollout row assembly.
"""
