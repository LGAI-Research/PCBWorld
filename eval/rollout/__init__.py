"""Rollout producers — Stage-1 of the eval pipeline.

Each module produces routed boards + canonical ``per_rollout`` rows that the
shared scoring/aggregation layer (``eval.metrics`` / ``eval.evaluator``) and the
post-hoc stages (``eval.pipeline``) consume.

One module per method family; the LLM producers live in
``methods.llm_agent.rollout``.

Modules
-------
rl           Stage-1 RL rollout driver (``eval_transformer``); the rollout loop
             itself stays in ``methods.rl_agent.rollout.transformer``.
rule_based    Rule-based router rollout (KRT / OrthoRoute; runs in the cadagent env).
"""
