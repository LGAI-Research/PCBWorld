"""Canonical Agent contract + the model/agent/manager/trainer taxonomy.

Naming taxonomy (the single vocabulary both branches follow):

  Model    Weights + forward pass only. RL: :class:`methods.rl_agent.policy.decoder_only_
           policy.KiCadRLModel` (network + tokenizer + heads); LLM: the
           weights behind vLLM / an API.
  Agent    Performs model-based inference — one decision step from
           observation(s)/prompt(s) to action(s)/text. Owns or calls a
           Model; holds no episode state. RL: :class:`methods.rl_agent.policy.agent.
           KiCadRLAgent`; LLM: :class:`methods.llm_agent.policy.agent.KiCadLLMAgent`.
  Manager  Vectorised episode orchestration over N envs — assembles
           observations/prompts and masks, calls an Agent, steps envs,
           keeps memory and episode-end scoring. Uses an Agent; is NOT one.
  Trainer  The learning loop (:class:`methods._shared.trainer.base.Trainer`).

The :class:`Agent` protocol below is the cross-branch text-wire contract
(historically ``methods.llm_agent.policy.model_provider.Provider``): one generated
response per prompt pair, with raw env observations as an optional side
channel. The RL serving agent satisfies it by rendering decoded actions as
LLM-format ``<think>/<action>`` envelopes, so callers (visualizer, eval CLI)
drive RL and LLM agents through one interface.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, runtime_checkable

# One (system_prompt, user_prompt) pair per env / decision step.
PromptPair = Tuple[str, str]
# (system_tokens, user_tokens, response_tokens) per generated response.
TokenCounts = Tuple[int, int, int]


@runtime_checkable
class Agent(Protocol):
    """Minimal model-inference contract shared by the RL and LLM branches."""

    def generate(
        self,
        prompt_pairs: List[PromptPair],
        observations: Optional[list] = None,
    ) -> Tuple[List[str], List[TokenCounts]]:
        """Generate one response per prompt pair.

        ``observations`` is an optional parallel list of raw env obs dicts
        (one per prompt pair). Only RL agents consume it (they act on the
        obs dict, not the prompt text); LLM agents ignore it.
        """
        ...

    def close(self) -> None: ...


__all__ = ["Agent", "PromptPair", "TokenCounts"]
