"""KiCadLLMAgent — the LLM inference facade (Agent layer).

Taxonomy (see :mod:`methods._shared.agent`): the Model is the weights behind vLLM /
an API endpoint; this class is the Agent — it performs model-based
inference, one decision step from prompt pairs to response texts. The
generation backends (:class:`~methods.llm_agent.policy.model_provider.VLLMProvider` /
``APIProvider`` / ``RemoteLLMProvider``) each satisfy the
:class:`methods._shared.agent.Agent` protocol already; this facade adds the
mode-based factory so callers pick a *mode*, not a provider class.

Scope note: during verl training the verl actor (vLLM rollout workers
inside ``multi_turn_loop``) plays the Agent role — this class is the
first-class citizen of the eval / serve / visualize paths. Text→action
projection stays env-side (:mod:`methods.llm_agent.wrappers.action_converter`),
driven by the rollout manager.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from methods._shared.agent import PromptPair, TokenCounts

__all__ = ["KiCadLLMAgent"]


class KiCadLLMAgent:
    """Mode-selected LLM generation facade (conforms to ``methods._shared.agent.Agent``)."""

    #: mode -> backend class name in :mod:`methods.llm_agent.policy.model_provider`
    _BACKENDS = {
        "llm": "VLLMProvider",
        "api": "APIProvider",
        "remote": "RemoteLLMProvider",
    }

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    @classmethod
    def from_args(cls, args: Any, mode: Optional[str] = None) -> "KiCadLLMAgent":
        """Build from eval/CLI args. ``mode`` defaults to ``args.mode``."""
        import methods.llm_agent.policy.model_provider as mp

        mode = mode or getattr(args, "mode", "llm")
        cls_name = cls._BACKENDS.get(mode)
        if cls_name is None:
            raise ValueError(
                f"KiCadLLMAgent mode must be one of {sorted(cls._BACKENDS)}, "
                f"got {mode!r}"
            )
        return cls(getattr(mp, cls_name)(args))

    def generate(
        self,
        prompt_pairs: List[PromptPair],
        observations: Optional[list] = None,
        **kwargs: Any,
    ) -> Tuple[List[str], List[TokenCounts]]:
        return self.backend.generate(prompt_pairs, observations, **kwargs)

    def close(self) -> None:
        self.backend.close()
