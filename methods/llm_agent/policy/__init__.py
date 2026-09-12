"""LLM inference layer — the policy-side counterpart of ``policy/rl``.

Symmetric with ``policy/rl`` (RL model + agent + providers): there is no
in-process LLM *model* (it lives behind vLLM / an API), so this package holds
only the inference adapters — the generation providers and the
:class:`~methods.llm_agent.policy.agent.KiCadLLMAgent` facade. ``methods/llm_agent/training`` keeps just
the trainer + the verl rollout manager.

Modules
-------
agent           KiCadLLMAgent — mode-selected generation facade (methods._shared.agent.Agent).
model_provider  VLLMProvider / APIProvider / RemoteLLMProvider backends.
rate_limiter    SlidingWindowLimiter for API providers.
"""
