"""Versioned cadagent prompt templates (v1–v4) + dispatcher."""

from methods.llm_agent.wrappers.prompts.dispatcher import (
    CADAGENT_PROMPT_VERSIONS,
    CadagentPromptBundle,
    get_cadagent_prompts,
)

__all__ = [
    "CADAGENT_PROMPT_VERSIONS",
    "CadagentPromptBundle",
    "get_cadagent_prompts",
]
