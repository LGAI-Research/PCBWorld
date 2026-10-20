"""API-mode wiring the LLM rollout depends on.

Three behaviours land together with API mode and none of them is exercised by a
board rollout: the reasoning-model branch of ``_call_openai`` (a missing
``gpt-6`` prefix made every call 400 with ``Unsupported parameter``),
``--api_reasoning_effort`` reaching the provider, and the ``v7`` prompt bundle.
The last test pins which early-stop knob sees a non-progress streak -- the
geometry guard cannot, and a run that trusts it alone keeps paying per call.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from methods._shared.early_stop import NoGeometryProgressGuard
from methods.llm_agent.policy import model_provider as MP
from methods.llm_agent.wrappers.prompts import (
    PCBWORLD_PROMPT_VERSIONS,
    get_pcbworld_prompts,
)


def _fake_openai_module(capture: dict) -> types.ModuleType:
    """Stand-in ``openai`` module that records the ``create()`` kwargs."""

    class _Completions:
        def create(self, **kwargs):
            capture.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="<think>ok</think>"))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )

    class _OpenAI:
        def __init__(self, **client_kwargs):
            capture["client_kwargs"] = client_kwargs
            self.chat = SimpleNamespace(completions=_Completions())

    mod = types.ModuleType("openai")
    mod.OpenAI = _OpenAI
    return mod


@pytest.fixture
def sent(monkeypatch) -> dict:
    cap: dict = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_module(cap))
    return cap


def test_gpt6_counts_as_a_reasoning_model(sent):
    text, counts, _ = MP._call_openai(
        "sys", "usr", "gpt-6-astra", 0.7, 128, reasoning_effort="low",
    )
    assert text == "<think>ok</think>" and counts == (11, 7)
    assert sent["max_completion_tokens"] == 128   # not max_tokens: 400 otherwise
    assert "max_tokens" not in sent
    assert sent["reasoning_effort"] == "low"
    assert "temperature" not in sent              # reasoning models reject it
    assert sent["messages"][0]["role"] == "developer"


def test_plain_chat_model_keeps_temperature(sent):
    MP._call_openai("sys", "usr", "gpt-4o", 0.7, 128, reasoning_effort="low")
    assert sent["max_tokens"] == 128
    assert "max_completion_tokens" not in sent
    assert "reasoning_effort" not in sent
    assert sent["temperature"] == 0.7
    assert sent["messages"][0]["role"] == "system"


def test_api_provider_forwards_reasoning_effort(monkeypatch):
    seen: dict = {}

    def _recorder(*args, **kwargs):
        seen.update(kwargs)
        seen["positional"] = args
        return "<think>ok</think>", (3, 4), {}

    monkeypatch.setitem(MP._API_CALLERS, "openai", _recorder)
    provider = MP.APIProvider(SimpleNamespace(
        api_provider="openai", api_model="gpt-6-astra", temperature=0.7,
        max_new_tokens=64, api_key=None, api_reasoning_effort="high",
    ))
    provider.generate([("sys", "usr")])
    assert seen["reasoning_effort"] == "high"      # declared in schema only, once
    assert seen["positional"][2:5] == ("gpt-6-astra", 0.7, 64)


def test_v7_is_discovered_and_mentions_no_finish():
    assert "v7" in PCBWORLD_PROMPT_VERSIONS
    bundle = get_pcbworld_prompts("v7")
    texts = [
        bundle.PCBWORLD_SYSTEM_PROMPT, bundle.PCBWORLD_USER_PROMPT,
        bundle.PCBWORLD_USER_PROMPT_NO_HIS, bundle.PCBWORLD_TEMPLATE,
        bundle.PCBWORLD_TEMPLATE_NO_HIS,
        bundle.get_state_format_desc("sexpr"),
        bundle.get_state_format_desc("xml"),
    ]
    assert all(isinstance(t, str) and t for t in texts)
    # v7 exists for --masking_rule default_no_finish: naming the action there
    # drives a start_route -> finish -> start_route loop.
    assert "finish" not in "\n".join(texts).lower()


def test_geometry_guard_cannot_see_steps_without_board_state():
    """parse-fail / mask-reject / idle carry no board-state measures.

    ``--early_stop_no_geometry_progress`` therefore never fires on such a
    streak; ``--early_stop_no_progress`` is the knob that does.
    """
    guard = NoGeometryProgressGuard(threshold=2)
    for _ in range(5):
        assert guard.update(info={"idle": True}) is False
        assert guard.update(info={"parse_invalid": True}) is False
    assert guard.count == 0

    valid = {"track_count": 3, "via_count": 1, "wirelength": 10.0,
             "ratsnest_reduction": 0.5}
    assert guard.update(info=valid) is False      # first: baseline
    assert guard.update(info=dict(valid)) is False
    assert guard.update(info=dict(valid)) is True  # 2 unchanged steps >= threshold
