"""Prompt v7 = v6 without the ``finish`` action (for ``default_no_finish`` masking).

``test_llm_api_wiring.py`` checks that v7 never names ``finish``; this pins the
other half of the contract: the check is not vacuous (v6 does name it) and v7
still offers every other routing action v6 offers.
"""
import re

from methods.llm_agent.wrappers.action_converter import ACTION_TYPE_MAP
from methods.llm_agent.wrappers.prompts import get_pcbworld_prompts

_FIELDS = ("PCBWORLD_SYSTEM_PROMPT", "PCBWORLD_USER_PROMPT_NO_HIS",
           "PCBWORLD_USER_PROMPT", "PCBWORLD_TEMPLATE_NO_HIS", "PCBWORLD_TEMPLATE")


def _text(version: str) -> str:
    bundle = get_pcbworld_prompts(version)
    return "\n".join(getattr(bundle, f) for f in _FIELDS)


def test_v6_names_finish():
    assert _names("finish", _text("v6"))


def _names(action: str, text: str) -> bool:
    """Whole-word match, so a name inside a longer identifier does not count."""
    return re.search(rf"(?<![A-Za-z_]){re.escape(action)}(?![A-Za-z_])", text) is not None


def test_v7_keeps_every_other_action_of_v6():
    v6, v7 = _text("v6"), _text("v7")
    offered = [a for a in ACTION_TYPE_MAP if a != "finish" and _names(a, v6)]
    assert offered                                  # the comparison has content
    assert [a for a in offered if not _names(a, v7)] == []
