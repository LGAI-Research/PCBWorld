"""Every env knob reaches the LLM prompt by a decided route (prompts/latest.py ENV_KNOBS).

A new EnvConfig field fails here until it is classified; a ``prompt`` knob must change the
generated prompt (or stop the build when the glossary has no sentence for the value), so a
rule the RL agent is held to cannot change without the LLM being told.
The glossary those sentences come from
may carry only reviewed ``origin`` marks.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from configs.loader.schema import EnvConfig, LLMConfig
from methods.llm_agent.wrappers.prompts import get_pcbworld_prompts, latest
from pcb_world.core.action_schema import ACT_IDLE, ACTION_NAMES

REPO = Path(__file__).resolve().parents[1]
ROUTES = {"prompt", "obs", "engine", "rl_only"}

# a value of each ``prompt`` knob other than the LLM default
OTHER_VALUE = {
    "max_steps": 512,
    "masking_rule": "default_no_finish",
    "net_reselect": True,
    "reward_rule": "drc_linear",
    "reward": {"via_penalty": 0.5},
    "drc_promoted_codes": [12, 13],
    "keep_routing_fraction": [0.2, 0.5],
    "force_walkaround": True,
    "mask_start_point": False,
    "offboard_mask": False,
    "action_target": "candidates",
    "net_constraint_obs": False,
    "emit_drc_tokens": False,
}


def _text(env: EnvConfig, guidelines: bool = True) -> str:
    b = get_pcbworld_prompts("latest", env_config=env, guidelines=guidelines)
    mask = [n in b.actions for n in ACTION_NAMES[:ACT_IDLE]]
    return "\n".join([b.PCBWORLD_SYSTEM_PROMPT, b.get_state_format_desc("sexpr"),
                      b.format_valid_actions(mask)])


def test_every_env_knob_has_a_route_to_the_llm():
    assert set(latest.ENV_KNOBS) == {f.name for f in dataclasses.fields(EnvConfig)}
    for name, (route, why) in latest.ENV_KNOBS.items():
        assert route in ROUTES and why.strip(), name


def test_every_prompt_knob_changes_the_prompt_or_stops_the_build():
    base_env = LLMConfig().env
    base = _text(base_env)
    prompt_knobs = {k for k, (route, _) in latest.ENV_KNOBS.items() if route == "prompt"}
    assert prompt_knobs == set(OTHER_VALUE)
    for name, value in OTHER_VALUE.items():
        env = (dataclasses.replace(base_env, reward=dataclasses.replace(base_env.reward, **value))
               if name == "reward" else base_env.with_flat({name: value}))
        try:
            text = _text(env)
        except ValueError as e:
            assert "glossary" in str(e), (name, e)
            continue
        assert text != base, f"{name}={value!r} leaves the latest prompt unchanged"


def _rules(sub: str) -> list[str]:
    return sorted(p.stem for d in (REPO / "configs" / sub, REPO / "experiments/kdd/configs" / sub)
                  if d.is_dir() for p in d.glob("*.yaml"))


@pytest.mark.parametrize("masking_rule", _rules("masking"))
@pytest.mark.parametrize("net_reselect", [False, True])
@pytest.mark.parametrize("action_target", ["free", "candidates"])
def test_latest_builds_for_every_shipped_masking_rule(masking_rule, net_reselect, action_target):
    """Every masking rule in the tree has glossary sentences for the actions it grants."""
    env = LLMConfig().env.with_flat({"masking_rule": masking_rule, "net_reselect": net_reselect,
                                     "action_target": action_target})
    for guidelines in (True, False):
        text = _text(env, guidelines)
        assert ("# Routing Guidelines" in text) is guidelines


@pytest.mark.parametrize("reward_rule", _rules("reward"))
def test_latest_describes_each_reward_rule_or_names_the_missing_sentence(reward_rule):
    env = LLMConfig().env.with_flat({"reward_rule": reward_rule})
    try:
        text = _text(env)
    except ValueError as e:
        assert "glossary" in str(e)
        return
    assert "the score rewards:" in text


def test_legacy_versions_keep_their_guidelines():
    with pytest.raises(ValueError, match="guidelines"):
        get_pcbworld_prompts("v8", guidelines=False)
    with pytest.raises(ValueError, match="EnvConfig"):
        get_pcbworld_prompts("latest")


@pytest.mark.parametrize("potential", [
    {"unconnected_penalty": 1.0, "drc_severity_mode": "errors_and_promoted"},   # default completion, linear DRC
    {**{k: v for k, v in yaml.safe_load((REPO / "configs/reward/pcbworld_reward.yaml")
                                        .read_text())["potential"].items()
        if k != "wirelength_bbox_normalize"}},                          # raw-length wirelength
], ids=["minimal", "no_bbox_normalize"])
def test_a_reward_rule_relying_on_potential_defaults_stops_the_build(tmp_path, potential):
    """An argument the rule leaves out is read as its PotentialReward default (as the env
    does); a default the glossary cannot describe stops the build like a set value."""
    rule = tmp_path / "minimal_rule.yaml"
    rule.write_text(yaml.safe_dump({"name": "minimal_rule", "mode": "per_step",
                                    "potential": potential}))
    with pytest.raises(ValueError, match="glossary"):
        _text(LLMConfig().env.with_flat({"reward_rule": str(rule)}))


@pytest.mark.parametrize("forced", [0, 1, 2])
def test_the_forced_routing_mode_env_var_reaches_the_prompt_or_stops_the_build(monkeypatch, forced):
    """PCBWORLD_FORCE_ROUTING_MODE forces the LLM's mode like force_walkaround does."""
    from pcb_world.core import constraints
    from pcb_world.core.action_schema import MODE_WALKAROUND

    monkeypatch.setattr(constraints, "FORCE_ROUTING_MODE", forced)
    env = LLMConfig().env.with_flat({"force_walkaround": False})
    if forced != MODE_WALKAROUND:
        with pytest.raises(ValueError, match="glossary"):
            _text(env)
        return
    forced_line = latest.glossary()["modes"]["forced_walkaround"]["text"]
    b = get_pcbworld_prompts("latest", env_config=env)
    assert forced_line in b.PCBWORLD_SYSTEM_PROMPT and forced_line in b.format_valid_actions(
        [n == "make_line" for n in ACTION_NAMES[:ACT_IDLE]])


REVIEWED_ORIGINS = {"v7", "v7-fixed", "new"}
NOT_A_SENTENCE = {"origin", "when", "element", "example", "v7"}   # glossary keys that hold no prompt text


def _sentences(node, path=(), origin=None):
    """``(path, origin)`` of every prompt string in the glossary: the ``origin`` mark of the
    node that holds it, else of the nearest group above (None when there is none)."""
    if isinstance(node, dict):
        origin = node.get("origin", origin)
        for key, value in node.items():
            if key not in NOT_A_SENTENCE:
                yield from _sentences(value, (*path, key), origin)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _sentences(value, (*path, i), origin)
    elif isinstance(node, str):
        yield "/".join(map(str, path)), origin


def test_a_sentence_takes_its_own_mark_or_its_groups():
    node = {"a": {"origin": "new", "line": "x", "b": {"origin": "unreviewed", "text": "y", "when": "k"}},
            "c": [{"text": "no mark"}, {"origin": "v7", "text": "z"}]}
    assert list(_sentences(node)) == [("a/line", "new"), ("a/b/text", "unreviewed"),
                                      ("c/0/text", None), ("c/1/text", "v7")]


def test_every_glossary_sentence_carries_a_reviewed_origin_mark():
    """A draft written with the generator (``origin: unreviewed``) is reviewed before it
    lands — the prompt sends every glossary sentence to the agent as a rule of the env —
    and a sentence with no mark, or a misspelt one, does not pass as reviewed."""
    sentences = list(_sentences(latest.glossary()))
    assert sentences, "no sentence found: the walk no longer matches the glossary's shape"
    not_reviewed = sorted(f"{path} ({origin})" for path, origin in sentences
                          if not isinstance(origin, str) or origin not in REVIEWED_ORIGINS)
    assert not not_reviewed, f"glossary.yaml sentences without a reviewed origin mark: {not_reviewed}"
