"""LLM prompt text: frozen legacy output, obs-element coverage, ``latest`` description.

The fixture ``tests/fixtures/llm_prompts/obs_all_knobs.json`` is a JSON observation with
every obs knob on (arc outline, simplified outline, per-net constraints) plus every optional
element (thru-hole pad, keepout polygon, track, via, DRC marker, action history). The legacy
v1–v7 converter + template output must stay byte-identical; ``legacy_sha256.json`` holds the
sha256 of each version's text as recorded before the obs-element table existed (the texts are
in the development history at ``SNAPSHOT_COMMIT``; a failing test writes the current render to
its pytest ``tmp_path`` for the diff).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from methods.llm_agent.wrappers import state_converter as sc
from methods.llm_agent.wrappers.prompts import (
    LATEST,
    PCBWORLD_PROMPT_VERSIONS,
    check_prompt_masking,
    get_pcbworld_prompts,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llm_prompts"
OBS = json.loads((FIXTURES / "obs_all_knobs.json").read_text())
LEGACY_SHA256 = json.loads((FIXTURES / "legacy_sha256.json").read_text())
SNAPSHOT_COMMIT = "0f4e246e79ab745879d927138b6b39507a36d5f1"   # development history: the frozen texts
LEGACY = [v for v in PCBWORLD_PROMPT_VERSIONS if v != LATEST]
FORMATS = {"sexpr": sc.format_state_split_sexpr, "xml": sc.format_state_split}
MASK = [False, True, True, True, True, True, False]


def _render(version: str, fmt: str) -> str:
    """The snapshot layout: full state, system prompt, user prompt with and without history."""
    b = get_pcbworld_prompts(version)
    static, dynamic = FORMATS[fmt](OBS, version, net_constraint_obs=True)
    va = sc.format_valid_actions(MASK)
    system = b.PCBWORLD_SYSTEM_PROMPT.format(state_format_desc=b.get_state_format_desc(fmt),
                                             board_static=static)
    user = b.PCBWORLD_USER_PROMPT_NO_HIS.format(dynamic_observation=dynamic, valid_actions=va)
    user_his = b.PCBWORLD_USER_PROMPT.format(
        current_step=6, valid_step_count=2,
        action_history="Step 1: net_select 1\nStep 2: start_route 20.0 5.0 1",
        dynamic_observation=dynamic, valid_actions=va, rejected_attempts="")
    return "\n".join(["===== state =====", sc.serialize_obs(OBS, fmt, version, net_constraint_obs=True),
                      "===== system =====", system, "===== user =====", user,
                      "===== user_history =====", user_his, ""])


def test_legacy_versions_are_v1_to_v8():
    assert LEGACY == [f"v{n}" for n in range(1, 9)]


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("version", LEGACY)
def test_legacy_prompt_text_is_frozen(version, fmt, tmp_path):
    text = _render(version, fmt)
    digest, frozen = hashlib.sha256(text.encode()).hexdigest(), LEGACY_SHA256[f"{version}_{fmt}"]
    if digest != frozen:
        now = tmp_path / f"{version}_{fmt}.txt"
        now.write_text(text, encoding="utf-8")
        pytest.fail(f"{version} {fmt} prompt text changed (sha256 {digest} != frozen {frozen}); current render: "
                    f"{now} — diff it against the frozen "
                    f"text (development history): git show {SNAPSHOT_COMMIT}:tests/fixtures/llm_prompts/{version}_{fmt}.txt")


def test_every_legacy_version_has_a_frozen_hash():
    assert sorted(LEGACY_SHA256) == sorted(f"{v}_{f}" for v in LEGACY for f in FORMATS)


def _uncovered(node, prefix: str = "") -> set[str]:
    """JSON paths under ``node`` with no OBS_ELEMENTS entry (``net_3`` -> ``net_*``,
    list item -> ``[]``); a dropped element's subtree is not visited."""
    if isinstance(node, list):
        return set().union(*(_uncovered(item, prefix + "[]") for item in node if isinstance(item, dict)))
    if not isinstance(node, dict):
        return set()
    missing: set[str] = set()
    for key, value in node.items():
        key = re.sub(r"_[0-9]+$", "_*", key)
        path = f"{prefix}.{key}" if prefix else key
        element = sc.OBS_ELEMENTS.get(path)
        if element is None:
            missing.add(path)
        elif element.kind == "render":
            missing |= _uncovered(value, path)
    return missing


def test_every_fixture_obs_key_has_a_render_or_drop_decision():
    assert _uncovered(OBS) == set()


def test_every_live_obs_key_has_a_render_or_drop_decision():
    """A new obs field in the env must get a render/drop entry before it ships."""
    from pcb_world.engine import engine_available
    if not engine_available():
        pytest.skip("KiCad engine not available")
    from configs.loader.schema import EnvConfig
    from pcb_world.core.env import PCBWorld

    cfg = dataclasses.replace(EnvConfig(), outline_obs="arc", simplify_outline=True,
                              net_constraint_obs=True, emit_drc_tokens=True,
                              action_history_len=2, max_steps=20)
    env = PCBWorld(board_path=str(FIXTURES.parent / "donut_cutout_board.kicad_pcb"), env_config=cfg)
    try:
        env.reset(seed=0)
        obs, *_ = env.step({"action_type": 0, "net_id": 1})
        obs, *_ = env.step({"action_type": 1, "x_mm": 20.0, "y_mm": 5.0, "layer": 1})
        obs, *_ = env.step({"action_type": 3, "x_mm": 20.0, "y_mm": 9.0, "routing_mode": 2})
    finally:
        env.close()
    assert "constraints" in obs["board_static"]["nets"]["net_1"]
    assert _uncovered(obs) == set()


def _latest(**env):
    from configs.loader.schema import LLMConfig

    return get_pcbworld_prompts(LATEST, env_config=LLMConfig().env.with_flat(env))


def test_v8_renders_per_net_constraints_and_v7_does_not():
    for fmt, split in FORMATS.items():
        v8_static, _ = split(OBS, "v8", net_constraint_obs=True)
        v7_static, _ = split(OBS, "v7", net_constraint_obs=True)
        tag = "(constraints " if fmt == "sexpr" else "<constraints "
        assert tag in v8_static and tag not in v7_static
        assert "0.250" in v8_static           # net 2's clearance
        # v8 = the v7 text plus the constraint lines only
        stripped = "\n".join(ln for ln in v8_static.splitlines() if tag not in ln)
        assert stripped == v7_static, fmt


def test_an_env_without_net_constraint_obs_renders_no_constraints():
    """With the knob off the env emits no per-net ``constraints``: the versions that render
    them render the rest of their elements (they used to index the missing key)."""
    obs = copy.deepcopy(OBS)
    for net in obs["board_static"]["nets"].values():
        del net["constraints"]
    for version, fmt in ((LATEST, "sexpr"), ("v8", "sexpr"), ("v8", "xml")):
        split = FORMATS[fmt]
        tag = "(constraints " if fmt == "sexpr" else "<constraints "
        off, _ = split(obs, version, net_constraint_obs=False)
        on, _ = split(OBS, version, net_constraint_obs=True)
        assert off == "\n".join(ln for ln in on.splitlines() if tag not in ln), (version, fmt)
        assert sc.serialize_obs(obs, fmt, version, net_constraint_obs=False).startswith(off)


def _tags(text: str) -> set[str]:
    return set(re.findall(r"\(([a-z_]+)", text))


def test_latest_description_mentions_every_tag_it_renders():
    desc = _latest().get_state_format_desc("sexpr")
    missing = {t for t in _tags(sc.serialize_obs(OBS, "sexpr", LATEST, net_constraint_obs=True))
               if not re.search(rf"(?<![a-z_]){t}(?![a-z_])", desc)}
    assert missing == set()


def test_latest_tells_net_end_may_release_an_unfinished_net():
    b = _latest()
    actions = b.format_valid_actions([False, False, True, False, False, False])
    assert "Allowed whenever no route is in progress, but final" in actions
    assert "finish" not in re.sub(r"finished", "", b.PCBWORLD_SYSTEM_PROMPT)
    assert "can be selected again" in _latest(net_reselect=True).format_valid_actions(
        [False, False, True, False, False, False])


def test_obs_elements_reads_the_version_case_insensitively_like_the_prompts():
    """``get_pcbworld_prompts`` accepts ``LATEST``; the converter must render the same
    elements for it instead of rejecting the version."""
    from configs.loader.schema import LLMConfig

    assert get_pcbworld_prompts("LATEST", env_config=LLMConfig().env).version == LATEST
    assert sc.obs_elements("LATEST") == sc.obs_elements(LATEST)
    assert sc.serialize_obs(OBS, "sexpr", "Latest", net_constraint_obs=True) == \
        sc.serialize_obs(OBS, "sexpr", LATEST, net_constraint_obs=True)


def test_obs_elements_rejects_an_unknown_version():
    with pytest.raises(ValueError, match="unknown prompt version"):
        sc.obs_elements("v99")


def test_content_hash_changes_with_a_template_the_element_set_or_the_env():
    latest = _latest()
    assert latest.content_hash == _latest().content_hash
    edited = dataclasses.replace(latest, PCBWORLD_SYSTEM_PROMPT=latest.PCBWORLD_SYSTEM_PROMPT + " ")
    assert edited.content_hash != latest.content_hash
    # Same templates, different rendered element set (legacy vs latest) -> different hash.
    as_legacy = dataclasses.replace(latest, version="v7")
    assert as_legacy.content_hash != latest.content_hash
    assert _latest(net_reselect=True).content_hash != latest.content_hash


@pytest.mark.parametrize("version,rule", [
    (LATEST, "pcbworld_masking"), (LATEST, "default"), (LATEST, "strict"),
    ("v8", "pcbworld_masking"), ("v7", "default_no_finish"), ("v5", "strict"),
    ("v1", "relaxed"), ("v4", "default"),
])
def test_matching_prompt_and_masking_pass(version, rule):
    check_prompt_masking(version, rule)


@pytest.mark.parametrize("version,rule", [
    ("v8", "default"), ("v8", "strict"), ("v5", "pcbworld_masking"), ("v7", "relaxed"),
    ("v5", "default_no_via"),
])
def test_mismatched_prompt_and_masking_raise(version, rule):
    with pytest.raises(ValueError, match="is written for the actions"):
        check_prompt_masking(version, rule)
