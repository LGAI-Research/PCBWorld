"""The LLM ``latest`` prompt shows exactly the observation the RL tokenizer reads.

The tokenizer is built from the RL policy defaults (configs/rl.yaml) and walks the all-knobs
fixture observation through a recording dict, which yields the obs paths it reads. The paths
``latest`` renders come from ``state_converter.OBS_ELEMENTS``. The two sets must match apart
from ``state_converter.PARITY_EXCEPTIONS`` (each with its reason), so an obs field the RL agent
starts to read reaches the LLM prompt in the same change, and the reverse.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
from pathlib import Path

from methods.llm_agent.wrappers import state_converter as sc

OBS = json.loads((Path(__file__).resolve().parent / "fixtures" / "llm_prompts"
                  / "obs_all_knobs.json").read_text())


def _key(k) -> str:
    return re.sub(r"_[0-9]+$", "_*", str(k))


class _Recording(dict):
    """A dict that records the JSON path of every key read through it (nested dicts and
    list items are wrapped on the way out)."""

    def __init__(self, data, path, seen):
        super().__init__(data)
        self._path, self._seen = path, seen

    def _out(self, k, v):
        path = f"{self._path}.{_key(k)}" if self._path else _key(k)
        self._seen.add(path)
        if isinstance(v, dict):
            return _Recording(v, path, self._seen)
        if isinstance(v, list):
            return [_Recording(x, path + "[]", self._seen) if isinstance(x, dict) else x for x in v]
        return v

    def __getitem__(self, k):
        return self._out(k, dict.__getitem__(self, k))

    def get(self, k, default=None):
        return self._out(k, dict.__getitem__(self, k)) if k in self else default

    def items(self):
        return [(k, self._out(k, v)) for k, v in dict.items(self)]

    def values(self):
        return [self._out(k, v) for k, v in dict.items(self)]


def _rl_read_paths() -> set[str]:
    from configs.loader.schema import RLPolicyConfig
    from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer

    params = inspect.signature(BatchedStateTokenizer).parameters
    policy = {f.name: getattr(RLPolicyConfig(), f.name) for f in dataclasses.fields(RLPolicyConfig)
              if f.name in params}
    tok = BatchedStateTokenizer(**{**policy, "d_model": 16, "n_freq": 4}).eval()
    seen: set[str] = set()
    tok._walk_dict([_Recording(OBS, "", seen)])
    return seen


def _leaves(paths: set[str]) -> set[str]:
    """Paths with no recorded child (a container read only to reach its children drops out)."""
    return {p for p in paths if not any(q.startswith(p + ".") or q.startswith(p + "[]")
                                        for q in paths)}


def _under(paths: set[str], roots) -> set[str]:
    """``paths`` with each one inside an exception subtree replaced by that root."""
    out = set()
    for p in paths:
        root = next((r for r in roots if p == r or p.startswith(r + ".") or p.startswith(r + "[]")), p)
        out.add(root)
    return out


def test_latest_renders_what_the_rl_tokenizer_reads():
    llm_only, rl_only = sc.PARITY_EXCEPTIONS["llm_only"], sc.PARITY_EXCEPTIONS["rl_only"]
    rl = _under(_leaves(_rl_read_paths()), rl_only)
    llm = _under(_leaves(set(sc.obs_elements("latest"))), llm_only)
    assert rl - llm == set(rl_only), "read by the RL tokenizer, not rendered for the LLM"
    assert llm - rl == set(llm_only), "rendered for the LLM, not read by the RL tokenizer"


def test_parity_exceptions_carry_a_reason_and_name_real_elements():
    for side in ("llm_only", "rl_only"):
        for path, reason in sc.PARITY_EXCEPTIONS[side].items():
            assert path in sc.OBS_ELEMENTS and reason.strip(), path


def test_latest_skips_rule_area_keepouts_as_the_rl_tokenizer_does():
    keepout = next(o for o in OBS["board_static"]["obstacles"].values() if o["shape"] == "polygon")
    plain = next(o for o in OBS["board_static"]["obstacles"].values() if o["shape"] != "polygon")
    latest = sc.serialize_obs(OBS, "sexpr", "latest", net_constraint_obs=True)
    v8 = sc.serialize_obs(OBS, "sexpr", "v8", net_constraint_obs=True)
    assert plain["id"] in latest and keepout["id"] not in latest
    assert keepout["id"] in v8


def test_latest_shows_drc_and_closed_nets_in_the_dynamic_part():
    static, dynamic = sc.format_state_split_sexpr(OBS, "latest", net_constraint_obs=True)
    assert "(drc_violations" in dynamic and "(closed_nets 3)" in dynamic
    assert "drc_violations" not in static and "board_constraints" not in static
    assert "via_drill" not in static


def test_latest_has_no_xml_form():
    """The glossary describes the S-expression syntax only."""
    import pytest

    for fn in (sc.format_state, sc.format_state_split):
        with pytest.raises(ValueError, match="sexpr state format only"):
            fn(OBS, "latest", net_constraint_obs=True)


def _drc_lines(obs, **kw) -> list[str]:
    _, dynamic = sc.format_state_split_sexpr(obs, "latest", net_constraint_obs=True, **kw)
    return [ln.strip() for ln in dynamic.splitlines() if ln.strip().startswith("(drc ")]


def test_latest_drc_layer_uses_the_prompt_layer_numbers():
    """DRC markers carry KiCad's PCB_LAYER_ID (F.Cu 0, B.Cu 2, inner 4, 6, …; -1 = no
    layer); the prompt numbers copper 1 = Top … N = Bottom, ``-`` for no copper layer."""
    obs = json.loads(json.dumps(OBS))
    obs["board_static"]["copper_layers"] = 4
    row = obs["drc_violations"][0]
    obs["drc_violations"] = [{**row, "layer": lid} for lid in (-1, 0, 4, 6, 2, 44)]
    assert [ln.split()[3] for ln in _drc_lines(obs)] == ["-", "1", "2", "3", "4", "-"]


def test_latest_drops_drc_violations_when_the_env_emits_no_drc_tokens():
    """``emit_drc_tokens: false`` leaves the obs list empty and the description silent:
    the element is not rendered at all (an empty ``(drc_violations)`` would read as a
    clean board)."""
    on = sc.serialize_obs(OBS, "sexpr", "latest", net_constraint_obs=True)
    off = sc.serialize_obs({**OBS, "drc_violations": []}, "sexpr", "latest",
                           net_constraint_obs=True, emit_drc_tokens=False)
    assert "(drc_violations" in on and "(drc_violations" not in off
    block = on[on.index("(drc_violations"):on.index("(closed_nets")]
    assert off == on.replace(block, "")
