"""The KDD plan-only LLM recipe scripts emit free (x, y) waypoints, so the env they
build must take ``action_target: free`` — the shipped default (``candidates``) would
refuse every off-pool waypoint as ``not_a_candidate``."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import pcb_world.core.env as env_mod

LLM_EVAL = Path(__file__).resolve().parent.parent / "experiments" / "kdd" / "llm_eval"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_kdd_{name}", LLM_EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # the script's dataclasses resolve their module
    spec.loader.exec_module(mod)
    return mod


class _Captured(Exception):
    pass


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake(board_path, env_config):
        seen["cfg"] = env_config
        raise _Captured

    monkeypatch.setattr(env_mod, "PCBWorld", fake)
    return seen


def test_v8_standalone_env_takes_free_targets(captured):
    with pytest.raises(_Captured):
        _load("eval_plan_only_llm_v8_standalone")._build_env(Path("b.kicad_pcb"))
    assert captured["cfg"].to_env_kwargs()["action_target"] == "free"


@pytest.mark.parametrize("call", [
    lambda m: m.replay_and_save(Path("b.kicad_pcb"), "", Path("o.kicad_pcb")),
    lambda m: m.extract_initial_board_state(Path("b.kicad_pcb")),
])
def test_fewshot_prep_env_takes_free_targets(captured, call):
    with pytest.raises(_Captured):
        call(_load("prepare_plan_only_fewshot"))
    assert captured["cfg"].to_env_kwargs()["action_target"] == "free"
