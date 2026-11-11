"""The LLM env is built from one EnvConfig: every env flag of the LLM eval parser reaches
the PCBWorld each Ray worker builds, the obs knobs share RL's ``env:`` defaults, and a
prompt / masking pair that disagree on the actions refuses to build."""
from __future__ import annotations

import dataclasses
import typing

import pytest


def _worker_env_config(monkeypatch, argv):
    """Parse ``argv`` with the batch LLM parser, build the vec env (Ray stubbed) and one
    worker from the ctor kwargs it hands out; return (the EnvConfig that worker's PCBWorld
    received, the worker)."""
    import pcb_world.core.env as env_mod
    from eval.args import build_eval_parser
    from methods.llm_agent.rollout import pcbworld as R
    from methods.llm_agent.wrappers.adapter import KiCadLLMWrapper
    from pcb_world.vec.backends.ray import RayVecBackend

    ctor = {}
    monkeypatch.setattr(RayVecBackend, "__init__",
                        lambda self, *, worker_ctor_kwargs, **_kw: ctor.update(worker_ctor_kwargs))
    built = {}
    monkeypatch.setattr(env_mod, "PCBWorld", lambda **kw: built.update(kw))
    R._build_envs(build_eval_parser().parse_args(argv))
    worker = KiCadLLMWrapper(board_path="b.kicad_pcb", seed=0, **ctor)
    return built["env_config"], worker


def _non_default_argv():
    """A non-default value for every parsed dest that is a flat EnvConfig key (what
    ``llm_env_config`` reads back), plus the bespoke env flags. Was: ``--drc_config_path``
    was dropped by the hand-listed worker kwargs, so a board that needed the YAML
    fallback failed with 'no drc_config_path was given'."""
    from configs.loader import _unwrap_optional
    from configs.loader.schema import _ENV_FLAT_KEYS, EnvConfig, LLMConfig, RewardOverrides
    from eval.args import build_eval_parser

    defaults = LLMConfig().env.to_env_kwargs()
    hints = {**typing.get_type_hints(EnvConfig), **typing.get_type_hints(RewardOverrides)}
    meta = {f.name: f.metadata for c in (EnvConfig, RewardOverrides) for f in dataclasses.fields(c)}
    argv, expect = [], {}
    dests = set(vars(build_eval_parser().parse_args([])))
    # corner_mode: below
    for name in sorted(dests & _ENV_FLAT_KEYS - {"corner_mode"}):
        default, (tp, _) = defaults[name], _unwrap_optional(hints[name])
        if tp is bool:
            argv.append(f"--no-{name}" if default else f"--{name}")
            expect[name] = not default
            continue
        if name == "masking_rule":
            value = "strict"                                   # a real rule matching v5
        elif "choices" in meta[name]:
            value = next(c for c in meta[name]["choices"] if c != default)
        elif tp is int:
            value = default + 1
        elif tp is float:
            value = 0.125
        else:
            value = f"{name}.yaml"
        argv += [f"--{name}", str(value)]
        expect[name] = value
    argv += ["--corner_mode", "90", "--no_drc_tokens", "--state_format", "xml", "--prompt_version", "v5"]
    expect.update(corner_mode=90, emit_drc_tokens=False)
    return argv, expect


def test_every_env_flag_reaches_the_worker_env(monkeypatch):
    argv, expect = _non_default_argv()
    cfg, worker = _worker_env_config(monkeypatch, argv)
    got = cfg.to_env_kwargs()
    assert {k: got[k] for k in expect} == expect
    assert (worker.state_format, worker.prompt_version) == ("xml", "v5")


def test_llm_env_is_the_shared_env_except_the_override_list():
    """The LLM eval and training envs are configs/env.yaml's env except the LLM_ENV_OVERRIDES
    keys, so a shared rule changed there reaches the LLM programs unchanged."""
    from configs.loader import schema

    rl = schema.EnvConfig().to_env_kwargs()
    for llm_env, section in ((schema.LLMConfig().env, schema._LLM["env"]),
                             (schema.LLMTrainConfig().env, schema._LLM_TRAIN)):
        llm = llm_env.to_env_kwargs()
        assert {k for k in llm if llm[k] != rl[k]} <= schema.LLM_ENV_OVERRIDES
        assert set(section) & schema._ENV_FLAT_KEYS <= schema.LLM_ENV_OVERRIDES


def test_llm_section_setting_a_shared_env_key_raises():
    """A shared env key in an LLM section with an override list refuses to load."""
    from configs.loader.schema import EnvConfig, LLMEnvConfig, _llm_env_config

    with pytest.raises(ValueError, match="shared env keys"):
        _llm_env_config(EnvConfig(), {"max_steps": 20}, LLMEnvConfig, "llm.env", frozenset())


def test_a_per_run_shared_env_value_is_a_reported_deviation(capsys):
    """A CLI value for a shared env key is taken, printed as a deviation and listed by
    env_deviations (what the run record stores); a default run has none."""
    from configs.loader.schema import EnvConfig, env_deviations
    from eval.args import llm_env_config, resolve_eval_args

    assert env_deviations(llm_env_config(resolve_eval_args([]))) == {}
    run = resolve_eval_args(["--max_steps", "512"])
    assert env_deviations(llm_env_config(run)) == {"max_steps": (EnvConfig().max_steps, 512)}
    assert "DEVIATION from the shared env (configs/env.yaml): max_steps = 512" in capsys.readouterr().out


def test_llm_defaults_are_latest_with_pcbworld_masking():
    from configs.loader.schema import LLMConfig, LLMTrainConfig
    from methods.llm_agent.wrappers.prompts import check_prompt_masking

    for env, prompt in ((LLMConfig().env, LLMConfig().prompt.prompt_version),
                        (LLMTrainConfig().env, LLMTrainConfig().prompt_version)):
        assert (env.masking_rule, prompt) == ("pcbworld_masking", "latest")
        check_prompt_masking(prompt, env.masking_rule)


def test_unknown_flat_env_key_raises():
    from configs.loader.schema import EnvConfig
    with pytest.raises(ValueError, match="not EnvConfig fields"):
        EnvConfig().with_flat({"masking_rul": "strict"})


def test_vec_env_refuses_a_prompt_masking_mismatch(monkeypatch):
    from configs.loader.schema import LLMConfig
    from methods.llm_agent.wrappers.factory import KiCadLLMVecEnv
    from pcb_world.vec.backends.ray import RayVecBackend

    monkeypatch.setattr(RayVecBackend, "__init__", lambda self, **_kw: pytest.fail("built"))
    env = LLMConfig().env.with_flat({"masking_rule": "strict"})
    with pytest.raises(ValueError, match="'v8' is written for the actions"):
        KiCadLLMVecEnv(env_config=env, state_format="sexpr", prompt_version="v8",
                       prompt_guidelines=True, seed=0, env_num=1, group_n=1,
                       resources_per_worker={}, board_path="b.kicad_pcb")


def test_rollout_manager_renders_the_prompt_the_env_was_built_for():
    """One source: the manager takes the prompt version from the vec env (whose obs
    elements and masking were checked for it), not from a second config read."""
    from types import SimpleNamespace

    from methods.llm_agent.training.manager import KiCadLLMRolloutManager

    from configs.loader.schema import EnvConfig

    envs = SimpleNamespace(prompt_version="v5", prompt_guidelines=True, state_format="sexpr",
                           env_config=EnvConfig())
    config = {"env": {"pcbworld": {"prompt_version": "v2"}}}
    assert KiCadLLMRolloutManager(envs, None, config)._prompts.version == "v5"


@pytest.mark.parametrize("corner", [45, 90])
def test_rollout_manager_scores_at_the_angle_the_env_routes(corner):
    """Episode-end scoring follows the workers' corner_mode; an explicit
    ``env.pcbworld.eval_check_angle`` may restate it, a differing one raises."""
    from types import SimpleNamespace

    from configs.loader.schema import EnvConfig
    from methods.llm_agent.training.manager import KiCadLLMRolloutManager

    envs = SimpleNamespace(prompt_version="v5", prompt_guidelines=True, state_format="sexpr",
                           env_config=EnvConfig().with_flat({"corner_mode": corner}))
    assert KiCadLLMRolloutManager(envs, None, {})._eval_check_angle == corner
    same = {"env": {"pcbworld": {"eval_check_angle": corner}}}
    assert KiCadLLMRolloutManager(envs, None, same)._eval_check_angle == corner
    other = {"env": {"pcbworld": {"eval_check_angle": 135 - corner}}}
    with pytest.raises(ValueError, match="env.pcbworld.eval_check_angle"):
        KiCadLLMRolloutManager(envs, None, other)


def test_llm_env_config_raises_without_the_env_group():
    """Every env flag is read from the parsed args; a namespace without the ``env``
    group raises instead of filling the missing knobs with the LLM defaults."""
    import argparse

    from eval.args import llm_env_config

    with pytest.raises(AttributeError):
        llm_env_config(argparse.Namespace(no_drc_tokens=False))
