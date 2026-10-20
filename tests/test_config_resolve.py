"""``configs.loader.resolve``: one resolve per config object, layers in order, the scope × mode table.

1. **Layers** — DEFAULTS < mode defaults < CKPT < recipe < cli ; code, with the chain of
   every layer that set a key; the result is frozen.
2. **Scope × mode** — resume pins experiment keys (``resume_mutable`` excepted), eval pins
   them (``eval_override`` excepted), arch keys equal the checkpoint in every ckpt mode, cli
   and code may not both set a key; every message names the fix.
3. **Training entry point** (``parse_train_args``) — ``--init-ckpt`` takes the checkpoint's arch
   keys and refuses a different one (also those the old ``INIT_CKPT_HARD_KEYS`` list skipped),
   the fine-tune optimizer defaults yield to an explicit flag, ``--resume`` pins experiment keys,
   ``--recipe`` sits under the command line, ``--expect`` stops a run whose config differs.
4. **Eval env** (``eval.rollout.rl.resolve_eval_env``) — train-only fields stay off, only
   ``eval_override`` fields may change, the DRC-off layer conflicts with a CLI DRC weight.
5. **Every schema field declares a scope.**
"""
from __future__ import annotations

import dataclasses

import pytest
import yaml

from configs.loader.knob import Policy, field_policies
from configs.loader.resolve import ConfigError, Layer, resolve
from tests.helpers.ckpt_golden import BOARD

POL = {"a": Policy("arch", False, False, "T.a"), "e": Policy("experiment", False, False, "T.e"),
       "m": Policy("experiment", True, False, "T.m"), "o": Policy("experiment", False, True, "T.o"),
       "r": Policy("runtime", False, False, "T.r")}
DEFAULTS = Layer("defaults", {"a": 1, "e": 1, "m": 1, "o": 1, "r": 1})
CKPT = Layer("ckpt", {"a": 2, "e": 2, "m": 2, "o": 2, "r": 2, "extra": 9}, sources={"e": "legacy"})


def _resolve(mode, *user, ckpt=CKPT):
    return resolve(mode, policies=POL, defaults=[DEFAULTS], ckpt=None if mode == "fresh" else ckpt,
                   user=list(user))


# ===================================================================
# 1. layers
# ===================================================================
def test_layer_order_and_provenance_chain():
    run = resolve("init-ckpt", policies=POL,
                  defaults=[DEFAULTS, Layer("defaults", {"e": 5}, where={"e": "mode default"})],
                  ckpt=CKPT, user=[Layer("recipe:r.yaml", {"e": 6, "r": 6}), Layer("cli", {"e": 7}),
                                   Layer("code:x.py", {"r": 8})])
    assert run.a == 2 and run.provenance["a"].source == "ckpt"          # init-ckpt: arch from ckpt
    assert [(c["source"], c["value"]) for c in run.provenance["e"].chain] == [
        ("defaults", 1), ("defaults", 5), ("recipe:r.yaml", 6), ("cli", 7)]
    assert (run.e, run.provenance["e"].source) == (7, "cli")
    assert (run.r, run.provenance["r"].source) == (8, "code:x.py")
    assert "extra" not in vars(run)                                      # ckpt-only keys are not config
    with pytest.raises(AttributeError, match="frozen"):
        run.e = 0


def test_ckpt_sources_carry_into_the_provenance():
    run = _resolve("resume")
    assert run.provenance["e"].source == "legacy" and run.provenance["a"].source == "ckpt"
    assert run.ckpt_config() == {"schema": 2, "mode": "resume", "values": vars(run),
                                 "sources": {k: e.source for k, e in run.provenance.items()}}


def test_unknown_and_undeclared_keys_raise():
    with pytest.raises(ConfigError, match="this config does not have"):
        _resolve("fresh", Layer("cli", {"nope": 1}))
    with pytest.raises(ConfigError, match="without a declared scope"):
        resolve("fresh", policies={}, defaults=[DEFAULTS])


# ===================================================================
# 2. scope x mode
# ===================================================================
@pytest.mark.parametrize("mode,key,ok", [
    ("fresh", "a", True), ("fresh", "e", True),
    ("resume", "a", False), ("resume", "e", False), ("resume", "m", True), ("resume", "o", False),
    ("resume", "r", True),
    ("init-ckpt", "a", False), ("init-ckpt", "e", True),
    ("eval", "a", False), ("eval", "e", False), ("eval", "m", False), ("eval", "o", True),
    ("eval", "r", True),
])
def test_scope_mode_table(mode, key, ok):
    layer = Layer("cli", {key: 3})
    if ok:
        assert getattr(_resolve(mode, layer), key) == 3
    else:
        with pytest.raises(ConfigError, match=f"may not set '{key}'=3 in mode {mode}"):
            _resolve(mode, layer)


def test_a_value_equal_to_the_checkpoint_is_allowed_and_recorded():
    run = _resolve("resume", Layer("cli", {"a": 2, "e": "2"}))
    assert run.provenance["e"].source == "cli" and len(run.provenance["e"].chain) == 3


@pytest.mark.parametrize("a,b,same", [
    (None, False, True), (None, "none", True), ("off", None, True), ("", None, True),
    (None, 0, False), (None, 0.0, False), ("off", 0, False), ("none", 0.0, False),
    (45, "45", True), (0.5, 0.25, False),
])
def test_same_value_reads_off_spellings_as_one_but_never_a_number(a, b, same):
    from configs.loader.resolve import same_value

    assert same_value(a, b) is same and same_value(b, a) is same


def test_violation_messages_name_the_fix():
    with pytest.raises(ConfigError, match=r"eval_override=False\); declare eval_override=True"):
        _resolve("eval", Layer("cli", {"e": 3}))
    with pytest.raises(ConfigError, match="resume_mutable=False"):
        _resolve("resume", Layer("cli", {"e": 3}))
    with pytest.raises(ConfigError, match="the weights depend on it"):
        _resolve("eval", Layer("cli", {"a": 3}))


def test_cli_and_code_may_not_set_one_key():
    with pytest.raises(ConfigError, match=r"\['r'\] set by both the command line and code:w.py"):
        _resolve("fresh", Layer("cli", {"r": 3}), Layer("code:w.py", {"r": 4}))


@pytest.mark.parametrize("source", ["legacy", "legacy:proxy", "legacy:partial"])
def test_init_ckpt_estimated_arch_value_yields_unless_strict(source):
    """init-ckpt: an arch value the checkpoint only estimates (legacy*) yields to USER with a
    warning; a recorded or derived one, or any under strict_ckpt, still raises."""
    ckpt = Layer("ckpt", {"a": 2, "e": 2}, sources={"a": source})
    with pytest.warns(UserWarning, match=r"a: the checkpoint's 2 is not recorded in it \(estimated/inferred, source legacy.*cli 3"):
        run = _resolve("init-ckpt", Layer("cli", {"a": 3}), ckpt=ckpt)
    assert (run.a, run.provenance["a"].source) == (3, "cli")
    with pytest.raises(ConfigError, match="may not set 'a'=3"):
        resolve("init-ckpt", policies=POL, defaults=[DEFAULTS], ckpt=ckpt,
                user=[Layer("cli", {"a": 3})], strict_ckpt=True)
    for recorded in ({}, {"a": "derived"}):
        with pytest.raises(ConfigError, match="may not set 'a'=3"):
            _resolve("init-ckpt", Layer("cli", {"a": 3}), ckpt=Layer("ckpt", {"a": 2, "e": 2}, sources=recorded))
    with pytest.raises(ConfigError, match="may not set 'a'=3"):                  # not init-ckpt
        _resolve("resume", Layer("cli", {"a": 3}), ckpt=ckpt)


# ===================================================================
# 3. the training entry point
# ===================================================================
def _ckpt_file(tmp_path, generation=None, **args):
    from tests.helpers.ckpt_payload import CURRENT, save_ckpt

    return str(save_ckpt(tmp_path / "ckpt.pt", generation or CURRENT, **args))


def _train(*argv):
    from methods.rl_agent.training.train_ppo import parse_train_args

    return parse_train_args(["--board", BOARD, *argv])


def test_policy_net_select_defaults_on_with_an_off_form():
    """A fresh run learns net ordering unless told otherwise (every real run trained with it)."""
    assert _train().policy_net_select is True
    assert _train("--no-policy-net-select").policy_net_select is False


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_init_ckpt_takes_the_checkpoint_arch_and_refuses_another(tmp_path):
    ckpt = _ckpt_file(tmp_path, time_feature_cap=7777, obstacle_obs=True)
    run = _train("--init-ckpt", ckpt)
    assert run.resolve_mode == "init-ckpt"
    assert (run.time_feature_cap, run.obstacle_obs) == (7777, True)
    assert run.provenance["time_feature_cap"].source == "ckpt"
    # time_feature_cap was not in the old INIT_CKPT_HARD_KEYS: a different value trained silently
    with pytest.raises(ConfigError, match="may not set 'time_feature_cap'=5000 in mode init-ckpt"):
        _train("--init-ckpt", ckpt, "--time-feature-cap", "5000")
    with pytest.raises(ConfigError, match="'obstacle_obs'=False"):
        _train("--init-ckpt", ckpt, "--no-obstacle-obs")
    assert _train("--init-ckpt", ckpt, "--masking-rule", "default").masking_rule == "default"


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_init_ckpt_arch_estimate_yields_to_the_command_line(tmp_path):
    """A checkpoint older than a knob carries no key for it: its value is the migration table's
    estimate (source legacy), so a run that sets the knob differently takes its own value with a
    warning naming both. --strict-ckpt refuses such a checkpoint outright."""
    from configs.loader.ckpt import CkptConfigError

    ckpt = _ckpt_file(tmp_path, "d2b_2026_08")                         # predates obstacle_obs
    assert _train("--init-ckpt", ckpt).obstacle_obs is False
    with pytest.warns(UserWarning, match=r"obstacle_obs: the checkpoint's False is not recorded in it \(estimated/inferred, source legacy\).*cli True"):
        run = _train("--init-ckpt", ckpt, "--obstacle-obs")
    assert (run.obstacle_obs, run.provenance["obstacle_obs"].source) == (True, "cli")
    with pytest.raises(ConfigError, match="'shape_obs'=True in mode init-ckpt"):   # derived from weights
        _train("--init-ckpt", ckpt, "--shape-obs")
    with pytest.raises(CkptConfigError, match="--strict-ckpt"):
        _train("--init-ckpt", ckpt, "--obstacle-obs", "--strict-ckpt")


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_fine_tune_defaults_yield_to_an_explicit_flag(tmp_path, monkeypatch):
    from configs.loader.schema import FINETUNE_DEFAULTS, RLTrainConfig

    # the decision reads the parsed argv list, not the process command line
    monkeypatch.setattr("sys.argv", ["train.py", "--lr", "9", "--clip-eps", "9"])
    ckpt = _ckpt_file(tmp_path)
    run = _train("--init-ckpt", ckpt)
    assert (run.lr, run.clip_eps) == (FINETUNE_DEFAULTS["lr"], FINETUNE_DEFAULTS["clip_eps"])
    assert run.provenance["lr"].where == "configs/rl.yaml rl_train.finetune.lr"
    explicit = _train("--init-ckpt", ckpt, "--lr", "1e-3")
    assert (explicit.lr, explicit.clip_eps) == (1e-3, FINETUNE_DEFAULTS["clip_eps"])
    assert _train("--init-ckpt", ckpt, "--clip-eps=0.3").clip_eps == 0.3
    # the shared default, given on the command line, is still the user's choice
    same = _train("--init-ckpt", ckpt, "--lr", str(RLTrainConfig().lr))
    assert same.lr == RLTrainConfig().lr and same.provenance["lr"].source == "cli"
    assert _train().lr == RLTrainConfig().lr                    # fresh: no fine-tune defaults


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_resume_pins_experiment_keys(tmp_path):
    ckpt = _ckpt_file(tmp_path, gamma=0.97, iterations=100, net_reselect=False, obstacle_obs=False)
    run = _train("--resume", ckpt, "--iterations", "300")         # resume_mutable
    assert (run.gamma, run.iterations, run.resolve_mode) == (0.97, 300, "resume")
    assert run.provenance["gamma"].source == "ckpt"
    # was CLI-wins with two pinned keys (_RESUME_PINNED_ARGS): gamma changed silently
    with pytest.raises(ConfigError, match="may not set 'gamma'=0.99 in mode resume"):
        _train("--resume", ckpt, "--gamma", "0.99")
    # the two old pinned keys add no weights: a flip would change the task mid-run
    assert _train("--resume", ckpt, "--no-net-reselect").net_reselect is False
    with pytest.raises(ConfigError, match="'net_reselect'=True in mode resume"):
        _train("--resume", ckpt, "--net-reselect")
    with pytest.raises(ConfigError, match="'obstacle_obs'=True in mode resume"):
        _train("--resume", ckpt, "--obstacle-obs")
    # a zero weight is not "unset": the checkpoint's None (the rule's value) stays pinned
    with pytest.raises(ConfigError, match="may not set 'drc_penalty'=0.0 in mode resume"):
        _train("--resume", ckpt, "--drc-penalty", "0")


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_resume_keeps_the_checkpoint_runtime_keys_and_frees_them(tmp_path, monkeypatch):
    """A resume may name the board by another path (another checkout) — recorded in the chain.
    Its runtime keys come from the checkpoint (a hand resume kept writing to ./checkpoints with
    async_val / wandb off when they came from DEFAULTS) and the command line may change any of
    them; a 4-GPU run resumed on a 1-GPU node then stops at the update group, naming the flag."""
    torch = pytest.importorskip("torch")
    from methods.rl_agent.training.ddp import DDPUpdateGroup

    ckpt = _ckpt_file(tmp_path, board="/other/checkout/" + BOARD, update_gpus=4, device="cuda:3",
                      save_dir=str(tmp_path / "run"))
    run = _train("--resume", ckpt)
    assert run.board == BOARD
    assert [c["source"] for c in run.provenance["board"].chain] == ["defaults", "ckpt", "cli"]
    assert (run.update_gpus, run.device, run.save_dir) == (4, "cuda:3", str(tmp_path / "run"))
    assert run.provenance["update_gpus"].source == "ckpt"
    moved = _train("--resume", ckpt, "--update-gpus", "1", "--device", "cuda:0")
    assert (moved.update_gpus, moved.device) == (1, "cuda:0")
    assert [c["source"] for c in moved.provenance["update_gpus"].chain] == ["defaults", "ckpt", "cli"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="pass --update-gpus 1"):
        DDPUpdateGroup(run, None, None, "cuda", use_critic=True)


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_resume_can_turn_off_wandb_and_bf16(tmp_path):
    """A resume starts its runtime keys from the checkpoint: W&B / bf16 on there need an off flag."""
    ckpt = _ckpt_file(tmp_path, wandb=True, bf16=True)
    assert (_train("--resume", ckpt).wandb, _train("--resume", ckpt).bf16) == (True, True)
    run = _train("--resume", ckpt, "--no-wandb", "--no-bf16")
    assert (run.wandb, run.bf16) == (False, False)
    assert run.provenance["wandb"].source == "cli"


def _resume_changeable_booleans(algo):
    """``{dest: {value: option}}`` of every boolean option of the ``algo`` training parser whose
    dest a resume takes from the checkpoint and lets the command line change (runtime or
    ``resume_mutable``; the process options never come from a checkpoint)."""
    import argparse

    from configs.loader.resolve import PROCESS_OPTIONS
    from configs.loader.schema import GRPOConfig, PPOConfig
    from methods.rl_agent.training import train_grpo, train_ppo
    from methods.rl_agent.training.args import train_policies

    module, cfg = {"ppo": (train_ppo, PPOConfig()), "grpo": (train_grpo, GRPOConfig())}[algo]
    policies = train_policies(cfg)
    out: dict = {}
    for a in module.build_arg_parser()._actions:
        if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            out.setdefault(a.dest, {})[a.const] = a.option_strings[0]
    return {d: opts for d, opts in out.items() if d not in PROCESS_OPTIONS
            and (policies[d].scope == "runtime" or policies[d].resume_mutable)}


@pytest.mark.parametrize("algo", ["ppo", "grpo"])
def test_every_resume_changeable_boolean_has_both_forms(algo):
    """A resume starts these keys from the checkpoint, so a flag with only one form could never
    undo the other value stored there (--wandb / --bf16 had no off form)."""
    booleans = _resume_changeable_booleans(algo)
    assert {"wandb", "no_vecenv", "no_eval_greedy"} <= set(booleans)
    assert {d: sorted(opts) for d, opts in booleans.items()} == \
        {d: [False, True] for d in booleans}


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
@pytest.mark.parametrize("algo", ["ppo", "grpo"])
def test_resume_sets_every_runtime_boolean_either_way(tmp_path, algo):
    from configs.loader.schema import GRPOConfig, PPOConfig
    from methods.rl_agent.training import train_grpo, train_ppo
    from methods.rl_agent.training.args import resolve_train_args

    module, cfg = {"ppo": (train_ppo, PPOConfig()), "grpo": (train_grpo, GRPOConfig())}[algo]
    for dest, opts in _resume_changeable_booleans(algo).items():
        for value in (False, True):
            ckpt = _ckpt_file(tmp_path, drop=("gamma",) if algo == "grpo" else (), **{dest: not value})
            run = resolve_train_args(module.build_arg_parser(), cfg,
                                     ["--board", BOARD, "--resume", ckpt, opts[value]])
            assert getattr(run, dest) is value, (dest, opts[value])


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_resume_reads_the_old_inverted_training_flags(tmp_path):
    """A paper-era checkpoint stores the PPO switches under their old inverted dests
    (``no_truncation_bootstrap=False`` = bootstrap on): a resume continues with the trained
    values (source ``renamed``), not the current defaults (truncation bootstrap off)."""
    ckpt = _ckpt_file(tmp_path, "kdd_2026_05", no_truncation_bootstrap=False, no_norm_reward=True,
                      no_normalize_adv=True)
    run = _train("--resume", ckpt)
    assert (run.truncation_bootstrap, run.norm_reward, run.normalize_adv) == (True, False, False)
    assert {run.provenance[k].source for k in ("truncation_bootstrap", "norm_reward",
                                                "normalize_adv")} == {"renamed"}


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_resume_warns_about_checkpoint_keys_it_drops(tmp_path):
    """A stored key no training option carries would be dropped silently: the resume names it;
    the known-irrelevant ones (IGNORED_CKPT_ARGS) pass quietly."""
    import warnings

    from methods.rl_agent.training.args import IGNORED_CKPT_ARGS

    quiet = _ckpt_file(tmp_path, "kdd_2026_05", **{k: None for k in IGNORED_CKPT_ARGS})
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not options of this trainer")
        _train("--resume", quiet)
    stale = _ckpt_file(tmp_path, "kdd_2026_05", gone_knob=3)
    with pytest.warns(UserWarning, match=r"checkpoint keys \['gone_knob'\] are not options of this trainer"):
        _train("--resume", stale)


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_resume_iteration_comes_from_the_one_checkpoint_load(tmp_path, monkeypatch):
    """The resumed run's record name needs the checkpoint iteration: resolve loads the
    checkpoint once and keeps it (the trainer used to torch.load the file again)."""
    torch = pytest.importorskip("torch")
    from configs.loader.record import RUN_RECORD
    from methods.rl_agent.training.loop import PPOTrainer

    ckpt = _ckpt_file(tmp_path)
    torch.save({**torch.load(ckpt, weights_only=False), "iteration": 41}, ckpt)
    run = _train("--resume", ckpt, "--save-dir", str(tmp_path / "run"))
    assert dict(run.ckpt_meta) == {"iteration": 41}
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("checkpoint loaded twice"))
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.args, trainer.boards = run, [BOARD]
    trainer._write_run_record()
    assert trainer._run_record_path.name == "run_record.resume_iter42.yaml" != RUN_RECORD


def _weights_ckpt(path, generation):
    """A ``generation`` checkpoint holding real weights of its own token layout."""
    import torch

    from configs.loader.schema import RLPolicyConfig
    from tests.helpers.ckpt_payload import ckpt_config, ckpt_payload

    payload = ckpt_payload(generation)
    payload["policy_state_dict"] = RLPolicyConfig.from_checkpoint(ckpt_config(generation)).build().state_dict()
    torch.save(payload, path)
    return str(path)


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
@pytest.mark.parametrize("flag", ["--resume", "--init-ckpt"])
@pytest.mark.parametrize("generation", ["d2b_2026_06", "d2b_2026_08"])
def test_a_checkpoint_token_layout_is_the_runs(tmp_path, flag, generation):
    """--resume / --init-ckpt build the checkpoint's token layout (derived from its weights) and
    load its weights with nothing missing: the August run (no history_age projection) loaded
    into the current layout with the age keys waved through as "missing", the June run (3-term
    NET tokens) raised a size mismatch. A layout field is an arch key: no layer may change it."""
    torch = pytest.importorskip("torch")
    from configs.loader.schema import CHECKPOINT_LAYOUT_FIELDS, RLPolicyConfig
    from methods.rl_agent.training.loop import PPOTrainer
    from methods.rl_agent.training.train_ppo import parse_train_args
    from tests.helpers.ckpt_payload import fixture

    ckpt = _weights_ckpt(tmp_path / "ckpt.pt", generation)
    run = _train(flag, ckpt)
    layout = {k: fixture(generation)["expected"]["policy"][k] for k in CHECKPOINT_LAYOUT_FIELDS}
    assert {k: getattr(run, k) for k in CHECKPOINT_LAYOUT_FIELDS} == layout
    assert {run.provenance[k].source for k in CHECKPOINT_LAYOUT_FIELDS} == {"derived"}
    policy = RLPolicyConfig.from_namespace(run, use_critic=True).build()
    policy.load_state_dict(torch.load(ckpt, weights_only=False)["policy_state_dict"])   # strict
    if flag == "--init-ckpt":   # the trainer's own load path
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.args, trainer.device, trainer.policy = run, "cpu", policy
        trainer._init_from_ckpt()
    with pytest.raises(ConfigError, match="may not set 'legacy_action_history'=False"):
        parse_train_args(["--board", BOARD, flag, ckpt],
                         code=Layer("code:t.py", {"legacy_action_history": False}))
    fresh = _train()
    assert {k: (getattr(fresh, k), fresh.provenance[k].source) for k in CHECKPOINT_LAYOUT_FIELDS} == \
        dict.fromkeys(CHECKPOINT_LAYOUT_FIELDS, (False, "defaults"))


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_resume_refuses_a_save_dir_with_later_checkpoints(tmp_path):
    """Resuming an earlier checkpoint of a run in place would overwrite its later
    policy_iter_*.pt / policy_last.pt: refused before the run record is written."""
    torch = pytest.importorskip("torch")
    from methods.rl_agent.training.loop import PPOTrainer

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = run_dir / "policy_iter_40.pt"
    _ckpt_file(tmp_path)
    torch.save({**torch.load(tmp_path / "ckpt.pt", weights_only=False), "iteration": 40}, ckpt)
    trainer = PPOTrainer.__new__(PPOTrainer)

    def check(resume=ckpt):
        trainer.args = _train("--resume", str(resume), "--save-dir", str(run_dir))
        trainer._refuse_resume_overwrite()

    torch.save({"iteration": 40}, run_dir / "policy_last.pt")
    (run_dir / "policy_iter_x.pt").touch()             # a non-integer stem is not a checkpoint
    check()                                            # nothing past iteration 40: resumes
    (run_dir / "policy_iter_60.pt").touch()
    with pytest.raises(RuntimeError, match=r"overwrite later checkpoints.*policy_iter_60\.pt.*fresh --save-dir"):
        check()
    (run_dir / "policy_iter_60.pt").unlink()
    torch.save({"iteration": 45}, run_dir / "policy_last.pt")
    with pytest.raises(RuntimeError, match=r"policy_last\.pt \(iteration 45\)"):
        check()
    torch.save({**torch.load(ckpt, weights_only=False), "iteration": 45}, run_dir / "policy_last.pt")
    check(run_dir / "policy_last.pt")                  # resuming policy_last itself


def test_recipe_sits_under_the_command_line(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"gamma": 0.9, "lr": "2e-4", "masking_rule": "default"}))
    run = _train("--recipe", str(recipe), "--gamma", "0.8")
    assert (run.gamma, run.lr, run.masking_rule) == (0.8, 2e-4, "default")
    assert [c["source"] for c in run.provenance["gamma"].chain] == [
        "defaults", f"recipe:{recipe}", "cli"]
    assert run.provenance["lr"].source == f"recipe:{recipe}"
    recipe.write_text(yaml.safe_dump({"not_a_flag": 1}))
    with pytest.raises(ConfigError, match="not config options"):
        _train("--recipe", str(recipe))
    recipe.write_text(yaml.safe_dump({"coord_encoding": "nope"}))
    with pytest.raises(ConfigError, match="is not one of"):
        _train("--recipe", str(recipe))


@pytest.mark.parametrize("key", ["resume", "init_ckpt", "strict_ckpt", "expect"])
def test_a_recipe_may_not_set_a_process_option(tmp_path, key):
    """The resolve mode follows --resume / --init-ckpt before the recipe is read: a recipe value
    there was recorded but never acted on (a recipe 'resume' resolved a fresh run)."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({key: "x"}))
    with pytest.raises(ConfigError, match=rf"\['{key}'\] steer this process"):
        _train("--recipe", str(recipe))


def test_a_repeatable_option_on_the_command_line_is_refused():
    from configs.loader.cli import StrictArgumentParser
    from configs.loader.resolve import add_run_args, user_layers

    p = StrictArgumentParser()
    add_run_args(p)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("-v", action="count", default=0)
    for argv, dest in ((["--tag", "a", "--tag", "b"], "tag"), (["-v", "-v"], "v")):
        with pytest.raises(ConfigError, match=rf"\['{dest}'\] are repeatable options"):
            user_layers(p, p.parse_args(argv))
    assert user_layers(p, p.parse_args([]))[-1].values == {}


def test_eval_pipeline_recipe_sets_only_the_env_overrides(monkeypatch, tmp_path):
    from configs.loader.resolve import user_layers
    from eval import pipeline

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"max_steps": 64, "n_envs": 3}))
    monkeypatch.setattr("sys.argv", ["eval", "--ckpt", "p.pt", "--boards-dir", str(tmp_path), "--seed", "1",
                                     "--n-rollouts", "1", "--recipe", str(recipe), "--corner-mode", "90"])
    parser, args = pipeline._parse_args()
    with pytest.raises(ConfigError, match=r"\['n_envs'\] are not config options .* sets only \['corner_mode'"):
        user_layers(parser, args, keys=pipeline.ENV_OVERRIDE_DESTS)
    recipe.write_text(yaml.safe_dump({"max_steps": 64}))
    assert [dict(layer.values) for layer in user_layers(parser, args, keys=pipeline.ENV_OVERRIDE_DESTS)] == [
        {"max_steps": 64}, {"corner_mode": 90}]


def test_expect_mismatch_stops_the_run(tmp_path):
    expect = tmp_path / "expect.yaml"
    expect.write_text(yaml.safe_dump({"gamma": 0.9, "lr": {"value": 1e-3, "source": "cli"}}))
    assert _train("--expect", str(expect), "--gamma", "0.9", "--lr", "1e-3").gamma == 0.9
    with pytest.raises(ConfigError) as err:
        _train("--expect", str(expect), "--gamma", "0.995")
    assert "gamma: expected 0.9, resolved 0.995" in str(err.value)
    assert "lr: expected source 'cli', resolved from 'defaults'" in str(err.value)
    expect.write_text(yaml.safe_dump({"nope": 1}))
    with pytest.raises(ConfigError, match="nope: not a key of this config"):
        _train("--expect", str(expect))


def test_code_layer_sets_a_key_no_flag_has():
    from methods.rl_agent.training.train_ppo import parse_train_args

    run = parse_train_args(["--board", BOARD], code=Layer("code:t.py", {"obs_format": "json"}))
    assert (run.obs_format, run.provenance["obs_format"].source) == ("json", "code:t.py")


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_watcher_config_is_the_checkpoint_plus_its_own_pool_size(tmp_path):
    """The async-val watcher's namespace: the checkpoint's training config in mode eval, this
    node's --n-envs as a code layer (was a Namespace assignment after loading)."""
    from methods.rl_agent.training.args import resolve_ckpt_train_config

    ckpt = _ckpt_file(tmp_path, n_envs=64, gamma=0.97)
    run = resolve_ckpt_train_config(ckpt, strict=False,
                                    code=Layer("code:async_val.py", {"n_envs": 8}))
    assert (run.resolve_mode, run.gamma, run.n_envs) == ("eval", 0.97, 8)
    assert run.provenance["n_envs"].source == "code:async_val.py"
    assert run.provenance["gamma"].source == "ckpt"


def test_llm_eval_resolves_its_command_line(tmp_path):
    from eval.args import resolve_eval_args

    recipe = tmp_path / "r.yaml"
    recipe.write_text(yaml.safe_dump({"max_steps": 30, "temperature": 0.5}))
    run = resolve_eval_args(["--recipe", str(recipe), "--max_steps", "40", "--no_drc_tokens"])
    assert (run.resolve_mode, run.max_steps, run.temperature, run.no_drc_tokens) == ("fresh", 40, 0.5, True)
    assert run.provenance["max_steps"].source == "cli"
    assert run.provenance["temperature"].source == f"recipe:{recipe}"


def test_llm_eval_mode_reaches_the_rollout_dispatch(monkeypatch):
    """The LLM option ``--mode`` is a config key: the RunConfig's own resolve mode must not
    shadow it (it did — ``args.mode`` read 'fresh' and no rollout ran)."""
    from eval.args import resolve_eval_args
    from methods.llm_agent.rollout import pcbworld

    run = resolve_eval_args(["--mode", "api"])
    assert (run.mode, run.values["mode"]) == ("api", "api")
    called = []
    for name in ("run_fixed", "run_llm", "run_api"):
        monkeypatch.setattr(pcbworld, name, lambda args, name=name: called.append((name, args.mode)))
    monkeypatch.setattr("sys.argv", ["pcbworld", "--mode", "api", "--board_path", BOARD])
    pcbworld.main()
    assert called == [("run_api", "api")]


def test_a_config_key_named_like_a_run_config_attribute_is_refused():
    pol = {**POL, "provenance": Policy("runtime", False, False, "T.provenance")}
    with pytest.raises(ConfigError, match=r"\[.provenance.\] collide with RunConfig attributes"):
        resolve("fresh", policies=pol, defaults=[Layer("defaults", {**DEFAULTS.values, "provenance": 1})])


# ===================================================================
# 4. the eval env
# ===================================================================
def _eval_env(**kw):
    from configs.loader.schema import RLEnvConfig
    from eval.rollout.rl import resolve_eval_env

    base = RLEnvConfig().with_flat({"keep_routing_fraction": [0.1, 0.2], "aug_flip": True,
                                    "net_reselect": True, "max_steps": 300})
    return resolve_eval_env(base, where="ckpt.pt", **kw)


def test_eval_env_pins_train_only_fields_and_follows_the_checkpoint():
    run = _eval_env(step_drc=True)
    assert run.keep_routing_fraction is None and run.aug_flip is False
    assert run.provenance["keep_routing_fraction"].source == "defaults"
    assert (run.net_reselect, run.max_steps) == (True, 300)


def test_eval_env_overrides_follow_eval_override():
    run = _eval_env(step_drc=True, user=[Layer("cli", {"max_steps": 64, "corner_mode": 90})])
    assert (run.max_steps, run.corner_mode) == (64, 90)
    with pytest.raises(ConfigError, match="'net_reselect'=False in mode eval"):
        _eval_env(step_drc=True, user=[Layer("cli", {"net_reselect": False})])


def test_eval_env_drc_off_is_a_mode_default_a_cli_weight_overrides():
    """An old checkpoint trained without DRC tokens (step_drc=False) evaluated with an explicit
    --drc-penalty: the CLI weight wins over the DRC-off layer and both are in the chain (was a
    'set by both' error)."""
    run = _eval_env(step_drc=False)
    assert (run.emit_drc_tokens, run.drc_penalty) == (False, 0.0)
    assert run.provenance["drc_penalty"].source == "mode-default"
    run = _eval_env(step_drc=False, user=[Layer("cli", {"drc_penalty": 0.5})])
    assert (run.emit_drc_tokens, run.drc_penalty, run.drc_log_scale) == (False, 0.5, 0.0)
    assert [c["source"] for c in run.provenance["drc_penalty"].chain] == [
        "defaults", "ckpt", "mode-default", "cli"]



def test_eval_env_overrides_are_reported_against_the_checkpoint():
    """The eval's 'differs from training' report compares an override with the checkpoint's
    value, not with the layer under it: a command line restating the checkpoint value over a
    recipe (or over the DRC-off layer) is no override; a recipe value is reported as ckpt -> value."""
    from eval.pipeline import env_overrides_vs_ckpt

    recipe = Layer("recipe:r.yaml", {"max_steps": 64, "corner_mode": 90})
    run = _eval_env(step_drc=True, user=[recipe, Layer("cli", {"max_steps": 300})])
    assert env_overrides_vs_ckpt(run, "ckpt.pt") == {"corner_mode": (45, 90)}
    ckpt_drc = _eval_env(step_drc=True).drc_penalty
    run = _eval_env(step_drc=False, user=[Layer("cli", {"drc_penalty": ckpt_drc})])
    assert env_overrides_vs_ckpt(run, "ckpt.pt") == {}

# ===================================================================
# 5. every field declares a scope
# ===================================================================
def test_every_schema_field_declares_a_scope():
    from configs.loader import schema

    for cls in vars(schema).values():
        if isinstance(cls, type) and dataclasses.is_dataclass(cls) and cls.__module__ == schema.__name__:
            field_policies(cls())        # raises on a leaf without a scope


def test_a_field_without_a_scope_is_refused():
    @dataclasses.dataclass(frozen=True)
    class Bare:
        x: int = 1

    with pytest.raises(ValueError, match="Bare.x declares no scope"):
        field_policies(Bare())
