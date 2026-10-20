"""Structural guards for the env-config plumbing.

Background. Training and eval once built their env kwargs through two different
paths, and a hand-written kwarg list dropped what it did not name: five obs
knobs ran on the factory's signature defaults in training (2026-08-20 audit),
``connectivity_filter`` had no schema field at all, and the engine knobs
(``engine_seed`` / ``shove_iter_limit`` / ``followbranch_iter_limit``) were
parsed and stored in every checkpoint but never reached the engine (2026-10-17).

The env is now one :class:`RLEnvConfig` carried whole: the factories take it,
``PCBWorld`` takes its ``env``, ``KiCadEngine`` takes that config's ``engine_kwargs()``,
and none of the three holds a default for a schema field. These tests pin:

1. no factory / ``PCBWorld`` / ``KiCadEngine`` signature default for a schema field;
2. every training env flag, set to a non-default value, arrives in the
   ``PCBWorld`` / wrapper / ``KiCadEngine`` actually constructed;
3. every hand-written CLI flag maps onto a schema field, and every config flag
   of the training parser is read into the env config;
4. eval adjusts the config only in its one resolver.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs.loader.schema import TRAIN_LAYER, EnvConfig, RLEnvConfig
from methods.rl_agent.wrappers import factory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Process/backend plumbing: how workers are spawned, not what the env is.
_PROCESS_KNOBS = {"start_method", "backend", "resources_per_worker"}

#: Schema fields whose training-namespace dest is inverted (``--no-x``).
INVERTED_ALIASES = {
    "no_drc_tokens": "emit_drc_tokens",
    "no_mask_start_point": "mask_start_point",
}


def _leaves() -> set[str]:
    return set(RLEnvConfig().to_record())


# ---------------------------------------------------------------------------
# 1. no layer below the definition point holds a default
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [factory.make_decoder_env, factory.make_decoder_env_pool],
                         ids=lambda f: f.__name__)
def test_factory_holds_no_value_default(fn):
    params = inspect.signature(fn).parameters
    defaulted = sorted(n for n, p in params.items()
                       if p.default is not inspect.Parameter.empty and n not in _PROCESS_KNOBS)
    assert not defaulted, f"{fn.__name__} value-defaults: {defaulted}"
    assert "env_config" in params


def test_pcbworld_and_engine_signatures_hold_no_schema_defaults():
    """Neither constructor may name a schema field with a default: a field it does
    not receive must raise, not fall back to a second copy of the YAML value."""
    from pcb_world.core.env import PCBWorld
    from pcb_world.engine.kicad_engine import KiCadEngine

    leaves = _leaves()
    for cls in (PCBWorld, KiCadEngine):
        params = inspect.signature(cls.__init__).parameters
        defaulted = sorted(n for n, p in params.items()
                           if n in leaves and p.default is not inspect.Parameter.empty)
        assert not defaulted, f"{cls.__name__}.__init__ defaults schema fields {defaulted}"
    pcbworld_fields = sorted(set(inspect.signature(PCBWorld.__init__).parameters) & leaves)
    assert not pcbworld_fields, (f"PCBWorld takes {pcbworld_fields} as keywords — env-core "
                                 "fields travel in env_config only")
    engine = set(inspect.signature(KiCadEngine.__init__).parameters)
    assert set(EnvConfig().engine_kwargs()) <= engine


# ---------------------------------------------------------------------------
# 2. every training env value arrives in what is constructed
# ---------------------------------------------------------------------------
def _perturbed(name: str, default):
    """A valid non-default value for field ``name`` (types from the YAML default)."""
    special = {"corner_mode": 90, "outline_obs": "poly16", "obs_format": "indexed",
               "wire_via_emission": "on_net_end", "directional_candidates": "grid16",
               "drc_config_path": "/tmp/drc.yaml", "keep_routing_fraction": [0.1, 0.2],
               "drc_promoted_codes": [12, 13],
               "rules_source": "board_or_yaml",
               "masking_rule": "probe_masking", "reward_rule": "probe_reward"}
    if name in special:
        assert special[name] != default, name
        return special[name]
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 3
    if default is None or isinstance(default, float):
        return (default or 0.0) + 0.25
    raise AssertionError(f"no perturbation for {name}={default!r}")


def _training_namespace():
    """The trainer's parsed defaults with EVERY env field set away from its default
    (field list from the schema, not by hand). Returns ``(namespace, expected record)``."""
    from methods.rl_agent.training.train_ppo import build_arg_parser

    ns = build_arg_parser().parse_args(["--board", "b.kicad_pcb"])
    expected = {}
    for name, default in RLEnvConfig().to_record().items():
        expected[name] = _perturbed(name, default)
        inverted = {v: k for k, v in INVERTED_ALIASES.items()}.get(name)
        setattr(ns, inverted or name, (not expected[name]) if inverted else expected[name])
    return ns, expected


def test_every_training_env_value_reaches_the_constructed_env_and_engine(monkeypatch):
    """Training namespace → ``RLEnvConfig.for_training`` → ``make_decoder_env`` →
    ``PCBWorld`` / ``KiCadRLWrapper`` → ``KiCadEngine``: every leaf of the config
    arrives with the namespace's value. Before 2026-10-17 the factory rebuilt the
    env from a hand list without the engine knobs, so a run asking for
    engine_seed 5 / shove 9 / followbranch 11 routed with 77 / 250 / 1e6."""
    import pcb_world.core.env as env_mod
    ns, expected = _training_namespace()
    cfg = RLEnvConfig.for_training(ns)

    seen: dict = {}

    class _Stop(Exception):
        pass

    def engine(board_path, **kw):
        seen["engine"] = kw
        raise _Stop

    def wrapper(env, **kw):
        seen["wrapper"] = kw
        return env

    real_pcbworld = env_mod.PCBWorld

    def pcbworld(board_path, env_config, **kw):
        seen["env_config"] = env_config
        try:
            real_pcbworld(board_path, env_config, **kw)     # runs up to KiCadEngine(...)
        except _Stop:
            pass
        return SimpleNamespace()

    monkeypatch.setattr(env_mod, "KiCadEngine", engine)
    monkeypatch.setattr(factory, "PCBWorld", pcbworld)
    monkeypatch.setattr(factory, "KiCadRLWrapper", wrapper)
    factory.make_decoder_env("b.kicad_pcb", cfg, seed=0, policy_net_select=True)

    got = {**seen["env_config"].to_env_kwargs(), **seen["wrapper"]}
    missing = sorted(set(expected) - set(got))
    assert not missing, f"{missing} never reached PCBWorld / the wrapper"
    wrong = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    assert not wrong, f"constructed with other values than the training namespace: {wrong}"
    engine_expected = {k: expected[k] for k in EnvConfig().engine_kwargs()}
    assert {k: seen["engine"][k] for k in engine_expected} == engine_expected
    assert seen["engine"]["allow_default_rules"] is True     # rules_source=board_or_yaml


def test_every_config_flag_is_read_into_the_env_config():
    """Each flag the training parser generates for an env field, set alone, changes
    exactly that field of the training env config — a flag that parses but is not
    read (the engine knobs before 2026-10-17) fails here."""
    from methods.rl_agent.training.train_ppo import build_arg_parser

    leaves = _leaves()
    dests = {a.dest for a in build_arg_parser()._actions}
    flagged = {INVERTED_ALIASES.get(d, d) for d in dests} & leaves
    base_ns = build_arg_parser().parse_args(["--board", "b.kicad_pcb"])
    base = RLEnvConfig.for_training(base_ns).to_record()
    for name in sorted(flagged):
        ns = SimpleNamespace(**vars(base_ns))
        inverted = {v: k for k, v in INVERTED_ALIASES.items()}.get(name)
        value = _perturbed(name, base[name])
        setattr(ns, inverted or name, (not value) if inverted else value)
        got = RLEnvConfig.for_training(ns).to_record()
        changed = sorted(k for k in got if got[k] != base[k])
        assert changed == [name], f"--{name.replace('_', '-')} changed {changed}"


def test_train_layer_is_training_only():
    """Validation builds without the training-only layer; the trainer adds it."""
    ns, expected = _training_namespace()
    train = RLEnvConfig.for_training(ns).to_record()
    val = RLEnvConfig.from_namespace(ns).to_record()
    assert {k for k in train if train[k] != val[k]} == set(TRAIN_LAYER)
    assert {k: train[k] for k in TRAIN_LAYER} == {k: expected[k] for k in TRAIN_LAYER}


def test_trainer_carries_the_config_object():
    """loop.py passes the config object to the pool, never schema fields by name."""
    tree = ast.parse((PROJECT_ROOT / "methods" / "rl_agent" / "training" / "loop.py").read_text())
    named = {kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") in ("make_decoder_env", "make_decoder_env_pool")
             for kw in node.keywords if kw.arg}
    assert not named & _leaves(), f"loop.py names {sorted(named & _leaves())} at the factory"


# ---------------------------------------------------------------------------
# 3. no CLI flag may bypass the schema
# ---------------------------------------------------------------------------
# Flags that steer the *process*, not the env/policy the run produces, so they
# have no schema field by design.
NON_CONFIG_FLAGS = {"board", "no_vecenv", "wandb", "no_eval_greedy"}


def _schema_field_names() -> set[str]:
    import dataclasses

    from configs.loader import schema as s

    names: set[str] = set()
    for attr in vars(s).values():
        if isinstance(attr, type) and dataclasses.is_dataclass(attr):
            names |= {f.name for f in dataclasses.fields(attr)}
    return names


def test_every_hand_written_cli_flag_has_a_schema_field():
    args_py = PROJECT_ROOT / "methods" / "rl_agent" / "training" / "args.py"
    known = _schema_field_names() | set(INVERTED_ALIASES) | NON_CONFIG_FLAGS
    orphans = []
    for node in ast.walk(ast.parse(args_py.read_text())):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            continue
        flag = next((a.value for a in node.args
                     if isinstance(a, ast.Constant) and str(a.value).startswith("--")),
                    None)
        if flag is None:
            continue
        dest = next((kw.value.value for kw in node.keywords
                     if kw.arg == "dest" and isinstance(kw.value, ast.Constant)), None)
        name = dest or flag[2:].replace("-", "_")
        if name not in known:
            orphans.append(flag)
    assert not orphans, (
        f"{orphans} bypass the schema — create the field in RLEnvConfig (or a "
        "sibling) first. (An inverted alias belongs in INVERTED_ALIASES, a "
        "process-control flag in NON_CONFIG_FLAGS — but registering there is a "
        "deliberate declaration of an exception.)"
    )


# ---------------------------------------------------------------------------
# 4. eval must not adjust the env config outside the shared resolver
# ---------------------------------------------------------------------------
def test_eval_transformer_builds_the_env_config_it_is_given():
    """Keeps the trainer's startup prediction of the val env honest.

    ``RLTrainer._record_env_contract`` records ``resolve_eval_env``'s result — the config
    validation will build — so the record (and the ``--expect-env-diff`` gate) is exact only
    while every eval-side adjustment is a layer of that resolve. An adjustment added straight
    into ``eval_transformer`` would silently stale the record — the same shape of bug this
    whole module exists to prevent.
    """
    src = (PROJECT_ROOT / "eval" / "rollout" / "rl.py").read_text()
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "eval_transformer"
    )
    offenders: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            offenders += [f"line {node.lineno}: env_config reassignment"
                          for tgt in targets if getattr(tgt, "id", "") == "env_config"]
        # env_config.with_flat(...) — a derived config built in place
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and getattr(node.func.value, "id", "") == "env_config"
                and node.func.attr == "with_flat"):
            offenders.append(f"line {node.lineno}: env_config.{node.func.attr}(...)")
    assert not offenders, (
        "eval_transformer changes the env config it was given — make it a layer of "
        "eval.rollout.rl.resolve_eval_env:\n  " + "\n  ".join(offenders)
    )


def test_recorded_val_kwargs_cross_check():
    from eval.rollout.rl import assert_env_kwargs_as_recorded

    assert_env_kwargs_as_recorded({"a": 1, "seed": 7}, {"a": 1, "seed": 7})
    with pytest.raises(RuntimeError, match="startup record"):
        assert_env_kwargs_as_recorded({"a": 1, "seed": 7}, {"a": 2, "seed": 7})
    with pytest.raises(RuntimeError, match="startup record"):   # a new key appeared
        assert_env_kwargs_as_recorded({"a": 1, "new": 3}, {"a": 1})


def test_trainer_records_the_val_env_and_gates_its_difference(tmp_path):
    """The validation env is a run record of its own (``run_record.val.yaml``) that the
    watchers read back in the ``env_record`` form; train vs val is compared over the two
    resolved configs, and only an undeclared difference halts. A stock launch passes: the
    validation seed (``--eval-base-seed``) is the harness's own and is not compared."""
    from configs.loader.record import VAL_RUN_RECORD
    from eval.rollout.rl import assert_env_kwargs_as_recorded, eval_env_config
    from methods.rl_agent.training.async_val import load_recorded_val_env
    from methods.rl_agent.training.loop import PPOTrainer, val_env_config
    from methods.rl_agent.training.train_ppo import parse_train_args

    def contract(*argv):
        t = PPOTrainer.__new__(PPOTrainer)
        t.args = parse_train_args(["--board", "b.kicad_pcb", "--save-dir", str(tmp_path), *argv])
        t.val_env, t.eval_boards = val_env_config(t.args), None
        t.agent = SimpleNamespace(policy_net_select=True)
        t._record_env_contract()
        return t

    t = contract()
    recorded = load_recorded_val_env(tmp_path)
    assert recorded == t._recorded_val_env
    assert_env_kwargs_as_recorded(factory.env_record(
        eval_env_config(t.val_env), seed=t.args.eval_base_seed, policy_net_select=True), recorded)
    assert (tmp_path / VAL_RUN_RECORD).is_file() and not (tmp_path / "env_records").exists()
    with pytest.raises(SystemExit, match="--expect-env-diff aug_flip"):
        contract("--aug-flip")
    contract("--aug-flip", "--expect-env-diff", "aug_flip")


def test_keep_routing_fraction_cli_default_comes_from_yaml():
    """The bespoke (nargs=2) flag's default is locked to the YAML (EnvConfig).

    Unlike the auto-generated flags, this hand-written flag alone used to
    bypass the YAML default with a ``default=None`` of its own (260827 review).
    ``add_shared_args`` now takes the default from the schema instance.
    """
    from configs.loader.cli import StrictArgumentParser
    from configs.loader.schema import EnvConfig
    from methods.rl_agent.training.args import add_shared_args

    p = StrictArgumentParser()
    add_shared_args(p)
    assert (p.get_default("keep_routing_fraction")
            == EnvConfig().keep_routing_fraction)
    # Same class of bypass: bespoke --pad-graze-margin-mm is locked to the schema value too
    from configs.loader.schema import RLEnvConfig
    assert (p.get_default("pad_graze_margin_mm")
            == RLEnvConfig().pad_graze_margin_mm)
