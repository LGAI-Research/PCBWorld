"""Structural guards for the env-config plumbing.

Background (2026-08-20 audit). Training and eval built their env kwargs through
two different paths: eval splatted ``RLEnvConfig.to_pool_kwargs()``, while
``TrainerBase._build_envs`` hand-listed 28 keyword arguments. Five knobs
(``action_history_len`` / ``net_constraint_obs`` / ``outline_obs`` /
``simplify_outline`` / ``keep_routing_fraction``) were never added to that
hand-list, so training silently ran on the factory's signature defaults while
val used the CLI value — a train/val observation drift that no test, no CSV
column and no checkpoint field could reveal, because every one of those records
the *intent* (the parsed CLI value) rather than what the env actually received.

A sixth knob, ``connectivity_filter``, failed the other way round: it was a
hand-written argparse flag with no schema field at all, so ``to_pool_kwargs()``
had no slot to carry it and eval could never see it.

These tests pin the three structural properties that make both classes of bug
impossible rather than merely detectable:

1. the pool factory has no signature default for any config-surface knob, so a
   knob that is not passed raises instead of falling back;
2. the trainer builds its kwargs from the shared surface rather than a
   hand-list, so a knob added to the schema reaches training with no edit;
3. every hand-written CLI flag maps onto a schema field, so nothing can enter
   the run without a slot in the config that both sides build from;
4. every schema field rides ``to_pool_kwargs()`` (or is a declared exemption),
   so a field added to the schema but forgotten in the surface dict cannot
   become a silently inert flag.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from configs.loader.schema import RLEnvConfig
from methods.rl_agent.wrappers import factory

PROJECT_ROOT = Path(__file__).resolve().parent.parent


#: ``seed`` and ``policy_net_select`` are not part of ``to_pool_kwargs()`` but
#: both train and eval pass them explicitly, so they are ordinary required
#: arguments whose *values* may differ — not a train-only bundle.
_ALSO_REQUIRED = {"seed", "policy_net_select"}

#: Process/backend plumbing: how workers are spawned, not what the env is.
_PROCESS_KNOBS = {"start_method", "backend", "resources_per_worker"}

_FACTORIES = (factory.make_decoder_env, factory.make_decoder_env_pool)


def _surface() -> set[str]:
    """The shared env-config surface: what train and eval both build."""
    return set(RLEnvConfig().to_pool_kwargs())


def _required() -> set[str]:
    return _surface() | _ALSO_REQUIRED


# ---------------------------------------------------------------------------
# 1. transport layer must not supply defaults
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", _FACTORIES, ids=lambda f: f.__name__)
def test_factory_has_no_default_for_contract_knobs(fn):
    params = inspect.signature(fn).parameters
    filled = sorted(
        name for name in _required()
        if name in params and params[name].default is not factory._REQ
    )
    assert not filled, (
        f"{filled} regained a default in the {fn.__name__} signature — "
        "the transport layer may not hold defaults for env-contract knobs "
        "(not passing one must raise TypeError)"
    )


@pytest.mark.parametrize("fn", _FACTORIES, ids=lambda f: f.__name__)
def test_factory_covers_the_whole_surface(fn):
    unknown = sorted(_required() - set(inspect.signature(fn).parameters))
    assert not unknown, (
        f"{unknown} is on the config surface but {fn.__name__} does not "
        "accept it — the config surface and the transport layer disagree"
    )


@pytest.mark.parametrize("fn", _FACTORIES, ids=lambda f: f.__name__)
def test_train_extras_is_the_only_value_default(fn):
    """No parameter may silently supply an env value.

    The single non-process default is ``train_extras``, and ``None`` there
    means "no training layer at all" rather than a per-knob value — eval omits
    the bundle instead of accidentally selecting ``aug_flip=False``.
    """
    params = inspect.signature(fn).parameters
    defaulted = sorted(
        name for name, prm in params.items()
        if prm.default is not inspect.Parameter.empty
        and prm.default is not factory._REQ
        and name not in _PROCESS_KNOBS
    )
    assert defaulted == ["train_extras"], (
        f"{fn.__name__} value-defaults: {defaulted} — 'train_extras' must be the only one"
    )


@pytest.mark.parametrize("fn", _FACTORIES, ids=lambda f: f.__name__)
def test_missing_contract_knob_raises(fn):
    args = ("board.kicad_pcb", 1) if fn is factory.make_decoder_env_pool else ("board.kicad_pcb",)
    with pytest.raises(TypeError, match="missing env-contract knob"):
        fn(*args, max_steps=10)


def test_train_extras_rejects_unknown_keys():
    with pytest.raises(TypeError, match="unknown key"):
        factory._unpack_train_extras({"aug_flip": True, "nope": 1}, where="t")


# ---------------------------------------------------------------------------
# 2. the trainer must not re-grow a hand-list
# ---------------------------------------------------------------------------
def _factory_call_keywords(path: Path, func: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == func:
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError(f"no {func}(...) call found in {path.name}")


def test_trainer_builds_env_kwargs_from_the_shared_surface():
    loop = PROJECT_ROOT / "methods" / "rl_agent" / "training" / "loop.py"
    named = _factory_call_keywords(loop, "make_decoder_env_pool")
    leaked = sorted(named & _surface())
    assert not leaked, (
        f"loop.py hand-lists {leaked} — env kwargs must be built only by "
        "to_pool_kwargs() so that a new knob reaches training automatically"
    )


# ---------------------------------------------------------------------------
# 3. no CLI flag may bypass the schema
# ---------------------------------------------------------------------------
# ``--no-X`` style flags whose value lives in the schema under the positive
# name. Adding an entry here is a deliberate act, not a silent bypass.
INVERTED_ALIASES = {
    "no_drc_tokens": "emit_drc_tokens",
    "no_mask_start_point": "mask_start_point",
}

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
# 4. every schema field must ride the surface (or be a declared exemption)
# ---------------------------------------------------------------------------
# Closes the last silent gap in the CLI -> schema -> surface chain: a field
# added to RLEnvConfig but forgotten in ``to_pool_kwargs()`` is carried by
# nobody — the flag parses, no factory ever receives it, and no TypeError can
# fire because nothing passes it. Guards 1-3 only see knobs that are already
# on the surface.
#: Fields that intentionally do NOT travel through ``to_pool_kwargs()``.
#: An entry here is a declaration that the knob reaches the env another way —
#: not a shortcut for "forgot to add it to the surface".
SURFACE_EXEMPT = {
    # Train-only bundle (factory._TRAIN_EXTRAS): eval omits the whole bundle,
    # so these must stay off the shared surface by design.
    "reward_noise_std",
    "aug_bbox_shifted", "aug_flip", "aug_rotate", "aug_trans", "aug_zoom",
    # Defaults-file constants: the factory builds EnvConfig(...) without them,
    # so every run (train and eval alike) uses the schema default — engine
    # tuning constants, not per-run knobs. Slated for the defaults-elimination
    # backlog.
    "engine_seed", "shove_iter_limit", "followbranch_iter_limit",
    "reject_if_stuck",
}


def _schema_leaves(obj: object) -> set[str]:
    """Leaf field names of a dataclass instance, nested dataclasses flattened."""
    import dataclasses

    leaves: set[str] = set()
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            leaves |= _schema_leaves(value)
        else:
            leaves.add(f.name)
    return leaves


def test_every_schema_field_rides_the_surface():
    leaves = _schema_leaves(RLEnvConfig())
    missing = sorted(leaves - _surface() - SURFACE_EXEMPT)
    assert not missing, (
        f"{missing} is in RLEnvConfig but does not ride to_pool_kwargs() — "
        "nobody passes it to the factory, so the flag is silently dead. "
        "Add it to to_pool_kwargs(), or write down why it travels by another "
        "path and register it in SURFACE_EXEMPT."
    )


def test_surface_exempt_list_stays_honest():
    # An exemption that joined the surface (or left the schema) is stale —
    # a rotten allowlist would quietly re-widen the gap this section closes.
    stale = sorted(SURFACE_EXEMPT & _surface())
    assert not stale, f"{stale} now rides the surface — remove it from SURFACE_EXEMPT"
    ghosts = sorted(SURFACE_EXEMPT - _schema_leaves(RLEnvConfig()))
    assert not ghosts, f"{ghosts} is not in the schema — remove it from SURFACE_EXEMPT"


# ---------------------------------------------------------------------------
# 5. eval must not adjust env kwargs outside the shared resolver
# ---------------------------------------------------------------------------
def test_eval_transformer_touches_env_kwargs_only_via_the_resolver():
    """Keeps the trainer's startup prediction of the val kwargs honest.

    ``TrainerBase._record_env_contract`` calls ``resolve_eval_env_kwargs`` to
    learn what validation will build, so the record (and the
    ``--expect-env-diff`` gate) is exact only while that function is the sole
    place eval adjusts what it was handed. A fourth adjustment added straight
    into ``eval_transformer`` would silently stale the record — the same shape
    of bug this whole module exists to prevent.
    """
    src = (PROJECT_ROOT / "eval" / "rollout" / "rl.py").read_text()
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "eval_transformer"
    )
    offenders: list[str] = []
    for node in ast.walk(fn):
        # env_kwargs[...] = ...
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and getattr(tgt.value, "id", "") == "env_kwargs"):
                    offenders.append(f"line {node.lineno}: env_kwargs[...] assignment")
                if getattr(tgt, "id", "") == "env_kwargs":
                    call = node.value
                    ok = (isinstance(call, ast.Call)
                          and getattr(call.func, "id", "") == "resolve_eval_env_kwargs")
                    if not ok:
                        offenders.append(f"line {node.lineno}: env_kwargs reassignment")
        # apply_drc_off(env_kwargs) / anything mutating it in place
        if isinstance(node, ast.Call):
            takes_it = any(getattr(a, "id", "") == "env_kwargs" for a in node.args)
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if takes_it and name not in ("resolve_eval_env_kwargs", "dict"):
                offenders.append(f"line {node.lineno}: {name}(env_kwargs)")
    assert not offenders, (
        "eval_transformer touches env_kwargs outside the resolver — "
        "move it into resolve_eval_env_kwargs:\n  " + "\n  ".join(offenders)
    )


def test_recorded_val_kwargs_cross_check():
    from eval.rollout.rl import assert_env_kwargs_as_recorded

    assert_env_kwargs_as_recorded({"a": 1, "seed": 7}, {"a": 1, "seed": 7})
    with pytest.raises(RuntimeError, match="startup record"):
        assert_env_kwargs_as_recorded({"a": 1, "seed": 7}, {"a": 2, "seed": 7})
    with pytest.raises(RuntimeError, match="startup record"):   # a new key appeared
        assert_env_kwargs_as_recorded({"a": 1, "new": 3}, {"a": 1})


def test_harness_seed_diff_needs_no_declaration():
    """A stock launch must pass the gate — the seed diff is the harness's own.

    ``_record_env_contract`` records the training seed (``--seed``) against the
    validation seed (``--eval-base-seed``): validation is deliberately seeded
    apart from the training stream, so the two differ on the shipped defaults
    and EVERY documented training command died at startup demanding
    ``--expect-env-diff seed`` — a flag no shipped doc or script passes, and
    one ``train_policy.sh``'s whitelist rejects outright.
    """
    from argparse import ArgumentParser

    from methods._shared.config_dump import _diff, check_expected_env_diff
    from methods.rl_agent.training.args import add_shared_args

    parser = ArgumentParser()
    add_shared_args(parser)
    seed, eval_seed = (parser.get_default("seed"),
                       parser.get_default("eval_base_seed"))
    assert seed != eval_seed, "defaults agree — this guard would be vacuous"
    diff = _diff({"seed": seed}, {"seed": eval_seed})
    assert set(diff) == {"seed"}
    check_expected_env_diff(diff, "")        # must not raise

    # The gate still halts on a real undeclared difference, and on a seed that
    # is absent on one side (the five-knob drift's own signature).
    with pytest.raises(SystemExit):
        check_expected_env_diff(_diff({"outline_obs": True},
                                      {"outline_obs": False}), "")
    with pytest.raises(SystemExit):
        check_expected_env_diff(_diff({"seed": seed}, {}), "")


def test_train_extras_hand_list_matches_factory_bundle():
    """Hand-sync guard: loop._TRAIN_EXTRAS_ARGS ↔ factory._TRAIN_EXTRAS.

    A new train-only knob put in the factory bundle but left out of the trainer
    tuple still parses on the CLI while the training env receives the bundle
    default — the same class as the five-knob drift, yet it left only a trace
    in the env_records diff with no automatic failure. ``advance_rng_on_reload``
    is a pool-only knob with no CLI flag, so it belongs outside the tuple (the
    trainer always adds it as True).
    """
    loop_src = (Path(__file__).resolve().parents[1]
                / "methods" / "rl_agent" / "training" / "loop.py")
    names = None
    for node in ast.walk(ast.parse(loop_src.read_text())):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_TRAIN_EXTRAS_ARGS"
                for t in node.targets):
            names = ast.literal_eval(node.value)
    assert names is not None, "_TRAIN_EXTRAS_ARGS not found in loop.py"
    assert set(names) == set(factory._TRAIN_EXTRAS) - {"advance_rng_on_reload"}


def test_keep_routing_fraction_cli_default_comes_from_yaml():
    """The bespoke (nargs=2) flag's default is locked to the YAML (EnvConfig).

    Unlike the auto-generated flags, this hand-written flag alone used to
    bypass the YAML default with a ``default=None`` of its own (260827 review).
    ``add_shared_args`` now takes the default from the schema instance.
    """
    from argparse import ArgumentParser

    from configs.loader.schema import EnvConfig
    from methods.rl_agent.training.args import add_shared_args

    p = ArgumentParser()
    add_shared_args(p)
    assert (p.get_default("keep_routing_fraction")
            == EnvConfig().keep_routing_fraction)
    # Same class of bypass: bespoke --pad-graze-margin-mm is locked to the schema value too
    from configs.loader.schema import RLEnvConfig
    assert (p.get_default("pad_graze_margin_mm")
            == RLEnvConfig().pad_graze_margin_mm)


def test_all_default_train_extras_bundle_pops_from_records():
    """An all-default run's train_extras bundle is popped from env_records.

    ``_env_kwargs`` builds the bundle from the CLI defaults and always forces
    ``advance_rng_on_reload=True``. The pop compares against
    ``{**factory._TRAIN_EXTRAS, advance_rng_on_reload: True}``, and the two must
    agree — otherwise an all-default run is forced into a meaningless
    ``--expect-env-diff train_extras`` declaration (260827 review: the RNG fix
    had made this comparison permanently false — a dead branch).
    """
    from argparse import ArgumentParser

    from methods.rl_agent.training.args import add_shared_args
    from methods.rl_agent.training.loop import RLTrainer

    parser = ArgumentParser()
    add_shared_args(parser)
    bundle = {n: parser.get_default(n) for n in RLTrainer._TRAIN_EXTRAS_ARGS}
    bundle["advance_rng_on_reload"] = True      # same value _env_kwargs always forces
    assert bundle == {**factory._TRAIN_EXTRAS, "advance_rng_on_reload": True}
