"""The one function that makes a run's config: layers in, a frozen config with per-key provenance out.

:func:`resolve` stacks the layers of one config object and decides every key once::

    DEFAULTS  the schema defaults (configs/{env,rl,llm}.yaml), then the mode's defaults
              (MODE_DEFAULTS: an init-ckpt run starts from rl_train.finetune)
      → CKPT  eval / resume: every key the checkpoint carries (a resume keeps its save_dir,
              async_val, wandb ... — USER may change a runtime key); init-ckpt: its arch keys
              (configs.loader.ckpt.load_ckpt_config output — legacy keys already filled)
      → USER  recipe:<file> (--recipe)  <  cli (the dests the command line gave)  ;  code:<file>

and checks every USER key against the scope × mode table (:mod:`configs.loader.knob`):

=============  =====  ==========================  ==============  ===========================
scope          fresh  resume                      init-ckpt       eval
=============  =====  ==========================  ==============  ===========================
arch           USER   = ckpt                      = ckpt          = ckpt
experiment     USER   = ckpt unless resume_mutable USER           = ckpt unless eval_override
runtime        USER   USER                        USER            USER
=============  =====  ==========================  ==============  ===========================

"= ckpt" means a USER value that differs from the checkpoint's raises (an equal one is
recorded) — except in init-ckpt for an arch key whose checkpoint value is not recorded in it
(source ``legacy*``: estimated or inferred from configs/ckpt_migrations.yaml, not derived from
the weights): the USER value is taken with a warning naming both, unless ``strict_ckpt``.
A key set by both ``cli`` and a ``code:`` layer raises; a key a layer sets that
the config does not have raises; a key without a declared policy raises.

The result is a :class:`RunConfig`: the values as read-only attributes (a frozen namespace —
``vars(run)`` is the flat value dict a checkpoint stores) plus ``run.provenance[key]`` = the
final value, its source (``defaults`` | ``ckpt`` | ``legacy:*`` | ``derived`` | ``renamed`` |
``recipe:<file>`` | ``cli`` | ``code:<file>``), where it was set, and the chain of every layer
that set it. :mod:`configs.loader.record` writes it, :mod:`configs.loader.explain` prints it.

:func:`resolve_parsed` builds the DEFAULTS / recipe / cli / code layers of a
:class:`~configs.loader.cli.StrictArgumentParser` parse and checks ``--expect``; entry points
call it once per config object, right after parsing and before any env or model exists.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Collection, Mapping, Sequence

from configs.loader.knob import Policy, field_policies, runtime

MODES = ("fresh", "resume", "init-ckpt", "eval")

#: Options every resolving entry point carries that steer this process only.
PROCESS_OPTIONS = ("recipe", "expect", "strict_ckpt", "dump_config_only")
#: Options a recipe may not set: read before the recipe (the resolve mode follows ``resume`` /
#: ``init_ckpt``), so a recipe value would be recorded but never acted on.
COMMAND_LINE_ONLY = (*PROCESS_OPTIONS, "resume", "init_ckpt")


class ConfigError(ValueError):
    """A layer set a key it may not set, or a key has no value or policy."""


@dataclass(frozen=True)
class Layer:
    """One source of values: ``source`` names it, ``where`` / ``sources`` refine it per key
    (a checkpoint layer's ``sources`` are the loader's ``ckpt`` / ``legacy:*`` / ``derived``);
    ``meta`` is what a checkpoint layer knows besides its config (its ``iteration``)."""

    source: str
    values: Mapping[str, Any]
    where: Mapping[str, str] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def entry(self, key: str) -> dict[str, Any]:
        return {"source": self.sources.get(key, self.source), "value": self.values[key],
                "where": self.where.get(key, self.source)}


@dataclass(frozen=True)
class Entry:
    """The provenance of one key: the final value and every layer that set it, in order."""

    value: Any
    source: str
    where: str
    chain: tuple[Mapping[str, Any], ...]


class RunConfig:
    """A resolved config: the values as read-only attributes, :attr:`provenance` per key,
    :attr:`resolve_mode` the :data:`MODES` member it was resolved in, :attr:`ckpt_meta` the
    checkpoint layer's ``meta`` (empty without one). A config key may not share a name with
    these attributes (:func:`resolve` refuses it)."""

    __slots__ = ("resolve_mode", "ckpt_meta", "provenance", "__dict__")

    def __init__(self, resolve_mode: str, provenance: Mapping[str, Entry],
                 ckpt_meta: Mapping[str, Any] = MappingProxyType({})) -> None:
        object.__setattr__(self, "resolve_mode", resolve_mode)
        object.__setattr__(self, "ckpt_meta", MappingProxyType(dict(ckpt_meta)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(provenance)))
        self.__dict__.update({k: e.value for k, e in provenance.items()})

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"RunConfig is frozen: cannot set {key!r} after resolve — add a layer "
                             "(code:<file> for a value code decides) instead")

    __delattr__ = __setattr__

    @property
    def values(self) -> Mapping[str, Any]:
        return MappingProxyType(self.__dict__)

    def ckpt_config(self) -> dict[str, Any]:
        """The ``config`` entry a checkpoint stores (schema 2)."""
        return {"schema": 2, "mode": self.resolve_mode, "values": dict(self.__dict__),
                "sources": {k: e.source for k, e in self.provenance.items()}}

    def __repr__(self) -> str:
        return f"RunConfig(resolve_mode={self.resolve_mode!r}, {len(self.__dict__)} keys)"


#: The non-value attributes of a :class:`RunConfig` (a config key of the same name would hide one).
_RUN_CONFIG_ATTRS = frozenset(n for n in dir(RunConfig) if not n.startswith("_"))


def same_value(a: Any, b: Any) -> bool:
    """Equal as a run setting: ``None`` / ``False`` / ``""`` / ``"none"`` / ``"off"`` are one
    "off" — never a number: ``0`` is a value, not "unset" —, and a path or number spelled
    differently (``45`` vs ``"45"``) is the same value."""
    if _is_off(a) or _is_off(b):
        return _is_off(a) and _is_off(b)
    return a == b or str(a) == str(b)


def _is_off(x: Any) -> bool:
    return x is None or x is False or x in ("", "none", "off")


def resolve(mode: str, *, policies: Mapping[str, Policy], defaults: Sequence[Layer],
            ckpt: Layer | None = None, user: Sequence[Layer] = (), strict_ckpt: bool = False) -> RunConfig:
    """Resolve one config object — module docstring. ``policies`` covers every key of the
    first ``defaults`` layer, which defines the key set; ``strict_ckpt`` (``--strict-ckpt``)
    refuses an init-ckpt override of an estimated checkpoint value too."""
    if mode not in MODES:
        raise ConfigError(f"unknown resolve mode {mode!r} (one of {MODES})")
    if (ckpt is None) != (mode == "fresh"):
        raise ConfigError(f"mode {mode!r} {'needs' if ckpt is None else 'takes no'} checkpoint layer")
    keys = list(defaults[0].values)
    clash = sorted(set(keys) & _RUN_CONFIG_ATTRS)
    if clash:
        raise ConfigError(f"config keys {clash} collide with RunConfig attributes — rename the option")
    undeclared = sorted(set(keys) - set(policies))
    if undeclared:
        raise ConfigError(f"keys without a declared scope: {undeclared} — declare them with "
                          "configs.loader.knob.knob (schema field) or in the entry point's policy table")
    for layer in [*defaults[1:], *user]:
        unknown = sorted(set(layer.values) - set(keys))
        if unknown:
            raise ConfigError(f"layer {layer.source} sets {unknown}, which this config does not have")
    cli = next((layer.values for layer in user if layer.source == "cli"), {})
    for layer in user:
        both = sorted(set(layer.values) & set(cli)) if layer.source.startswith("code:") else []
        if both:
            raise ConfigError(f"{both} set by both the command line and {layer.source} — give it once")

    provenance: dict[str, Entry] = {}
    for key in keys:
        pol = policies[key]
        chain = [layer.entry(key) for layer in defaults if key in layer.values]
        # A checkpoint value stands unless the mode frees the key: init-ckpt takes only the arch keys.
        pinned = (ckpt is not None and key in ckpt.values and key not in PROCESS_OPTIONS
                  and (mode != "init-ckpt" or pol.scope == "arch"))
        if pinned:
            chain.append(ckpt.entry(key))
        for layer in user:
            if key not in layer.values:
                continue
            got = layer.entry(key)
            if pinned and not same_value(got["value"], ckpt.values[key]) and not _may_change(mode, pol):
                estimate = ckpt.entry(key)["source"]
                if mode == "init-ckpt" and not strict_ckpt and estimate.partition(":")[0] == "legacy":
                    warnings.warn(
                        f"{key}: the checkpoint's {ckpt.values[key]!r} is not recorded in it "
                        f"(estimated/inferred, source {estimate}) — taking {got['source']} "
                        f"{got['value']!r} (--strict-ckpt refuses)", UserWarning, stacklevel=2)
                    chain.append(got)
                    continue
                flag = ("resume_mutable" if mode == "resume" else "eval_override")
                raise ConfigError(
                    f"layer {got['source']} may not set {key!r}={got['value']!r} in mode {mode} "
                    f"(scope={pol.scope}, checkpoint has {ckpt.values[key]!r}; {pol.where}"
                    + (f", {flag}=False); declare {flag}=True in configs/loader/schema.py if intended"
                       if pol.scope == "experiment" else "); the weights depend on it — train a new policy")
                    + " or drop the override")
            chain.append(got)
        last = chain[-1]
        provenance[key] = Entry(last["value"], last["source"], last["where"],
                                tuple(MappingProxyType(c) for c in chain))
    return RunConfig(mode, provenance, {} if ckpt is None else ckpt.meta)


def _may_change(mode: str, pol: Policy) -> bool:
    """Whether USER may set a key to another value than the checkpoint's (the table above)."""
    if pol.scope == "runtime" or mode == "fresh":
        return True
    if pol.scope == "arch":
        return False
    return {"resume": pol.resume_mutable, "init-ckpt": True, "eval": pol.eval_override}[mode]


# ---------------------------------------------------------------------------
# From a parsed command line
# ---------------------------------------------------------------------------

def dest_policies(*configs: Any, inverted: Mapping[str, str] = MappingProxyType({}),
                  extra: Mapping[str, Policy] = MappingProxyType({})) -> dict[str, Policy]:
    """``{dest: Policy}`` for a parser built from the schema dataclass instances ``configs``:
    their leaf fields, the ``inverted`` dests (``no_x`` → field ``x``) and the ``extra`` options
    that are not schema fields; the :data:`PROCESS_OPTIONS` are runtime. A field declared twice
    with different policies raises."""
    out: dict[str, Policy] = {k: runtime("process option") for k in PROCESS_OPTIONS}
    for cfg in configs:
        for name, pol in field_policies(cfg).items():
            if name in out and (out[name].scope, out[name].resume_mutable, out[name].eval_override) != \
                    (pol.scope, pol.resume_mutable, pol.eval_override):
                raise ConfigError(f"{name}: {out[name].where} and {pol.where} declare different policies")
            out.setdefault(name, pol)
    out.update({dest: out[name] for dest, name in inverted.items() if name in out})
    out.update(extra)
    return out


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """The ``--recipe`` / ``--expect`` options of a resolving entry point."""
    parser.add_argument("--recipe", default=None, metavar="YAML",
                        help="Experiment recipe: a YAML mapping of option dests to values, applied "
                             "under the command line (a flag given there wins; both are recorded).")
    parser.add_argument("--expect", default=None, metavar="YAML",
                        help="Expected config: a YAML mapping key -> value or {value, source}; the "
                             "run stops before building anything when the resolved config differs.")


def recipe_layer(parser: argparse.ArgumentParser, path: str | Path, *,
                 keys: Collection[str] | None = None) -> Layer:
    """The ``recipe:<file>`` layer: a YAML mapping of the parser's dests (only ``keys`` when
    given), each value converted by its option's ``type`` and checked against its ``choices``.
    A :data:`COMMAND_LINE_ONLY` option raises."""
    from configs.loader import load_config

    raw = load_config(path)
    process = sorted(set(raw) & set(COMMAND_LINE_ONLY))
    if process:
        raise ConfigError(f"recipe {path}: {process} steer this process (what to resume or start "
                          "from, the recipe / expect files) and are read before the recipe — give "
                          "them on the command line")
    actions: dict[str, argparse.Action] = {}
    for a in parser._actions:   # first action per dest: an off form (--no-x) carries no type/choices
        if a.default is not argparse.SUPPRESS and (keys is None or a.dest in keys):
            actions.setdefault(a.dest, a)
    unknown = sorted(set(raw) - set(actions))
    if unknown:
        raise ConfigError(f"recipe {path}: {unknown} are not config options of this entry point"
                          + ("" if keys is None else f" (a recipe here sets only {sorted(actions)})"))
    values = {}
    for key, value in raw.items():
        action = actions[key]
        if isinstance(value, str) and callable(action.type):
            value = action.type(value)
        if action.choices is not None and value not in action.choices:
            raise ConfigError(f"recipe {path}: {key}={value!r} is not one of {list(action.choices)}")
        values[key] = value
    return Layer(f"recipe:{path}", values, where={k: f"{path}: {k}" for k in values})


def cli_layer(parser: Any, args: argparse.Namespace, keys: Collection[str] | None = None) -> Layer:
    """The ``cli`` layer of a :class:`~configs.loader.cli.StrictArgumentParser` parse: the dests
    the command line gave — the options it named (``parser.explicit_options``) and every required
    positional —, limited to ``keys`` when given. A repeatable option (``append`` / ``count``)
    given on the command line raises: the parser does not record it as given (it may repeat), so
    its value would be lost."""
    positional = {a.dest: a.dest for a in parser._actions if not a.option_strings and a.required}
    given = {k: opt for k, opt in {**positional, **parser.explicit_options}.items()
             if keys is None or k in keys}
    repeatable = sorted(a.dest for a in parser._actions if isinstance(a, parser._REPEATABLE)
                        and (keys is None or a.dest in keys)
                        and getattr(args, a.dest) != parser.get_default(a.dest))
    if repeatable:
        raise ConfigError(f"{repeatable} are repeatable options — a resolved config takes each "
                          "option once; make it a single-valued option")
    return Layer("cli", {k: getattr(args, k) for k in given},
                 where={k: f"{opt} {getattr(args, k)!r}" for k, opt in given.items()})


def user_layers(parser: Any, args: argparse.Namespace, keys: Collection[str] | None = None
                ) -> list[Layer]:
    """The USER layers of a :class:`~configs.loader.cli.StrictArgumentParser` parse: ``--recipe``
    < :func:`cli_layer`, both limited to ``keys`` when given."""
    cli = cli_layer(parser, args, keys)
    return ([] if args.recipe is None else [recipe_layer(parser, args.recipe, keys=keys)]) + [cli]


def resolve_parsed(parser: Any, args: argparse.Namespace, mode: str, *,
                   policies: Mapping[str, Policy], ckpt: Layer | None = None,
                   mode_defaults: Layer | None = None, code: Layer | None = None) -> RunConfig:
    """Resolve the config a :class:`~configs.loader.cli.StrictArgumentParser` parse describes:
    DEFAULTS = the parser defaults (+ ``mode_defaults``), USER = :func:`user_layers` ;
    ``code``. Checks ``--expect`` on the result."""
    user = user_layers(parser, args) + ([] if code is None else [code])
    run = resolve(mode, policies=policies,
                  defaults=[parser_defaults(parser, policies), *([mode_defaults] if mode_defaults else [])],
                  ckpt=ckpt, user=user, strict_ckpt=ckpt is not None and args.strict_ckpt)
    if args.expect is not None:
        from configs.loader.expect import check_expect
        check_expect(args.expect, run)
    return run


def parser_defaults(parser: argparse.ArgumentParser, policies: Mapping[str, Policy]) -> Layer:
    """The DEFAULTS layer of a parser: every dest with its default (``set_defaults`` included)."""
    values = {a.dest: parser.get_default(a.dest) for a in parser._actions
              if a.default is not argparse.SUPPRESS}
    values.update({k: parser.get_default(k) for k in parser._defaults})
    return Layer("defaults", values, where={k: policies[k].where for k in values if k in policies})


def ckpt_layer(values: Mapping[str, Any], sources: Mapping[str, str], where: str,
               meta: Mapping[str, Any] = MappingProxyType({})) -> Layer:
    """The CKPT layer of :func:`configs.loader.ckpt.load_ckpt_config` output."""
    return Layer("ckpt", values, where={k: where for k in values}, sources=sources, meta=meta)


__all__ = ["COMMAND_LINE_ONLY", "MODES", "PROCESS_OPTIONS", "ConfigError", "Entry", "Layer", "RunConfig", "add_run_args",
           "ckpt_layer", "cli_layer", "dest_policies", "parser_defaults", "recipe_layer", "resolve", "resolve_parsed",
           "same_value", "user_layers"]
