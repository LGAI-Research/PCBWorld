"""The repo's argparse parser class, and argparse arguments generated from a config dataclass.

:class:`StrictArgumentParser` is the parser every entry point builds (``engine/`` aside — a standalone
bundle): exact option names only, each dest at most once, and the set of dests the command line
gave kept on the parser.

The companion to :mod:`configs.loader.schema`: the dataclasses there are the single
*visible* list of every run variable + its default (YAML-backed). This module
turns those fields into argparse flags so the CLI is *derived* from the schema
rather than hand-maintained in parallel (no add-a-field-but-forget-the-flag
drift). Both branches call :func:`add_dataclass_args` (RL with dash-style flags,
LLM/eval with underscore-style) — one symmetric path.

Mapping (per ``dataclasses.field``), default = the field's value on the passed
instance (so YAML/override flows through):

  * nested dataclass field  → recurse (flatten, e.g. ``EnvConfig.reward``)
  * ``bool`` (default False) → ``--flag`` ``store_true``
  * ``int`` / ``float`` / ``str``        → ``--flag`` ``type=...`` ``default=...``
  * ``X | None`` (Optional)  → ``type=X`` ``default=None``

Per-field ``metadata`` keys (``dataclasses.field(metadata={...})``):

  * ``cli_skip``  : ``True`` → don't emit (handled explicitly — legacy-named
    inverted bools like ``--no-drc-tokens``, sugar like ``--corner-mode``,
    or required positionals like ``--board``). The variable still lives in the
    schema (visible); only its bespoke CLI form is added by the caller.
  * ``help``      : help string.
  * ``choices``   : argparse choices (applied to both branches).
  * ``metavar``   : argparse metavar.
  * ``off_flag``  : ``True`` → also emit ``--no-<name>`` storing ``None`` (an optional value
    whose default is set).

Anything the generator can't represent (inverted/sugar/required) is listed in
each field's ``cli_skip`` and added explicitly by the small tail in the args
builders — but every such variable is still a schema field, so the full list is
readable in one place (``configs/loader/schema.py``).
"""
from __future__ import annotations

import argparse
import dataclasses
import typing
from typing import Any

from configs.loader import _unwrap_optional


_ENGINE_KNOB = "no flag any more: set env.{} in configs/env.yaml"
#: Options a released version had and this one dropped → what to do instead. A parser names the ones
#: it had (``StrictArgumentParser(removed=...)``); given one of those, its parse error ends with the text
#: (the option is never accepted).
REMOVED_OPTIONS: dict[str, str] = {
    "--use-yaml-drc-fallback": "use --rules-source board_or_yaml",
    "--no-use-yaml-drc-fallback": "use --rules-source board (the default)",
    "--use_yaml_drc_fallback": "use --rules_source board_or_yaml",
    "--no-use_yaml_drc_fallback": "use --rules_source board (the default)",
    "--finetune-lr": "drop it: under --init-ckpt the default --lr is the fine-tune one",
    "--finetune-clip-eps": "drop it: under --init-ckpt the default --clip-eps is the fine-tune one",
    "--engine-seed": _ENGINE_KNOB.format("engine_seed"),
    "--shove-iter-limit": _ENGINE_KNOB.format("shove_iter_limit"),
    "--followbranch-iter-limit": _ENGINE_KNOB.format("followbranch_iter_limit"),
    "--easy-train-only-glob": "drop it: D3-A excludes 0096_karabas-nano_wifi_revA for good (configs/datasets/README.md)",
}
#: The removed options of the RL trainers (v1.0.1 ``train_ppo`` / ``train_grpo``).
REMOVED_TRAIN_OPTIONS = ("--use-yaml-drc-fallback", "--no-use-yaml-drc-fallback", "--engine-seed",
                         "--shove-iter-limit", "--followbranch-iter-limit")


def _removed_options(parser: argparse.ArgumentParser) -> set[str]:
    """The removed options ``parser`` and its subcommands' parsers name."""
    found = set(getattr(parser, "removed_options", ()))
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                found |= _removed_options(sub)
    return found


class StrictArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` that sets each dest at most once and records which dests the command line set.

    * ``allow_abbrev=False`` by default — ``--clip 0.3`` is an error, not ``--clip-eps 0.3``.
    * Setting one dest twice — the same flag twice, ``--x`` and ``--no-x``, or two aliases — is a parse
      error instead of argparse's silent last-wins. Accumulating actions (``append``, ``append_const``,
      ``extend``, ``count``) exist to repeat and are exempt.
    * An unrecognized option that a released version of this parser had (``removed=``, keys of
      :data:`REMOVED_OPTIONS`) is still an error, with its replacement named.
    * After a parse, :attr:`explicit_dests` (:attr:`explicit_options`: with the option used) holds the dests of the options the command line gave, so a
      caller can tell "given" from "left at its default". It lives on the parser, not on the Namespace,
      because ``vars(args)`` is what checkpoints record. A subcommand's options are recorded on its own
      subparser (``add_subparsers`` builds subparsers of the parent's class).
    """

    _REPEATABLE = (argparse._AppendAction, argparse._AppendConstAction, argparse._CountAction)

    def __init__(self, *args: Any, allow_abbrev: bool = False, removed: tuple[str, ...] = (),
                 **kwargs: Any) -> None:
        super().__init__(*args, allow_abbrev=allow_abbrev, **kwargs)
        unknown = [opt for opt in removed if opt not in REMOVED_OPTIONS]
        if unknown:
            raise KeyError(f"{unknown} are not in REMOVED_OPTIONS")
        self.removed_options = tuple(removed)
        self._given: dict[str, argparse.Action] | None = None
        self._parsing = False

    @property
    def explicit_dests(self) -> frozenset[str]:
        """Dests set by an option on the command line of the last parse."""
        return frozenset(self._given_actions())

    @property
    def explicit_options(self) -> dict[str, str]:
        """``{dest: the option string that set it}`` for :attr:`explicit_dests`."""
        return {dest: action.option_strings[0] for dest, action in self._given_actions().items()}

    def _given_actions(self) -> dict[str, argparse.Action]:
        if self._given is None:
            raise RuntimeError("explicit_dests read before the parser parsed a command line")
        return self._given

    def parse_known_args(self, args: Any = None, namespace: Any = None) -> Any:
        return self._recording(super().parse_known_args, args, namespace)

    def parse_known_intermixed_args(self, args: Any = None, namespace: Any = None) -> Any:
        return self._recording(super().parse_known_intermixed_args, args, namespace)

    def _recording(self, parse: Any, args: Any, namespace: Any) -> Any:
        # Every public parse method ends in one of the two above; the intermixed one parses in two
        # internal passes (through ``parse_known_args`` on older Pythons), so only the outermost call
        # starts a new record — one command line, one record.
        if self._parsing:
            return parse(args, namespace)
        self._given, self._parsing = {}, True
        try:
            return parse(args, namespace)
        finally:
            self._parsing = False

    def parse_args(self, args: Any = None, namespace: Any = None) -> Any:
        parsed, extras = self.parse_known_args(args, namespace)
        self._reject_unrecognized(extras)
        return parsed

    def parse_intermixed_args(self, args: Any = None, namespace: Any = None) -> Any:
        parsed, extras = self.parse_known_intermixed_args(args, namespace)
        self._reject_unrecognized(extras)
        return parsed

    def _reject_unrecognized(self, extras: list[str]) -> None:
        # argparse's own check, plus the replacement of each removed option (:data:`REMOVED_OPTIONS`).
        if not extras:
            return
        msg = argparse._("unrecognized arguments: %s") % " ".join(extras)
        had = _removed_options(self)
        hints = [f"{opt}: {REMOVED_OPTIONS[opt]}"
                 for opt in dict.fromkeys(tok.split("=", 1)[0] for tok in extras) if opt in had]
        if hints:
            msg += " — removed: " + "; ".join(hints)
        if self.exit_on_error:
            self.error(msg)
        raise argparse.ArgumentError(None, msg)

    def _get_values(self, action: argparse.Action, arg_strings: list[str]) -> Any:
        # argparse calls this once per option occurrence, just before the action stores its value —
        # after ``--flag=value``, short-option clusters and the ``--`` terminator are resolved.
        if action.option_strings and not isinstance(action, self._REPEATABLE):
            prev = self._given.get(action.dest)
            if prev is action:
                raise argparse.ArgumentError(action, "given more than once")
            if prev is not None:
                raise argparse.ArgumentError(
                    action, f"conflicts with {'/'.join(prev.option_strings)} given earlier "
                            f"(both set {action.dest!r}; give one)")
            self._given[action.dest] = action
        return super()._get_values(action, arg_strings)


def _flag(name: str, style: str) -> str:
    """Field name → CLI flag (``dash`` = ``--a-b``, ``underscore`` = ``--a_b``)."""
    return "--" + (name.replace("_", "-") if style == "dash" else name)


def add_dataclass_args(
    parser: argparse._ActionsContainer,
    config: Any,
    *,
    style: str = "dash",
    skip: tuple[str, ...] = (),
) -> None:
    """Append one argparse argument per (non-skipped) field of ``config``.

    ``config`` is a dataclass *instance* (defaults = its field values, so a
    YAML-loaded / overridden instance flows its values through as the CLI
    defaults). Nested dataclass fields are flattened. ``style`` picks the flag
    naming; ``skip`` (plus per-field ``metadata['cli_skip']``) excludes fields
    handled explicitly by the caller.
    """
    cls = type(config)
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if f.name in skip or f.metadata.get("cli_skip"):
            continue
        value = getattr(config, f.name)
        if dataclasses.is_dataclass(value):
            add_dataclass_args(parser, value, style=style, skip=skip)
            continue

        tp, _opt = _unwrap_optional(hints.get(f.name, type(value)))
        flag = _flag(f.name, style)
        help_txt = f.metadata.get("help", "")
        kwargs: dict[str, Any] = {"default": value, "help": help_txt}
        if "metavar" in f.metadata:
            kwargs["metavar"] = f.metadata["metavar"]

        if tp is bool:
            # Every bool gets the pair ``--flag`` / ``--no-flag`` writing the same dest, with
            # the YAML value as the default. Whichever flag matches the default is a no-op, so
            # a launch script written under the other default keeps working when the default
            # flips (e.g. ``--no-truncation-bootstrap`` after it became off by default).
            parser.add_argument(flag, dest=f.name, action="store_true", default=value,
                                help=help_txt + (" (on by default)" if value else ""))
            parser.add_argument("--no-" + flag[2:], dest=f.name, action="store_false",
                                help=f"Turn off {flag}." + ("" if value else " (off by default)"))
            continue

        if "choices" in f.metadata:
            kwargs["choices"] = f.metadata["choices"]
        # str args use argparse's default (identity) type — only int/float need
        # an explicit converter (matches the hand-written parsers).
        if tp in (int, float):
            kwargs["type"] = tp
        parser.add_argument(flag, **kwargs)
        if f.metadata.get("off_flag"):
            if not _opt:
                raise TypeError(f"{cls.__name__}.{f.name}: off_flag needs an optional field")
            parser.add_argument("--no-" + flag[2:], dest=f.name, action="store_const", const=None,
                                help=f"Unset {flag} (None).")


__all__ = ["REMOVED_OPTIONS", "REMOVED_TRAIN_OPTIONS", "StrictArgumentParser", "add_dataclass_args"]
