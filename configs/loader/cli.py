"""Generate argparse arguments from a config dataclass.

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


def _flag(name: str, style: str) -> str:
    """Field name → CLI flag (``dash`` = ``--a-b``, ``underscore`` = ``--a_b``)."""
    return "--" + (name.replace("_", "-") if style == "dash" else name)


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Return ``(inner_type, is_optional)`` for ``X | None`` / ``Optional[X]``."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is getattr(__import__("types"), "UnionType", None):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


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
            if value:
                raise ValueError(
                    f"{cls.__name__}.{f.name}: default-True bool needs an "
                    f"explicit inverted flag (set metadata cli_skip=True)"
                )
            parser.add_argument(flag, action="store_true", default=value,
                                help=help_txt)
            continue

        if "choices" in f.metadata:
            kwargs["choices"] = f.metadata["choices"]
        # str args use argparse's default (identity) type — only int/float need
        # an explicit converter (matches the hand-written parsers).
        if tp in (int, float):
            kwargs["type"] = tp
        parser.add_argument(flag, **kwargs)


__all__ = ["add_dataclass_args"]
