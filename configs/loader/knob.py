"""Config field declaration: every schema field names its scope and what may override it.

:func:`knob` is the ``dataclasses.field`` every leaf field of :mod:`configs.loader.schema` is
declared with. Besides the CLI metadata :func:`configs.loader.cli.add_dataclass_args` reads
(``help`` / ``choices`` / ``metavar`` / ``cli_skip``) it carries the field's override policy,
which :func:`configs.loader.resolve.resolve` enforces (the scope × mode table):

* ``scope`` — ``arch`` (the weights, or the observation/action space they were trained on,
  depend on it: equal to the checkpoint whenever there is one), ``experiment`` (env, reward,
  masking, optimisation: pinned to the checkpoint on resume and eval unless the flags below
  say otherwise) or ``runtime`` (process plumbing: free in every mode).
* ``resume_mutable`` — an experiment field a resumed run may change (recorded).
* ``eval_override`` — an experiment field an eval of a checkpoint may change (recorded).

Changing a policy is a contract change, so it lives here in code (reviewed), not in YAML.
"""
from __future__ import annotations

import dataclasses
from dataclasses import MISSING, dataclass
from typing import Any

SCOPES = ("arch", "experiment", "runtime")


def knob(default: Any = MISSING, scope: str | None = None, *, default_factory: Any = MISSING,
         resume_mutable: bool = False, eval_override: bool = False, cli_skip: bool = False,
         help: str | None = None, choices: Any = None, metavar: Any = None, off_flag: bool = False,
         readback: Any = MISSING, readback_reason: str | None = None) -> Any:
    """A schema field: its default (from YAML), its :data:`SCOPES` member and override policy,
    its CLI metadata (``off_flag``: an optional field whose default is set also gets
    ``--no-<flag>``, which sets it to None), and — an ``EnvConfig`` field — where ``PCBWorld.effective_config`` reads
    it back (``readback``: an attribute path, or ``None`` with a ``readback_reason``)."""
    if scope not in SCOPES:
        raise ValueError(f"knob scope must be one of {SCOPES}, got {scope!r}")
    if (resume_mutable or eval_override) and scope != "experiment":
        raise ValueError("resume_mutable / eval_override relax an experiment field only "
                         f"(scope={scope!r})")
    meta = {"scope": scope, "resume_mutable": resume_mutable, "eval_override": eval_override}
    if readback is not MISSING:
        meta["readback"] = readback
    for key, value in (("cli_skip", cli_skip or None), ("off_flag", off_flag or None), ("help", help), ("choices", choices),
                       ("metavar", metavar), ("readback_reason", readback_reason)):
        if value is not None:
            meta[key] = value
    return dataclasses.field(default=default, default_factory=default_factory, metadata=meta)


@dataclass(frozen=True)
class Policy:
    """The override policy of one config key; ``where`` names its declaration."""

    scope: str
    resume_mutable: bool
    eval_override: bool
    where: str


def field_policies(config: Any) -> dict[str, Policy]:
    """``{field name: Policy}`` over the leaf fields of the dataclass instance ``config``
    (nested dataclass fields are flattened, like the CLI). A leaf without a scope raises."""
    out: dict[str, Policy] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if dataclasses.is_dataclass(value):
            out.update(field_policies(value))
            continue
        where = f"{type(config).__name__}.{f.name}"
        if "scope" not in f.metadata:
            raise ValueError(f"{where} declares no scope — declare it with configs.loader.knob.knob")
        out[f.name] = Policy(f.metadata["scope"], f.metadata["resume_mutable"],
                             f.metadata["eval_override"], where)
    return out


def runtime(where: str) -> Policy:
    """The policy of a parser option that is not a schema field and only steers the process."""
    return Policy("runtime", False, False, where)


__all__ = ["SCOPES", "Policy", "field_policies", "knob", "runtime"]
