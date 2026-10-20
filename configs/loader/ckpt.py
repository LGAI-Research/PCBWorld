"""Checkpoint config loader — the only code that reads a checkpoint's ``config`` / ``args``.

:func:`load_ckpt_config` turns a training checkpoint payload (``torch.load`` of a
``policy_*.pt``) — its schema-2 ``config`` or, older, its ``args`` — into ``(values, sources)``:

* ``values`` — a read-only mapping holding every arg the checkpoint stores plus
  every :data:`CKPT_FIELDS` key, so readers index ``values[k]`` and never carry a
  default of their own;
* ``sources`` — for every :data:`CKPT_FIELDS` key, where its value came from:
  ``ckpt`` | ``renamed`` | ``derived`` | ``legacy`` | ``legacy:proxy`` |
  ``legacy:partial``.

Steps, in order:

1. **RENAMES** — keys no writer stores any more: ``directional_grid_size`` →
   ``directional_candidates="grid<N>"``, ``use_yaml_drc_fallback`` → ``rules_source``
   (:data:`RULES_SOURCE_OF_YAML_FALLBACK`; beside a stored ``rules_source`` — the mirror
   :func:`legacy_args` writes — it must agree and is dropped), the inverted training flags of
   :data:`INVERTED_RENAMES` (``no_x`` → ``x = not no_x``); :data:`REMOVED` keys are dropped and
   named in the warning. The trainer stores ``emit_drc_tokens`` /
   ``mask_start_point`` under their inverted CLI dests (``no_drc_tokens`` /
   ``no_mask_start_point``, :data:`CLI_FORMS`); the field is added beside the
   stored dest, so a training namespace rebuilt from ``values`` still parses.
2. **DERIVED** — from the weights (:func:`_derive`): ``shape_obs``,
   ``legacy_pad_layer_encoding``, ``legacy_net_encoding``,
   ``legacy_edge_encoding``, ``legacy_action_history`` (+ ``action_history_len=1``
   for the single-entry layout), ``use_critic``, ``n_max_slots``. A stored value
   that contradicts the weights raises.
3. **LEGACY** — ``configs/ckpt_migrations.yaml`` (``ckpt_args:``): the pre-knob
   value of a key the checkpoint predates. ``reject_if_stuck`` was never stored;
   its value follows the ``shove_iter_limit`` proxy (see the table).
4. **CHECK** — a :data:`CKPT_FIELDS` key still missing raises (add it to the
   table). A key missing from an ``args`` checkpoint is always filled from the table,
   whatever its ``provenance.repo_version`` (public and private version lines differ,
   so a version cannot tell an old checkpoint from a writer bug); the warning names it
   and ``--strict-ckpt`` refuses it.

**Field set** :data:`CKPT_FIELDS` = every field of ``EnvConfig`` (reward
overrides flattened), the RL-wrapper fields of ``RLEnvConfig`` and every field of
``RLPolicyConfig``, minus :data:`RUNTIME_FIELDS` — values that are not a property
of the trained policy and that ``RLEnvConfig.from_checkpoint`` pins itself
(``obs_format``, train-only ``reward_noise_std`` / ``keep_routing_fraction`` /
``slot_perm`` / ``aug_*``). ``corner_mode`` is the routing angle (45 or 90);
anything else raises.

**Schema 2.** A checkpoint the trainers write now stores ``config: {schema: 2, mode, values,
sources}`` (:meth:`configs.loader.resolve.RunConfig.ckpt_config`; beside it the same values as
``args`` for older loaders, :func:`ckpt_entries`, which this loader ignores): its values are read
as stored — a value whose recorded source is ``legacy*`` (through ``init_from`` for a key a BC
checkpoint took from its base) keeps that source, the rest are ``ckpt``. CLI forms and weight
derivation apply, renames do not, the legacy
table only for a key its writer (``payload["trainer"]``, :data:`TRAINERS`) never records
(``trainers:``, e.g. GRPO's ``detach_critic``),
and any other missing field raises (a writer bug). It loads without a warning.

**Old checkpoints** (``args``, no ``config``) are read through the steps above. A load warns
once per process for each distinct set of resolved keys (legacy / derived / renamed keys with
their sources; a checkpoint without ``obs_probe`` — pre-v0.31 — is named as such);
``strict=True`` (``--strict-ckpt`` on every checkpoint-consuming entry point,
:func:`add_strict_ckpt_arg`) rejects it instead. The warned messages are the
module global ``_warned`` — the only process state here; ``strict`` is always
passed explicitly.

Torch-free at import; :func:`read_ckpt_config` imports torch lazily.
"""
from __future__ import annotations

import warnings
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from configs.loader import load_config
from configs.loader.schema import (
    CHECK_ANGLES, INVERTED_DESTS, RUNTIME_FIELDS, TRAIN_LAYER, EnvConfig, RewardOverrides, RLEnvConfig,
    RLPolicyConfig, _none_if_off,
)

_MIGRATIONS_PATH = Path(__file__).resolve().parent.parent / "ckpt_migrations.yaml"
_MIGRATIONS = load_config(_MIGRATIONS_PATH)
_TABLE = _MIGRATIONS["ckpt_args"]
#: ``key -> table entry`` over every section of ``ckpt_args:`` in configs/ckpt_migrations.yaml.
LEGACY: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    k: v for section in _TABLE.values() for k, v in section.items()
})

#: Keys no writer stores any more and that map to no field (dropped, named in the warning).
REMOVED = ("allow_incomplete_net_end",)   # v0.11.26: the masking rule owns the net_end precondition

#: Inverted training flags (``--no-x`` stored ``no_x``) the ``--x`` / ``--no-x`` pair replaced
#: (6320a1c66): ``stored dest -> field``.
INVERTED_RENAMES: Mapping[str, str] = MappingProxyType({
    "no_truncation_bootstrap": "truncation_bootstrap", "no_norm_reward": "norm_reward",
    "no_normalize_adv": "normalize_adv"})

#: Stored ``use_yaml_drc_fallback`` (the bool ``rules_source`` replaced) -> ``rules_source``.
RULES_SOURCE_OF_YAML_FALLBACK: Mapping[bool, str] = MappingProxyType(
    {False: "board", True: "board_or_yaml"})
#: ``rules_source`` -> the ``use_yaml_drc_fallback`` a loader before schema 2 reads.
YAML_FALLBACK_OF_RULES_SOURCE: Mapping[str, bool] = MappingProxyType(
    {v: k for k, v in RULES_SOURCE_OF_YAML_FALLBACK.items()})

#: Checkpoint writers, stored as ``payload["trainer"]`` (the table's ``trainers:``).
TRAINERS = ("ppo", "grpo", "bc")

#: Every config field a checkpoint must yield (stored, renamed, derived or legacy).
CKPT_FIELDS = frozenset({f.name for cls in (EnvConfig, RewardOverrides, RLEnvConfig, RLPolicyConfig)
                         for f in fields(cls)} - {"env", "reward"} - RUNTIME_FIELDS)

#: Fields the trainer stores under an inverted CLI dest: ``field -> stored dest``.
CLI_FORMS: Mapping[str, str] = MappingProxyType(
    {name: dest for dest, name in INVERTED_DESTS.items() if name in CKPT_FIELDS})

#: ``key -> value`` an env record written before the key was recorded lacks: the table's
#: ``env_records:`` value, else the train-only layer's default (off), else the ``ckpt_args:``
#: ``env`` value.
_RECORD_FILL: Mapping[str, Any] = MappingProxyType({
    **{k: e["value"] for k, e in _TABLE["env"].items() if "value" in e},
    **{k: v for k, v in RLEnvConfig().to_record().items() if k in TRAIN_LAYER},
    **_MIGRATIONS["env_records"],
})

#: Engine corner codes an env record stored before ``corner_mode`` became degrees
#: (``pcb_world.core.env.corner_mode_code``: MITERED_45 / MITERED_90).
_CORNER_CODE_ANGLE = {0: 45, 2: 90}

_warned: set[str] = set()


class CkptConfigError(ValueError):
    """The checkpoint's args cannot be resolved (or strict mode refused them)."""


def add_strict_ckpt_arg(parser: Any) -> None:
    """The ``--strict-ckpt`` flag every checkpoint-consuming entry point carries."""
    parser.add_argument(
        "--strict-ckpt", action="store_true",
        help="Refuse a checkpoint without a schema-2 config (one written before the "
             "trainers stored `config`) instead of filling its missing keys from "
             "configs/ckpt_migrations.yaml with a warning.")


def ckpt_entries(config: Mapping[str, Any]) -> dict[str, Any]:
    """The config entries a writer puts in a checkpoint: the schema-2 ``config`` and, for loaders
    from before schema 2 (an older clone reading a new checkpoint), the same values as ``args``
    — with ``use_yaml_drc_fallback``, the key those loaders read instead of ``rules_source``.
    :func:`load_ckpt_config` reads ``config`` and ignores ``args`` beside it."""
    return {"config": config, "args": legacy_args(config["values"])}


def legacy_args(values: Mapping[str, Any]) -> dict[str, Any]:
    """Resolved config ``values`` as the ``args`` a loader before schema 2 reads."""
    return {**values, "use_yaml_drc_fallback": YAML_FALLBACK_OF_RULES_SOURCE[values["rules_source"]]}


def _recorded_source(cfg: Mapping[str, Any], key: str) -> str:
    """``legacy*`` when a schema-2 ``cfg`` recorded ``key`` as an estimate (directly or, for a key
    a BC checkpoint took from its base, in ``init_from.sources``), else ``ckpt``."""
    where = cfg["sources"]
    if key in where and where[key] == "init_from":
        where = cfg["init_from"]["sources"]
    if key not in where:
        raise CkptConfigError(f"schema-2 checkpoint config records {key!r} without a source "
                              "(a writer bug)")
    src = where[key]
    return src if src.partition(":")[0] == "legacy" else "ckpt"


def read_ckpt_config(path: str | Path, *, strict: bool):
    """:func:`load_ckpt_config` of the checkpoint file at ``path`` (CPU load)."""
    import torch

    return load_ckpt_config(torch.load(path, map_location="cpu", weights_only=False),
                            strict=strict)


def _derive(state_dict: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Config values the weights fix: their presence or shape is the record."""
    def required(key: str) -> Any:
        if key not in state_dict:
            raise CkptConfigError(f"policy_state_dict has no {key!r}; cannot derive the "
                                  "tokenizer layout from the weights")
        return state_dict[key]

    out: dict[str, Any] = {
        # Modules built only when the knob is on: presence is the knob.
        "shape_obs": "tokenizer.vocab.shape_embed.weight" in state_dict,
        "legacy_edge_encoding": "tokenizer.vocab.edge_mid_proj.weight" not in state_dict,
        "legacy_action_history": "tokenizer.vocab.history_age_proj.weight" not in state_dict,
        "use_critic": any(k.startswith("critic_head.") for k in state_dict),
        "n_max_slots": int(required("tokenizer.vocab.slot_emb_table").shape[0]),
    }
    if out["legacy_action_history"]:   # one prev-action entry, no age projection
        out["action_history_len"] = 1
    for key in ("coord_encoding", "n_freq", "d_model"):
        if key not in values:
            raise CkptConfigError(f"checkpoint args lack {key!r}, needed to read the "
                                  "tokenizer layout")
    f = 4 * int(values["n_freq"]) if values["coord_encoding"] == "fourier" else int(values["d_model"])
    pad_in = int(required("tokenizer.vocab.pad_proj.weight").shape[1])
    net_in = int(required("tokenizer.vocab.net_proj.weight").shape[1])
    layouts = {"legacy_pad_layer_encoding": (pad_in, 5 * f + 1, 6 * f + 1),
               "legacy_net_encoding": (net_in, 3 * f, 4 * f)}   # 4th NET term: per-episode `closed`
    for key, (got, legacy, current) in layouts.items():
        if got not in (legacy, current):
            raise CkptConfigError(f"{key}: tokenizer input dimension {got}, expected "
                                  f"{legacy} (legacy) or {current} (current)")
        out[key] = got == legacy
    return out


def load_ckpt_config(payload: Mapping[str, Any], *, strict: bool
                     ) -> tuple[Mapping[str, Any], Mapping[str, str]]:
    """Resolve a checkpoint payload into ``(values, sources)`` — module docstring."""
    if payload.get("trainer") not in (None, *TRAINERS):
        raise CkptConfigError(f"unknown checkpoint trainer {payload['trainer']!r} "
                              f"(expected one of {TRAINERS})")
    legacy = "config" not in payload   # pre-schema-2: stored `args`, keys filled below
    if not legacy:
        cfg = payload["config"]
        if not isinstance(cfg, dict) or cfg.get("schema") != 2:
            raise CkptConfigError(f"checkpoint config is not schema 2: {str(cfg)[:80]!r}")
        stored, removed = cfg["values"], []
    elif isinstance(payload.get("args"), dict):
        stored = payload["args"]
    else:
        raise CkptConfigError("checkpoint payload has neither a 'config' (schema 2) nor an 'args' dict")
    values = dict(stored)
    sources = {k: "ckpt" if legacy else _recorded_source(cfg, k) for k in CKPT_FIELDS if k in stored}

    # 1. RENAMES (pre-schema-2) — the int grid size became the "grid<N>" mode; the newer key
    # wins unless it is unset/off (the grid then still applies).
    if legacy:
        if "directional_grid_size" in values:
            grid = values.pop("directional_grid_size")
            if "directional_candidates" not in values or (
                    grid is not None and _none_if_off(values["directional_candidates"]) is None):
                values["directional_candidates"] = None if grid is None else f"grid{int(grid)}"
                sources["directional_candidates"] = "renamed"
        if "use_yaml_drc_fallback" in values:   # beside rules_source: the mirror legacy_args writes
            old = RULES_SOURCE_OF_YAML_FALLBACK[bool(values.pop("use_yaml_drc_fallback"))]
            if "rules_source" not in values:
                values["rules_source"], sources["rules_source"] = old, "renamed"
            elif values["rules_source"] != old:
                raise CkptConfigError(f"checkpoint args contradict: rules_source={values['rules_source']!r} "
                                      f"but use_yaml_drc_fallback reads as {old!r}")
        for dest, name in INVERTED_RENAMES.items():
            if dest in values:
                value = not bool(values.pop(dest))
                if name in values and values[name] != value:
                    raise CkptConfigError(f"checkpoint args contradict: {name}={values[name]!r} "
                                          f"but {dest} reads as {value!r}")
                values[name], sources[name] = value, "renamed"
        removed = [k for k in REMOVED if k in values]
        for k in removed:
            del values[k]
    for field_name, dest in CLI_FORMS.items():
        if dest in values:
            values[field_name] = not bool(values[dest])
            sources[field_name] = "ckpt"

    # 2. DERIVED
    for key, value in _derive(payload["policy_state_dict"], values).items():
        if key in values and values[key] != value:
            raise CkptConfigError(
                f"checkpoint args/weights contradiction: args say {key}={values[key]!r} "
                f"but the weights say {value!r} — refusing to guess which is the training truth")
        if key not in values:
            values[key], sources[key] = value, "derived"

    # 3. LEGACY. A schema-2 config holds every field its writer has; a table key its writer
    # never records (``trainers:``) still takes the table value.
    if legacy:
        _fill_legacy(stored, values, sources)
    else:
        for key in sorted(CKPT_FIELDS - values.keys()):
            entry = LEGACY.get(key)
            if entry is not None and payload["trainer"] not in entry.get("trainers", TRAINERS):
                values[key], sources[key] = entry["value"], "legacy"

    # 4. CHECK
    missing = sorted(CKPT_FIELDS - values.keys())
    if missing:
        raise CkptConfigError(
            f"checkpoint lacks {missing} and they are neither derivable from its weights "
            + ("nor listed in configs/ckpt_migrations.yaml — add an entry (the pre-knob value) there"
               if legacy else "— a schema-2 config records every field (writer bug)"))
    if values["corner_mode"] not in CHECK_ANGLES:
        raise CkptConfigError(f"corner_mode must be one of {CHECK_ANGLES} (degrees), "
                              f"got {values['corner_mode']!r}")
    if legacy:
        _warn_pre_schema2(payload, sources, removed, strict=strict)
    return MappingProxyType(values), MappingProxyType(sources)


def _fill_legacy(stored: Mapping[str, Any], values: dict[str, Any], sources: dict[str, str]) -> None:
    """Step 3: the ``ckpt_args:`` value of every table key the checkpoint lacks."""
    for key, entry in LEGACY.items():
        if key in values:
            continue
        if "proxy" in entry:
            present = entry["proxy"] in stored
            values[key] = entry["value_if_present" if present else "value_if_absent"]
            sources[key] = "legacy:proxy" if present else "legacy:partial"
        else:
            values[key] = entry["value"]
            sources[key] = "legacy"


def _warn_pre_schema2(payload: Mapping[str, Any], sources: Mapping[str, str], removed: list[str],
                      *, strict: bool) -> None:
    """Warn once per distinct message about a pre-schema-2 checkpoint; ``strict`` refuses it."""
    filled = {k: s for k, s in sorted(sources.items()) if s != "ckpt"}
    notes = [f"{k} ({s})" for k, s in filled.items()]
    notes += [f"{k} (removed — stored value ignored)" for k in removed]
    if not payload.get("obs_probe"):
        notes.append("no obs_probe (pre-v0.31 checkpoint: obs semantics are not verified)")
    msg = ("pre-schema-2 checkpoint (no recorded config): keys resolved outside the "
           "stored args — " + "; ".join(notes))
    if strict:
        raise CkptConfigError(f"--strict-ckpt: {msg}")
    if msg not in _warned:
        _warned.add(msg)
        warnings.warn(msg + " (warned once per process; --strict-ckpt refuses such "
                      "checkpoints)", stacklevel=3)


def with_legacy_env_defaults(record: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a recorded env dict (a run's ``run_record.val.yaml`` values, or an old run's
    ``env_records/val_env.yaml``) the way :func:`load_ckpt_config` fills that run's checkpoint.
    A key of the record format (``RLEnvConfig.to_record``) the record predates gets the value
    its env was built with (:data:`_RECORD_FILL`), whatever version wrote the record.
    A ``corner_mode`` recorded as the engine code (records before it became degrees)
    reads as its angle; a recorded ``use_yaml_drc_fallback`` reads as its ``rules_source``."""
    out = dict(record)
    if "use_yaml_drc_fallback" in out:
        out["rules_source"] = RULES_SOURCE_OF_YAML_FALLBACK[bool(out.pop("use_yaml_drc_fallback"))]
    for k in sorted(_RECORD_FILL.keys() - out.keys()):
        out[k] = _RECORD_FILL[k]
    if "corner_mode" in out and out["corner_mode"] in _CORNER_CODE_ANGLE:
        out["corner_mode"] = _CORNER_CODE_ANGLE[out["corner_mode"]]
    return out
