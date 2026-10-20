"""Config-loading code for the KiCad RL environment (the code half of ``configs/``).

The sibling directories under ``configs/`` are pure data (YAML/JSON); this
subpackage holds everything importable: YAML IO (``load_config`` /
``merge_configs``), rule-file validation (``check_mapping`` / ``load_rule_file``), the
typed schema (``schema``), the dataset/output path resolver (``paths``) and the CLI (``cli``).
"""

from __future__ import annotations

import copy
import functools
import subprocess
import types as _types
import typing
import warnings
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

# ``X | Y`` hints (3.10+). The push gate runs check_docs, which imports this package, under Python 3.9,
# where every union is ``typing.Union``.
UnionType = getattr(_types, "UnionType", typing.Union)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CONFIGS_DIR.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Parsed configuration as a nested dict.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_rule_file(path: str | Path, kind: str) -> Any:
    """Load a rule YAML of *kind* (``reward`` / ``masking``) for its schema.

    A file outside the tracked tree (an old checkpoint's rule written before keys became
    required) is migrated by ``configs/ckpt_migrations.yaml`` ``rule_files.<kind>`` — keys
    it lacks are filled, retired keys dropped — with one warning naming the file and the
    keys. A reward rule in the format before ``potential:`` is first converted by
    ``old_reward_format`` (:func:`_from_old_reward_format`). A tracked file is returned as
    written, so its schema refuses a missing key or the old format.
    """
    from pcb_world.core.reward_config import is_old_reward_format

    raw = load_config(path)
    old_format = kind == "reward" and is_old_reward_format(raw)
    migrated, keys = _legacy_fill(_from_old_reward_format(raw) if old_format else raw,
                                  _migrations()["rule_files"][kind])
    if not (keys or old_format) or _is_tracked(path):
        return raw
    old = _migrations()["old_reward_format"]
    ignored = sorted(set(raw) - {"name", "mode", "step_reward", "final_reward", *old["penalties"]}) if old_format else []
    what = (["the reward format before `potential:` (converted as the old loader read it"
             + (f"; ignored top-level keys {ignored}" if ignored else "") + ")"] if old_format else [])
    what += [f"legacy keys {keys}"] if keys else []
    warnings.warn(
        f"{kind} rule {path} predates the current schema; migrated {', '.join(what)} — update the file to match.",
        UserWarning,
        stacklevel=3,
    )
    return migrated


@functools.lru_cache(maxsize=None)
def _migrations() -> dict[str, Any]:
    """``configs/ckpt_migrations.yaml`` (parsed once; never mutated)."""
    return load_config(_CONFIGS_DIR / "ckpt_migrations.yaml")


def _from_old_reward_format(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The ``potential:`` rule the old loader built from an old-format reward rule
    (``old_reward_format`` in ``configs/ckpt_migrations.yaml`` says how)."""
    old = _migrations()["old_reward_format"]
    final = raw.get("final_reward") or {}
    potential = {**old["penalties"], **{k: raw[k] for k in old["penalties"] if k in raw},
                 **old["final_reward"], **{k: final[k] for k in old["final_reward"] if k in final}}
    potential["drc_penalty"] = (final.get("drc_penalty", old["drc_penalty"])
                                if final.get("use_drc", old["use_drc"]) else 0.0)
    return {**{k: raw[k] for k in ("name", "mode") if k in raw}, "potential": potential}


def _legacy_fill(raw: Any, fill: Mapping[str, Any], prefix: str = "") -> tuple[Any, list[str]]:
    """Return a copy of *raw* migrated by *fill*, and the dotted names of the keys it changed.

    A non-empty mapping in *fill* is a nested table, applied only where *raw* has that key —
    except a table with a ``"*"`` key, which describes a collection: an absent collection is
    filled as empty. ``"*"`` applies its table to every entry of *raw*; a null entry is read
    as empty. A null value drops the key where present; any other value (a scalar, ``{}``)
    is filled where absent. Non-mappings are returned unchanged for the schema to refuse.
    """
    if not isinstance(raw, Mapping):
        return raw, []
    out, keys = dict(raw), []
    for key, value in fill.items():
        for target in (list(out) if key == "*" else [key]):
            if isinstance(value, Mapping) and value:
                if "*" in value and target not in out:
                    out[target] = {}
                    keys.append(prefix + target)
                if target in out:
                    if key == "*" and out[target] is None:
                        out[target] = {}
                    out[target], sub = _legacy_fill(out[target], value, f"{prefix}{target}.")
                    keys += sub
            elif value is None:
                if target in out:
                    del out[target]
                    keys.append(prefix + target)
            elif target not in out:
                out[target] = copy.deepcopy(value)
                keys.append(prefix + target)
    return out, keys


def _is_tracked(path: str | Path) -> bool:
    """True when *path* is a file git tracks in this repository."""
    return _is_tracked_resolved(Path(path).resolve())


@functools.lru_cache(maxsize=None)
def _is_tracked_resolved(resolved: Path) -> bool:
    if not resolved.is_relative_to(_REPO_ROOT):
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "--error-unmatch", str(resolved.relative_to(_REPO_ROOT))],
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"cannot tell whether rule file {resolved} is tracked: the git executable was not found"
        ) from None
    return result.returncode == 0


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Return ``(inner_type, is_optional)`` for ``X | None`` / ``Optional[X]``."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def check_mapping(
    raw: Any, types: Mapping[str, Any], *, required: Collection[str] | None = None, where: str,
) -> Mapping[str, Any]:
    """Validate one YAML mapping against a key → type-hint table and return it.

    ``required`` defaults to every key of ``types``. Raises ``ValueError`` naming
    ``where`` and the key for: a non-mapping, a key not in ``types``, a required key
    that is absent, or a value whose type does not match its hint (``float`` accepts
    ints, never bools; ``X | None`` accepts ``None``). Supported hints: a class, a
    generic alias of one (checked by its origin) and ``X | None`` of those; any other
    hint (``Literal``, ``Any``, a union of several types) raises ``ValueError``.
    Nothing is filled in — an optional key that is absent stays absent.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(types))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} (allowed: {sorted(types)})")
    missing = [k for k in (types if required is None else required) if k not in raw]
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")
    for key, value in raw.items():
        tp, optional = _unwrap_optional(types[key])
        cls = typing.get_origin(tp) or tp
        if tp is Any or cls is typing.Union or cls is UnionType or not isinstance(cls, type):
            raise ValueError(f"{where}: key {key!r} has unsupported type hint {types[key]}")
        if value is None:
            ok = optional
        elif cls is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif cls is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        else:
            ok = isinstance(value, cls)
        if not ok:
            raise ValueError(f"{where}: key {key!r} must be {types[key]}, got {value!r}")
    return raw


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override config into base config.

    Args:
        base: Base configuration dict.
        override: Override values (takes precedence).

    Returns:
        Merged configuration dict.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
