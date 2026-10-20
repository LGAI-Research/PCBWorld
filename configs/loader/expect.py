"""Expected-config check: the launcher states what it meant, the run stops when the config differs.

An expect file is a YAML mapping of config keys to the value the launcher intends, or to
``{value: ..., source: ...}`` to also pin where it must come from (a source prefix:
``recipe`` matches ``recipe:<file>``, ``legacy`` matches ``legacy:proxy``). Only the keys
it names are checked. :func:`check_expect` runs right after :func:`configs.loader.resolve.resolve`
(``--expect`` of every resolving entry point), before any env or model is built.
:file:`tools/experiments/preflight_diff.py` (differences across cases) and ``--expect-env-diff``
(train vs validation env) stay as the thin checks above it.
"""
from __future__ import annotations

from pathlib import Path

from configs.loader.resolve import ConfigError, RunConfig, same_value


def check_expect(path: str | Path, run: RunConfig) -> None:
    """Raise :class:`ConfigError` listing every key of the expect file at ``path`` whose
    resolved value or source differs (or that the config does not have)."""
    from configs.loader import load_config

    expected = load_config(path)
    bad = []
    for key, want in expected.items():
        spec = want if isinstance(want, dict) else {"value": want}
        if set(spec) - {"value", "source"} or not spec:
            raise ConfigError(f"expect {path}: {key}: want a value or {{value, source}}, got {want!r}")
        if key not in run.provenance:
            bad.append(f"{key}: not a key of this config")
            continue
        got = run.provenance[key]
        if "value" in spec and not same_value(got.value, spec["value"]):
            bad.append(f"{key}: expected {spec['value']!r}, resolved {got.value!r} (from {got.where})")
        src = spec.get("source")
        if src is not None and got.source != src and not got.source.startswith(f"{src}:"):
            bad.append(f"{key}: expected source {src!r}, resolved from {got.source!r} ({got.where})")
    if bad:
        raise ConfigError(f"the resolved config differs from {path}:\n  " + "\n  ".join(bad))


__all__ = ["check_expect"]
