"""Single source of truth for dataset / output paths.


Two entry points:
  resolve_dataset(name) -- read path: staged local copy if present, else canonical
                           data root. For single-pass jobs (eval/DRC).
  stage_dataset(name)   -- guarantee a local real copy and return it. For
                           repeated-read jobs (training/sweep) where shared-NFS
                           tail latency would stall a sync wave (policy R3).

Paths come from configs/paths.yaml; every root is overridable via env (CADAGENT_*).
The datasets are not distributed with the repo: the data root is $CADAGENT_DATA_ROOT
(no baked-in default), and any resolution that actually needs it while the variable is
unset fails loudly naming the variable rather than guessing a path.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_PATHS_YAML = Path(__file__).resolve().parent.parent / "paths.yaml"
_REPO_ROOT = _PATHS_YAML.parent.parent
# ${VAR} or ${VAR:-default}. No nested braces (campaign is appended in code, §4).
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: str) -> str:
    """Expand shell-style ${VAR:-default} / ${VAR} from os.environ."""

    def repl(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default if default is not None else "")

    return _ENV_RE.sub(repl, value)


def _anchor(root: str) -> Path:
    """Absolute roots as-is; relative roots (./var/...) anchored to the repo root."""
    p = Path(_expand(root))
    return p if p.is_absolute() else (_REPO_ROOT / p)


def data_root_path(*parts: str) -> str:
    """Join ``*parts`` onto the resolved data root, or "" when it is empty.

    For module-level constants that serve as argparse defaults: returning ""
    (the root unset) lets the caller check the value
    before use, so an unconfigured run reports the missing root instead of
    silently reading from the current directory.
    """
    root = get_paths().data
    return str(root.joinpath(*parts)) if root is not None else ""


def expand_data_path(value: str, cfg: Paths | None = None) -> str:
    """Expand ``${ENV}`` / ``${ENV:-default}`` in a dataset path, failing loudly.

    Board-list manifests and split-json ``dataset_dirs`` entries name their
    location as ``${CADAGENT_DATA_ROOT}/<sub>`` rather than a machine-specific
    absolute path, so the tracked lists are portable. A path with no ``${...}``
    is returned unchanged, which
    keeps plain relative entries cwd-relative.

    Raises KeyError when a referenced variable has no value and no default —
    silently expanding it to "" would yield a truncated path that later
    surfaces as a confusing "board not found".
    """

    def lookup(var: str) -> str | None:
        val = os.environ.get(var)
        if val:  # empty counts as unset — "" would truncate the path silently
            return val
        if var == "CADAGENT_DATA_ROOT":
            data = (cfg or get_paths()).data
            if data is not None:
                return str(data)
        return None

    missing = sorted({
        m.group(1) for m in _ENV_RE.finditer(value)
        if m.group(2) is None and lookup(m.group(1)) is None
    })
    if missing:
        raise KeyError(
            f"dataset path {value!r} references unset environment variable(s): "
            f"{', '.join(missing)}. Point CADAGENT_DATA_ROOT at your dataset "
            f"root (expected layout: configs/paths.yaml)."
        )

    def repl(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        resolved = lookup(var)
        return resolved if resolved is not None else default

    return _ENV_RE.sub(repl, value)


@dataclass(frozen=True)
class Paths:
    campaign: str
    data: Path | None  # None when the data root resolves empty (env + yaml default)
    staged: Path
    out: Path  # campaign already appended
    ckpt: Path  # campaign already appended
    datasets: dict[str, str]  # logical name -> sub

    def _sub(self, name: str) -> str:
        try:
            return self.datasets[name]["sub"]
        except KeyError:
            raise KeyError(
                f"unknown dataset '{name}'; known: {sorted(self.datasets)}"
            ) from None


def load_paths(path: str | Path = _PATHS_YAML) -> Paths:
    """Build a Paths from a paths.yaml, expanding env overrides at call time."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    campaign = _expand(raw.get("campaign", "kdd"))
    roots = raw.get("roots", {})
    # Data root has no baked-in default: unset $CADAGENT_DATA_ROOT expands to "" → None,
    # and resolution fails loudly at the point a dataset actually needs it.
    data_raw = _expand(roots["data"])
    return Paths(
        campaign=campaign,
        data=_anchor(roots["data"]) if data_raw else None,
        staged=_anchor(roots["staged"]),
        out=_anchor(roots["out"]) / campaign,
        ckpt=_anchor(roots["ckpt"]) / campaign,
        datasets=raw.get("datasets", {}),
    )


_CFG: Paths | None = None


def get_paths() -> Paths:
    """Process-wide singleton (lazy). Call load_paths() directly for a fresh build."""
    global _CFG
    if _CFG is None:
        _CFG = load_paths()
    return _CFG


def _require_data_root(cfg: Paths, name: str) -> Path:
    if cfg.data is None:
        raise RuntimeError(
            f"dataset '{name}' has no staged copy and CADAGENT_DATA_ROOT is "
            "not set — export CADAGENT_DATA_ROOT pointing at your dataset "
            "root (expected layout: configs/paths.yaml)."
        )
    return cfg.data


def resolve_dataset(name: str, cfg: Paths | None = None) -> Path:
    """Read path for `name`: staged copy if it exists, else canonical data root."""
    cfg = cfg or get_paths()
    sub = cfg._sub(name)
    staged = cfg.staged / sub
    return staged if staged.exists() else _require_data_root(cfg, name) / sub


def resolve_dataset_or_empty(name: str, cfg: Paths | None = None) -> str:
    """``resolve_dataset`` for module-level / argparse defaults: returns ""
    instead of raising when the data root is empty, so importing the caller
    never needs the data root — the caller checks the value before use and an
    unconfigured run reports the missing root at that point."""
    try:
        return str(resolve_dataset(name, cfg))
    except RuntimeError:
        return ""


def stage_dataset(name: str, cfg: Paths | None = None) -> Path:
    """Guarantee a local real copy of `name` and return it (real bytes, not symlink).

    Full-directory copytree carries native .kicad_pro sidecars verbatim (policy R5).
    """
    cfg = cfg or get_paths()
    sub = cfg._sub(name)
    local = cfg.staged / sub
    if not local.exists():
        src = _require_data_root(cfg, name) / sub
        if not src.exists():
            raise FileNotFoundError(f"dataset '{name}' source missing: {src}")
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, local)
    return local


if __name__ == "__main__":
    # Tiny CLI for shell scripts (no internal roots hardcoded shell-side):
    #   python -m configs.loader.paths data-root        -> effective data root
    #   python -m configs.loader.paths resolve <name>   -> resolve_dataset(name)
    # Errors (empty root / unknown name) exit non-zero with the resolver's
    # message, so `set -e` callers fail loudly instead of using a blank path.
    import sys

    def _cli(argv: list[str]) -> int:
        usage = "usage: python -m configs.loader.paths {data-root | resolve <name>}"
        try:
            if argv[:1] == ["data-root"] and len(argv) == 1:
                cfg = get_paths()
                if cfg.data is None:
                    raise RuntimeError(
                        "CADAGENT_DATA_ROOT is not set and configs/paths.yaml "
                        "carries no default — export CADAGENT_DATA_ROOT "
                        "pointing at your dataset root."
                    )
                print(cfg.data)
            elif argv[:1] == ["resolve"] and len(argv) == 2:
                print(resolve_dataset(argv[1]))
            else:
                print(usage, file=sys.stderr)
                return 2
        except (KeyError, RuntimeError) as e:
            print(f"configs.loader.paths: {e}", file=sys.stderr)
            return 1
        return 0

    raise SystemExit(_cli(sys.argv[1:]))
