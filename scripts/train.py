#!/usr/bin/env python3
"""Unified training front door — the ``eval.py``/``profile.py`` peer for training.

Thin dispatch only; implementations live in ``methods/rl_agent/training/``.

Both subcommands train the decoder-only Transformer policy and need the built
C++ router (``kicad_rl_router``) importable — set ``PYTHONPATH`` as in
README.md before running.

Subcommands:
    ppo      Clipped-objective PPO (critic head + GAE + truncation bootstrap).
    grpo     GRPO (group-relative baseline; no critic / GAE).

Usage:
    python scripts/train.py ppo --board <path> --iterations 1000 --n-envs 4
    python scripts/train.py grpo --board <path> --iterations 200 --n-envs 32
    python scripts/train.py <subcommand> --help
"""
from __future__ import annotations

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
# Drop scripts/ from sys.path entirely (not just outrank it): with
# ``scripts/profile.py`` present, a later stdlib ``import profile`` (torch._dynamo
# -> cProfile -> profile) would resolve to that file and crash — the same
# shadowing hazard scripts/profile.py itself guards against. ``scripts/eval.py``
# shadowing the repo's ``eval`` package is covered by the same removal.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SUBCOMMANDS = {
    "ppo": "methods.rl_agent.training.train_ppo",
    "grpo": "methods.rl_agent.training.train_grpo",
}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in _SUBCOMMANDS:
        names = " | ".join(_SUBCOMMANDS)
        raise SystemExit(f"unknown subcommand {cmd!r} — expected: {names}")
    module = importlib.import_module(_SUBCOMMANDS[cmd])
    sys.argv = [f"{os.path.basename(sys.argv[0])} {cmd}"] + rest
    module.main()


if __name__ == "__main__":
    main()
