#!/usr/bin/env python3
"""Speed-profiler front door — the ``train.py``/``eval.py`` peer.

Thin dispatch only; the implementation lives in
``tools/diagnostics/speed_profiler/``. Profiles the Decoder-only PPO training loop
(3 phases + internal decomposition, H1 sync-barrier, phase-tagged util) on the
real PPOTrainer code paths.

Usage:
    python scripts/profile.py --dataset d2a --n-envs 64
    python scripts/profile.py --dataset d3b --n-envs 128 --host-tag l40 --waterfall
    python scripts/profile.py --help

Needs the ``cadagent`` conda env + the built router on PYTHONPATH/LD_LIBRARY_PATH.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
# CRITICAL: this file is named ``profile.py``; Python auto-adds ``scripts/`` to
# sys.path[0] when run as ``python scripts/profile.py``, so a later
# ``import profile`` (torch._dynamo -> cProfile -> import profile) would resolve
# to THIS file and crash. Drop scripts/ from sys.path and put the repo root first
# (the same shadowing hazard applies to the repo's ``eval`` package).
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    from tools.diagnostics.speed_profiler.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
