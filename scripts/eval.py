#!/usr/bin/env python3
"""Unified eval front door — the ``train.py``/``visualize.py`` peer for scoring.

Thin dispatch only; the 3-stage pipeline (rollout -> post-hoc DRC -> aggregate)
lives in ``eval/pipeline.py`` (single command, so no subcommands — all flags
pass straight through; ``--help`` shows the pipeline's own help). Needs the
built C++ router (``kicad_rl_router``) importable — set ``PYTHONPATH`` /
``LD_LIBRARY_PATH`` as in the README before running.

Usage:
    python scripts/eval.py --ckpt <path> --boards-dir <path> --seed 42 --n-rollouts 5
    python scripts/eval.py --help
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Must outrank sys.path[0] (this scripts/ dir): this file would otherwise
# shadow the repo's ``eval`` package on plain ``python scripts/eval.py``.
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    from eval.pipeline import main as pipeline_main

    return pipeline_main()


if __name__ == "__main__":
    raise SystemExit(main())
