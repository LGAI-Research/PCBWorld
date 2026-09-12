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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
# Drop scripts/ from sys.path entirely (not just outrank it), exactly as
# scripts/train.py does. Outranking alone fixes only THIS file shadowing the
# repo's ``eval`` package; it leaves scripts/ on the path, so the stdlib
# ``import profile`` that torch._dynamo reaches through cProfile resolves to
# scripts/profile.py and dies with "module 'profile' has no attribute 'run'"
# — i.e. plain ``python scripts/eval.py --ckpt ...`` could not load a policy.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    from eval.pipeline import main as pipeline_main

    return pipeline_main()


if __name__ == "__main__":
    raise SystemExit(main())
