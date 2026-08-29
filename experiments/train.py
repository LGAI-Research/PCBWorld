#!/usr/bin/env python3
"""Single-run router for KDD training entrypoints.

Thin pass-through: pick a per-experiment trainer shell and forward all remaining
args to it verbatim. The grid x seed loops live in each experiment's ``run.sh``
(bash); this router only resolves ``<experiment>`` -> the right shell and execs it.

    python experiments/train.py table1 --method ppo_per_step --seed 42
    python experiments/train.py reward --wirelength-penalty 0.002 --via-penalty 0.1 --seed 42
    python experiments/train.py d1-ppo --grid-size 100 --seed 42

Add ``--dry-run`` (consumed by the shells) to print the resolved command only.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# experiment token -> trainer shell (relative to experiments/)
TRAINERS = {
    "table1": "kdd/table1_rl/train_policy.sh",
    "reward": "kdd/figure6_reward/train_dense_reward_cell.sh",
    "d1-ppo": "kdd/figure5_d1/train_transformer_ppo.sh",
    "d1-jumanji": "kdd/figure5_d1/train_jumanji_a2c.sh",
    "d1-sable": "kdd/figure5_d1/train_sable.sh",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(TRAINERS),
                    help="which trainer to run")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args forwarded verbatim to the trainer shell")
    args = ap.parse_args(argv)

    shell = HERE / TRAINERS[args.experiment]
    if not shell.is_file():
        sys.exit(f"trainer shell not found: {shell}")

    # strip a leading '--' separator if present
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    cmd = ["bash", str(shell), *rest]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
