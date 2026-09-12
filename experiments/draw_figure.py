#!/usr/bin/env python3
"""Single dispatcher for KDD paper figures/tables.

Each extractor under ``_lib/metrics/`` exposes ``main(argv=None)`` and writes its
artifact into ``var/results/kdd/paper_outputs/`` (read-only on the results tree).
This script just routes ``--figure <key>`` to the matching module's ``main``.

    python experiments/draw_figure.py --figure fig6c
    python experiments/draw_figure.py --figure all
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Each metrics module does its own `import common as C` via a sibling sys.path
# insert, so the metrics dir must be importable here too.
sys.path.insert(0, str(Path(__file__).resolve().parent / "_lib" / "metrics"))

FIGURES = {
    "fig6c": "fig6c_d1_cleanpass",
    "table3": "table3",
    "table22": "table22",
    "table23": "table23",
    "fig8": "fig8_reward_sweep",
    "fig9": "fig9_openloop",
    "table24_25": "table24_25",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", required=True, choices=[*sorted(FIGURES), "all"],
                    help="paper artifact to (re)generate")
    args, rest = ap.parse_known_args(argv)

    targets = list(FIGURES.values()) if args.figure == "all" else [FIGURES[args.figure]]
    for mod_name in targets:
        print(f"[draw_figure] -> {mod_name}", file=sys.stderr)
        importlib.import_module(mod_name).main(rest)


if __name__ == "__main__":
    main()
