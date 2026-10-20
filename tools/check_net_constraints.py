#!/usr/bin/env python3
"""Preflight sweep for the ``net_constraint_obs`` env knob over a dataset.

With the knob ON, ``PCBWorld.__init__`` resolves every net's netclass DRC
values (track_width / clearance / via_diameter / via_drill; KiCad inherit →
Default fallback) and RAISES on any net whose resolved value is not
positive — loud failure instead of a silent 0 observation
(``PCBWorld._fill_net_constraint_obs``). In an ncobs cell that failure fires
only at board-load time, mid-run; this tool front-loads the check across a
whole dataset BEFORE dispatching such a cell, and summarises the value
distribution (unique constraint tuples).

There is no clamp: a board whose net class value sits below the board
minimum of the same quantity (where the value would have been changed) is
refused at construction with ``ValueError("inconsistent design rules ...")``
(``pcb_world.engine.drc_config.verify_netclass_rules``) and is reported
here as a FAIL like any other load error.

No re-implementation: each board is loaded through the real path — a
``PCBWorld`` construction with ``net_constraint_obs=True`` runs
``_fill_net_constraint_obs`` exactly as the cell would. Boards are checked
sequentially with a close() + gc between loads (engine liveness contract:
one live router per process). Engine assert spam on stderr (e.g.
``PCB_VIA::GetWidth``) is harmless noise, not a failure.

Usage (``pcbworld`` env; needs the built router on PYTHONPATH)::

    export PYTHONPATH=build_rl/pcbnew/python/rl:.
    python tools/check_net_constraints.py \
        --boards-json configs/datasets/local/d2b_geo.json --split val
    python tools/check_net_constraints.py --boards-dir <dir> --limit 32

Exit codes: 0 = every board loads with consistent design rules and resolves
4 positive fields for every net; 1 = at least one board failed (each printed
immediately: path + error).
"""
from __future__ import annotations

import argparse
import gc
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from methods._shared.board_loader import (  # noqa: E402
    load_boards_from_dir_or_list,
    load_boards_from_split_json,
)

_FIELDS = ("track_width", "clearance", "via_diameter", "via_drill")


def check_board(path: str) -> list[dict[str, float]]:
    """Load ``path`` with the knob ON and return the per-net resolved values.

    The construction itself IS the check — ``__init__`` runs
    ``verify_netclass_rules`` (raises on inconsistent design rules) and
    ``_fill_net_constraint_obs`` (raises on any non-positive field).
    The values for the distribution summary come from
    ``_resolve_net_rule_values`` — private access on purpose: it is the
    single source the fill uses, and copying the resolution logic here is
    exactly the drift this tool exists to catch.
    """
    from configs.loader.schema import EnvConfig
    from pcb_world.core.env import PCBWorld

    # Outline kept as stored: the check reads design rules, not outline geometry.
    env = PCBWorld(path, EnvConfig.with_overrides(max_steps=1, net_constraint_obs=True,
                                                  simplify_outline=False))
    try:
        rules = env._engine.get_design_rules()
        return [
            env._resolve_net_rule_values(code, rules)
            for code in env._board_info.nets
        ]
    finally:
        env.close()


def main() -> int:
    from configs.loader.cli import StrictArgumentParser
    ap = StrictArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--boards-json", type=Path,
                     help="split json (configs/datasets/...)")
    src.add_argument("--boards-dir", type=Path,
                     help="directory of *.kicad_pcb boards")
    ap.add_argument("--difficulty", default="easy",
                    help="split-json difficulty key (default: easy)")
    ap.add_argument("--split", default="val",
                    help="split-json split key (default: val)")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="check only the first N boards (default: all)")
    args = ap.parse_args()

    if args.boards_json is not None:
        boards = load_boards_from_split_json(
            args.boards_json, args.difficulty, args.split,
        )
    else:
        boards = load_boards_from_dir_or_list(boards_dir=args.boards_dir)
    if args.limit is not None:
        boards = boards[:args.limit]
    print(f"[ncobs-check] {len(boards)} boards", flush=True)

    failures: list[str] = []
    tuple_counts: Counter[tuple[float, ...]] = Counter()
    n_nets = 0

    for i, spec in enumerate(boards):
        try:
            per_net = check_board(spec.path)
        except Exception as e:  # noqa: BLE001 — report every failure mode
            print(f"FAIL {spec.path}\n     {type(e).__name__}: {e}",
                  flush=True)
            failures.append(spec.path)
        else:
            for values in per_net:
                n_nets += 1
                # round to nm resolution (1e-6 mm) — the engine's unit floor
                tuple_counts[
                    tuple(round(values[k], 6) for k in _FIELDS)
                ] += 1
        # Engine liveness contract: reclaim this board's router before the
        # next construction (also frees the half-built env of a failed load).
        gc.collect()
        done = i + 1
        if done % 25 == 0 or done == len(boards):
            print(f"  [{done}/{len(boards)}] ok={done - len(failures)} "
                  f"fail={len(failures)}", flush=True)

    print(f"[ncobs-check] nets: {n_nets} checked")
    print(f"[ncobs-check] unique (tw, cl, vd, drill) tuples: "
          f"{len(tuple_counts)}")
    for vals, cnt in tuple_counts.most_common(10):
        print(f"    {cnt:6d} nets  {vals}")
    if len(tuple_counts) > 10:
        print(f"    ... and {len(tuple_counts) - 10} more tuples")

    if failures:
        print(f"[ncobs-check] FAIL — {len(failures)}/{len(boards)} boards "
              "(paths above)")
        return 1
    print(f"[ncobs-check] PASS — all {len(boards)} boards load with "
          "consistent rules and resolve positive constraints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
