"""Generate grid-aligned synthetic boards with all params derived from --grid N.

Given --grid N (e.g. 10, 20, 50, 100, 1000), fixes board_size=100mm and derives:
  grid_spacing g = 100 / N         (mm per cell)
  clearance  c = g / 2
  trace_width w = g / 2            (= c)
  pad_size     = w                 (pad occupies exactly 1 grid cell-width,
                                    same as a trace)
  min_sep      = g                 (pad-pad centers may sit in adjacent cells;
                                    same-cell collision still rejected)

All other conventions fixed:
  --mode grid --pitch-formula c+w --central-frac 1.0
  num_layers 1 (single-sided, F.Cu only)
  10 nets x 2 pins per board (overridable via --nets / --pins-per-net)

Produces train + test directories by default; use --train-only / --test-only
to run a single side.

Examples:
  # 1000x1000 grid, 10k train + 128 test (default)
  python tools/datagen/synthetic_generator/generate_grid_boards.py --grid 1000

  # 100x100 grid, smoke-test 20 boards
  python tools/datagen/synthetic_generator/generate_grid_boards.py --grid 100 --n-train 20 --test-only
  python tools/datagen/synthetic_generator/generate_grid_boards.py --grid 100 --n-train 0 --n-test 20

  # Custom nets / pins
  python tools/datagen/synthetic_generator/generate_grid_boards.py --grid 50 --nets 5 --pins-per-net 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BOARD_SIZE_MM = 100.0
GEN_SCRIPT = Path(__file__).resolve().parent / "generate_synthetic_boards.py"


def derive_params(grid_n: int) -> dict[str, float]:
    if grid_n < 2:
        raise SystemExit(f"--grid must be >= 2, got {grid_n}")
    g = BOARD_SIZE_MM / grid_n
    c = g / 2
    w = g / 2
    return {
        "grid_spacing": g,
        "clearance": c,
        "trace_width": w,
        "pad_size": w,
        "min_sep": g,
    }


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def run_generator(n: int, seed: int, out_dir: Path, params: dict[str, float],
                  nets: int, pins_per_net: int, num_layers: int,
                  central_frac: float, via_dia: float, via_drill: float,
                  dry_run: bool = False) -> None:
    fixed = ",".join([str(pins_per_net)] * nets)
    cmd = [
        sys.executable, str(GEN_SCRIPT),
        "--n", str(n),
        "--seed", str(seed),
        "--seed-mode", "legacy",  # grid_scan datasets were made with the legacy formula
        "--mode", "grid",
        "--num-layers", str(num_layers),
        "--board-size", _fmt(BOARD_SIZE_MM),
        "--clearance", _fmt(params["clearance"]),
        "--trace-width", _fmt(params["trace_width"]),
        "--pitch-formula", "c+w",
        "--pad-size", _fmt(params["pad_size"]),
        "--min-sep", _fmt(params["min_sep"]),
        "--fixed-pads-per-net", fixed,
        "--central-frac", _fmt(central_frac),
        "--via-dia", _fmt(via_dia),
        "--via-drill", _fmt(via_drill),
        "--out-dir", str(out_dir),
    ]
    print(">>>", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, required=True,
                   help="grid size N (cells per side). e.g. 10, 20, 50, 100, 1000.")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--seed-train", type=int, default=0)
    p.add_argument("--seed-test", type=int, default=1)
    p.add_argument("--nets", type=int, default=10,
                   help="nets per board (default 10)")
    p.add_argument("--pins-per-net", type=int, default=2,
                   help="pins per net (default 2)")
    p.add_argument("--num-layers", type=int, default=1, choices=[1, 2])
    p.add_argument("--central-frac", type=float, default=1.0)
    p.add_argument("--via-dia", type=float, default=0.6)
    p.add_argument("--via-drill", type=float, default=0.3)
    p.add_argument("--out-prefix", default=None,
                   help="output directory prefix. default: "
                        "pcb_dataset_synthetic_{nets}net_{pins}pin_{layers}layer_grid{N}")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the sub-commands but do not execute")
    args = p.parse_args()

    if args.train_only and args.test_only:
        raise SystemExit("--train-only and --test-only are mutually exclusive")

    params = derive_params(args.grid)
    repo_root = Path(__file__).resolve().parents[2]

    prefix = args.out_prefix or (
        f"pcb_dataset_synthetic_{args.nets}net_{args.pins_per_net}pin_"
        f"{args.num_layers}layer_grid{args.grid}"
    )
    train_dir = repo_root / prefix
    test_dir = repo_root / f"{prefix}_test"

    print(f"grid = {args.grid} x {args.grid}  (board {BOARD_SIZE_MM} mm)")
    print(f"  grid_spacing = {params['grid_spacing']:.6g} mm")
    print(f"  clearance    = {params['clearance']:.6g} mm")
    print(f"  trace_width  = {params['trace_width']:.6g} mm")
    print(f"  pad_size     = {params['pad_size']:.6g} mm")
    print(f"  min_sep      = {params['min_sep']:.6g} mm  (= 1 grid cell)")
    print(f"  structure    = {args.nets} nets x {args.pins_per_net} pins "
          f"(total {args.nets * args.pins_per_net} pads)")
    print(f"  layers       = {args.num_layers}")
    print(f"  central_frac = {args.central_frac:.3g}")

    if not args.test_only:
        print(f"\n[train] n={args.n_train} seed={args.seed_train} -> {train_dir}")
        run_generator(args.n_train, args.seed_train, train_dir, params,
                      args.nets, args.pins_per_net, args.num_layers,
                      args.central_frac, args.via_dia, args.via_drill,
                      dry_run=args.dry_run)

    if not args.train_only:
        print(f"\n[test]  n={args.n_test} seed={args.seed_test} -> {test_dir}")
        run_generator(args.n_test, args.seed_test, test_dir, params,
                      args.nets, args.pins_per_net, args.num_layers,
                      args.central_frac, args.via_dia, args.via_drill,
                      dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
