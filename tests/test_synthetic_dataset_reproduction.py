"""Regression test: the synthetic generators must keep reproducing the paper
datasets board-for-board.

Both staged datasets are FIXED artifacts under the shared dataset root, produced
by the synthetic generator:

  - D1  $DATASET_ROOT/synthetic/synth_1L/grid{G}_5net_v15   (5 nets x 2 pins, 1L)
  - D2  $DATASET_ROOT/synthetic/synth_2L_v2                 (nets U{4,5,6}, 2L thru)

This test pins board_00000 of each task so that a generator change which shifts
the generated geometry — e.g. one that reorders the RNG draws — fails loudly
instead of silently diverging from the staged dataset.

Self-contained: regenerates one board per task with the canonical args and
checks a geometry signature (net count, pad count, hash of sorted pad-center
coordinates) against a hardcoded golden captured from the staged boards. Does
NOT require access to the staged dataset root.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = REPO_ROOT / "tools" / "datagen" / "synthetic_generator"
GRID_GEN = GEN_DIR / "generate_grid_boards.py"
SYNTH_GEN = GEN_DIR / "generate_synthetic_boards.py"

# Canonical 2L (synth_2L_v2) args — mirror generate_2layer_v2.sh COMMON_ARGS.
T2_COMMON_ARGS = [
    "--mode", "grid", "--pitch-formula", "c+w",
    "--board-size-min", "80.0", "--board-size-max", "120.0",
    "--clearance", "0.3", "--trace-width", "0.3", "--pad-size", "2.4",
    "--via-dia", "1.2", "--via-drill", "0.6",
    "--num-layers", "2", "--thru-hole-prob", "0.3", "--central-frac", "0.9",
    "--nets-min", "4", "--nets-max", "6",
    "--pads-per-net-min", "2", "--pads-per-net-max", "5",
    "--pads-per-net-weights", "0.6,0.2,0.1,0.1",
    "--min-sep-formula", "four-pitch",
    "--seed-mode", "legacy",  # the released synth_2L_v2 was made with the legacy seed formula
]

# Canonical d2b_geo args — mirror generate_D2B_geo.sh COMMON (minus --aspect-sigma,
# which each test sets). The shipped var/datasets/pcb_dataset_synthetic_d2b_geo pool
# was generated with these; `--rule-mode paired` is part of the recipe because the
# paired render draws the per-net rules too (dropping it shifts the RNG stream).
D2B_GEO_ARGS = [
    "--mode", "d2b", "--rule-mode", "paired", "--geo", "--seed-mode", "linear",
    "--clearance", "0.2", "--trace-width", "0.25", "--pad-size", "1.2",
    "--via-dia", "0.8", "--via-drill", "0.4",
    "--num-layers", "2", "--central-frac", "0.8",
    "--min-sep-formula", "four-pitch", "--thru-hole-prob", "0.5",
    "--board-lognormal-median", "45", "--board-lognormal-sigma", "0.30",
    "--board-clip-min", "26", "--board-clip-max", "90",
    "--net-density-k", "0.0010", "--net-ref-pitch", "0.45",
    "--min-nets", "6", "--max-nets", "80",
    "--rail-prob", "0.10", "--rail-median", "13", "--rail-sigma", "0.5",
    "--rail-min", "8", "--rail-max", "32",
    "--bulk-base", "3", "--bulk-lambda", "0.35",
    "--uni-clearance-min", "0.10", "--uni-clearance-max", "0.25",
    "--uni-clearance-step", "0.05",
    "--uni-width-factor-min", "1.0", "--uni-width-factor-max", "1.4",
    "--uni-pad-pitch-mult-min", "2.0", "--uni-pad-pitch-mult-max", "2.5",
    "--uni-via-drill-mult-min", "1.5", "--uni-via-drill-mult-max", "2.5",
    "--uni-via-dia-mult", "1.8",
]

# Golden signatures captured from the staged dataset boards (board_00000, seed 0):
#   (net_count, pad_count, sha256(sorted pad-center coords)[:16])
GOLDEN_T1_GRID50 = (5, 10, "3fe860fb5622a2aa")
GOLDEN_T2 = (5, 18, "9354de200fb4ff8f")
# d2b_geo train board_00000 — square (shipped pool) and --aspect-sigma 0.60.
GOLDEN_D2B_GEO = (14, 28, "90f10919646b4a2f")
GOLDEN_D2B_GEO_BOX = (42.585, 42.585)
GOLDEN_D2B_GEO_AR = (14, 28, "324aef593ea903e7")
GOLDEN_D2B_GEO_AR_BOX = (57.766, 31.393)


def _signature(board: Path) -> tuple[int, int, str]:
    text = board.read_text()
    nets = {n for i, n in re.findall(r'\(net (\d+) "([^"]+)"\)', text) if n}
    coords = re.findall(r"\(at ([-0-9.]+) ([-0-9.]+)", text)
    pads = len(re.findall(r"\(pad ", text))
    pos_hash = hashlib.sha256(repr(sorted(coords)).encode()).hexdigest()[:16]
    return len(nets), pads, pos_hash


def _edge_box(board: Path) -> tuple[float, float]:
    """Edge.Cuts bounding box (w, h) in mm from the raw gr_* primitives."""
    text = board.read_text()
    pts = [(float(a), float(b)) for a, b in re.findall(
        r"\((?:start|end|mid|center) (-?[0-9.]+) (-?[0-9.]+)\)", text)]
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def _gen_d2b_geo(tmp_path: Path, aspect_sigma: str) -> Path:
    out = tmp_path / f"d2b_geo_{aspect_sigma}"
    subprocess.run(
        [sys.executable, str(SYNTH_GEN),
         "--n", "1", "--seed", "0", "--start-index", "0",
         "--out-dir", str(out), "--paired-dir", str(tmp_path / "paired"),
         "--aspect-sigma", aspect_sigma, *D2B_GEO_ARGS],
        check=True, cwd=REPO_ROOT, capture_output=True,
    )
    return out / "board_00000.kicad_pcb"


def test_d1_grid_board_reproduction(tmp_path: Path) -> None:
    """1L grid50 board_00000 (5 nets x 2 pins) must match the staged geometry."""
    out = tmp_path / "t1_grid50"
    subprocess.run(
        [sys.executable, str(GRID_GEN), "--grid", "50",
         "--n-train", "1", "--n-test", "0", "--seed-train", "0", "--train-only",
         "--nets", "5", "--pins-per-net", "2", "--num-layers", "1",
         "--out-prefix", str(out)],
        check=True, cwd=REPO_ROOT, capture_output=True,
    )
    assert _signature(out / "board_00000.kicad_pcb") == GOLDEN_T1_GRID50


def test_d2a_synth2l_board_reproduction(tmp_path: Path) -> None:
    """2L synth_2L_v2 board_00000 (nets U{4,5,6}, thru) must match staged geometry."""
    out = tmp_path / "t2_synth2l"
    subprocess.run(
        [sys.executable, str(SYNTH_GEN),
         "--n", "1", "--seed", "0", "--start-index", "0", "--out-dir", str(out),
         *T2_COMMON_ARGS],
        check=True, cwd=REPO_ROOT, capture_output=True,
    )
    assert _signature(out / "board_00000.kicad_pcb") == GOLDEN_T2


def test_d2b_geo_board_reproduction(tmp_path: Path) -> None:
    """d2b_geo train board_00000 must stay square and byte-stable.

    Pins the shipped ``pcb_dataset_synthetic_d2b_geo`` recipe: ``--aspect-sigma``
    defaults to 0, which must consume no RNG at all, so adding board-shape knobs
    cannot silently shift the geometry of the existing pool.
    """
    board = _gen_d2b_geo(tmp_path, "0")
    assert _signature(board) == GOLDEN_D2B_GEO
    w, h = _edge_box(board)
    assert (round(w, 3), round(h, 3)) == GOLDEN_D2B_GEO_BOX


def test_d2b_geo_aspect_board_reproduction(tmp_path: Path) -> None:
    """``--aspect-sigma 0.60`` stretches the board box but preserves its area."""
    board = _gen_d2b_geo(tmp_path, "0.60")
    assert _signature(board) == GOLDEN_D2B_GEO_AR
    w, h = _edge_box(board)
    assert (round(w, 3), round(h, 3)) == GOLDEN_D2B_GEO_AR_BOX
    sq_w, sq_h = GOLDEN_D2B_GEO_BOX
    assert w * h == pytest.approx(sq_w * sq_h, rel=1e-6)


if __name__ == "__main__":
    import shutil

    for name, fn in [
        ("D1 grid50", test_d1_grid_board_reproduction),
        ("D2 synth_2L_v2", test_d2a_synth2l_board_reproduction),
        ("D2-B-geo square", test_d2b_geo_board_reproduction),
        ("D2-B-geo aspect", test_d2b_geo_aspect_board_reproduction),
    ]:
        d = Path(tempfile.mkdtemp())
        try:
            fn(d)
            print(f"PASS  {name}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("All reproduction checks passed.")
