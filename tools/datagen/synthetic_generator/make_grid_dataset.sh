#!/usr/bin/env bash
# Build a complete 1L 10net x 2pin grid<N> dataset (train + test + val) and
# wire it into a combined_v2 dir + split JSON sibling to the existing
# grid1000 layout.
#
# Layout produced (under repo root):
#   pcb_dataset_synthetic_10net_2pin_1layer_grid<N>/         train, .pcb + .pro
#   pcb_dataset_synthetic_10net_2pin_1layer_grid<N>_test/    test,  .pcb + .pro
#   pcb_dataset_synthetic_10net_2pin_1layer_grid<N>_val/     val,   .pcb + .pro
#   pcb_dataset_10net_2pin_1layer_grid<N>_combined_v2/       symlinks: board_/testboard_/valboard_
#   configs/datasets/grids/10net_2pin_1layer_grid<N>_v2.json         {"easy": {train, test, val}}
#
# Seeds (mirror the grid1000 convention):
#   train -> 0, test -> 1, val -> 2
#
# Usage:
#   bash tools/datagen/synthetic_generator/make_grid_dataset.sh <grid>
#   N_TRAIN=200 N_TEST=20 N_VAL=20 bash tools/datagen/synthetic_generator/make_grid_dataset.sh 10
set -euo pipefail

cd "$(dirname "$0")/../../.."
GEN_DIR="tools/datagen/synthetic_generator"

GRID="${1:?Usage: $0 <grid_size>  (e.g. 10, 30, 100, 300, 1000)}"
N_TRAIN="${N_TRAIN:-10000}"
N_TEST="${N_TEST:-128}"
N_VAL="${N_VAL:-128}"
SEED_TRAIN="${SEED_TRAIN:-0}"
SEED_TEST="${SEED_TEST:-1}"
SEED_VAL="${SEED_VAL:-2}"
WORKERS="${WORKERS:-16}"

PREFIX="var/datasets/synthetic/pcb_dataset_synthetic_10net_2pin_1layer_grid${GRID}"
TRAIN_DIR="$PREFIX"
TEST_DIR="${PREFIX}_test"
VAL_DIR="${PREFIX}_val"
COMBINED="pcb_dataset_10net_2pin_1layer_grid${GRID}_combined_v2"
SPLIT_JSON="configs/datasets/grids/10net_2pin_1layer_grid${GRID}_v2.json"

echo "=== make_grid_dataset grid=${GRID} ==="
echo "  train=${N_TRAIN} (seed=${SEED_TRAIN})  test=${N_TEST} (seed=${SEED_TEST})  val=${N_VAL} (seed=${SEED_VAL})"
echo "  combined -> ${COMBINED}"
echo "  split    -> ${SPLIT_JSON}"

# --- step 1: train + test (+ in-place v2 migration) ---------------------------
N_TRAIN="$N_TRAIN" N_TEST="$N_TEST" SEED_TRAIN="$SEED_TRAIN" SEED_TEST="$SEED_TEST" \
  WORKERS_UPGRADE="$WORKERS" \
  bash "$GEN_DIR/generate_grid_dataset.sh" "$GRID"

# --- step 2: val (use derive_params() to mirror grid_dataset geometry) --------
echo "[val] generate n=${N_VAL} seed=${SEED_VAL} -> ${VAL_DIR}"
python - <<EOF
import subprocess, sys
from pathlib import Path
sys.path.insert(0, "$GEN_DIR")
from generate_grid_boards import derive_params, BOARD_SIZE_MM, _fmt
p = derive_params(${GRID})
fixed = ",".join(["2"] * 10)
cmd = [
    sys.executable, "$GEN_DIR/generate_synthetic_boards.py",
    "--n", "${N_VAL}", "--seed", "${SEED_VAL}", "--seed-mode", "legacy",
    "--mode", "grid", "--num-layers", "1",
    "--board-size", _fmt(BOARD_SIZE_MM),
    "--clearance", _fmt(p["clearance"]),
    "--trace-width", _fmt(p["trace_width"]),
    "--pitch-formula", "c+w",
    "--pad-size", _fmt(p["pad_size"]),
    "--min-sep", _fmt(p["min_sep"]),
    "--fixed-pads-per-net", fixed,
    "--central-frac", "1.0",
    "--via-dia", "0.6", "--via-drill", "0.3",
    "--out-dir", "$VAL_DIR",
]
print(">>>", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
EOF

echo "[val] migrate -> v2 (in-place)"
python "$GEN_DIR/migrate_dataset_to_pro.py" \
  --src "$VAL_DIR" --dst "$VAL_DIR" --overwrite --workers "$WORKERS"

# --- step 3: combined_v2 dir + split JSON -------------------------------------
echo "[combined] symlink train/test/val into ${COMBINED}"
python - <<EOF
import json
from pathlib import Path

TRAIN = Path("$TRAIN_DIR")
TEST  = Path("$TEST_DIR")
VAL   = Path("$VAL_DIR")
DST   = Path("$COMBINED")
SPLIT = Path("$SPLIT_JSON")

DST.mkdir(exist_ok=True)
SPLIT.parent.mkdir(parents=True, exist_ok=True)

def link_pair(src_pcb: Path, dst_pcb: Path) -> None:
    src_pro = src_pcb.with_suffix(".kicad_pro")
    dst_pro = dst_pcb.with_suffix(".kicad_pro")
    for d in (dst_pcb, dst_pro):
        if d.is_symlink() or d.exists():
            d.unlink()
    dst_pcb.symlink_to(src_pcb.resolve())
    if src_pro.exists():
        dst_pro.symlink_to(src_pro.resolve())

train_ids: list[str] = []
for f in sorted(TRAIN.glob("board_*.kicad_pcb"), key=lambda p: int(p.stem.split("_")[1])):
    bid = f.stem
    link_pair(f, DST / f"{bid}.kicad_pcb")
    train_ids.append(bid)

test_ids: list[str] = []
for f in sorted(TEST.glob("board_*.kicad_pcb"), key=lambda p: int(p.stem.split("_")[1])):
    n = int(f.stem.split("_")[1])
    bid = f"testboard_{n:05d}"
    link_pair(f, DST / f"{bid}.kicad_pcb")
    test_ids.append(bid)

val_ids: list[str] = []
for f in sorted(VAL.glob("board_*.kicad_pcb"), key=lambda p: int(p.stem.split("_")[1])):
    n = int(f.stem.split("_")[1])
    bid = f"valboard_{n:05d}"
    link_pair(f, DST / f"{bid}.kicad_pcb")
    val_ids.append(bid)

split = {"easy": {"train": train_ids, "test": test_ids, "val": val_ids}}
SPLIT.write_text(json.dumps(split, indent=2))

print(f"[combined] {len(train_ids)} train + {len(test_ids)} test + {len(val_ids)} val "
      f"linked into {DST}")
print(f"[split]    {SPLIT}")
EOF

echo "=== grid=${GRID} done ==="
