#!/usr/bin/env bash
# Add 128 validation boards to both combined_v2 datasets, side-by-side with the
# existing train/test splits. Idempotent: skips per-stage if outputs already
# exist (use FORCE=1 to rebuild the val source dirs from scratch).
#
# Pipeline (per dataset):
#   1. generate_synthetic_boards.py with a fresh seed (no overlap with train/test).
#   2. migrate_dataset_to_pro.py to attach .kicad_pro (v2 layout).
#   3. symlink valboard_NNNNN.{kicad_pcb,kicad_pro} into combined_v2/.
#   4. update configs/datasets/{10net_2pin_1layer_v2,multi_pin_2layer_v2}.json
#      — adds an "easy.val" key with the 128 valboard_* ids.
#   5. dereferenced cp into every combined_v2__r0X (repo).
#      (Direct generation into the shared dataset root is prohibited — shared
#      storage is read-only; distribute manually if deployment is needed.)
#
# Seeds (see README.md for the full collision table):
#   1L val: --seed 2     (train=0, test=1)
#   2L val: --seed 1234  (train shards 0..7, test=9999)
#
# Usage:
#   bash tools/datagen/synthetic_generator/generate_validation_set.sh
#   FORCE=1 bash tools/datagen/synthetic_generator/generate_validation_set.sh   # rebuild val sources

set -euo pipefail

cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
GEN_DIR="tools/datagen/synthetic_generator"


VAL_N="${VAL_N:-128}"
SEED_1L="${SEED_1L:-2}"
SEED_2L="${SEED_2L:-1234}"
WORKERS="${WORKERS:-16}"

SRC_1L_VAL=pcb_dataset_synthetic_10net_2pin_1layer_grid1000_val
SRC_1L_VAL_V2=pcb_dataset_synthetic_10net_2pin_1layer_grid1000_val_v2
SRC_2L_VAL=pcb_dataset_synthetic_multi_pin_2layer_val
SRC_2L_VAL_V2=pcb_dataset_synthetic_multi_pin_2layer_val_v2

COMBINED_1L=pcb_dataset_10net_2pin_1layer_combined_v2
COMBINED_2L=pcb_dataset_multi_pin_2layer_combined_v2

N_R_1L=10  # combined_v2__r01..r10
N_R_2L=6   # combined_v2__r01..r06

if [[ "${FORCE:-0}" == "1" ]]; then
  rm -rf "$SRC_1L_VAL" "$SRC_1L_VAL_V2" "$SRC_2L_VAL" "$SRC_2L_VAL_V2"
fi

# ---- step 1: generate raw .kicad_pcb -----------------------------------------
if [[ ! -d "$SRC_1L_VAL" ]]; then
  echo "[1L] generate val (n=$VAL_N seed=$SEED_1L)"
  python "$GEN_DIR/generate_synthetic_boards.py" \
    --n "$VAL_N" --seed "$SEED_1L" --seed-mode legacy \
    --mode grid --num-layers 1 \
    --board-size 100 --clearance 0.05 --trace-width 0.05 \
    --pitch-formula c+w --pad-size 0.05 --min-sep 0.1 \
    --fixed-pads-per-net 2,2,2,2,2,2,2,2,2,2 \
    --central-frac 1 --via-dia 0.6 --via-drill 0.3 \
    --out-dir "$SRC_1L_VAL"
else
  echo "[1L] skip generate (dir exists: $SRC_1L_VAL)"
fi

if [[ ! -d "$SRC_2L_VAL" ]]; then
  echo "[2L] generate val (n=$VAL_N seed=$SEED_2L)"
  python "$GEN_DIR/generate_synthetic_boards.py" \
    --n "$VAL_N" --seed "$SEED_2L" --seed-mode legacy \
    --mode grid --pitch-formula c+w \
    --board-size 100.0 --clearance 0.3 --trace-width 0.3 \
    --pad-size 2.4 --via-dia 1.2 --via-drill 0.6 \
    --num-layers 2 --central-frac 0.8 \
    --fixed-pads-per-net 2,2,2,3,4 --min-sep-formula four-pitch \
    --out-dir "$SRC_2L_VAL"
else
  echo "[2L] skip generate (dir exists: $SRC_2L_VAL)"
fi

# ---- step 2: migrate to v2 ---------------------------------------------------
if [[ ! -d "$SRC_1L_VAL_V2" ]]; then
  echo "[1L] migrate -> v2"
  python "$GEN_DIR/migrate_dataset_to_pro.py" \
    --src "$SRC_1L_VAL" --dst "$SRC_1L_VAL_V2" --workers "$WORKERS"
else
  echo "[1L] skip migrate (dir exists: $SRC_1L_VAL_V2)"
fi

if [[ ! -d "$SRC_2L_VAL_V2" ]]; then
  echo "[2L] migrate -> v2"
  python "$GEN_DIR/migrate_dataset_to_pro.py" \
    --src "$SRC_2L_VAL" --dst "$SRC_2L_VAL_V2" --workers "$WORKERS"
else
  echo "[2L] skip migrate (dir exists: $SRC_2L_VAL_V2)"
fi

# ---- step 3+4: symlink into combined_v2/ + update split JSON -----------------
python - <<'PY'
import json
from pathlib import Path

REPO = Path.cwd()
specs = [
    ("1L", REPO / "pcb_dataset_synthetic_10net_2pin_1layer_grid1000_val_v2",
            REPO / "pcb_dataset_10net_2pin_1layer_combined_v2",
            REPO / "configs/datasets/grids/10net_2pin_1layer_v2.json"),
    ("2L", REPO / "pcb_dataset_synthetic_multi_pin_2layer_val_v2",
            REPO / "pcb_dataset_multi_pin_2layer_combined_v2",
            REPO / "configs/datasets/misc/multi_pin_2layer_v2.json"),
]
for name, src, combined, split_json in specs:
    combined.mkdir(exist_ok=True)
    val_ids = []
    for sp in sorted(src.glob("board_*.kicad_pcb")):
        idx = int(sp.stem.split("_")[1])
        bid = f"valboard_{idx:05d}"
        for ext in (".kicad_pcb", ".kicad_pro"):
            s = sp.with_suffix(ext).resolve()
            d = combined / f"{bid}{ext}"
            if d.is_symlink() or d.exists():
                d.unlink()
            d.symlink_to(s)
        val_ids.append(bid)
    d = json.loads(split_json.read_text())
    d.setdefault("easy", {})["val"] = val_ids
    split_json.write_text(json.dumps(d, indent=2))
    print(f"[{name}] linked {len(val_ids)} val pairs into {combined.name}")
PY

# ---- step 5: dereferenced cp into every repo-local __r0X copy ----------------
distribute() {
    local src=$1 dst=$2
    if [[ ! -d "$dst" ]]; then echo "  [skip] $dst missing"; return 0; fi
    local n=0
    for f in "$src"/board_*.kicad_pcb; do
        idx=$(basename "$f" .kicad_pcb | cut -d_ -f2)
        cp -f "$f" "$dst/valboard_${idx}.kicad_pcb"
        cp -f "${f%.kicad_pcb}.kicad_pro" "$dst/valboard_${idx}.kicad_pro"
        n=$((n+1))
    done
    echo "  [ok] $n -> $dst"
}

echo "[distribute] 1L val into ${COMBINED_1L}__r01..r${N_R_1L}"
for i in $(seq 1 "$N_R_1L"); do
    idx=$(printf '%02d' "$i")
    distribute "$SRC_1L_VAL_V2" "${COMBINED_1L}__r${idx}"
done

echo "[distribute] 2L val into ${COMBINED_2L}__r01..r${N_R_2L}"
for i in $(seq 1 "$N_R_2L"); do
    idx=$(printf '%02d' "$i")
    distribute "$SRC_2L_VAL_V2" "${COMBINED_2L}__r${idx}"
done

echo "Done."
