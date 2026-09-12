#!/usr/bin/env bash
# Launcher for a full 10k-train + 128-test grid dataset.
#
# Produces under the repo root:
#   pcb_dataset_synthetic_<nets>net_<pins>pin_<layers>layer_grid<N>        (train)
#   pcb_dataset_synthetic_<nets>net_<pins>pin_<layers>layer_grid<N>_test   (test)
#
# Defaults: 10 nets * 2 pins, 1 copper layer, central_frac=1.0,
# seeds: train=0, test=1.  Override via env vars (NETS, PINS, LAYERS, SEED_TRAIN,
# SEED_TEST) if needed.
#
# Usage:
#   tools/datagen/synthetic_generator/generate_grid_dataset.sh <grid_size>
#   tools/datagen/synthetic_generator/generate_grid_dataset.sh 1000
#   NETS=5 PINS=3 tools/datagen/synthetic_generator/generate_grid_dataset.sh 100
set -euo pipefail

GRID="${1:?Usage: $0 <grid_size>  (e.g. 10, 20, 50, 100, 1000)}"
NETS="${NETS:-10}"
PINS="${PINS:-2}"
LAYERS="${LAYERS:-1}"
SEED_TRAIN="${SEED_TRAIN:-0}"
SEED_TEST="${SEED_TEST:-1}"
N_TRAIN="${N_TRAIN:-10000}"
N_TEST="${N_TEST:-128}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
GEN_DIR="tools/datagen/synthetic_generator"

PREFIX="var/datasets/synthetic/pcb_dataset_synthetic_${NETS}net_${PINS}pin_${LAYERS}layer_grid${GRID}"

echo "== grid dataset generation =="
echo "  grid      = ${GRID}x${GRID}"
echo "  nets      = ${NETS} x ${PINS} pins  layers=${LAYERS}"
echo "  train dir = ${PREFIX}/            (n=${N_TRAIN}, seed=${SEED_TRAIN})"
echo "  test  dir = ${PREFIX}_test/       (n=${N_TEST},  seed=${SEED_TEST})"
echo

python "$GEN_DIR/generate_grid_boards.py" \
    --grid "${GRID}" \
    --n-train "${N_TRAIN}" \
    --n-test "${N_TEST}" \
    --seed-train "${SEED_TRAIN}" \
    --seed-test "${SEED_TEST}" \
    --nets "${NETS}" \
    --pins-per-net "${PINS}" \
    --num-layers "${LAYERS}" \
    --out-prefix "${PREFIX}"

echo
echo "Generated:"
echo "  ${REPO_ROOT}/${PREFIX}/        ($(ls -1 ${PREFIX} 2>/dev/null | wc -l) boards)"
echo "  ${REPO_ROOT}/${PREFIX}_test/   ($(ls -1 ${PREFIX}_test 2>/dev/null | wc -l) boards)"

# Promote to v2 format: strip the legacy (net_class) block and emit a
# sibling .kicad_pro with full design rules (auto-derived from each
# board's default_netclass; copper_edge_clearance = clearance). In-place.
WORKERS_UPGRADE="${WORKERS_UPGRADE:-16}"
echo
echo "[upgrade] adding .kicad_pro to ${PREFIX}/ and ${PREFIX}_test/  (${WORKERS_UPGRADE} workers)"
python "$GEN_DIR/migrate_dataset_to_pro.py" \
  --src "${PREFIX}" --dst "${PREFIX}" --overwrite \
  --workers "${WORKERS_UPGRADE}"
python "$GEN_DIR/migrate_dataset_to_pro.py" \
  --src "${PREFIX}_test" --dst "${PREFIX}_test" --overwrite \
  --workers "${WORKERS_UPGRADE}"
