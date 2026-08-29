#!/usr/bin/env bash
# Generate 1M synthetic 2-layer boards with FIXED net/pin structure
# (5 nets: 3x2-pin + 1x3-pin + 1x4-pin = 13 pads) and 2x realistic design rules.
#
#   board: 100 x 100 mm (actual ~101.2 with edge margin)
#   clearance = trace_width = 0.3 mm
#   via = 1.2 / 0.6 mm, pad = 2.4 mm
#   min_sep = pad + 4*(c + w) = 4.8 mm
#   grid: c+w pitch (0.6 mm), pads snap to cell centers
#
# Shards 1M train into N parallel workers (default 8). Test set (2K) runs
# single-process with a different seed.
#
# Usage:
#   bash tools/datagen/synthetic_generator/generate_multi_pin_2layer.sh              # 1M train + 2K test
#   TRAIN_N=10000 SHARDS=4 bash tools/datagen/synthetic_generator/generate_multi_pin_2layer.sh   # smaller
#   SKIP_TRAIN=1 bash tools/datagen/synthetic_generator/generate_multi_pin_2layer.sh # test only
set -euo pipefail

cd "$(dirname "$0")/../../.."
GEN_DIR="tools/datagen/synthetic_generator"

TRAIN_N="${TRAIN_N:-1000000}"
TEST_N="${TEST_N:-2000}"
SHARDS="${SHARDS:-8}"
TRAIN_DIR="${TRAIN_DIR:-var/datasets/synthetic/pcb_dataset_synthetic_multi_pin_2layer}"
TEST_DIR="${TEST_DIR:-var/datasets/synthetic/pcb_dataset_synthetic_multi_pin_2layer_test}"

COMMON_ARGS=(
  --mode grid --pitch-formula c+w
  --board-size 100.0
  --clearance 0.3
  --trace-width 0.3
  --pad-size 2.4
  --via-dia 1.2
  --via-drill 0.6
  --num-layers 2
  --central-frac 0.8
  --fixed-pads-per-net 2,2,2,3,4
  --min-sep-formula four-pitch
  --seed-mode legacy   # legacy dataset — pin so default flip to linear doesn't change output
)

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if (( TRAIN_N % SHARDS != 0 )); then
    echo "ERROR: TRAIN_N ($TRAIN_N) must be divisible by SHARDS ($SHARDS)" >&2
    exit 1
  fi
  PER_SHARD=$(( TRAIN_N / SHARDS ))
  mkdir -p "$TRAIN_DIR"
  echo "[train] $TRAIN_N boards, $SHARDS shards x $PER_SHARD, dir=$TRAIN_DIR"
  pids=()
  for (( s=0; s<SHARDS; s++ )); do
    START=$(( s * PER_SHARD ))
    LOG="$TRAIN_DIR/.shard_${s}.log"
    (
      python "$GEN_DIR/generate_synthetic_boards.py" \
        --n "$PER_SHARD" --start-index "$START" --seed "$s" \
        --out-dir "$TRAIN_DIR" "${COMMON_ARGS[@]}"
    ) > "$LOG" 2>&1 &
    pids+=($!)
    echo "  shard $s: pid=${pids[-1]} range=[$START, $((START + PER_SHARD)))  log=$LOG"
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=1; fi
  done
  if (( fail )); then
    echo "[train] one or more shards failed; see $TRAIN_DIR/.shard_*.log" >&2
    exit 1
  fi
  echo "[train] done: $(ls "$TRAIN_DIR"/board_*.kicad_pcb 2>/dev/null | wc -l) files"
fi

if [[ "${SKIP_TEST:-0}" != "1" ]]; then
  mkdir -p "$TEST_DIR"
  echo "[test] $TEST_N boards, seed=9999, dir=$TEST_DIR"
  python "$GEN_DIR/generate_synthetic_boards.py" \
    --n "$TEST_N" --seed 9999 \
    --out-dir "$TEST_DIR" "${COMMON_ARGS[@]}"
  echo "[test] done: $(ls "$TEST_DIR"/board_*.kicad_pcb 2>/dev/null | wc -l) files"
fi

# Promote each board to v2 format: strip the legacy (net_class) block and
# emit a sibling .kicad_pro carrying the full design rules (incl.
# copper_edge_clearance = clearance, derived from each board's own
# default_netclass). In-place; idempotent.
WORKERS_UPGRADE="${WORKERS_UPGRADE:-16}"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "[upgrade] adding .kicad_pro to $TRAIN_DIR ($WORKERS_UPGRADE workers)"
  python "$GEN_DIR/migrate_dataset_to_pro.py" \
    --src "$TRAIN_DIR" --dst "$TRAIN_DIR" --overwrite \
    --workers "$WORKERS_UPGRADE"
fi
if [[ "${SKIP_TEST:-0}" != "1" ]]; then
  echo "[upgrade] adding .kicad_pro to $TEST_DIR ($WORKERS_UPGRADE workers)"
  python "$GEN_DIR/migrate_dataset_to_pro.py" \
    --src "$TEST_DIR" --dst "$TEST_DIR" --overwrite \
    --workers "$WORKERS_UPGRADE"
fi

echo "All done."
