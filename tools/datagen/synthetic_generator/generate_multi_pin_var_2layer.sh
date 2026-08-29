#!/usr/bin/env bash
# Variable-net 2-layer multi-pin synthetic generator.
#
# Distribution (per board, sampled iid):
#   - nets         : Uniform{2,3,4,5,6,7,8}
#   - pads/net     : Categorical{2: 0.60, 3: 0.30, 4: 0.10}
#
# Geometry (matches generate_multi_pin_2layer.sh):
#   board ~101.2 mm, c = w = 0.3 mm, via 1.2/0.6 mm, pad 2.4 mm,
#   grid pitch c+w = 0.6 mm, num_layers = 2, central_frac = 0.8.
#
# Output (v2 format: .kicad_pcb stripped of legacy net_class + sibling
# .kicad_pro carrying full design rules incl. copper_edge_clearance = 0.3).
#
# Usage:
#   bash tools/datagen/synthetic_generator/generate_multi_pin_var_2layer.sh                    # 10K train + 1K test
#   TRAIN_N=200 TEST_N=20 bash tools/datagen/synthetic_generator/generate_multi_pin_var_2layer.sh
#   SHARDS=1 bash tools/datagen/synthetic_generator/generate_multi_pin_var_2layer.sh           # single-process
#   TRAIN_DIR=foo bash tools/datagen/synthetic_generator/generate_multi_pin_var_2layer.sh      # custom dir
set -euo pipefail

cd "$(dirname "$0")/../../.."
GEN_DIR="tools/datagen/synthetic_generator"

TRAIN_N="${TRAIN_N:-10000}"
TEST_N="${TEST_N:-1000}"
SHARDS="${SHARDS:-8}"
TRAIN_DIR="${TRAIN_DIR:-var/datasets/synthetic/pcb_dataset_synthetic_multi_pin_var_2layer}"
TEST_DIR="${TEST_DIR:-var/datasets/synthetic/pcb_dataset_synthetic_multi_pin_var_2layer_test}"

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
  --nets-min 2
  --nets-max 8
  --pads-per-net-min 2
  --pads-per-net-max 4
  --pads-per-net-weights 0.6,0.3,0.1
  --min-sep-formula four-pitch
  --seed-mode legacy   # legacy dataset — pin so default flip to linear doesn't change output
)

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if (( SHARDS > 1 )) && (( TRAIN_N % SHARDS != 0 )); then
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

# Promote to v2 format (auto-derive design rules from each board's netclass;
# strip legacy block; emit sibling .kicad_pro). In-place; idempotent.
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
