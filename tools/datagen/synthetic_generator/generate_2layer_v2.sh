#!/usr/bin/env bash
# 2-layer synthetic generator — v2 spec.
#
# Per-board distribution:
#   - board size  : Uniform(80..120) mm x Uniform(80..120) mm (independent W/H)
#   - nets        : Uniform{4,5,6}
#   - pads/net    : Categorical{2: 0.60, 3: 0.20, 4: 0.10, 5: 0.10}
#   - pad type    : 30% through-hole (PTH, *.Cu), 70% SMD with random F.Cu/B.Cu
#   - pad XY      : unique across layers (min_sep on (x,y) regardless of side)
#   - placement   : central 90% of the board only
#
# Geometry (matches generate_multi_pin_2layer.sh):
#   c = w = 0.3 mm, via 1.2/0.6 mm, pad 2.4 mm,
#   grid pitch c+w = 0.6 mm, num_layers = 2,
#   min_sep = pad + 4*(c + w) = 4.8 mm (four-pitch)
#
# Output (v2 format: .kicad_pcb stripped of legacy net_class + sibling
# .kicad_pro carrying full design rules incl. copper_edge_clearance = 0.3).
#
# Usage:
#   bash tools/datagen/synthetic_generator/generate_2layer_v2.sh                      # 10K train + 1K test
#   TRAIN_N=200 TEST_N=20 bash tools/datagen/synthetic_generator/generate_2layer_v2.sh
#   SHARDS=1 bash tools/datagen/synthetic_generator/generate_2layer_v2.sh             # single-process
#   TRAIN_DIR=foo bash tools/datagen/synthetic_generator/generate_2layer_v2.sh        # custom dir
set -euo pipefail

cd "$(dirname "$0")/../../.."
GEN_DIR="tools/datagen/synthetic_generator"

TRAIN_N="${TRAIN_N:-10000}"
VAL_N="${VAL_N:-0}"
TEST_N="${TEST_N:-1000}"
SHARDS="${SHARDS:-8}"
TRAIN_DIR="${TRAIN_DIR:-var/datasets/synthetic/pcb_dataset_synthetic_2layer_v2}"
VAL_DIR="${VAL_DIR:-var/datasets/synthetic/pcb_dataset_synthetic_2layer_v2_val}"
TEST_DIR="${TEST_DIR:-var/datasets/synthetic/pcb_dataset_synthetic_2layer_v2_test}"
VAL_SEED="${VAL_SEED:-100}"
TEST_SEED="${TEST_SEED:-9999}"

COMMON_ARGS=(
  --mode grid --pitch-formula c+w
  --board-size-min 80.0
  --board-size-max 120.0
  --clearance 0.3
  --trace-width 0.3
  --pad-size 2.4
  --via-dia 1.2
  --via-drill 0.6
  --num-layers 2
  --thru-hole-prob 0.3
  --central-frac 0.9
  --nets-min 4
  --nets-max 6
  --pads-per-net-min 2
  --pads-per-net-max 5
  --pads-per-net-weights 0.6,0.2,0.1,0.1
  --min-sep-formula four-pitch
  --seed-mode legacy   # reproduces the released synth_2L_v2 dataset bit-for-bit
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

if [[ "${SKIP_VAL:-0}" != "1" ]] && (( VAL_N > 0 )); then
  mkdir -p "$VAL_DIR"
  echo "[val] $VAL_N boards, seed=$VAL_SEED, dir=$VAL_DIR"
  python "$GEN_DIR/generate_synthetic_boards.py" \
    --n "$VAL_N" --seed "$VAL_SEED" \
    --out-dir "$VAL_DIR" "${COMMON_ARGS[@]}"
  echo "[val] done: $(ls "$VAL_DIR"/board_*.kicad_pcb 2>/dev/null | wc -l) files"
fi

if [[ "${SKIP_TEST:-0}" != "1" ]]; then
  mkdir -p "$TEST_DIR"
  echo "[test] $TEST_N boards, seed=$TEST_SEED, dir=$TEST_DIR"
  python "$GEN_DIR/generate_synthetic_boards.py" \
    --n "$TEST_N" --seed "$TEST_SEED" \
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
if [[ "${SKIP_VAL:-0}" != "1" ]] && (( VAL_N > 0 )); then
  echo "[upgrade] adding .kicad_pro to $VAL_DIR ($WORKERS_UPGRADE workers)"
  python "$GEN_DIR/migrate_dataset_to_pro.py" \
    --src "$VAL_DIR" --dst "$VAL_DIR" --overwrite \
    --workers "$WORKERS_UPGRADE"
fi
if [[ "${SKIP_TEST:-0}" != "1" ]]; then
  echo "[upgrade] adding .kicad_pro to $TEST_DIR ($WORKERS_UPGRADE workers)"
  python "$GEN_DIR/migrate_dataset_to_pro.py" \
    --src "$TEST_DIR" --dst "$TEST_DIR" --overwrite \
    --workers "$WORKERS_UPGRADE"
fi

echo "All done."
