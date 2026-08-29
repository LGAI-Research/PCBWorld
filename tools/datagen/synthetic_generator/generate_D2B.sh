#!/usr/bin/env bash
# D2-B / D2-B-V synthetic 2L dataset — the OFFICIAL recipe of
# pcb_dataset_synthetic_d2b{,v} (train 10k / val 128 / test 128) and of the
# 100k train extension.
#
# These are the verified production arguments: they reproduce the distributed
# boards bit-for-bit apart from uuids.
#   board side ~ lognormal(median 45, sigma 0.30) clip [26, 90] mm
#   nets ~ Poisson(0.0010 * (side/0.45)^2), min 6, max 80
#   fanout: 90% (3+Poisson(0.35)) + 10% rail lognormal(med 13, s 0.5)[8,32]
#   D2-B rules fixed: clearance 0.2 / width 0.25 / pad 1.2 / via 0.8/0.4
#   D2-B-V per-net uniform: clearance U{0.10..0.25/0.05}, w=c*U[1.0,1.4],
#     pad=pitch*U[2.0,2.5], via_drill=w*U[1.5,2.5], via_dia=drill*1.8
#   paired mode: BOTH sets from one shared layout (identical boards/pads,
#   only rules differ). The conservative worst-rule grid also drives the
#   capacity clip — running rule-mode fixed instead inflates pads/board ~+22%.
#
# Requires the engine importable (conda cadagent + PYTHONPATH=build_rl/...)
# for the per-board .kicad_pro round-trip.
#
# Usage:
#   bash tools/datagen/synthetic_generator/generate_D2B.sh              # 10k/128/128
#   TRAIN_N=64 VAL_N=8 TEST_N=8 SHARDS=4 bash .../generate_D2B.sh      # smoke
set -euo pipefail
cd "$(dirname "$0")/../../.."
GEN_DIR="tools/datagen/synthetic_generator"
GEN="$GEN_DIR/generate_synthetic_boards.py"
CONV="$GEN_DIR/convert_pernet_to_pro.py"

TRAIN_N="${TRAIN_N:-10000}"
VAL_N="${VAL_N:-128}"
TEST_N="${TEST_N:-128}"
SHARDS="${SHARDS:-8}"
OUT_ROOT="${OUT_ROOT:-var/datasets/pcb_dataset_synthetic_d2b}"
V_ROOT="${V_ROOT:-var/datasets/pcb_dataset_synthetic_d2bv}"
WORKERS="${WORKERS:-24}"
VAL_SEED="${VAL_SEED:-7777}"
TEST_SEED="${TEST_SEED:-9999}"

python -c 'import pcb_world.engine' 2>/dev/null \
  || { echo "ERROR: engine not importable — conda cadagent + PYTHONPATH=build_rl/pcbnew/python/rl:. required" >&2; exit 1; }

# --seed-mode legacy pinned: reproduces the existing released d2b datasets.
COMMON=( --mode d2b --rule-mode paired --seed-mode legacy
  --clearance 0.2 --trace-width 0.25 --pad-size 1.2 --via-dia 0.8 --via-drill 0.4
  --num-layers 2 --central-frac 0.8 --min-sep-formula four-pitch --thru-hole-prob 0.5
  --board-lognormal-median 45 --board-lognormal-sigma 0.30
  --board-clip-min 26 --board-clip-max 90
  --net-density-k 0.0010 --net-ref-pitch 0.45 --min-nets 6 --max-nets 80
  --rail-prob 0.10 --rail-median 13 --rail-sigma 0.5 --rail-min 8 --rail-max 32
  --bulk-base 3 --bulk-lambda 0.35
  --uni-clearance-min 0.10 --uni-clearance-max 0.25 --uni-clearance-step 0.05
  --uni-width-factor-min 1.0 --uni-width-factor-max 1.4
  --uni-pad-pitch-mult-min 2.0 --uni-pad-pitch-mult-max 2.5
  --uni-via-drill-mult-min 1.5 --uni-via-drill-mult-max 2.5 --uni-via-dia-mult 1.8 )

mkdir -p "$OUT_ROOT"/{train,val,test} "$V_ROOT"/{train,val,test}

# train: sharded; legacy seed = shard_id * 1_000_003 + global_index
if (( TRAIN_N % SHARDS != 0 )); then
  echo "ERROR: TRAIN_N ($TRAIN_N) must be divisible by SHARDS ($SHARDS)" >&2; exit 1
fi
per=$(( TRAIN_N / SHARDS ))
echo "[gen] train: $TRAIN_N boards, $SHARDS shards x $per (paired)"
pids=()
for (( s=0; s<SHARDS; s++ )); do
  python "$GEN" --n "$per" --start-index $(( s * per )) --seed "$s" \
    --out-dir "$OUT_ROOT/train" --paired-dir "$V_ROOT/train" "${COMMON[@]}" \
    > "$OUT_ROOT/train/.shard_${s}.log" 2>&1 &
  pids+=($!)
done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
(( fail )) && { echo "[gen] shard failure; see $OUT_ROOT/train/.shard_*.log" >&2; exit 1; }

echo "[gen] val: $VAL_N (seed=$VAL_SEED)  test: $TEST_N (seed=$TEST_SEED)"
python "$GEN" --n "$VAL_N"  --seed "$VAL_SEED"  --out-dir "$OUT_ROOT/val"  --paired-dir "$V_ROOT/val"  "${COMMON[@]}"
python "$GEN" --n "$TEST_N" --seed "$TEST_SEED" --out-dir "$OUT_ROOT/test" --paired-dir "$V_ROOT/test" "${COMMON[@]}"

# per-board engine round-trip: strips legacy net_class blocks, emits .kicad_pro
# (doubles as a full engine-load smoke over every board)
for d in "$OUT_ROOT"/{train,val,test} "$V_ROOT"/{train,val,test}; do
  echo "[convert] $d"
  python "$CONV" --src "$d" --workers "$WORKERS"
done

echo "D2-B done: train=$(ls "$OUT_ROOT/train"/board_*.kicad_pcb | wc -l)" \
     "val=$(ls "$OUT_ROOT/val"/board_*.kicad_pcb | wc -l)" \
     "test=$(ls "$OUT_ROOT/test"/board_*.kicad_pcb | wc -l)  -> $OUT_ROOT (+$V_ROOT)"
