#!/usr/bin/env bash
# D2-B-geo / D2-B-V-geo — the D2-B recipe (see generate_D2B.sh) plus
# real-board geometry objects sampled per board (--geo):
#   outline mix: 4x gr_line rect ~53% | corner fillet (gr_arc) ~25% |
#                rectilinear polygon ~20% | gr_circle ~2%  (gr_rect: never)
#   internal cutouts ~5% · NPTH mounting holes ~25% (drill 3.2 dominant) ·
#   oval-drill slots ~10% · diversified THT pads (drill 0.8-1.0, oval/rect)
# Rates/ranges: outline_geometry.py constants (census of real d3a/d3b
# boards). Placement respects outline/holes (keepouts), capacity is derated
# by usable area — pads/cm² lands ~1.5% below the non-geo d2b pool.
#
# ASPECT_SIGMA=0.60 additionally draws a non-square board box (log(long/short)
# ~ |N(0, sigma)|, area unchanged) and writes to the `_ar` dataset roots. The
# default 0 reproduces the shipped square pool byte-for-byte.
#
# Seeds are --seed-mode linear (shard-count independent):
#   train base 0 · val base 1_000_000_000 · test base 2_000_000_000.
# Train shares topology-seed prefix 0..N-1 with the original d2b, so old/new
# boards pair up for geometry-effect comparisons.
#
# Requires the engine importable (conda cadagent + PYTHONPATH=build_rl/...).
#
# Usage:
#   bash tools/datagen/synthetic_generator/generate_D2B_geo.sh            # 10k/128/128
#   TRAIN_N=64 VAL_N=8 TEST_N=8 SHARDS=4 bash .../generate_D2B_geo.sh    # smoke
#   ASPECT_SIGMA=0.60 bash .../generate_D2B_geo.sh                       # _ar pool
set -euo pipefail
cd "$(dirname "$0")/../../.."
GEN_DIR="tools/datagen/synthetic_generator"
GEN="$GEN_DIR/generate_synthetic_boards.py"
CONV="$GEN_DIR/convert_pernet_to_pro.py"

TRAIN_N="${TRAIN_N:-10000}"
VAL_N="${VAL_N:-128}"
TEST_N="${TEST_N:-128}"
SHARDS="${SHARDS:-16}"
ASPECT_SIGMA="${ASPECT_SIGMA:-0}"
# non-zero aspect => separate dataset roots, so a stretched run can never
# overwrite the shipped square pool by omitting OUT_ROOT.
SUF=""; [ "$ASPECT_SIGMA" != "0" ] && SUF="_ar"
OUT_ROOT="${OUT_ROOT:-var/datasets/pcb_dataset_synthetic_d2b_geo$SUF}"
V_ROOT="${V_ROOT:-var/datasets/pcb_dataset_synthetic_d2bv_geo$SUF}"
WORKERS="${WORKERS:-24}"
VAL_BASE="${VAL_BASE:-1000000000}"
TEST_BASE="${TEST_BASE:-2000000000}"

python -c 'import pcb_world.engine' 2>/dev/null \
  || { echo "ERROR: engine not importable — conda cadagent + PYTHONPATH=build_rl/pcbnew/python/rl:. required" >&2; exit 1; }

COMMON=( --mode d2b --rule-mode paired --geo --seed-mode linear
  --aspect-sigma "$ASPECT_SIGMA"
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

# train: sharded by index; linear mode => same boards for any SHARDS value
if (( TRAIN_N % SHARDS != 0 )); then
  echo "ERROR: TRAIN_N ($TRAIN_N) must be divisible by SHARDS ($SHARDS)" >&2; exit 1
fi
per=$(( TRAIN_N / SHARDS ))
echo "[gen] train: $TRAIN_N boards, $SHARDS shards x $per (paired, geo)"
pids=()
for (( s=0; s<SHARDS; s++ )); do
  python "$GEN" --n "$per" --start-index $(( s * per )) --seed 0 \
    --out-dir "$OUT_ROOT/train" --paired-dir "$V_ROOT/train" "${COMMON[@]}" \
    > "$OUT_ROOT/train/.shard_${s}.log" 2>&1 &
  pids+=($!)
done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
(( fail )) && { echo "[gen] shard failure; see $OUT_ROOT/train/.shard_*.log" >&2; exit 1; }

echo "[gen] val: $VAL_N (base=$VAL_BASE)  test: $TEST_N (base=$TEST_BASE)"
python "$GEN" --n "$VAL_N"  --seed "$VAL_BASE"  --out-dir "$OUT_ROOT/val"  --paired-dir "$V_ROOT/val"  "${COMMON[@]}"
python "$GEN" --n "$TEST_N" --seed "$TEST_BASE" --out-dir "$OUT_ROOT/test" --paired-dir "$V_ROOT/test" "${COMMON[@]}"

# per-board engine round-trip: strips legacy net_class blocks, emits .kicad_pro
# (doubles as a full engine-load smoke over every board, incl. arc/circle/
# cutout/NPTH/slot geometry)
for d in "$OUT_ROOT"/{train,val,test} "$V_ROOT"/{train,val,test}; do
  echo "[convert] $d"
  python "$CONV" --src "$d" --workers "$WORKERS"
done

echo "D2-B-geo done: train=$(ls "$OUT_ROOT/train"/board_*.kicad_pcb | wc -l)" \
     "val=$(ls "$OUT_ROOT/val"/board_*.kicad_pcb | wc -l)" \
     "test=$(ls "$OUT_ROOT/test"/board_*.kicad_pcb | wc -l)  -> $OUT_ROOT (+$V_ROOT)"
