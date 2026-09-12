#!/usr/bin/env bash
# d2c — 2-layer synthetic dataset matched to the d3b (real PCB) distribution.
#
# What sets it apart from d2b is not "bigger/denser" but **which axes are
# matched**. The values below come from measuring the 50 d3b boards on the same
# axes as d2a/d2b (measurement basis: README, "Matching the real-board (d3b)
# distribution"):
#
#   pads per net    Zipf s=2.955 (MLE; KS 0.034 < 0.044 critical @n=953) plus a
#                   tail correction of 16:0.018
#                   -> 2-pin 62% (d3b 64%), mean 3.12 (d3b 3.29)
#   spatial locality --net-locality 0.7 + size decay 10
#                   -> 2-pin net span / board diagonal 0.20 (d3b 0.20; d2b 0.39)
#   board content   nets 20.1 / pads 62.8 / connections 42.7  (d3b 19.1 / 68.7 / 43.7)
#   pad size        sampled per board in 1.0~1.9mm -> median 1.43 (d3b 1.40,
#                   p75 1.70, a match)
#   board scale     38~58mm rectangles. The board is drawn AFTER the nets and
#                   enlarged to fit their pad count, staying under the RSA limit
#                   (_min_area_for_pads) — drawing the size independently makes
#                   placement impossible on boards where Zipf drew several large
#                   nets (the old code failed at board 17,198 of 100k with 131
#                   pads at 99% saturation).
#                   Result: density p25~p90 3.5~6.2 (d3b 2.8~14.6),
#                   W+H 76.9 (d3b 83.9)
#
# **Vias deliberately depart from d3b.** The d3b via-free routability ceiling is
# 0.985 and 84% of its boards need no via at all — under that distribution most
# episodes never present the moment where a via is required. --thru-hole-prob
# 0.30 over-samples the positive cases, yielding 4.78 mandatory vias per board
# and 0% of boards with no mandatory via (on the assumption that the decision
# rule itself is frequency-independent and therefore transfers).
#
# Train with --directional-candidates ring_1mm: the default 0.5mm ring is
# smaller than a pad, so at the start of a route all 8 candidates land inside
# the starting pad (open space 0.0% on this dataset, 25.9% on d3b). At 1.0mm
# those become 99.8% / 86.5%.
set -euo pipefail
cd "$(dirname "$0")/../../.."
GEN=tools/datagen/synthetic_generator/generate_synthetic_boards.py
MIG=tools/datagen/synthetic_generator/migrate_dataset_to_pro.py
OUT="${OUT:-$PWD/var/datasets/synthetic/pcb_dataset_synthetic_d2c_100k}"
TRAIN_N="${TRAIN_N:-100000}"; VAL_N="${VAL_N:-128}"; TEST_N="${TEST_N:-128}"
WORKERS="${WORKERS:-16}"
COMMON=(--mode grid --pitch-formula c+w
        --board-size 40 --board-size-min 38 --board-size-max 58
        --pad-size 1.4 --pad-size-min 1.0 --pad-size-max 1.9
        --clearance 0.3 --trace-width 0.3 --via-dia 1.2 --via-drill 0.6
        --num-layers 2 --central-frac 0.8 --thru-hole-prob 0.30
        --nets-min 13 --nets-max 27 --pads-per-net-min 2 --pads-per-net-max 20
        --pads-per-net-zipf 2.955 --pads-per-net-zipf-tail 16:0.018
        --net-locality 0.7 --net-locality-decay 10
        --size-board-for-pads
        --seed-mode linear)
# linear seed bands: train 0.., val 1e9.., test 2e9.. (README "seed modes" convention)
gen(){ local split="$1" n="$2" seed="$3"
  echo "$(date '+%H:%M:%S') [$split] $n boards (seed base $seed)"
  python "$GEN" "${COMMON[@]}" --n "$n" --seed "$seed" --out-dir "$OUT/_raw_$split" > "$OUT/_raw_$split.log" 2>&1
  python "$MIG" --src "$OUT/_raw_$split" --dst "$OUT/_pro_$split" --rules 2layer_multi_pin \
    --workers "$WORKERS" > "$OUT/_pro_$split.log" 2>&1
  # rename the directory itself — with 100k pairs `mv dir/* dst/` blows the
  # argument-list limit.
  rm -rf "$OUT/$split"; mv "$OUT/_pro_$split" "$OUT/$split"
  rm -rf "$OUT/_raw_$split"
  echo "$(date '+%H:%M:%S') [$split] done: $(find "$OUT/$split" -name '*.kicad_pcb' | wc -l) pcb / $(find "$OUT/$split" -name '*.kicad_pro' | wc -l) pro"; }
mkdir -p "$OUT"
gen val   "$VAL_N"   1000000000
gen test  "$TEST_N"  2000000000
gen train "$TRAIN_N" 0
echo "$(date '+%F %T %Z') D2C DONE -> $OUT"
du -sh "$OUT"
