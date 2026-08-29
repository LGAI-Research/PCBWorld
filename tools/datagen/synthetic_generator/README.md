# synthetic_generator

Synthetic PCB dataset generation / migration / split pipeline.
The grid and multi-pin orchestrators write directly under the repo root; the
D2-B family writes under `var/datasets/`.

## Core building blocks

| File | Role |
|---|---|
| [generate_synthetic_boards.py](generate_synthetic_boards.py) | Generates N `.kicad_pcb` files in a single worker (deterministic). `--mode d2b` = the real-board-matched 2L sampler (paired mode emits the D2-B fixed twin and the D2-B-V per-net-rule twin together). The per-board seed follows `--seed-mode`: **`linear` (default)** → `base_seed + (start_index+i)`; **`legacy`** → `base_seed*1_000_003 + (start_index+i)`. |
| [outline_geometry.py](outline_geometry.py) | `--geo` geometry sampler — 4 outline kinds (4×gr_line rect / corner-fillet arc / rectilinear polygon / gr_circle) plus internal cutouts, NPTH holes, oval slots and diversified THT pads. The board box is `w×h`, so with `--aspect-sigma` on, outline, holes, slots and cutouts stretch together (only circles keep a square bbox — they use the geometric mean side instead). Includes placement keepout and capacity derate. Distribution constants come from a census of the real d3a/d3b boards. |
| [generate_grid_boards.py](generate_grid_boards.py) | Wrapper deriving grid spacing / clearance / trace_width / pad_size / min_sep from the single `--grid N` argument. |
| [migrate_dataset_to_pro.py](migrate_dataset_to_pro.py) | Strips the `(net_class …)` blocks and emits a companion `.kicad_pro` (v2 format). The first board is round-tripped through the KiCad engine to extract a template `.kicad_pro`; the rest are processed in bulk with text operations. **Shared-rule datasets only** — for per-net rules use convert_pernet below. |
| [convert_pernet_to_pro.py](convert_pernet_to_pro.py) | **Per-board** KiCad engine round-trip (`KiCadEngine(board).save()`): strips legacy net_class blocks and emits a `.kicad_pro` holding the per-net rules. Required for D2-B/D2-B-V, and doubles as an engine-load smoke test over every board. Throughput ~25 boards/s (24 workers) · ~61 boards/s (48 workers); the first start can be slow because of NFS cold-start. Transient failures that occasionally appear at high worker counts are retried once by a 4-worker retry pass; anything still failing exits 1. |

## End-to-end orchestrators

| File | Output dataset prefix | Composition |
|---|---|---|
| [generate_grid_dataset.sh](generate_grid_dataset.sh) | `pcb_dataset_synthetic_<nets>net_<pins>pin_<L>layer_grid<N>(_test)` | 10K train (seed=0) + 128 test (seed=1). `_v2` is added in-place in the same dir. |
| [generate_multi_pin_2layer.sh](generate_multi_pin_2layer.sh) | `pcb_dataset_synthetic_multi_pin_2layer(_test)` | 1M train (8 parallel shards, seed=0..7) + 2K test (seed=9999). |
| [generate_multi_pin_var_2layer.sh](generate_multi_pin_var_2layer.sh) | `pcb_dataset_synthetic_multi_pin_var_2layer(_test)` | Variable net/pin distribution variant. |
| [setup_10net_2pin_1layer_split_v2.py](setup_10net_2pin_1layer_split_v2.py) | `pcb_dataset_10net_2pin_1layer_combined_v2/` + `configs/datasets/grids/10net_2pin_1layer_v2.json` | Symlinks the `_v2` train (10K) + test (128) sets, renaming test boards to `testboard_NNNNN`. |
| [setup_multi_pin_synthetic_split_v2.py](setup_multi_pin_synthetic_split_v2.py) | `pcb_dataset_multi_pin_2layer_combined_v2/` + `configs/datasets/misc/multi_pin_2layer_v2.json` | Same as above for the 2L side. Train is `_v2` (a copy trimmed to the first 10K). |
| [setup_indexed_datasets.sh](setup_indexed_datasets.sh) | `..._combined_v2__r{01..10}` (1L) / `..._combined_v2__r{01..06}` (2L) | Dereferenced real-file copies that avoid NFS lock contention. |
| [generate_validation_set.sh](generate_validation_set.sh) | `..._val(_v2)` + `valboard_NNNNN` symlinks in combined_v2 + real copies in every `__r0X` | Adds 128 validation boards with no seed collisions. |
| [generate_D2B.sh](generate_D2B.sh) | `var/datasets/pcb_dataset_synthetic_d2b{,v}/{train,val,test}` | **The official d2b/d2bv recipe** (10k/128/128, paired, `--seed-mode legacy`). The arguments are verified to reproduce the distributed boards bit-for-bit apart from uuids. |
| [generate_D2B_geo.sh](generate_D2B_geo.sh) | `var/datasets/pcb_dataset_synthetic_d2b_geo{,v_geo}{,_ar}/{train,val,test}` | The D2B recipe plus `--geo` (real-board geometry: outline / cutouts / NPTH / slots / THT). `--seed-mode linear` (train 0 / val 1e9 / test 2e9); train shares the topology-seed prefix with the plain d2b set, so boards can be compared one to one. Census distributions land within a few percentage points of d3b, pads/cm² −1.5%. `ASPECT_SIGMA=0.60` gives non-square boards under a separate `_ar` root (the default 0 keeps the square pool byte-identical). |

## Current canonical datasets

Two datasets under `$CADAGENT_DATA_ROOT/` (and their repo-local
`__r01..r10` / `__r01..r06` copies):

```
pcb_dataset_10net_2pin_1layer_combined_v2__r01/  # 1L, 10nets x 2pin
  board_NNNNN.{kicad_pcb,kicad_pro}      # train, n=10000  (seed=0)
  testboard_NNNNN.{kicad_pcb,kicad_pro}  # test,  n=128    (seed=1)
  valboard_NNNNN.{kicad_pcb,kicad_pro}   # val,   n=128    (seed=2)

pcb_dataset_multi_pin_2layer_combined_v2__r01/   # 2L, 5nets (2,2,2,3,4) pin
  board_NNNNN.{kicad_pcb,kicad_pro}      # train, n=10000  (seed=0 shard0)
  testboard_NNNNN.{kicad_pcb,kicad_pro}  # test,  n=128    (seed=9999)
  valboard_NNNNN.{kicad_pcb,kicad_pro}   # val,   n=128    (seed=1234)
```

The matching split definitions (`easy.train` / `easy.test` / `easy.val`) live in
`configs/datasets/grids/10net_2pin_1layer_v2.json` and
`configs/datasets/misc/multi_pin_2layer_v2.json`.

## Reproduction (from scratch)

```bash
# 1L (10K train + 128 test)
bash tools/datagen/synthetic_generator/generate_grid_dataset.sh 1000

# 2L (10K train + 128 test — for 10K instead of 1M, set SHARDS=1 TRAIN_N=10000)
bash tools/datagen/synthetic_generator/generate_multi_pin_2layer.sh

# the setup_*.py scripts do the v2 trimming and build the reference dirs:
#   - _v2 (10K train + 128 test) symlinked into combined_v2/
python tools/datagen/synthetic_generator/setup_10net_2pin_1layer_split_v2.py
python tools/datagen/synthetic_generator/setup_multi_pin_synthetic_split_v2.py

# r0X copies for parallel runs
bash tools/datagen/synthetic_generator/setup_indexed_datasets.sh

# 128 extra validation boards (separate seeds, no overlap with train/test)
bash tools/datagen/synthetic_generator/generate_validation_set.sh

# D2-B family (engine required: conda cadagent + PYTHONPATH=build_rl/pcbnew/python/rl:.)
bash tools/datagen/synthetic_generator/generate_D2B.sh       # d2b/d2bv (legacy seed)
bash tools/datagen/synthetic_generator/generate_D2B_geo.sh   # d2b_geo/d2bv_geo (geometry extension)
```

## Matching the real-board (d3b) distribution — `--pads-per-net-zipf` / `--net-locality`

The axis on which synthetic boards differ from real PCBs (d3b) is **not the need
for vias**. Measured over the 50 d3b boards: the via-free routability ceiling is
**0.985**, higher than d2b's 0.976, and only 3.5% of nets are topologically
forced to use a via (62.6% of the pads are thru-hole, so most nets need no layer
change at all). The three axes that do diverge are:

| Axis | d2b | d3b (50 boards) | How to match it |
|---|---|---|---|
| pads per net | 62% peak at 3 pins | **Zipf**, 64% at 2 pins | `--pads-per-net-zipf` |
| spatial locality (2-pin net span / board diagonal) | 0.386 | **0.203** | `--net-locality` |
| pad density (per 100mm²) | 2.96 | 7.04 | `--nets-*` / `--board-size` |

**`--pads-per-net-zipf S`** draws the pad count per net from a discrete power law
`P(k) ~ k^-S`. The d3b MLE is `S=2.955` (KS distance 0.034 < the 0.044 critical
value at n=953, so the fit is not rejected). Real boards have a heavier tail than
a pure power law (power nets carry 20–42 pins), so `--pads-per-net-zipf-tail
FROM:MASS` lifts `P(k>=FROM)` to a target mass — for d3b that is `16:0.018`.

**`--net-locality L`** — pad placement (`_place_pads`) is net-agnostic and
`_render` assigns nets by slicing the placement order, so at the default 0.0 net
membership is spatially random (a 2-pin net spans ~52% of a board edge, the
expected distance between two uniform points). With `L>0` each next pad is picked
from the `K = ceil(remaining^(1-L))` nearest candidates. **`--net-locality-decay
K`** fades L linearly with net size, reaching 0 at K pins — on real boards only
the small nets are local, while power nets reach across the whole board.

Measured with `--net-locality 0.7 --net-locality-decay 10` (40 generated boards
vs the 50 d3b boards):

| net span / board diagonal | d2b | generated | d3b50 |
|---|---|---|---|
| 2 pins | 0.386 | **0.202** | 0.203 |
| 3 pins | 0.561 | **0.327** | 0.321 |
| 4-5 pins | 0.687 | **0.441** | 0.440 |
| 6-9 pins | 0.806 | 0.610 | 0.694 |
| 10+ pins | 0.899 | 0.949 | 0.895 |

`--net-locality 0` (the default) returns the pad list unchanged, so **existing
datasets reproduce bit-identically**.

### A d3b-approximating recipe

`--min-sep 3.0` / `--board-size 42` are back-derived from d2b_100k measurements
(nearest pad distance 3.02mm, pad bbox 33x33mm). Density stays at the d2b level
(2.92) and only the distribution changes — raising it to the d3b level (7.04)
requires a larger `--nets-max`, and the `min_sep` constraint can then make
`_place_pads` fail.

```bash
python tools/datagen/synthetic_generator/generate_synthetic_boards.py \
  --mode grid --pitch-formula c+w --board-size 42 \
  --clearance 0.3 --trace-width 0.3 --pad-size 2.4 --via-dia 1.2 --via-drill 0.6 \
  --num-layers 2 --central-frac 0.8 --min-sep 3.0 --seed-mode linear \
  --thru-hole-prob 0.62 --nets-min 6 --nets-max 18 \
  --pads-per-net-min 2 --pads-per-net-max 42 \
  --pads-per-net-zipf 2.955 --pads-per-net-zipf-tail 16:0.018 \
  --net-locality 0.7 --net-locality-decay 10 \
  --n 10000 --seed 0 --out-dir var/datasets/synthetic/d2b_zipfloc/train
```

### `--size-board-for-pads` — growing the board to fit the nets

With a heavy-tailed distribution such as `--pads-per-net-zipf`, drawing the board
size **independently** of the nets produces unplaceable combinations. Random
sequential adsorption of equal disks saturates at an area fill of **0.547**, and
`_place_pads` only reports a demand beyond that limit as a failure **after
burning its whole try budget** — for example a draw of 131 pads at `min_sep`
2.524 into 1210mm² of usable area asks for 99% of saturation.

With this flag the net structure is drawn **first** and `cfg_factory(rng,
total_pads=...)` then enlarges the board to fit those pads (`_min_area_for_pads`,
with a safety factor of 0.50 that fills only half of saturation).

**It is OFF by default** — the reordering changes RNG consumption, so with the
flag on the same seed produces a different board. Existing datasets (d1 grid50,
d2a synth_2L_v2, …) depend on the original order and
[tests/test_synthetic_dataset_reproduction.py](../../../tests/test_synthetic_dataset_reproduction.py)
enforces it. Turn the flag on for new datasets only (the d2c recipe does).

### roundrect 45° clearance

`_place_pads` filters candidates by **Euclidean distance**, but SMD pads are
square `roundrect`. Two of them facing each other at 45° come much closer than
their center distance suggests. For side `s` and corner radius `r = 0.25s`, the
gap between the facing corner arcs is at least `clearance` only when

```
d >= sqrt(2)*(s - 2r) + 2r + clearance          (_min_sep_for_clearance)
```

A denser trial placement produced a violating gap of **0.1314mm**, exactly the
value this formula predicts. Lower-density datasets simply stay far enough from
the bound for it to matter.

## Seed modes & collision guide

The per-board RNG seed is set by `--seed-mode`:

- **`linear` (default)** — `seed = base_seed + (start_index + i)`. Board content is independent of the shard count (changing SHARDS yields the same boards), and splits are separated into non-overlapping base-offset bands. Example: train base `0` → seeds `0..N-1`, val base `1_000_000_000`, test base `2_000_000_000` (this is what the `D2-A-100k` dataset uses).
- **`legacy`** — `seed = base_seed * 1_000_003 + (start_index + i)`. Reproduces the datasets in the table below bit-identically. Pin it only when regenerating those datasets (every existing orchestrator already fixes `--seed-mode legacy`).

Legacy seed bands (avoid these when adding a new split in legacy mode):

| dataset | split | seed | idx range | board RNG seed range |
|---|---|---|---|---|
| 1L | train | 0 | 0..9999 | 0 .. 9999 |
| 1L | test  | 1 | 0..127  | 1_000_003 .. 1_000_130 |
| 1L | val   | 2 | 0..127  | 2_000_006 .. 2_000_133 |
| 2L | train | 0..7 (shard) | 0..124999 | 0 .. ~7_125_000 |
| 2L | test  | 9999 | 0..127 | 9_999_029_997 .. 9_999_030_124 |
| 2L | val   | 1234 | 0..127 | 1_234_003_702 .. 1_234_003_829 |

## Reproducibility check

The determinism of `generate_synthetic_boards.py` can be checked as follows — the
result must be byte-identical to the existing test set:

```bash
# 1L test — byte-diff board_00000..03 (reproducing a legacy dataset, so --seed-mode legacy is required)
python tools/datagen/synthetic_generator/generate_synthetic_boards.py \
    --n 4 --seed 1 --seed-mode legacy --mode grid --num-layers 1 --board-size 100 \
    --clearance 0.05 --trace-width 0.05 --pitch-formula c+w \
    --pad-size 0.05 --min-sep 0.1 \
    --fixed-pads-per-net 2,2,2,2,2,2,2,2,2,2 \
    --central-frac 1 --via-dia 0.6 --via-drill 0.3 \
    --out-dir /tmp/repro_1L
for i in 0 1 2 3; do
  f=$(printf "board_%05d.kicad_pcb" $i)
  cmp /tmp/repro_1L/$f pcb_dataset_synthetic_10net_2pin_1layer_grid1000_test/$f
done
```
