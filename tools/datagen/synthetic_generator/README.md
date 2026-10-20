# synthetic_generator

Generates synthetic PCB boards — unrouted `.kicad_pcb` pad/net layouts plus the matching
`.kicad_pro` design rules — and assembles them into train/val/test dataset directories. The
grid and multi-pin orchestrators write under the repo root, the D2-B/D2-C families under
`var/datasets/`.

## Building blocks

One generator; everything else wraps it or post-processes its output.

| File | Role |
|---|---|
| [generate_synthetic_boards.py](generate_synthetic_boards.py) | The sampler — N boards in one deterministic worker. `--mode d2b` is the real-board-matched 2L variant (paired mode emits the fixed-rule and per-net-rule twins together); `--geo` pulls in the geometry sampler below. |
| [outline_geometry.py](outline_geometry.py) | `--geo` geometry: 4 outline kinds (rect / corner-fillet arc / rectilinear polygon / circle), internal cutouts, NPTH holes, oval slots, diversified THT pads, placement keepout and capacity derate. Distribution constants come from a census of the real d3a/d3b boards. The board box is `w×h`, so `--aspect-sigma` stretches outline, holes, slots and cutouts together — only circles keep a square bbox (they use the geometric-mean side). |
| [generate_grid_boards.py](generate_grid_boards.py) | Derives grid spacing / clearance / trace_width / pad_size / min_sep from the single `--grid N` argument. |
| [migrate_dataset_to_pro.py](migrate_dataset_to_pro.py) | Strips the `(net_class …)` blocks and emits a companion `.kicad_pro` (v2 format): one engine round-trip for a template, then bulk text edits. **Shared-rule datasets only.** |
| [convert_pernet_to_pro.py](convert_pernet_to_pro.py) | Per-board engine round-trip (`load_and_save_via_engine`) emitting a `.kicad_pro` that holds per-net rules — required for D2-B/D2-B-V, and doubling as an engine-load smoke test over every board. Failures that appear at high worker counts get one 4-worker retry pass; anything still failing exits 1. |

## Orchestrators

| File | Output |
|---|---|
| [generate_grid_dataset.sh](generate_grid_dataset.sh) | `pcb_dataset_synthetic_<nets>net_<pins>pin_<L>layer_grid<N>(_test)` — 10K train (seed 0) + 128 test (seed 1), `_v2` added in place |
| [generate_multi_pin_2layer.sh](generate_multi_pin_2layer.sh) | `pcb_dataset_synthetic_multi_pin_2layer(_test)` — 1M train (8 shards, seeds 0..7) + 2K test (seed 9999) |
| [generate_multi_pin_var_2layer.sh](generate_multi_pin_var_2layer.sh) | the same with a variable net/pin distribution |
| [setup_10net_2pin_1layer_split_v2.py](setup_10net_2pin_1layer_split_v2.py) | trims to `_v2`, symlinks train+test into `pcb_dataset_10net_2pin_1layer_combined_v2/` (test boards renamed `testboard_NNNNN`) and writes the split json |
| [setup_multi_pin_synthetic_split_v2.py](setup_multi_pin_synthetic_split_v2.py) | the same for the 2L side (train `_v2` = a copy trimmed to the first 10K) |
| [setup_indexed_datasets.sh](setup_indexed_datasets.sh) | `..._combined_v2__r01..` dereferenced real-file copies that avoid NFS lock contention |
| [generate_validation_set.sh](generate_validation_set.sh) | adds 128 validation boards (fresh seeds) to `combined_v2` and to every `__r0X` |
| [generate_D2B.sh](generate_D2B.sh) | `var/datasets/pcb_dataset_synthetic_d2b{,v}/{train,val,test}` — **the official d2b/d2bv recipe** (10k/128/128, paired); its arguments are verified to reproduce the distributed boards bit-for-bit apart from uuids |
| [generate_D2B_geo.sh](generate_D2B_geo.sh) | the same plus `--geo`; train shares the topology-seed prefix with plain d2b, so boards compare one to one. `ASPECT_SIGMA=0.60` writes non-square boards under a separate `_ar` root (the default 0 keeps the square pool byte-identical) |
| [generate_d2c.sh](generate_d2c.sh) | `var/datasets/synthetic/pcb_dataset_synthetic_d2c_100k/` — the d3b-matched recipe (below); its header records the measurement behind every value |

## Dataset layout

Two canonical datasets under `$PCBWORLD_DATA_ROOT/`, plus their `__r01..r10` (1L) / `__r01..r06` (2L) copies:

```
pcb_dataset_10net_2pin_1layer_combined_v2__r01/  # 1L, 10 nets x 2 pins
  board_NNNNN.{kicad_pcb,kicad_pro}      # train, n=10000  (seed 0)
  testboard_NNNNN.{kicad_pcb,kicad_pro}  # test,  n=128    (seed 1)
  valboard_NNNNN.{kicad_pcb,kicad_pro}   # val,   n=128    (seed 2)

pcb_dataset_multi_pin_2layer_combined_v2__r01/   # 2L, 5 nets (2,2,2,3,4) pins
  board_NNNNN.{kicad_pcb,kicad_pro}      # train, n=10000  (seed 0, shard 0)
  testboard_NNNNN.{kicad_pcb,kicad_pro}  # test,  n=128    (seed 9999)
  valboard_NNNNN.{kicad_pcb,kicad_pro}   # val,   n=128    (seed 1234)
```

Consumers reach them through the `easy.train` / `easy.test` / `easy.val` split definitions in
`experiments/kdd/configs/datasets/{grids/10net_2pin_1layer_v2,misc/multi_pin_2layer_v2}.json`.

## Reproduction

```bash
cd tools/datagen/synthetic_generator
bash generate_grid_dataset.sh 1000          # 1L: 10K train + 128 test
bash generate_multi_pin_2layer.sh           # 2L: SHARDS=1 TRAIN_N=10000 for the 10K variant
python setup_10net_2pin_1layer_split_v2.py  # _v2 trim + combined_v2 symlinks + split json
python setup_multi_pin_synthetic_split_v2.py
bash setup_indexed_datasets.sh              # __r0X copies for parallel runs
bash generate_validation_set.sh             # +128 val boards on non-colliding seeds
bash generate_D2B.sh                        # d2b / d2bv
bash generate_D2B_geo.sh                    # d2b_geo / d2bv_geo
bash generate_d2c.sh                        # d2c
```

The D2-* recipes need the engine: conda env `pcbworld` + `PYTHONPATH=build_rl/pcbnew/python/rl:.`.

## Seeds

`--seed-mode` fixes the per-board RNG seed from the split's base seed and the board index:

- **`linear` (default)** — `base + idx`. Board content is independent of the shard count, and
  splits are kept apart by base-offset bands (train 0, val 1e9, test 2e9 — what D2-A-100k,
  d2b_geo and d2c use).
- **`legacy`** — `base * 1_000_003 + idx`. Every orchestrator that reproduces an
  already-released dataset pins it, so a default flip cannot change those boards. A new legacy
  split needs a base seed whose band overlaps none of those listed above.

## Matching the real-board (d3b) distribution

The axis on which synthetic boards differ from real PCBs is **not** the need for vias: over the 50
d3b boards the via-free routability ceiling is 0.985, above d2b's 0.976 (62.6% of the pads are
thru-hole). Three axes do diverge:

| Axis | d2b | d3b (50 boards) | Flag |
|---|---|---|---|
| pads per net | 62% peak at 3 pins | Zipf, 64% at 2 pins | `--pads-per-net-zipf`, `--pads-per-net-zipf-tail` |
| 2-pin net span / board diagonal | 0.386 | 0.203 | `--net-locality`, `--net-locality-decay` |
| pad density per 100mm² | 2.96 | 7.04 | `--nets-*`, `--board-size` |

- `--pads-per-net-zipf S` draws the pad count per net from `P(k) ~ k^-S` (d3b MLE `S=2.955`). Real
  boards have a heavier tail than a pure power law (power nets carry 20–42 pins), so
  `--pads-per-net-zipf-tail FROM:MASS` lifts `P(k>=FROM)` to a target mass (d3b: `16:0.018`).
- `--net-locality L` — `_place_pads` is net-agnostic and `_render` assigns nets by slicing the
  placement order, so at L=0 net membership is spatially random and the pad list comes back
  unchanged (**existing datasets reproduce bit-identically**). With L>0 each next pad is drawn
  from the `K = ceil(remaining^(1-L))` nearest candidates, and `--net-locality-decay K` fades L
  linearly to 0 at K pins, because on real boards only the small nets are local.

[generate_d2c.sh](generate_d2c.sh) is the tuned combination of all three; its header carries the
measured result per axis, and where it deliberately departs from d3b.

## Pitfalls

| Issue | What to do |
|---|---|
| **`--size-board-for-pads` moves the RNG order** | Drawing the board size independently of a heavy-tailed net draw produces unplaceable boards — random sequential adsorption of equal disks saturates at an area fill of 0.547, and `_place_pads` reports the overshoot only after burning its whole try budget. The flag draws the nets first, then enlarges the board to fit (`_min_area_for_pads`, safety factor 0.50). It is **off by default** because the reordering changes what a seed produces; turn it on for new datasets only. |
| **Any reordering of RNG draws** | Silently changes every existing dataset. [tests/test_synthetic_dataset_reproduction.py](../../../tests/test_synthetic_dataset_reproduction.py) regenerates one board per canonical task and compares a geometry signature — run it before trusting a regenerated set. |
| **roundrect pads meeting at 45°** | `_place_pads` filters candidates by Euclidean center distance, but SMD pads are square roundrects, and two facing corners come much closer than the center distance suggests. `_min_sep_for_clearance` bounds it — for side `s` and corner radius `r = 0.25s`, `d >= sqrt(2)*(s - 2r) + 2r + clearance`. Only dense datasets reach the bound (a denser trial placement produced exactly the predicted 0.1314mm gap). |
| **Per-net rules need `convert_pernet_to_pro.py`** | `migrate_dataset_to_pro.py` understands shared rules only; run it on a D2-B set and the per-net rules are lost. |
| **`.kicad_pro` conversion throughput** | ~25 boards/s at 24 workers, ~61 at 48; the first start can be slow from NFS cold-start. |
