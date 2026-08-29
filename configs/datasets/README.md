# Dataset index — only what the tree does not show (mapping + gotchas)

**Source of truth = the split json itself + [`configs/quickstart/kdd/splits.json`](../quickstart/kdd/splits.json) (the machine map the code consumes).**
The filename *is* the logical id (`d2a.json`·`d3.json`…), so most of it can be read straight off the tree. This document adds only *what the tree does not show* —
the d3a/b/c difficulty mapping, the gotchas, and where a file comes from. **Board counts, dataset_dir and split membership are deliberately not copied here** (drift prevention) —
read the JSON directly, or go through the loader: `load_boards_from_split_json(json, diff, split)` / for training `resolve_board_list(...)`
([`methods/_shared/board_loader.py`](../../methods/_shared/board_loader.py)).

## Layout

```
configs/datasets/
  d2a.json  d3.json    ← core logical datasets (tracked)
  d3_public_811.json   ← the d3 board set re-split ≈8:1:1 train/val/test (tracked)
  grids/  10net_2pin_1layer_v2.json   ← synth 1-layer split (generator output)
  misc/   multi_pin_2layer_v2.json    ← synth 2-layer split (generator output)
  local/                              ← personal / not promoted (gitignored; personal dataset_dir OK)
```

The `grids/` and `misc/` entries are written by
[`tools/datagen/synthetic_generator/`](../../tools/datagen/synthetic_generator/): the setup
scripts build the board set and its split json in one pass, so a missing file there is
regenerated rather than restored.

## Logical id → coordinates (what the filename alone does not give: the d3 difficulty axis · the d1 family)

| id | split json | difficulty | what |
|----|-----------|-----------|------|
| **d1** | **not distributed** — no split json ships for it | `easy` | synth 1-layer grid sweep (G=10…1000). [`experiments/kdd/figure5_d1/train_transformer_ppo.sh`](../../experiments/kdd/figure5_d1/train_transformer_ppo.sh) looks for `<dataset root>/synthetic/splits/synth_1L_grid{G}_*v{NN}_local.json` (the gitignored `local/` convention) or takes `--split-json`, and exits 2 with a notice when neither resolves. What each D1 script needs: [`experiments/kdd/figure5_d1/README.md`](../../experiments/kdd/figure5_d1/README.md) |
| **d2a** | `d2a.json` | `easy` | synth 2-layer v2 (D2-A in the paper) |
| **d3a** | `d3.json` | `easy` | real boards, small (PCBench) |
| **d3b** | `d3.json` | `medium` | real boards, medium |
| **d3c** | `d3.json` | `hard` | real boards, large |
| — | `d3_public_811.json` | `easy`·`medium`·`hard` | the **same** board set as `d3.json`, re-split into disjoint `train`/`val`/`test`, stratified by `(tier, layers, wire_type)`. The `811` in the name is the *target* ratio and is applied per stratum, so the realised split is only **approximately** 8:1:1 (gotcha below). Its own `_split_meta` / `_stratum_stats` blocks record the exact parameters and per-stratum counts |

- **d3a/b/c are one file, `d3.json`** + difficulty (a=easy·b=medium·c=hard). Eval / best-ckpt conventionally use `split=test` (= `quickstart/kdd/splits.json`). **D3 is evaluation-only by design** — the shipped RL recipes train on synthetic boards and evaluate zero-shot on D3; nothing trains on `d3.json`.
- The d3 real-board json uses a per-board-dir layout (`<root>/<bid>/<board_filename>`) — the loader handles it through the top-level `board_filename` key.
- `d3.json` is rebuilt by [`experiments/kdd/d3_dataset/`](../../experiments/kdd/d3_dataset/); `d3_public_811.json` is a re-split of the same board set and is not produced by that builder.

## Gotchas (derived facts the tree does not show)

- **`d3.json`'s `train` key mirrors the full list — deliberately.** `d3.json` is the canonical
  full board list, consumed test-only (see above); the `train` key keeps the full list as
  provision for a possible future D3-train split, so `test` ⊂ `train` in every difficulty.
  That is intentional, not leakage — the benchmark's models never train on it. To actually
  train on D3, use `d3_public_811.json` (disjoint, ≈8:1:1).
- ⚠️ **`d3_public_811.json` is ≈8:1:1, not exactly 8:1:1**: the target ratio is applied *inside* each
  stratum (`i%10==0`→test, `1`→val, else→train), and restarting that rule on each of the 21 strata rounds
  val and test up. The realised totals are **train 530 / val 74 / test 74** of the 678 boards
  = 78.2 / 10.9 / 10.9 %. That is a consequence of the documented per-stratum rule, not a defect — the
  splits are disjoint and the per-tier `_stratum_stats` sum to the list lengths.
- ⚠️ **`0170_hackaday_esp-14…__autosave` is permanently excluded from the d3 pool**: its only netclass has
  clearance 0.0 and BDS min 0.0 → an autosave leftover with an effective clearance of 0. DRC cannot catch
  zero-gap routing there, so it does not qualify as a benchmark board (the only such case among the 679
  exacad boards). Both `d3.json` and `d3_public_811.json` therefore carry **678** boards. The excluded
  board sits in the medium tier, so the medium bucket holds 286 boards — one fewer than the 287 the
  `d3_dataset` builder classifies from the raw CSV.
- **Naming convention**: tasks and datasets all use a lowercase `d`, **filename = logical id** (`d2a.json`·`d3.json`), personal / not-yet-promoted files go in `local/`, grids and the rest in `grids/` / `misc/`.
- **Legacy on-disk alias**: cell paths under `var/results/kdd/` keep the legacy `t3/{t3a,t3b,t3c}` directories (the loader applies a d→t alias) — artifact names are unchanged.
- **External data root**: dataset locations live outside the repo under `$CADAGENT_DATA_ROOT`
  (layout: [configs/paths.yaml](../paths.yaml)), so their directory names are not rename targets
  (`dataset_dirs` values and SOURCE_PATH.txt stay as they are).
- **The data root is read-only from this repo**: no direct creation or writes — the synthetic generator
  writes to `var/datasets/synthetic/` by default.
