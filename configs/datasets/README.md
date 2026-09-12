# Dataset index — only what the tree does not show (mapping + gotchas)

**Source of truth = the split json itself + [`experiments/kdd/configs/quickstart/splits.json`](../../experiments/kdd/configs/quickstart/splits.json)
(the machine map the recipes consume).** The filename *is* the logical id, so most of it can be read straight off the tree. This document adds
only *what the tree does not show* — the d3a/b/c difficulty mapping, the gotchas, and where a file comes from. **Board counts, dataset_dir and
split membership are deliberately not copied here** (drift prevention) — read the JSON directly, or go through the loader:
`load_boards_from_split_json(json, diff, split)` / for training `resolve_board_list(...)` ([`methods/_shared/board_loader.py`](../../methods/_shared/board_loader.py)).

## Layout

```
configs/datasets/
  d3.json        ← the ONE shipped dataset: the D3 real boards (679 PCBench boards under $PCBWORLD_DATA_ROOT/pcbench/exacad_sorted)
  local/         ← personal / not promoted (gitignored; personal dataset_dir OK)

experiments/kdd/configs/datasets/     ← paper-only splits, kept with the paper recipes
  d2a.json                            synth 2-layer v2 (D2-A in the paper; README §2 generates it)
  grids/ 10net_2pin_1layer_v2.json    synth 1-layer split (generator output, Figure 5)
  misc/  multi_pin_2layer_v2.json     synth 2-layer split (generator output)
```

The `grids/` and `misc/` entries are written by [`tools/datagen/synthetic_generator/`](../../tools/datagen/synthetic_generator/):
the setup scripts build the board set and its split json in one pass, so a missing file there is regenerated rather than restored.

## Logical id → coordinates (the d3 difficulty axis · the d1 family)

| id | split json | difficulty | what |
|----|-----------|-----------|------|
| **d3a** | `d3.json` | `easy` | real boards, small (PCBench). The 679 paper boards; README Quick start §3 rebuilds them from the public PCBench clone with [`tools/datagen/pcbench_prep/`](../../tools/datagen/pcbench_prep/README.md) into `pcbench/exacad_sorted` |
| **d3b** | `d3.json` | `medium` | real boards, medium (same set) |
| **d3c** | `d3.json` | `hard` | real boards, large (same set) |
| **d2a** | [`experiments/kdd/configs/datasets/d2a.json`](../../experiments/kdd/configs/datasets/d2a.json) | `easy` | synth 2-layer v2 (D2-A in the paper) |
| **d1** | **not distributed** — no split json ships for it | `easy` | synth 1-layer grid sweep (G=10…1000). [`experiments/kdd/figure5_d1/train_transformer_ppo.sh`](../../experiments/kdd/figure5_d1/train_transformer_ppo.sh) loads the grid folders directly |

- **d3a/b/c are one file, `d3.json`** + difficulty (a=easy·b=medium·c=hard). Eval / best-ckpt conventionally use `split=test`. **D3 is
  evaluation-only by design** — the shipped RL recipes train on synthetic boards; every D3 number is zero-shot.
- The d3 json uses a per-board-dir layout (`<root>/<bid>/<board_filename>`) — the loader handles it through the top-level `board_filename` key.
- `d3.json` is rebuilt by [`experiments/kdd/d3_dataset/`](../../experiments/kdd/d3_dataset/) from the corpus directory.

## Gotchas (derived facts the tree does not show)

- **Rebuilt vs paper files.** A rebuild by the pcbench_prep chain reproduces the paper's 679 boards, order, splits and `.kicad_pro` rules; the
  guide boards differ only in the per-net track width chosen on 40 boards (the paper's set is one sample of a then non-deterministic `kicad-cli`
  DRC — details: pcbench_prep README, "Expected result"). One split file serves both: what sits in `pcbench/exacad_sorted` is your D3.
- **`d3.json`'s `train` key mirrors the full list — deliberately.** It is the canonical full board list, consumed test-only; the `train` key
  keeps the full list as provision for a possible future D3-train split, so `test` ⊂ `train` in every difficulty. Not leakage — the
  benchmark's models never train on it.
- ⚠️ **`0170_hackaday_esp-14…__autosave` is permanently excluded from the d3 pool**: its only netclass has clearance 0.0 and BDS min 0.0 →
  an autosave leftover with an effective clearance of 0. DRC cannot catch zero-gap routing there, so it does not qualify as a benchmark
  board (the only such case among the 679 exacad boards). `d3.json` therefore carries **678** boards; the excluded board
  sits in the medium tier, so the medium bucket holds 286 boards — one fewer than the 287 the `d3_dataset` builder classifies from the raw CSV.
- **Naming convention**: tasks and datasets all use a lowercase `d`, **filename = logical id**; personal / not-yet-promoted files go in `local/`.
- **Legacy on-disk alias**: cell paths under `var/results/kdd/` keep the legacy `t3/{t3a,t3b,t3c}` directories (the loader applies a d→t alias).
- **External data root**: dataset locations live outside the repo under `$PCBWORLD_DATA_ROOT` (layout: [configs/paths.yaml](../paths.yaml)).
