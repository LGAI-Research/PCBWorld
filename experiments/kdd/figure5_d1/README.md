# figure5_d1 — D1 grid scalability (paper Figure 5 / Figure 6c)

> **The D1 corpus is not distributed with this repository.** Nothing in this
> folder can be reproduced from a fresh clone: every script here needs a
> synthetic 1-layer grid dataset (and, for two of the baselines, runner scripts)
> that this repository does not ship and that no generator here reproduces.
> The scripts detect that up front and exit `2` with a notice naming the paths
> they wanted. Figure 5 is a figure in the paper, not a reproducible pipeline —
> the recipes are kept as a record of how the published numbers were produced.

## What is here

| file | role |
| --- | --- |
| `run.sh` | orchestrator — `train [transformer\|jumanji\|sable]` · `eval` · `figure` · `all` |
| `train_transformer_ppo.sh` | PCBWorld Transformer PPO trainer (one grid × seed) |
| `train_jumanji_a2c.sh` | Jumanji A2C baseline trainer (one grid × seed) |
| `train_sable.sh` | SABLE/Mava baseline trainer (one grid × seed) |
| `cases.sh` | the published hyperparameters per grid size, and the shared preflight |
| `plot_grid_scenarios.py` | the Connector-v2 grid scenario illustration |

`run.sh figure` (`draw_figure.py --figure fig6c`) is the one stage that always
runs: it reads `var/results/kdd/` and, with no D1 cells present, writes a figure
whose rows all read `(absent/OOM)`.

## What each script needs, and where it would come from

The recipes grew against four different on-disk conventions. None of them is
produced by anything in this repository — they are listed so the paths in the
preflight notices are legible, not as a build recipe.

| script | required input |
| --- | --- |
| `run.sh eval` | boards `$DATASET_ROOT/synthetic/synth_1L/grid<G>_5net_v15/test`, checkpoints `$CKPT_ROOT/Transformer_1L/grid<G>/seed<S>/policy_best.pt` |
| `train_transformer_ppo.sh` | split json `$DATASET_ROOT/synthetic/splits/synth_1L_grid<G>_*v<NN>_local.json` (or `--split-json`) |
| `train_jumanji_a2c.sh`, `train_sable.sh` | arrays `$DATASET_ROOT/synthetic/connector_v2/grid<G>/{train,val}.npz` |
| `plot_grid_scenarios.py` | arrays `$DATASET_ROOT/synthetic/connector_v2/grid<G>/test.npz` |

Notes on why none of these resolves out of the box:

- The `_local` suffix in the split-json name is the *gitignored, personal* split
  convention ([configs/datasets/README.md](../../../configs/datasets/README.md)),
  so such a file is never tracked.
- The shipped synthetic generators
  ([tools/datagen/synthetic_generator/](../../../tools/datagen/synthetic_generator/))
  write `var/datasets/synthetic/pcb_dataset_synthetic_<N>net_<P>pin_<L>layer_grid<G>`
  plus a split json under `configs/datasets/grids/`, and `make_grid_dataset.sh`
  is fixed at 10 nets × 2 pins. D1 is 5 nets × 2 pins under different directory
  names, so those generators do not reproduce it.
- The Connector-v2 `.npz` arrays are a Jumanji-side board encoding; no encoder
  for them is part of this tree.
- The rule-based D1 baselines use yet another root
  (`synthetic/synth_1L_grid<G>_5net_v02`, selected by `SYNTH1L_PCB_ROOT_<G>` —
  [methods/baselines/rule_based/_lib/datasets.py](../../../methods/baselines/rule_based/_lib/datasets.py)).

## Baselines that cannot be retrained here at all

`train_jumanji_a2c.sh` and `train_sable.sh` shell out to `run_v56_jumanji_a2c.py`
/ `run_v56_mava_sable.py` (and, for SABLE, a Mava source tree). Those runners are
not part of this repository, so these two scripts refuse immediately — including
under `--dry-run`, which would otherwise print a command line that could not run.
The hyperparameters those runs used are recorded in `cases.sh`
(`T1_JUMANJI_*`, `T1_SABLE_*`).

## Provenance

Checkpoint and dataset provenance for the published D1 numbers:
[experiments/kdd/PROVENANCE.md](../PROVENANCE.md) §1–§2.
