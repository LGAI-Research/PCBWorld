# figure5_d1 — D1 grid scalability (paper Figure 5 / Figure 6c)

> **The D1 corpus is not distributed with this repository.** The published
> numbers cannot be reproduced from a fresh clone: the paper's boards are not
> shipped, and two of the three baselines also need runner scripts that are not
> in this tree. What *can* be run from a fresh clone is the PCBWorld row's
> protocol on a locally generated, D1-shaped corpus — `run.sh train transformer`
> and `run.sh eval` default to what `make_grid_dataset.sh` writes, and every
> stage exits `2` with a notice naming that command when the inputs are absent.
> Boards generated that way are not the paper's boards (below), so the curve
> reproduces the protocol, not the published values.

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

## Running the PCBWorld row from a fresh clone

```bash
bash tools/datagen/synthetic_generator/make_grid_dataset.sh 10   # ... 50 100 200 500
bash experiments/kdd/figure5_d1/run.sh train transformer
bash experiments/kdd/figure5_d1/run.sh eval
```

The generator writes the boards under `var/datasets/synthetic/` and a split json
under `experiments/kdd/configs/datasets/grids/`; those two paths are the stage
defaults, named in one place (`cases.sh` `d1_generated_*`) so the recipe and the
generator cannot drift. `N_TRAIN`/`N_TEST`/`N_VAL` size the corpus.

## What each script needs, and where it would come from

| script | input, default first |
| --- | --- |
| `run.sh eval` | boards `var/datasets/synthetic/pcb_dataset_synthetic_10net_2pin_1layer_grid<G>_test` (paper: `BOARDS_DIR=$DATASET_ROOT/synthetic/synth_1L/grid<G>_5net_v15/test`), checkpoints `$CKPT_ROOT/Transformer_1L_300/grid<G>/seed<S>/policy_iter_300.pt` (the paper's: the last checkpoint, not `policy_best.pt`) |
| `train_transformer_ppo.sh` | split json `experiments/kdd/configs/datasets/grids/10net_2pin_1layer_grid<G>_v2.json`; a staged `$DATASET_ROOT/synthetic/splits/synth_1L_grid<G>_*v<NN>_local.json` wins when present, `--split-json` over both |
| `train_jumanji_a2c.sh`, `train_sable.sh` | arrays `$DATASET_ROOT/synthetic/connector_v2/grid<G>/{train,val}.npz` |
| `plot_grid_scenarios.py` | arrays `$DATASET_ROOT/synthetic/connector_v2/grid<G>/test.npz` |

Notes:

- **The generated corpus is not the paper corpus.** `make_grid_dataset.sh` is
  fixed at 10 nets × 2 pins and its board seeds are not the published ones; the
  paper's D1 is 5 nets × 2 pins (`synth_1L/grid<G>_5net_v15`). Grid geometry,
  the per-grid step penalty and every trainer knob in `cases.sh` are the
  published ones, so the scalability *protocol* is what carries over.
- The paper's split json uses the `_local` suffix — the *gitignored, personal*
  split convention ([configs/datasets/README.md](../../../configs/datasets/README.md)) —
  so it is never tracked, and only a staged dataset root has it.
- The Connector-v2 `.npz` arrays are a Jumanji-side board encoding; no encoder
  for them is part of this tree.
- The rule-based D1 baselines read
  `synthetic/synth_1L_grid<G>_5net_v02` (selected by `SYNTH1L_PCB_ROOT_<G>` —
  [methods/baselines/rule_based/_lib/datasets.py](../../../methods/baselines/rule_based/_lib/datasets.py)).

## Baselines that cannot be retrained here at all

`train_jumanji_a2c.sh` and `train_sable.sh` shell out to `run_v56_jumanji_a2c.py`
/ `run_v56_mava_sable.py` (and, for SABLE, a Mava source tree). Those runners are
not part of this repository, so these two scripts check for their entrypoint and exit
`2` with a notice that the runner is absent — including under `--dry-run`. They are kept as
the record of the published runs, not as something to launch.
The hyperparameters those runs used are recorded in `cases.sh`
(`T1_JUMANJI_*`, `T1_SABLE_*`).

This is settled, not an open defect: their jax/mava dependencies are deliberately absent
from [environment.yml](../../../environment.yml), and Fig5/6c takes those two rows from
checkpoints — so the missing runners and the `connector_v2/grid<G>/*.npz` arrays
they would read are expected here and are not tracked as issues.

## Provenance

Checkpoint and dataset provenance for the published D1 numbers:
the checkpoint/dataset provenance record, which is kept outside the public tree.
