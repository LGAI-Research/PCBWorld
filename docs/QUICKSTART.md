# Quick Start — reproducing the KDD benchmark

The paper reproduction runs from the in-repo **`var/`** tree
(`results/kdd` · `datasets` · `checkpoints/kdd`).

---

## 0. Environment

```bash
git clone --recursive https://github.com/LGAI-Research/PCBWorld.git pcbworld && cd pcbworld

# One shot: conda env -> submodules -> pinned baseline downloads -> C++ build -> smoke
bash tools/setup/setup_all.sh

conda activate cadagent
export PYTHONPATH=build_rl/pcbnew/python/rl:.
python -c 'import kicad_rl_router as krl; print("OK")'
```

- **Single env**: transformer/KiCad rollout AND the rule-based baselines
  (FreeRouting/KRT/OrthoRoute) all run in the `cadagent` env —
  [environment.yml](../environment.yml) carries openjdk 21 + rust.
- Every data/checkpoint/result root lives under the in-repo `var/`:

```bash
export DATASET_ROOT="$PWD/var/datasets"          # campaign-flat (shared)
export CKPT_ROOT="$PWD/var/checkpoints/kdd"
export EXPR_ROOT="$PWD/var/results/kdd"
```

> The defaults in `experiments/_lib/env.sh` (sourced by the dispatch scripts; also sets
> PYTHONPATH/LD_LIBRARY_PATH) point at the same in-repo `var/` roots. To use an external
> staged tree, override explicitly, e.g. `DATASET_ROOT=$KDD_BENCH_ROOT/dataset`, where
> `$KDD_BENCH_ROOT` is wherever you keep that tree — it has no default and nothing reads
> it directly; it is only a convenience name for writing the overrides below.

**Two root schemes, and how they relate.** `DATASET_ROOT` / `CKPT_ROOT` / `EXPR_ROOT` are
the **shell variables the `experiments/` dispatch scripts read** — they name the three
in-repo `var/` trees this reproduction reads and writes. `CADAGENT_DATA_ROOT` (used
throughout [README.md](../README.md) and resolved by `configs/loader/paths.py`) is a
different thing: it points at the **read-only dataset corpus** — your own copy of the
synthetic and PCBench board sets, laid out with the `sub` paths in
`configs/paths.yaml`. Python code that resolves a logical dataset name (`d2a`,
`synth_2L_v2`, …) goes through `CADAGENT_DATA_ROOT`; the shell dispatch scripts pass
explicit directories built from `DATASET_ROOT`. Point `DATASET_ROOT` at
`$CADAGENT_DATA_ROOT` when the two trees are the same copy:

```bash
export CADAGENT_DATA_ROOT="$PWD/var/datasets"   # dataset corpus (Python resolver)
export DATASET_ROOT="$CADAGENT_DATA_ROOT"       # same tree, for the dispatch scripts
```

**Checkpoints are not distributed with this repository.** A fresh clone has no `var/`
tree at all — it is created on first use by whatever writes there (generators, training,
eval). Every checkpoint path below refers to a policy you must train yourself first —
see §3.5 (`experiments/train.py`) for the recipe that produces each one.

## 1. `var/` layout

```text
var/datasets/                    # synthetic/ (synth_1L, synth_2L_v2/{train,val,test}), real_board/ (PCBench)
var/checkpoints/kdd/             # Transformer_1L_300, Transformer_2L/{default,Episodic,GRPO}, 1L_grid_rl_baselines, ...
var/results/kdd/<task>/<cell>/   # per-cell rollout + eval outputs
var/results/kdd/paper_outputs/   # paper figures/tables
var/results/kdd/legacy/          # flat-manifest inputs (aggregated_metrics/...) read by a few extractors
```

- **task** = `d1/d1_grid{10,50,100,200,500}`, `d2a`, `d3/{d3a,d3b,d3c}`. The loader also
  accepts the `t1/t2/t3*` directory names as read-aliases.
- **cell** (= atomic (task, method) unit; the directory name is the method tag):
  `transformer_pcbworld[_episodic|_grpo]`, `reward_w{0,0.001,0.002}_v{0,0.05,0.1}`,
  `freerouting`, `krt`, `ortho`, `<agent>_<backbone>` (agent ∈ interactive/plan_only/engine_free),
  `jumanji`, `sable`, `reference`. The `pcbworld`/`apiseq`/`cadgen` cell prefixes and the
  cell name `human` are accepted as read-aliases.

### Unified cell layout (flat `boards/` + `per_rollout.csv`)

Every method's rollout artifacts land **flat** under `<cell>/boards/` with a single filename
grammar (one format across all independently-sampled methods):

```text
<cell>/boards/<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb   # routed board of one rollout
<cell>/per_rollout.csv                                     # per-rollout metrics/timing (RL, freerouting)
```

- `board_id`: `board_NNNNN` for D1/D2, the actual PCBench board name for D3.
- `s<SS>` = seed (transformer/jumanji/sable = model seeds 42–45; freerouting = 20 stochastic
  runs re-bucketed as 4 seeds × 5 rollouts (`s00–03`); LLM = single `s00`), `r<RR>` =
  rollout/sample index. The rollout's `.kicad_pro`/`.kicad_prl` ship with the same stem.
- `per_rollout.csv` follows `eval/eval_utils.PER_ROLLOUT_FIELDS`; `artifact_path` links each
  row to its `boards/` file (crashed rollouts without an artifact leave it blank).
- For D3 rule-based/LLM cells the per-board shared `<board_id>.kicad_pro` (project file) is
  not per-rollout and is kept as-is.

## 2. Stage-1 rollout reproduction (producing boards)

Goal: route boards with each checkpoint/router and produce every cell's rollout boards
(DRC/aggregation are separate stages).

**Transformer (RL) — `eval/pipeline.py`** (rollout only; `--skip-drc`):

```bash
# One cell (d2a PPO — the checkpoint the §3.5 table1 recipe trains)
CUDA_VISIBLE_DEVICES=0 python -u eval/pipeline.py \
  --ckpt var/outputs/training_logs/table1_synth2l_t3a/ppo_per_step/checkpoints/policy_best.pt \
  --boards-dir "$DATASET_ROOT/synthetic/synth_2L_v2/test" \
  --seed 5600 --n-rollouts 5 --n-envs 1 --rollout-mode serial \
  --env-drc off --skip-drc --save-artifacts on \
  --output-dir "$EXPR_ROOT/t2/transformer_pcbworld"   # t2 = the d2a read-alias
```

(`experiments/train.py table1` saves under
`var/outputs/training_logs/table1_synth2l_t3a/<method>/checkpoints/`; pass
`--output-root` to the trainer to keep several seeds' checkpoints apart.)

A pipeline rollout flattens its own outputs into the §1 unified layout: the saved
artifacts are renamed to `boards/<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb` (`<cell>` =
the `--output-dir` basename, `s<SS>` = the checkpoint's training seed) and
`per_rollout.csv`'s `artifact_path` column is rewritten to match — Stages 2–3 (§3)
then consume the cell directly, with no manual step. One invocation covers one
checkpoint; a multi-seed cell (the paper's transformer cells pool ckpt seeds 42–45)
is assembled by rolling out each seed's checkpoint and merging the `boards/` files
and `per_rollout.csv` rows into one cell by hand.

**FreeRouting (rule-based)**: `methods/baselines/rule_based/run_rule_based_routers.py`
(or `eval/rollout/rule_based.py`) with `--no-eval`. These write
`<board_id>_<algo>.kicad_pcb` per seed, NOT the unified grammar — rename the routed
boards into `<cell>/boards/` under the §1 grammar before §3 (Stage 2 synthesises the
per-rollout rows from those filenames).

## 3. Evaluation + aggregation (Stages 2–3)

Score the unified cells (`boards/` + `per_rollout.csv`) from Stage 1 with **the same post-hoc
DRC (Stage 2) and standard per-board aggregation (Stage 3) for every method**. Select stages
via `--stages` of `eval/pipeline.py` (run in the `cadagent` env with the §0 roots exported).

```bash
# One cell: DRC (Stage 2) + aggregation (Stage 3). No ckpt needed (works from boards/).
CUDA_VISIBLE_DEVICES=0 python -u eval/pipeline.py \
  --stages eval,aggregate --output-dir "$EXPR_ROOT/t2/transformer_pcbworld" \
  --rollout-mode parallel --n-envs 64 \
  --selection-method posthoc_drc_aware --check-angle 45

# Re-run aggregation only (DRC already done; e.g. to compare selection modes)
python -u eval/pipeline.py --stages aggregate \
  --output-dir "$EXPR_ROOT/t2/transformer_pcbworld" --selection-method final_potential
```

- **`--stages`**: comma set of `rollout` / `eval` (=DRC) / `aggregate` (aliases `drc`, `agg`).
  Unset = all. The older `--skip-rollout/--skip-drc/--skip-aggregate` flags still work
  (`--stages` wins when both are given).
- **Parallel DRC** needs `--rollout-mode parallel --n-envs N` (default serial forces n-envs 1).
- **D1 (1-layer)** uses `--check-angle 90`, everything else `45`. Selection:
  `final_potential` (default) or `posthoc_drc_aware` (DRC-aware).
- Boards-only cells (LLM/rule-based, no `per_rollout.csv`) get their rows synthesised from
  `boards/` filenames in Stage 2 — same scoring path.

### Stage 2 — post-hoc DRC
Scores each completed rollout's `boards/*.kicad_pcb` with the KiCad evaluator
(`eval.metrics.evaluate_one`) and merges the DRC columns
(routability/drv_errors_only/clean_pass/…) into `per_rollout.csv`. RL, freerouting,
rule-based and LLM all share this path (`eval_kicad_pcb` in `eval/pipeline.py`).

### Stage 3 — standard per-board aggregation (`eval.aggregation.aggregate_boards`)
Reads the cell's `per_rollout.csv` and writes three files. Also runnable standalone:
`python eval/aggregation.py --cell <cell> [--selection-mode final_potential|posthoc_drc_aware]`

```text
<cell>/per_boards_ckpts.csv     # board × ckpt: <m>_avg / <m>_best / time_avg / time_best
<cell>/per_boards_overall.csv   # board: <m>_avg/_avg_std (all runs), <m>_best/_best_std (per-ckpt best), time
<cell>/per_boards_summary.csv   # one model row: board-averaged summary
```

- **ckpt** = the `s<SS>` in `boards/` filenames (distinct from the csv `seed` column, which is
  the rollout RNG seed).
- `<m>_avg` = mean over rollouts, `<m>_best` = the `selection-method` winner rollout's value.
  Metric vocabulary = `eval_utils.CADAGENT_VALUE_METRIC_FIELDS` (avg/best reuse the same
  `_per_board_summary` as the inline path).
- overall: `_avg` = mean over all runs / `_avg_std` = run sample std; `_best` = mean of
  per-ckpt bests / `_best_std` = their sample std (blank when <2 samples).
- time: `_avg` = **mean** rollout time, `_best` = **sum** of rollout times.

> **Paper-table reporting is a separate step** from the aggregation above:
> Table 1's passed-board-only (success) means and Table 2's P@k/CP@k live in
> `experiments/_lib/metrics/score_rollouts.py`, which reads the §3 outputs.

## 3.5 Training · figures/tables — `experiments/train.py` · `draw_figure.py`

`experiments/` is a single layer. Each paper artifact folder carries its own light shell
driver: `figure5_d1`, `figure6_reward`, `table1_rl`, `table1_llm`, `appendix_diagnostics`
and `d3_dataset` each have a `run.sh`; `table2` is split by LLM agent mode into
`run_interactive.sh`, `run_plan_only.sh`, `run_engine_free.sh` plus `eval.sh`. The two
shared entrypoints are below — full map in
[experiments/README.md](../experiments/README.md).

**Training — `train.py`** (single-run router: method → one trainer shell exec; grid×seed
loops live in each `run.sh`):

```bash
python experiments/train.py table1 --method ppo_per_step|ppo_terminal|grpo --seed 42   # → kdd/table1_rl/train_policy.sh
python experiments/train.py reward --wirelength-penalty 0.002 --via-penalty 0.1 --seed 42
python experiments/train.py d1-ppo --grid-size 100 --seed 42
# or run a folder's full sweep: bash experiments/kdd/figure5_d1/run.sh train transformer
```

- `table1` / `reward` train on the [d2a split](../configs/datasets/d2a.json) (10 000
  `synth_2L_v2/train` boards). For a locally generated set, pass your own split file with
  `--split-json <file>` — a board listed there but missing on disk is warned about and
  skipped, so a partial set trains on what exists instead of crashing mid-run.

> **D1 (Figure 5) is not reproducible from this repository.** Its corpus is not
> distributed here and the `d1-jumanji`/`d1-sable` runners are not part of this
> tree, so the `kdd/figure5_d1/` train/eval entrypoints exit 2 with a notice naming
> what they need. `run.sh figure` is the exception: it runs (exit 0) and renders the
> figure with `(absent/OOM)` placeholders —
> see [experiments/kdd/figure5_d1/README.md](../experiments/kdd/figure5_d1/README.md).

**Figures/tables — `draw_figure.py`** (read-only; writes to `var/results/kdd/paper_outputs/`):

```bash
python experiments/draw_figure.py --figure fig6c|table3|table22|table23|fig8|fig9|table24_25|all
```

> `eval` has no separate entrypoint — use `eval/pipeline.py` as in §2–§3. The D1 held-out
> eval is `bash experiments/kdd/figure5_d1/run.sh eval` (internally
> `eval/pipeline.py --check-angle 90`).

## 4. Per-experiment map (summary)

| Paper item | task | main cells | folder · figure |
| --- | --- | --- | --- |
| Figure 5 / Fig 6c (D1 grid scalability) — corpus not distributed | `d1/d1_grid{10,50,100,200,500}` | `transformer_pcbworld`, `jumanji`, `sable` | `kdd/figure5_d1/` · `--figure fig6c` |
| Table 1 RL rows (D2/D3 quality) | `d2a`, `d3/{d3a,d3b}` | `transformer_pcbworld{,_episodic,_grpo}` | `kdd/table1_rl/` · `--figure table3\|table22\|table24_25` |
| Table 1(b)/Table 2 LLM rows | `d2a`, `d3/d3a` | `interactive_*`, `plan_only_*`, `engine_free_*` (legacy disk names read-aliased) | `kdd/table1_llm/`, `kdd/table2/` · `--figure fig9` |
| Rule-based baselines | `d2a`, `d3/*` | `freerouting`, `krt`, `ortho` | `methods/baselines/rule_based/` |
| Figure 6 (reward ablation) | `d2a` | `reward_w{0,0.001,0.002}_v{0,0.05,0.1}` | `kdd/figure6_reward/` · `--figure fig8` |

Rollout for each item follows §2, training/figure generation §3.5, evaluation/aggregation §3.
Per-folder entrypoints: [experiments/README.md](../experiments/README.md). Dataset/checkpoint
paths and generators: the §1 tree and `tools/datagen/synthetic_generator/README.md`.
