# Reproducing the KDD 2026 workshop paper

One index for the paper's experiments. Every folder here is a self-contained recipe (`run.sh`,
`cases.sh`, trainer shells); this page only says which folder produces which table or figure,
what it needs, and where the paper's configuration lives. Mechanics of the shared pipeline
(rollout → post-hoc DRC → aggregate, the `var/` layout) are in [docs/QUICKSTART.md](../../docs/QUICKSTART.md).

## Prerequisites

1. The conda env and a built engine — root README, Quick start *Setup*.
2. `PCBWORLD_DATA_ROOT` set; the synthetic 2-layer boards (`tools/quickstart/train_rl.sh` generates them; `--paper`
   for the paper-scale set) and the D3 real boards (`tools/quickstart/prepare_pcbench.sh`, the PCBench rebuild chain). Every `run.sh` sources
   `experiments/_lib/env.sh`, which resolves dataset and output paths through `configs/paths.yaml`.
3. LLM rows need an API key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`, or an
   OpenAI-compatible endpoint via `OPENAI_BASE_URL`); model aliases are in
   [configs/quickstart/models.json](configs/quickstart/models.json).

## Paper item → recipe

| Paper item | Folder | Command | Output |
|---|---|---|---|
| Table 1 — RL rows (PPO per-step, PPO terminal, GRPO) | [table1_rl/](table1_rl/) | `bash experiments/kdd/table1_rl/run.sh train` (method × seed sweep, cells in `cases.sh`); `run.sh eval` (per task × method: each seed's rollout pooled into one cell under `var/results/kdd/`, then post-hoc DRC + aggregation; the cells' settings: [configs/paper_train_flags.sh](configs/paper_train_flags.sh) `PAPER_ROLLOUT_FLAGS`/`PAPER_SCORING_FLAGS` + `cases.sh` `TABLE1_PAPER_*`); `run.sh figure` | checkpoints, `per_rollout.csv` cells, Table 3 / 22 / 24–25 |
| Table 1 — LLM rows | [table1_llm/](table1_llm/) | `bash experiments/kdd/table1_llm/run.sh --model <alias> --split d2a --out <dir>` (and `d3a`) | LLM rollout cells for the same aggregation |
| Table 2 — LLM interaction levels (interactive / plan-only / engine-free) | [table2/](table2/) | `run_interactive.sh` · `run_plan_only.sh --model <alias> --split d2a` · `run_engine_free.sh --model <alias> --split d2a` (each also with `--split d3a`), then `eval.sh`; `plot_gpt_levels.py` | per-level cells, Table 2 plot |
| Figure 5 — D1 grid scalability (and Figure 6c) | [figure5_d1/](figure5_d1/) | `bash experiments/kdd/figure5_d1/run.sh <stage>`, `<stage>` = `train` / `eval` / `figure` | **the D1 corpus is not distributed** — see that folder's README |
| Figure 6 / Figure 8 — dense reward ablation | [figure6_reward/](figure6_reward/) | `bash experiments/kdd/figure6_reward/run.sh <stage>`, `<stage>` = `train` (wire × via × seed sweep) / `plot` / `figure` | reward-sweep figure from `per_rollout.csv` |
| Appendix — validation curves | [appendix_diagnostics/](appendix_diagnostics/) | `bash experiments/kdd/appendix_diagnostics/run.sh` (`FETCH_WANDB=1` to re-pull) | appendix figures |
| D3 split file | [d3_dataset/](d3_dataset/) | `bash experiments/kdd/d3_dataset/run.sh` | rebuilds `configs/datasets/d3.json` from the corpus |

**D3-A boards.** Every recipe here runs the 99-board D3-A, `easy.test` of
[configs/datasets/d3.json](../../configs/datasets/d3.json) ([configs/datasets/README.md](../../configs/datasets/README.md)) —
e.g. `table2/run_engine_free.sh --split d3a` routes those 99. The DRV of every table row (RL, rule-based, LLM) counts DRC
errors only.

Two front doors sit above the folders: `python experiments/train.py table1|reward|d1-ppo …`
launches the matching trainer shell, and `python experiments/draw_figure.py --figure
{fig6c,table3,table22,table23,fig8,fig9,table24_25,all}` draws a table or figure from the
cells' CSVs.

## The paper's configuration is pinned here

[configs/](configs/) holds exactly what the paper ran with — its reward and masking rules, the
synthetic-dataset DRC rules, the dataset splits (`d2a.json`, the grid families) and the
checkpoint/split maps under `quickstart/`. The recipes name these explicitly — the rule files by name, and the training knobs whose
shipped defaults moved on after the paper (observation tokens, outline representation, directional
candidates, time feature, the connectivity candidate filter, the engine's legacy `make_line` (`reject_if_stuck` false),
PPO truncation bootstrap) through [configs/paper_train_flags.sh](configs/paper_train_flags.sh) —
so they do not depend on the repository's shipped defaults (`configs/env.yaml`, `configs/rl.yaml`, `configs/llm.yaml` and the `pcbworld_*` rules),
which are newer and differ. In particular the paper's RL cells were scored with the plain `drc_dense_promoted`
(errors + the three promoted warnings, no wirelength/via term), every recipe's `eval` stage passes those settings
([configs/paper_train_flags.sh](configs/paper_train_flags.sh) `PAPER_ROLLOUT_FLAGS`/`PAPER_SCORING_FLAGS`), while the shipped
evaluation default is the shipped training reward `pcbworld_reward`.
Checkpoints from the paper era carry no value for knobs added later; the checkpoint loader fills those
with their pre-knob values from [configs/ckpt_migrations.yaml](../../configs/ckpt_migrations.yaml).

Which config file each experiment used, row by row: [configs/quickstart/README.md](configs/quickstart/README.md).
