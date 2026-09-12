# Experiments — paper reproduction recipes

Recipes nest per campaign (`experiments/<campaign>/`). The campaign-agnostic
routers (`train.py` · `draw_figure.py`) and the shared scaffolding `_lib/` sit at
the top level; deliverables live under their campaign folder. The current
campaign is `kdd/` (paper reproduction). Future campaigns join by reusing
`_lib/` + the kdd recipes and share the same output-tree prefixes
(`var/results/<campaign>/` · `var/checkpoints/<campaign>/`).
The overall reproduction flow: [docs/QUICKSTART.md](../docs/QUICKSTART.md).

## Layout

```
experiments/
  train.py            # campaign-agnostic router: <experiment> → one exec of a kdd/<deliverable>/ trainer shell
  draw_figure.py      # figure/table dispatcher: --figure fig6c|table3|table22|table23|fig8|fig9|table24_25|all
  _lib/               # shared across campaigns: env.sh, llm_lib.sh, metrics/ (extractor modules)
  kdd/                # ── KDD campaign ──
    figure5_d1/         # Fig5 / Fig6c — D1 grid scalability
    figure6_reward/     # Fig6 / Fig8 — reward ablation
    table1_rl/          # Tables 3/22/24/25 — RL policy quality (D2/D3)
    table1_llm/         # Table1(b)/Fig9 — LLM PCBWorld agent
    table2/             # LLM plan_only/engine_free/interactive levels + plot_gpt_levels.py
    llm_eval/           # LLM eval implementation (plan_only_v8 / engine_free_v3 standalone, fewshot prep, P@K aggregator)
    d3_dataset/         # D3 real-board split JSON build (data-prep)
    appendix_diagnostics/  # appendix training curves
```

- **Eval reuses the top-level [eval/pipeline.py](../eval/pipeline.py)** (3-stage
  rollout/eval/aggregate) — `experiments/` carries no separate eval.py.
- Each experiment folder is a light `run.sh` (train/eval/figure subcommands) +
  `train_*.sh` (trainers) + `cases.sh` (case registry).
- All path defaults are in-repo `var/` (datasets/checkpoints/results); external
  dataset stores are reached only via env-var overrides (`_lib/env.sh`).

## Entrypoints per paper item

| item | train | eval | figure |
| --- | --- | --- | --- |
| Fig5 / Fig6c (D1 grid) — **corpus not distributed**, see [kdd/figure5_d1/README.md](kdd/figure5_d1/README.md) | `kdd/figure5_d1/run.sh train [transformer\|jumanji\|sable]` (or `train.py d1-ppo`) | `kdd/figure5_d1/run.sh eval` (eval/pipeline.py, `--check-angle 90`) | `draw_figure.py --figure fig6c` |
| Fig6 / Fig8 (reward) | `kdd/figure6_reward/run.sh train` (or `train.py reward`) | shared cell path | `draw_figure.py --figure fig8` |
| Tables 3/22/24/25 (RL) | `kdd/table1_rl/run.sh train` (or `train.py table1 --method …`) | `eval/pipeline.py` (rollout → `--stages eval,aggregate --check-angle 45`) | `draw_figure.py --figure table3\|table22\|table24_25` |
| Table1(b)/Fig9 (LLM) | — | `kdd/table1_llm/run.sh`, `kdd/table1_llm/baselines/*.sh` | `draw_figure.py --figure fig9` |
| Table 2 (LLM levels) | — | `kdd/table2/run_{plan_only,engine_free,interactive}.sh` → `kdd/table2/eval.sh` | — |
| Appendix training curves | — | — | `kdd/appendix_diagnostics/run.sh` |
| D3 split (data-prep) | `kdd/d3_dataset/run.sh` | — | — |

Common flags: `DRY_RUN=1` (print commands only) · `SMOKE=1` (1-iter smoke) ·
`GPU=N`, plus the `L1_GRIDS`/`SEEDS`/`WIRES`/`VIAS`/`METHODS` overrides.
`DRY_RUN=1` does not bypass a missing prerequisite: the D1 scripts check their
inputs first and exit 2 either way, rather than printing a command that could
not run.

## Manual helper tools (not wired into run.sh — run directly)

- [kdd/table1_rl/render_best_only_visualizations.py](kdd/table1_rl/render_best_only_visualizations.py) —
  picks each group's best final board from `rollouts.csv` and renders SVGs via
  kicad-cli (`PCBRenderer`).
  `python experiments/kdd/table1_rl/render_best_only_visualizations.py --rollouts-csv <csv> --output-dir <dir>`
- [kdd/table2/plot_gpt_levels.py](kdd/table2/plot_gpt_levels.py) — GPT-family ×
  eval-level bar chart.
  `python experiments/kdd/table2/plot_gpt_levels.py --metrics-csv <csv>`

## D1 (Fig5 / Fig6c) is not reproducible here

The D1 corpus is not distributed with this repository and no generator here
reproduces it, so every script under `kdd/figure5_d1/` refuses up front with a
notice naming the paths it wanted. On top of that,
`kdd/figure5_d1/train_jumanji_a2c.sh` · `train_sable.sh` invoke external runner
scripts (run_v56_jumanji_a2c / run_v56_mava_sable + their connector), which are
not part of this tree — so retraining those two baselines is not supported here
under any conditions. The jumanji·sable **numbers in Fig5/6c come from
checkpoints**, which the eval path covers without those runners.
Full breakdown: [kdd/figure5_d1/README.md](kdd/figure5_d1/README.md).
