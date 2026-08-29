# `experiments/_lib/metrics/` — paper figure/table extraction

Read-only extractors that turn the per-cell **common aggregation**
(`per_rollout.csv` under `var/results/kdd/`) into the paper's figures and
tables. Outputs (CSV + Markdown + PDF/PNG) go to `var/results/kdd/paper_outputs/`.
**Nothing under the results tree is written outside `paper_outputs/`** (enforced
by `common.open_ro` / `common.assert_output_path`).

Each extractor is a module exposing `main(argv=None)`. Invoke them through the
single dispatcher `experiments/draw_figure.py` (don't run the modules directly).

## Run

```bash
conda activate cadagent          # for the eval.eval_utils import
cd <repo>
python experiments/draw_figure.py --figure fig6c
python experiments/draw_figure.py --figure table3
python experiments/draw_figure.py --figure table22
python experiments/draw_figure.py --figure table23
python experiments/draw_figure.py --figure fig8
python experiments/draw_figure.py --figure fig9
python experiments/draw_figure.py --figure table24_25
python experiments/draw_figure.py --figure all      # regenerate everything
```

Each is idempotent (pure read → rewrite of its own `paper_outputs/` files) and
safe to re-run as the dispatch fills more cells. Cells whose DRC stage is still
running print a `[warn] … only N/M populated` line; their numbers are
provisional until the dispatch completes.

| script | paper artifact |
|---|---|
| `fig6c_d1_cleanpass.py` | Fig 6c — D1 clean-pass@5 vs grid |
| `table3.py` | Table 3 — clean_pass / potential / routability / time (D2, D3-A, **D3-B**) |
| `table22.py` | Table 22 — Table 3 + DRV/WL/Via, all `mean ± std` |
| `table23.py` | Table 23 — D1 grid sweep `mean ± std` |
| `fig8_reward_sweep.py` | Fig 8 — reward sweep (WL/Via marginals, zoomed y-axes + seed-std error bars; no Freerouting ref) |
| `fig9_openloop.py` | Fig 9 — interactive vs plan-only vs engine-free (CP@5/Pot/Rout/DRV) |
| `table24_25.py` | Tables 24/25 — PPO (per-step)/GRPO/PPO (terminal) quality + timing (+**D3-B**) |

## Conventions (locked with the user)

- **best = `final_potential` winner** per `(board, seed)`. The reward potential
  is built so a clean rollout always wins, hence `winner.clean_pass` == CP@5.
  Every reported metric (incl. WL/Via/DRV) is read off that single winner;
  WL/Via/DRV are averaged over **all** boards (no routed-only filter).
- **`±`** = sample std (ddof=1) across seeds; `†` marks single-seed cells.
- **DRV** column = `drv_errors_only_count` (errors-only, same bucket as
  `clean_pass`). `common.METRIC_COL` also exposes `drv_errors_promoted` and
  `total_drv` if a stricter count is ever wanted.
- **Parse-fail** (Fig 9): generations whose `.kicad_pcb` failed to parse/evaluate
  (`eval_status == 'error'`); surfaced as its own panel + count rows since these
  rollouts have blank metrics and are otherwise dropped (`common.parse_fail_stats`).
- **`clean_pass`** is the engine column (legacy CSVs call it `clean_success`;
  the loader aliases it), whose current definition is
  `routed && drv_errors_only == 0` (errors-only). This is why a row can show
  `clean_pass = 1.00` while the `DRV` (errors+promoted) column is > 0. If the
  paper wants the stricter Appendix-I.1 definition (`total_drv_count == 0`),
  switch `METRIC_COL["clean_pass"]` logic in `common.py` — it is the single knob.
- **Potential is reported as `potential_gain` = final − initial**, where
  `initial_potential` is Φ of the bare board (raw Φ is often negative). This is
  now computed **in the per_rollout pipeline itself**, no side script:
  the gym env records `initial_potential` at reset and exposes
  `initial_potential`/`potential_gain` in the terminal `info`
  (`pcb_world/core/env.py`), the rollout writes them to the row
  (`eval/rollout/rl.py`), and the post-hoc DRC path recomputes them in
  `eval.metrics.compute_metrics` (overriding only when it actually reset
  the engine; it leaves the keys untouched otherwise so it never
  clobbers the env value). So any fresh rollout / `--stages eval` run populates
  the native `initial_potential`/`potential_gain` columns.
  *Transitional only:* `compute_initial_potential.py` backfills a per-board cache
  (`paper_outputs/initial_potential.json`) for cells that predate this change;
  `common.reduce_cell` uses it **only** when the native `potential_gain` column is
  absent. Once the dispatch is re-run, the cache and that script are unnecessary.
- **time**: RL/rule-based cells use the measured `per_board_rollout_time` (RL is
  the serial seed42 pass; Freerouting from its router wrapper). LLM cells
  (PCBWorld/API/Code) are boards-only with no `per_board_rollout_time`, so
  `common.reduce_cell` falls back to the model-level `sec/rollout` (= paper
  sec/ep) from `var/results/kdd/legacy/aggregated_metrics/llm_eval_metrics.csv`,
  mapped by (interface, model, namespace) with `-think-off` stripped
  (`common.llm_time_for`). That csv only covers D2 (`synth2L`) and D3-A
  (`PCBench`), so **LLM D3-B time stays `—`**. KRT/OrthoRoute are also boards-only
  → time `—` (their runner logs aren't wired in).

## LLM routing time — `llm_time_pattern.py`

LLM cells store only DRC time in `per_rollout.csv`; the real per-episode latency
lives in the original agentic rollout logs. This module is the **reusable
back-fill pattern** for a *separate* agent: it maps `(board_id) → seconds` from
an LLM-rollout root and (default) prints a **dry-run** plan. It only writes
`per_board_rollout_time` when explicitly run with `--write` (loud banner) — the
table scripts never call it.

```bash
# dry-run (read-only): show what would be filled
python experiments/_lib/metrics/llm_time_pattern.py \
    --cell d2a/pcbworld_gpt-5.4 --llm-root <ORIGINAL_LLM_ROLLOUT_ROOT>
```

The back-fill agent should confirm the actual time field name (the module tries
`per_board_rollout_time`, `routing_time_sec`, `wall_time_sec`, `sec_per_ep`, …)
and the root layout (`**/per_board/<id>/aggregate.json` or `**/summary.csv`)
before `--write`.
