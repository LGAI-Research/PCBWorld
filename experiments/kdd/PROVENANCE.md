# KDD result provenance — checkpoints · datasets · metrics conventions · incident record

Provenance needed to reproduce and interpret the paper (KDD/PCBWorld) results that cannot be
recovered from the code/tree alone. This file is the authoritative record.

## 1. Checkpoint provenance (on-disk record = `var/checkpoints/kdd/*/SOURCE.txt`)

- **d1 (1L)**: authoritative = **`policy_iter_300.pt`** from the 1L grid-scan training run
  (2026-05-22) (→ `var/checkpoints/kdd/Transformer_1L_300`). **Do not use `policy_best.pt`** — its best
  sits at ~iter25, far too early. The original iter10..500 checkpoints + optimizer state (17G) are kept
  in that run's checkpoint directory, outside this repo.
- **d2a PPO default seed42**: `default/seed42/policy_best.pt` is the file replaced on 2026-05-30 with the
  true best (iter169, reward 16.30) of the figure6 reward-ablation (wire0.002, via0.1) seed42 cell.
  Rollouts predating that replacement (`eval_260528`) are STALE.
- **Authoritative 36 cells of the reward ablation (fig8)** = the
  `figure6_reward_ablation/training/checkpoints/` tree of the archived training outputs
  (MANIFEST.csv, sha256-verified).

## 2. Authoritative datasets

- **d1** = `synth_1L/grid{10,50,100,200,500}_5net_v15/` — **5 nets × 2 pins** (not the generator's 10-net
  default; to regenerate, `NETS=5 PINS=2 generate_grid_dataset.sh <G>`, generator =
  [tools/datagen/synthetic_generator/](../../tools/datagen/synthetic_generator/)). train 10000 / val 128 / test 128.
- **d2a** = `synth_2L_v2` — generator `generate_2layer_v2.sh` (nets U{4,5,6}, pads/net {2:.6, 3:.2, 4:.1, 5:.1}).
  ⚠ The HEAD generator drifted in geometry on 2026-05-28 → reverted to the pre-05/28
  behaviour; the golden regression test
  [tests/test_synthetic_dataset_reproduction.py](../../tests/test_synthetic_dataset_reproduction.py) guards it.
  **Reproduce from the staged copies only**: local `var/datasets/synthetic/synth_2L_v2{,_test_128}`,
  authoritative copy `$CADAGENT_DATA_ROOT/synthetic/*`.
  split = [configs/datasets/d2a.json](../../configs/datasets/d2a.json).
- **d3 (real)** = easy/medium/hard in [configs/datasets/d3.json](../../configs/datasets/d3.json) =
  d3a(99)/d3b(10)/d3c(10); board lists `var/outputs/rollouts_t3_2l/_boards_lists/t3{a,b,c}.txt`.
  Registry = [configs/datasets/README.md](../../configs/datasets/README.md)
  (⚠ includes the d3b train ⊃ test leakage gotcha).
- **LLM coverage gap**: all 7 LLMs generated **d3a only** → the d3b/d3c LLM cells score 0 clean because the
  inputs are absent (expected).

## 3. Official bench source (ground-truth routed boards + native rule `.kicad_pro` files)

`$CADAGENT_DATA_ROOT/KDD_benchmark/bench_results/bench_results_260501/bench_results_official_kicadpcb/`

- Mapping: `PCBench`=d3a, `PCBench_medium`=d3b, `synthetic_2L_v2`=d2a, `synthetic_1L_grid10_5net_v15`=d1;
  `Transformer_PPO`→transformer_pcbworld, `gpt-…/qwen…`→pcbworld_<m>; apiseq/cadgen (= plan_only/engine_free)
  are absent from the official routed tree.
- Convention: **one native rule set shared per board**. On 38/100 d3a boards the Transformer `.kicad_pro`
  differs only in net-class (GND) membership (the min_* thresholds are identical).
- zero-shot outputs =
  `$CADAGENT_DATA_ROOT/cadagent_baseline_summaries/{cadgen,apiseq}_{real,synth}_zs_<model>/per_board/`
  (external directories keep the old names).

## 4. Metrics extraction conventions ([experiments/_lib/metrics/](../_lib/metrics/), user-confirmed 2026-05-31)

- **best = the final_potential winner** per (board, seed); every metric is read from that winner. WL/Via/DRV
  are averaged over all boards. **Exception — fig8**: a plain mean over all rollouts + SEM (do not "fix" this
  to the winner convention). `±` = std across seeds (`†` = single seed).
- The DRV column = `drv_errors_only_count` (errors only). Potential is reported as the **gain**
  (final − initial), with the native column preferred (the `compute_initial_potential.py` cache is
  transitional). d3a = **99 boards** (198 is an artifact of the cache union).
- **Time sources**: RL/Freerouting = measured `per_board_rollout_time` (for RL, only the serial seed42 run is
  attributed; parallel = NaN); LLM = the sec/rollout column of
  `var/results/kdd/legacy/aggregated_metrics/llm_eval_metrics.csv` (`common.llm_time_for`; there is no LLM d3b
  experiment); KRT/Ortho = `common.RULE_BASED_TIME_PAPER` (krt d3b=3.265 is measured from the 5/28 run,
  ortho d3b=—).
- On-disk cell paths keep the legacy `t1/t2/t3` names; the code aliases d→t
  ([experiments/_lib/metrics/common.py](../_lib/metrics/common.py) `_LEGACY_TASK_DIRS`).

## 5. Two resolved incidents (recurrence-prevention summary)

- **Design-rule mismatch (resolved 2026-05-31 – 06-01)**: the LLM cells' `.kicad_pro` files were loose
  TEMPLATE rules and were bulk-replaced with the native ones (backup
  `var/results/kdd/legacy/llm_template_pros_backup_260531.tar.gz`). The swap is geometry-blind, so **only d2a
  apiseq (plan_only) was re-rolled out** (native; reproduction tool =
  [methods/llm_agent/rollout/plan_only.py](../../methods/llm_agent/rollout/plan_only.py); the old template
  boards are at `<cell>/_boards_template_bak/`). The remaining LLM cells (cadgen d2a, pcbworld d2a, d3a) keep
  their originals and stay geometry↔rule consistent — no re-rollout needed. Root cause: the old apiseq script
  hard-coded `use_yaml_drc_fallback=True`.
- **potential_gain DRC asymmetry (resolved 2026-06-29)**: only the initial snapshot ran with run_drc=False, so
  the gain was under-reported (a per-board constant bias — ranking, selection and training unaffected, only the
  absolute value). Fixed, with the regression test
  [tests/test_eval_initial_potential_drc.py](../../tests/test_eval_initial_potential_drc.py). Absolute gain
  values in CSVs written before the fix carry this bias.

## 6. Old names must not come back (S1–S6 rename, completed 2026-07-03)

`ppo_dense`→`ppo_per_step` · `sparse/dense` mode→`terminal/per_step` · `clean_success`→`clean_pass` ·
t-series→d-series (t2→d2a etc.) · apiseq/cadgen/pcbworld→plan_only/engine_free/interactive.
Path-rename registry = `DEAD_PATHS` in check_docs.py (symbol renames are not registered — this list is the
reference).
Exception: **on-disk paths, legacy CSVs and the external dataset/result directories keep the old names**
(the code resolves them through aliases).
