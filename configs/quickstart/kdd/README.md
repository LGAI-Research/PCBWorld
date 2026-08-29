# configs/quickstart — KDD reproduction config index

The two JSON files in this folder are the LLM-eval alias tables that
[experiments/_lib/llm_lib.sh](../../../experiments/_lib/llm_lib.sh) reads mechanically:

- [models.json](models.json) — model alias → `api_provider`/`api_model` (`qs_resolve_model`)
- [splits.json](splits.json) — task alias (d2a·d3a·d3b·d3c; legacy t2·t3* accepted) → boards_json/difficulty/split (`qs_resolve_split`)

The tables below are a **lookup index of the config files actually used by each KDD experiment**. The single
source of truth for the values is each experiment's `cases.sh` / defaults YAML (last column); this table only
tells you where to go.

## Training (RL)

| Experiment (paper) | reward | masking | DRC config | boards | source of truth |
|---|---|---|---|---|---|
| Figure 5 — D1 grid scalability | [jumanji_connector_wirelength_dense](../../reward/jumanji_connector_wirelength_dense.yaml) | [default_no_via](../../masking/default_no_via.yaml) | – (native) | `synth_1L/grid{G}_5net_v15` (call-site path, [run.sh](../../../experiments/kdd/figure5_d1/run.sh)) — **corpus not distributed**, see [figure5_d1/README.md](../../../experiments/kdd/figure5_d1/README.md) | [figure5_d1/cases.sh](../../../experiments/kdd/figure5_d1/cases.sh) |
| Figure 6 — dense reward ablation | [drc_dense_promoted](../../reward/drc_dense_promoted.yaml) | [default](../../masking/default.yaml) | [drc/synth_2L_v2](../../drc/synth_2L_v2.yaml) | [datasets/d2a.json](../../datasets/d2a.json) | [figure6_reward/cases.sh](../../../experiments/kdd/figure6_reward/cases.sh) |
| Table 1 — PPO per-step | [drc_dense_promoted](../../reward/drc_dense_promoted.yaml) | [default](../../masking/default.yaml) | [drc/synth_2L_v2](../../drc/synth_2L_v2.yaml) | [datasets/d2a.json](../../datasets/d2a.json) | [table1_rl/cases.sh](../../../experiments/kdd/table1_rl/cases.sh) |
| Table 1 — PPO terminal | [drc_sparse_promoted_ppo](../../reward/drc_sparse_promoted_ppo.yaml) | [default](../../masking/default.yaml) | same | same | same |
| Table 1 — GRPO | [drc_sparse_promoted_grpo](../../reward/drc_sparse_promoted_grpo.yaml) | [default](../../masking/default.yaml) | same | same | same |

## Evaluation (shared 3-stage) · LLM

| Pipeline | reward / masking | boards | source of truth |
|---|---|---|---|
| RL eval ([eval/pipeline.py](../../../eval/pipeline.py)) | DRC scoring reward = [drc_dense_errors_only_eval](../../reward/drc_dense_errors_only_eval.yaml); env/masking are inherited from the ckpt. The rules come from each board's native `.kicad_pro` | per experiment (table above) | [defaults/rl_eval.yaml](../../defaults/rl_eval.yaml) |
| LLM eval (Table 1 LLM rows) | reward = [grpo_final](../../reward/grpo_final.yaml), masking = `strict` (compatibility-resolved to [default](../../masking/default.yaml)) | [splits.json](splits.json) alias: d2a → [d2a.json](../../datasets/d2a.json), d3a/b/c → [d3.json](../../datasets/d3.json) | [defaults/llm.yaml](../../defaults/llm.yaml) · [table1_llm/run.sh](../../../experiments/kdd/table1_llm/run.sh) |

> **Note**: the `reward_rule: drc_only_dense` / `masking_rule: default_no_finish` in
> [configs/defaults/env.yaml](../../defaults/env.yaml) are **not the KDD settings** — they are development
> defaults. KDD reproduction must always go through each experiment's `run.sh` (→ `cases.sh` overrides).
> Full reproduction procedure: [docs/QUICKSTART.md](../../../docs/QUICKSTART.md).
