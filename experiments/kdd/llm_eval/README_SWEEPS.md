# LLM-baseline eval — code map

Open-loop (one-shot) LLM eval for KiCad PCB routing. The LLM gets one completion per
board — no env stepping, no intermediate state. Output is patched / replayed
through the same scoring backend the RL pipeline uses, so metrics compare
apples-to-apples.

Two task framings live in this directory:

| task     | LLM emits                                  | entry point                    |
|----------|--------------------------------------------|---------------------------------|
| Engine-free (CAD-Gen)  | raw `(segment ...)` / `(via ...)` lines    | `eval_engine_free_llm_v3_standalone.py` |
| Plan-only (API-Seq)  | `<actions>net_select 1\nstart_route ...</actions>` | `eval_plan_only_llm_v8_standalone.py` |

CAD-Gen patches its routing into the unrouted board and scores the result
via `eval.metrics.evaluate_one`. API-Seq parses one action per line and
replays them through a fresh `PCBWorld`, then scores the final state.

## Versioned prompts

Each task has a prompt iteration chain — `v2`, `v3`, … layered on top of v1
via monkey-patch (`vN.main()` patches `vN-1._SYSTEM_PROMPT`, forwards). The
**finals**:

- **CAD-Gen v3** — octilinear prompt + 45° audit + `--strict-angle`
  (`success_strict = success AND every segment within 0.5° of {0,45,90,135}°`).
- **API-Seq v8** — per-topology emission rules (Case A/B/C/D), explicit
  layer-choice rule for `th` pads, `make_via` → `start_route` restart pattern
  for cross-layer nets, anti-pattern list.

Standalone equivalents (no v1/v2/v3 imports — single self-contained file):
`experiments/kdd/llm_eval/eval_engine_free_llm_v3_standalone.py`, `experiments/kdd/llm_eval/eval_plan_only_llm_v8_standalone.py`.

## Few-shot pools

Few-shot mode (`--mode few_shot --fewshot-pool DIR`) needs a pre-built pool:

```bash
python experiments/kdd/llm_eval/prepare_synth_fewshot.py        # PNS engine-routed synth pool (engine-free)
python experiments/kdd/llm_eval/prepare_plan_only_fewshot.py    # deterministic auto-router action pool
python experiments/kdd/llm_eval/prepare_plan_only_fewshot_llm.py # LLM-generated routability=1.0 action pool
```

## Provider sweep — Qwen on Together

`experiments/kdd/table1_llm/baselines/run_qwen_together_sweep.sh` wraps both eval scripts and sweeps the
Qwen3 chat family (+ Coder-480B) against Together's serverless endpoint.

```bash
export TOGETHER_API_KEY=...
bash experiments/kdd/table1_llm/baselines/run_qwen_together_sweep.sh dry       # smoke
bash experiments/kdd/table1_llm/baselines/run_qwen_together_sweep.sh main      # 8 models × engine-free+plan-only × synth (128 boards)
python experiments/kdd/llm_eval/aggregate_qwen_together_sweep.py \
    eval_out/qwen_together_sweep/<DATE_TAG>
```

Default sample budget = 25 per board; ks reported = `{1, 5, 10, 25}`. The
aggregator produces a `comparison.csv` + a model × k pivot in `comparison.md`,
cells = `pass@k_unb / rb_best / rb_mean`.

## Direct invocation (single model)

```bash
# CAD-Gen v3 standalone, synth zero-shot, OpenAI
OPENAI_API_KEY=... bash experiments/kdd/table1_llm/baselines/run_engine_free_llm_v3_standalone.sh synth_zs

# API-Seq v8 standalone, synth few-shot, Together / Qwen3-8B
TOGETHER_API_KEY=... API_PROVIDER=together API_MODEL=Qwen/Qwen3-8B \
    bash experiments/kdd/table1_llm/baselines/run_plan_only_llm_v8_standalone.sh synth_fs

# Strict 45° aggregation on engine-free v3
STRICT_ANGLE=1 bash experiments/kdd/table1_llm/baselines/run_engine_free_llm_v3_standalone.sh synth_zs
```

## Outputs

Each per-(set, mode) run writes:
- `per_board/<board_id>/sample_NN.{kicad_pcb,kicad_pro,json,response.txt}`
- `per_board/<board_id>/aggregate.json`
- `summary.csv`, `overall.json`
- `overall_multi_k.json` + `summary_k{1,5,10,25}.csv` when `--ks` is set
- `audit.json` + `audit_summary.csv` (engine-free v2+ with 45° audit)

Re-aggregate without re-spending API tokens:
```bash
python experiments/kdd/llm_eval/eval_engine_free_llm_v3_standalone.py --reaggregate -o eval_out/<existing_run>
```

## Helpers

- `experiments/_lib/metrics/score_rollouts.py` — common (4) eval stage: re-scores
  every `.kicad_pcb` under a rollout root against `eval.metrics.evaluate_one`.
  Auto-detects per_board/<id>/sample_NN.kicad_pcb and PCBWORLD
  *_episode_NN_env_NN.kicad_pcb layouts.
