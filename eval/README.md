# `eval/` — KiCad PCB evaluation

`eval.metrics.evaluate_one` is the canonical DRC scorer for routed
`.kicad_pcb` files: it evaluates a routed board against the matching source
`.kicad_pro` design rules and returns a metric dict. The central
`eval.evaluator.Evaluator` orchestrates the two evaluation modes — rollout eval
(`Evaluator.run`) and post-hoc board scoring (`Evaluator.score_boards`) — and
both feed the same `eval.metrics.EvalSummary` summary into the sinks
(logger / CSV / JSON). The staged command-line entry point is
`python -m eval.pipeline`.

## Module map

| File | Role |
|---|---|
| `metrics.py` (`evaluate_one`, `compute_metrics`, `compute_metrics_inline`) | Scoring kernel and metric source of truth for routed `.kicad_pcb` files. Use this when you want Rout., DRV, WL, Via, and `final_potential`. `compute_metrics_inline` is the non-destructive live-env entry (u_0 from the env's reset-time capture) that both branch wrappers expose as their `eval_inline_drc` `env_method` hook. Also defines `EvalSummary`, the sink-agnostic summary. |
| `evaluator.py` (`Evaluator`) | Central evaluator: `Evaluator.run()` (rollout, mode A) and `Evaluator.score_boards()` (post-hoc board scoring, mode B). Plus the CSV/JSON sinks (`export_csv` / `export_json` / `emit_csv_artifacts`). |
| `pipeline.py` (`main`, `eval_kicad_pcb`) | The `python -m eval.pipeline` CLI orchestration (3 stages) and the post-hoc DRC stage `eval_kicad_pcb()` that scores saved `.kicad_pcb` artifacts. |
| `aggregation.py` (`aggregate_boards`) | Per-board / overall aggregation (Stage 3). |
| `eval_utils.py` | Stdlib-only CSV / schema / metric flattening helpers. Also home to the runtime metric-semantics kernel: `runtime_metrics_from_info()` (+ `success_from` / `clean_pass_from`) is the single place where live-env per-rollout derivations (success, clean_pass, ratsnest_reduction, ...) are defined — every producer (RL rollout, LLM live/plan-only) calls it instead of computing its own. |
| `rollout/` | Stage-1 rollout producers (see below). |

Support modules that used to live here (still used by the pipeline): board
loaders (`BoardSpec`) → `methods/_shared/board_loader.py`, metric-logging
sinks → `methods/_shared/logger.py`, checkpoint → policy/env-kwargs builders →
`methods/rl_agent/models/loader.py`, the forkserver parallel DRC worker pool
(`SubprocEvalPool`) → `pcb_world/vec/subproc_pool.py`.

### `rollout/` — Stage-1 rollout producers

Each module produces routed boards plus canonical `per_rollout` rows that the
shared scoring/aggregation layer consumes:

| File | Role |
|---|---|
| `rollout/rl.py` (`eval_transformer`) | RL decoder-policy rollout **driver** — packs `(board, rollout)` jobs into waves that fill all `n_envs` slots (`plan_job_schedule`, boards ordered small-first to cut straggler wait), drives the loop in `methods/rl_agent/rollout/transformer.py` (`_run_one_batch`; per-step transition = the shared primitive `methods/rl_agent/rollout/primitive.py`), flushes per-rollout rows. This is the rollout function the CLI injects into the rollout stage. |
| `rollout/rule_based.py` | Rule-based router rollout (KRT / OrthoRoute; runs in the single `cadagent` env). |
| (LLM producers) | live in `methods/llm_agent/rollout/` — `cadagent.py` (live vLLM / API rollout) and `plan_only.py` (API-sequence replay, no live LLM calls). |

## What it computes

For every routed board (`evaluate_one` / `compute_metrics`):

| metric | source |
|---|---|
| `success` | every net's pads form one connectivity group (`Gᵢ(t) == 1` for all nets) — equivalent to an empty ratsnest when no dangling copper is present |
| `routability` | `Σᵢ(Gᵢ(0) − Gᵢ(t)) / Σᵢ(Gᵢ(0) − 1)` where `Gᵢ(s)` = pad groups on net i (connectivity clusters holding ≥1 pad, via `KiCadEngine.get_pad_groups()`). Initial board = exactly 0, fully connected = exactly 1; dangling copper holds no pad so it cannot enter the metric; a lower-bound violation (pads that started joined ending up split) raises instead of emitting a negative score. Baseline `Gᵢ(0)`: disk path = fully stripped board; inline path = the episode-reset capture (`env._initial_pad_groups`). Filled by the DRC eval stage **only** — the rollout stage leaves the column NaN; the env-side per-step proxy is `ratsnest_reduction` = `(u₀ − u_t) / u₀` (signed: negative when the board grew more islands than it closed connections) |
| `track_count`, `via_count`, `wirelength_mm` | `KiCadEngine.get_reward_snapshot(run_drc=True)` |
| `drv_errors_only_count` | DRC violations whose severity == ERROR |
| `drv_errors_and_promoted_count` | ERROR + 3 promoted warnings (`DANGLING_VIA`/`DANGLING_TRACK`/`NET_CONFLICT`, codes 12/13/37) |
| `drv_violations` | full per-violation list with severity, error_code, error_type, x_mm, y_mm, layer, net_names, `is_error`, `is_promoted` flags |
| `final_potential` | `PotentialReward.compute_final(state)` — Φ(s) using the reward config (default `drc_dense_errors_only_eval` ⇒ `drc_severity_mode = errors_only`). Board-dependent terms (`completion_bonus_log_scale` · `clean_completion_bonus_log_scale` · `wirelength_bbox_normalize` · `net_bonus_size_log_scale`) are resolved by the training env's own definition, [`PotentialReward.bind_board`](../pcb_world/core/reward.py) (static group from the board meta, per-reset group from the same bare-board pad groups as the routability baseline) — offline Φ == training Φ, pinned by [tests/test_reward_parity.py](../tests/test_reward_parity.py) |
| `initial_potential` | Φ of the *bare* board (all tracks/vias deleted), snapshot with `run_drc=True` — same DRC convention as `final_potential` so `potential_gain` does not mix a DRC-free initial with a DRC-included final. `None` on the inline path (`u₀` supplied) |
| `potential_gain` | `final_potential − initial_potential` (`None` if no baseline) |
| `phi_components` | breakdown of the five base Φ terms only: `completion_bonus`, `−unconnected`, `−drc`, `−wirelength`, `−via` — ladder / clean-completion terms are not itemized, so `total` ≠ `final_potential` under ladder rules |
| `phi_weights` | the reward config's weights as resolved on this board (so you can re-derive Φ): `completion_bonus`, `clean_completion_bonus`, `net_completion_bonus`, `net_clean_bonus`, `net_bonus_size_log_scale`, `net_size_weights` (`{net_code: wᵢ}` for the size-weighted ladder, else `None`), `unconnected_penalty`, DRC shape/scales/severity, `wirelength_penalty` (post-normalization), `via_penalty`, `step_penalty` |
| `extras.per_net` | per-net wirelength / track / via / unrouted-edges |
| `extras.board_meta` | bbox, net count, copper layer count |

The reward config controls how DRC is shaped (linear / saturating / `log_per_net`)
and which severity is counted. The default `drc_dense_errors_only_eval` is the
canonical eval setting (`drc_severity_mode = errors_only`, so `clean_pass`
counts only true ERROR violations); pass `--reward-config <name>` to override
(e.g. `drc_dense_promoted` ⇒ `errors_and_promoted`).

## Required setup

The C++ KiCad RL router must be built and the env vars set:

```bash
conda activate <your-cadagent-env>
cd <cadagent-repo>
export PYTHONPATH=build_rl/pcbnew/python/rl:.
```

## Source `.pro` matching

`evaluate_one(routed_pcb, pro_path)` takes the routed board and the path to its
source `.kicad_pro` directly. When scoring through the pipeline (`--boards-dir`),
the source `.pro` is resolved per board via `eval.eval_utils.resolve_pro_path`,
which strips the rollout filename suffix
(`<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb`) to find the matching source design.

## CLI — `python -m eval.pipeline`

The pipeline runs up to three stages: **(1) rollout** a live policy over a board
set, **(2) post-hoc DRC** scoring of the saved `.kicad_pcb` artifacts, and
**(3) aggregate** per-board / overall metrics.

### Full run (rollout + score + aggregate)

```bash
python -m eval.pipeline \
  --ckpt <path/to/checkpoint> \
  --boards-dir <path/to/board/dir> \
  --seed 42 \
  --n-rollouts 5 \
  --n-envs 1
```

`--ckpt`, one of `--boards-dir`/`--boards-list`, `--seed`, and `--n-rollouts`
are all required unless `--skip-rollout` is set.

### Stage selection

```bash
# only the post-hoc DRC + aggregate stages over an existing rollout dir
python -m eval.pipeline --skip-rollout --output-dir <rollout-dir> \
  --stages eval,aggregate
```

`--skip-rollout` requires `--output-dir` pointing at a directory that already
contains a `per_rollout.csv` (or a `boards/` directory of routed
`.kicad_pcb` files, from which the per-rollout rows are reconstructed). This is
how rollouts produced by the LLM / rule-based `eval.rollout.*` modules are scored
through the same path.

### Useful flags

| flag | meaning |
|---|---|
| `--ckpt PATH` | Policy checkpoint to roll out (Stage 1). Required unless `--skip-rollout`. |
| `--boards-dir DIR` / `--boards-list FILE` | Board source for the rollout (mutually exclusive). |
| `--seed N` | Base rollout seed. |
| `--n-rollouts N` | Rollouts per board. |
| `--n-envs N` | Number of envs (also the post-hoc DRC worker count). Default `1`. |
| `--rollout-mode {serial,parallel}` | Default `serial` (requires `--n-envs 1`). |
| `--inline-drc {on,off}` | `on` (serial only): score DRC inline on the live engine and skip Stage 2. `off` (default): score saved `.kicad_pcb` post-hoc in Stage 2. |
| `--env-drc {on,off}` | Env DRC reward/tokens during rollout. Omitted: follow the ckpt's training `emit_drc_tokens`. |
| `--reward-config NAME` | DRC scoring config in `configs/reward/`. Default `drc_dense_errors_only_eval` (errors_only). e.g. `drc_dense_promoted` (errors_and_promoted). Warns when it differs from the ckpt's training `reward_rule`. |
| `--check-angle {45,90}` | DRC track-angle check. Omitted: inherit the ckpt's `corner_mode`. |
| `--selection-method {final_potential,posthoc_drc_aware,none}` | Per-board best-rollout selection. Default `final_potential`. |
| `--save-artifacts {on,off}` | Save routed `.kicad_pcb` per rollout. Default `on` (required for post-hoc Stage 2). |
| `--output-dir DIR` | Output root. Default `outputs/eval_overall/<ts>_<ckpt>_seed<seed>`. |
| `--skip-rollout` / `--skip-drc` / `--skip-aggregate` | Skip individual stages. |
| `--stages rollout,eval,aggregate` | Positive stage selector (aliases `drc`=eval, `agg`=aggregate); overrides the `--skip-*` flags. |
| `--override-n-max-slots N` | Lift the ckpt's trained slot cap at inference. Default `1280`. |
| `--dry-run` | Print the resolved plan and exit. |

## Library API — post-hoc board scoring

To score finished boards without the CLI, use either entry point:

```python
# Pipeline function: scores a rollout dir's saved artifacts, merging DRC
# metrics into its per_rollout.csv (the same Stage 2 the CLI runs).
from eval.pipeline import eval_kicad_pcb
eval_kicad_pcb(rollout_dir, n_workers=8, boards_dir=source_dir)

# Evaluator: score an explicit list of (routed_pcb, pro_path) pairs into an
# EvalSummary summary (parallel>1 uses SubprocEvalPool).
from eval.evaluator import Evaluator
metrics = Evaluator.score_boards(
    [("board_00000_freerouting.kicad_pcb", "board_00000.kicad_pro"), ...],
    parallel=8,
)
```

Or, for a single board, call the kernel directly:

```python
from eval.metrics import evaluate_one
result = evaluate_one(routed_pcb, pro_path, reward_config_name="drc_dense_errors_only_eval")
```

## Output tree

A pipeline run writes into `--output-dir`:

```
<output-dir>/
├── boards/                    # routed rollouts, unified cell grammar (when --save-artifacts on):
│                              #   <board_id>_<cell>_s<SS>_r<RR>.kicad_pcb (+ .kicad_pro/.kicad_prl)
├── per_rollout.csv            # one row per rollout, DRC columns filled by Stage 2
├── per_board_avg.csv          # per-board aggregates (alias: per_board.csv)
├── manifest.json              # env_kwargs + resolved args
└── eval_overall_summary.json  # overall summary + stage wall times
```

With `--save-artifacts on`, Stage 1 saves each rollout under a temporary
`artifacts/` staging dir and then flattens it into `boards/` under the unified
cell grammar (`flatten_rollout_artifacts`; `<cell>` = the output dir basename,
`s<SS>` = the ckpt's training seed), rewriting the `artifact_path` column to
match. Stage 3 (`aggregate_boards`) parses exactly that grammar from
`per_rollout.csv` and **fails loudly** (`SystemExit`) when no row matches it —
it never exits 0 with nothing written.

`per_rollout.csv` is flushed incrementally during the rollout and updated in
place by the post-hoc DRC stage (keyed by the saved board filename, which is
unique per rollout across ckpt seeds). Re-running with `--skip-rollout
--stages eval,aggregate` re-scores / re-aggregates an existing dir.

## Reading the DRV breakdown

`drv_breakdown.errors_only_by_type` and `drv_breakdown.errors_and_promoted_by_type`
each list `{severity, error_code, error_type, count}` rows. KiCad error_code
mapping for the 3 promoted warnings:

| code | error_type | meaning |
|---|---|---|
| 12 | via_dangling | a via with no track ending on either copper layer |
| 13 | track_dangling | a track segment with one end floating |
| 37 | net_conflict | a track/via assigned to a net that disagrees with the connectivity (potential short) |

`severity` is the human label (`ERROR` / `WARNING`); the integer KiCad value is
also kept in `drv_violations[i].severity`.

## Caveats

* **Pro file matters.** Without a source `.pro`, the engine falls back to KiCad's
  compile-time default rules (more permissive); `Track width` / `Via diameter`
  / `Hole size` violations may be undercounted. Always supply the source pro
  (`--boards-dir` resolves it per board; the library API takes it explicitly).
* **Singleton C++ engine.** Only one `RLRouter` instance can live at a time
  within a process, so serial scoring is sequential. Parallel scoring fans out
  across `forkserver` subprocesses via `SubprocEvalPool` (`--n-envs N` /
  `parallel=N`), each its own process.
* **`final_potential` ≠ training reward at convergence.** The training reward
  is the per-step potential delta plus step penalty, not Φ itself. Φ is what
  `info["final_potential"]` exposes at episode termination — i.e. the score of
  a finished board under the reward config's weights.
