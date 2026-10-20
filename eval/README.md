# `eval/` — KiCad PCB evaluation

`eval.metrics.evaluate_one` is the canonical DRC scorer for a routed `.kicad_pcb`: it scores the board against
its source `.kicad_pro` design rules and returns a metric dict. `eval.evaluator.Evaluator` drives both evaluation
modes — rollout eval (`Evaluator.run`) and post-hoc board scoring (`Evaluator.score_boards`) — feeding the same
`eval.metrics.EvalSummary` into the sinks (logger / CSV / JSON). The staged CLI is `python -m eval.pipeline`, fronted
by [scripts/eval.py](../scripts/eval.py); setup [docs/QUICKSTART.md](../docs/QUICKSTART.md), numbers [docs/METRICS.md](../docs/METRICS.md).

## Module map

| File | Role |
|---|---|
| `metrics.py` | Scoring kernel and metric source of truth: `evaluate_one` / `compute_metrics` (disk board + `.pro`) and `compute_metrics_inline`, the non-destructive live-env entry (u₀ from the env's reset-time capture) that both branch wrappers expose as their `eval_inline_drc` `env_method` hook. Also defines `EvalSummary`, the sink-agnostic summary. |
| `evaluator.py` | `Evaluator.run()` (rollout, mode A) · `Evaluator.score_boards()` (post-hoc, mode B) + the CSV/JSON sinks (`export_csv` / `export_json` / `emit_csv_artifacts`). |
| `pipeline.py` | The `python -m eval.pipeline` 3-stage CLI (`main`) and the post-hoc DRC stage `eval_kicad_pcb()` over saved artifacts. |
| `aggregation.py` | Stage 3: `aggregate_boards` (post-hoc, from `per_rollout.csv`) and `aggregate_inline` (in-memory, training validation). |
| `eval_utils.py` | Stdlib-only CSV / schema / flattening helpers, plus the runtime metric-semantics kernel: `runtime_metrics_from_info()` (+ `success_from` / `clean_pass_from`) is the single definition of the live-env per-rollout derivations (success, clean_pass, ratsnest_reduction, …) — every producer calls it instead of computing its own. |
| `args.py` | Shared argparse builders for the LLM-env eval entry points; the flag list derives from the config schema. |
| `rollout/rl.py` | Stage-1 RL rollout **driver** (`eval_transformer`): packs `(board, rollout)` jobs into waves that fill all `n_envs` slots (`plan_job_schedule`, boards small-first to cut straggler wait), drives `methods/rl_agent/rollout/transformer.py`, flushes per-rollout rows. This is what the CLI injects into Stage 1. |
| `rollout/rule_based.py` | Stage-1 rule-based router rollout (FreeRouting / KRT / OrthoRoute). |

Same pipeline, other packages: LLM Stage-1 producers in `methods/llm_agent/rollout/`; board loaders (`BoardSpec`)
in `methods/_shared/board_loader.py`; metric-logging sinks in `methods/_shared/logger.py`; policy/env-kwargs
builders in `methods/rl_agent/models/loader.py`; the DRC worker pool (`SubprocEvalPool`) in `pcb_world/vec/subproc_pool.py`.

## Metric columns

Produced per routed board by `evaluate_one` / `compute_metrics`, written to `per_rollout.csv`:

| metric | meaning |
|---|---|
| `success` | every net's pads form one connectivity group — equivalent to an empty ratsnest when no dangling copper is present |
| `routability` | `Σᵢ(Gᵢ(0) − Gᵢ(t)) / Σᵢ(Gᵢ(0) − 1)` over pad groups `Gᵢ` (connectivity clusters holding ≥1 pad): exactly 0 on the bare board, exactly 1 fully connected; dangling copper holds no pad so it cannot enter the metric, and a lower-bound violation raises instead of emitting a negative score. Filled by the DRC stage **only** — the rollout stage leaves the column NaN; the env-side per-step proxy is `ratsnest_reduction` (signed) |
| `track_count`, `via_count`, `wirelength_mm` | engine reward snapshot (`run_drc=True`) |
| `drv_errors_only_count` · `drv_errors_and_promoted_count` | ERROR-severity violations · ERROR + the 3 promoted warnings (codes 12/13/37, below) |
| `drv_violations` | per-violation records: severity (int) + label, error_code, error_type, x_mm, y_mm, layer, net_names, `is_error`, `is_promoted` |
| `final_potential` · `initial_potential` · `potential_gain` | Φ(routed) · Φ(bare board, all tracks/vias deleted) · their difference. Both ends score with `run_drc=True`, so the gain never mixes a DRC-free initial with a DRC-included final. Board-dependent Φ terms resolve through the training env's own [`PotentialReward.bind_board`](../pcb_world/core/reward.py) (per-reset group = the same bare-board pad groups as the routability baseline), so offline Φ == training Φ — pinned by [tests/test_reward_parity.py](../tests/test_reward_parity.py). On the inline path Φ₀ is the env's reset-time bare-board state priced with the scoring config (equal to post-hoc), blank (with a warning) when that reset ran no DRC and the scoring config reads DRC — Stage 2 then prices the input board and Stage 3 fills it — never the env's training-rule Φ₀ |
| `phi_components` · `phi_weights` | the five base Φ terms plus their `total` (penalties signed negative; ladder / clean-completion terms are **not** itemized, so `total` ≠ `final_potential` under ladder rules) · the reward config's weights as resolved on this board, enough to re-derive Φ |
| `extras.per_net` · `extras.board_meta` | per-net wirelength / track / via / unrouted-edges · bbox, net count, copper layer count |

Which violations count, and how the DRC term is shaped, is the reward config's business (`configs/reward/`, knob
`drc_severity_mode`); `--reward-config NAME` scores under another rule, warning (rollout and re-scoring) when its Φ differs from the ckpt's training reward (rule + per-run overrides).

## Source `.pro` matching

`evaluate_one(routed_pcb, pro_path)` takes the project path directly; through the pipeline it is resolved per board
by `eval.eval_utils.resolve_pro_path`, in order: the co-located `<stem>.kicad_pro`; for the cell grammar, the longest
co-located `<board_id>.kicad_pro` whose stem prefixes the file; then, under `--boards-dir`, `<stem>.kicad_pro` with
and without the staging suffix `_rollout_<idx>`.

## CLI — `python -m eval.pipeline`

Three stages: **(1) rollout** a live policy over a board set, **(2) post-hoc DRC** over the saved `.kicad_pcb` artifacts, **(3) aggregate**.

```bash
python -m eval.pipeline --ckpt <ckpt> --boards-dir <dir> --seed 42 --n-rollouts 5 --n-envs 1
python -m eval.pipeline --skip-rollout --output-dir <rollout-dir> --stages eval,aggregate   # score + aggregate only
```

Every flag is in `--help` and every default in the config schema; only these carry a constraint:

* `--ckpt`, one of `--boards-dir` / `--boards-list`, `--seed`, `--n-rollouts` — required unless `--skip-rollout`.
* `--skip-rollout` needs `--output-dir` pointing at a dir that already holds a `per_rollout.csv` (or a `boards/`
  dir of routed `.kicad_pcb` to reconstruct the rows from) — this is how the rule-based and LLM producers are
  scored through the same path. Pass the cell's `--boards-list` there too: it is the expected board set (Stage 2
  prices the failed boards' Φ₀, Stage 3 counts a board without rows); `--boards-dir` there only locates `.kicad_pro`.
* `--stages rollout,eval,aggregate` is the positive selector (aliases `drc`=eval, `agg`=aggregate); it overrides `--skip-rollout` / `--skip-drc` / `--skip-aggregate`.
* `--rollout-mode serial` requires `--n-envs 1`; `--n-envs` is also the Stage-2 DRC worker count.
* `--inline-drc on` scores on the live engine and skips Stage 2 — serial only, and incompatible with `--skip-rollout`; `--save-artifacts off` leaves nothing for Stage 2 to score.
* Omitted, `--env-drc` inherits the ckpt's `emit_drc_tokens` and `--check-angle` the env's final `corner_mode` (with a ckpt, any other explicit angle is an error).
* The rollout's `run_record.yaml` records the scoring (`extra.scoring`: the env's `check_angle`; `reward_config` only when the run scores or names it, with `training_reward_rule`, `training_reward_overrides` and `matches_training_rule` beside it; `selection_method` only when it aggregates or names it); re-scoring (`--skip-rollout` + Stage 2 and/or 3) reuses what the cell records (the last re-scoring's `run_record.rescore.yaml`, else the rollout's record, else the rows' columns for the DRC scoring and, for the selection, the rollout's `manifest.json` when that run aggregated)
  and writes `run_record.rescore.yaml` (every setting the cell records, the ones it decided included; with the training comparison of the cell's rule) once Stage 2 succeeds and again once Stage 3 does, adding the selection — a failed stage or `--dry-run` leaves the record as it was. An explicit flag wins: another value than the recorded one re-scores the cell under it (one warning naming both; the record's provenance chain keeps the cell's value before the `cli` entry), and it picks among several recorded values or supplies a missing one (an unrecorded selection is the default; an unrecorded DRC scoring key has no default, e.g. `--reward-config` for a `--skip-drc` rollout); without the flag a cell whose rows carry several values raises — one error names every unrecorded or ambiguous key. The Φ check against the ckpt's training reward runs on the rule actually used. Stage 3 refuses rows scored under several `reward_config` or `check_angle` values (an interrupted Stage 2): rerun Stage 2 with one.

## Library API — post-hoc board scoring

```python
eval_kicad_pcb(rollout_dir, n_workers=8, boards_dir=src)           # eval.pipeline — Stage 2 over a rollout dir
Evaluator.score_boards([(routed_pcb, pro_path), ...], parallel=8)  # eval.evaluator — pairs -> EvalSummary
evaluate_one(routed_pcb, pro_path, reward_config_name="...")       # eval.metrics — one board, the kernel
```
`eval_kicad_pcb` merges its DRC columns back into the dir's `per_rollout.csv`; `parallel>1` fans out over `SubprocEvalPool`.

## Output tree

A pipeline run writes into `--output-dir`:

```
<output-dir>/
├── boards/                    # routed rollouts, unified cell grammar (when --save-artifacts on):
│                              #   <board_id>_<cell>_s<SS>_r<RR>.kicad_pcb (+ .kicad_pro/.kicad_prl)
├── per_rollout.csv            # one row per rollout, DRC columns filled by Stage 2
├── per_boards_ckpts.csv       # Stage 3: one row per board x ckpt seed
├── per_boards_overall.csv     # Stage 3: one row per board
├── per_boards_summary.csv     # Stage 3: one model-level row
├── manifest.json · run_record*.yaml  # env_kwargs + resolved args · Stage 1's per-key config record; .aggregate (python -m eval.aggregation) names it by hash
└── eval_overall_summary.json  # overall summary + stage wall times
```

Stage 1 saves each rollout under a temporary `artifacts/` staging dir, then flattens it into `boards/` under
the cell grammar (`flatten_rollout_artifacts`; `<cell>` = the output dir basename, `s<SS>` = the ckpt's
training seed), rewriting the `artifact_path` column to match. Stage 3 parses exactly that grammar from
`per_rollout.csv`, first removing the previous `per_boards_*.csv`, and **fails loudly** (`SystemExit`) when no row matches
but a row names an artifact or is `completed` (a cell whose rows all failed — no artifact, a failure status — is aggregated, with a warning), and when the cell has routed rows but the scorer (Stage 2, or inline DRC) scored none of them (all `eval_status=error`: a scorer failure, not a 0% result — the message names the first errors and the rerun; `aggregate_inline` only warns, since its callers return the rows).
A row with no artifact in that grammar (crash, nothing routed, unconvertible output, timeout, incompatible board) is a
**failed attempt**, never dropped — the rule the inline (training val) aggregation shares, where a crashed or
skipped-incompatible row is one (`Evaluator.score_boards` marks a scoring error `crashed` only when the board file is
missing; in the inline aggregation a failed attempt's Φ₀ is the one its board's scored rows carry, else the env's reset value, which the rollout copies onto crash and skip rows,
when that is the scorer's Φ₀ too — a run with no scored row, or one whose scored rows all record `env_phi0_matches` (the env's reward and the scoring config define the same Φ) — and a board still without one is counted in `n_failed_boards_without_phi0`, with a warning): `success` / `clean_pass` / `routability` score 0 (not injected when the cell's routed
rows do not measure them, e.g. before Stage 2, unless every routed row failed scoring — then all three are 0; every view — `_avg`, `_best`, the selection — skips a blank bool metric instead of reading it as 0), the potential is the bare board's Φ₀
(`final_potential` = `initial_potential`, `potential_gain` 0 — the board as if left untouched; a crash's partial routing
is not scored), the routed-board measurements (wirelength, vias, DRV) are left blank, so those average only attempts
that produced a board, a failed attempt wins the selection only on a board with no routed attempt (and a routed row whose scoring failed, still carrying env values, never wins over a scored one), and time still
counts. Stage 2 records Φ₀ by scoring the input board of every board with a failed attempt, a ckpt whose routed attempts all failed scoring, or a scored attempt without Φ₀ (inline scoring whose reset ran no DRC; Stage 3 fills those rows' `initial_potential` / `potential_gain` from it and counts a scored row still without a gain in `n_scored_without_potential_gain`, with a warning) (`initial_potentials.csv`;
rules from its own `.kicad_pro`, else the board-list entry's;
Φ₀ from another reward config than the routed rows' is an error; a board without Φ₀ leaves the potential blank, with a
warning and `n_failed_boards_without_phi0`, which also counts a board whose unscored attempts below found no Φ₀). An existing board file outside the grammar, a failed row without a
`board_id`, or a `ckpt` column that contradicts the artifact name is an error, not a failure — in Stage 2's Φ₀ step
too (`classify_cell_rows`). The board set is Stage 1's, or `--boards-list` on `--skip-rollout` (a board with
no row is one failed attempt per ckpt; a row outside it is an error); without either it is the boards in
`per_rollout.csv`, with a warning. A board with no row under one of the cell's ckpts failed there (`missing_row`). A failed attempt is charged to its row's `ckpt` column (Stage 1 stamps it on every
row) or the cell's only ckpt; otherwise it enters only the board's `_avg` (`n_unattributed_failed`, with a warning — a board with a routed attempt but a missing ckpt gets an optimistic `_best`),
and a board whose every attempt is such a failure takes the failure as its `_best`; such a board's missing ckpts
are not filled. A routed row whose scoring failed (`eval_status=error`) is not a failure: its blank metrics are skipped the same way, while the env rollout values it still carries (e.g. `success`, `final_potential`) are read unrescored — except where no routed attempt of the selection group (Stage 3: board × ckpt; inline: board) was scored: those rows are priced exactly like failed attempts (Φ₀, pass-type 0, no wirelength/vias/DRV), never by their env values. Both count in `n_eval_error`, not `n_failed` (with a warning; inline `overall` too).
Every `per_boards_*.csv` carries `n_failed`, `n_crashed` (the failed attempts with status `crashed`) and `n_eval_error`; the summary
adds `expected_boards` (`given` / `per_rollout_rows`) and, when RL rows carry the give-up record (`giveup_count` · `giveup_events` · `unrouted_nets`), the diagnostic `giveup_*` keys of [`giveup_quality`](aggregation.py) (`aggregate_inline` adds them to `overall` too), and, when LLM rows carry
`api_failed_calls` (steps whose provider call failed and fell back to `idle` — still scored; blank = not recorded), [`api_failure_summary`](aggregation.py)'s sum `api_failed_calls` and `api_failed_boards` (also in `overall`), warning when nonzero; per-board rows sum their rollouts' `api_failed_calls`.
`per_rollout.csv` is flushed incrementally during the rollout and updated in place by the post-hoc DRC stage,
keyed by the saved board filename (unique per rollout across ckpt seeds). Live-env rows (RL and LLM) carry `rules_digest` (`PCBWorld.rules_digest`: digest of the board minima, size presets and net class sizes the board was routed under — not the net→class assignment or custom `.kicad_dru` rules) and `rules_substitution` (JSON of the YAML minima that replaced a board without rules under `rules_source=board_or_yaml`; blank when the board's own rules applied); both from the env's terminal info (RL) or reset info (LLM, so early-stopped episodes keep them), blank without a live env and on a crash.

## Reading the DRV breakdown

`drv_breakdown.errors_only_by_type` and `drv_breakdown.errors_and_promoted_by_type` each list
`{severity, error_code, error_type, count}` rows, where `severity` is the human label (`ERROR` / `WARNING`)
and the integer KiCad value stays in `drv_violations[i].severity`. The 3 promoted warnings:

| code | error_type | meaning |
|---|---|---|
| 12 | via_dangling | a via with no track ending on either copper layer |
| 13 | track_dangling | a track segment with one end floating |
| 37 | net_conflict | a track/via on a net that disagrees with the connectivity (potential short) |

## Caveats

* **The source `.pro` is mandatory.** Without it the rules would fall back to KiCad's compile-time defaults
  (more permissive — `Track width` / `Via diameter` / `Hole size` undercounted), so `KiCadEngine` raises
  `RuntimeError` instead of scoring.
* **Singleton C++ engine.** One `RLRouter` per process, so serial scoring is sequential; parallel scoring
  fans out across `forkserver` subprocesses (`SubprocEvalPool`, `--n-envs N` / `parallel=N`).
* **`final_potential` ≠ the training reward at convergence.** The training reward is the per-step potential
  delta plus step penalty, not Φ itself; `final_potential` is Φ of the finished board under the config's weights.
