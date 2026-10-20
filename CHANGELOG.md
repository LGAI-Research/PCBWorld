# Changelog

Release notes for the PCBWorld environment, newest first. The engine is a separate program
with its own version line — its notes are in the
[PCBWorld-Engine](https://github.com/LGAI-Research/PCBWorld-Engine) repository's `CHANGELOG.md`
— and every environment tag pins exactly one engine commit: check out the tag and run
`git submodule update --init` to get the engine it was tested with.

## v1.1.0 — 2026-10-20

Configuration and checkpoint-compatibility release. Observations, actions, the shipped reward and
masking rules and the wire protocol are unchanged; v1.0.x checkpoints load with their training
settings and route synthetic boards exactly as under v1.0.1. D3 numbers are not comparable with
v1.0.1 (99-board D3-A, per-net-class route sizes, failed attempts in the denominators). A clone that
fetched the withdrawn 2026-09 `v1.1.0` tag updates it with `git fetch --tags --force`.

### Added
- Run records — training, evaluation and re-scoring write `run_record.yaml` (every setting's value
  and source); `python -m configs.loader.explain <run dir | checkpoint>` prints one
- `net_reselect`, `trim_at_contact` — opt-in env knobs, off by default

### Changed
- Configuration — defaults in `configs/env.yaml`, `rl.yaml`, `llm.yaml` (`configs/pcbworld.yaml`
  removed), experiment values via `--recipe`; `PCBWorld(board_path, env_config)` takes one `EnvConfig`
- Command line — abbreviated or repeated options are errors; `--use-yaml-drc-fallback` →
  `--rules-source`, engine-knob and `--finetune-*` flags → config keys (the error names the replacement)
- Checkpoints — store the resolved config plus `args`, so v1.0.1 reads them; keys an older file lacks
  come from `configs/ckpt_migrations.yaml` (warning; `--strict-ckpt` refuses)
- New-run defaults — RL: `policy_net_select`, `disable_slot_emb`, validation on `val` every 10
  iterations; LLM: `latest` prompt, `pcbworld_masking`, stop after 10 steps without geometry change
- Benchmark and evaluation — D3-A is 99 boards, rule-based runners take `--difficulty`; failed and
  crashed attempts stay in every denominator; re-scoring reuses the cell's recorded scoring

### Fixed
- Routing — each net uses its own net class's track and via sizes (a Default-class net took the previous net's)
- Training — PPO/GRPO updates apply the sampling-time layer mask; `--resume` keeps the LR schedule,
  board order and best checkpoint
- Checkpoints and paper recipes — default eval accepts a default-trained checkpoint; the recipes and
  KDD-era checkpoints use the paper's `connectivity_filter` / `reject_if_stuck` (off)

## v1.0.1 — 2026-09-16

Patch release. Observations, actions, reward, masking, DRC rules, benchmark splits and the wire
protocol unchanged — v1.0.0 numbers carry over except for the net-subset fix.

### Added
- Quick start — three front-door scripts under `tools/quickstart/`, and two guides,
  `docs/ENVIRONMENT.md` and `docs/METRICS.md`
- Router provenance — build stamps `ENGINE_CPP_HASH`, environment refuses a router from other
  sources (`PCBWORLD_ENGINE_ALLOW_MISMATCH=1` → warning)
- `KICAD_ENGINE_MAX_RSS_MB` (default 1024, `0` off) — over-budget engine server replaced, not reused

### Fixed
- Net-subset (`target_nets`) — DRC scored whole-board after a board reload; whole-board unaffected
- D1 recipes — two undefined helpers, so exit 127 instead of the documented exit 2
- Checkpoint without `policy_net_select` read as a net index
- Synthetic grid datasets — relative `--out-prefix` outside the repo root, split json without
  `dataset_dirs`

### Changed
- README → short landing page; `docs/QUICKSTART.md` on D1 — protocol runs on a locally generated
  corpus, the paper's own numbers do not

### Removed
- `external/RAGEN` and its overlay (`external/patcher.sh` takes `verl-agent` or `all`), and the
  rule-based upstream notes — angle guidance moved to `methods/baselines/rule_based/README.md`
- The documentation-consistency checker — development tooling, no longer shipped

## v1.0.0 — 2026-09-11

First public release. PCBWorld wraps KiCad's push-and-shove router as a Gymnasium environment
for PCB routing and ships the benchmark and the training/evaluation code around it.

### The environment
- `PCBWorld`: a Gymnasium env over a real `.kicad_pcb` board — six routing actions plus an
  LLM-only `idle`, hierarchical JSON-dict observations, every step scored by KiCad's DRC through
  the engine's push-and-shove router. Deterministic: the same board geometry and the same action
  give the same result across re-saved files and reused engine servers.
- The engine runs as a separate GPLv3 program ([PCBWorld-Engine](https://github.com/LGAI-Research/PCBWorld-Engine),
  pinned as the `engine/` submodule) that the environment talks to over a unix socket; no combined
  artifact is built or distributed.
- One shipped configuration: the reward rule `pcbworld_reward` (a linear pad-group DRC ladder),
  the masking rule `pcbworld_masking` (net_end free, no finish action), the fallback DRC rules
  `pcbworld_drc`, and one defaults file, `configs/pcbworld.yaml`, whose sections feed every
  command line. Observations carry obstacle and pad-shape tokens, arc outlines, per-net
  constraint channels and multi-resolution directional candidates by default.

### The benchmark
- Synthetic board generators (2-layer `synth_2L`, 1-layer grid families) and the D3 real-board
  set: 679 PCBench boards, rebuilt from the public PCBench clone by the shipped preparation chain
  (`tools/datagen/pcbench_prep/`; README Quick start §3), listed in `configs/datasets/d3.json`
  with easy / medium / hard difficulty splits. D3 is evaluation-only.
- One uniform three-stage evaluation (`eval/pipeline.py`: rollout → post-hoc DRC → aggregate)
  applied identically to every method; DRC scoring uses the same `pcbworld_reward` rule as training.

### Methods
- A decoder-only PPO / GRPO transformer trainer (`scripts/train.py ppo|grpo`; flex-attention
  path, BC-initialised fine-tuning options with a critic warm-up and a reference-KL penalty).
- LLM tool-calling agents (`OPENAI` / `Anthropic` / `Google` / OpenAI-compatible endpoints) over
  the same environment, and classical rule-based routers (FreeRouting, OrthoRoute, KRT) through
  the same evaluation pipeline.

### Paper reproduction
- The KDD 2026 workshop paper's recipes live under `experiments/kdd/` together with the exact
  configuration the paper used (`experiments/kdd/configs/`: its reward and masking rules, DRC
  rules, dataset splits and checkpoint map). Those rules resolve by name from the recipes, so the
  paper's numbers reproduce on this release even though the repository's shipped defaults differ.
- Checkpoints saved with earlier, paper-era settings evaluate with those settings
  (`experiments/kdd/configs/legacy_ckpt_defaults.yaml`; `tools/ckpt_migrate.py` writes them into a
  checkpoint file).

### Licensing
- This repository is BSD-3-Clause; the engine repository is GPLv3 (derived from KiCad).
