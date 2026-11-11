# Changelog

Release notes for the PCBWorld environment, newest first. The engine is a separate program
with its own version line — its notes are in the
[PCBWorld-Engine](https://github.com/LGAI-Research/PCBWorld-Engine) repository's `CHANGELOG.md`
— and every environment tag pins exactly one engine commit: check out the tag and run
`git submodule update --init` to get the engine it was tested with.

## v1.2.1 — 2026-11-11

Documentation release. Code, configuration defaults and results are as in v1.2.0; the engine pin moves to v1.2.1, which
changes only its README.

### Added
- README "Benchmark results" — the paper's D3-A table (Table 3): Reference, Freerouting, KRT, four LLM agents and PPO
- Hugging Face checkpoints — README links the paper's trained policies (`LG-AI-Research/PCBWorld`), `docs/QUICKSTART.md` adds the
  download command; trained on an earlier release (results can differ from the paper), non-commercial license separate from the code

### Changed
- Contact — yj.park@lgresearch.ai
- LLM prompt glossary — every sentence carries a reviewed `origin` mark and `tests/test_llm_env_knobs.py` refuses an unreviewed or
  unmarked one; the prompt text is unchanged

## v1.2.0 — 2026-11-11

Platform and agent-interface release, pinned to engine v1.2.0. The RL observation format, the action set, the shipped
reward and masking rules and the wire protocol are unchanged, and v1.1.x checkpoints load with their stored settings.
Not comparable with v1.1.1: routing results (the ratsnest is now ordered by content; when that fix landed, 32 of the 240 D3-A / D3-B
test boards routed differently, other boards were not compared), and LLM-agent runs — under the defaults (generated prompt, richer
observation, 1024-step budget, `pcbworld_reward`), and with pinned settings too, since every caller now gets the `same_point` /
`off_board` / `net_closed` refusals.
Code that steps `PCBWorld` directly with coordinates outside the candidate set now needs `action_target: free`.

### Added
- Apple Silicon install (experimental) — `tools/setup/setup_all.sh` picks `environment.macos.yml` and `requirements-macos.txt` on
  an arm64 Mac, builds the engine from the conda env alone, RL recipes train on MPS; verified scope in the README
- `action_target` (`candidates` | `free`) — how an agent names a target point: a candidate-set point (default, the RL action
  space) or any coordinate (LLM default); under `candidates` the LLM observation lists the points each action may take

### Changed
- Action constraints — `PCBWorld.step` applies the rules in `pcb_world/core/constraints.py` to every caller (a refusal is
  `mask_reject` plus `info["constraint"]`); the six constraint knobs moved from `RLEnvConfig` to `EnvConfig`, flags and values kept
- LLM `latest` prompt — generated per run from `methods/llm_agent/wrappers/prompts/glossary.yaml` and the env config, sexpr only
  (a setting with no sentence stops the run); the hand-written text is frozen as `v8`, `--no-prompt_guidelines` drops the hints
- LLM `latest` observation — what the RL tokenizer reads: adds pad size and shape, obstacle shape, net-less pads, outline, track
  and via sizes, DRC violations and closed nets; drops the board-wide constraints, per-net `via_drill` and rule-area keepouts
- LLM env — follows `configs/env.yaml` (`max_steps` 1024 and `pcbworld_reward`; eval was 20 steps with `grpo_final`); `llm.env` may
  set only `action_target`, a run's other env values are printed as `DEVIATION` lines and kept in `run_record.yaml`
- Setup — `setup_all.sh` needs conda's libmamba solver or `mamba` in conda's base and stops otherwise (`SKIP_ENV=1` keeps an
  existing env), and refuses a `BUILD_DIR` from the shell; `tools/setup/build_engine.sh` rebuilds the engine; no Homebrew path

### Fixed
- Determinism — same-seed `--deterministic` rollouts could differ between runs on boards with pads of one net at one position;
  the engine now picks ratsnest anchors by content, not memory order
- Fresh Linux env — the `BUILD_PCBNEW=1` build failed against newly resolved OCCT 8 / SWIG 4.5 / NNG 2; `environment.yml` pins
  `occt=7.9`, `swig=4.4`, `nng<2`
- Closed nets — a caller without the RL net mask (the LLM path) could `net_select` a net it had released with `net_end`; the env
  refuses it (`net_closed`) unless `net_reselect` is on

### Documentation
- `docs/QUICKSTART.md` "Crash logs" — where crash traces go (`var/crashlogs/`), `KICAD_CRASH_LOG_DIR`, `KICAD_CRASH_DIAG`;
  `Notice.md` lists `scipy`, now a direct dependency

## v1.1.1 — 2026-10-21

PCBench preparation fix. Rebuild the D3 set with `tools/quickstart/prepare_pcbench.sh` to pick it up: D3 results on the 72 affected
boards are not comparable with a v1.1.0-built set. The environment, the engine (v1.1.0, unchanged) and checkpoints are as in v1.1.0.

### Fixed
- PCBench preparation — diff-pair width / gap that the source board's net classes left unset now follow the class track width / clearance,
  not KiCad's 0.2 / 0.25 mm defaults, which routed name-paired nets (e.g. `D+`/`D-`, `CAN_P`/`CAN_N`) at 0.2 mm or the board minimum on 72 of D3's 677 boards

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
