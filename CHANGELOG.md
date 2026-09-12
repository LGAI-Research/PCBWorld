# Changelog

Release notes for the PCBWorld environment, newest first. The engine is a separate program
with its own version line — its notes are in the
[PCBWorld-Engine](https://github.com/LGAI-Research/PCBWorld-Engine) repository's `CHANGELOG.md`
— and every environment tag pins exactly one engine commit (see `Versioning` in the README).

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
