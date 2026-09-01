# Changelog

Release notes for the PCBWorld environment, newest first. The engine is a separate program
with its own version line — its notes are in the
[PCBWorld-Engine](https://github.com/LGAI-Research/PCBWorld-Engine) repository's `CHANGELOG.md`
— and every environment tag pins exactly one engine commit (see `Versioning` in the README).

## v1.1.0 — 2026-09-01

### Added
- Directional candidate preset `mres8` (8 directions × 8 log-scale distances, 0.2–50 mm) and an
  optional off-board pointer mask for directional candidates (`offboard_mask`, default off;
  checkpoints trained without it load unchanged).
- PCBench → D3 reconstruction chain in `tools/datagen/pcbench_prep/`: KiCad 5→9 conversion, DRC
  repair/filter, deterministic guide generation and the difficulty sort rebuild the paper's
  679-board D3 set from a public PCBench clone (README §3).
- Flex-attention path for the policy network, a cleaned-up attention-mask form and a
  branch-free pointer block.
- Qwen sweep recipes for the LLM baseline (`experiments/kdd/table1_llm/baselines/`).
- Dataset registry entries in `configs/paths.yaml` for the rule-based grid families, the KDD
  figure inputs and the D3 DSN mirror.
- README: a `Versioning` section (how the two repositories are numbered, and that an environment
  tag pins exactly one engine commit) and an explicit statement that D3 is evaluation-only.
- The D3 chain's KiCad tools can now be built from the engine's own pinned source:
  `BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh` adds `kicad-cli` and the `pcbnew`
  Python module to `build_rl/`, and the chain scripts resolve them automatically (printing the
  choice; `KICAD_CLI` / `PCBNEW_PYTHON` override). README Quick start §3 is now a runnable
  block: a shallow PCBench clone plus a bounded trial of the four-step chain.
- README Quick start §6 — routing a board with an LLM agent over the same environment (one API
  call per step, the same seven actions). Its first command needs no API key: it renders the
  exact prompt the agent receives and executes one action; the API rollout in the same block
  runs when a key is present (OpenAI / Anthropic / Google / Together, and any OpenAI-compatible
  endpoint through `OPENAI_BASE_URL`) and skips itself otherwise.

### Changed
- **One KiCad, no containers.** Everything that needs KiCad — the router, `kicad-cli` for DRC, and
  the `pcbnew` Python module for board conversion — now comes from a single build of the engine's
  pinned, patched 9.0.8 source (`BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh`). No
  step of the build, the setup scripts or the data preparation installs a distribution KiCad or
  runs a container, so the KiCad the data is made with cannot drift from the one the environment
  routes with. A host KiCad still works if you prefer one (`KICAD_CLI` / `PCBNEW_PYTHON`).
- Engine pin moves to PCBWorld-Engine 1.1.0 — **rebuild required**. The DRC per-type report cap
  (199 / 499) is removed, and the connected-points query is fixed at thru-hole pad / via
  centres. **Behaviour change**: on boards with thru-hole pads or vias the already-connected-point
  cluster is now populated, so candidate sets — and therefore rollouts — differ from 1.0.0 there;
  DRC counts on boards with more than 199 / 499 violations of one type are now exact.
- `$KDD_BENCH_ROOT` defaults to `$CADAGENT_DATA_ROOT/KDD_benchmark` when the data root is set.
- Rule-based runners, the KDD recipes and plan-only evaluation resolve dataset paths through
  `configs/paths.yaml` like everything else.
- D3 rebuild: with the engine's uncapped DRC, the guide step narrows only the nets DRC actually
  blames — the narrow-all fallback for capped violation lists no longer applies on the 15
  high-violation boards (it remains in effect under a stock, capped `kicad-cli`, which therefore
  reproduces the set on all but those boards). The `d3_v2` guide boards now match the paper's on
  639 of 679; the guide step is byte-deterministic across reruns.
- The OrthoRoute baseline's ORP conversion runs with the locally resolved `pcbnew` interpreter;
  an interpreter that cannot import `pcbnew` is reported instead of silently worked around.
- Direct Python dependencies are trimmed to what this repository imports (`fastapi`, `uvicorn`,
  `requests`, `pydantic` are no longer direct dependencies; every pin in `requirements.txt` is
  unchanged) and `Notice.md` is reconciled with them.
- `.gitignore` folder patterns are anchored to the folder they mean (a bare `name/` no longer
  swallows a same-named folder elsewhere); `tools/docs/check_docs.py` gains a `gitignore-traps`
  check that fails when a tracked folder would silently ignore a new file.

### Fixed
- Flex attention: the same-net bias is applied as a `score_mod` instead of absorbing it into the
  q/k channels (head-dim blow-up and crash), and compilation is dynamic so chunk-size changes no
  longer exhaust the recompile limit.

### Removed
- The KDD recipes' `PROVENANCE.md` working record — internal notes, not part of the reproduction recipe.

## v1.0.0 — 2026-08-29

Initial public release.
