#!/usr/bin/env python3
"""Deterministic doc/code consistency checker for cadagent (Tier-1 doc-sync).

The doc-sync automation has three tiers (see `.claude/rules/docs.md`):
  Tier 1 — THIS script: mechanical checks that need no judgement. High trust,
           wired into `internal/githooks/pre-push`. Read-only — reports drift, never edits.
  Tier 2 — `/sync-docs` skill: LLM-assisted review of semantic drift (prose vs code).
  Tier 3 — `.claude/rules/docs.md`: the path-scoped contract that loads while editing.

Pure stdlib (no PyYAML/torch) so it runs in any env, incl. `cadagent-classical`.

Checks (run all by default; `--only <name>` to run one):
  md-links     — every local [text](path) link in live docs resolves to a real file/dir
  version-sync — pyproject.toml `version` == README.md <!--VERSION--> marker
  history      — internal/docs/HISTORY.md has a non-empty `## Unreleased` section (skipped
                 when the private tree is absent)
                 (empty is tolerated only right after a `## vX.Y.Z` version freeze)
  dead-paths   — removed path prefixes (DEAD_PATHS) appear in no live tracked file
  gitignore-traps — no tracked directory swallows a new file through an ancestor's ignore
                 rule (bare `name/` patterns match at every depth: anchor them as `/name/`)
  cpp-bump     — engine/kicad-patches/ (C++, rebuild) changed vs origin/develop
                 ⇒ the branch must carry a minor bump (push-batch rule, README.md
                 version-rules section; also invoked warn-only by engine/build_rl_router.sh)
  import-hygiene — the GPL/NC boundary: GPL imports only inside engine//
                 pcb_world/engine/ (dev-only in-process escape)/native test
                 bundle; no engine import outside the bundle; the bundle
                 imports nothing from the environment side
  wire-sync    — pcb_world/engine/wire.py == engine/engine_server/wire.py (byte-identical)
  public-hygiene — no public file imports or names anything under internal/ (names derived
                 from the tree), nor .claude/ / CLAUDE.md; no Korean; no identity/infra seed
                 (internal/hygiene_patterns.txt); dataset configs only under ${CADAGENT_DATA_ROOT}
  upstream-diff-sync — engine/docs/upstream-diff/ (generated KiCad-diff view) matches a fresh
                 regeneration byte for byte, and the build never names it
  test-durations — tests/durations.json (xdist scheduling seed, tests/conftest.py)
                 has an entry for every tests/**/test_*.py and no stale entries;
                 refresh with `pytest <paths> --update-durations`
  mirror-sync  — speed_profiler timed mirrors (hooks.py/worker_shim.py) are in
                 sync with their base functions: source digests pinned in
                 tools/diagnostics/speed_profiler/mirror_contract.py (static ast
                 extraction, no imports); on drift, re-sync the mirror then
                 `python -m tools.diagnostics.speed_profiler.mirror_contract`

Usage:
  python tools/docs/check_docs.py                 # all checks; exit 1 on any failure
  python tools/docs/check_docs.py --only md-links
  SKIP_DOC_CHECK=1 git push                       # emergency bypass of the pre-push gate
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Removed path/module prefixes that must not reappear in live tracked files.
# Grows when you rename something — the `/rename-symbol` skill appends here so a
# rename can never silently regress. Keep entries fully-qualified and unambiguous
# (a prefix that is also a substring of a LIVE path would false-positive).
#   (dead_prefix, hint_for_the_fix)
DEAD_PATHS: list[tuple[str, str]] = [
    # Pre-refactor env-layer package names; the env lives under pcb_world/.
    ("envs/gym_env", "use pcb_world/core (env.py, action.py, masking.py, action_schema.py)"),
    ("wrappers/common", "use pcb_world/vec (slots.py, candidate_pool.py)"),
    # The eval package drops the redundant eval_ prefix (see eval/__init__.py).
    ("eval_kicadpcb", "use eval/metrics.py — module eval.metrics (evaluate_one)"),
    ("eval/rollout/transformer.py", "renamed to eval/rollout/rl.py"),
    # The router-baselines layer is named "rule-based".
    ("methods/baselines/classical", "renamed to methods/baselines/rule_based"),
    ("run_classical_routers", "renamed to run_rule_based_routers (methods/baselines/rule_based/)"),
    ("eval/rollout/classical.py", "renamed to eval/rollout/rule_based.py"),
    (".claude/rules/classical.md", "renamed to .claude/rules/rule-based.md"),
    ("classical_routers", "pre-refactor top-level name; code lives in methods/baselines/rule_based"),
    # Dependency single-sourcing: one pinned pyproject + one compiled lock.
    ("requirements-all.txt", "replaced by requirements.txt (uv pip compile of the pinned pyproject)"),
    ("requirements-kicad-only.txt", "removed — one-shot install policy; use requirements.txt"),
    ("tools/serve/requirements.txt", "absorbed into the pinned requirements (private-only pins live beside the private tree)"),
    # NB: archived handoffs (internal/deprecated/internal/docs/handoff/) are NOT listed here — the
    # old path is a substring of the new (internal/deprecated/…) path, so a substring
    # DEAD_PATHS guard would false-positive on the valid new references. The
    # md-links check already enforces that moved links resolve.
    ("tests/crashtrace.c", "moved to pcb_world/diag/crashtrace.c (shared via pcb_world.diag)"),
    ("logs/crash", "cwd-relative default removed — crash artifacts live in var/crashlogs (pcb_world.diag.default_log_dir)"),
    # d3b train-set consolidation: the count-naming variants collapse to a single d3b_train.json.
    ("experiments/d2b_midboard/d3b_train277.json",
     "consolidated into internal/experiments/d2b_midboard/d3b_train.json (275 parseable; the 277 original is in git history)"),
    ("experiments/d2b_midboard/d3b_train275_parseable.json",
     "renamed to internal/experiments/d2b_midboard/d3b_train.json"),
    # Dataset-axis generalization: the d2a-only recipe becomes the --dataset d2a|d2b|d3b axis.
    ("experiments/d2b_midboard/train_d2a_timeblind.sh",
     "renamed to internal/experiments/d2b_midboard/train_timeblind.sh (--dataset axis)"),
    # Suite-speed split — a one-file compile guard serialized ~22s on a
    # single xdist worker; now one file per compile region under the package dir.
    ("tests/test_speed_knobs.py",
     "split into tests/test_speed_knobs/ (test_compile_{stack,decode,heads,encode}.py + test_bf16.py + test_cli_wiring.py)"),
    ("tools/diagnostics/speed_profiler/schema.py",
     "absorbed into tools/diagnostics/speed_profiler/instrument.py (fingerprint/stats/write_run)"),
    # Dead-output cleanup: the only display surface is waterfall.py — report/flops removed.
    ("tools/diagnostics/speed_profiler/report.py",
     "removed (superseded by waterfall.py — the single display surface)"),
    ("tools/diagnostics/speed_profiler/flops.py",
     "removed with the unread mfu/flops JSON blocks (no consumer after report.py removal)"),
    ("configs/datasets/misc/pcb_dataset_difficulty.json",
     "unused legacy split moved to internal/deprecated/configs/datasets/ (2026-07-17; dead boards_json defaults dropped to null)"),
    ("training/rl/", "moved to methods/rl_agent/training/ (use module path methods.rl_agent.training.*)"),
    ("scripts/paper_repro/", "consolidated into experiments/kdd/ (figure5_d1, table1_rl, ...)"),
    ("experiments/paper_repro/", "consolidated into experiments/kdd/ (e.g. experiments.kdd.figure5_t1)"),
    # Campaign nesting: flat KDD deliverables moved under experiments/kdd/.
    # Each prefix is the pre-move flat form; the live form inserts /kdd/, so these never
    # substring-match the new paths (experiments/figure5_t1 ⊄ experiments/kdd/figure5_t1).
    ("experiments/figure5_t1", "moved to experiments/kdd/figure5_d1 (module experiments.kdd.figure5_d1; task t1 -> d1)"),
    ("experiments/figure6_reward", "moved to experiments/kdd/figure6_reward"),
    ("experiments/table1_rl", "moved to experiments/kdd/table1_rl"),
    ("experiments/table1_llm", "moved to experiments/kdd/table1_llm"),
    ("experiments/table2/", "moved to experiments/kdd/table2/"),
    ("experiments/llm_eval", "moved to experiments/kdd/llm_eval"),
    ("experiments/t3_dataset", "moved to experiments/kdd/d3_dataset (task t3 -> d3)"),
    ("experiments/appendix_diagnostics", "moved to experiments/kdd/appendix_diagnostics"),
    ("experiments.figure5_t1", "moved to experiments.kdd.figure5_d1"),
    # RL codec/net lives under methods/rl_agent/models/v1/.
    # Fully qualified (rl_agent.*) so the live llm_agent codec twin is not matched.
    ("methods.rl_agent.policy.decoder_only_policy", "moved to methods.rl_agent.models.v1.net (KiCadRLModel) + .spec (action constants)"),
    ("methods.rl_agent.policy.token_vocabulary", "moved to methods.rl_agent.models.v1.{embedding (TokenVocabulary), spec (enums), encoding (encode_layer)}"),
    ("methods.rl_agent.policy.state_tokenizer_batched", "moved to methods.rl_agent.models.v1.tokenizer (BatchedStateTokenizer)"),
    ("methods.rl_agent.models.v1.encoder", "renamed to methods.rl_agent.models.v1.tokenizer (resolves the encoding/encoder near-dup)"),
    ("methods.rl_agent.policy.state_tokenizer_common", "moved to methods.rl_agent.models.v1.encoding (norm helpers + dataclasses)"),
    ("methods.rl_agent.policy.model", "moved to methods.rl_agent.models.v1.blocks (transformer blocks)"),
    ("methods.rl_agent.wrappers.state_converter", "moved to methods.rl_agent.models.v1.encoding (codec)"),
    ("methods.rl_agent.wrappers.action_converter", "moved to methods.rl_agent.models.v1.encoding (codec)"),
    ("methods.rl_agent.wrappers.mask", "moved to methods.rl_agent.models.v1.encoding (codec)"),
    # decoder_common split into buffer/collect/algorithms; trainer renamed to loop.
    ("methods.rl_agent.training.decoder_common", "carved into training/{buffer,collect} + algorithms/{ppo,grpo,_common} (C1-b); RunningRewardStd/RewardNormalizer/explained_variance/auto_device → training.utils, make_decoder_env* → wrappers.factory, resolve_board_list → _shared.board_loader"),
    ("methods.rl_agent.training.trainer", "renamed to methods.rl_agent.training.loop (RLTrainer/PPOTrainer/GRPOTrainer)"),
    # Vendored classical libs are relocated to external/.
    ("methods/baselines/classical/OrthoRoute", "moved to external/OrthoRoute (git submodule)"),
    ("methods/baselines/classical/freerouting", "moved to external/freerouting (auto-downloaded jar)"),
    # Viz cleanup — orphan V5 leftovers moved to internal/deprecated/gym_env/.
    ("env/trajectory/trajectory_replay", "moved to internal/deprecated/gym_env/trajectory_replay.py (orphan; TrajectoryLogger stays live)"),
    ("env.trajectory.trajectory_replay", "moved to deprecated.gym_env.trajectory_replay"),
    ("env/core/wrappers", "package removed — LoggingWrapper moved to internal/deprecated/gym_env/logging_wrapper.py"),
    ("env.core.wrappers", "package removed — use deprecated.gym_env.logging_wrapper"),
    # Viz naming is role-based: what it draws is rollouts/MDP state.
    ("tools/viz/visualize", "renamed to internal/tools/viz/rollout_viz.py"),
    # make_line mode verification: the manual main() script was ported to pytest.
    ("tests/integration/test_make_line_modes", "ported to tests/test_engine_api/test_make_line_modes.py (mark_obstacles=commit-reject verdict pinned)"),
    ("tools.viz.visualize", "renamed to internal.tools.viz.rollout_viz"),
    # xdist loadfile balance — a 17.5s single-file chain split per topic.
    ("test_drc_incremental_keying", "split into tests/test_drc_incremental/{test_keying,test_equivalence,test_restore}.py (shared helpers: tests/helpers/drc_keying.py)"),
    # llm_agent legacy-path sweep — old top-level LLM paths folded into methods/llm_agent/.
    # S1'-port: the GPL-side code is consolidated into the top-level engine/ bundle.
    ("external/kicad-patches", "moved to engine/kicad-patches (GPL bundle consolidation)"),
    ("external/kicad-python", "moved to engine/kicad-python (submodule path; .gitmodules updated)"),
    ("external/engine_server", "moved to engine/engine_server"),
    ("tools/build/build_rl_router", "moved to engine/build_rl_router.sh"),
    ("_dataset_prep", "moved to engine/pcbnew_prep (the scripts import GPL pcbnew)"),
    # release-structure P1 (2026-08-30): the GPL bundle takes the path the public tree
    # pins it at as a submodule, so the two trees agree byte-for-byte on every path.
    ("gpl_engine", "moved to engine/ (release-structure P1 — same path as the public engine submodule)"),
    # release-structure P4: the KiCad-diff tools belong to the engine (they read its tree)
    ("tools/build/make_kicad_patch_series", "moved to engine/tools/make_upstream_diff.sh (writes the generated view engine/docs/upstream-diff/)"),
    ("tools/build/diff_patches", "moved to engine/tools/diff_patches.sh"),
    ("envs/llm_env", "moved to methods/llm_agent/ (policy·rollout·training·wrappers)"),
    ("envs.llm_env", "moved to methods.llm_agent.* (eval_cadagent → methods.llm_agent.rollout.cadagent)"),
    ("policy/llm", "moved to methods/llm_agent/policy/"),
    ("training/llm", "moved to methods/llm_agent/training/"),
    ("eval/rollout/cadagent", "moved to methods/llm_agent/rollout/cadagent.py"),
    ("eval/rollout/apiseq", "moved to methods/llm_agent/rollout/apiseq.py"),
    ("eval_cadagent", "renamed to methods/llm_agent/rollout/cadagent.py (module methods.llm_agent.rollout.cadagent)"),
    # llm_agent slot consistency — orphan re-export shim removed (zero consumers).
    ("methods/llm_agent/wrappers/projection", "removed — import from wrappers/{state_converter,action_converter} directly"),
    ("methods.llm_agent.wrappers.projection", "removed — import methods.llm_agent.wrappers.{state_converter,action_converter} directly"),
    # docs/design/ carve — detailed design records split from top-level reference docs.
    ("docs/INCREMENTAL_DRC.md", "moved to docs/design/INCREMENTAL_DRC.md (summary: internal/docs/ENGINE.md §5)"),
    # Legacy masking YAML removal — 4 unloadable-by-file variants removed; the names still map via _NAME_COMPAT.
    ("configs/masking/strict.yaml", "removed — name-only clone of configs/masking/default.yaml (name 'strict' still maps via _NAME_COMPAT)"),
    ("configs/masking/strict_no_finish.yaml", "removed — name-only clone of configs/masking/default_no_finish.yaml (name still maps via _NAME_COMPAT)"),
    ("configs/masking/strict_phase.yaml", "removed — old phases: format (parser in internal/deprecated/gym_env/yaml_mask_legacy.py); name maps to 'default'"),
    ("configs/masking/relaxed_phase.yaml", "removed — old phases: format; name maps to 'relaxed'"),
    # Naming pass — singular→plural widget package; eval/rollout symmetry.
    ("gui_component/", "renamed to internal/tools/viz/interactive/gui_components/ (plural)"),
    # tools/interactive relocate — serve pulled up to a top-level
    # internal/tools/serve/; the REPL/GUI app nested under internal/tools/viz/interactive/ (viz =
    # all visualization tooling). Register serve-specific prefixes first.
    ("tools/interactive/serve", "moved to internal/tools/serve/"),
    ("tools.interactive.serve", "moved to internal.tools.serve"),
    ("tools/interactive", "moved to internal/tools/viz/interactive/ (serve → internal/tools/serve/)"),
    ("tools.interactive", "moved to internal.tools.viz.interactive (serve → internal.tools.serve)"),
    ("eval/stage1_rollout", "moved to eval/rollout/rl.py (rollout/ = one module per method family)"),
    ("eval.stage1_rollout", "moved to eval.rollout.rl"),
    # The top-level package env/ is now pcb_world/ — kills the venv-name
    # collision (.gitignore needed a `!/env/` carve-out) and the hyper-generic import name.
    ("env/core", "renamed to pcb_world/core"),
    ("env/engine", "renamed to pcb_world/engine"),
    ("env/vec", "renamed to pcb_world/vec"),
    ("env/rendering", "renamed to pcb_world/rendering"),
    ("env/trajectory", "renamed to pcb_world/trajectory"),
    ("env.core", "renamed to pcb_world.core"),
    ("env.engine", "renamed to pcb_world.engine"),
    ("env.vec", "renamed to pcb_world.vec"),
    ("env.rendering", "renamed to pcb_world.rendering"),
    ("env.trajectory", "renamed to pcb_world.trajectory"),
    # configs code/data split — importable code lives under configs/loader/;
    # configs/ top level is data-only (defaults/ datasets/ drc/ masking/ reward/ paths.yaml).
    ("configs/schema.py", "moved to configs/loader/schema.py"),
    ("configs.schema", "moved to configs.loader.schema"),
    ("configs/paths.py", "moved to configs/loader/paths.py"),
    ("configs.paths", "moved to configs.loader.paths"),
    ("configs/cli.py", "moved to configs/loader/cli.py"),
    ("configs.cli", "moved to configs.loader.cli"),
    ("from configs import", "load_config/merge_configs moved — use `from configs.loader import`"),
    # Test decoupling from internal/deprecated/ — the reference tokenizer became a test fixture.
    ("deprecated/decoder_v1", "state_tokenizer.py moved to tests/helpers/reference_tokenizer.py (frozen parity reference); decoder_v1/ removed"),
    ("deprecated.decoder_v1", "moved to tests.helpers.reference_tokenizer"),
    # quickstart campaign scoping: alias JSONs moved under kdd/.
    # Old paths lack the /kdd/ segment, so they never substring-match the new ones.
    ("configs/quickstart/models.json", "moved to configs/quickstart/kdd/models.json"),
    ("configs/quickstart/splits.json", "moved to configs/quickstart/kdd/splits.json"),
    # Root README casing — standard uppercase; version-marker source.
    ("readme.md", "renamed to README.md (root version-marker source; case-sensitive match)"),
    # Naming-audit accuracy renames.
    ("BoardInfoJson", "renamed to BoardStatic (held no JSON; aligns with the board_static obs section)"),
    ("GeometricObjects", "renamed to NetGeometry (per-net dynamic geometry; pairs with NetContext)"),
    ("pcb_world/engine/dataclasses", "renamed to pcb_world/engine/containers.py (named for content, not the stdlib decorator)"),
    ("pcb_world.engine.dataclasses", "renamed to pcb_world.engine.containers"),
    ("schema.RewardConfig", "renamed to RewardOverrides (kills the collision with the pcb_world.core.reward_config Protocol)"),
    ("CandType", "renamed to CandidateType (models/v1/spec.py; matches its spelled-out siblings)"),
    ("DRCHelper", "renamed to DRCUtils"),
    # Package relocations — vec backend stutter fix + orphan top-level search/ fold.
    ("pcb_world/vec/vec", "renamed to pcb_world/vec/backends (de-stutter; disambiguates from subproc_pool.py)"),
    ("pcb_world.vec.vec", "renamed to pcb_world.vec.backends"),
    ("search/mcts", "core moved to methods/_shared/mcts/ (branch-agnostic); mcts_compare.py -> methods/rl_agent/policy/"),
    ("search.mcts", "moved to methods._shared.mcts (compare: methods.rl_agent.policy.mcts_compare)"),
    # Visual tests are named for what they produce — artifacts are visual (SVG/PNG).
    ("tests/visible", "renamed to internal/tests/visual (+ pytest marker visible -> visual)"),
    ("tests.visible", "renamed to tests.visual"),
    ("mark.visible", "marker renamed to visual"),
    ("test_visible_scenarios", "renamed to internal/tests/visual/test_visual_scenarios.py"),
    ("EvaluationMetrics", "renamed to EvalSummary (eval/metrics.py; consumer-side summary over EvalResult)"),
    ("edatool", "pyproject distribution renamed to pcbworld (paper name)"),
    # There is no low/high-level split — the HL prefix no longer applies (was tied to the V5 env).
    ("KiCadHLEnv", "renamed to PCBWorld (via KiCadEnv; the HL/LL distinction no longer exists)"),
    ("HLActionDispatcher", "renamed to ActionDispatcher"),
    ("test_hl_env_checkpoint", "renamed to tests/test_env_checkpoint.py"),
    ("test_hl_env_incremental_restore", "renamed to tests/test_env_incremental_restore.py"),
    ("test_hl_env_wrapper_decoder", "renamed to tests/test_env_wrapper_decoder.py"),
    ("test_hl_unit", "renamed to tests/test_env/test_unit.py"),
    ("test_hl_integration", "renamed to tests/test_env/test_integration.py"),
    # This module builds the gym observation; the actual routing state lives
    # in pcb_world/engine/containers.py.
    ("pcb_world/core/state", "renamed to pcb_world/core/observation.py"),
    ("pcb_world.core.state", "renamed to pcb_world.core.observation"),
    # Paper alignment — the env class and dataset task IDs take the paper's
    # names (PCBWorld; d1/d2a/d3*). var/results/kdd keeps the old t-series
    # dirs on disk (read-only) — loaders read-alias them; only NEW outputs use d*.
    ("KiCadEnv", "renamed to PCBWorld (paper environment name; pcb_world/core/env.py)"),
    ("experiments/kdd/figure5_t1", "renamed to experiments/kdd/figure5_d1 (task t1 -> d1)"),
    ("experiments/kdd/t3_dataset", "renamed to experiments/kdd/d3_dataset (task t3 -> d3)"),
    ("paper_repro_load_t1_grid_case", "renamed to paper_repro_load_d1_grid_case"),
    ("fig6c_t1_cleanpass", "renamed to experiments/_lib/metrics/fig6c_d1_cleanpass.py"),
    ("table23_t1_gridsweep", "output table renamed to table23_d1_gridsweep"),
    ("test_t1_grid_scenarios_hybrid", "renamed to tests/test_d1_grid_scenarios_hybrid.py"),
    # Dataset config folder + classical eval dir naming: splits→datasets, t3→d3.
    ("configs/splits/", "renamed to configs/datasets/ (split-config folder split→datasets)"),
    ("t3_pcb_dataset", "renamed to configs/datasets/d3.json (t3→d3)"),
    ("methods/baselines/classical/eval/T2", "renamed to methods/baselines/rule_based/eval/d2a (T2→d2a)"),
    ("methods/baselines/classical/eval/T3-A", "renamed to methods/baselines/rule_based/eval/d3a (T3-A→d3a)"),
    # Dataset config file naming — filename = logical id; flat layout → grids//misc//local/.
    ("d3_pcb_dataset", "renamed to configs/datasets/d3.json"),
    # Paper alignment — LLM cell prefixes take the paper's mode names
    # (interactive / plan_only / engine_free); the open-loop category
    # replaces "one-shot". Existing var/ cells + NFS response/cache dirs keep
    # the legacy prefixes (read-aliased); guards are fully-qualified so the
    # legacy-compat literals (apiseq_<split>_zs_*, cache paths) don't match.
    ("methods/llm_agent/rollout/apiseq", "renamed to methods/llm_agent/rollout/plan_only.py"),
    ("methods.llm_agent.rollout.apiseq", "renamed to methods.llm_agent.rollout.plan_only"),
    ("eval_apiseq_llm_v8_standalone", "renamed to experiments/kdd/llm_eval/eval_plan_only_llm_v8_standalone.py"),
    ("eval_cadgen_llm_v3_standalone", "renamed to experiments/kdd/llm_eval/eval_engine_free_llm_v3_standalone.py"),
    ("prepare_apiseq_fewshot", "renamed to experiments/kdd/llm_eval/prepare_plan_only_fewshot[_llm].py"),
    ("run_apiseq_llm_v8_standalone", "renamed to experiments/kdd/table1_llm/baselines/run_plan_only_llm_v8_standalone.sh"),
    ("run_cadgen_llm_v3_standalone", "renamed to experiments/kdd/table1_llm/baselines/run_engine_free_llm_v3_standalone.sh"),
    ("table2/run_pcbworld.sh", "renamed to experiments/kdd/table2/run_interactive.sh"),
    ("table2/run_apilevel.sh", "renamed to experiments/kdd/table2/run_plan_only.sh"),
    ("table2/run_codelevel.sh", "renamed to experiments/kdd/table2/run_engine_free.sh"),
    ("fig9_oneshot", "renamed to experiments/_lib/metrics/fig9_openloop.py (open-loop category)"),
    ("apiseq_responses", "paths.yaml key renamed to plan_only_responses (path value unchanged)"),
    # The pygame frontend is removed — the interactive GUI is tkinter-only;
    # the toolkit-agnostic orchestrator is gui.py -> session.py.
    ("tools/viz/interactive/gui.py", "split: Session -> internal/tools/viz/interactive/session.py, widgets -> internal/tools/viz/interactive/tk/"),
    ("tools.viz.interactive.gui", "renamed to internal.tools.viz.interactive.session (widgets: internal.tools.viz.interactive.tk)"),
    ("tools/viz/interactive/gui_components", "removed with the pygame frontend (tk widgets: internal/tools/viz/interactive/tk/)"),
    ("tools.viz.interactive.gui_components", "removed with the pygame frontend (tk widgets: internal.tools.viz.interactive.tk)"),
    ("tools/viz/interactive/load_modal", "removed with the pygame frontend (tk: internal/tools/viz/interactive/tk/modals.py TkLoadModal)"),
    ("tools/viz/interactive/mode_modal", "removed with the pygame frontend (tk: internal/tools/viz/interactive/tk/modals.py TkModeModal)"),
    ("--toolkit", "removed — the interactive GUI has a single (tkinter) frontend"),
]

# Pathspecs excluded from the dead-paths scan: archives + files that legitimately
# record the old names (changelog, refactor map, this tooling, the rules/skills).
DEAD_PATH_EXCLUDES = [
    ":!internal/deprecated",
    # ONLY the third-party submodule mount points — never the `external/` prefix.
    # `external/README.md`, `external/patcher.sh` and `external/*-patch/**` are OUR
    # files and they ship, so they must stay in scope.
    ":!external/OrthoRoute",
    ":!external/RAGEN",
    ":!external/verl-agent",
    # vendored upstream full copies (+ our C++ patch forks) — same rationale as
    # the submodule mount points; e.g. upstream CMakeLists legitimately contains '--toolkit'.
    ":!engine/kicad-patches",
    ":!var",
    ":!build_rl",
    ":!internal/docs/HISTORY.md",
    ":!internal/docs/DEPRECATED.md",
    ":!internal/docs/REFACTOR_PLAN.md",
    ":!internal/docs/handoff",
    ":!internal/docs/releases",   # generated per-release records quoting commit subjects verbatim
    ":!internal/dead_paths.txt",  # the private dead-path list names the dead strings by definition
    ":!tools/docs",
    # auto-generated map of LIVE collected test files (a live filename may
    # embed a dead-path substring, e.g. test_eval_cadagent_canonical.py ⊃
    # 'eval_cadagent'); ghost entries are caught by check_test_durations.
    ":!tests/durations.json",
    ":!internal/githooks",
    ":!.claude/rules/docs.md",
    ":!.claude/skills",
]

# Directories excluded from the markdown link scan (vendored / archived / generated).
# Under external/ only the submodule mount points are vendored — our own
# external/README.md and external/*-patch/** stay in scope.
MD_EXCLUDE_DIRS = ("internal/deprecated/", "external/OrthoRoute/", "external/RAGEN/",
                   "external/verl-agent/", "var/", "build_rl/")

# Docs that legitimately reference removed paths (point-in-time changelog +
# old→new maps) — their links are *expected* to go stale, so skip link-checking them.
MD_LINK_EXCLUDE_FILES = {"internal/docs/HISTORY.md", "internal/docs/DEPRECATED.md", "internal/docs/REFACTOR_PLAN.md"}

# Binary/asset link targets are out of scope — the checker guards doc/code
# cross-references, not committed images/PDFs (which rarely move with code).
ASSET_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf", ".ico")

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def _c(s: str, color: str) -> str:
    return f"{color}{s}{RESET}" if sys.stdout.isatty() else s


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def git_tracked_md(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "*.md", "*.markdown"],
        capture_output=True, text=True, check=True,
    )
    files = []
    for rel in out.stdout.splitlines():
        if rel and not rel.startswith(MD_EXCLUDE_DIRS):
            files.append(root / rel)
    return files


# --------------------------------------------------------------------------- #
# Checks — each returns a list of human-readable failure strings (empty == pass)
# --------------------------------------------------------------------------- #

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def check_md_links(root: Path) -> list[str]:
    failures: list[str] = []
    for md in git_tracked_md(root):
        if md.relative_to(root).as_posix() in MD_LINK_EXCLUDE_FILES:
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        # Strip code (fenced ``` blocks + inline `spans`) so example link syntax
        # shown as documentation isn't mistaken for a real link.
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`[^`\n]*`", "", text)
        for m in _LINK_RE.finditer(text):
            target = m.group(1).strip().split()[0].strip("<>")  # drop optional "title"
            if (not target or target.startswith(("#", "http://", "https://", "mailto:"))
                    or "://" in target):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:  # pure in-page anchor
                continue
            if path_part.lower().endswith(ASSET_EXTS):  # images/PDFs out of scope
                continue
            resolved = (md.parent / path_part).resolve()
            if not resolved.exists():
                rel = md.relative_to(root)
                failures.append(f"{rel}: broken link → {target}")
    return failures


def check_version_sync(root: Path) -> list[str]:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    m_py = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if not m_py:
        return ["pyproject.toml: no `version = \"...\"` field found"]
    py_ver = m_py.group(1).lstrip("v")

    readme = (root / "README.md").read_text(encoding="utf-8")
    m_rd = re.search(r"<!--VERSION-->(.*?)<!--/VERSION-->", readme)
    if not m_rd:
        return ["README.md: missing <!--VERSION-->...<!--/VERSION--> marker "
                "(needed so bump_version.py can keep it in sync with pyproject.toml)"]
    rd_ver = m_rd.group(1).strip().lstrip("v")

    if py_ver != rd_ver:
        return [f"version mismatch: pyproject.toml={py_ver} vs README.md={rd_ver} "
                f"(run `python internal/tools/docs/bump_version.py` or sync by hand)"]
    return []


def check_history(root: Path) -> list[str]:
    hist = root / "internal" / "docs" / "HISTORY.md"
    if not hist.exists():
        return []   # public clone: the changelog is not distributed
    text = hist.read_text(encoding="utf-8")
    m = re.search(r"\n##\s+Unreleased\s*\n", text)
    if not m:
        return ["internal/docs/HISTORY.md: no `## Unreleased` section found"]
    rest = text[m.end():]
    nxt = re.search(r"\n##\s", rest)
    body = rest[: nxt.start()] if nxt else rest
    if not re.search(r"(^|\n)\s*(-|\*|###)\s", body):
        # An empty `## Unreleased` is allowed only right after a version freeze
        # (bump_version.py stamps the batch into `## vX.Y.Z — DATE` and leaves a
        # fresh empty Unreleased on top); a `## vX.Y.Z` section following it marks
        # that clean-slate state. Otherwise it's a genuine "you forgot to log" miss.
        if nxt and re.match(r"\n##\s+v\d", rest[nxt.start():]):
            return []
        return ["internal/docs/HISTORY.md: `## Unreleased` section is empty "
                "(add an entry for the latest change — see README.md, change-log section)"]
    return []


def _private_dead_paths(root: Path) -> list[tuple[str, str]]:
    """Dead paths whose NAME is itself an identity seed (personal script names, site labels) live
    in the private tree — this file ships publicly and is exempt from the hygiene scan, so a seed
    written here would leave with the export. Format: `dead<TAB>hint`, `#` comments."""
    f = root / PRIVATE_DEAD_PATHS
    if not f.exists():
        return []
    out = []
    for ln in f.read_text().splitlines():
        if ln.strip() and not ln.startswith("#") and "\t" in ln:
            dead, hint = ln.split("\t", 1)
            out.append((dead.strip(), hint.strip()))
    return out


def check_dead_paths(root: Path) -> list[str]:
    # ONE combined `git grep` over every dead-path pattern instead of one
    # subprocess per pattern. On large/slow working trees the per-invocation
    # tree walk dominates (~2s each × 100+ patterns ≈ minutes); folding them
    # into a single `-e … -e …` scan cuts wall time ~60x. Attribution back to
    # each pattern is recovered from the matched line's content (`-F` ⇒ literal
    # substring), so the per-dead-path hint output is unchanged.
    dead_paths = DEAD_PATHS + _private_dead_paths(root)
    patterns = [dead for dead, _ in dead_paths]
    grep_args = ["git", "-C", str(root), "grep", "-nIF"]
    for dead in patterns:
        grep_args += ["-e", dead]
    grep_args += ["--", *DEAD_PATH_EXCLUDES]

    proc = subprocess.run(grep_args, capture_output=True, text=True)
    if proc.returncode == 1:   # 1 == no matches == clean
        return []
    if proc.returncode > 1:    # real git error
        return [f"git grep failed for dead paths: {proc.stderr.strip()}"]

    # rc == 0: at least one hit. Bucket every matched line under each dead path
    # its *content* literally contains (matching the path prefix would misfile a
    # line whose path merely happens to embed another pattern). A line may trip
    # more than one pattern → report it under all, as the per-pattern loop did.
    hits_by_dead: dict[str, list[str]] = {dead: [] for dead in patterns}
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)          # <path>:<lineno>:<content>
        content = parts[2] if len(parts) == 3 else line
        for dead in patterns:
            if dead in content:
                hits_by_dead[dead].append(line)

    failures: list[str] = []
    for dead, hint in dead_paths:
        hits = hits_by_dead[dead]
        if hits:
            failures.append(f"dead path '{dead}' → {hint}")
            failures.extend(f"    {h}" for h in hits)
    return failures


# --------------------------------------------------------------------------- #
# import-hygiene: the GPL/NC boundary gate (internal/docs/ENGINE.md, engine IPC mode section).
# engine/ is the single top-level GPL bundle (S1'-form consolidation):
# every component that links or imports GPL KiCad code lives there
# (kicad-python submodule, kicad-patches full copies, engine_server/,
# pcbnew_prep/, build_rl_router.sh). Three machine-checked rules keep the
# boundary real inside the dev tree:
#
#  (1) GPL python modules (kicad_rl_router / pcbnew) may be imported ONLY by:
#      - engine/       : the GPL-side bundle (engine server, pcbnew prep)
#      - pcb_world/engine/ : the sole NC-side loader — a DEV-ONLY escape
#                            hatch (`KICAD_ENGINE_IPC=0` in-process mode);
#                            default IPC mode never executes those imports,
#                            and the release split (S4a) deletes them.
#      - tests/test_engine_api/, tests/stress/, internal/tests/visual/ : the
#                            engine-direct (native) test bundle — runs the
#                            binding in-process by design.
#  (2) Nothing outside engine/ imports engine internals. The IPC
#      client (pcb_world/engine/router_client.py) is the only bridge, and it
#      SPAWNS the server by file path — it never imports it.
#  (3) Python files inside engine/ import nothing from the environment side:
#      the bundle is self-contained (the wire schema is a byte-identical copy
#      on both sides — the `wire-sync` check below — and the thread-pool cap
#      and crash diagnostics are bundle-local), so the same bytes also work
#      as the standalone engine repository.
#
# Everything else must stay import-clean so the NC release evidence
# (zero GPL-import grep hits) holds by construction.
# --------------------------------------------------------------------------- #

GPL_BUNDLE_PREFIX = "engine/"

IMPORT_HYGIENE_ALLOWED_PREFIXES = (
    GPL_BUNDLE_PREFIX,
    "pcb_world/engine/",
    "tests/test_engine_api/",
    "tests/stress/",
    "internal/tests/visual/",
)

# Environment-side modules the GPL bundle may import (rule 3): none — the bundle
# is self-contained (wire.py ships as a byte-identical copy on both sides,
# thread_cap.py / crash_diag.py are bundle-local) so the same bytes work as the
# standalone engine repository.
IMPORT_HYGIENE_BUNDLE_ENV_ALLOWED: tuple[str, ...] = ()

_GPL_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:kicad_rl_router|pcbnew)\b")
_BUNDLE_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+engine\b")
_ENV_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(pcb_world[.\w]*)\s+import|import\s+(pcb_world[.\w]*))")


def _env_module_allowed(module: str) -> bool:
    return any(module == a or module.startswith(a + ".")
               for a in IMPORT_HYGIENE_BUNDLE_ENV_ALLOWED)


def check_import_hygiene(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "*.py"],
        capture_output=True, text=True, check=True)
    failures: list[str] = []
    for rel in proc.stdout.splitlines():
        inside_bundle = rel.startswith(GPL_BUNDLE_PREFIX)
        gpl_import_ok = rel.startswith(IMPORT_HYGIENE_ALLOWED_PREFIXES)
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if not gpl_import_ok and _GPL_IMPORT_RE.match(line):
                failures.append(
                    f"{rel}:{lineno}: GPL import outside the boundary "
                    f"({line.strip()}) — go through pcb_world.engine, or "
                    "move the file into engine/ or the native test dirs")
            if not inside_bundle and _BUNDLE_IMPORT_RE.match(line):
                failures.append(
                    f"{rel}:{lineno}: engine internals imported outside "
                    f"the bundle ({line.strip()}) — the IPC client "
                    "(pcb_world/engine/router_client.py) is the only bridge; "
                    "it spawns the server by path, never by import")
            if inside_bundle:
                m = _ENV_IMPORT_RE.match(line)
                if m:
                    module = m.group(1) or m.group(2)
                    if not _env_module_allowed(module):
                        failures.append(
                            f"{rel}:{lineno}: GPL bundle imports a "
                            f"non-sanctioned environment module ({line.strip()})"
                            " — allowed: "
                            + ", ".join(IMPORT_HYGIENE_BUNDLE_ENV_ALLOWED))
    return failures



# --------------------------------------------------------------------------- #
# public-hygiene: the public/private boundary (internal/docs/handoff/260830_release_structure.md §3).
# Everything outside internal/ and the two root exclusions is the PUBLIC SET — the bytes
# a release exports as-is — so no public file may import or name anything that stays
# behind. The forbidden names are derived from the tree, not maintained by hand: every
# path under internal/ whose public mirror does not exist (internal/tools/viz ↔ no
# tools/viz) is a private leaf, and its slash form, dotted-module form and relative-
# import form are forbidden in public files. Adding a file under internal/ extends the
# guard automatically. Files under internal/ may reference the public tree freely.
PRIVATE_ROOT = "internal/"
ROOT_EXCLUDED = (".claude/", "CLAUDE.md", "CLAUDE.local.md")   # dev-only, never exported
# Skipped: the two boundary checkers (their tables name private paths on purpose) and the
# generated durations seed. Everything else in the public set is scanned.
PUBLIC_HYGIENE_SKIP = (
    "tools/docs/check_docs.py",
    "tools/check_separation.py",     # names the excluded roots by definition (its skip list)
    "tests/durations.json",
)
# Rule 3 seeds live in the private tree (identity / infrastructure regexes; `allow:` literals
# are exempt). Absent in a public clone → rule 3 is skipped there.
HYGIENE_PATTERNS = PRIVATE_ROOT + "hygiene_patterns.txt"
PRIVATE_DEAD_PATHS = PRIVATE_ROOT + "dead_paths.txt"   # dead paths whose name is a seed (see _private_dead_paths)
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]")
_HYGIENE_BINARYISH = {".kicad_pcb", ".kicad_dru", ".kicad_pro", ".png", ".jpg", ".pdf", ".pt", ".npz",
                      ".npy", ".zip", ".gz", ".pyc", ".ico", ".svg", ".mp4", ".gif", ".ttf", ".woff", ".woff2"}


def _private_forms(root: Path) -> dict[str, str]:
    """{regex: what it names} for the literal roots + every private leaf under internal/."""
    forms = {re.escape(PRIVATE_ROOT): "the private root",
             re.escape(".claude/"): "Claude Code working files (never exported)",
             r"(?<![\w/])CLAUDE(?:\.local)?\.md\b": "the Claude Code instructions (never exported)"}
    ls = subprocess.run(["git", "-C", str(root), "ls-files", "--", PRIVATE_ROOT],
                        capture_output=True, text=True)
    leaves: dict[str, bool] = {}
    for rel in ls.stdout.splitlines():
        parts = rel.split("/")[1:]
        for i in range(1, len(parts) + 1):
            cand = "/".join(parts[:i])
            if not (root / cand).exists():
                leaves[cand] = i == len(parts)
                break
    for leaf, is_file in sorted(leaves.items()):
        what = PRIVATE_ROOT + leaf
        if is_file:
            stem = leaf[:-3] if leaf.endswith(".py") else leaf
            forms[r"(?<![\w/])" + re.escape(stem) + r"(?![\w-])"] = what
            if leaf.endswith(".py"):
                forms[r"(?<![\w.])" + re.escape(stem.replace("/", ".")) + r"\b"] = what
                base = stem.rsplit("/", 1)[-1]
                forms[r"from\s+\.+(?:[\w.]*\.)?" + re.escape(base) + r"\b"] = what
                if base.startswith("_"):      # private-by-convention names: forbid the bare stem too
                    forms[r"(?<![\w/.])" + re.escape(base) + r"\b"] = what
        elif "/" in leaf:
            forms[r"(?<![\w/])" + re.escape(leaf) + r"(?![\w-])"] = what
            forms[r"(?<![\w.])" + re.escape(leaf.replace("/", ".")) + r"\b"] = what
        else:                                  # single-component dir: require the slash (plain word otherwise)
            forms[r"(?<![\w/])" + re.escape(leaf) + "/"] = what
    return forms


def _identity_seeds(root: Path) -> tuple[re.Pattern | None, list[str]]:
    seeds, allow = [], []
    p = root / HYGIENE_PATTERNS
    if not p.exists():
        return None, allow
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("allow:"):
            allow.append(line[len("allow:"):].strip())
        else:
            seeds.append(line)
    return (re.compile("|".join(f"(?:{s})" for s in seeds)) if seeds else None), allow


def check_public_hygiene(root: Path) -> list[str]:
    if not (root / PRIVATE_ROOT).is_dir():
        return []                              # public clone: nothing private to name
    forms = _private_forms(root)
    regex = re.compile("|".join(f"(?P<f{i}>{pat})" for i, pat in enumerate(forms)))
    names = list(forms.values())
    ident, allow = _identity_seeds(root)
    proc = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True)
    failures: list[str] = []
    for rel in proc.stdout.splitlines():
        if rel.startswith(PRIVATE_ROOT) or rel.startswith(ROOT_EXCLUDED) or rel.startswith(PUBLIC_HYGIENE_SKIP):
            continue
        if ident is not None and ident.search(rel):
            failures.append(f"{rel}: file NAME matches an identity/infra seed (rule 3) — rename or move it under internal/")
        if Path(rel).suffix in _HYGIENE_BINARYISH:
            continue
        try:
            path = root / rel
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        korean = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in regex.finditer(line):
                idx = int(m.lastgroup[1:])
                failures.append(f"{rel}:{lineno}: public file names private path "
                                f"'{m.group(0)}' ({names[idx]}) — public code must not import or "
                                "reference internal/; move the consumer under internal/ or cut the dependency")
            if _HANGUL_RE.search(line):
                korean += 1
            if ident is not None:
                probe = line
                for lit in allow:
                    probe = probe.replace(lit, "")
                hit = ident.search(probe)
                if hit:
                    failures.append(f"{rel}:{lineno}: identity/infra seed '{hit.group(0)}' (rule 3 — the "
                                    "public set carries no internal names, hosts or paths; see "
                                    f"{HYGIENE_PATTERNS})")
        if korean:
            # Rule 2 of the boundary: the public set is English-only (code comments, strings,
            # and the docs that ship). Korean prose belongs under internal/ (docs, handoff).
            failures.append(f"{rel}: {korean} line(s) contain Korean — the public set is "
                            "English-only; translate, or move the file under internal/")
    failures += _check_config_data_roots(root)
    return failures


_DATA_ROOT_KEYS = ("dataset_dirs", "source_dir", "source_csv")


def _check_config_data_roots(root: Path) -> list[str]:
    """Rule 4: shipped dataset configs name locations only as ${CADAGENT_DATA_ROOT}/<sub>
    (or repo-relative paths) — never an absolute path, and paths.yaml subs are relative."""
    import json
    failures: list[str] = []
    for p in sorted((root / "configs" / "datasets").rglob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        def walk(node, key=None):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, k)
            elif isinstance(node, list):
                for v in node:
                    walk(v, key)
            elif isinstance(node, str) and key in _DATA_ROOT_KEYS or (
                    isinstance(node, str) and key in ("train", "val", "test") and node.startswith("/")):
                if node.startswith("/"):
                    failures.append(f"{p.relative_to(root)}: absolute dataset path under '{key}' "
                                    f"({node[:60]}…) — use ${{CADAGENT_DATA_ROOT}}/<sub> (rule 4)")
        walk(doc)
    yaml_p = root / "configs" / "paths.yaml"
    if yaml_p.exists():
        for lineno, line in enumerate(yaml_p.read_text(encoding="utf-8").splitlines(), 1):
            m = re.search(r"\{sub:\s*(/[^}\s]*)", line)
            if m:
                failures.append(f"configs/paths.yaml:{lineno}: absolute sub {m.group(1)} — subs are relative to the data root (rule 4)")
    return failures


# wire-sync: the engine-IPC wire schema ships as one file present on BOTH sides
# of the GPL boundary — pcb_world/engine/wire.py and engine/engine_server/wire.py —
# so neither program imports a module from the other. The copies must stay byte
# identical; edit one and copy it over the other.
WIRE_COPIES = ("pcb_world/engine/wire.py", "engine/engine_server/wire.py")


def check_wire_sync(root: Path) -> list[str]:
    a, b = (root / p for p in WIRE_COPIES)
    if not b.exists():
        # A tree without the engine checked out (the public tree pins it as a
        # submodule) has nothing to compare; tools/check_separation.py covers it.
        return []
    if a.read_bytes() != b.read_bytes():
        return [f"{WIRE_COPIES[0]} and {WIRE_COPIES[1]} differ — the wire schema is a "
                "byte-identical copy on both sides of the GPL boundary; copy the edited "
                "one over the other"]
    return []



# upstream-diff-sync: engine/docs/upstream-diff/ is a GENERATED read-only view of what the
# engine changes in KiCad (engine/tools/make_upstream_diff.sh). The full copies under
# engine/kicad-patches/kicad/ are the only source and the only build input; the view must
# match them byte for byte, and the build must never read it (no fallback, ever).
def check_upstream_diff_sync(root: Path) -> list[str]:
    eng = root / "engine"
    gen, view = eng / "tools" / "make_upstream_diff.sh", eng / "docs" / "upstream-diff"
    if not gen.exists() or not (eng / "kicad-python" / "kicad" / "pcbnew").is_dir():
        return []   # engine absent (public clone) or upstream not checked out: nothing to compare
    failures: list[str] = []
    for f in [eng / "build_rl_router.sh", *sorted((eng / "kicad-patches").rglob("CMakeLists.txt"))]:
        if "upstream-diff" in f.read_text(encoding="utf-8", errors="replace"):
            failures.append(f"{f.relative_to(root)}: names docs/upstream-diff — the build must never read "
                            "the generated view (no fallback)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "view"
        proc = subprocess.run(["bash", str(gen), str(out)], capture_output=True, text=True)
        if proc.returncode != 0:
            return failures + [f"engine/tools/make_upstream_diff.sh failed: {proc.stderr.strip()[-300:]}"]
        want = {q.relative_to(out).as_posix(): q.read_bytes() for q in out.rglob("*") if q.is_file()}
        have = {q.relative_to(view).as_posix(): q.read_bytes() for q in view.rglob("*") if q.is_file()} if view.is_dir() else {}
        stale = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
        if stale:
            failures.append(f"engine/docs/upstream-diff/ is stale ({len(stale)} file(s): {', '.join(stale[:3])}"
                            f"{'…' if len(stale) > 3 else ''}) — regenerate: bash engine/tools/make_upstream_diff.sh")
    return failures


# C++ patch tree whose changes mandate a MINOR bump (same rule as bump_version.py).
CPP_PREFIX = "engine/kicad-patches/"

_PYPROJECT_VER_RE = re.compile(
    r'^version\s*=\s*"v?(\d+)\.(\d+)\.(\d+)(\+b\d+)?"', re.MULTILINE)


def _pyproject_minor(text: str) -> tuple[int, int] | None:
    m = _PYPROJECT_VER_RE.search(text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_cpp_bump(root: Path) -> list[str]:
    """C++ change under engine/kicad-patches/ ⇒ minor bump (README.md, version rules).

    Push-batch semantics (same base as check_version_bumped.py): fails only when
    the branch changes CPP_PREFIX relative to origin/develop (merge-base diff or
    uncommitted changes) while neither HEAD nor the working-tree pyproject carries
    a (major, minor) ahead of the FORK POINT's (merge-base, not develop's tip —
    the local number line ignores develop's parallel advancement). One minor bump per branch batch —
    further C++ edits/rebuilds after the bump stay green. Passes silently when
    origin/develop is unresolvable (fresh clone / detached fetch state) — this
    gate must never block unrelated workflows. Detect-and-remind only — never
    bumps anything itself.
    """
    if not (root / "internal").is_dir():
        return []   # public clone: a develop-workflow gate (the version tooling lives in internal/)

    def git(*a: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *a],
                              capture_output=True, text=True)

    wt_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    wt_m = _PYPROJECT_VER_RE.search(wt_text)
    if wt_m is None:
        return ["pyproject.toml: no parsable `version = \"X.Y.Z[+bN]\"` field"]
    if wt_m.group(4):
        # Personal branch line (vX.Y.Z+bN): release digits are frozen at the
        # fork, so the C++-minor decision is DEFERRED to the MR rebaseline —
        # `bump_version.py --mr` inspects the whole branch batch and picks
        # minor there (bump_version.py detect_level). Nothing to enforce here.
        return []
    wt = _pyproject_minor(wt_text)
    # Version base = merge-base with origin/develop (fork point), not develop's
    # tip — the branch's local number line ignores develop's parallel
    # advancement (check_version_bumped.py docstring; docs.md, bump-timing section).
    mb = git("merge-base", "origin/develop", "HEAD")
    base_ref = mb.stdout.strip() if mb.returncode == 0 and mb.stdout.strip() else "origin/develop"
    base_show = git("show", f"{base_ref}:pyproject.toml")
    if base_show.returncode != 0:  # base unresolvable — never block
        return []
    base = _pyproject_minor(base_show.stdout)
    if base is None:
        return []
    head_show = git("show", "HEAD:pyproject.toml")
    head = _pyproject_minor(head_show.stdout) if head_show.returncode == 0 else None
    if wt > base or (head is not None and head > base):
        return []  # minor (or major) bump already carried by this batch

    failures: list[str] = []
    hint = "run `python internal/tools/docs/bump_version.py --minor --apply` (C++ change = minor, rebuild)"
    # ENGINE_VERSION is the major.minor stamp bump_version.py itself rewrites
    # (patch bumps included, e.g. when flooring at a develop that minor-advanced
    # elsewhere) — it is metadata, never a C++ change.
    stamp = CPP_PREFIX + "ENGINE_VERSION"
    # Exact (100%) renames are excluded: a pure relocation of the patch tree
    # (e.g. external/kicad-patches → engine/kicad-patches, the S1' bundle
    # consolidation) changes no C++ content, so the built engine identity is
    # unchanged and no minor bump is owed. Rename detection needs the WHOLE
    # diff (a pathspec would hide the delete side and turn renames into adds),
    # so the prefix filter happens in Python.
    committed = [ln for ln in git("diff", "--name-only", "--find-renames=100%",
                                  "--diff-filter=r",
                                  "origin/develop...HEAD").stdout.splitlines()
                 if ln.startswith(CPP_PREFIX) and ln.strip() != stamp]
    if committed:
        failures.append(f"{CPP_PREFIX} changed vs the develop fork point (v{base[0]}.{base[1]}.x) in "
                        f"{len(committed)} file(s) but the branch carries no minor bump — {hint}")
        failures.extend(f"    {p}" for p in committed)
    dirty_tracked = [ln for ln in git("diff", "HEAD", "--name-only",
                                      "--find-renames=100%",
                                      "--diff-filter=r").stdout.splitlines()
                     if ln.startswith(CPP_PREFIX) and ln.strip() != stamp]
    untracked = [ln for ln in git("ls-files", "--others", "--exclude-standard",
                                  "--", CPP_PREFIX).stdout.splitlines()
                 if ln.strip()]
    dirty = dirty_tracked + untracked
    if dirty:
        failures.append(f"{CPP_PREFIX} has uncommitted changes and no minor bump is pending "
                        f"vs origin/develop — before pushing, {hint}")
    return failures


def check_test_durations(root: Path) -> list[str]:
    """tests/durations.json (xdist scheduling seed) covers every test file.

    tests/conftest.py orders files longest-first for xdist from this tracked
    seed; a missing entry schedules the file as instantly-fast, a stale entry is
    rot. Refresh = `pytest <paths> --update-durations` (merges files that ran,
    prunes deleted). Files pytest never runs by default (fully `deprecated`-
    marked, or scripts that collect no tests) keep a manual 0.0 entry.
    """
    import json

    seed_path = root / "tests" / "durations.json"
    try:
        seed = set(json.loads(seed_path.read_text(encoding="utf-8")))
    except OSError:
        return ["tests/durations.json missing — regenerate: `pytest tests/ --update-durations`"]
    except ValueError as e:
        return [f"tests/durations.json unparsable ({e}) — regenerate: `pytest tests/ --update-durations`"]

    fs: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root / "tests"):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = [d for d in dirnames
                       if d != "__pycache__" and not (rel_dir == "tests" and d == "deprecated")]
        fs.update(f"{rel_dir}/{fn}" for fn in filenames
                  if fn.startswith("test_") and fn.endswith(".py"))

    failures: list[str] = []
    missing = sorted(fs - seed)
    if missing:
        failures.append("tests/durations.json: missing entries (new test file?) — run "
                        "`pytest <file> --update-durations`; for deprecated-only/"
                        "non-collected files add a manual 0.0 entry")
        failures.extend(f"    {m}" for m in missing)
    stale = sorted(seed - fs)
    if stale:
        failures.append("tests/durations.json: entries for deleted files — "
                        "`pytest tests/ --update-durations` prunes them")
        failures.extend(f"    {s}" for s in stale)
    return failures


def check_mirror_sync(root: Path) -> list[str]:
    """speed_profiler timed mirrors: every base function unchanged since the
    last mirror re-sync.

    hooks.py / worker_shim.py copy base control flow with timers added; a base
    edit without a mirror re-sync makes the profiler measure a stale path
    (precedent: a stale digest once stayed red for two days).
    mirror_contract extracts base source statically (ast — stdlib-only), so
    this runs in any env, matching this script's contract.
    """
    import importlib.util

    mc_path = root / "tools" / "diagnostics" / "speed_profiler" / "mirror_contract.py"
    spec = importlib.util.spec_from_file_location("_mirror_contract", mc_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [
        f"{mirror}: base {base} changed (digest {expected} -> {actual}) — port the "
        f"change into the mirror, then refresh digests: "
        f"`python -m tools.diagnostics.speed_profiler.mirror_contract`"
        for mirror, base, expected, actual in mod.check(root)
    ]


# gitignore-traps: a bare `name/` ignore pattern matches at every depth, so ignoring one folder
# silently swallows same-named folders elsewhere — new files under a TRACKED directory then never
# show up in `git status` / `git add` (it hid the internal/ mirrors of experiments/ and
# configs/datasets/, and a `!dir/` without `!dir/**` did the same under .claude/). Probe every
# directory that holds tracked files with a hypothetical new file; a match from a .gitignore that
# lives ABOVE the directory is a trap. A directory's own .gitignore may ignore its content — that
# is that tree's decision (.claude/.gitignore does). The fix is always to anchor: `/name/`.
def check_gitignore_traps(root: Path) -> list[str]:
    ls = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True).stdout
    dirs = sorted({p.rsplit("/", 1)[0] for p in ls.split("\0") if "/" in p})
    probes = "".join(f"{d}/.gitignore-probe\n" for d in dirs)
    proc = subprocess.run(["git", "check-ignore", "-v", "--stdin"], cwd=root, input=probes, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return [f"git check-ignore failed: {proc.stderr.strip()}"]
    failures, local = [], []
    for line in proc.stdout.splitlines():
        rule, _, path = line.partition("\t")
        source, lineno, pattern = rule.split(":", 2)
        if pattern.startswith("!"):
            continue                                   # re-included, not ignored
        d = path.rsplit("/", 1)[0]
        if str(Path(source).parent) == d:
            continue                                   # the directory's own .gitignore
        if Path(source).is_absolute() or source.startswith(".git/"):
            local.append(f"{d}/ ← {source}:{lineno} `{pattern}`")   # this machine's config, not the repo's
            continue
        failures.append(f"{d}/: a new file there is ignored by {source}:{lineno} `{pattern}` — anchor the "
                        f"pattern to the folder it means (`/{pattern.lstrip('/')}`), or move the rule into "
                        f"{d}/.gitignore if that tree really ignores its own content")
    if local:
        # Not a repo defect, but the same silent loss on THIS machine: say it loudly, do not fail.
        print("  ⚠ local exclude (.git/info/exclude or core.excludesFile) swallows new files under tracked "
              "directories — delete or anchor the line:")
        for item in local:
            print(f"      {item}")
    return failures


CHECKS = {
    "md-links": check_md_links,
    "version-sync": check_version_sync,
    "history": check_history,
    "dead-paths": check_dead_paths,
    "gitignore-traps": check_gitignore_traps,
    "cpp-bump": check_cpp_bump,
    "test-durations": check_test_durations,
    "mirror-sync": check_mirror_sync,
    "import-hygiene": check_import_hygiene,
    "wire-sync": check_wire_sync,
    "public-hygiene": check_public_hygiene,
    "upstream-diff-sync": check_upstream_diff_sync,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(CHECKS), help="run a single check")
    args = ap.parse_args(argv)

    root = repo_root()
    names = [args.only] if args.only else list(CHECKS)
    total_failures = 0
    for name in names:
        failures = CHECKS[name](root)
        if failures:
            total_failures += sum(1 for f in failures if not f.startswith("    "))
            print(_c(f"✗ {name}", RED))
            for f in failures:
                print(f"  {f}")
        else:
            print(_c(f"✓ {name}", GREEN))

    if total_failures:
        print(_c(f"\n{total_failures} doc-sync issue(s). Fix, or bypass with SKIP_DOC_CHECK=1.", BOLD))
        return 1
    print(_c("\nAll doc-sync checks passed.", GREEN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
