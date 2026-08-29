#!/usr/bin/env python3
"""Deterministic doc/code consistency checker (read-only — reports drift, never edits).

Pure stdlib (no PyYAML/torch) so it runs in any env.

Checks (run all by default; `--only <name>` to run one):
  md-links     — every local [text](path) link in live docs resolves to a real file/dir
  version-sync — pyproject.toml `version` == README.md <!--VERSION--> marker
  dead-paths   — removed path prefixes (DEAD_PATHS) appear in no live tracked file
  import-hygiene — the GPL/NC boundary: GPL imports only inside the engine/
                 submodule/pcb_world/engine/ (dev-only in-process escape)/
                 native test bundle; nothing outside the engine imports its
                 internals
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
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Removed path/module prefixes that must not reappear in live tracked files.
# Grows when something is renamed — append the dead prefix here so the rename
# can never silently regress. Keep entries fully-qualified and unambiguous
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
    ("classical_routers", "pre-refactor top-level name; code lives in methods/baselines/rule_based"),
    # Dependency single-sourcing: one pinned pyproject + one compiled lock.
    ("requirements-all.txt", "replaced by requirements.txt (uv pip compile of the pinned pyproject)"),
    ("requirements-kicad-only.txt", "removed — one-shot install policy; use requirements.txt"),
    ("tests/crashtrace.c", "moved to pcb_world/diag/crashtrace.c (shared via pcb_world.diag)"),
    ("logs/crash", "cwd-relative default removed — crash artifacts live in var/crashlogs (pcb_world.diag.default_log_dir)"),
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
     "unused legacy split removed (dead boards_json defaults dropped to null)"),
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
    ("env/trajectory/trajectory_replay", "removed (orphan; TrajectoryLogger stays live)"),
    ("env.trajectory.trajectory_replay", "removed"),
    ("env/core/wrappers", "package removed (LoggingWrapper gone with it)"),
    ("env.core.wrappers", "package removed"),
    # make_line mode verification: the manual main() script was ported to pytest.
    ("tests/integration/test_make_line_modes", "ported to tests/test_engine_api/test_make_line_modes.py (mark_obstacles=commit-reject verdict pinned)"),
    # xdist loadfile balance — a 17.5s single-file chain split per topic.
    ("test_drc_incremental_keying", "split into tests/test_drc_incremental/{test_keying,test_equivalence,test_restore}.py (shared helpers: tests/helpers/drc_keying.py)"),
    # llm_agent legacy-path sweep — old top-level LLM paths folded into methods/llm_agent/.
    # The GPL-side code lives in the engine repository, pinned as engine/.
    ("external/kicad-patches", "moved to engine/kicad-patches (the engine repository)"),
    ("external/kicad-python", "moved to engine/kicad-python (submodule of the engine)"),
    ("external/engine_server", "moved to engine/engine_server"),
    ("tools/build/build_rl_router", "moved to engine/build_rl_router.sh"),
    ("_dataset_prep", "moved to engine/pcbnew_prep (the scripts import GPL pcbnew)"),
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
    ("docs/INCREMENTAL_DRC.md", "moved to docs/design/INCREMENTAL_DRC.md"),
    # Legacy masking YAML removal — 4 unloadable-by-file variants removed; the names still map via _NAME_COMPAT.
    ("configs/masking/strict.yaml", "removed — name-only clone of configs/masking/default.yaml (name 'strict' still maps via _NAME_COMPAT)"),
    ("configs/masking/strict_no_finish.yaml", "removed — name-only clone of configs/masking/default_no_finish.yaml (name still maps via _NAME_COMPAT)"),
    ("configs/masking/strict_phase.yaml", "removed — old phases: format; name maps to 'default'"),
    ("configs/masking/relaxed_phase.yaml", "removed — old phases: format; name maps to 'relaxed'"),
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
    ("EvaluationMetrics", "renamed to EvalSummary (eval/metrics.py; consumer-side summary over EvalResult)"),
    ("edatool", "pyproject distribution renamed to pcbworld (paper name)"),
    # There is no low/high-level split — the HL prefix no longer applies.
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
]

# Pathspecs excluded from the dead-paths scan: vendored/generated trees + files
# that legitimately record the old names (this tooling).
DEAD_PATH_EXCLUDES = [
    ":!external",
    ":!var",
    ":!build_rl",
    ":!tools/docs",
    # auto-generated map of LIVE collected test files (a live filename may
    # embed a dead-path substring, e.g. test_eval_cadagent_canonical.py ⊃
    # 'eval_cadagent'); ghost entries are caught by check_test_durations.
    ":!tests/durations.json",
]

# Directories excluded from the markdown link scan (vendored / generated).
MD_EXCLUDE_DIRS = ("external/", "var/", "build_rl/")

# Docs that legitimately reference removed paths — their links are *expected*
# to go stale, so skip link-checking them. (Currently none.)
MD_LINK_EXCLUDE_FILES: set[str] = set()

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
                "(needed to keep it in sync with pyproject.toml)"]
    rd_ver = m_rd.group(1).strip().lstrip("v")

    if py_ver != rd_ver:
        return [f"version mismatch: pyproject.toml={py_ver} vs README.md={rd_ver} "
                f"(sync by hand)"]
    return []


def check_dead_paths(root: Path) -> list[str]:
    # ONE combined `git grep` over every dead-path pattern instead of one
    # subprocess per pattern. On large/slow working trees the per-invocation
    # tree walk dominates (~2s each × 100+ patterns ≈ minutes); folding them
    # into a single `-e … -e …` scan cuts wall time ~60x. Attribution back to
    # each pattern is recovered from the matched line's content (`-F` ⇒ literal
    # substring), so the per-dead-path hint output is unchanged.
    patterns = [dead for dead, _ in DEAD_PATHS]
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
    for dead, hint in DEAD_PATHS:
        hits = hits_by_dead[dead]
        if hits:
            failures.append(f"dead path '{dead}' → {hint}")
            failures.extend(f"    {h}" for h in hits)
    return failures


# --------------------------------------------------------------------------- #
# import-hygiene: the GPL/NC boundary gate.
# The engine — every component that links or imports GPL KiCad code — is a
# separate repository, pinned here as the `engine/` submodule and tracked in
# this tree as nothing but a URL and a commit hash. Two machine-checked rules
# keep the boundary real in the files this repository does ship:
#
#  (1) GPL python modules (kicad_rl_router / pcbnew) may be imported ONLY by:
#      - engine/           : the engine checkout itself (nothing tracked here)
#      - pcb_world/engine/ : the sole environment-side loader — a DEV-ONLY
#                            escape hatch (`KICAD_ENGINE_IPC=0` in-process
#                            mode); the default IPC mode never executes those
#                            imports, and the module it names is not part of
#                            this repository.
#      - tests/test_engine_api/, tests/stress/ : the
#                            engine-direct (native) test bundle — runs the
#                            binding in-process by design.
#  (2) Nothing imports the engine's internals. The IPC client
#      (pcb_world/engine/router_client.py) is the only bridge, and it SPAWNS
#      the server by file path — it never imports it.
#
# Everything else must stay import-clean, so that a grep for GPL imports
# outside the boundary returns nothing by construction. The cross-repository
# half of the boundary (no GPL bytes here, the two wire.py copies identical)
# is checked by tools/check_separation.py.
# --------------------------------------------------------------------------- #

GPL_BUNDLE_PREFIX = "engine/"

IMPORT_HYGIENE_ALLOWED_PREFIXES = (
    GPL_BUNDLE_PREFIX,
    "pcb_world/engine/",
    "tests/test_engine_api/",
    "tests/stress/",
)

_GPL_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:kicad_rl_router|pcbnew)\b")
# The engine's own top-level modules. Importing one from this repository would
# mean loading engine code in-process, which is exactly what the socket avoids.
_ENGINE_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:engine|engine_server)\b")


def check_import_hygiene(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "*.py"],
        capture_output=True, text=True, check=True)
    failures: list[str] = []
    for rel in proc.stdout.splitlines():
        inside_engine = rel.startswith(GPL_BUNDLE_PREFIX)
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
                    "move the file into the engine repository or the native "
                    "test dirs")
            if not inside_engine and _ENGINE_IMPORT_RE.match(line):
                failures.append(
                    f"{rel}:{lineno}: engine internals imported "
                    f"({line.strip()}) — the IPC client "
                    "(pcb_world/engine/router_client.py) is the only bridge; "
                    "it spawns the server by path, never by import")
    return failures


def check_test_durations(root: Path) -> list[str]:
    """tests/durations.json (xdist scheduling seed) covers every test file.

    tests/conftest.py orders files longest-first for xdist from this tracked
    seed; a missing entry schedules the file as instantly-fast, a stale entry is
    rot. Refresh = `pytest <paths> --update-durations` (merges files that ran,
    prunes deleted). Files pytest collects no tests from keep a manual 0.0 entry.
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
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        fs.update(f"{rel_dir}/{fn}" for fn in filenames
                  if fn.startswith("test_") and fn.endswith(".py"))

    failures: list[str] = []
    missing = sorted(fs - seed)
    if missing:
        failures.append("tests/durations.json: missing entries (new test file?) — run "
                        "`pytest <file> --update-durations`; for non-collected "
                        "files add a manual 0.0 entry")
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
    edit without a mirror re-sync makes the profiler measure a stale path.
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


CHECKS = {
    "md-links": check_md_links,
    "version-sync": check_version_sync,
    "dead-paths": check_dead_paths,
    "test-durations": check_test_durations,
    "mirror-sync": check_mirror_sync,
    "import-hygiene": check_import_hygiene,
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
