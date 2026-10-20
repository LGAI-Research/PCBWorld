"""Contract of the KDD paper recipes against the shipped defaults.

The paper's rules live under ``experiments/kdd/configs/`` and resolve by name, but the
training KNOBS whose shipped defaults moved on after the paper (observation tokens, outline
representation, directional candidates, connectivity filter, make_line's reject-if-stuck,
time feature, PPO truncation bootstrap) are only
paper-faithful if the recipe shells pass them explicitly (``configs/paper_train_flags.sh``).
The v1.0.0 restart nearly shipped Table 1 PPO-terminal broken exactly this way: the recipe
relied on ``truncation_bootstrap`` being on by default, and the reward rule
(``truncation_mode: none``) refuses to train without it.

Each test runs a recipe shell in ``--dry-run`` (it echoes the assembled trainer command and
returns — milliseconds, no GPU / engine / dataset), parses that command with the trainer's own
argparse and checks the resolved values against the pre-knob values of the checkpoint
migration table ``configs/ckpt_migrations.yaml`` and the trainer's truncation rule.
A future default flip that the recipes do not pin fails here, before any release is cut.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
# A proxy entry (reject_if_stuck) takes its value-if-absent: the paper checkpoints predate the proxy key.
def _paper_value(key, entry):
    if "value" in entry:
        return entry["value"]
    if "proxy" in entry:
        return entry["value_if_absent"]
    raise AssertionError(f"ckpt_migrations.yaml {key}: an entry shape this test does not know: {entry}")


LEGACY = {section: {k: _paper_value(k, e) for k, e in entries.items()}
          for section, entries in yaml.safe_load(
              (REPO / "configs/ckpt_migrations.yaml").read_text())["ckpt_args"].items()}
# The knobs the recipes must pin (the paper-era values), by section of configs/ckpt_migrations.yaml.
PINNED_ENV = ("outline_obs", "simplify_outline", "net_constraint_obs", "directional_candidates", "offboard_mask",
              "net_reselect", "connectivity_filter", "reject_if_stuck")
PINNED_POLICY = ("time_feature", "obstacle_obs")
# Where a recipe's paper value is not the pre-knob value: D1 (Figure 5) trained on the grid<N>
# candidates (the original recipe's --directional-grid-size), not the 8-direction ring.
PAPER_OVERRIDES = {"figure5_d1": {"directional_candidates": "grid10"}}

RECIPES = [
    ("table1_rl:ppo_per_step", ["experiments/kdd/table1_rl/train_policy.sh", "--method", "ppo_per_step"]),
    ("table1_rl:ppo_terminal", ["experiments/kdd/table1_rl/train_policy.sh", "--method", "ppo_terminal"]),
    ("table1_rl:grpo", ["experiments/kdd/table1_rl/train_policy.sh", "--method", "grpo"]),
    ("figure6_reward", ["experiments/kdd/figure6_reward/train_dense_reward_cell.sh",
                        "--wirelength-penalty", "0.002", "--via-penalty", "0.1"]),
    ("figure5_d1", ["experiments/kdd/figure5_d1/train_transformer_ppo.sh", "--grid-size", "10",
                    "--split-json", "tests/fixtures/simple_routing_board.kicad_pcb"]),
]


def _launch_argv(recipe_cmd: list[str], tmp_path: Path) -> list[str]:
    proc = subprocess.run(
        ["bash", *recipe_cmd, "--seed", "42", "--output-root", str(tmp_path), "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120, env={**os.environ, "NO_WANDB": "1"},
    )
    assert proc.returncode == 0, f"dry-run failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    line = next((ln for ln in (proc.stderr + proc.stdout).splitlines()
                 if "methods.rl_agent.training.train_" in ln), None)
    assert line, f"no trainer launch echoed:\n{proc.stderr}"
    argv = shlex.split(line.split("CUDA_VISIBLE_DEVICES=", 1)[-1])
    return argv[argv.index("-m") + 1:]          # ["methods.rl_agent.training.train_ppo", flags...]


def _parse(module: str, flags: list[str]):
    if module.endswith("train_grpo"):
        from methods.rl_agent.training.train_grpo import build_arg_parser
        return build_arg_parser().parse_args(flags)
    from methods.rl_agent.training.train_ppo import parse_train_args
    return parse_train_args(flags)


@pytest.mark.parametrize("name,recipe", RECIPES, ids=[r[0] for r in RECIPES])
def test_recipe_pins_the_paper_knobs(name, recipe, tmp_path):
    from configs.loader.schema import RLEnvConfig
    from pcb_world.core.masking import get_masking_rule
    from pcb_world.core.reward_config import get_reward_config

    module, *flags = _launch_argv(recipe, tmp_path)
    args = _parse(module, flags)
    env = RLEnvConfig.from_namespace(args).to_record()
    paper = {**LEGACY["env"], **PAPER_OVERRIDES.get(name, {})}
    for key in PINNED_ENV:
        assert env[key] == paper[key], f"{name}: {key}={env[key]!r}, paper value {paper[key]!r}"
    # the trainer's in-run validation: 10 rollouts every 20 iterations (the cells' run args)
    assert (args.eval_n_rollouts, args.eval_every) == (10, 20), name
    for key in PINNED_POLICY:
        assert getattr(args, key) == LEGACY["rl_policy"][key], \
            f"{name}: {key}={getattr(args, key)!r}, paper value {LEGACY['rl_policy'][key]!r}"
    # The paper's rules resolve by name (configs/ first, then experiments/kdd/configs/).
    reward = get_reward_config(args.reward_rule)
    get_masking_rule(args.masking_rule)
    # PPO: a terminal-mode reward (truncation_mode none) needs the truncation bootstrap the
    # trainer refuses to run without — the paper ran with it on for every PPO cell.
    if not module.endswith("train_grpo"):
        assert args.truncation_bootstrap is True, f"{name}: truncation bootstrap must be on (paper default)"
        assert getattr(reward, "truncation_mode", None) != "none" or args.truncation_bootstrap, name


# The D1 stages are the one public path whose inputs are absent BY DESIGN (the paper
# corpus is not distributed), so the absent branch — not the present one the tests above
# exercise — is what a fresh clone actually runs. It shipped calling two functions that
# were defined nowhere, so it died with `command not found` (127) instead of the exit 2 +
# notice QUICKSTART documents.
D1_STAGES = {
    "eval": ["experiments/kdd/figure5_d1/run.sh", "eval"],
    "train": ["experiments/kdd/figure5_d1/train_transformer_ppo.sh", "--grid-size", "10", "--dry-run"],
}


# The jumanji/sable rows are a record, not runnable here (experiments/kdd/figure5_d1/README.md
# says why): the launchers must say so rather than exec a missing file.
@pytest.mark.parametrize("launcher", ["train_jumanji_a2c.sh", "train_sable.sh"])
def test_d1_baseline_launcher_refuses_and_says_the_runner_is_absent(launcher):
    r = subprocess.run(["bash", f"experiments/kdd/figure5_d1/{launcher}",
                        "--grid-size", "10", "--seed", "42", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, f"{launcher}: exit {r.returncode}\n{r.stdout[-400:]}{r.stderr[-400:]}"
    assert "not part of this repository" in r.stderr, f"{launcher}: notice must say the runner is absent\n{r.stderr[-400:]}"
    assert "run_v56" in r.stderr, f"{launcher}: notice must name the runner\n{r.stderr[-400:]}"
    assert "run_v56" not in r.stdout, f"{launcher}: must not log a command it cannot run\n{r.stdout[-400:]}"


@pytest.mark.parametrize("stage", sorted(D1_STAGES), ids=sorted(D1_STAGES))
def test_d1_stage_without_inputs_exits_2_with_the_generator_command(stage, tmp_path):
    env = dict(os.environ, L1_GRIDS="10", SEEDS="42",
               BOARDS_DIR=str(tmp_path / "no-boards"), CKPT=str(tmp_path / "no.pt"),
               SPLIT_JSON=str(tmp_path / "no-split.json"), DATASET_ROOT=str(tmp_path / "no-root"))
    r = subprocess.run(["bash", *D1_STAGES[stage]], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, f"{stage}: exit {r.returncode} (127 = a preflight helper is undefined)\n{r.stderr[-800:]}"
    assert "make_grid_dataset.sh" in r.stderr, f"{stage}: the notice must name the generator\n{r.stderr[-800:]}"


def test_table2_interactive_runs_the_table1_llm_runner(tmp_path):
    """Table 2's interactive column is an alias of the Table 1 LLM runner (``--samples`` becomes
    ``--rollouts``). Its exec path once named a folder that does not exist (``table1/llm/``), so the
    command the KDD README lists exited 127; the dry-run needs no API key, data root or engine."""
    proc = subprocess.run(
        ["bash", "experiments/kdd/table2/run_interactive.sh", "--model", "gpt-5.4-mini", "--split", "d2a",
         "--out", str(tmp_path), "--samples", "2", "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    cmd = next((ln for ln in out.splitlines() if "methods.llm_agent.rollout.pcbworld" in ln), None)
    assert cmd, out
    assert "--rollout_episodes 2" in cmd


def _llm_args(tmp_path, *extra, model="gpt-5.4-mini"):
    """The rollout args ``experiments/kdd/table1_llm/run.sh --dry-run`` prints, parsed (no API key,
    data root or engine needed)."""
    from eval.args import build_eval_parser

    proc = subprocess.run(
        ["bash", "experiments/kdd/table1_llm/run.sh", "--model", model, "--split", "d2a",
         "--out", str(tmp_path), *extra, "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    cmd = next(ln for ln in out.splitlines() if "methods.llm_agent.rollout.pcbworld" in ln)
    argv = shlex.split(cmd.split("&&", 1)[1].rstrip(")"))
    return build_eval_parser().parse_args(argv[argv.index("methods.llm_agent.rollout.pcbworld") + 1:])


def test_table1_llm_recipe_pins_the_paper_knobs(tmp_path):
    """The Table 1 LLM recipe pins tess outlines, no outline simplification, no per-net
    constraints, the paper-era make_line (reject_if_stuck off), the strict masking rule and prompt v5 (the shipped LLM defaults are arc /
    simplified / constraints / pcbworld_masking / latest), a base budget of 20 steps (raised
    per board to max(base, 3*pins + 2*nets)), a stop after 8 non-progress actions and none on a
    geometry stall, the last 4 turns as history and a 256-token turn budget for the OpenAI
    models."""
    from eval.args import llm_env_config
    from methods.llm_agent.wrappers.prompts import check_prompt_masking

    args = _llm_args(tmp_path)
    env = llm_env_config(args)
    for key in ("outline_obs", "simplify_outline", "net_constraint_obs", "net_reselect", "reject_if_stuck"):
        assert getattr(env, key) == LEGACY["env"][key], f"{key}={getattr(env, key)!r}"
    assert (env.masking_rule, args.prompt_version) == ("strict", "v5")
    check_prompt_masking(args.prompt_version, env.masking_rule)
    assert (env.max_steps, env.reward_rule, env.corner_mode) == (20, "grpo_final", 45)
    assert (args.history_length, args.early_stop_no_progress, args.early_stop_no_geometry_progress) == (4, 8, 0)
    assert (args.max_new_tokens, args.temperature, args.rollout_episodes) == (256, 0.7, 5)


def test_table1_llm_recipe_gives_the_together_model_its_paper_token_budget(tmp_path):
    """A together-served model (the Qwen row) gets a 512-token turn budget."""
    assert _llm_args(tmp_path, model="qwen3.5-397b").max_new_tokens == 512
    assert _llm_args(tmp_path, "--max-new-tokens", "1024").max_new_tokens == 1024
    assert _llm_args(tmp_path, "--early-stop", "0").early_stop_no_progress == 0


def test_table1_llm_recipe_runs_gpt54_without_the_early_stop():
    """gpt-5.4 runs with --early_stop_no_progress 0, every other model with 8."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert _llm_args(Path(d), model="gpt-5.4").early_stop_no_progress == 0
        assert _llm_args(Path(d), "--early-stop", "8", model="gpt-5.4").early_stop_no_progress == 8


def test_table1_llm_recipe_takes_another_masking_rule(tmp_path):
    """``--masking`` replaces the pinned strict rule (prompt v7 needs a no-finish rule): the old
    advice, ``--extra-args "--masking_rule ..."``, now gives the flag twice and the rollout refuses."""
    from eval.args import llm_env_config
    from methods.llm_agent.wrappers.prompts import check_prompt_masking

    args = _llm_args(tmp_path, "--prompt", "v7", "--masking", "default_no_finish")
    assert llm_env_config(args).masking_rule == "default_no_finish"
    check_prompt_masking(args.prompt_version, "default_no_finish")


@pytest.mark.parametrize("wandb,flag", [("0", "--no-wandb"), ("1", "--wandb")])
def test_launchers_say_wandb_off_explicitly(wandb, flag):
    """``public_wandb_args`` (every KDD / d2b trainer launcher) passes ``--no-wandb`` when W&B is
    off: a ``--resume`` starts its runtime keys from the checkpoint, so silence kept a W&B run's on."""
    proc = subprocess.run(["bash", "-c", 'source experiments/_lib/env.sh; public_wandb_args r g; '
                           'echo "${PUBLIC_WANDB_ARGS[0]}"'], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "WANDB": wandb, "WANDB_MODE": ""}, timeout=60)
    assert proc.stdout.strip() == flag, proc.stderr


def _pipeline_runs(stage_cmd: list[str], env: dict[str, str], monkeypatch):
    """Every ``eval/pipeline.py`` command a recipe stage echoes under ``DRY_RUN=1``, parsed with
    the pipeline's own parser: ``(rollout runs, scoring runs)``."""
    from eval import pipeline

    proc = subprocess.run(["bash", *stage_cmd], cwd=REPO, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "DRY_RUN": "1", **env})
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    runs = []
    for line in (proc.stdout + proc.stderr).splitlines():
        if "eval/pipeline.py" in line and "CUDA_VISIBLE_DEVICES=" in line:
            argv = shlex.split(line.split("CUDA_VISIBLE_DEVICES=", 1)[1])
            monkeypatch.setattr(sys, "argv", ["pipeline.py", *argv[argv.index("eval/pipeline.py") + 1:]])
            runs.append(pipeline._parse_args()[1])
    assert runs, proc.stdout + proc.stderr
    return [a for a in runs if not a.skip_rollout], [a for a in runs if a.skip_rollout]


def _data_root(tmp_path: Path) -> Path:
    """A dataset root holding (empty) files for every D2 / D3 test board the splits list."""
    from configs.loader.paths import expand_data_path
    import json

    root = tmp_path / "data"
    for split_json, diff in (("experiments/kdd/configs/datasets/d2a.json", "easy"),
                             ("configs/datasets/d3.json", "easy"), ("configs/datasets/d3.json", "medium")):
        data = json.loads((REPO / split_json).read_text())
        base = Path(data["dataset_dirs"]["test"].replace("${PCBWORLD_DATA_ROOT}", str(root)))
        for bid in data[diff]["test"]:
            path = base / bid / data["board_filename"] if data.get("board_filename") else base / f"{bid}.kicad_pcb"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return root


# What the paper's RL cells ran (the unified-pipeline cells of 2026-05-30/31 behind the published
# numbers: eval_overall_summary.json + per_rollout.csv). Rollout: sampled, at most 256 steps from
# env seed 5600 + 100*board + r, a stop after 10 finish actions without progress, 64 parallel envs.
# Scoring: post-hoc DRC with the plain drc_dense_promoted rule (errors + the three promoted warnings,
# NO wirelength/via term), the rollout of largest final potential. The recipes used to name an
# errors-only rule carrying the training wirelength/via terms — other potentials, other winners.
PAPER_ROLLOUT = dict(seed=5600, max_steps=256, early_stop_finish_no_progress=10, rollout_mode="parallel",
                     n_envs=64, env_drc="off", inline_drc="off", save_artifacts="on")
PAPER_SCORING = dict(reward_config="drc_dense_promoted", selection_method="final_potential",
                     rollout_mode="parallel", n_envs=64)


def _assert_values(a, expected):
    for key, value in expected.items():
        assert getattr(a, key) == value, f"{key}={getattr(a, key)!r}, paper cell {value!r}"


def test_paper_scoring_rule_has_no_wirelength_or_via_term():
    from pcb_world.core.reward_config import get_reward_config

    reward = get_reward_config("drc_dense_promoted").build_reward()
    assert (reward.drc_severity_mode, reward.wirelength_penalty, reward.via_penalty) == \
        ("errors_and_promoted", 0.0, 0.0)


def test_table1_rl_eval_scores_like_the_paper_cells(tmp_path, monkeypatch):
    """Table 1 (paper Table 3): each seed's policy_best.pt rolled out (5 rollouts per board, 45
    degrees, the trained 64-slot net table), pooled into one cell per (task, method) where the
    figures read it, scored there; D2 (128) / D3-A (99) / D3-B (10) test boards."""
    results = tmp_path / "results"
    rollouts, scorings = _pipeline_runs(
        ["experiments/kdd/table1_rl/run.sh", "eval"],
        {"LOCAL_OUT": str(tmp_path), "EXPR_ROOT": str(results), "PCBWORLD_DATA_ROOT": str(_data_root(tmp_path)),
         "SEEDS": "42 43", "METHODS": "ppo_per_step grpo"}, monkeypatch)
    assert len(rollouts) == 3 * 2 * 2 and len(scorings) == 3 * 2   # task x method (x seed)
    for a in rollouts:
        _assert_values(a, {**PAPER_ROLLOUT, "n_rollouts": 5, "check_angle": 45, "override_n_max_slots": 64})
        assert a.ckpt.name == "policy_best.pt" and a.ckpt.parts[-4] in ("seed42", "seed43")
        assert a.output_dir.name in ("transformer_pcbworld", "transformer_pcbworld_grpo")   # the cell name
    for a in scorings:
        _assert_values(a, {**PAPER_SCORING, "check_angle": 45, "skip_drc": False, "skip_aggregate": False})
        assert a.n_rollouts is None   # rollout-only flags stay with Stage 1
    assert {str(a.output_dir.relative_to(results)) for a in scorings} == {
        f"{t}/{c}" for t in ("d2a", "d3/d3a", "d3/d3b") for c in ("transformer_pcbworld", "transformer_pcbworld_grpo")}
    assert sorted({len(a.boards_list.read_text().splitlines()) for a in scorings}) == [10, 99, 128]


def test_table1_rl_eval_refuses_one_ckpt_for_several_seeds(tmp_path):
    r = subprocess.run(["bash", "experiments/kdd/table1_rl/run.sh", "eval"], cwd=REPO, capture_output=True,
                       text=True, timeout=120, env={**os.environ, "DRY_RUN": "1", "CKPT": str(tmp_path / "x.pt"),
                                                    "SEEDS": "42 43", "METHODS": "grpo"})
    assert r.returncode == 2 and "single METHODS and SEEDS" in r.stderr, r.stderr[-400:]


def test_reward_sweep_eval_takes_one_rollout_per_board(tmp_path, monkeypatch):
    """Figure 8 (paper Figure 9): each cell's policy_best.pt per seed, ONE rollout per D2 test board
    at 45 degrees, pooled into d2a/reward_w<wire>_v<via> — the figure averages every rollout.
    The recipe had no eval stage at all."""
    results = tmp_path / "results"
    rollouts, scorings = _pipeline_runs(
        ["experiments/kdd/figure6_reward/run.sh", "eval"],
        {"LOCAL_OUT": str(tmp_path), "EXPR_ROOT": str(results), "PCBWORLD_DATA_ROOT": str(_data_root(tmp_path)),
         "SEEDS": "42 43", "WIRES": "0 0.002", "VIAS": "0.1"}, monkeypatch)
    assert len(rollouts) == 4 and len(scorings) == 2
    for a in rollouts:
        _assert_values(a, {**PAPER_ROLLOUT, "n_rollouts": 1, "check_angle": 45})
    for a in scorings:
        _assert_values(a, {**PAPER_SCORING, "check_angle": 45})
    assert {str(a.output_dir.relative_to(results)) for a in scorings} == {"d2a/reward_w0_v0.1", "d2a/reward_w0.002_v0.1"}


def test_d1_eval_scores_like_the_paper_cells(tmp_path, monkeypatch):
    """Figure 5 (paper Figure 6): the staged paper checkpoint is the run's LAST one,
    policy_iter_300.pt (its policy_best.pt sits near iteration 25), 5 rollouts per test board,
    the 90-degree grid angle, pooled into d1/d1_grid<G>/transformer_pcbworld."""
    (tmp_path / "boards").mkdir()
    (tmp_path / "boards" / "board_00000.kicad_pcb").touch()
    for seed in (42, 43):
        ckpt = tmp_path / "Transformer_1L_300" / "grid10" / f"seed{seed}" / "policy_iter_300.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
    results = tmp_path / "results"
    rollouts, scorings = _pipeline_runs(
        ["experiments/kdd/figure5_d1/run.sh", "eval"],
        {"L1_GRIDS": "10", "SEEDS": "42 43", "CKPT_ROOT": str(tmp_path), "BOARDS_DIR": str(tmp_path / "boards"),
         "LOCAL_OUT": str(tmp_path), "EXPR_ROOT": str(results)}, monkeypatch)
    assert [a.ckpt.parts[-2:] for a in rollouts] == [("seed42", "policy_iter_300.pt"), ("seed43", "policy_iter_300.pt")]
    for a in rollouts:
        _assert_values(a, {**PAPER_ROLLOUT, "n_rollouts": 5, "check_angle": 90})
    assert len(scorings) == 1
    _assert_values(scorings[0], {**PAPER_SCORING, "check_angle": 90})
    assert scorings[0].output_dir == results / "d1" / "d1_grid10" / "transformer_pcbworld"


@pytest.mark.parametrize("recipe,env,expected", [
    ("experiments/kdd/table1_rl/run.sh", {"SEEDS": "42 43", "METHODS": "grpo"},
     ["seed42/grpo", "seed43/grpo"]),   # train_policy.sh adds the method
    ("experiments/kdd/figure6_reward/run.sh", {"SEEDS": "42", "WIRES": "0.002", "VIAS": "0 0.1"},
     ["w0.002_v0/seed42", "w0.002_v0.1/seed42"]),
    ("experiments/kdd/figure5_d1/run.sh", {"SEEDS": "42 43", "L1_GRIDS": "10", "SPLIT_JSON": "x.json"},
     ["grid10/seed42", "grid10/seed43"]),
], ids=["table1", "reward", "d1"])
def test_train_sweeps_keep_runs_apart(recipe, env, expected, tmp_path):
    """The trainer shells write <output-root>/.../checkpoints: each sweep gives every run its own
    root, so a later seed or cell no longer overwrites an earlier one's checkpoints (which eval reads)."""
    (tmp_path / "x.json").touch()
    proc = subprocess.run(["bash", recipe, "train"], cwd=REPO, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "DRY_RUN": "1", "LOCAL_OUT": str(tmp_path), "TRAIN_ROOT": str(tmp_path / "t"),
                               **{k: str(tmp_path / v) if k == "SPLIT_JSON" else v for k, v in env.items()}})
    assert proc.returncode == 0, proc.stderr[-800:]
    save_dirs = [shlex.split(ln)[shlex.split(ln).index("--save-dir") + 1]
                 for ln in proc.stderr.splitlines() if "--save-dir" in ln]
    assert save_dirs == [f"{tmp_path}/t/{e}/checkpoints" for e in expected]


def test_pool_seed_cells_moves_boards_and_concatenates_rows(tmp_path, monkeypatch):
    """A pipeline run rolls out one checkpoint; the pooled cell carries every seed's boards and rows,
    with artifact_path at the moved file, and refuses to pool twice or a part named otherwise."""
    from eval import eval_utils as u

    monkeypatch.syspath_prepend(str(REPO / "experiments" / "_lib"))
    from pool_seed_cells import pool

    parts = []
    for seed in (42, 43):
        part = tmp_path / "parts" / str(seed) / "cellx"
        (part / "boards").mkdir(parents=True)
        board = part / "boards" / f"b0_cellx_s{seed}_r00.kicad_pcb"
        board.write_text(str(seed))
        board.with_suffix(".kicad_pro").write_text("pro")
        u.write_csv(part / "per_rollout.csv", [{"board_id": "b0", "rollout_idx": 0, "artifact_path": str(board)},
                                               {"board_id": "b1", "rollout_idx": 0, "artifact_path": ""}],
                    u.PER_ROLLOUT_FIELDS)
        parts.append(part)
    cell = tmp_path / "results" / "cellx"
    assert pool(cell, parts) == 4
    rows = u.read_csv(cell / "per_rollout.csv")
    moved = [Path(r["artifact_path"]) for r in rows if r["artifact_path"]]
    assert [p.read_text() for p in moved] == ["42", "43"] and all(p.parent == cell / "boards" for p in moved)
    assert (cell / "boards" / "b0_cellx_s42_r00.kicad_pro").is_file()
    with pytest.raises(SystemExit, match="already has per_rollout.csv"):
        pool(cell, parts)
    with pytest.raises(SystemExit, match="not named like the cell"):
        pool(tmp_path / "other" / "celly", parts)


def test_pool_seed_cells_checks_everything_before_moving_anything(tmp_path, monkeypatch):
    """A failed pool leaves the parts as they were: a missing artifact (or a clash) found in the
    second part must not have moved the first part's boards."""
    from eval import eval_utils as u

    monkeypatch.syspath_prepend(str(REPO / "experiments" / "_lib"))
    from pool_seed_cells import pool

    parts = []
    for seed in (42, 43):
        part = tmp_path / str(seed) / "cellx"
        (part / "boards").mkdir(parents=True)
        board = part / "boards" / f"b0_cellx_s{seed}_r00.kicad_pcb"
        if seed == 42:
            board.write_text("42")
        u.write_csv(part / "per_rollout.csv", [{"board_id": "b0", "rollout_idx": 0, "artifact_path": str(board)}],
                    u.PER_ROLLOUT_FIELDS)
        parts.append(part)
    with pytest.raises(SystemExit, match="is missing"):
        pool(tmp_path / "results" / "cellx", parts)
    assert (parts[0] / "boards" / "b0_cellx_s42_r00.kicad_pcb").read_text() == "42"
    assert not (tmp_path / "results" / "cellx" / "boards").exists()
    with pytest.raises(SystemExit, match="written twice"):
        pool(tmp_path / "results" / "cellx", [parts[0], parts[0]])


def test_eval_stage_refuses_an_existing_cell_before_rolling_out(tmp_path):
    """A pooled cell is written once: an existing one stops the stage before any rollout runs."""
    cell = tmp_path / "results" / "d2a" / "reward_w0_v0.1"
    cell.mkdir(parents=True)
    (cell / "per_rollout.csv").write_text("board_id\n")
    r = subprocess.run(["bash", "experiments/kdd/figure6_reward/run.sh", "eval"], cwd=REPO, capture_output=True,
                       text=True, timeout=300, env={**os.environ, "DRY_RUN": "1", "LOCAL_OUT": str(tmp_path),
                                                    "EXPR_ROOT": str(tmp_path / "results"),
                                                    "PCBWORLD_DATA_ROOT": str(_data_root(tmp_path)),
                                                    "SEEDS": "42", "WIRES": "0", "VIAS": "0.1"})
    assert r.returncode == 2 and "already holds a cell" in r.stderr, r.stderr[-400:]
    assert "eval/pipeline.py" not in r.stderr


def test_task_boards_list_resolves_through_the_llm_split_aliases(tmp_path):
    """kdd_task_boards_list resolves a task as the LLM recipes do (qs_resolve_split): legacy t-names
    and a QS_SPLITS_CONFIG override work, and a board missing on disk is an error."""
    root = _data_root(tmp_path)
    run = lambda task, **env: subprocess.run(  # noqa: E731
        ["bash", "-c", f'source experiments/_lib/env.sh; kdd_task_boards_list {task} "$OUT"'], cwd=REPO,
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PCBWORLD_DATA_ROOT": str(root), "OUT": str(tmp_path / "list.txt"), **env})
    assert run("t3b").returncode == 0
    assert len((tmp_path / "list.txt").read_text().splitlines()) == 10
    cfg = tmp_path / "splits.json"
    cfg.write_text('{"mine": {"boards_json": "configs/datasets/d3.json", "difficulty": "easy", "boards_split": "test"}}')
    assert run("mine", QS_SPLITS_CONFIG=str(cfg)).returncode == 0
    assert len((tmp_path / "list.txt").read_text().splitlines()) == 99
    Path((tmp_path / "list.txt").read_text().splitlines()[0]).unlink()
    r = run("mine", QS_SPLITS_CONFIG=str(cfg))
    assert r.returncode != 0 and "missing" in r.stderr, r.stderr[-400:]


class _Built(Exception):
    """Raised by the fake PCBWorld once it has seen the env config."""


# The KDD plan-only paths (Table 2 replay, few-shot harvest) build their own env; the paper's
# plan-only cells were rolled out on 2026-05-31 - 06-01, before reject_if_stuck existed.
PLAN_ONLY_ENVS = [
    ("eval_plan_only_llm_v8_standalone.py", "_build_env", ()),
    ("prepare_plan_only_fewshot.py", "replay_and_save", ("", Path("out.kicad_pcb"))),
    ("prepare_plan_only_fewshot.py", "extract_initial_board_state", ()),
    ("prepare_plan_only_fewshot_llm.py", "_board_static_for", ()),
]


LLM_EVAL = REPO / "experiments/kdd/llm_eval"


@pytest.fixture
def import_llm_eval(monkeypatch):
    """Import KDD llm_eval scripts; their module-level sys.path inserts and the sibling modules
    they import are undone after the test."""
    import importlib

    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    yield lambda script: importlib.import_module(f"experiments.kdd.llm_eval.{Path(script).stem}")
    for name in set(sys.modules) - before:
        if (getattr(sys.modules[name], "__file__", None) or "").startswith(str(LLM_EVAL) + os.sep):
            del sys.modules[name]


@pytest.mark.parametrize("script,func,rest", PLAN_ONLY_ENVS, ids=[f"{s}:{f}" for s, f, _ in PLAN_ONLY_ENVS])
def test_plan_only_env_pins_the_paper_knobs(script, func, rest, monkeypatch, import_llm_eval):
    import pcb_world.core.env as env_mod

    seen = {}

    def fake_pcbworld(board, config):
        seen["config"] = config
        raise _Built

    monkeypatch.setattr(env_mod, "PCBWorld", fake_pcbworld)
    mod = import_llm_eval(script)
    with pytest.raises(_Built):
        getattr(mod, func)(Path("board.kicad_pcb"), *rest)
    for key in ("outline_obs", "simplify_outline", "net_constraint_obs", "reject_if_stuck"):
        assert getattr(seen["config"], key) == LEGACY["env"][key], f"{script}:{func} {key}"


def test_synth_fewshot_engine_pins_the_paper_make_line(monkeypatch, import_llm_eval):
    """The engine-free few-shot pool (Table 1 baselines) routes with the paper-era make_line."""
    import pcb_world.engine.kicad_engine as ke

    seen = {}

    def fake_engine(board, *args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(ke, "KiCadEngine", fake_engine)
    mod = import_llm_eval("prepare_synth_fewshot.py")
    with pytest.raises(_Built):
        mod.route_board(Path("board.kicad_pcb"), Path("out.kicad_pcb"))
    assert seen["reject_if_stuck"] is LEGACY["env"]["reject_if_stuck"] is False
    assert seen["simplify_outline"] is False   # the outline kept as stored, as KiCadEngine.for_board_file does
