"""Contract of the KDD paper recipes against the shipped defaults.

The paper's rules live under ``experiments/kdd/configs/`` and resolve by name, but the
training KNOBS whose shipped defaults moved on after the paper (observation tokens, outline
representation, directional candidates, time feature, PPO truncation bootstrap) are only
paper-faithful if the recipe shells pass them explicitly (``configs/paper_train_flags.sh``).
The v1.0.0 restart nearly shipped Table 1 PPO-terminal broken exactly this way: the recipe
relied on ``truncation_bootstrap`` being on by default, and the reward rule
(``truncation_mode: none``) refuses to train without it.

Each test runs a recipe shell in ``--dry-run`` (it echoes the assembled trainer command and
returns — milliseconds, no GPU / engine / dataset), parses that command with the trainer's own
argparse and checks the resolved values against the paper-era table
``experiments/kdd/configs/legacy_ckpt_defaults.yaml`` and the trainer's truncation rule.
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
LEGACY = yaml.safe_load((REPO / "experiments/kdd/configs/legacy_ckpt_defaults.yaml").read_text())
# The knobs the recipes must pin (the paper-era values), by section of legacy_ckpt_defaults.yaml.
PINNED_ENV = ("outline_obs", "simplify_outline", "net_constraint_obs", "directional_candidates", "offboard_mask")
PINNED_POLICY = ("time_feature", "obstacle_obs")

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
    else:
        from methods.rl_agent.training.train_ppo import build_arg_parser
    return build_arg_parser().parse_args(flags)


@pytest.mark.parametrize("name,recipe", RECIPES, ids=[r[0] for r in RECIPES])
def test_recipe_pins_the_paper_knobs(name, recipe, tmp_path):
    from configs.loader.schema import RLEnvConfig
    from pcb_world.core.masking import get_masking_rule
    from pcb_world.core.reward_config import get_reward_config

    module, *flags = _launch_argv(recipe, tmp_path)
    args = _parse(module, flags)
    env = RLEnvConfig.from_namespace(args).to_pool_kwargs()
    for key in PINNED_ENV:
        assert env[key] == LEGACY["env"][key], f"{name}: {key}={env[key]!r}, paper value {LEGACY['env'][key]!r}"
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
