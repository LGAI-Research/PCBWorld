"""Startup smoke for the training front door — ``experiments/train.py`` dispatch.

Nothing exercised the paper-reproduction entrypoint or the ``train_policy.sh``
recipe it execs, so a startup regression could ship unseen: the recipe shell
parses its arguments against a fixed whitelist and exits 2 on anything else, so
a flag a newcomer is told to add (once, ``--expect-env-diff``) cannot even be
passed, and a broken ``cases.sh`` lookup would surface only in a real run.

``--dry-run`` stops at the assembled launch command, so this costs milliseconds
and needs no GPU, engine or dataset. The other half of the same startup path —
the env-contract gate that used to halt every stock launch — is pinned by
``test_env_contract.test_harness_seed_diff_needs_no_declaration``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("method", ["ppo_per_step", "ppo_terminal", "grpo"])
def test_table1_dispatch_reaches_a_launch(method, tmp_path):
    proc = subprocess.run(
        [sys.executable, "experiments/train.py", "table1",
         "--method", method, "--seed", "42",
         "--output-root", str(tmp_path), "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
        env={**os.environ, "WANDB": "0"},
    )
    assert proc.returncode == 0, (
        f"dispatch failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    # public_cuda_run echoes the resolved command to stderr, then returns.
    launch = proc.stderr
    module = "train_grpo" if method == "grpo" else "train_ppo"
    assert f"methods.rl_agent.training.{module}" in launch, launch
    assert "--seed 42" in launch, launch
    assert "--boards-order per_env_epoch" in launch, launch
