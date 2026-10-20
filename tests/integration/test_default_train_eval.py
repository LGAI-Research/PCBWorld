"""Default training settings compose with the default eval settings: a PPO run on the
training defaults (``configs/rl.yaml`` ``rl_train`` / ``rl_policy``) writes a checkpoint the
eval pipeline on its defaults (``rl_eval``) loads and scores. The rules that make this hold are
unit-tested in ``tests/test_default_settings.py``."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pcb_world.engine import engine_available

REPO = Path(__file__).resolve().parents[2]
BOARD = REPO / "tests" / "fixtures" / "simple_routing_board.kicad_pcb"


@pytest.mark.slow
@pytest.mark.skipif(not engine_available(), reason="kicad_rl_router binding not built")
def test_default_training_checkpoint_loads_in_default_eval(tmp_path):
    """Only sizes (iterations, steps, envs, device) leave their defaults."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    train = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train.py"), "ppo", "--board", str(BOARD),
         "--iterations", "1", "--n-envs", "1", "--n-steps", "8", "--batch-size", "8", "--max-steps", "8",
         "--device", "cpu", "--save-dir", str(tmp_path / "ckpt"), "--log-dir", str(tmp_path / "tb")],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert train.returncode == 0, train.stdout[-3000:] + train.stderr[-3000:]
    assert "validation: off" in train.stdout

    boards = tmp_path / "boards"
    boards.mkdir()
    for suffix in (".kicad_pcb", ".kicad_pro"):
        shutil.copy(BOARD.with_suffix(suffix), boards)
    ev = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "eval.py"), "--ckpt", str(tmp_path / "ckpt" / "policy_final.pt"),
         "--boards-dir", str(boards), "--n-rollouts", "1", "--seed", "0", "--device", "cpu",
         "--output-dir", str(tmp_path / "eval")],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert ev.returncode == 0, ev.stdout[-3000:] + ev.stderr[-3000:]
    assert "[eval] n_max_slots: 512 -> 1280" in ev.stdout
    assert (tmp_path / "eval" / "eval_overall_summary.json").is_file()
