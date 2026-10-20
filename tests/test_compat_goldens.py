"""2026-10-18 compatibility audit, as fast regression tests (no engine, no weights):

* real KDD checkpoints (args + weight shapes, tests/fixtures/ckpt_real/) resolve to their golden
  configs — eval load, ``--resume``, ``--init-ckpt`` (a resume took the current defaults for the
  paper's inverted training switches) — and the checkpoint the resume writes reads the same through
  ``args`` as through ``config``;
* real KDD launcher command lines (tests/fixtures/recipe_argv/) parse and resolve to their golden values
  (the LLM v7 one pins the whole env config the run.sh ``--masking`` path builds).

Regenerate a golden: tests/helpers/ckpt_golden.py, tests/helpers/recipe_golden.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import ckpt_golden, recipe_golden

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("path", ckpt_golden.golden_params(FIXTURES / "ckpt_real"))
def test_real_checkpoint_resolves_to_its_golden_config(path, tmp_path):
    ckpt_golden.check(path, tmp_path)


@pytest.mark.parametrize("path", ckpt_golden.golden_params(FIXTURES / "recipe_argv"))
def test_launcher_command_resolves_to_its_golden_config(path, tmp_path):
    recipe_golden.check(path, FIXTURES / "ckpt_real", tmp_path)


def test_a_grpo_checkpoint_without_its_marker_is_read_as_grpo(tmp_path):
    """The KDD GRPO checkpoints predate the ``trainer`` marker: the watchers' config resolve
    reads them through the GRPO parser (their ``group_size`` says so), not PPO's, which dropped it."""
    import json
    import warnings

    from methods.rl_agent.training.args import resolve_ckpt_train_config
    from tests.helpers.ckpt_payload import save_ckpt

    fx = json.loads((FIXTURES / "ckpt_real" / "kdd_2l_grpo_s42.json").read_text())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = resolve_ckpt_train_config(save_ckpt(tmp_path / "ckpt.pt", fx), strict=False)
    assert run.group_size == fx["args"]["group_size"]
    assert not [w for w in caught if "not options of this trainer" in str(w.message)]
