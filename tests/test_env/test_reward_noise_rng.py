"""``reward_noise_std`` must draw from the env's own RNG, not global numpy.

Two defects in one line: ``np.random.normal`` is the legacy PROCESS-GLOBAL
generator, so (a) the noise was unseeded and unreproducible, and (b) every
worker forked from one forkserver shared the same global state, which would
hand all envs the same noise sequence. The draw now comes from the env's
gymnasium ``np_random``, seeded at construction.

Terminal-mode reward only (the noise sits in the terminal Φ branch), so these
use ``drc_only_sparse`` with ``max_steps=1`` — one step truncates and pays it.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from tests.helpers.pro_sidecar import write_default_pro
from tests.test_env.test_action_class_empty import BOARD_TEMPLATE

ACT_NET_SELECT = 0

NOISE_STD = 5.0


def _skip_if_no_kicad():
    pytest.importorskip("kicad_rl_router")


def _truncating_reward(board: str, *, seed: int | None) -> float:
    """Run one step (=truncation at max_steps=1) and return its reward."""
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(
        board_path=board,
        reward_rule="drc_only_sparse",
        reward_noise_std=NOISE_STD,
        max_steps=1,
        seed=seed,
    )
    try:
        env.reset()
        _, reward, _, truncated, _ = env.step(
            {"action_type": ACT_NET_SELECT, "net_id": 1}
        )
        assert truncated, "max_steps=1 must truncate on the first step"
        return float(reward)
    finally:
        env.close()


@pytest.fixture
def board(tmp_path):
    _skip_if_no_kicad()
    p = tmp_path / "two_pad_net1.kicad_pcb"
    p.write_text(BOARD_TEMPLATE)
    write_default_pro(p)   # engine load contract: pro sibling required
    return str(p)


def test_noise_reproducible_from_ctor_seed(board):
    """Same ctor seed -> same terminal noise; different seeds -> different."""
    assert _truncating_reward(board, seed=3) == _truncating_reward(board, seed=3)
    rewards = {_truncating_reward(board, seed=s) for s in range(5)}
    assert len(rewards) > 1, f"noise did not vary with the seed: {rewards}"


def test_noise_ignores_global_numpy_seed(board):
    """Reseeding global numpy must not move the draw — that is the shared
    forkserver state the env RNG replaces."""
    np.random.seed(0)
    a = _truncating_reward(board, seed=7)
    np.random.seed(12345)
    b = _truncating_reward(board, seed=7)
    assert a == b
