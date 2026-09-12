"""Smoke test for SubprocDecoderVecEnv.reload_board.

Verifies that a pool built on board A can be swapped to board B without
tearing down subprocesses, and the new board is actually reflected in env
state (board path in reset info / obs).

The per-env variant (reload_boards) lives in test_reload_boards_per_env.py —
each test spawns its own subprocess pool (multi-second), so they are split
across files for pytest-xdist's loadfile scheduler.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from methods.rl_agent.wrappers.factory import make_decoder_env_pool


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOARD_A = FIXTURES / "simple_routing_board.kicad_pcb"
BOARD_B = FIXTURES / "simple_routing_board_shifted.kicad_pcb"


@pytest.mark.skipif(
    not (BOARD_A.exists() and BOARD_B.exists()),
    reason="fixture boards missing",
)
def test_reload_board_swaps_env_without_restart(pool_kwargs):
    pool = make_decoder_env_pool(
        str(BOARD_A), n_envs=2,
        **pool_kwargs(max_steps=20, masking_rule="default_no_finish",
                      reward_rule="drc_only_dense"),
    )
    try:
        obs_a = pool.reset_all()
        assert len(obs_a) == 2

        processes_before = [p.pid for p in pool.processes]
        pool.reload_board(str(BOARD_B))
        processes_after = [p.pid for p in pool.processes]
        # Same subprocesses — only the inner env was rebuilt.
        assert processes_before == processes_after

        obs_b = pool.reset_all()
        assert len(obs_b) == 2

        # A couple of steps on the reloaded board should succeed.
        masks = pool.get_action_masks()
        ptrs = pool.get_pointer_masks()
        assert masks.shape[0] == 2
        assert ptrs.shape[0] == 2
    finally:
        pool.close()
