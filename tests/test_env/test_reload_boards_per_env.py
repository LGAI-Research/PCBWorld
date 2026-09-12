"""Smoke test for SubprocDecoderVecEnv.reload_boards (per-env variant).

Split from test_reload_board.py — each test spawns its own subprocess pool
(multi-second), so they are split across files for pytest-xdist's loadfile
scheduler.
"""
from __future__ import annotations

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
def test_reload_boards_per_env_different_boards(pool_kwargs):
    """Each worker reloads to a DIFFERENT board concurrently.

    Guards against the prior KiCad engine failure mode (RLRouter pointer
    aliasing across workers) and confirms per-env curriculum is viable.
    """
    pool = make_decoder_env_pool(
        str(BOARD_A), n_envs=2,
        **pool_kwargs(max_steps=20, masking_rule="default_no_finish",
                      reward_rule="drc_only_dense"),
    )
    try:
        pool.reset_all()
        pids_before = [p.pid for p in pool.processes]

        pool.reload_boards([str(BOARD_A), str(BOARD_B)])

        pids_after = [p.pid for p in pool.processes]
        assert pids_before == pids_after, "workers must not be respawned"

        obs = pool.reset_all()
        assert len(obs) == 2
        masks = pool.get_action_masks()
        ptrs = pool.get_pointer_masks()
        assert masks.shape[0] == 2 and ptrs.shape[0] == 2

        pool.step_async(pool.get_pointer_masks())  # pointer indices as actions
        # Don't assert on step result — just ensure no crash/hang.
        pool.step_wait()
    finally:
        pool.close()


@pytest.mark.skipif(
    not (BOARD_A.exists() and BOARD_B.exists()),
    reason="fixture boards missing",
)
def test_reload_boards_respawns_dead_worker(pool_kwargs):
    """A worker killed before ``reload_boards`` is respawned on its NEW board.

    Regression guard for the 260803 campaign crash: a worker segfault landing
    between iterations made ``reload_boards`` raise EOFError and killed the
    whole trainer (recovery only covered the step path).
    """
    import os
    import signal

    pool = make_decoder_env_pool(
        str(BOARD_A), n_envs=2,
        **pool_kwargs(max_steps=20, masking_rule="default_no_finish",
                      reward_rule="drc_only_dense"),
    )
    try:
        pool.reset_all()
        victim = pool.processes[1]
        os.kill(victim.pid, signal.SIGKILL)
        victim.join(timeout=5)

        pool.reload_boards([str(BOARD_A), str(BOARD_B)])  # must not raise

        assert pool.processes[1].pid != victim.pid, "dead worker must be respawned"
        assert pool.processes[1].is_alive()
        obs = pool.reset_all()
        assert len(obs) == 2
        masks = pool.get_action_masks()
        assert masks.shape[0] == 2
    finally:
        pool.close()
