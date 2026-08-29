"""reload_board: a board_factory failure must surface the ORIGINAL exception.

The worker's except-handler must not itself fail — e.g. by referencing ``env``
after ``del env``. If it does, a factory exception turns into an
UnboundLocalError inside the handler, the worker dies uncaught, and the parent
sees only EOF / a silent respawn, leaving the real cause recoverable only from
var/crashlogs stderr files.

Own file: spawns a subprocess pool (multi-second) — pytest-xdist loadfile.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from methods.rl_agent.wrappers.factory import make_decoder_env_pool


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOARD_A = FIXTURES / "simple_routing_board.kicad_pcb"
MISSING = FIXTURES / "does_not_exist_260825.kicad_pcb"


@pytest.mark.skipif(not BOARD_A.exists(), reason="fixture board missing")
def test_reload_board_factory_error_is_reported(pool_kwargs):
    pool = make_decoder_env_pool(
        str(BOARD_A), n_envs=1,
        **pool_kwargs(max_steps=20, masking_rule="default_no_finish",
                      reward_rule="drc_only_dense"),
    )
    try:
        pool.reset_all()
        pids_before = [p.pid for p in pool.processes]
        # The factory raises inside the worker; the parent must re-raise it
        # with the worker traceback (not swallow it as a dead-worker respawn).
        with pytest.raises(RuntimeError, match="Worker exception") as ei:
            pool.reload_board(str(MISSING))
        msg = str(ei.value)
        assert "UnboundLocalError" not in msg
        assert MISSING.name in msg
        # The worker process itself survived (no crash → no respawn).
        assert [p.pid for p in pool.processes] == pids_before
    finally:
        pool.close()
