"""Regression guards for DRC-cache clearing (env.reset / clear_drc_cache).

Two bugs fixed together (commit "Bug fix. trackwidth setting"):

  1. ``DRCUtils.clear()`` must empty BOTH caches — the object cache
     (``_cached_violations``) AND the dict cache (``_cached_violations_dict``).
  2. ``KiCadEngine.clear_drc_cache()`` — the sole DRC-clearing call made by
     ``PCBWorld.reset()`` — must ALSO clear the Python-side ``drc_helper``, not
     just the C++ incremental-DRC state (``m_drcViolations`` + ``m_drcItemSig``).
     Otherwise the previous episode's DRC results linger until the next
     ``run_drc()`` overwrites them; in terminal-mode training (reset does NOT
     re-run DRC) they leak into the next episode's observation as phantom DRC
     tokens (reward/metrics stay correct — they recompute fresh at the terminal
     step — but the policy's *input* is polluted).
"""

import os
import types

import pytest

from pcb_world.engine import engine_available

from pcb_world.engine.drc import DRCUtils

# A board with inherent DRC violations (11 per its companion .kicad_dru), so we
# never depend on the shove router actually producing an overlap.
VIOLATION_PCB = "tests/fixtures/sample_drc_violation.kicad_pcb"
VIOLATION_DRU = "tests/fixtures/sample_drc_violation.kicad_dru"
# A plain routable board for the env-reset path.
BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def _fake_violation():
    return types.SimpleNamespace(
        error_code=1, error_type="clearance", message="test",
        x_mm=1.0, y_mm=2.0, layer=0, net_names=["NET1"], severity=0x20,
    )


# --------------------------------------------------------------------------
# 1. Pure-Python guard on drc.py — clear() empties BOTH caches. No engine.
# --------------------------------------------------------------------------

class TestDRCUtilsClear:
    def test_clear_empties_both_caches(self):
        helper = DRCUtils()
        helper.update([_fake_violation(), _fake_violation()])
        assert helper._cached_violations != []
        assert helper._cached_violations_dict != []      # populated by update()
        assert helper.get_violation_count() == 2
        assert helper.get_sorted() != []

        helper.clear()
        assert helper._cached_violations == []
        assert helper._cached_violations_dict == []       # BUG guard: dict cache too
        assert helper.get_violation_count() == 0
        assert helper.get_sorted() == []


# --------------------------------------------------------------------------
# Engine / env fixtures (need the built C++ router).
# --------------------------------------------------------------------------

def _skip_if_unavailable(board=VIOLATION_PCB):
    if not os.path.exists(board):
        pytest.skip(f"Board not found: {board}")
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


@pytest.fixture
def violation_engine():
    _skip_if_unavailable(VIOLATION_PCB)
    from pcb_world.engine.kicad_engine import KiCadEngine
    e = KiCadEngine(VIOLATION_PCB)
    e.build_connectivity()
    yield e
    e.close()


# --------------------------------------------------------------------------
# 2. Engine guard — clear_drc_cache() empties the Python drc_helper.
# --------------------------------------------------------------------------

class TestClearDRCCacheClearsHelper:
    def test_clear_drc_cache_clears_python_helper(self, violation_engine):
        violation_engine.run_drc(VIOLATION_DRU)
        assert violation_engine.get_drc_violation_count() > 0, "setup: board must violate DRC"
        assert violation_engine.drc_helper.get_sorted() != []

        # The fix: clear_drc_cache (env.reset's DRC-clearing call) must ALSO empty
        # the Python helper, not just the C++ incremental state. Pre-fix this line
        # left the helper untouched -> the following asserts would fail.
        violation_engine.clear_drc_cache()
        assert violation_engine.get_drc_violation_count() == 0
        assert violation_engine.drc_helper.get_violations() == []
        assert violation_engine.drc_helper.get_sorted() == []
        assert violation_engine.drc_helper._cached_violations_dict == []


# --------------------------------------------------------------------------
# 3. End-to-end guard — env.reset() drops the previous episode's DRC so no
#    phantom tokens can leak into the next observation.  Uses a terminal-mode
#    DRC config (run_drc_on_reset is False), so the ONLY thing that clears the
#    helper on reset is the fix — a reset-time full DRC does not mask it.
# --------------------------------------------------------------------------

class TestResetClearsStaleDRC:
    def test_reset_clears_python_drc_helper(self):
        _skip_if_unavailable(BOARD)
        from pcb_world.core.env import PCBWorld
        env = PCBWorld(
            board_path=BOARD, max_steps=200,
            masking_rule="strict_no_finish", reward_rule="drc_sparse_error",
        )
        try:
            env.reset()
            # terminal mode => reset does NOT re-run DRC (would otherwise mask the fix)
            assert env._reward.run_drc_on_reset is False
            eng = env._engine

            # Simulate a stale helper as if the previous episode ended with
            # violations (inject directly so the guard never depends on the
            # shove router actually producing an overlap).
            eng.drc_helper.update([_fake_violation(), _fake_violation()])
            assert eng.get_drc_violation_count() == 2

            env.reset()  # must clear the Python helper (bug: stale carry-over)
            assert eng.get_drc_violation_count() == 0
            assert eng.drc_helper.get_sorted() == []
        finally:
            env.close()
