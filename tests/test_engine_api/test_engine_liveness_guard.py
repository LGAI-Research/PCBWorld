"""One-live-RLRouter-per-process guard (``_assert_no_live_router``).

Two live routers in one process share KiCad global state (the PNS::ROUTER
singleton, BOARD/VIA aliasing): the stale one's late destruction nulls the live
router's singleton and segfaults it mid-routing (260722 MCTS crash). The guard
turns that into deterministic, attributable failures at construction time:

  - a second live engine → RuntimeError naming the offender's creation site
  - an engine dropped without close() (GC cycle) → reclaimed before the new
    router exists, with a loud RuntimeWarning naming the leak
  - close()-then-create → clean, no warning
  - INTENDED coexistence (trainer --no-vecenv list mode, comparison tests) →
    explicit opt-in via ``allow_router_coexistence(reason)`` at the call site
"""
from __future__ import annotations

import gc
import warnings

import pytest

from pcb_world.engine.kicad_engine import KiCadEngine, allow_router_coexistence

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


@pytest.fixture(autouse=True)
def _flush_prior_leaks():
    """Reclaim engines leaked by earlier tests in this worker, so each scenario
    starts from zero live routers (their rescue warning is not ours to assert)."""
    gc.collect()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = KiCadEngine(BOARD)
        e.close()
    yield


def test_second_live_engine_raises():
    e1 = KiCadEngine(BOARD)
    try:
        with pytest.raises(RuntimeError, match="one live router per process"):
            KiCadEngine(BOARD)
    finally:
        e1.close()


def test_close_then_create_is_clean():
    e1 = KiCadEngine(BOARD)
    e1.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        e2 = KiCadEngine(BOARD)
    e2.close()


def test_leaked_engine_is_reclaimed_with_loud_warning():
    e1 = KiCadEngine(BOARD)
    e1._cycle = e1          # reference cycle: refcount alone can never free it
    del e1                  # dropped without close() — the leak under test
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e2 = KiCadEngine(BOARD)
    e2.close()
    rescued = [w for w in caught if "dropped without close()" in str(w.message)]
    assert rescued, "leak was not reported"
    assert "created at" in str(rescued[0].message)   # names the offender


def test_explicit_coexistence_opt_in():
    with allow_router_coexistence("guard test: intentional side-by-side pair"):
        e1 = KiCadEngine(BOARD)
        e2 = KiCadEngine(BOARD)   # sanctioned — must not raise
    e1.close()
    e2.close()
    # outside the scope the contract is strict again
    e3 = KiCadEngine(BOARD)
    with pytest.raises(RuntimeError, match="one live router per process"):
        KiCadEngine(BOARD)
    e3.close()


def test_coexistence_requires_reason():
    with pytest.raises(ValueError, match="non-empty reason"):
        with allow_router_coexistence(""):
            pass
