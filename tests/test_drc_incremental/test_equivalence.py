"""C++ run_drc_incremental() == run_drc(), per-step (full DRC after every delete).
Accumulated / make_line variants: test_equivalence_accumulated.py (loadfile split)."""
from tests.helpers.drc_keying import _assert_eq, _full, _incr


def test_run_drc_incremental_perstep(engine):
    engine.run_drc()  # baseline + snapshot
    for i in range(15):
        assert engine.delete_track_by_index(0)
        incr = _incr(engine)
        full = _full(engine)   # ground truth (also resets snapshot)
        _assert_eq(incr, full, f"step {i}")
