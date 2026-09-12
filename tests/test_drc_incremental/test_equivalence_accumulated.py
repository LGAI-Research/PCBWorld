"""C++ run_drc_incremental() == run_drc(): accumulated snapshots and the real
RL ADD action (make_line commit path). Per-step variant: test_equivalence.py
(loadfile split)."""
from pcb_world.engine.kicad_engine import KiCadEngine
from tests.helpers.drc_keying import _assert_eq, _full, _incr


def test_run_drc_incremental_accumulated(engine):
    engine.run_drc()  # baseline
    for _ in range(15):
        assert engine.delete_track_by_index(0)
        engine.run_drc_incremental()        # accumulate snapshot, no full in between
    incr = engine.get_drc_violations()
    full = _full(engine)
    _assert_eq(incr, full, "accumulated")


def test_run_drc_incremental_make_line():
    """Routing a new track (the real RL ADD action) → incremental == full, end-to-end.
    Exercises the commit path (fix_route) feeding the snapshot diff, on a routable board."""
    e = KiCadEngine("tests/fixtures/simple_routing_board.kicad_pcb")
    e.build_connectivity()
    try:
        e.run_drc()                                    # baseline + snapshot
        e.start_route(25.0, 5.0, 1); e.fix_route(25.0, 9.0, True)
        _assert_eq(_incr(e), _full(e), "make_line #1")
        e.start_route(25.0, 9.0, 1); e.fix_route(20.0, 9.0, True)
        _assert_eq(_incr(e), _full(e), "make_line #2")
    finally:
        if e.is_routing():
            e.cancel_route()
        e.close()
