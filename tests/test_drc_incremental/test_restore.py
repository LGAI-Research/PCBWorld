"""Incremental DRC across checkpoint()/restore_incremental(): the DRC state
survives a restore and the following incremental still equals full."""
from tests.helpers.drc_keying import _assert_eq, _clr, _full, _incr


def test_run_drc_incremental_remove_then_restore(engine):
    engine.run_drc()
    cp = engine.checkpoint()
    for _ in range(20):
        assert engine.delete_track_by_index(0)
    engine.run_drc_incremental()            # remove
    engine.restore_incremental(cp)
    incr = _incr(engine)                     # add back
    full = _full(engine)
    _assert_eq(incr, full, "remove+restore")


def test_drc_state_survives_restore(engine):
    """checkpoint() stores the DRC state; restore brings it back verbatim (not cleared),
    so it's available immediately after a restore without re-running DRC — and a following
    incremental DRC retains instead of recomputing the whole board. Without this, the
    retain set is empty after a restore and run_drc_incremental drops the unchanged
    violations."""
    keys0 = _clr(_full(engine))            # baseline DRC + snapshot
    n0 = len(engine.get_drc_violations())
    assert keys0
    cp = engine.checkpoint()
    # Replace the live DRC state with a clearly different one (delete enough tracks to
    # change the clearance set, so the restore below is not a vacuous no-op).
    for _ in range(300):
        assert engine.delete_track_by_index(0)
    assert _clr(_full(engine)) != keys0, "deletes did not change DRC — test is vacuous"
    # Restore must bring the checkpoint's DRC state back verbatim, WITHOUT any run_drc.
    engine.restore_incremental(cp)
    restored = engine.get_drc_violations()
    assert len(restored) == n0, "restore did not bring back the checkpoint DRC state"
    assert _clr(restored) == keys0
    # The restored snapshot is consistent: a following incremental still equals full.
    _assert_eq(_incr(engine), _full(engine), "post-restore incremental")
