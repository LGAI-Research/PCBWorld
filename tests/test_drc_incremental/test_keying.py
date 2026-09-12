"""KIID-keying foundation (Python-simulated recipe): clearance keys are stable
values, and clearance is pairwise-local (retain + scoped recompute == full)."""
from tests.helpers.drc_keying import _clr, _full, _is_clr, _items, _key


def test_clearance_keys_stable_across_restore(engine):
    """Clearance keys are bit-stable across a remove + restore_incremental cycle."""
    keys0 = _clr(_full(engine))
    assert keys0 and all(v.item_a for v in engine.get_drc_violations())
    cp0 = engine.checkpoint()
    for _ in range(30):
        assert engine.delete_track_by_index(0)
    assert _clr(_full(engine)) != keys0
    engine.restore_incremental(cp0)
    assert _clr(_full(engine)) == keys0


def test_clearance_is_pairwise_local(engine):
    """retain(clearance not involving changed) + recompute(changed) == full clearance,
    both directions (REMOVE delete, ADD restore). Connectivity is global (excluded)."""
    viols_a = _full(engine)
    clr_a = _clr(viols_a)
    tracks_a = {t.uuid for t in engine.get_tracks()}
    cp_a = engine.checkpoint()
    assert clr_a

    for _ in range(30):
        assert engine.delete_track_by_index(0)
    viols_b = _full(engine)
    tracks_b = {t.uuid for t in engine.get_tracks()}
    removed = tracks_a - tracks_b
    retain = {_key(v) for v in viols_a if _is_clr(v) and not (_items(v) & removed)}
    assert retain == _clr(viols_b), "REMOVE clearance not pairwise-local"

    engine.restore_incremental(cp_a)
    viols_a2 = _full(engine)
    readded = {t.uuid for t in engine.get_tracks()} - tracks_b
    retain2 = {_key(v) for v in viols_b if _is_clr(v) and not (_items(v) & readded)}
    fresh2 = {_key(v) for v in viols_a2 if _is_clr(v) and (_items(v) & readded)}
    assert (retain2 | fresh2) == _clr(viols_a2), "ADD clearance merge != full"
