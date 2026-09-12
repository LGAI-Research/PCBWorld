"""Route-AFTER-restore equivalence — guards the virtual-via (VVIA) regeneration
that a full ``restore()`` gets for free via ClearWorld+SyncWorld+FixupVirtualVias
but ``restore_incremental()`` must reproduce explicitly.

VVIAs are a *derived* projection of the joint topology (width-change / T-junction /
locked joints), not board items, so the incremental track-diff never touches them.
Before the fix, a VVIA minted for the pre-restore state lingered as a phantom
obstacle and deflected the FIRST route done after an incremental restore — invisible
to a board *snapshot* (VVIAs aren't board vias), so the existing snapshot-equality
tests could not catch it. These tests route a net *after* each restore path and
assert the resulting geometry matches, which is exactly what exercises VVIAs.
"""

import pytest

from pcb_world.engine.kicad_engine import KiCadEngine

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def _snap(e):
    tracks = sorted(
        (round(t.x1_mm, 4), round(t.y1_mm, 4), round(t.x2_mm, 4), round(t.y2_mm, 4),
         round(t.width_mm, 4), t.layer, t.net_code)
        for t in e.get_tracks()
    )
    vias = sorted(
        (round(v.x_mm, 4), round(v.y_mm, 4), round(v.diameter_mm, 4), v.net_code)
        for v in e.get_vias()
    )
    return tracks, vias


def _route(e, sx, sy, tx, ty, layer=1):
    e.start_route(sx, sy, layer)
    e.fix_route(tx, ty, True)


@pytest.fixture
def engine():
    e = KiCadEngine(BOARD)
    e.build_connectivity()
    yield e
    if e.is_routing():
        e.cancel_route()
    e.close()


def _route_probe(e):
    """Route NET3 straight through (25, 10) — the point where the width-change
    joint (and thus a VVIA) is created below."""
    e.set_track_width(0.25)
    _route(e, 25.0, 5.0, 25.0, 25.0, layer=1)
    return _snap(e)


def _make_width_change_vvia(e):
    """Route NET1 as two different-width segments meeting at (25, 10): the joint
    there is a width-change joint, so FixupVirtualVias mints a VVIA at (25, 10)."""
    e.set_track_width(0.20)
    _route(e, 10.0, 10.0, 25.0, 10.0)
    e.set_track_width(0.40)
    _route(e, 25.0, 10.0, 40.0, 10.0)


def test_route_after_incremental_matches_full(engine):
    """A route done after restore_incremental() must be identical to the same
    route after the full-swap restore() — i.e. the VVIA set is regenerated so no
    phantom obstacle from the pre-restore state deflects it."""
    h0 = engine.checkpoint()                 # empty board: correct VVIA set = none
    assert _snap(engine)[0] == []

    _make_width_change_vvia(engine)          # mints a VVIA at (25, 10)
    assert len(_snap(engine)[0]) == 2

    engine.restore_incremental(h0)           # VVIA must be dropped (was stale)
    assert _snap(engine)[0] == []
    inc = _route_probe(engine)

    engine.restore(h0)                       # full-swap oracle (clean)
    assert _snap(engine)[0] == []
    full = _route_probe(engine)

    engine.release_checkpoint(h0)
    assert inc == full


def test_incremental_regenerates_vvia_when_it_should_exist(engine):
    """The mirror case: restoring *to* a state that legitimately has a VVIA must
    reinstate it, so a route that should be deflected is deflected identically on
    both paths (guards against over-deletion / missing regeneration)."""
    _make_width_change_vvia(engine)          # checkpoint state HAS a VVIA at (25,10)
    h = engine.checkpoint()

    engine.restore_incremental(h)
    inc = _route_probe(engine)

    engine.restore(h)
    full = _route_probe(engine)

    engine.release_checkpoint(h)
    assert inc == full
