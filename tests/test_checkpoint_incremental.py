"""Incremental restore (diff-at-restore) — correctness vs full-swap.

restore_incremental updates only the changed tracks in the PNS world (keeping
invariant pad/footprint obstacles) instead of a full ClearWorld+SyncWorld. It
must produce the SAME board as the full-swap restore() — which is the oracle.
This file asserts correctness (toAdd / toRemove / unchanged-skip + full ==
incremental agreement).
"""

import os

import pytest

from pcb_world.engine.kicad_engine import KiCadEngine

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"
PIC = "build_rl/kicad_src/demos/pic_programmer/pic_programmer.kicad_pcb"


def _snap(e):
    """Board + connectivity snapshot. Includes unrouted_count and ratsnest so
    the oracle also validates the incremental connectivity path (ratsnest
    recomputed without a full BuildConnectivity)."""
    tracks = sorted(
        (round(t.x1_mm, 6), round(t.y1_mm, 6), round(t.x2_mm, 6), round(t.y2_mm, 6),
         round(t.width_mm, 6), t.layer, t.net_code)
        for t in e.get_tracks()
    )
    vias = sorted(
        (round(v.x_mm, 6), round(v.y_mm, 6), round(v.diameter_mm, 6),
         round(v.drill_mm, 6), v.top_layer, v.bottom_layer, v.net_code)
        for v in e.get_vias()
    )
    unrouted = e.get_unrouted_count()
    ratsnest = sorted(
        (round(r.x1_mm, 4), round(r.y1_mm, 4), round(r.x2_mm, 4), round(r.y2_mm, 4),
         r.net_code)
        for r in e.get_ratsnest()
    )
    return tracks, vias, unrouted, ratsnest


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


def test_incremental_toadd_with_unchanged(engine):
    """Deleting one of two tracks then restoring re-adds it (toAdd) while the
    other is left in place (unchanged-skip) — same board as the checkpoint."""
    _route(engine, 25.0, 5.0, 25.0, 9.0)
    _route(engine, 25.0, 9.0, 20.0, 9.0)
    s = _snap(engine)
    assert len(s[0]) == 2
    h = engine.checkpoint()
    engine.delete_track_by_index(0)
    assert len(_snap(engine)[0]) == 1
    engine.restore_incremental(h)
    assert _snap(engine) == s
    engine.release_checkpoint(h)


def test_incremental_toremove(engine):
    """A track added after the checkpoint is removed on incremental restore."""
    _route(engine, 25.0, 5.0, 25.0, 9.0)
    s = _snap(engine)
    h = engine.checkpoint()
    _route(engine, 25.0, 9.0, 20.0, 9.0)   # extra track not in the checkpoint
    assert len(_snap(engine)[0]) == 2
    engine.restore_incremental(h)
    assert _snap(engine) == s               # extra removed
    engine.release_checkpoint(h)


def test_full_and_incremental_agree(engine):
    """restore() (full-swap) and restore_incremental() reach the same board."""
    _route(engine, 25.0, 5.0, 25.0, 9.0)
    _route(engine, 25.0, 9.0, 20.0, 9.0)
    s = _snap(engine)
    h = engine.checkpoint()

    engine.delete_track_by_index(0)
    engine.restore(h)                       # full
    s_full = _snap(engine)

    engine.delete_track_by_index(0)
    engine.restore_incremental(h)           # incremental
    s_incr = _snap(engine)

    assert s_full == s == s_incr
    engine.release_checkpoint(h)


def test_incremental_reopens_session(engine):
    """Session re-open works through the incremental path too."""
    engine.start_route(25.0, 5.0, 1)
    h = engine.checkpoint()
    engine.fix_route(25.0, 9.0, True)
    assert not engine.is_routing() and len(_snap(engine)[0]) == 1
    engine.restore_incremental(h)
    assert engine.is_routing()
    assert len(_snap(engine)[0]) == 0
    engine.release_checkpoint(h)


@pytest.mark.skipif(not os.path.exists(PIC), reason="pic_programmer demo not built")
def test_scale_oracle_pic_programmer():
    """On a real board (370 tracks, 247 pads), incremental restore reproduces
    the full-swap board exactly with a many-track 'unchanged' majority."""
    e = KiCadEngine(PIC)
    e.build_connectivity()
    s0 = _snap(e)
    assert len(s0[0]) > 300
    h = e.checkpoint()
    for _ in range(5):
        e.delete_track_by_index(0)
    assert len(_snap(e)[0]) == len(s0[0]) - 5
    e.restore_incremental(h)
    assert _snap(e) == s0                    # 5 re-added, ~365 unchanged

    for _ in range(5):
        e.delete_track_by_index(0)
    e.restore(h)
    s_full = _snap(e)
    assert s_full == s0
    e.release_checkpoint(h)
    e.close()
