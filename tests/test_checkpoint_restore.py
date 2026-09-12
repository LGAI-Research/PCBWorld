"""Checkpoint / restore round-trip oracle (MCTS tree search).

Stage-1 full-swap baseline correctness gate. Future incremental optimizations
(diff-at-restore, stored-delta) must keep these passing.

Covers: identity, revert, stale-handle no-op, multi-handle, mid-route session
re-open, and the reset+replay vs checkpoint/restore oracle.

NOTE: KiCadEngine aliases KiCad global state, so only ONE RLRouter may be alive
at a time — every engine is closed before the next is created.
"""

import pytest

from pcb_world.engine.kicad_engine import KiCadEngine

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def _snap(e):
    """Order-independent geometry+net+layer snapshot of tracks and vias."""
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
    return tracks, vias


def _route_track(e, sx, sy, tx, ty, layer=1):
    e.start_route(sx, sy, layer)
    e.fix_route(tx, ty, True)


@pytest.fixture
def engine():
    e = KiCadEngine(BOARD)
    e.build_connectivity()
    yield e
    if e.is_routing():
        e.cancel_route()
    if e.is_dragging():
        e.cancel_drag()
    e.close()


def test_identity(engine):
    """restore right after checkpoint leaves the board unchanged."""
    _route_track(engine, 25.0, 5.0, 25.0, 9.0)
    a = _snap(engine)
    assert len(a[0]) == 1
    h = engine.checkpoint()
    engine.restore(h)
    assert _snap(engine) == a


def test_revert(engine):
    """checkpoint -> mutate -> restore brings the board back exactly."""
    _route_track(engine, 25.0, 5.0, 25.0, 9.0)
    a = _snap(engine)
    h = engine.checkpoint()
    engine.delete_track_by_index(0)
    assert _snap(engine) != a
    engine.restore(h)
    assert _snap(engine) == a


def test_stale_handle_is_noop(engine):
    """restore / release on an already-released handle leaves the board unchanged,
    and restore() reports it (False) so the caller can detect the stale handle."""
    _route_track(engine, 25.0, 5.0, 25.0, 9.0)
    h = engine.checkpoint()
    engine.release_checkpoint(h)
    before = _snap(engine)
    assert engine.restore(h) is False        # released -> not restored, board unchanged
    assert engine.has_checkpoint(h) is False
    engine.release_checkpoint(h)             # double release -> no-op
    assert _snap(engine) == before


def test_handles_unique_and_reset_invalidates(engine):
    """Handles are globally unique; reset_checkpoints() invalidates every prior handle
    with NO aliasing (a new handle never reuses an old idx), and has_checkpoint /
    restore-return expose validity. Guards the RL 'reset then re-create' footgun."""
    _route_track(engine, 25.0, 5.0, 25.0, 9.0)
    h1 = engine.checkpoint()
    h2 = engine.checkpoint()
    assert h1 != h2 and engine.has_checkpoint(h1) and engine.has_checkpoint(h2)
    assert engine.restore(h1) is True

    engine.reset_checkpoints()
    assert not engine.has_checkpoint(h1) and not engine.has_checkpoint(h2)
    before = _snap(engine)
    assert engine.restore(h1) is False               # stale -> board unchanged
    assert engine.restore_incremental(h2) is False
    assert _snap(engine) == before

    h3 = engine.checkpoint()                          # re-create after reset
    assert h3 != h1 and h3 != h2                      # never aliases an old handle
    assert engine.has_checkpoint(h3) and engine.restore(h3) is True


def test_multi_handle(engine):
    """Two live checkpoints restore independently, in arbitrary order."""
    h0 = engine.checkpoint()                  # empty board
    _route_track(engine, 25.0, 5.0, 25.0, 9.0)
    s1 = _snap(engine)
    h1 = engine.checkpoint()                  # 1 track
    engine.delete_track_by_index(0)
    assert len(_snap(engine)[0]) == 0
    engine.restore(h1)
    assert _snap(engine) == s1
    engine.restore(h0)
    assert len(_snap(engine)[0]) == 0
    engine.release_checkpoint(h0)
    engine.release_checkpoint(h1)


def test_session_reopen(engine):
    """checkpoint while is_routing -> restore re-opens the session and reverts
    the board; finishing the re-opened session reproduces the track."""
    engine.start_route(25.0, 5.0, 1)
    assert engine.is_routing()
    h = engine.checkpoint()
    engine.fix_route(25.0, 9.0, True)          # commit -> session ends, 1 track
    assert not engine.is_routing()
    assert len(_snap(engine)[0]) == 1
    engine.restore(h)
    assert engine.is_routing()                 # session re-opened
    assert len(_snap(engine)[0]) == 0          # board reverted
    engine.fix_route(25.0, 9.0, True)          # finish re-opened session
    assert len(_snap(engine)[0]) == 1
    engine.release_checkpoint(h)


def test_replay_vs_checkpoint_oracle():
    """reset+replay and checkpoint/restore reach the same final board.

    Two engines run sequentially (only one RLRouter alive at a time).
    """
    er = KiCadEngine(BOARD)
    er.build_connectivity()
    _route_track(er, 25.0, 5.0, 25.0, 9.0)
    _route_track(er, 25.0, 9.0, 20.0, 9.0)
    full_replay = _snap(er)
    if er.is_routing():
        er.cancel_route()
    er.close()

    ec = KiCadEngine(BOARD)
    ec.build_connectivity()
    _route_track(ec, 25.0, 5.0, 25.0, 9.0)
    h = ec.checkpoint()
    _route_track(ec, 25.0, 9.0, 20.0, 9.0)
    full_ckpt = _snap(ec)
    ec.restore(h)
    _route_track(ec, 25.0, 9.0, 20.0, 9.0)     # re-route after restore
    after_restore = _snap(ec)
    ec.release_checkpoint(h)
    if ec.is_routing():
        ec.cancel_route()
    ec.close()

    assert after_restore == full_ckpt          # route-after-restore reproduces
    assert full_replay == full_ckpt            # replay == checkpoint/restore
