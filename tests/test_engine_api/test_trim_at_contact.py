"""``fix_route(trim_at_contact=True)`` — no new copper on top of existing same-net copper.

PNS's walkaround ignores same-net copper, so a new route can run on top of an
existing same-net track (overlapping copper) and can end inside one without a
joint there — stock KiCad avoids both only because a human clicks on the track and
the router splits it at that end item. With ``trim_at_contact`` the engine edits
the fixed route before the commit: stretches lying collinearly on existing same-net
copper are removed (that copper already carries the connection) and existing
segments are split wherever a kept route end lands inside them. The route is never
recomputed.

Boards (all pads NET1, A–C routed first as one straight track on y=10):
``shared_path_board`` A(10,10) C(40,10) B(25,20);
``bridge_exit_board`` A(10,10) C(40,10) B(20,20) D(45,10);
``bridge_multi_board`` A(10,10) C(40,10) B(15,20) D(35,20).
``offgrid_track_board`` (shared_path pads, A–C pre-routed 1 nm off y=10) pins the nm
slack; copies with other pre-routed tracks are written per test.
``two_net_multiterm_board`` (NET1 A(10,10) C(40,10) P(25,5); NET2 pads on y≥20) is the
shove case: a NET2 route pushes NET1 aside.
"""

import os
import re

import pytest

from pcb_world.core.action import finish, make_line, make_via
from tests.helpers.prerouted_board import offgrid_board_with_tracks
from tests.helpers.engine import open_engine

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
SHOVE, WALKAROUND = 1, 2   # routing_mode: PNS::RM_Shove / RM_Walkaround


def _segs(eng, net=None):
    return sorted(
        (round(t.x1_mm, 3), round(t.y1_mm, 3), round(t.x2_mm, 3), round(t.y2_mm, 3))
        for t in eng.get_tracks() if net is None or t.net_code == net
    )


def _nm_segs(eng):
    return sorted((round(t.x1_mm * 1e6), round(t.y1_mm * 1e6), round(t.x2_mm * 1e6), round(t.y2_mm * 1e6))
                  for t in eng.get_tracks())


@pytest.fixture(params=[False, True], ids=["trim_off", "trim_on"])
def trim(request):
    return request.param


@pytest.fixture
def open_board(trim):
    """Open ``<name>.kicad_pcb`` with A–C already routed along y=10."""
    engines = []

    def _open(name):
        e = open_engine(
            os.path.join(FIXTURES, f"{name}.kicad_pcb"),
            trim_at_contact=trim,
        )
        engines.append(e)
        e.build_connectivity()
        assert e.start_route(10.0, 10.0, 1)
        assert make_line(e, 40.0, 10.0)[0]
        return e

    yield _open
    for e in engines:
        if e.is_routing():
            e.cancel_route()
        e.close()


def _assert_connected(eng):
    eng.build_connectivity()
    assert eng.get_unrouted_count() == 0


def test_route_running_onto_track_to_its_end(open_board, trim):
    """B→C, diagonal first, meets y=10 at (35,10) and runs on top of A–C to C:
    the overlapping stretch goes, A–C is split at (35,10)."""
    eng = open_board("shared_path_board")
    assert eng.start_route(25.0, 20.0, 1)
    eng.flip_posture()   # diagonal-first posture: (25,20)→(35,10)→(40,10)
    assert make_line(eng, 40.0, 10.0)[0]

    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 35.0, 10.0),
            (25.0, 20.0, 35.0, 10.0),
            (35.0, 10.0, 40.0, 10.0),
        ]
    else:
        assert _segs(eng) == [
            (10.0, 10.0, 40.0, 10.0),
            (25.0, 20.0, 35.0, 10.0),
            (35.0, 10.0, 40.0, 10.0),
        ]
    _assert_connected(eng)


def test_route_leaving_shared_stretch_keeps_its_exit(open_board, trim):
    """B→D runs on A–C from (30,10) to C and on past it to D(45,10). Only the shared
    30→40 stretch is removed; the 40→45 exit that reaches D stays."""
    eng = open_board("bridge_exit_board")
    assert eng.start_route(20.0, 20.0, 1)
    eng.flip_posture()   # (20,20)→(30,10)→(45,10)
    assert make_line(eng, 45.0, 10.0)[0]

    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 30.0, 10.0),
            (20.0, 20.0, 30.0, 10.0),
            (30.0, 10.0, 40.0, 10.0),
            (40.0, 10.0, 45.0, 10.0),
        ]
    else:
        assert _segs(eng) == [
            (10.0, 10.0, 40.0, 10.0),
            (20.0, 20.0, 30.0, 10.0),
            (30.0, 10.0, 45.0, 10.0),
        ]
    _assert_connected(eng)


def test_bridge_built_over_several_actions(open_board, trim):
    """B→J1 onto A–C, J1→J2 along it, J2→D off it. The along-the-track action adds
    no copper; the joins at J1 and J2 are joints."""
    eng = open_board("bridge_multi_board")
    for start, end in (((15.0, 20.0), (15.0, 10.0)),
                       ((15.0, 10.0), (35.0, 10.0)),
                       ((35.0, 10.0), (35.0, 20.0))):
        assert eng.start_route(*start, 1)
        assert make_line(eng, *end)[0]

    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 15.0, 10.0),
            (15.0, 10.0, 35.0, 10.0),
            (15.0, 20.0, 15.0, 10.0),
            (35.0, 10.0, 35.0, 20.0),
            (35.0, 10.0, 40.0, 10.0),
        ]
    else:
        assert _segs(eng) == [
            (10.0, 10.0, 40.0, 10.0),
            (15.0, 10.0, 35.0, 10.0),
            (15.0, 20.0, 15.0, 10.0),
            (35.0, 10.0, 35.0, 20.0),
        ]
    _assert_connected(eng)


def test_along_track_action_adds_no_copper(open_board, trim):
    """Right after J1→J2 (before any later action touches J2): trimmed, the new
    segment is gone at once rather than cleaned up by a later split."""
    eng = open_board("bridge_multi_board")
    assert eng.start_route(15.0, 20.0, 1)
    assert make_line(eng, 15.0, 10.0)[0]
    before = len(eng.get_tracks())
    assert eng.start_route(15.0, 10.0, 1)
    assert make_line(eng, 35.0, 10.0)[0]

    assert len(eng.get_tracks()) == (before if trim else before + 1)


def test_endpoint_on_track_interior_splits_it(open_board, trim):
    """A route ending mid-track: nothing overlaps, but the track is split at the
    endpoint so the T-junction is a joint."""
    eng = open_board("shared_path_board")
    assert eng.start_route(25.0, 20.0, 1)
    assert make_line(eng, 25.0, 10.0)[0]

    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 25.0, 10.0),
            (25.0, 10.0, 40.0, 10.0),
            (25.0, 20.0, 25.0, 10.0),
        ]
    else:
        assert _segs(eng) == [(10.0, 10.0, 40.0, 10.0), (25.0, 20.0, 25.0, 10.0)]
    _assert_connected(eng)


def test_overlap_on_start_side_copper_is_removed(open_board, trim):
    """A→B leaves A along its own A–C track to (15,10) before turning: that stretch
    duplicates copper already joined to the start and goes too."""
    eng = open_board("shared_path_board")
    assert eng.start_route(10.0, 10.0, 1)
    assert make_line(eng, 25.0, 20.0)[0]

    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 15.0, 10.0),
            (15.0, 10.0, 25.0, 20.0),
            (15.0, 10.0, 40.0, 10.0),
        ]
    else:
        assert _segs(eng) == [
            (10.0, 10.0, 15.0, 10.0),
            (10.0, 10.0, 40.0, 10.0),
            (15.0, 10.0, 25.0, 20.0),
        ]
    _assert_connected(eng)


FINISH_TRIMMED = [(10.0, 10.0, 15.0, 10.0), (15.0, 10.0, 40.0, 10.0), (25.0, 20.0, 15.0, 10.0)]
FINISH_UNTRIMMED = [(10.0, 10.0, 40.0, 10.0), (15.0, 10.0, 10.0, 10.0), (25.0, 20.0, 15.0, 10.0)]


@pytest.mark.parametrize("mode", [WALKAROUND, SHOVE], ids=["walkaround", "shove"])
def test_finish_commit_is_trimmed_too(open_board, trim, mode):
    """``finish`` from B, diagonal first, meets A–C at (15,10) and runs on top of it to A,
    the anchor it picks. The router joins only the route's end through its end item, so
    this commit needs the same cleanup as make_line's: the overlapping stretch goes and
    A–C is split at (15,10)."""
    eng = open_board("shared_path_board")
    assert eng.start_route(25.0, 20.0, 1)
    eng.flip_posture()   # diagonal-first posture: (25,20)→(15,10)→(10,10)
    assert finish(eng, routing_mode=mode)[0]

    assert _segs(eng) == (FINISH_TRIMMED if trim else FINISH_UNTRIMMED)
    _assert_connected(eng)


def test_finish_trim_argument_overrides_the_engine_policy(open_board, trim):
    """``KiCadEngine.finish(trim_at_contact=...)`` wins over the engine-level policy, in
    both directions — same route as above with the policy inverted per call."""
    eng = open_board("shared_path_board")
    assert eng.start_route(25.0, 20.0, 1)
    eng.flip_posture()
    assert eng.finish(trim_at_contact=not trim)

    assert _segs(eng) == (FINISH_UNTRIMMED if trim else FINISH_TRIMMED)
    _assert_connected(eng)


def test_net_moved_by_shove_is_cleaned_too(open_board, trim):
    """The cleanup covers every net the commit writes, not only the routed one. NET1 holds
    duplicate copper made without trim: A–C on y=10, and P→(30,10)→C running on top of it
    from x=30. A Shove-mode NET2 route then pushes A–C aside around x=18; shove re-adds
    that whole line, so its tail still lies on the untouched 30→40 stretch. Trimmed, the
    tail stops at (30,10): one segment covers 30→40, three meet at that joint, and NET1
    stays connected. (Shove coordinates are not pinned — only the structure is.)"""
    eng = open_board("two_net_multiterm_board")
    assert eng.start_route(25.0, 5.0, 1)
    eng.flip_posture()   # diagonal-first posture: (25,5)→(30,10)→(40,10)
    assert eng.fix_route(40.0, 10.0, trim_at_contact=False)
    net1 = eng.get_tracks()[0].net_code
    assert {(10.0, 10.0, 40.0, 10.0), (30.0, 10.0, 40.0, 10.0)} <= set(_segs(eng, net1)), \
        "precondition: P's route lies on A–C from x=30"

    assert eng.start_route(10.0, 20.0, 1)
    assert make_line(eng, 18.0, 5.0, routing_mode=SHOVE)[0]

    segs = _segs(eng, net1)
    assert (10.0, 10.0, 40.0, 10.0) not in segs, "precondition: the NET2 route shoves A–C"
    covering_35 = [s for s in segs if s[1] == s[3] == 10.0 and min(s[0], s[2]) < 35.0 < max(s[0], s[2])]
    ends_at_30 = [s for s in segs if (30.0, 10.0) in (s[:2], s[2:])]
    assert len(covering_35) == (1 if trim else 2)
    assert len(ends_at_30) == (3 if trim else 2)
    eng.build_connectivity()
    assert eng.get_pad_groups()[net1] == 1


def test_shove_across_a_three_way_joint_splits_the_net(open_board, trim):
    """Known upstream limitation, pinned in both arms so that any change in it is noticed.
    P is dropped onto A–C at (25,10): trimmed, that is a 3-way joint; untrimmed, P's line
    just ends on the through track. A Shove-mode NET2 route through the point pushes each
    NET1 line on its own — PNS ShoveObstacleLine lets its third attempt move a line end off
    a non-via joint, and nothing re-joins the lines that met there. With the joint, the A
    side ends on the hull boundary while P–C goes round the other way and A drops off the
    net; without it the through track is shoved as one line and NET1 stays connected.
    When a shove change fixes the joint case, flip that arm."""
    eng = open_board("two_net_multiterm_board")
    assert eng.start_route(25.0, 5.0, 1)
    assert make_line(eng, 25.0, 10.0)[0]
    net1 = eng.get_tracks()[0].net_code
    at_joint = [s for s in _segs(eng, net1) if (25.0, 10.0) in (s[:2], s[2:])]
    assert len(at_joint) == (3 if trim else 1), "precondition: joint only when trimmed"
    eng.build_connectivity()
    assert eng.get_pad_groups()[net1] == 1

    assert eng.start_route(10.0, 20.0, 1)
    assert make_line(eng, 30.0, 5.0, routing_mode=SHOVE)[0]
    eng.build_connectivity()
    assert eng.get_pad_groups()[net1] == (2 if trim else 1)


def test_overlap_left_untrimmed_is_split_by_a_later_shove(open_board, trim):
    """The other direction of the same limitation: duplicate copper is a split source too,
    and trim removes it. ``finish`` from P, diagonal first, runs on top of A–C from (20,10)
    to A. A Shove-mode NET2 route to (20,5) then splits NET1 when that overlap was left in
    place, and leaves it connected when the commit was trimmed."""
    eng = open_board("two_net_multiterm_board")
    assert eng.start_route(25.0, 5.0, 1)
    eng.flip_posture()   # diagonal-first posture: (25,5)→(20,10)→(10,10)
    assert finish(eng)[0]
    net1 = eng.get_tracks()[0].net_code
    assert ((10.0, 10.0, 20.0, 10.0) if trim else (20.0, 10.0, 10.0, 10.0)) in _segs(eng, net1), \
        "precondition: the finish route ran onto A–C (split there when trimmed, overlap when not)"
    eng.build_connectivity()
    assert eng.get_pad_groups()[net1] == 1

    assert eng.start_route(10.0, 20.0, 1)
    assert make_line(eng, 20.0, 5.0, routing_mode=SHOVE)[0]
    eng.build_connectivity()
    assert eng.get_pad_groups()[net1] == (1 if trim else 2)


def _vias(eng):
    return [(round(v.x_mm, 3), round(v.y_mm, 3)) for v in eng.get_vias()]


def test_make_via_on_track_interior_splits_under_the_via(open_board, trim):
    """make_via ending mid-track: the via stays at its target and, trimmed, the track
    is split under it."""
    eng = open_board("shared_path_board")
    assert eng.start_route(25.0, 20.0, 1)
    assert make_via(eng, 25.0, 10.0)[0]

    assert _vias(eng) == [(25.0, 10.0)]
    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 25.0, 10.0),
            (25.0, 10.0, 40.0, 10.0),
            (25.0, 20.0, 25.0, 10.0),
        ]
    else:
        assert _segs(eng) == [(10.0, 10.0, 40.0, 10.0), (25.0, 20.0, 25.0, 10.0)]
    _assert_connected(eng)


def test_make_via_route_onto_track_keeps_the_via(open_board, trim):
    """make_via B→(38,10), diagonal first, runs on A–C from (35,10) to the via. The
    overlapping stretch is dropped but the via stays at (38,10), with A–C split at
    35 and under the via."""
    eng = open_board("shared_path_board")
    assert eng.start_route(25.0, 20.0, 1)
    eng.flip_posture()   # (25,20)→(35,10)→(38,10)
    assert make_via(eng, 38.0, 10.0)[0]

    assert _vias(eng) == [(38.0, 10.0)]
    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 35.0, 10.0),
            (25.0, 20.0, 35.0, 10.0),
            (35.0, 10.0, 38.0, 10.0),
            (38.0, 10.0, 40.0, 10.0),
        ]
    else:
        assert _segs(eng) == [
            (10.0, 10.0, 40.0, 10.0),
            (25.0, 20.0, 35.0, 10.0),
            (35.0, 10.0, 38.0, 10.0),
        ]
    _assert_connected(eng)


def test_crossing_becomes_a_joint_that_survives_the_track_cleaner(open_board, trim):
    """B→(25,5) crosses A–C at (25,10) and dead-ends past it. Copper overlap connects B,
    but the cleaner's dangling test is anchor-based: without a joint it strips B's
    whole line and B drops off the net. Trimmed, both tracks are split at the
    crossing, so the cleaner removes only the dead-end stub."""
    from pcb_world.engine.kicad_engine import CLEANUP_FINALIZE

    eng = open_board("shared_path_board")
    assert eng.start_route(25.0, 20.0, 1)
    assert make_line(eng, 25.0, 5.0)[0]

    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 25.0, 10.0),
            (25.0, 10.0, 25.0, 5.0),
            (25.0, 10.0, 40.0, 10.0),
            (25.0, 20.0, 25.0, 10.0),
        ]
    else:
        assert _segs(eng) == [(10.0, 10.0, 40.0, 10.0), (25.0, 20.0, 25.0, 5.0)]
    _assert_connected(eng)

    eng.cleanup_tracks(dry_run=False, **CLEANUP_FINALIZE)
    eng.build_connectivity()
    if trim:
        assert _segs(eng) == [
            (10.0, 10.0, 25.0, 10.0),
            (25.0, 10.0, 40.0, 10.0),
            (25.0, 20.0, 25.0, 10.0),
        ]
        assert eng.get_unrouted_count() == 0
    else:
        assert _segs(eng) == [(10.0, 10.0, 40.0, 10.0)]
        assert eng.get_unrouted_count() == 1


@pytest.mark.parametrize("trim_on", [False, True], ids=["trim_off", "trim_on"])
def test_overlap_with_copper_one_nm_off_the_line_is_removed(trim_on):
    """PNS rounds diagonals to the nm grid, so copper that should coincide can sit 1 nm
    off. Here the pre-routed A–C ends 1 nm off y=10 (A(10,10)→(40,10.000001)); the new
    B→C route runs exactly on y=10 from (35,10). An exact collinearity test misses the
    overlap; the slack-tolerant one removes it."""
    e = open_engine(
        os.path.join(FIXTURES, "offgrid_track_board.kicad_pcb"),
        trim_at_contact=trim_on,
    )
    try:
        e.build_connectivity()
        assert e.start_route(25.0, 20.0, 1)
        e.flip_posture()   # (25,20)→(35,10)→(40,10)
        assert make_line(e, 40.0, 10.0)[0]
        nm = _nm_segs(e)
        if trim_on:
            # New (35,10)→(40,10) dropped; the off-line A–C split at the junction.
            assert nm == [(10_000_000, 10_000_000, 35_000_000, 10_000_000),
                          (25_000_000, 20_000_000, 35_000_000, 10_000_000),
                          (35_000_000, 10_000_000, 40_000_000, 10_000_001)]
        else:
            assert nm == [(10_000_000, 10_000_000, 40_000_000, 10_000_001),
                          (25_000_000, 20_000_000, 35_000_000, 10_000_000),
                          (35_000_000, 10_000_000, 40_000_000, 10_000_000)]
        _assert_connected(e)
    finally:
        e.close()


@pytest.mark.parametrize("trim_on", [False, True], ids=["trim_off", "trim_on"])
def test_overlap_ending_nm_short_leaves_no_nm_fragment(trim_on, tmp_path):
    """The pre-routed A–C ends 2 nm short of C along the line ((10,10)→(39.999998,10));
    the new B→C route runs on it from (35,10) to C. The uncovered 2 nm at C is rounding:
    trimmed, the new stretch goes whole instead of leaving a 2 nm segment behind."""
    e = open_engine(
        offgrid_board_with_tracks(tmp_path, [(10, 10, 39.999998, 10)]),
        trim_at_contact=trim_on,
    )
    try:
        e.build_connectivity()
        assert e.start_route(25.0, 20.0, 1)
        e.flip_posture()   # (25,20)→(35,10)→(40,10)
        assert make_line(e, 40.0, 10.0)[0]
        if trim_on:
            assert _nm_segs(e) == [(10_000_000, 10_000_000, 35_000_000, 10_000_000),
                                   (25_000_000, 20_000_000, 35_000_000, 10_000_000),
                                   (35_000_000, 10_000_000, 39_999_998, 10_000_000)]
        else:
            assert _nm_segs(e) == [(10_000_000, 10_000_000, 39_999_998, 10_000_000),
                                   (25_000_000, 20_000_000, 35_000_000, 10_000_000),
                                   (35_000_000, 10_000_000, 40_000_000, 10_000_000)]
        _assert_connected(e)
    finally:
        e.close()


ACUTE_PAIR = [(25, 20, 25, 12), (25, 12, 33, 20)]   # B→V(25,12) vertical, V→(33,20) diagonal: 45° apart at V


@pytest.mark.parametrize("trim_on", [False, True], ids=["trim_off", "trim_on"])
def test_split_just_past_the_slack_of_a_vertex_leaves_nm_fragments(trim_on, tmp_path):
    """Known limitation of the split step, pinned so that a fix is noticed. The route
    from C ends at T(25,12.000014), on the vertical 14 nm from V. T is past the 10 nm
    "already a joint" slack of V yet within 10 nm of both centerlines (9.9 nm off the
    diagonal), so both tracks are cut at T: the 14 nm V–T is added once per track and
    the diagonal now leaves from T instead of V. Untrimmed, nothing is cut.
    Connectivity is the same either way."""
    e = open_engine(
        offgrid_board_with_tracks(tmp_path, ACUTE_PAIR),
        trim_at_contact=trim_on,
    )
    try:
        e.build_connectivity()
        assert e.start_route(40.0, 10.0, 1)
        assert make_line(e, 25.0, 12.000014)[0]
        route = [(27_000_014, 10_000_000, 25_000_000, 12_000_014),
                 (40_000_000, 10_000_000, 27_000_014, 10_000_000)]
        if trim_on:
            assert _nm_segs(e) == sorted([
                (25_000_000, 12_000_000, 25_000_000, 12_000_014),   # V–T, cut off the diagonal
                (25_000_000, 12_000_014, 25_000_000, 12_000_000),   # T–V, cut off the vertical
                (25_000_000, 12_000_014, 33_000_000, 20_000_000),   # the diagonal, now from T
                (25_000_000, 20_000_000, 25_000_000, 12_000_014),
                *route])
        else:
            assert _nm_segs(e) == sorted([
                (25_000_000, 12_000_000, 33_000_000, 20_000_000),
                (25_000_000, 20_000_000, 25_000_000, 12_000_000),
                *route])
        e.build_connectivity()
        assert e.get_unrouted_count() == 1   # A is never routed here; B–C are joined in both arms
    finally:
        e.close()


@pytest.mark.parametrize("k_nm, segments, duplicates",
                         [(10, 4, 0), (11, 6, 1), (14, 6, 1), (15, 11, 5), (16, 8, 2), (20, 8, 2), (21, 7, 0)])
def test_split_outcome_by_distance_from_the_vertex(k_nm, segments, duplicates, tmp_path):
    """The same limitation along the distance k of the committed end T from V, trimmed
    (segment count, how many of them repeat another). Up to 10 nm T is V's joint and
    nothing is cut. 11–14: T cuts both tracks, the case pinned above — one piece, twice.
    From 15 nm the route's crossing X with the diagonal is no longer within the slack
    of T and is a split point too; while it is within 10 nm of the vertical (to 20 nm)
    it cuts that as well, so pieces come out twice. At 15 nm T still cuts the diagonal
    (10.6 nm truncates to 10) and X then cuts five segments: T–X is there five times.
    From 21 nm X is clear of the vertical and nothing repeats."""
    e = open_engine(
        offgrid_board_with_tracks(tmp_path, ACUTE_PAIR),
        trim_at_contact=True,
    )
    try:
        e.build_connectivity()
        assert e.start_route(40.0, 10.0, 1)
        assert make_line(e, 25.0, 12.0 + k_nm * 1e-6)[0]
        segs = _nm_segs(e)
        distinct = {(min(s[:2], s[2:]), max(s[:2], s[2:])) for s in segs}
        assert (len(segs), len(segs) - len(distinct)) == (segments, duplicates)
        e.build_connectivity()
        assert e.get_unrouted_count() == 1
    finally:
        e.close()


@pytest.mark.parametrize("trim_on", [False, True], ids=["trim_off", "trim_on"])
def test_split_point_on_overlapping_tracks_repeats_the_cut_piece(trim_on, tmp_path):
    """The same limitation on copper that already overlaps: pre-routed B(25,20)→(25,12)
    and (25,16)→(25,12) run on top of each other. The route from C ends at (25,14),
    inside both; trimmed, each is cut there and (25,14)–(25,12) is on the board twice —
    a split does not check whether the piece it adds is already there. Untrimmed,
    nothing is cut."""
    e = open_engine(
        offgrid_board_with_tracks(tmp_path, [(25, 20, 25, 12), (25, 16, 25, 12)]),
        trim_at_contact=trim_on,
    )
    try:
        e.build_connectivity()
        assert e.start_route(40.0, 10.0, 1)
        assert make_line(e, 25.0, 14.0)[0]
        route = [(29.0, 10.0, 25.0, 14.0), (40.0, 10.0, 29.0, 10.0)]
        if trim_on:
            assert _segs(e) == sorted([(25.0, 14.0, 25.0, 12.0), (25.0, 14.0, 25.0, 12.0),
                                       (25.0, 16.0, 25.0, 14.0), (25.0, 20.0, 25.0, 14.0), *route])
        else:
            assert _segs(e) == sorted([(25.0, 16.0, 25.0, 12.0), (25.0, 20.0, 25.0, 12.0), *route])
    finally:
        e.close()


def _locked_segments(eng, tmp_path):
    """(x1, y1, x2, y2, locked) per segment of the board as saved — the lock flag is
    not on the track info, the file carries it as ``(locked yes)``."""
    out = tmp_path / "saved.kicad_pcb"
    eng.save(str(out))
    segs = []
    for body in re.findall(r"\(segment\b(.*?)\(uuid", out.read_text(encoding="utf-8"), re.S):
        (x1, y1), (x2, y2) = (tuple(float(v) for v in re.search(rf"\({k} (\S+) (\S+)\)", body).groups())
                              for k in ("start", "end"))
        segs.append((x1, y1, x2, y2, "(locked yes)" in body))
    return sorted(segs)


def test_split_locked_track_stays_locked(open_board, trim, tmp_path):
    """Locked A–C, then B→(25,10) ends inside it. Trimmed, A–C is split at (25,10) and
    BOTH halves stay locked — the half the router re-creates carries the lock too, so
    shove keeps treating the whole locked net as immovable."""
    eng = open_board("shared_path_board")
    net = eng.get_tracks()[0].net_code
    assert eng.lock_net(net, True) == 1

    assert eng.start_route(25.0, 20.0, 1)
    assert make_line(eng, 25.0, 10.0)[0]

    if trim:
        assert _locked_segments(eng, tmp_path) == [
            (10.0, 10.0, 25.0, 10.0, True),
            (25.0, 10.0, 40.0, 10.0, True),
            (25.0, 20.0, 25.0, 10.0, False),
        ]
    else:
        assert _locked_segments(eng, tmp_path) == [
            (10.0, 10.0, 40.0, 10.0, True),
            (25.0, 20.0, 25.0, 10.0, False),
        ]
    _assert_connected(eng)
