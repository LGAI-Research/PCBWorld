"""``KiCadEngine.optimize_all_lines`` — a finishing pass that runs the router's PNS
optimizer over every committed line, like ``cleanup_tracks`` for geometry.

Board: ``shared_path_board`` — NET1 pads A(10,10), C(40,10), B(25,20).
The collinear case runs on a copy of ``offgrid_track_board`` (same pads) with its own
pre-routed tracks.
"""

import os

from pcb_world.core.action import make_line
from pcb_world.engine.kicad_engine import CLEANUP_TOPOLOGY_PRESERVING
from tests.helpers.prerouted_board import offgrid_board_with_tracks
from tests.helpers.engine import open_engine

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _segs(e):
    return sorted((round(t.x1_mm, 3), round(t.y1_mm, 3), round(t.x2_mm, 3), round(t.y2_mm, 3))
                  for t in e.get_tracks())


def _engine():
    e = open_engine(
        os.path.join(FIXTURES, "shared_path_board.kicad_pcb"),
    )
    e.build_connectivity()
    return e


def test_straightens_a_path_built_over_two_actions():
    """A→(20,15) then (20,15)→C: each action's head was optimized alone, so the detour
    through (20,15) survives the routing; the finishing pass straightens the whole
    A–C line without changing what is connected."""
    e = _engine()
    try:
        assert e.start_route(10.0, 10.0, 1) and make_line(e, 20.0, 15.0)[0]
        assert e.start_route(20.0, 15.0, 1) and make_line(e, 40.0, 10.0)[0]
        groups = dict(e.get_pad_groups())
        assert len(_segs(e)) == 4

        assert e.optimize_all_lines() == 1
        assert _segs(e) == [(10.0, 10.0, 40.0, 10.0)]
        e.build_connectivity()
        assert dict(e.get_pad_groups()) == groups
    finally:
        e.close()


def test_zero_passes_is_a_no_op():
    """``passes`` bounds the pass count: 0 runs none, so the two-action detour stays."""
    e = _engine()
    try:
        assert e.start_route(10.0, 10.0, 1) and make_line(e, 20.0, 15.0)[0]
        assert e.start_route(20.0, 15.0, 1) and make_line(e, 40.0, 10.0)[0]
        before = _segs(e)

        assert e.optimize_all_lines(passes=0) == 0
        assert _segs(e) == before
    finally:
        e.close()


def test_refused_during_a_routing_session():
    e = _engine()
    try:
        assert e.start_route(10.0, 10.0, 1) and make_line(e, 20.0, 15.0)[0]
        assert e.start_route(20.0, 15.0, 1)
        assert e.optimize_all_lines() == -1
        e.cancel_route()
        assert e.optimize_all_lines() == 0      # a single segment: nothing to merge
    finally:
        e.close()


def test_optimized_line_does_not_run_on_top_of_same_net_copper():
    """A–C straight, then B→C diagonal-first lands on A–C at (35,10) and — without
    trim — overlaps it to C. After the finishing pass no same-net copper overlaps:
    the committed copper gets the same cleanup as trim_at_contact."""
    e = _engine()
    try:
        assert e.start_route(10.0, 10.0, 1) and make_line(e, 40.0, 10.0)[0]
        assert e.start_route(25.0, 20.0, 1)
        e.flip_posture()
        assert make_line(e, 40.0, 10.0)[0]
        e.optimize_all_lines()
        on_y10 = [s for s in _segs(e) if s[1] == s[3] == 10.0]
        covered = sorted((min(s[0], s[2]), max(s[0], s[2])) for s in on_y10)
        for (a0, a1), (b0, b1) in zip(covered, covered[1:]):
            assert a1 <= b0, f"overlap on y=10: {covered}"
        e.build_connectivity()
        assert e.get_unrouted_count() == 0
    finally:
        e.close()


def test_collinear_only_result_is_left_to_the_track_cleaner(tmp_path):
    """A–C pre-routed as two tracks meeting on a straight run at (25,10). Dropping that
    vertex is all the optimizer can do, and a result of unchanged shape is not committed:
    the pass replaces nothing and the tracks stay two. Merging them is the track
    cleaner's ``merge_segments``."""
    e = open_engine(
        offgrid_board_with_tracks(tmp_path, [(10, 10, 25, 10), (25, 10, 40, 10)]),
    )
    try:
        e.build_connectivity()
        two = [(10.0, 10.0, 25.0, 10.0), (25.0, 10.0, 40.0, 10.0)]
        assert _segs(e) == two

        assert e.optimize_all_lines() == 0
        assert _segs(e) == two

        assert e.cleanup_tracks(dry_run=False, **CLEANUP_TOPOLOGY_PRESERVING).ran
        assert _segs(e) == [(10.0, 10.0, 40.0, 10.0)]
    finally:
        e.close()
