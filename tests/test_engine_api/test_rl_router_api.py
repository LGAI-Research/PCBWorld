"""Full RLRouter API functional tests.

Test board: simple_obstacle_board.kicad_pcb
- P1 (0,0) -> P2 (3,5) route, NET1
- Obstacle OBS1: 2mm x 2mm rectangle centered at (2,0), [1,-1]~[3,1]

APIs covered:
  - RLRouter(board_path)           : constructor
  - set_routing_mode(mode)         : set the routing mode
  - set_track_width(width_mm)      : set the track width
  - start_route(x, y, layer)      : start a route
  - move(x, y)                    : move the route head
  - fix_route(x, y, force_finish) : commit a route
  - cancel_route()                : cancel a route
  - get_tracks()                  : list tracks (TrackInfo)
  - get_track_count()             : get the track count
  - is_routing()                  : query routing state
  - delete_track_by_index(idx)    : delete a track by index
  - delete_track_near(...)        : delete a track by coordinate
  - save(path)                    : save the board
  - MODE_MARK_OBSTACLES / MODE_SHOVE / MODE_WALKAROUND : module constants
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build_rl"
RL_MODULE_DIR = BUILD_DIR / "pcbnew" / "python" / "rl"
BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"
OUTPUT_DIR = PROJECT_ROOT / "var" / "tests" / "output"

sys.path.insert(0, str(RL_MODULE_DIR))

START = (0.0, 0.0)
END = (3.0, 5.0)


@pytest.fixture(autouse=True)
def _ensure_output_dir() -> None:
    """Creates the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def board_path() -> str:
    """Path to the test board."""
    if not BOARD_PATH.exists():
        pytest.skip(f"test board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


def _import_krl():
    """Imports the kicad_rl_router module."""
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip(
            f"kicad_rl_router module not found. "
            f"build path: {RL_MODULE_DIR}"
        )


def _route_simple(r, krl) -> bool:
    """Helper for a simple P1->P2 route. Returns whether it succeeded."""
    r.set_routing_mode(krl.MODE_WALKAROUND)
    r.start_route(START[0], START[1], 0)
    r.move(1.5, 2.5)
    r.fix_route(1.5, 2.5, force_finish=False)
    return r.fix_route(END[0], END[1])


# ──────────────────────────────────────────────
# 1. Module constants
# ──────────────────────────────────────────────
class TestModuleConstants:
    """Checks module-level constants exist and have the right values."""

    def test_mode_constants_exist(self) -> None:
        krl = _import_krl()
        assert hasattr(krl, "MODE_MARK_OBSTACLES")
        assert hasattr(krl, "MODE_SHOVE")
        assert hasattr(krl, "MODE_WALKAROUND")

    def test_mode_constants_values(self) -> None:
        krl = _import_krl()
        assert krl.MODE_MARK_OBSTACLES == 0
        assert krl.MODE_SHOVE == 1
        assert krl.MODE_WALKAROUND == 2

    def test_mode_constants_match_action_schema(self) -> None:
        """Pure-Python ``action_schema.MODE_*`` must equal the engine's
        ``krl.MODE_*`` — these are the routing_mode ints the codec/adapter send
        to the engine, so any drift silently mis-routes."""
        krl = _import_krl()
        from pcb_world.core.action_schema import (
            MODE_MARK_OBSTACLES,
            MODE_SHOVE,
            MODE_WALKAROUND,
        )
        assert MODE_MARK_OBSTACLES == krl.MODE_MARK_OBSTACLES
        assert MODE_SHOVE == krl.MODE_SHOVE
        assert MODE_WALKAROUND == krl.MODE_WALKAROUND
        # DRC-severity / codec / eval cross-module parity lives in
        # tests/test_constant_consistency.py (pure — no router needed).


# ──────────────────────────────────────────────
# 2. Constructor
# ──────────────────────────────────────────────
class TestConstructor:
    """RLRouter constructor checks."""

    def test_load_valid_board(self, board_path: str) -> None:
        """Loads a valid board file."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r is not None

    def test_load_invalid_path(self) -> None:
        """Raises when loading a nonexistent file."""
        krl = _import_krl()
        with pytest.raises(Exception):
            krl.RLRouter("/nonexistent/path/board.kicad_pcb")


# ──────────────────────────────────────────────
# 3. set_routing_mode()
# ──────────────────────────────────────────────
class TestSetRoutingMode:
    """set_routing_mode() checks."""

    @pytest.mark.parametrize(
        "mode_name,mode_value",
        [
            ("mark_obstacles", 0),
            ("shove", 1),
            ("walkaround", 2),
        ],
    )
    def test_set_mode_and_route(
        self, board_path: str, mode_name: str, mode_value: int
    ) -> None:
        """Checks that routing works correctly in each mode."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(mode_value)

        started = r.start_route(START[0], START[1], 0)
        assert started, f"start_route failed (mode={mode_name})"

        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)
        success = r.fix_route(END[0], END[1])

        assert success, f"fix_route failed (mode={mode_name})"
        assert r.get_track_count() > 0

    def test_set_mode_with_constant(self, board_path: str) -> None:
        """Sets the mode using a module constant."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        started = r.start_route(START[0], START[1], 0)
        assert started


# ──────────────────────────────────────────────
# 4. set_track_width()
# ──────────────────────────────────────────────
class TestSetTrackWidth:
    """set_track_width() checks."""

    def test_default_width(self, board_path: str) -> None:
        """Design-rule width applies automatically (width=0)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_track_width(0)

        success = _route_simple(r, krl)
        assert success

        tracks = r.get_tracks()
        assert len(tracks) > 0
        assert tracks[0].width_mm > 0

    def test_custom_width(self, board_path: str) -> None:
        """Sets a custom track width (0.25mm)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        width_mm = 0.25
        r.set_track_width(width_mm)

        success = _route_simple(r, krl)
        assert success

        tracks = r.get_tracks()
        assert len(tracks) > 0
        assert tracks[0].width_mm == pytest.approx(width_mm, abs=0.01)

    @pytest.mark.parametrize(
        "width_mm_a,width_mm_b",
        [(0.2, 0.5)],
    )
    def test_different_widths_produce_different_tracks(
        self, board_path: str, width_mm_a: float, width_mm_b: float
    ) -> None:
        """Checks that different width settings produce different track widths.

        Compares route -> query -> delete -> reroute on a single RLRouter.
        """
        krl = _import_krl()
        r = krl.RLRouter(board_path)

        r.set_track_width(width_mm_a)
        _route_simple(r, krl)
        width_a = r.get_tracks()[0].width_mm

        while r.get_track_count() > 0:
            r.delete_track_by_index(0)

        r.set_track_width(width_mm_b)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)
        r.fix_route(END[0], END[1])
        width_b = r.get_tracks()[0].width_mm

        assert width_a != pytest.approx(width_b, abs=0.01)


# ──────────────────────────────────────────────
# 5. start_route()
# ──────────────────────────────────────────────
class TestStartRoute:
    """start_route() checks."""

    def test_start_on_pad(self, board_path: str) -> None:
        """Succeeds starting at a pad position."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        started = r.start_route(START[0], START[1], 0)
        assert started is True

    def test_start_on_empty_area(self, board_path: str) -> None:
        """Can also start in an empty area with no pads/tracks (PNS does not refuse)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        started = r.start_route(100.0, 100.0, 0)
        assert started is True
        assert r.is_routing() is True


# ──────────────────────────────────────────────
# 6. is_routing() state
# ──────────────────────────────────────────────
class TestIsRouting:
    """is_routing() checks."""

    def test_false_before_start(self, board_path: str) -> None:
        """False before routing starts."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.is_routing() is False

    def test_true_during_routing(self, board_path: str) -> None:
        """True while routing is in progress."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)

        assert r.is_routing() is True

    def test_true_after_move(self, board_path: str) -> None:
        """Still True after move()."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)

        assert r.is_routing() is True

    def test_true_after_waypoint_fix(self, board_path: str) -> None:
        """Still True after force_finish=False (routing continues)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)

        assert r.is_routing() is True

    def test_false_after_fix_finish(self, board_path: str) -> None:
        """False after committing with force_finish=True."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)
        r.fix_route(END[0], END[1])

        assert r.is_routing() is False

    def test_false_after_cancel(self, board_path: str) -> None:
        """False after cancel_route()."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.cancel_route()

        assert r.is_routing() is False


# ──────────────────────────────────────────────
# 7. cancel_route()
# ──────────────────────────────────────────────
class TestCancelRoute:
    """cancel_route() checks."""

    def test_cancel_does_not_add_tracks(self, board_path: str) -> None:
        """Cancel does not add a track to the board."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        count_before = r.get_track_count()

        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.cancel_route()

        assert r.get_track_count() == count_before

    def test_can_reroute_after_cancel(self, board_path: str) -> None:
        """Routing is possible again after a cancel."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.cancel_route()

        started = r.start_route(START[0], START[1], 0)
        assert started is True

        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)
        success = r.fix_route(END[0], END[1])
        assert success


# ──────────────────────────────────────────────
# 8. get_tracks() / TrackInfo fields
# ──────────────────────────────────────────────
class TestGetTracks:
    """get_tracks() and TrackInfo field checks."""

    def test_empty_before_routing(self, board_path: str) -> None:
        """Queries the initial track list before routing."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        tracks = r.get_tracks()
        assert isinstance(tracks, list)

    def test_tracks_added_after_routing(self, board_path: str) -> None:
        """Tracks are added after routing."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        count_before = len(r.get_tracks())

        _route_simple(r, krl)

        tracks = r.get_tracks()
        assert len(tracks) > count_before

    def test_track_count_matches_get_tracks(self, board_path: str) -> None:
        """get_track_count() matches len(get_tracks())."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        assert r.get_track_count() == len(r.get_tracks())

    def test_trackinfo_has_all_fields(self, board_path: str) -> None:
        """TrackInfo has all required fields."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        track = r.get_tracks()[0]
        assert hasattr(track, "x1_mm")
        assert hasattr(track, "y1_mm")
        assert hasattr(track, "x2_mm")
        assert hasattr(track, "y2_mm")
        assert hasattr(track, "width_mm")
        assert hasattr(track, "layer")
        assert hasattr(track, "net_code")
        assert hasattr(track, "net_name")

    def test_trackinfo_coordinate_types(self, board_path: str) -> None:
        """TrackInfo coordinates are float."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        track = r.get_tracks()[0]
        assert isinstance(track.x1_mm, float)
        assert isinstance(track.y1_mm, float)
        assert isinstance(track.x2_mm, float)
        assert isinstance(track.y2_mm, float)
        assert isinstance(track.width_mm, float)

    def test_trackinfo_layer_and_net(self, board_path: str) -> None:
        """TrackInfo's layer and net info are valid."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        track = r.get_tracks()[0]
        assert isinstance(track.layer, int)
        assert track.layer >= 0
        assert isinstance(track.net_code, int)
        assert isinstance(track.net_name, str)
        assert len(track.net_name) > 0

    def test_trackinfo_width_positive(self, board_path: str) -> None:
        """Track width is positive."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        for track in r.get_tracks():
            assert track.width_mm > 0


# ──────────────────────────────────────────────
# 9. delete_track_by_index()
# ──────────────────────────────────────────────
class TestDeleteTrackByIndex:
    """delete_track_by_index() checks."""

    def test_delete_first_track(self, board_path: str) -> None:
        """Deletes the first track."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        count_before = r.get_track_count()
        assert count_before > 0

        result = r.delete_track_by_index(0)
        assert result is True
        assert r.get_track_count() == count_before - 1

    def test_delete_last_track(self, board_path: str) -> None:
        """Deletes the last track."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        count_before = r.get_track_count()
        last_idx = count_before - 1

        result = r.delete_track_by_index(last_idx)
        assert result is True
        assert r.get_track_count() == count_before - 1

    def test_delete_invalid_index(self, board_path: str) -> None:
        """Returns False for an out-of-range index."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        count = r.get_track_count()
        result = r.delete_track_by_index(count + 100)
        assert result is False
        assert r.get_track_count() == count

    def test_delete_all_tracks(self, board_path: str) -> None:
        """Deletes every track one at a time."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        while r.get_track_count() > 0:
            r.delete_track_by_index(0)

        assert r.get_track_count() == 0
        assert len(r.get_tracks()) == 0


# ──────────────────────────────────────────────
# 10. delete_track_near()
# ──────────────────────────────────────────────
class TestDeleteTrackNear:
    """delete_track_near() checks."""

    def test_delete_existing_track(self, board_path: str) -> None:
        """Deletes by the coordinates of a routed track."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        count_before = r.get_track_count()
        track = r.get_tracks()[0]

        result = r.delete_track_near(
            track.x1_mm, track.y1_mm,
            track.x2_mm, track.y2_mm,
            track.layer, track.net_code,
            0.1,
        )
        assert result is True
        assert r.get_track_count() == count_before - 1

    def test_delete_nonexistent_track(self, board_path: str) -> None:
        """Returns False when deleting at a nonexistent position."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        count_before = r.get_track_count()
        result = r.delete_track_near(999.0, 999.0, 998.0, 998.0, 0, 1, 0.01)
        assert result is False
        assert r.get_track_count() == count_before


# ──────────────────────────────────────────────
# 11. fix_route() force_finish parameter
# ──────────────────────────────────────────────
class TestFixRouteForceFinish:
    """Checks fix_route()'s force_finish parameter."""

    def test_force_finish_false_continues_routing(
        self, board_path: str
    ) -> None:
        """Routing can continue after force_finish=False."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        r.start_route(START[0], START[1], 0)
        r.move(1.0, 1.0)
        r.fix_route(1.0, 1.0, force_finish=False)

        assert r.is_routing() is True

        r.move(2.0, 3.0)
        r.fix_route(2.0, 3.0, force_finish=False)

        assert r.is_routing() is True

    def test_force_finish_true_commits(self, board_path: str) -> None:
        """force_finish=True performs the final commit."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        r.start_route(START[0], START[1], 0)
        r.move(1.5, 2.5)
        r.fix_route(1.5, 2.5, force_finish=False)
        success = r.fix_route(END[0], END[1], force_finish=True)

        assert success is True
        assert r.is_routing() is False
        assert r.get_track_count() > 0

    def test_multiple_waypoints_then_finish(self, board_path: str) -> None:
        """Multiple waypoints followed by a final commit."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)

        r.start_route(START[0], START[1], 0)

        waypoints = [(-0.5, 1.0), (0.0, 3.0), (1.5, 4.0)]
        for wx, wy in waypoints:
            r.move(wx, wy)
            r.fix_route(wx, wy, force_finish=False)
            assert r.is_routing() is True

        success = r.fix_route(END[0], END[1])
        assert success is True
        assert r.is_routing() is False


# ──────────────────────────────────────────────
# 12. save()
# ──────────────────────────────────────────────
class TestSave:
    """save() checks."""

    def test_save_creates_file(self, board_path: str) -> None:
        """Confirms a file is created after saving."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        output = OUTPUT_DIR / "api_save_test.kicad_pcb"
        if output.exists():
            output.unlink()

        r.save(str(output))
        assert output.exists()
        assert output.stat().st_size > 0

    def test_save_preserves_tracks(self, board_path: str) -> None:
        """Tracks are preserved when reloaded after saving."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        _route_simple(r, krl)

        track_count = r.get_track_count()
        output = OUTPUT_DIR / "api_save_reload.kicad_pcb"
        r.save(str(output))

        r2 = krl.RLRouter(str(output))
        assert r2.get_track_count() == track_count


# ──────────────────────────────────────────────
# 13. move()
# ──────────────────────────────────────────────
class TestMove:
    """move() checks."""

    def test_move_returns_true_during_routing(self, board_path: str) -> None:
        """move() returns True while routing."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)

        result = r.move(1.5, 2.5)
        assert result is True

    def test_move_returns_false_without_routing(
        self, board_path: str
    ) -> None:
        """move() returns False before routing has started."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)

        result = r.move(1.5, 2.5)
        assert result is False

    def test_multiple_moves(self, board_path: str) -> None:
        """Consecutive move() calls (only the final position matters)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_routing_mode(krl.MODE_WALKAROUND)
        r.start_route(START[0], START[1], 0)

        r.move(0.5, 1.0)
        r.move(1.0, 2.0)
        r.move(1.5, 2.5)

        r.fix_route(1.5, 2.5, force_finish=False)
        success = r.fix_route(END[0], END[1])
        assert success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
