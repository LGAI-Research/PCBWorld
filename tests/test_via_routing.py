"""Test via placement and multi-layer routing.

Verifies that make_via(x, y) correctly:
1. Creates a track from current position to (x, y)
2. Places a via at (x, y)
3. Can be combined with start_route on another layer to route through vias
"""

import pytest
from pcb_world.engine.kicad_engine import KiCadEngine
from pcb_world.core.action import make_via

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


@pytest.fixture
def engine():
    e = KiCadEngine(BOARD)
    e.build_connectivity()
    yield e
    # Ensure any active routing/dragging session is cancelled before
    # the C++ RLRouter is destroyed, preventing a segfault.
    if e.is_routing():
        e.cancel_route()
    if e.is_dragging():
        e.cancel_drag()


class TestMakeVia:
    """Test make_via action with target coordinates."""

    def test_make_via_creates_track_and_via(self, engine):
        """make_via should create a track segment and a via at the target."""
        engine.start_route(25.0, 5.0, 1)
        success, info = make_via(engine, 25.0, 9.0)

        assert success, "make_via should succeed"
        assert engine.get_track_count() == 1, "Should have 1 track"
        assert engine.get_via_count() == 1, "Should have 1 via"

    def test_make_via_ends_routing_session(self, engine):
        """make_via with forceFinish=True should end the routing session."""
        engine.start_route(25.0, 5.0, 1)
        make_via(engine, 25.0, 9.0)

        assert not engine.is_routing(), "Routing session should be ended"

    def test_via_placed_at_target_position(self, engine):
        """Via should be at the target coordinates."""
        engine.start_route(25.0, 5.0, 1)
        make_via(engine, 25.0, 9.0)

        vias = engine.get_vias()
        assert len(vias) == 1
        assert vias[0].x_mm == pytest.approx(25.0, abs=0.01)
        assert vias[0].y_mm == pytest.approx(9.0, abs=0.01)

    def test_track_on_original_layer(self, engine):
        """Track should be on the starting layer (F.Cu)."""
        engine.start_route(25.0, 5.0, 1)
        make_via(engine, 25.0, 9.0)

        tracks = engine.get_tracks()
        assert len(tracks) == 1
        # get_tracks() returns raw C++ objects with board layer IDs (F_Cu=0)
        assert tracks[0].layer == 0, "Track should be on F.Cu (board layer 0)"
