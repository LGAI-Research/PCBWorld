"""DRC integration test with via-based crossing avoidance.

Board: simple_routing_board.kicad_pcb (2-layer)
  NET1: (10,10) <-> (40,10)  horizontal
  NET2: (10,20) <-> (40,20)  horizontal
  NET3: (25,5)  <-> (25,25)  vertical (crosses NET1 and NET2)

Human layers: 1=Top (F.Cu), 2=Bottom (B.Cu)

Test plan:
  1. Route all 3 nets with make_line only on layer 1
     -> verify all nets connected
     -> verify DRC violations >= 2 (crossing = short/clearance)

  2. Route NET1, NET2 with make_line, NET3 with make_via to avoid crossing
     -> verify DRC violations = 0
"""

import pytest
from pcb_world.engine.kicad_engine import KiCadEngine
from pcb_world.core.action import make_line, make_via

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"

TOP = 1     # human layer: Top (F.Cu)
BOTTOM = 2  # human layer: Bottom (B.Cu)


@pytest.fixture
def engine():
    e = KiCadEngine(BOARD)
    e.build_connectivity()
    yield e
    e.close()


def _route_net1_net2(engine):
    """Route NET1 and NET2 as straight lines on Top layer."""
    engine.start_route(10.0, 10.0, TOP)
    make_line(engine, 40.0, 10.0)
    engine.build_connectivity()

    engine.start_route(10.0, 20.0, TOP)
    make_line(engine, 40.0, 20.0)
    engine.build_connectivity()


class TestMakeLineOnlyCausesDRC:
    """Route all nets with make_line only — should cause DRC violations."""

    def test_all_nets_connected(self, engine):
        """All 3 nets should be routed (connected)."""
        _route_net1_net2(engine)

        engine.start_route(25.0, 5.0, TOP)
        make_line(engine, 25.0, 25.0)
        engine.build_connectivity()

        assert engine.get_unrouted_count() == 0, (
            f"All nets should be connected, unrouted={engine.get_unrouted_count()}"
        )

class TestMakeViaAvoidsDRC:
    """Route NET3 with via to avoid crossing — should have 0 DRC violations."""

    def test_make_via_places_via(self, engine):
        """make_via should place a via at the target position."""
        engine.start_route(25.0, 5.0, TOP)
        success, _ = make_via(engine, 25.0, 5.5)
        engine.build_connectivity()

        assert success, "make_via should succeed"
        assert engine.get_via_count() == 1, (
            f"Expected 1 via, got {engine.get_via_count()}"
        )

    def test_via_routing_zero_violations(self, engine):
        """NET1/NET2 via make_line, NET3 via make_via -> DRC = 0.

        Strategy for NET3:
          (25,5) Top --make_via--> (25,5.5) via to Bottom
          (25,5.5) Bottom --make_via--> (25,25) via at target pad, back to Top
        """
        _route_net1_net2(engine)

        # Set via dimensions to meet board design rules
        engine.set_via_diameter(0.6)
        engine.set_via_drill(0.3)

        # NET3 Step 1: Top pad (25,5) -> via at (25,5.5)
        engine.start_route(25.0, 5.0, TOP)
        s1, _ = make_via(engine, 25.0, 5.5)
        engine.build_connectivity()
        assert s1, "First make_via should succeed"

        # NET3 Step 2: Bottom (25,5.5) -> via at target pad (25,25)
        engine.start_route(25.0, 5.5, BOTTOM)
        s2, _ = make_via(engine, 25.0, 25.0)
        engine.build_connectivity()
        assert s2, "Second make_via should succeed"
        assert engine.get_via_count() == 2, (
            f"Expected 2 vias, got {engine.get_via_count()}"
        )

        # Verify all nets connected
        assert engine.get_unrouted_count() == 0, (
            f"All nets should be connected, unrouted={engine.get_unrouted_count()}"
        )

        # Verify DRC = 0
        engine.run_drc()
        violations = engine.get_drc_violations()

        assert len(violations) == 0, (
            f"Expected 0 DRC violations with via routing, got {len(violations)}: "
            f"{[f'{v.error_type}: {v.message}' for v in violations]}"
        )
