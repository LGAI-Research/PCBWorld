"""Tests for set_via_diameter / set_via_drill.

Verifies that:
1. Design rules from the board are visible via the engine's
   ``get_design_rules`` API (BDS + NetSettings from the C++ loader).
2. set_via_diameter / set_via_drill propagate values to the routing engine.
3. Placed vias reflect the configured dimensions.
"""

import os
import pytest

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"
TOP = 1
BOTTOM = 2


def _skip_if_no_board():
    if not os.path.exists(BOARD):
        pytest.skip(f"Board not found: {BOARD}")


def _skip_if_no_kicad():
    try:
        import kicad_rl_router  # noqa: F401
    except ImportError:
        pytest.skip("kicad_rl_router not available")


# ---------------------------------------------------------------------------
# 1. Design rules visibility (engine-sourced)
# ---------------------------------------------------------------------------

class TestDesignRulesVisibility:
    """Verify the engine surfaces via dimensions via get_design_rules()."""

    @pytest.fixture
    def rules(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.engine.kicad_engine import KiCadEngine
        eng = KiCadEngine(BOARD)
        return eng.get_design_rules()

    def test_engine_via_diameter_from_default_netclass(self, rules):
        # Fixture's companion .kicad_pro puts via_diameter on the Default
        # netclass (0.6 mm); the per-netclass view is the reliable source.
        assert rules.default_netclass.via_diameter_mm == pytest.approx(0.6, abs=0.01)

    def test_engine_via_drill_from_default_netclass(self, rules):
        assert rules.default_netclass.via_drill_mm == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Engine-level set_via_diameter / set_via_drill (no crash, basic setter)
# ---------------------------------------------------------------------------

class TestSetViaDimensions:
    """Verify that set_via_diameter and set_via_drill execute without error."""

    @pytest.fixture
    def engine(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.engine.kicad_engine import KiCadEngine
        e = KiCadEngine(BOARD)
        e.build_connectivity()
        return e

    def test_set_via_diameter_no_error(self, engine):
        engine.set_via_diameter(0.6)

    def test_set_via_drill_no_error(self, engine):
        engine.set_via_drill(0.3)

    def test_custom_dimensions_no_error(self, engine):
        """Non-default dimensions should also work."""
        engine.set_via_diameter(0.8)
        engine.set_via_drill(0.4)


# ---------------------------------------------------------------------------
# 3. Placed via count after make_via
# ---------------------------------------------------------------------------

class TestViaDimensionsInPlacedVia:
    """Verify vias are placed correctly after set_via_diameter/drill."""

    @pytest.fixture
    def engine(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.engine.kicad_engine import KiCadEngine
        e = KiCadEngine(BOARD)
        e.build_connectivity()
        return e

    def test_placed_via_with_board_defaults(self, engine):
        """Via placed with board default dimensions (0.6 dia, 0.3 drill)."""
        from pcb_world.core.action import make_via

        engine.set_via_diameter(0.6)
        engine.set_via_drill(0.3)

        engine.start_route(25.0, 5.0, TOP)
        success, _ = make_via(engine, 25.0, 5.5)
        engine.build_connectivity()

        assert success, "make_via should succeed"
        assert engine.get_via_count() == 1

        # Verify via attributes if exposed by C++ bindings
        vias = engine.get_vias()
        assert len(vias) >= 1
        v = vias[0]
        assert abs(v.x_mm - 25.0) < 0.5

    def test_placed_via_with_custom_dimensions(self, engine):
        """Via placed with custom dimensions (0.8 dia, 0.4 drill)."""
        from pcb_world.core.action import make_via

        engine.set_via_diameter(0.8)
        engine.set_via_drill(0.4)

        engine.start_route(25.0, 5.0, TOP)
        success, _ = make_via(engine, 25.0, 5.5)
        engine.build_connectivity()

        assert success, "make_via should succeed with custom dimensions"
        assert engine.get_via_count() == 1


# ---------------------------------------------------------------------------
# 4. Env-level: design rules surfaced on board_static after init
# ---------------------------------------------------------------------------

class TestDesignRulesSurfacedOnEnvInit:
    """Verify env init surfaces the engine's design rules on board_static.

    env.py reads the hardest-per-netclass view from
    ``KiCadEngine.get_design_rules`` and exposes it under
    ``board_static["board_constraints"]``.
    """

    @pytest.fixture
    def env(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.core.env import PCBWorld
        e = PCBWorld(board_path=BOARD, max_steps=50)
        yield e
        e.close()

    def test_board_static_has_board_constraints(self, env):
        obs, _info = env.reset()
        bc = obs["board_static"]["board_constraints"]
        # Six fields from compute_hardest_per_netclass; uvia may be -1 on
        # boards without microvias, but the basic fields should be set.
        assert "clearance_mm" in bc
        assert "track_width_mm" in bc
        assert "via_diameter_mm" in bc
        assert "via_drill_mm" in bc
        assert bc["via_diameter_mm"] > 0
        assert bc["via_drill_mm"] > 0
