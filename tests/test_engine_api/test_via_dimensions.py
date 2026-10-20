"""Tests for set_via_diameter / set_via_drill.

Verifies that:
1. Design rules from the board are visible via the engine's
   ``get_design_rules`` API (BDS + NetSettings from the C++ loader).
2. set_via_diameter / set_via_drill override the current routing session
   (set after start_route; the next start_route re-imports the net class).
3. Placed vias reflect the session's dimensions.
"""

import os
import pytest
from configs.loader.schema import EnvConfig
from tests.helpers.engine import open_engine

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
        eng = open_engine(BOARD)
        return eng.get_design_rules()

    def test_engine_via_diameter_from_default_netclass(self, rules):
        # Fixture's companion .kicad_pro puts via_diameter on the Default
        # netclass (0.6 mm). The per-netclass view is the reliable
        # source post-engine-refactor.
        assert rules.default_netclass.via_diameter_mm == pytest.approx(0.6, abs=0.01)

    def test_engine_via_drill_from_default_netclass(self, rules):
        assert rules.default_netclass.via_drill_mm == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Engine-level set_via_diameter / set_via_drill (session override)
# ---------------------------------------------------------------------------

class TestSetViaDimensions:
    """set_via_diameter / set_via_drill override the CURRENT routing session."""

    @pytest.fixture
    def engine(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        e = open_engine(BOARD)
        e.build_connectivity()
        return e

    def test_start_route_imports_netclass_dimensions(self, engine):
        """start_route sets the sizes from the start net's class (0.6 / 0.3)."""
        assert engine.start_route(25.0, 5.0, TOP)
        sizes = engine.get_route_sizes()
        assert sizes["via_diameter"] == pytest.approx(0.6, abs=1e-6)
        assert sizes["via_drill"] == pytest.approx(0.3, abs=1e-6)

    def test_set_via_diameter_overrides_session(self, engine):
        assert engine.start_route(25.0, 5.0, TOP)
        engine.set_via_diameter(0.8)
        sizes = engine.get_route_sizes()
        assert sizes["via_diameter"] == pytest.approx(0.8, abs=1e-6)
        assert sizes["via_drill"] == pytest.approx(0.3, abs=1e-6)

    def test_set_via_drill_overrides_session(self, engine):
        assert engine.start_route(25.0, 5.0, TOP)
        engine.set_via_drill(0.4)
        sizes = engine.get_route_sizes()
        assert sizes["via_diameter"] == pytest.approx(0.6, abs=1e-6)
        assert sizes["via_drill"] == pytest.approx(0.4, abs=1e-6)

    @pytest.mark.parametrize("setter", ["set_via_diameter", "set_via_drill"])
    def test_set_while_idle_raises(self, engine, setter):
        """No session open: the next start_route would overwrite the value without a
        trace, so the call is an error instead of a silent no-op."""
        with pytest.raises(RuntimeError, match="while a route is in progress"):
            getattr(engine, setter)(0.8)
        assert engine.start_route(25.0, 5.0, TOP)
        sizes = engine.get_route_sizes()
        assert sizes["via_diameter"] == pytest.approx(0.6, abs=1e-6)
        assert sizes["via_drill"] == pytest.approx(0.3, abs=1e-6)

    @pytest.mark.parametrize("setter", ["set_via_diameter", "set_via_drill"])
    @pytest.mark.parametrize("value", [0, -0.3])
    def test_non_positive_value_raises(self, engine, setter, value):
        with pytest.raises(ValueError, match="must be > 0"):
            getattr(engine, setter)(value)


# ---------------------------------------------------------------------------
# 3. Dimensions of the placed via after make_via
# ---------------------------------------------------------------------------

class TestViaDimensionsInPlacedVia:
    """Verify the placed via carries the session's via dimensions."""

    @pytest.fixture
    def engine(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        e = open_engine(BOARD)
        e.build_connectivity()
        return e

    def test_placed_via_with_board_defaults(self, engine):
        """No override: the via gets the net class dimensions (0.6 dia, 0.3 drill)."""
        from pcb_world.core.action import make_via

        engine.start_route(25.0, 5.0, TOP)
        success, _ = make_via(engine, 25.0, 5.5)
        engine.build_connectivity()

        assert success, "make_via should succeed"
        assert engine.get_via_count() == 1

        vias = engine.get_vias()
        assert len(vias) == 1
        v = vias[0]
        assert abs(v.x_mm - 25.0) < 0.5
        assert v.diameter_mm == pytest.approx(0.6, abs=1e-6)
        assert v.drill_mm == pytest.approx(0.3, abs=1e-6)

    def test_placed_via_with_custom_dimensions(self, engine):
        """Override after start_route: the via is placed at 0.8 dia, 0.4 drill."""
        from pcb_world.core.action import make_via

        engine.start_route(25.0, 5.0, TOP)
        engine.set_via_diameter(0.8)
        engine.set_via_drill(0.4)
        success, _ = make_via(engine, 25.0, 5.5)
        engine.build_connectivity()

        assert success, "make_via should succeed with custom dimensions"
        assert engine.get_via_count() == 1
        v = engine.get_vias()[0]
        assert v.diameter_mm == pytest.approx(0.8, abs=1e-6)
        assert v.drill_mm == pytest.approx(0.4, abs=1e-6)

    def test_override_does_not_carry_to_next_route(self, engine):
        """The next start_route re-imports the net class dimensions."""
        from pcb_world.core.action import make_via

        engine.start_route(25.0, 5.0, TOP)
        engine.set_via_diameter(0.8)
        engine.set_via_drill(0.4)
        success, _ = make_via(engine, 25.0, 5.5)
        assert success

        assert engine.start_route(25.0, 5.5, BOTTOM)
        sizes = engine.get_route_sizes()
        assert sizes["via_diameter"] == pytest.approx(0.6, abs=1e-6)
        assert sizes["via_drill"] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Env-level: design rules surfaced on board_static after init
# ---------------------------------------------------------------------------

class TestDesignRulesSurfacedOnEnvInit:
    """Verify env init surfaces the engine's design rules on board_static.

    Under the engine-based refactor, env.py no longer copies pcb-setup
    via dimensions into the engine; instead it reads the hardest-per
    -netclass view from ``KiCadEngine.get_design_rules`` and exposes it
    under ``board_static["board_constraints"]``.
    """

    @pytest.fixture
    def env(self):
        _skip_if_no_board()
        _skip_if_no_kicad()
        from pcb_world.core.env import PCBWorld
        e = PCBWorld(BOARD, EnvConfig.with_overrides(max_steps=50))
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


def test_size_and_via_setters_are_visible_to_the_next_query(board_path=BOARD):
    """get -> set -> get over the default IPC client: the setters change what
    get_route_sizes / is_placing_via return, so they must flush the query cache."""

    e = open_engine(board_path)
    try:
        pad = e.get_pads()[0]
        assert e.start_route(pad.x_mm, pad.y_mm, 1)
        # One setter between two queries each: a later invalidating call must not
        # mask a setter that left a stale cached answer.
        for key, setter, delta in (("track_width", "set_track_width", 0.3),
                                   ("via_diameter", "set_via_diameter", 0.2),
                                   ("via_drill", "set_via_drill", 0.1)):
            before = e.get_route_sizes()[key]
            getattr(e, setter)(before + delta)
            assert e.get_route_sizes()[key] == pytest.approx(before + delta), setter
        assert e.is_placing_via() is False
        e.set_via_placement(True)
        assert e.is_placing_via() is True
    finally:
        if e.is_routing():
            e.cancel_route()
        e.close()
