"""Unit tests for PCBWorld env components.

Tests:
1. pcb_file_parser: engine-backed dict view (board edges, vias, etc.)
2. state_json: BoardStatic.from_board(), build_json_observation()
3. phase_tracker: state transitions, sync safety
4. action_mask: all masking rules, registry
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. pcb_file_parser — engine-backed dict view
# ---------------------------------------------------------------------------

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "simple_obstacle_board.kicad_pcb"
)


class TestPcbFileParser:
    """Engine-backed dict view shape + value sanity."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_board(self):
        if not os.path.exists(BOARD_PATH):
            pytest.skip(f"Board not found: {BOARD_PATH}")

    @pytest.fixture
    def parsed(self):
        from pcb_world.engine.kicad_engine import KiCadEngine
        from pcb_world.engine.pcb_file_parser import parse_pcb_file
        engine = KiCadEngine(BOARD_PATH)
        try:
            return parse_pcb_file(BOARD_PATH, engine)
        finally:
            engine.close()

    def test_returns_expected_keys(self, parsed):
        assert "board_meta" in parsed
        assert "board_snapshot" in parsed
        assert "net_names" in parsed
        assert "board_edges" in parsed
        assert "parse_stats" in parsed
        # Design rules are now supplied by the engine
        # (see compute_hardest_per_netclass) — parser no longer owns them.
        assert "design_rules" not in parsed

    def test_board_edges_are_list(self, parsed):
        edges = parsed["board_edges"]
        assert isinstance(edges, list)
        # board should have at least some outline
        if len(edges) > 0:
            e = edges[0]
            assert hasattr(e, "x1_mm")
            assert hasattr(e, "y1_mm")
            assert hasattr(e, "x2_mm")
            assert hasattr(e, "y2_mm")
            assert hasattr(e, "width_mm")

    def test_board_snapshot_has_vias(self, parsed):
        snapshot = parsed["board_snapshot"]
        assert hasattr(snapshot, "vias")
        assert isinstance(snapshot.vias, list)

    def test_parse_stats_includes_new_fields(self, parsed):
        stats = parsed["parse_stats"]
        assert "vias" in stats
        assert "board_edges" in stats

    def test_net_names_are_dict(self, parsed):
        net_names = parsed["net_names"]
        assert isinstance(net_names, dict)
        # net 0 is usually "" (unconnected)
        for code, name in net_names.items():
            assert isinstance(code, int)
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# 2. state_json — BoardStatic.from_board(), build_json_observation()
# ---------------------------------------------------------------------------

class TestStateJson:
    """Test state_json dataclasses and observation builders."""

    @pytest.fixture
    def board_info(self):
        """Build a BoardStatic with mock data (no C++ needed)."""
        from pcb_world.engine.containers import BoardMeta
        from pcb_world.engine.pcb_file_parser import BoardEdge
        from pcb_world.core.observation import BoardStatic

        # Tiny pad mock — duck-typed for ``BoardStatic.from_board``.
        # The real producer is the engine; here we only need attribute
        # access for x/y/width/height/layer/net_code/shape.
        def PadInfoCompat(**kw):
            return SimpleNamespace(shape="", **kw)

        meta = BoardMeta(
            bbox_x=100.0, bbox_y=50.0,
            bbox_w=60.0, bbox_h=40.0,
            net_count=3, copper_layers=2,
        )
        pads = [
            PadInfoCompat(x_mm=110.0, y_mm=60.0, width_mm=1.0, height_mm=1.0, layer=1, net_code=1),
            PadInfoCompat(x_mm=150.0, y_mm=80.0, width_mm=1.0, height_mm=1.0, layer=1, net_code=1),
            PadInfoCompat(x_mm=120.0, y_mm=70.0, width_mm=0.8, height_mm=0.8, layer=2, net_code=2),
        ]
        edges = [
            BoardEdge(x1_mm=100.0, y1_mm=50.0, x2_mm=160.0, y2_mm=50.0, width_mm=0.1),
            BoardEdge(x1_mm=160.0, y1_mm=50.0, x2_mm=160.0, y2_mm=90.0, width_mm=0.1),
            BoardEdge(x1_mm=160.0, y1_mm=90.0, x2_mm=100.0, y2_mm=90.0, width_mm=0.1),
            BoardEdge(x1_mm=100.0, y1_mm=90.0, x2_mm=100.0, y2_mm=50.0, width_mm=0.1),
        ]
        net_names = {1: "VCC", 2: "GND"}
        # Shape matches ``compute_hardest_per_netclass`` output so production
        # and test paths exercise the same dict contract.
        board_constraints = {
            "clearance_mm":     0.20,
            "track_width_mm":   0.25,
            "via_diameter_mm":  0.60,
            "via_drill_mm":     0.30,
            "uvia_diameter_mm": -1.0,
            "uvia_drill_mm":    -1.0,
        }

        return BoardStatic.from_board(meta, pads, edges, net_names, board_constraints)

    def test_from_board_boardlines(self, board_info):
        assert len(board_info.boardlines) == 4
        # all coordinates should be in mm (within board bbox)
        for line in board_info.boardlines:
            assert 100.0 <= line.p1.x <= 160.0
            assert 50.0 <= line.p1.y <= 90.0

    def test_from_board_nets(self, board_info):
        assert 1 in board_info.nets
        assert 2 in board_info.nets
        assert board_info.nets[1].net_name == "VCC"
        assert len(board_info.nets[1].pads) == 2  # two pads on net 1
        assert len(board_info.nets[2].pads) == 1  # one pad on net 2

    def test_from_board_constraints(self, board_info):
        bc = board_info.board_constraints
        assert bc["track_width_mm"] > 0
        assert bc["clearance_mm"] > 0
        assert bc["via_diameter_mm"] > 0
        # uvia fields were -1 in the fixture (no microvias); the from_board
        # path passes the dict through untouched so the sentinel survives.
        assert bc["uvia_diameter_mm"] == -1.0

    def test_from_board_scale(self, board_info):
        # scale = max(60, 40) = 60
        assert board_info.scale == 60.0

    def test_build_json_observation_structure(self, board_info):
        from pcb_world.engine.containers import BoardSnapshot, RoutingSessionState
        from pcb_world.core.observation import build_json_observation

        snapshot = BoardSnapshot(tracks=[], vias=[], pads=[], ratsnest=[])
        router_session = RoutingSessionState()

        obs = build_json_observation(
            snapshot, router_session, board_info,
            step_count=5, max_steps=100,
            current_net_id=1, routing_mode=2,
        )

        assert "board_static" in obs
        assert "routing_geometry" in obs
        assert "router_head" in obs
        # board_meta fields are now part of board_static
        assert "bbox_x" in obs["board_static"]
        assert "scale" in obs["board_static"]
        assert "net_count" in obs["board_static"]

    def test_router_head_has_new_fields(self, board_info):
        from pcb_world.engine.containers import BoardSnapshot, RoutingSessionState
        from pcb_world.core.observation import build_json_observation

        snapshot = BoardSnapshot(tracks=[], vias=[], pads=[], ratsnest=[])
        router_session = RoutingSessionState()

        obs = build_json_observation(
            snapshot, router_session, board_info,
            current_net_id=1, routing_mode=1,
        )
        mode = obs["router_head"]
        # current_net_id=1, is_routing=False -> net_phase=1 (START_ROUTE equivalent)
        assert mode["current_net_phase"] == 1
        assert mode["routing_mode"] == 1
        assert "via_toggle" in mode
        assert "step_ratio" in mode

    def test_board_static_has_boardlines(self, board_info):
        from pcb_world.engine.containers import BoardSnapshot, RoutingSessionState
        from pcb_world.core.observation import build_json_observation

        snapshot = BoardSnapshot(tracks=[], vias=[], pads=[], ratsnest=[])
        router_session = RoutingSessionState()

        obs = build_json_observation(snapshot, router_session, board_info)
        ctx = obs["board_static"]
        assert len(ctx["boardlines"]) == 4
        assert len(ctx["nets"]) == 2
        assert "board_constraints" in ctx

    def test_json_serializable(self, board_info):
        """Observation should be JSON-serializable (for LLM agents)."""
        from pcb_world.engine.containers import BoardSnapshot, RoutingSessionState
        from pcb_world.core.observation import build_json_observation

        snapshot = BoardSnapshot(tracks=[], vias=[], pads=[], ratsnest=[])
        router_session = RoutingSessionState()

        obs = build_json_observation(snapshot, router_session, board_info)
        text = json.dumps(obs)
        assert len(text) > 0
        roundtrip = json.loads(text)
        assert roundtrip["router_head"]["step"] == 0


# ---------------------------------------------------------------------------
# 4. action_mask — masking rules and registry
# ---------------------------------------------------------------------------

class TestActionMask:
    """Test action masking rules and registry."""

    def test_strict_net_select(self):
        """No net selected, not routing -> only net_select allowed."""
        from pcb_world.core.masking import (
            build_action_mask, MaskContext,
            ACT_NET_SELECT, NUM_ACTIONS,
        )
        ctx = MaskContext(has_net=False, is_routing=False)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask.shape == (NUM_ACTIONS,)
        assert mask[ACT_NET_SELECT] == True
        assert mask.sum() == 1

    def test_strict_start_route_not_connected(self):
        """Net selected, not routing, not connected -> only start_route."""
        from pcb_world.core.masking import (
            build_action_mask, MaskContext,
            ACT_START_ROUTE, ACT_NET_END,
        )
        ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=False)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_START_ROUTE] == True
        assert mask[ACT_NET_END] == False
        assert mask.sum() == 1

    def test_strict_start_route_connected(self):
        """Net selected, not routing, connected -> net_end ONLY.

        A completed net cannot be re-routed: start_route requires
        ``net_fully_connected: false``, so laying redundant copper on a
        finished net is not even reachable.
        """
        from pcb_world.core.masking import (
            build_action_mask, MaskContext,
            ACT_START_ROUTE, ACT_NET_END,
        )
        ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=True)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_START_ROUTE] == False
        assert mask[ACT_NET_END] == True
        assert mask.sum() == 1

    def test_strict_routing(self):
        """Actively routing -> make_line, make_via, finish."""
        from pcb_world.core.masking import (
            build_action_mask, MaskContext,
            ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH,
        )
        ctx = MaskContext(has_net=True, is_routing=True)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_MAKE_LINE] == True
        assert mask[ACT_MAKE_VIA] == True
        assert mask[ACT_FINISH] == True
        assert mask.sum() == 3

    def test_backward_compat_rule_names(self):
        """Old rule name 'strict_phase' should still resolve (now -> 'default')."""
        from pcb_world.core.masking import build_action_mask, MaskContext, ACT_NET_SELECT
        ctx = MaskContext(has_net=False, is_routing=False)
        mask = build_action_mask(ctx, rule_name="strict_phase")
        assert mask[ACT_NET_SELECT] == True

    def test_unknown_rule_raises(self):
        from pcb_world.core.masking import build_action_mask, MaskContext
        ctx = MaskContext()
        with pytest.raises(KeyError, match="Unknown masking rule"):
            build_action_mask(ctx, rule_name="nonexistent_rule")

    def test_register_custom_rule(self):
        from pcb_world.core.masking import (
            MaskContext, NUM_ACTIONS,
            register_masking_rule, build_action_mask,
        )

        class AllowAllMask:
            def build_mask(self, ctx):
                return np.ones(NUM_ACTIONS, dtype=bool)

        register_masking_rule("allow_all", AllowAllMask())
        ctx = MaskContext()
        mask = build_action_mask(ctx, rule_name="allow_all")
        assert mask.all()

    def test_mask_dtype_is_bool(self):
        from pcb_world.core.masking import build_action_mask, MaskContext
        ctx = MaskContext(has_net=True, is_routing=True)
        mask = build_action_mask(ctx)
        assert mask.dtype == bool
