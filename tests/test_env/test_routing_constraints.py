"""Tests for routing constraint enforcement.

Verifies three key constraints during net routing:

1. Net-filtered candidates: After net_select, only the selected net's
   geometry (pads, tracks, ratsnest, vias) appears in the candidate pool.
2. Directional candidates: 8-direction candidates are added ONLY when
   is_routing=True, not during net_select or start_route phases.
3. current_net_id lifecycle: current_net_id persists from net_select
   through all routing actions until net_end clears it.

Tests 1-2 are pure unit tests (no C++ dependency).
Test 3 uses the ActionDispatcher with a mock engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pcb_world.vec.candidate_pool import (
    CAND_FEATURES,
    CTYPE_DIRECTIONAL,
    CTYPE_PAD,
    CTYPE_RATSNEST,
    CTYPE_TRACK,
    MAX_CANDIDATES,
    NUM_CAND_TYPES,
    build_candidates_mlp,
    build_directional_candidates,
)
from pcb_world.core.masking import (
    ACT_FINISH,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    MaskContext,
    YamlConditionMask,
    build_action_mask,
)
from pcb_world.core.action import ActionDispatcher


# ---------------------------------------------------------------------------
# Helpers — obs builders (same style as test_candidate_builder.py)
# ---------------------------------------------------------------------------

def _pad(x, y, layer=1):
    return {"center": {"xy": [x, y]}, "layer": layer}


def _track(x1, y1, x2, y2, layer=1):
    return {"p1": {"xy": [x1, y1]}, "p2": {"xy": [x2, y2]}, "layer": layer}


def _ratsnest(x, y, layer=1):
    return {"xy": [x, y], "layer": layer}


def _via(x, y, layer_start=1, layer_end=2):
    return {"center": {"xy": [x, y]}, "layer_start": layer_start, "layer_end": layer_end}


def _make_multi_net_obs(
    net1_pads=None, net1_tracks=None, net1_ratsnest=None,
    net2_pads=None, net2_tracks=None, net2_ratsnest=None,
    bbox_x=0.0, bbox_y=0.0, scale=100.0,
):
    """Build obs with two nets (net_1 and net_2) for filtering tests."""
    board_static = {
        "bbox_x": bbox_x, "bbox_y": bbox_y, "scale": scale,
        "nets": {},
    }
    routing_geometry = {}

    for net_id, pads, tracks, ratsnest in [
        (1, net1_pads, net1_tracks, net1_ratsnest),
        (2, net2_pads, net2_tracks, net2_ratsnest),
    ]:
        net_key = f"net_{net_id}"
        if pads is not None:
            board_static["nets"][net_key] = {
                "pads": {f"pad_{i}": p for i, p in enumerate(pads)},
            }
        net_geom = {}
        if tracks is not None:
            net_geom["tracks"] = {f"track_{i}": t for i, t in enumerate(tracks)}
        if ratsnest is not None:
            net_geom["points"] = ratsnest
        if net_geom:
            routing_geometry[net_key] = net_geom

    return {"board_static": board_static, "routing_geometry": routing_geometry}


# ---------------------------------------------------------------------------
# 1. Net-filtered candidates
# ---------------------------------------------------------------------------

class TestNetFilteredCandidates:
    """Candidates must only contain geometry from the selected net."""

    def test_selects_only_net1_pads(self):
        """When current_net_id=1, only net_1 pads appear."""
        obs = _make_multi_net_obs(
            net1_pads=[_pad(10, 10), _pad(20, 20)],
            net2_pads=[_pad(50, 50), _pad(60, 60)],
        )
        _, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert mask.sum() == 2
        coords = {(x, y) for x, y, _ in mm}
        assert (10, 10) in coords
        assert (20, 20) in coords
        assert (50, 50) not in coords
        assert (60, 60) not in coords

    def test_selects_only_net2_pads(self):
        """When current_net_id=2, only net_2 pads appear."""
        obs = _make_multi_net_obs(
            net1_pads=[_pad(10, 10)],
            net2_pads=[_pad(50, 50)],
        )
        _, mask, mm = build_candidates_mlp(obs, current_net_id=2, route_head_mm=(0, 0))
        assert mask.sum() == 1
        assert mm[0][:2] == (50, 50)

    def test_selects_only_current_net_tracks(self):
        """Tracks from other nets are excluded."""
        obs = _make_multi_net_obs(
            net1_tracks=[_track(0, 0, 10, 10)],
            net2_tracks=[_track(50, 50, 60, 60)],
        )
        _, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        coords = {(x, y) for x, y, _ in mm}
        # net_1 track endpoints
        assert (0, 0) in coords or (10, 10) in coords
        # net_2 track endpoints must NOT appear
        assert (50, 50) not in coords
        assert (60, 60) not in coords

    def test_ratsnest_is_not_pointer_selectable(self):
        # Ratsnest points are context-only (state token) and not a
        # pointer-selectable candidate — per-endpoint layer is not
        # reliably available, so they would emit a fake layer signal and
        # steal the slot of the real pad. See candidate_builder's
        # ``collect_raw_candidates`` for the rationale.
        obs = _make_multi_net_obs(
            net1_ratsnest=[_ratsnest(10, 10)],
            net2_ratsnest=[_ratsnest(90, 90)],
        )
        _, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        coords = {(x, y) for x, y, _ in mm}
        assert mask.sum() == 0
        assert (10, 10) not in coords
        assert (90, 90) not in coords

    def test_no_net_returns_empty_candidates(self):
        """When current_net_id=None, no geometry candidates are returned."""
        obs = _make_multi_net_obs(
            net1_pads=[_pad(10, 10)],
            net2_pads=[_pad(50, 50)],
        )
        _, mask, mm = build_candidates_mlp(obs, current_net_id=None, route_head_mm=(0, 0))
        assert mask.sum() == 0
        assert len(mm) == 0

    def test_switching_net_changes_candidates(self):
        """Changing current_net_id switches the candidate pool entirely."""
        obs = _make_multi_net_obs(
            net1_pads=[_pad(10, 10)],
            net2_pads=[_pad(50, 50)],
        )
        # Select net 1
        _, mask1, mm1 = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        # Select net 2
        _, mask2, mm2 = build_candidates_mlp(obs, current_net_id=2, route_head_mm=(0, 0))

        assert mask1.sum() == 1
        assert mm1[0][:2] == (10, 10)
        assert mask2.sum() == 1
        assert mm2[0][:2] == (50, 50)


# ---------------------------------------------------------------------------
# 2. Directional candidates only during routing
# ---------------------------------------------------------------------------

class TestDirectionalCandidatesTiming:
    """8-direction candidates must appear ONLY when is_routing=True."""

    def test_directional_not_added_without_extra(self):
        """Without extra_candidates, no DIRECTIONAL type appears."""
        obs = _make_multi_net_obs(net1_pads=[_pad(10, 10)])
        feat, mask, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(10, 10))
        for i in range(int(mask.sum())):
            assert feat[i, 3 + CTYPE_DIRECTIONAL] == 0.0

    def test_directional_added_with_extra(self):
        """With extra_candidates from build_directional_candidates, DIRECTIONAL appears."""
        obs = _make_multi_net_obs(net1_pads=[_pad(10, 10)])
        dir_cands = build_directional_candidates((10, 10), current_layer=1)
        feat, mask, _ = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(10, 10), extra_candidates=dir_cands,
        )
        # Should have pad(s) + directional candidates
        directional_count = sum(
            1 for i in range(int(mask.sum()))
            if feat[i, 3 + CTYPE_DIRECTIONAL] == 1.0
        )
        assert directional_count == len(dir_cands)

    def test_directional_candidates_are_8_directions(self):
        """build_directional_candidates produces 8 distinct direction vectors."""
        hx, hy = 50.0, 50.0
        cands = build_directional_candidates((hx, hy), current_layer=1)
        # Extract direction vectors (normalized)
        directions = set()
        for x, y, _, _ in cands:
            dx, dy = x - hx, y - hy
            # Normalize to unit direction
            mag = (dx ** 2 + dy ** 2) ** 0.5
            if mag > 0:
                directions.add((round(dx / mag, 3), round(dy / mag, 3)))
        assert len(directions) == 8

    def test_directional_candidates_use_current_layer(self):
        """All directional candidates must be on the specified layer."""
        cands = build_directional_candidates((50, 50), current_layer=3)
        for _, _, layer, _ in cands:
            assert layer == 3

    def test_directional_candidates_all_typed_correctly(self):
        """All directional candidates must have CTYPE_DIRECTIONAL."""
        cands = build_directional_candidates((50, 50), current_layer=1)
        for _, _, _, ctype in cands:
            assert ctype == CTYPE_DIRECTIONAL

    def test_wrapper_logic_routing_adds_directional(self):
        """Simulate the wrapper logic: directional only when is_routing=True.

        This mirrors the exact logic in KiCadHLWrapper._do_build_candidates_mlp().
        """
        obs = _make_multi_net_obs(net1_pads=[_pad(10, 10)])
        head_mm = (10.0, 10.0)

        # Phase 1: NOT routing — no directional candidates
        mode_not_routing = {"is_routing": False, "current_layer": 0}
        extra = None
        if mode_not_routing.get("is_routing", False):
            extra = build_directional_candidates(head_mm, mode_not_routing["current_layer"])
        feat1, mask1, _ = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=head_mm, extra_candidates=extra,
        )
        dir_count_1 = sum(
            1 for i in range(int(mask1.sum()))
            if feat1[i, 3 + CTYPE_DIRECTIONAL] == 1.0
        )
        assert dir_count_1 == 0, "No directional candidates when not routing"

        # Phase 2: IS routing — directional candidates added
        mode_routing = {"is_routing": True, "current_layer": 0}
        extra = None
        if mode_routing.get("is_routing", False):
            extra = build_directional_candidates(head_mm, mode_routing["current_layer"])
        feat2, mask2, _ = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=head_mm, extra_candidates=extra,
        )
        dir_count_2 = sum(
            1 for i in range(int(mask2.sum()))
            if feat2[i, 3 + CTYPE_DIRECTIONAL] == 1.0
        )
        assert dir_count_2 > 0, "Directional candidates must appear when routing"


# ---------------------------------------------------------------------------
# 3. current_net_id lifecycle (net_select → routing → net_end)
# ---------------------------------------------------------------------------

class TestNetIdLifecycle:
    """current_net_id must persist from net_select until net_end."""

    @pytest.fixture
    def dispatcher(self):
        return ActionDispatcher()

    @pytest.fixture
    def mock_engine(self):
        """Mock engine that simulates routing lifecycle.

        ``get_ratsnest`` returns edges for the net_ids these tests select
        (3 and 5) so the per-net validation in ``net_select`` accepts them.
        Tests that exercise the "net is fully routed" path (e.g.
        :meth:`test_net_end_clears_net_id_when_connected`) override this to
        ``[]`` after net_select succeeds.
        """
        engine = MagicMock()
        engine.is_routing.return_value = False
        engine.start_route.return_value = True
        engine.fix_route.return_value = True
        engine.finish.return_value = True
        engine.set_routing_mode.return_value = None
        engine.toggle_via.return_value = None
        engine.build_connectivity.return_value = None
        # make_via confirms the via landed by the via_count delta: 0 -> 1.
        engine.get_via_count.side_effect = [0, 1] * 16
        engine.get_ratsnest.return_value = [
            SimpleNamespace(net_code=3),
            SimpleNamespace(net_code=5),
        ]
        return engine

    def test_initial_state_no_net(self, dispatcher):
        """At init, current_net_id is None."""
        assert dispatcher.current_net_id is None

    def test_net_select_sets_net_id(self, dispatcher, mock_engine):
        """net_select sets current_net_id."""
        result = dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        assert result.success
        assert dispatcher.current_net_id == 5

    def test_start_route_preserves_net_id(self, dispatcher, mock_engine):
        """start_route does NOT change current_net_id."""
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        dispatcher.dispatch(mock_engine, ACT_START_ROUTE, {
            "x_mm": 10.0, "y_mm": 20.0, "layer": 1,
        })
        assert dispatcher.current_net_id == 5

    def test_make_line_preserves_net_id(self, dispatcher, mock_engine):
        """make_line does NOT change current_net_id."""
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        dispatcher.dispatch(mock_engine, ACT_START_ROUTE, {
            "x_mm": 10.0, "y_mm": 20.0, "layer": 1,
        })
        dispatcher.dispatch(mock_engine, ACT_MAKE_LINE, {
            "x_mm": 30.0, "y_mm": 40.0, "routing_mode": 2,
        })
        assert dispatcher.current_net_id == 5

    def test_make_via_preserves_net_id(self, dispatcher, mock_engine):
        """make_via does NOT change current_net_id."""
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        dispatcher.dispatch(mock_engine, ACT_START_ROUTE, {
            "x_mm": 10.0, "y_mm": 20.0, "layer": 1,
        })
        dispatcher.dispatch(mock_engine, ACT_MAKE_VIA, {
            "x_mm": 30.0, "y_mm": 40.0, "routing_mode": 2,
        })
        assert dispatcher.current_net_id == 5

    def test_finish_preserves_net_id(self, dispatcher, mock_engine):
        """finish does NOT change current_net_id."""
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        dispatcher.dispatch(mock_engine, ACT_START_ROUTE, {
            "x_mm": 10.0, "y_mm": 20.0, "layer": 1,
        })
        dispatcher.dispatch(mock_engine, ACT_FINISH, {"routing_mode": 2})
        assert dispatcher.current_net_id == 5

    def test_net_end_clears_net_id_when_connected(self, dispatcher, mock_engine):
        """net_end clears current_net_id when net is fully connected."""
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        # Engine returns empty ratsnest → net is fully connected
        mock_engine.get_ratsnest.return_value = []
        result = dispatcher.dispatch(mock_engine, ACT_NET_END, {})
        assert result.success
        assert dispatcher.current_net_id is None

    def test_net_end_deselects_even_when_not_connected(self, dispatcher, mock_engine):
        """The handler deselects regardless of remaining edges — the
        "fully connected" precondition is owned by the masking rule and
        enforced by env.step before dispatch (see test_net_end_masking.py)."""
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        # Simulate remaining ratsnest edge for net 5
        remaining_edge = SimpleNamespace(net_code=5)
        mock_engine.get_ratsnest.return_value = [remaining_edge]
        result = dispatcher.dispatch(mock_engine, ACT_NET_END, {})
        assert result.success
        assert result.info["remaining"] == 1
        assert dispatcher.current_net_id is None

    def test_full_lifecycle(self, dispatcher, mock_engine):
        """Full cycle: net_select → start_route → make_line → finish → net_end."""
        # Step 1: net_select
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 3})
        assert dispatcher.current_net_id == 3

        # Step 2: start_route
        dispatcher.dispatch(mock_engine, ACT_START_ROUTE, {
            "x_mm": 10.0, "y_mm": 20.0, "layer": 1,
        })
        assert dispatcher.current_net_id == 3

        # Step 3: make_line (multiple)
        for _ in range(3):
            dispatcher.dispatch(mock_engine, ACT_MAKE_LINE, {
                "x_mm": 30.0, "y_mm": 40.0, "routing_mode": 2,
            })
            assert dispatcher.current_net_id == 3

        # Step 4: finish
        dispatcher.dispatch(mock_engine, ACT_FINISH, {"routing_mode": 2})
        assert dispatcher.current_net_id == 3

        # Step 5: net_end (fully connected)
        mock_engine.get_ratsnest.return_value = []
        result = dispatcher.dispatch(mock_engine, ACT_NET_END, {})
        assert result.success
        assert dispatcher.current_net_id is None

    def test_reset_clears_net_id(self, dispatcher, mock_engine):
        """reset() clears current_net_id."""
        dispatcher.dispatch(mock_engine, ACT_NET_SELECT, {"net_id": 5})
        assert dispatcher.current_net_id == 5
        dispatcher.reset()
        assert dispatcher.current_net_id is None

    def test_cannot_select_new_net_without_net_end(self):
        """Masking prevents net_select while current_net_id is set."""
        # has_net=True → net_select is masked out
        ctx = MaskContext(has_net=True, is_routing=False)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_NET_SELECT] == False

    def test_can_select_net_after_net_end(self):
        """After net_end clears net_id, net_select becomes available again."""
        ctx = MaskContext(has_net=False, is_routing=False)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_NET_SELECT] == True


# ---------------------------------------------------------------------------
# 4. Masking transitions through routing phases
# ---------------------------------------------------------------------------

class TestMaskingPhaseTransitions:
    """Verify that masking correctly restricts actions at each phase."""

    def test_phase_initial_only_net_select(self):
        """No net, not routing → only net_select allowed."""
        ctx = MaskContext(has_net=False, is_routing=False)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_NET_SELECT] == True
        assert mask.sum() == 1

    def test_phase_net_selected_only_start_route(self):
        """Net selected, not routing, not connected → only start_route."""
        ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=False)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_START_ROUTE] == True
        assert mask[ACT_NET_END] == False
        assert mask[ACT_NET_SELECT] == False
        assert mask.sum() == 1

    def test_phase_routing_only_line_via_finish(self):
        """Routing active → only make_line, make_via, finish."""
        ctx = MaskContext(has_net=True, is_routing=True)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_MAKE_LINE] == True
        assert mask[ACT_MAKE_VIA] == True
        assert mask[ACT_FINISH] == True
        assert mask[ACT_NET_SELECT] == False
        assert mask[ACT_START_ROUTE] == False
        assert mask[ACT_NET_END] == False
        assert mask.sum() == 3

    def test_phase_connected_allows_only_net_end(self):
        """Net fully connected, not routing → net_end ONLY.

        The completed net is closed out, never re-opened: start_route carries a
        ``net_fully_connected: false`` condition so the agent cannot lay
        redundant copper on a net that is already done.
        """
        ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=True)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_START_ROUTE] == False
        assert mask[ACT_NET_END] == True
        assert mask.sum() == 1

    def test_strict_no_finish_disables_finish(self):
        """strict_no_finish rule: finish action is never available."""
        ctx = MaskContext(has_net=True, is_routing=True)
        mask = build_action_mask(ctx, rule_name="strict_no_finish")
        assert mask[ACT_MAKE_LINE] == True
        assert mask[ACT_MAKE_VIA] == True
        assert mask[ACT_FINISH] == False
        assert mask.sum() == 2
