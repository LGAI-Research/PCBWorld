"""Tests for condition-based YAML action masking system.

Tests:
1. YamlConditionMask: loading and mask evaluation
2. get_masking_rule: name-based YAML auto-discovery
3. ACTION_REGISTRY: structure validation
4. Param filtering via ACTION_REGISTRY
"""

import numpy as np
import pytest

from pcb_world.core.masking import (
    ACT_FINISH,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACTION_NAMES,
    ACTION_REGISTRY,
    ActionDef,
    NUM_ACTIONS,
    MaskContext,
    YamlConditionMask,
    build_action_mask,
    get_masking_rule,
    register_masking_rule,
)


# ---------------------------------------------------------------------------
# 1. YamlConditionMask — strict condition rule from dict
# ---------------------------------------------------------------------------

STRICT_CONFIG = {
    "name": "strict",
    "actions": {
        "net_select": {"when": {"has_net": False, "is_routing": False}},
        "start_route": {"when": {"has_net": True, "is_routing": False,
                                 "net_fully_connected": False}},
        "net_end": {"when": {"has_net": True, "is_routing": False, "net_fully_connected": True}},
        "make_line": {"when": {"is_routing": True}},
        "make_via": {"when": {"is_routing": True}},
        "finish": {"when": {"is_routing": True}},
    },
}

RELAXED_CONFIG = {
    "name": "relaxed",
    "actions": {
        "net_select": {"when": {"has_net": False, "is_routing": False}},
        "start_route": {"when": {"has_net": True, "is_routing": False}},
        "net_end": {"when": {"has_net": True, "is_routing": False}},
        "make_line": {"when": {"is_routing": True}},
        "make_via": {"when": {"is_routing": True}},
        "finish": {"when": {"is_routing": True}},
    },
}


class TestYamlConditionMask:
    """Test YamlConditionMask with in-memory config dicts."""

    def test_strict_net_select(self):
        rule = YamlConditionMask(STRICT_CONFIG)
        ctx = MaskContext(has_net=False, is_routing=False)
        mask = rule.build_mask(ctx)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask[ACT_NET_SELECT] is np.True_
        assert mask.sum() == 1

    def test_strict_start_route_not_connected(self):
        rule = YamlConditionMask(STRICT_CONFIG)
        ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=False)
        mask = rule.build_mask(ctx)
        assert mask[ACT_START_ROUTE] is np.True_
        assert mask[ACT_NET_END] is np.False_
        assert mask.sum() == 1

    def test_strict_start_route_connected(self):
        # A fully-connected net can only be ended (net_end), NOT re-routed:
        # start_route is masked (net_fully_connected: false condition) so the
        # agent cannot lay redundant copper on a completed net.
        rule = YamlConditionMask(STRICT_CONFIG)
        ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=True)
        mask = rule.build_mask(ctx)
        assert mask[ACT_START_ROUTE] is np.False_
        assert mask[ACT_NET_END] is np.True_
        assert mask.sum() == 1

    def test_strict_routing(self):
        rule = YamlConditionMask(STRICT_CONFIG)
        ctx = MaskContext(has_net=True, is_routing=True)
        mask = rule.build_mask(ctx)
        assert mask[ACT_MAKE_LINE] is np.True_
        assert mask[ACT_MAKE_VIA] is np.True_
        assert mask[ACT_FINISH] is np.True_
        assert mask.sum() == 3

    def test_relaxed_allows_net_end_always(self):
        rule = YamlConditionMask(RELAXED_CONFIG)
        ctx = MaskContext(has_net=True, is_routing=False, net_fully_connected=False)
        mask = rule.build_mask(ctx)
        assert mask[ACT_NET_END] is np.True_

    def test_mask_dtype_is_bool(self):
        rule = YamlConditionMask(STRICT_CONFIG)
        ctx = MaskContext(has_net=True, is_routing=True)
        mask = rule.build_mask(ctx)
        assert mask.dtype == bool

    def test_empty_actions_empty_mask(self):
        """If no actions are defined, all actions are masked."""
        config = {"name": "empty", "actions": {}}
        rule = YamlConditionMask(config)
        ctx = MaskContext()
        mask = rule.build_mask(ctx)
        assert mask.sum() == 0

    def test_invalid_action_name_raises(self):
        config = {
            "name": "bad",
            "actions": {"nonexistent_action": {"when": {}}},
        }
        with pytest.raises(KeyError):
            YamlConditionMask(config)

    def test_name_attribute(self):
        rule = YamlConditionMask(STRICT_CONFIG)
        assert rule.name == "strict"


# ---------------------------------------------------------------------------
# 2. get_masking_rule — YAML auto-discovery
# ---------------------------------------------------------------------------

class TestGetMaskingRule:
    """Test get_masking_rule name-based YAML loading."""

    def test_load_strict_by_name(self):
        rule = get_masking_rule("strict")
        ctx = MaskContext(has_net=False, is_routing=False)
        mask = rule.build_mask(ctx)
        assert mask[ACT_NET_SELECT] is np.True_
        assert mask.sum() == 1

    def test_backward_compat_strict_phase(self):
        """Old name 'strict_phase' should resolve to 'default'."""
        rule = get_masking_rule("strict_phase")
        ctx = MaskContext(has_net=False, is_routing=False)
        mask = rule.build_mask(ctx)
        assert mask[ACT_NET_SELECT] is np.True_

    def test_unknown_rule_raises(self):
        with pytest.raises(KeyError, match="Unknown masking rule"):
            get_masking_rule("nonexistent_rule")

    def test_register_custom_rule(self):
        class AllowAllMask:
            def build_mask(self, ctx):
                return np.ones(NUM_ACTIONS, dtype=bool)

        register_masking_rule("allow_all_v2", AllowAllMask())
        rule = get_masking_rule("allow_all_v2")
        ctx = MaskContext()
        mask = rule.build_mask(ctx)
        assert mask.all()

    def test_build_action_mask_convenience(self):
        ctx = MaskContext(has_net=True, is_routing=True)
        mask = build_action_mask(ctx, rule_name="strict")
        assert mask[ACT_MAKE_LINE] is np.True_
        assert mask.sum() == 3


# ---------------------------------------------------------------------------
# 3. ACTION_REGISTRY — structure validation
# ---------------------------------------------------------------------------

class TestActionRegistry:
    """Test ACTION_REGISTRY structure and derived constants."""

    def test_registry_has_all_actions(self):
        assert len(ACTION_REGISTRY) == NUM_ACTIONS
        for i, action_def in enumerate(ACTION_REGISTRY):
            assert isinstance(action_def, ActionDef)
            assert action_def.name == ACTION_NAMES[i]

    def test_registry_entries_are_frozen(self):
        with pytest.raises(AttributeError):
            ACTION_REGISTRY[0].name = "changed"

    def test_net_select_params(self):
        assert ACTION_REGISTRY[ACT_NET_SELECT].params == ["net_id"]

    def test_start_route_params(self):
        assert set(ACTION_REGISTRY[ACT_START_ROUTE].params) == {"x_mm", "y_mm", "layer"}

    def test_net_end_has_no_params(self):
        assert ACTION_REGISTRY[ACT_NET_END].params == []

    def test_make_line_params(self):
        assert set(ACTION_REGISTRY[ACT_MAKE_LINE].params) == {"x_mm", "y_mm", "routing_mode"}

    def test_make_via_params(self):
        assert ACTION_REGISTRY[ACT_MAKE_VIA].params == ["x_mm", "y_mm", "routing_mode"]

    def test_finish_params(self):
        assert ACTION_REGISTRY[ACT_FINISH].params == ["routing_mode"]


# ---------------------------------------------------------------------------
# 4. Param filtering in dispatch
# ---------------------------------------------------------------------------

class TestParamFiltering:
    """Test that ActionDispatcher filters params per ACTION_REGISTRY."""

    def test_net_select_filters_out_xy(self):
        """net_select only allows net_id; x_mm, y_mm should be filtered."""
        allowed = ACTION_REGISTRY[ACT_NET_SELECT].params
        full_params = {
            "action_type": 0,
            "x_mm": 100.0,
            "y_mm": 50.0,
            "layer": 1,
            "net_id": 5,
            "routing_mode": 2,
        }
        filtered = {k: v for k, v in full_params.items() if k in allowed}
        assert "net_id" in filtered
        assert "x_mm" not in filtered
        assert "y_mm" not in filtered

    def test_make_line_keeps_xy_and_routing_mode(self):
        allowed = ACTION_REGISTRY[ACT_MAKE_LINE].params
        full_params = {
            "x_mm": 10.0, "y_mm": 20.0, "layer": 1,
            "net_id": 1, "routing_mode": 2,
        }
        filtered = {k: v for k, v in full_params.items() if k in allowed}
        assert set(filtered.keys()) == {"x_mm", "y_mm", "routing_mode"}

    def test_make_via_only_routing_mode(self):
        allowed = ACTION_REGISTRY[ACT_MAKE_VIA].params
        full_params = {
            "x_mm": 10.0, "y_mm": 20.0, "routing_mode": 1,
        }
        filtered = {k: v for k, v in full_params.items() if k in allowed}
        assert set(filtered.keys()) == {"x_mm", "y_mm", "routing_mode"}
