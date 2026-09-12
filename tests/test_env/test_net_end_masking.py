"""net_end precondition ownership: the masking rule, not the handler.

Contract (v0.11.26): ``net_end``'s mechanical semantics is "deselect the
current net" and the handler never fails on an unfinished net — the
"fully connected" precondition lives ONLY in the masking rule
(``configs/masking/default*.yaml`` require ``net_fully_connected``), which
``env.step`` enforces before dispatch for every caller (mask_reject skips
dispatch). A masking rule that exposes net_end early therefore enables
give-up/skip semantics with no env-side knob.

Pure-Python: stub engine (net_end touches only build_connectivity +
get_ratsnest); mask evaluated directly via MaskContext.
"""

from __future__ import annotations

from types import SimpleNamespace

from pcb_world.core.action import ActionDispatcher, net_end
from pcb_world.core.action_schema import ACT_NET_END
from pcb_world.core.masking import MaskContext, YamlConditionMask, get_masking_rule


class _StubEngine:
    def __init__(self, remaining: int, net_code: int = 1) -> None:
        self._edges = [SimpleNamespace(net_code=net_code)] * remaining

    def build_connectivity(self) -> None:
        pass

    def get_ratsnest(self):
        return self._edges


def _ctx(fully_connected: bool) -> MaskContext:
    return MaskContext(
        has_net=True, is_routing=False, net_fully_connected=fully_connected,
        step_count=0, max_steps=200, unrouted_count=3,
    )


class TestHandlerNeverGuards:
    """The handler deselects regardless of completion (mask's job)."""

    def test_net_end_succeeds_on_incomplete_net(self):
        ok, info = net_end(_StubEngine(remaining=2), 1)
        assert ok
        assert info["remaining"] == 2

    def test_net_end_succeeds_on_complete_net(self):
        ok, info = net_end(_StubEngine(remaining=0), 1)
        assert ok
        assert info["remaining"] == 0

    def test_dispatch_deselects_incomplete_net(self):
        d = ActionDispatcher()
        d.current_net_id = 1
        result = d.dispatch(_StubEngine(remaining=3), ACT_NET_END, {})
        assert result.success
        assert d.current_net_id is None

    def test_dispatch_without_net_still_fails(self):
        # The null-safety guard (not a state predicate) stays.
        d = ActionDispatcher()
        result = d.dispatch(_StubEngine(remaining=0), ACT_NET_END, {})
        assert not result.success


class TestMaskOwnsThePrecondition:
    """Default rules require fully-connected; a permissive rule may not."""

    def test_default_rule_blocks_early_net_end(self):
        rule = get_masking_rule("default")
        assert not rule.build_mask(_ctx(fully_connected=False))[ACT_NET_END]
        assert rule.build_mask(_ctx(fully_connected=True))[ACT_NET_END]

    def test_default_no_finish_rule_blocks_early_net_end(self):
        rule = get_masking_rule("default_no_finish")
        assert not rule.build_mask(_ctx(fully_connected=False))[ACT_NET_END]

    def test_permissive_rule_exposes_early_net_end(self):
        rule = YamlConditionMask({
            "name": "netend_free_inline",
            "actions": {
                "net_end": {"when": {"has_net": True, "is_routing": False}},
            },
        })
        assert rule.build_mask(_ctx(fully_connected=False))[ACT_NET_END]
