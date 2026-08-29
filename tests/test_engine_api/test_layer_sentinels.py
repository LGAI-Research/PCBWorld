"""Layer-sentinel contract regression tests (requires the v0.26+ engine).

Covers — [tests/fixtures/layer_sentinels_board.kicad_pcb](../fixtures/layer_sentinels_board.kicad_pcb):
- ``RatsnestEdge.layer1/2``: single-layer parent=PCB_LAYER_ID, multi-layer parent=-2 (RL_LAYER_SPANS_COPPER)
- ``get_pads().layer``: THT/multi-layer pad=-2 (older engines collapse this to F.Cu)
- ``get_routing_target``: the third element is the layer of the **target anchor**, not the head
- ``delete_track_near``/``delete_via_near``: layer/net filters are required

Skipped entirely on a pre-v0.26 ``.so`` — active after a rebuild
(engine/build_rl_router.sh).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

BOARD = PROJECT_ROOT / "tests" / "fixtures" / "layer_sentinels_board.kicad_pcb"

F_CU, B_CU = 0, 2                 # PCB_LAYER_ID
SPANS = -2                        # RL_LAYER_SPANS_COPPER


def _engine_has_layer_sentinels() -> bool:
    try:
        import kicad_rl_router as krl
    except ImportError:
        return False
    return hasattr(krl.RatsnestEdge, "layer1")


pytestmark = pytest.mark.skipif(
    not _engine_has_layer_sentinels(),
    reason="RatsnestEdge.layer1/2 requires the v0.26+ engine (skip before rebuild)",
)


@pytest.fixture
def eng():
    from pcb_world.engine.kicad_engine import KiCadEngine

    e = KiCadEngine(board_path=str(BOARD))
    e.build_connectivity()
    yield e
    e.close()


class TestRatsnestEdgeLayers:
    def test_stacked_smd_pair_carries_both_layers(self, eng) -> None:
        edges = [e for e in eng.get_ratsnest() if e.net_code == 1]
        assert len(edges) == 1
        assert {edges[0].layer1, edges[0].layer2} == {F_CU, B_CU}

    def test_tht_anchor_reports_spans_sentinel(self, eng) -> None:
        edges = [e for e in eng.get_ratsnest() if e.net_code == 2]
        assert len(edges) == 1
        assert {edges[0].layer1, edges[0].layer2} == {SPANS, F_CU}


class TestPadLayerSentinel:
    def test_pad_layers(self, eng) -> None:
        by_name = {p.pad_name: p for p in eng.get_pads()}
        assert by_name["1"].layer == F_CU
        assert by_name["2"].layer == B_CU
        assert by_name["3"].layer == SPANS      # older engines collapse this to F_CU


class TestRoutingTargetLayer:
    def test_target_layer_is_targets_not_heads(self, eng) -> None:
        # Start on THT(105,105) from the back side (human 2) — the target is
        # the F.Cu SMD pad at (108,105); the returned layer must be the
        # target's layer (1), not the head's layer (2).
        assert eng.start_route(105.0, 105.0, 2)
        try:
            x, y, layer = eng.get_routing_target()
            assert (round(x, 3), round(y, 3)) == (108.0, 105.0)
            assert layer == 1.0
        finally:
            eng.cancel_route()


class TestFixRouteExpectedLayer:
    def test_wrong_expected_layer_rejects_and_right_one_commits(self, eng) -> None:
        # When a make_line-style caller declares its intended layer, a commit
        # whose head arrives on a different layer is rejected as stuck.
        before = eng.get_track_count()
        assert eng.start_route(105.0, 105.0, 1)
        assert not eng.fix_route(106.0, 105.0, force_finish=True, expected_layer=2)
        assert not eng.is_routing()               # session ended via cancelRoute
        assert eng.get_track_count() == before    # nothing committed

        assert eng.start_route(105.0, 105.0, 1)
        assert eng.fix_route(106.0, 105.0, force_finish=True, expected_layer=1)
        assert eng.get_track_count() == before + 1


class TestDeleteFilters:
    def test_delete_track_near_respects_layer(self, eng) -> None:
        assert eng.delete_track_near(101, 108, 103, 108, human_layer=2, net_code=3)
        remaining = [t.layer for t in eng.get_tracks() if t.net_code == 3]
        assert remaining == [F_CU]
        assert not eng.delete_track_near(101, 108, 103, 108, human_layer=2, net_code=3)

    def test_delete_via_near_respects_net(self, eng) -> None:
        assert not eng.delete_via_near(104, 108, net_code=1)
        assert eng.delete_via_near(104, 108, net_code=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
