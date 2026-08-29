"""Connectivity and board query API tests (Step 3).

Tests:
- build_connectivity / recalculate_ratsnest
- get_net_count / get_board_net_count
- get_board_bbox / get_copper_layer_count
- get_router_state / get_failure_reason
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"

START = (0.0, 0.0)
END = (3.0, 5.0)


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


@pytest.fixture
def board_path() -> str:
    if not BOARD_PATH.exists():
        pytest.skip(f"Test board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


def _route_simple(r, krl) -> bool:
    r.set_routing_mode(krl.MODE_WALKAROUND)
    r.start_route(START[0], START[1], 0)
    r.move(1.5, 2.5)
    r.fix_route(1.5, 2.5, force_finish=False)
    return r.fix_route(END[0], END[1])


class TestBuildConnectivity:
    def test_build_connectivity_no_error(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.build_connectivity()

    def test_unrouted_decreases_after_routing(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)

        before = r.get_unrouted_count()
        assert before > 0

        _route_simple(r, krl)
        r.build_connectivity()

        after = r.get_unrouted_count()
        assert after < before

    def test_recalculate_ratsnest(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.recalculate_ratsnest()
        # Should not raise


class TestNetCount:
    def test_get_net_count(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        count = r.get_net_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_board_net_count(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        count = r.get_board_net_count()
        assert isinstance(count, int)
        assert count >= 2  # At least NET1 + NET_OBSTACLE + unconnected


class TestBoardBBox:
    def test_get_board_bbox(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        bbox = r.get_board_bbox()

        assert hasattr(bbox, "x_mm")
        assert hasattr(bbox, "y_mm")
        assert hasattr(bbox, "width_mm")
        assert hasattr(bbox, "height_mm")

        assert isinstance(bbox.x_mm, float)
        assert isinstance(bbox.width_mm, float)
        assert bbox.width_mm > 0
        assert bbox.height_mm > 0

    def test_bbox_contains_pads(self, board_path: str) -> None:
        """Bounding box should contain all pad positions."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        bbox = r.get_board_bbox()
        pads = r.get_pads()

        x_max = bbox.x_mm + bbox.width_mm
        y_max = bbox.y_mm + bbox.height_mm

        for p in pads:
            assert bbox.x_mm <= p.x_mm <= x_max, f"Pad {p.pad_name} x={p.x_mm} out of bbox"
            assert bbox.y_mm <= p.y_mm <= y_max, f"Pad {p.pad_name} y={p.y_mm} out of bbox"


class TestCopperLayerCount:
    def test_get_copper_layer_count(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        count = r.get_copper_layer_count()
        assert isinstance(count, int)
        assert count >= 2  # At least F.Cu + B.Cu


class TestRouterState:
    def test_idle_initially(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        assert r.get_router_state() == krl.STATE_IDLE

    def test_route_track_during_routing(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.start_route(0.0, 0.0, 0)
        assert r.get_router_state() == krl.STATE_ROUTE_TRACK
        r.cancel_route()

    def test_back_to_idle_after_cancel(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.start_route(0.0, 0.0, 0)
        r.cancel_route()
        assert r.get_router_state() == krl.STATE_IDLE


class TestFailureReason:
    def test_failure_reason_is_string(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        reason = r.get_failure_reason()
        assert isinstance(reason, str)


class TestBoundingBoxRepr:
    def test_repr(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        bbox = r.get_board_bbox()
        s = repr(bbox)
        assert "BoundingBox" in s


class TestCopperShapeRatsnestCoverage:
    """Copper shape clusters (e.g. circles) are also captured by the ratsnest —
    a regression guard for KiCad 9's ``PCB_SHAPE::GetConnectionPoints``, which
    gives filled shapes a centroid anchor point. If this behavior regresses
    upstream, the net count would silently drop.
    """

    def test_isolated_copper_circle_gets_ratsnest_edge(self) -> None:
        from pcb_world.engine.kicad_engine import KiCadEngine

        board = PROJECT_ROOT / "tests" / "fixtures" / "zero_anchor_board.kicad_pcb"
        eng = KiCadEngine(board_path=str(board))
        try:
            eng.build_connectivity()
            assert eng.get_unrouted_count() == 1
            edges = list(eng.get_ratsnest())
            assert len(edges) == 1
            pts = {(edges[0].x1_mm, edges[0].y1_mm), (edges[0].x2_mm, edges[0].y2_mm)}
            assert pts == {(95.0, 95.0), (100.0, 100.0)}   # pad <-> circle centroid
        finally:
            eng.close()


class TestRatsnestPadGroupInvariant:
    """Per-net invariant ``ratsnest edges >= pad_groups - 1`` — an early-warning
    guard against ratsnest undercounting.

    KiCad's ratsnest counts (RN-participating clusters - 1) edges per net, and
    pad groups (clusters containing a pad) are a subset of those clusters, so on
    any board a violation of per-net ``rats >= G - 1`` means an engine/KiCad
    change has swallowed a needed connection (e.g. the same-coordinate-anchor
    special case in ratsnest_data.cpp). On a stripped corpus board (pads/tracks/
    vias only, no dangling copper) the totals match exactly. The targets are
    three real d3b boards with same-coordinate pad stacks (zero-length
    ratsnest); skipped when the corpus is not available (the real-PCB corpus
    lives at ``$CADAGENT_DATA_ROOT/pcbench/exacad_sorted``, not in the repo).
    """

    BOARDS = [
        "0218_kitspace_GPSMux",
        "0225_mac-pro-conversion_front-panel-power-adapter",
        "0367_pcb-usb-ft245r-parallel-adapter_pcb-usb-ft245r-parallel-adapter",
    ]

    @staticmethod
    def _corpus() -> Path:
        import os

        root = os.environ.get("CADAGENT_DATA_ROOT")
        if not root:
            pytest.skip("CADAGENT_DATA_ROOT unset — real-PCB corpus unavailable")
        return Path(root) / "pcbench" / "exacad_sorted"

    @pytest.mark.parametrize("board", BOARDS)
    def test_unrouted_counts_agree(self, board: str) -> None:
        corpus = self._corpus()
        path = corpus / board / "processed_v9_guide_v3_unrouted.kicad_pcb"
        if not path.exists():
            pytest.skip(f"d3b corpus not available: {path}")
        # The _unrouted variant carries no .kicad_pro of its own (and the
        # corpus is read-only) — pass the routed sibling's project explicitly:
        # same project, same rules.
        pro = corpus / board / "processed_v9_guide_v3.kicad_pro"
        from collections import Counter

        from pcb_world.engine.kicad_engine import KiCadEngine

        eng = KiCadEngine(board_path=str(path), project_path=str(pro))
        try:
            eng.build_connectivity()
            rats = Counter(int(e.net_code) for e in eng.get_ratsnest())
            groups = eng.get_pad_groups()
        finally:
            eng.close()

        assert groups, "pad groups empty — corpus board loaded wrong"
        for net, g in sorted(groups.items()):
            assert rats.get(net, 0) >= g - 1, (
                f"{board} net {net}: ratsnest {rats.get(net, 0)} < pad_groups-1 {g - 1}"
                " — ratsnest has swallowed a needed connection (undercount)"
            )
        assert sum(rats.values()) == sum(g - 1 for g in groups.values() if g >= 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
