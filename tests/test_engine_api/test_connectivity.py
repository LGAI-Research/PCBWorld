"""Connectivity and board query API tests (Step 3).

Tests:
- build_connectivity / recalculate_ratsnest
- get_net_count / get_board_net_count
- get_board_bbox / get_copper_layer_count
- get_router_state / get_failure_reason
"""

import itertools
import sys
from pathlib import Path

import pytest
from tests.helpers.engine import open_engine

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
        board = PROJECT_ROOT / "tests" / "fixtures" / "zero_anchor_board.kicad_pcb"
        eng = open_engine(board_path=str(board))
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
    is the ``pcbench_exacad`` dataset of configs/paths.yaml under the data
    root, which ``PCBWORLD_DATA_ROOT`` overrides).
    """

    BOARDS = [
        "0218_kitspace_GPSMux",
        "0225_mac-pro-conversion_front-panel-power-adapter",
        "0367_pcb-usb-ft245r-parallel-adapter_pcb-usb-ft245r-parallel-adapter",
    ]

    @staticmethod
    def _corpus() -> Path:
        from configs.loader.paths import resolve_dataset

        try:
            corpus = resolve_dataset("pcbench_exacad")
        except RuntimeError as e:  # empty data root (release configuration)
            pytest.skip(str(e))
        if not corpus.is_dir():
            pytest.skip(
                f"d3b corpus not available: {corpus} — point "
                "PCBWORLD_DATA_ROOT at a dataset root holding the "
                "pcbench_exacad layout (configs/paths.yaml)."
            )
        return corpus

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

        eng = open_engine(
            board_path=str(path),
            project_path=str(pro),
        )
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


class TestRatsnestContentOrder:
    """The ratsnest is a function of board content, not of item order.

    A front and a back SMD pad stacked at one spot are two clusters at one
    position; the MST edge from the third pad may land on either. Which one
    used to follow the CN_ITEM heap-address order (SearchClusters'
    ``std::set<CN_ITEM*>``), so identical boards gave different ratsnest
    layers run to run (D3-A 0067, 2026-10-20). Two boards differing only in
    the pad order of the footprint — so in allocation order — must agree.
    """

    PADS = {
        "a_front": '(pad "1" smd rect (at 0 0) (size 0.6 0.6) (layers "F.Cu") (net 1 "N1")'
                   ' (uuid "61000000-0000-4000-8000-000000000001"))',
        "a_back": '(pad "2" smd rect (at 0 0) (size 0.6 0.6) (layers "B.Cu") (net 1 "N1")'
                  ' (uuid "62000000-0000-4000-8000-000000000002"))',
        "c": '(pad "3" smd rect (at 5 0) (size 0.6 0.6) (layers "F.Cu") (net 1 "N1")'
             ' (uuid "63000000-0000-4000-8000-000000000003"))',
        "d": '(pad "4" smd rect (at 5 4) (size 0.6 0.6) (layers "F.Cu") (net 1 "N1")'
             ' (uuid "64000000-0000-4000-8000-000000000004"))',
    }

    # Same pad, other copper layers: with four layers, three clusters of N1 share (0, 0).
    PADS_4L = {
        **PADS,
        "a_in1": '(pad "5" smd rect (at 0 0) (size 0.6 0.6) (layers "In1.Cu") (net 1 "N1")'
                 ' (uuid "65000000-0000-4000-8000-000000000005"))',
    }

    def _ratsnest(self, tmp_path: Path, order: list[str], four_layer: bool = False) -> list[tuple]:
        from tests.helpers.pro_sidecar import write_default_pro

        fixture = (PROJECT_ROOT / "tests" / "fixtures" / "layer_sentinels_board.kicad_pcb").read_text()
        header = fixture[:fixture.index("  (footprint")]
        if four_layer:
            header = header.replace('(31 "B.Cu" signal)',
                                    '(1 "In1.Cu" signal)\n    (2 "In2.Cu" signal)\n    (31 "B.Cu" signal)')
        pad_defs = self.PADS_4L if four_layer else self.PADS
        pads = "\n".join(f"    {pad_defs[k]}" for k in order)
        pcb = tmp_path / f"stack_{'_'.join(order)}.kicad_pcb"
        pcb.write_text(header + '  (footprint "TEST:STACK" (layer "F.Cu")\n'
                       '    (uuid "60000000-0000-4000-8000-000000000000") (at 100 100)\n'
                       f"{pads}\n  )\n)\n")
        write_default_pro(pcb)
        eng = open_engine(board_path=str(pcb))
        try:
            eng.build_connectivity()
            return sorted((e.net_code, e.x1_mm, e.y1_mm, e.layer1, e.x2_mm, e.y2_mm, e.layer2)
                          for e in eng.get_ratsnest())
        finally:
            eng.close()

    def test_pad_order_does_not_change_ratsnest(self, tmp_path: Path) -> None:
        front_first = self._ratsnest(tmp_path, ["a_front", "a_back", "c", "d"])
        back_first = self._ratsnest(tmp_path, ["a_back", "a_front", "c", "d"])
        assert len(front_first) == 3        # c-d, c-stack, front-back (zero length)
        assert front_first == back_first

    def test_three_stacked_clusters_order_does_not_change_ratsnest(self, tmp_path: Path) -> None:
        # Two coincident clusters leave one zero-length edge, whose endpoint order the
        # binding canonicalises — so a chain ordered by CN_CLUSTER* address passes the
        # test above. With three, the chain order picks WHICH two pairs get edges
        # (F-In1 + In1-B vs F-B + B-In1) and which layer the edge to "c" lands on.
        orders = [list(p) + ["c", "d"] for p in itertools.permutations(["a_front", "a_in1", "a_back"])]
        orders.append(["d", "c", "a_back", "a_in1", "a_front"])
        results = [self._ratsnest(tmp_path, order, four_layer=True) for order in orders]
        stack = [e for e in results[0] if (e[1], e[2]) == (e[4], e[5]) == (100.0, 100.0)]
        assert len(results[0]) == 4         # c-d, c-stack, two zero-length edges in the stack
        assert len({e[3] for e in stack} | {e[6] for e in stack}) == 3   # F.Cu, In1.Cu, B.Cu
        for order, result in zip(orders[1:], results[1:]):
            assert result == results[0], order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
