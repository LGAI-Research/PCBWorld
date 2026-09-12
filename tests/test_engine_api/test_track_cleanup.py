"""Track cleaner API tests (RLRouter.cleanup_tracks / KiCadEngine.cleanup_tracks).

The fixture board (tests/fixtures/cleanup_board.kicad_pcb) is hand-built so every
pass has exactly one target — routing itself almost never produces cleanable
geometry, because the PNS optimizer already merges collinear segments on commit:

  NET1  P1(0,0) → P2(0,6): three collinear segments a001/a002/a003, plus an exact
        duplicate of the first (a004) and a zero-length track on the end pad
        (a005). A merge run drops the duplicate and the zero-length track first
        (they would otherwise read as nodes and block the merge), then folds the
        remaining collinear run into the lowest-UUID survivor.
  NET2  a stub b001 hanging off P3: connected at (5,0), free at (5,2) — the
        dangling pass's target, and nothing else's.
  NET3  F.Cu → via → B.Cu, with a second via (c004) superimposed on c003.

Contracts under test: dry runs never mutate, an open session is rejected rather
than silently cancelled, merge preserves topology, cleanup is idempotent and
undoable through checkpoint/restore, and the result is byte-identical across
processes (the pointer-ordering hardening in tracks_cleaner_rl.cpp).
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "cleanup_board.kicad_pcb"

# Segment / via UUID suffixes, as laid out in the fixture.
A1, A2, A3, A4, A5 = "a001", "a002", "a003", "a004", "a005"
B1 = "b001"
C1, C2, C3, C4 = "c001", "c002", "c003", "c004"


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


@pytest.fixture
def cleanup_board_path() -> str:
    if not BOARD_PATH.exists():
        pytest.skip(f"Test board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


@pytest.fixture
def router(cleanup_board_path: str):
    krl = _import_krl()
    r = krl.RLRouter(cleanup_board_path)
    r.build_connectivity()
    yield r
    del r


def _digest(r) -> list:
    """Order-independent copper fingerprint: geometry keyed by UUID."""
    tracks = [
        (t.uuid, round(t.x1_mm, 6), round(t.y1_mm, 6), round(t.x2_mm, 6),
         round(t.y2_mm, 6), t.layer, t.net_code)
        for t in r.get_tracks()
    ]
    vias = [(v.uuid, round(v.x_mm, 6), round(v.y_mm, 6), v.net_code) for v in r.get_vias()]
    return sorted(tracks) + sorted(vias)


def _tags(uuids) -> set:
    """Last four hex digits — the fixture's readable item tags."""
    return {u[-4:] for u in uuids}


class TestDryRun:
    def test_dry_run_reports_without_touching_the_board(self, router) -> None:
        before = _digest(router)

        result = router.cleanup_tracks(dry_run=True, merge_segments=True, clean_vias=True)

        assert result.ran
        assert _digest(router) == before
        assert result.removed == []
        assert result.modified == []
        assert {i.code_name for i in result.items} == {
            "duplicate_track", "zero_length_track", "redundant_via", "merge_tracks",
        }

    def test_dry_run_is_repeatable(self, router) -> None:
        first = router.cleanup_tracks(dry_run=True, merge_segments=True, clean_vias=True)
        second = router.cleanup_tracks(dry_run=True, merge_segments=True, clean_vias=True)

        assert [(i.code_name, i.item_a) for i in first.items] \
            == [(i.code_name, i.item_a) for i in second.items]

    def test_no_passes_selected_is_a_no_op(self, router) -> None:
        before = _digest(router)

        result = router.cleanup_tracks(dry_run=False)

        # Duplicate removal always runs (stock behaviour), nothing else does.
        assert result.ran
        assert _tags(result.removed) == {A1}
        assert len(_digest(router)) == len(before) - 1


class TestQuiescentOnly:
    def test_rejected_while_routing(self, router) -> None:
        krl = _import_krl()
        router.set_routing_mode(krl.MODE_WALKAROUND)
        assert router.start_route(5.0, 6.0, 0)
        before = _digest(router)

        result = router.cleanup_tracks(dry_run=False, merge_segments=True)

        assert not result.ran
        assert result.reject_reason == "routing_session_active"
        assert result.items == []
        assert _digest(router) == before
        router.cancel_route()

    def test_runs_once_the_session_is_closed(self, router) -> None:
        krl = _import_krl()
        router.set_routing_mode(krl.MODE_WALKAROUND)
        router.start_route(5.0, 6.0, 0)
        router.cancel_route()

        assert router.cleanup_tracks(dry_run=True, merge_segments=True).ran


class TestMergePreservesTopology:
    def test_merge_folds_the_collinear_run(self, router) -> None:
        unrouted_before = router.get_unrouted_count()

        result = router.cleanup_tracks(dry_run=False, merge_segments=True)

        assert result.ran
        # a001 goes as the duplicate, a005 as the zero-length track, a003/a004 into
        # the merge survivor a002 — which now spans the whole P1→P2 run.
        assert _tags(result.removed) == {A1, A3, A4, A5}
        assert _tags(result.modified) == {A2}

        net1 = [t for t in router.get_tracks() if t.net_code == 1]
        assert len(net1) == 1
        survivor = net1[0]
        assert survivor.uuid[-4:] == A2
        assert (survivor.x1_mm, survivor.y1_mm) == (0.0, 0.0)
        assert (survivor.x2_mm, survivor.y2_mm) == (0.0, 6.0)

        # Topology-preserving: no net gained or lost a connection.
        assert router.get_unrouted_count() == unrouted_before

    def test_merge_leaves_other_nets_alone(self, router) -> None:
        router.cleanup_tracks(dry_run=False, merge_segments=True)

        assert _tags(t.uuid for t in router.get_tracks() if t.net_code == 2) == {B1}
        assert _tags(v.uuid for v in router.get_vias()) == {C3, C4}

    def test_idempotent(self, router) -> None:
        router.cleanup_tracks(dry_run=False, merge_segments=True, clean_vias=True)
        after_first = _digest(router)

        second = router.cleanup_tracks(dry_run=False, merge_segments=True, clean_vias=True)

        assert second.items == []
        assert second.removed == [] and second.modified == []
        assert _digest(router) == after_first


class TestPassSelection:
    def test_clean_vias_drops_the_superimposed_via(self, router) -> None:
        result = router.cleanup_tracks(dry_run=False, clean_vias=True)

        assert _tags(result.removed) >= {C3}
        assert _tags(v.uuid for v in router.get_vias()) == {C4}

    def test_dangling_pass_removes_the_stub(self, router) -> None:
        result = router.cleanup_tracks(dry_run=False, dangling_tracks=True)

        # b001 is the stub; a005 (zero length, on the pad) is dangling too; a001
        # goes to the duplicate pass, which always runs. The collinear run itself
        # is untouched — no merge was requested.
        assert _tags(result.removed) == {A1, A5, B1}
        assert {t.uuid[-4:] for t in router.get_tracks()} == {A2, A3, A4, C1, C2}

    def test_net_filter_scopes_every_pass(self, router) -> None:
        # NET2-scoped: the NET1 duplicate and zero-length track must survive even
        # though their passes would otherwise fire.
        result = router.cleanup_tracks(dry_run=False, dangling_tracks=True, net_codes=[2])

        assert result.ran
        assert _tags(result.removed) == {B1}
        assert {t.uuid[-4:] for t in router.get_tracks()} == {A1, A2, A3, A4, A5, C1, C2}

    def test_net_filter_excludes_out_of_scope_nets(self, router) -> None:
        result = router.cleanup_tracks(dry_run=False, dangling_tracks=True, net_codes=[1])

        assert result.ran
        assert B1 in {t.uuid[-4:] for t in router.get_tracks()}
        assert B1 not in _tags(result.removed)

    def test_merge_pass_does_not_touch_dangling_items(self, router) -> None:
        router.cleanup_tracks(dry_run=False, merge_segments=True)

        assert B1 in {t.uuid[-4:] for t in router.get_tracks()}


class TestUndoThroughCheckpoint:
    def test_restore_reverts_a_cleanup(self, router) -> None:
        before = _digest(router)
        handle = router.checkpoint()

        router.cleanup_tracks(dry_run=False, merge_segments=True, clean_vias=True)
        assert _digest(router) != before

        assert router.restore_incremental(handle)
        router.build_connectivity()
        assert _digest(router) == before

    def test_restore_full_path_reverts_a_cleanup(self, router) -> None:
        before = _digest(router)
        handle = router.checkpoint()

        router.cleanup_tracks(dry_run=False, merge_segments=True, clean_vias=True)

        assert router.restore(handle)
        router.build_connectivity()
        assert _digest(router) == before


_DETERMINISM_PROBE = """
import sys
sys.path.insert(0, {rl_dir!r})
import kicad_rl_router as krl

r = krl.RLRouter({board!r})
r.build_connectivity()
res = r.cleanup_tracks(dry_run=False, merge_segments=True, clean_vias=True)
rows = sorted(
    (t.uuid, t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm, t.layer) for t in r.get_tracks()
)
print([i.code_name for i in res.items])
print(sorted(res.removed))
print(rows)
del r
"""


class TestCrossProcessDeterminism:
    """The stock cleaner picks the merge survivor by heap address; the fork keys
    every ordering decision on KIID. Two fresh processes must agree exactly."""

    def test_same_result_in_a_fresh_process(self, cleanup_board_path: str) -> None:
        _import_krl()
        probe = _DETERMINISM_PROBE.format(
            rl_dir=str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"),
            board=cleanup_board_path,
        )
        runs = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300,
            )
            assert proc.returncode == 0, proc.stderr
            runs.append(proc.stdout)

        assert runs[0] == runs[1]
        assert "merge_tracks" in runs[0]


class TestEngineWrapper:
    def test_engine_cleanup_tracks(self, cleanup_board_path: str) -> None:
        pytest.importorskip("pcb_world.engine")
        from pcb_world.engine import (
            CLEANUP_TOPOLOGY_PRESERVING,
            KiCadEngine,
        )

        engine = KiCadEngine(cleanup_board_path)
        try:
            engine.build_connectivity()

            dry = engine.cleanup_tracks(**CLEANUP_TOPOLOGY_PRESERVING)
            assert dry.ran and not dry.changed
            assert dry.counts()["merge_tracks"] >= 1

            live = engine.cleanup_tracks(dry_run=False, **CLEANUP_TOPOLOGY_PRESERVING)
            assert live.ran and live.changed
            assert _tags(live.modified) == {A2}
        finally:
            engine.close()
