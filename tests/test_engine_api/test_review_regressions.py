"""Regression pins for a set of engine contracts around rounding, checkpointing,
mid-session updates, deferred deletion, incremental DRC, duplicate segments, and
kiid/world-stats consistency.

Covers:

- mm→nm conversion rounds (pcbIUScale delegation) instead of truncating —
  a double sitting one ULP below a clean value must land on the intended nm.
- checkpoint()/restore() carries the runtime-mutable shove iteration limit.
- Size setters propagate into an ACTIVE routing session (ROUTER::UpdateSizes).
- delete_track_* defers the C++ free until connectivity is purged — querying
  the ratsnest right after a delete must be safe.
- set_design_rules() invalidates the incremental-DRC baseline: violations
  computed under OLD rules must not be retained across a rule change.
- Exact-duplicate segments (same geometry+net, distinct UUIDs — retrace
  artifacts observed on real boards) survive SyncWorld and can be routed
  around. Routing near them drives the comparator invariant self-check
  compiled into compareObstacleItems (wx-assert: distinct items must never
  compare equal), which is the actual collapse detector.
- kiid_get/set_generator_state round-trips and reflects stream advancement.
- world_stats() is consistent across resync_world(), which also closes an
  open routing session (dangling-placer guard).
"""

import math
import re
import sys
import uuid as uuid_mod
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


def _route_and_commit(krl, r) -> None:
    """Commit the fixture's canonical NET1 route (walkaround past the obstacle)."""
    r.set_routing_mode(krl.MODE_WALKAROUND)
    assert r.start_route(*START, 0), "start_route failed"
    assert r.fix_route(*END, True, False), "fix_route failed"


class TestMmNmRounding:
    """nmFromMm delegates to pcbIUScale.mmToIU (rounds, does not truncate)."""

    def test_value_below_clean_target_rounds_up(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        rules = r.get_design_rules()

        below = math.nextafter(0.2, 0.0)      # one ULP below 0.2 mm
        rules.min_clearance_mm = below
        r.set_design_rules(rules)
        assert r.get_design_rules().min_clearance_mm == pytest.approx(0.2, abs=1e-9), (
            "truncation regression: value one ULP below 0.2 must round to 200000 nm"
        )

        above = math.nextafter(0.2, 1.0)      # one ULP above 0.2 mm
        rules.min_clearance_mm = above
        r.set_design_rules(rules)
        assert r.get_design_rules().min_clearance_mm == pytest.approx(0.2, abs=1e-9)


class TestCheckpointShoveLimit:
    """checkpoint carries the runtime-mutable shove iteration limit."""

    def test_restore_recovers_limit(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        assert r.get_shove_iter_limit() == 250

        handle = r.checkpoint()
        r.set_shove_iter_limit(17)
        assert r.get_shove_iter_limit() == 17

        assert r.restore(handle)
        assert r.get_shove_iter_limit() == 250, (
            "checkpoint must snapshot shove_iter_limit (runtime-mutable via set_shove_iter_limit)"
        )
        r.release_checkpoint(handle)


class TestMidSessionSizes:
    """Size setters reach the ACTIVE placer (ROUTER::UpdateSizes)."""

    def test_track_width_set_after_start_route_applies(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        r.build_connectivity()
        r.set_routing_mode(krl.MODE_WALKAROUND)
        assert r.start_route(*START, 0)
        r.set_track_width(0.6)                # mid-session change
        assert r.fix_route(*END, True, False)

        widths = {round(t.width_mm, 6) for t in r.get_tracks()}
        assert widths == {0.6}, (
            f"mid-session set_track_width must apply to this session, got {widths}"
        )


class TestDeleteThenRatsnest:
    """Deferred free: ratsnest queries right after delete are safe."""

    def test_delete_track_then_query_ratsnest(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        r.build_connectivity()
        _route_and_commit(krl, r)
        assert r.get_track_count() > 0

        t = r.get_tracks()[0]
        assert r.delete_track_near(
            t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm, t.layer, t.net_code, 0.01)

        # BOARD::Remove only marks the CN_ITEM invalid; without deferring the
        # free, the C++ delete would leave it dangling until the next full
        # build_connectivity(). These queries walk connectivity WITHOUT an
        # intervening build_connectivity().
        edges = r.get_ratsnest()
        assert r.get_unrouted_count() >= 1
        assert all(e.net_code >= 0 for e in edges)


class TestIncrementalDrcRuleChange:
    """A rule change must invalidate the incremental-DRC baseline."""

    def test_rule_change_reflected_without_track_change(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        r.build_connectivity()
        _route_and_commit(krl, r)

        baseline = r.run_drc_incremental("")          # baseline under load-time rules
        n_before = len(baseline)

        rules = r.get_design_rules()
        rules.min_clearance_mm = 1.5                  # far above the walkaround gap
        r.set_design_rules(rules)

        after = r.run_drc_incremental("")             # NO track changed since baseline
        n_after = len(after)
        assert n_after > n_before, (
            "rule change with unchanged tracks must produce new violations "
            f"(before={n_before}, after={n_after}) — incremental baseline was "
            "not invalidated by set_design_rules()"
        )


def _duplicate_first_segment(board_text: str) -> str:
    """Append an exact copy (fresh UUID) of the first (segment ...) block."""
    m = re.search(r"\(segment\b", board_text)
    assert m, "no segment in saved board"
    i, depth = m.start(), 0
    for j in range(i, len(board_text)):
        if board_text[j] == "(":
            depth += 1
        elif board_text[j] == ")":
            depth -= 1
            if depth == 0:
                break
    block = board_text[i : j + 1]
    dup = re.sub(r'\(uuid "[^"]+"\)', f'(uuid "{uuid_mod.uuid4()}")', block)
    assert dup != block
    return board_text[: j + 1] + "\n  " + dup + board_text[j + 1 :]


class TestExactDuplicateSegments:
    """Exact-duplicate copper (retrace artifact) must survive sync and routing.

    Routing next to the duplicated pair exercises the obstacle/cluster sets on
    items whose (UUID-distinct, geometry-identical) key reaches the Serial
    tie-break — with the compareObstacleItems invariant assert armed, any
    set-collapse regression surfaces as a wx-assert on stderr and, for the
    world, as a lost segment below.
    """

    def test_duplicates_survive_sync_and_route(self, board_path: str, tmp_path) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        r.build_connectivity()
        _route_and_commit(krl, r)
        saved = tmp_path / "dup_board.kicad_pcb"
        r.save(str(saved))

        text = saved.read_text()
        n_orig = len(re.findall(r"\(segment\b", text))
        saved.write_text(_duplicate_first_segment(text))

        r2 = krl.RLRouter(str(saved), "", 77, 250, 1000000)
        r2.build_connectivity()
        assert r2.get_track_count() == n_orig + 1, "duplicate lost at board load"

        stats = r2.world_stats()
        assert stats["segments"] == n_orig + 1, (
            "SyncWorld dropped the exact-duplicate segment "
            f"(world={stats['segments']}, board={n_orig + 1})"
        )

        # Route the same net again next to the duplicated copper: query sets now
        # contain the UUID-distinct twins (per-item collision checks hit both).
        r2.set_routing_mode(krl.MODE_WALKAROUND)
        assert r2.start_route(*START, 0)
        r2.fix_route(*END, True, False)   # success not required; must not collapse/crash
        assert r2.get_track_count() >= n_orig + 1


class TestKiidStateAndWorldStats:
    """kiid stream round-trip + world_stats/resync coherence."""

    def test_kiid_state_roundtrip_and_advance(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        r.build_connectivity()

        s0 = krl.kiid_get_generator_state()
        assert isinstance(s0, bytes) and len(s0) > 0
        krl.kiid_set_generator_state(s0)
        assert krl.kiid_get_generator_state() == s0

        _route_and_commit(krl, r)              # commits mint UUIDs
        assert krl.kiid_get_generator_state() != s0, "stream did not advance"

    def test_world_stats_stable_across_resync(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path, "", 77, 250, 1000000)
        r.build_connectivity()
        _route_and_commit(krl, r)

        before = r.world_stats()
        assert before["segments"] == r.get_track_count()

        assert r.start_route(*START, 0)        # open a session, then resync
        r.resync_world()                       # must cancel the session first
        assert not r.is_routing(), "resync_world must close the open session"
        assert r.world_stats() == before, "resync changed world content"
