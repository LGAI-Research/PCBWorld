"""Version-upgrade round-trip test for a legacy KiCad 5 board.

Loads ``tests/fixtures/crossover_legacy.kicad_pcb`` (KiCad 5 format, no
companion ``.kicad_pro``), captures design rules, saves to a tmp_path
(which produces a modern KiCad 9 pcb + a companion .kicad_pro), then
reloads from that new location and verifies the design rules survived
intact.

This is the end-to-end check for the "legacy constraints are preserved
across format upgrade" contract, covering:

  - 9 global minima (BDS m_Min*, m_TrackMinWidth, etc.)
  - Default netclass (clearance/trace_width/via/uvia)
  - Additional netclass entries defined via ``(net_class ...)`` blocks,
    which on the legacy path go through the netclass migration branch
    inside ``BOARD::SetProject``.

The first load should trigger the legacy path
(``was_legacy_design_settings_loaded()==True``) while the reload should
pick up the freshly-written project file
(``was_project_loaded_from_file()==True``) — both asserted explicitly so
regressions show up as provenance failures rather than silent value drift.

Test inputs live under ``tests/fixtures/`` (copied there once, not
referenced in-place from ``pcb_dataset/``) so a broken test can never
mutate the training corpus.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))


# Legacy (KiCad 5) board with setup-block constraints AND net_class blocks:
# 4 nets, 2 netclasses (Default + "phat"). Small enough for a fast test.
LEGACY_PCB_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "crossover_legacy.kicad_pcb"


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

# Global minima fields we compare across the round-trip. Kept as a module-level
# constant so the same list drives both capture and assertion — drift in one
# place is impossible.
_GLOBAL_MINIMA_FIELDS = (
    "min_clearance_mm",
    "min_track_width_mm",
    "min_via_diameter_mm",
    "min_through_hole_mm",
    "min_via_annular_width_mm",
    "min_hole_to_hole_mm",
    "min_uvia_diameter_mm",
    "min_uvia_drill_mm",
    "copper_edge_clearance_mm",
)

_NETCLASS_FIELDS = (
    "clearance_mm",
    "track_width_mm",
    "via_diameter_mm",
    "via_drill_mm",
    "uvia_diameter_mm",
    "uvia_drill_mm",
)


def _snapshot_rules(rules) -> dict:
    """Flatten a DesignRules object into a plain dict for comparison.

    Netclasses are keyed by name so ordering differences between the two
    loads (e.g. std::map vs JSON object iteration) don't cause spurious
    failures.
    """
    snap: dict = {}
    for field in _GLOBAL_MINIMA_FIELDS:
        snap[field] = getattr(rules, field)

    snap["default_netclass_name"] = rules.default_netclass.name
    snap["default_netclass"] = {
        f: getattr(rules.default_netclass, f) for f in _NETCLASS_FIELDS
    }

    classes: dict[str, dict[str, float]] = {}
    for nc in rules.netclasses:
        classes[nc.name] = {f: getattr(nc, f) for f in _NETCLASS_FIELDS}
    snap["netclasses"] = classes
    return snap


def _assert_rules_equal(before: dict, after: dict, *, tol: float = 2e-6) -> None:
    """Compare two rule snapshots field-by-field with nm-level tolerance."""
    # Global minima
    for field in _GLOBAL_MINIMA_FIELDS:
        assert after[field] == pytest.approx(before[field], abs=tol), \
            f"{field}: before={before[field]}, after={after[field]}"

    # Default netclass
    assert after["default_netclass_name"] == before["default_netclass_name"]
    for f in _NETCLASS_FIELDS:
        b = before["default_netclass"][f]
        a = after["default_netclass"][f]
        assert a == pytest.approx(b, abs=tol), \
            f"default_netclass.{f}: before={b}, after={a}"

    # Non-default netclasses: same set of names, same field values.
    assert set(after["netclasses"]) == set(before["netclasses"]), \
        (f"netclass name mismatch: before={sorted(before['netclasses'])}, "
         f"after={sorted(after['netclasses'])}")
    for name, before_fields in before["netclasses"].items():
        after_fields = after["netclasses"][name]
        for f in _NETCLASS_FIELDS:
            b = before_fields[f]
            a = after_fields[f]
            assert a == pytest.approx(b, abs=tol), \
                f"netclass[{name!r}].{f}: before={b}, after={a}"


# ---------------------------------------------------------------------------
# Fixture: copy the legacy pcb into tmp_path alone
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_legacy_pcb(tmp_path: Path) -> Path:
    """Copy the legacy fixture into tmp_path with no companion .kicad_pro.

    Two-stage copy (pcb_dataset → tests/fixtures → tmp_path) ensures:
      - the training corpus in pcb_dataset/ is never touched by a test run
      - a failing test can't leave a .kicad_pro next to the fixture that
        would poison subsequent runs
      - the first-load provenance is a clean "legacy setup" path.
    """
    if not LEGACY_PCB_FIXTURE.exists():
        pytest.skip(f"Fixture not found: {LEGACY_PCB_FIXTURE}")
    dest = tmp_path / LEGACY_PCB_FIXTURE.name
    shutil.copy(LEGACY_PCB_FIXTURE, dest)
    return dest


# ---------------------------------------------------------------------------
# Test: legacy → upgrade → reload round-trip preserves design rules
# ---------------------------------------------------------------------------

def test_legacy_pcb_version_upgrade_preserves_drc_rules(
    isolated_legacy_pcb: Path, tmp_path: Path
) -> None:
    krl = _import_krl()

    # --- Step 1: load the legacy pcb, capture rules ---
    original = krl.RLRouter(str(isolated_legacy_pcb))
    assert not original.was_project_loaded_from_file(), \
        "legacy pcb has no sibling .kicad_pro, so project load must fail"
    assert original.was_legacy_design_settings_loaded(), \
        "legacy KiCad 5 setup block must trigger the legacy-load flag"
    before = _snapshot_rules(original.get_design_rules())

    # Sanity check the snapshot actually carries non-default values: if the
    # parser silently dropped the legacy tokens, this catches it before the
    # (trivially-symmetric) round-trip comparison.
    assert before["min_track_width_mm"] > 0, \
        "expected BDS::m_TrackMinWidth populated from legacy (trace_min)"
    assert len(before["netclasses"]) >= 1, \
        "crossover fixture defines at least one non-Default net_class"

    # --- Step 2: save (emits modern pcb + companion pro) ---
    out_dir = tmp_path / "upgraded"
    out_dir.mkdir()
    out_pcb = out_dir / (isolated_legacy_pcb.stem + ".kicad_pcb")
    original.save(str(out_pcb))

    assert out_pcb.exists(), "save() must produce the pcb"
    assert out_pcb.with_suffix(".kicad_pro").exists(), \
        "save() must emit a companion .kicad_pro — this is what preserves " \
        "BDS + NetSettings, since modern .kicad_pcb drops them"

    # Release the first router before constructing the second. The native
    # BOARD/VIA pointers alias into KiCad global state; overlapping routers
    # have been observed to crash (see KiCadEngine.close docstring).
    del original

    # --- Step 3: reload the saved pair, capture rules ---
    reloaded = krl.RLRouter(str(out_pcb))
    assert reloaded.was_project_loaded_from_file(), \
        "the freshly-saved .kicad_pro must be discovered and loaded"
    assert not reloaded.was_legacy_design_settings_loaded(), \
        "modern-format pcb must not trip the legacy flag"
    after = _snapshot_rules(reloaded.get_design_rules())

    # --- Compare ---
    _assert_rules_equal(before, after)
