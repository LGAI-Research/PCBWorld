"""Verify the ``use_yaml`` flag on apply_default_drc_if_fallback.

The flag toggles what happens when the board triggers the "real
default" fallback (both ``was_project_loaded_from_file`` and
``was_legacy_design_settings_loaded`` return False):

  - ``use_yaml=True``: push the YAML's global minima into BDS so DRC
    runs against a meaningful floor (with a ``UserWarning`` because
    every dataset we ship should already carry rules).
  - ``use_yaml=False``: raise ``ValueError``. Strict mode — refuses to
    silently route on KiCad compile-time defaults.

Regardless of the flag, boards that already carry authoritative rules
(a pro companion on disk OR legacy KiCad 5 setup tokens) must be left
untouched (no warning, no error).

Tests use ``kicad_rl_router.RLRouter`` directly: the ``KiCadEngine``
load contract refuses the orphan/legacy fixtures these tests construct
on purpose to reach each provenance state.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

from pcb_world.engine.drc_config import apply_default_drc_if_fallback  # noqa: E402


# Modern KiCad 9 pcb (no legacy setup tokens). Used as the source for
# orphan-pcb fixtures to guarantee the "real default" provenance state.
MODERN_PCB_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "simple_routing_board.kicad_pcb"
MODERN_PRO_FIXTURE = MODERN_PCB_FIXTURE.with_suffix(".kicad_pro")

# Legacy KiCad 5 pcb with (setup ...) + (net_class ...) tokens — triggers
# the ``was_legacy_design_settings_loaded==True`` path at load.
LEGACY_PCB_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "crossover_legacy.kicad_pcb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


# Mirrors the dict form RLRouter.DesignRules exposes so snapshots can
# be compared field-by-field without pybind objects in the mix.
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


def _global_minima_snapshot(rules) -> dict[str, float]:
    return {f: float(getattr(rules, f)) for f in _GLOBAL_MINIMA_FIELDS}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orphan_modern_pcb(tmp_path: Path) -> Path:
    """Copy a KiCad 9 pcb into tmp_path WITHOUT its .kicad_pro sibling.

    Absent pro + modern (non-legacy) setup → both provenance flags come
    back False. This is the only state where ``use_yaml`` can act.
    """
    if not MODERN_PCB_FIXTURE.exists():
        pytest.skip(f"Fixture not found: {MODERN_PCB_FIXTURE}")
    dest = tmp_path / "orphan.kicad_pcb"
    shutil.copy(MODERN_PCB_FIXTURE, dest)
    return dest


@pytest.fixture
def paired_modern_pcb(tmp_path: Path) -> Path:
    """Copy the KiCad 9 pcb AND its pro companion into tmp_path.

    The loader finds the pro → ``was_project_loaded_from_file==True`` →
    no fallback branch runs regardless of the ``use_yaml`` flag.
    """
    if not MODERN_PCB_FIXTURE.exists() or not MODERN_PRO_FIXTURE.exists():
        pytest.skip(f"Fixture pair not found: {MODERN_PCB_FIXTURE} / .kicad_pro")
    pcb = tmp_path / MODERN_PCB_FIXTURE.name
    pro = tmp_path / MODERN_PRO_FIXTURE.name
    shutil.copy(MODERN_PCB_FIXTURE, pcb)
    shutil.copy(MODERN_PRO_FIXTURE, pro)
    return pcb


@pytest.fixture
def legacy_pcb(tmp_path: Path) -> Path:
    """Copy the KiCad 5 crossover fixture into tmp_path alone.

    Legacy setup tokens → ``was_legacy_design_settings_loaded==True`` →
    no fallback branch runs regardless of the ``use_yaml`` flag.
    """
    if not LEGACY_PCB_FIXTURE.exists():
        pytest.skip(f"Fixture not found: {LEGACY_PCB_FIXTURE}")
    dest = tmp_path / LEGACY_PCB_FIXTURE.name
    shutil.copy(LEGACY_PCB_FIXTURE, dest)
    return dest


@pytest.fixture
def custom_drc_yaml(tmp_path: Path) -> Path:
    """YAML values deliberately different from repo defaults.

    Makes it unambiguous in assertions whether "YAML was applied" vs
    "BDS left at engine defaults" — the two paths produce measurably
    different global minima.
    """
    path = tmp_path / "custom_drc.yaml"
    path.write_text(yaml.safe_dump({
        "name": "test_custom_use_yaml",
        "global_minima": {
            "clearance":             0.37,
            "track_width":           0.41,
            "via_diameter":          0.71,
            "through_hole_drill":    0.27,
            "via_annular_width":     0.13,
            "hole_to_hole":          0.29,
            "uvia_diameter":         0.19,
            "uvia_drill":            0.09,
            "copper_edge_clearance": 0.17,
        },
    }))
    return path


# ---------------------------------------------------------------------------
# use_yaml on the fallback path
# ---------------------------------------------------------------------------

def test_use_yaml_true_mutates_bds_on_fallback(
    orphan_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """Opt-in: YAML minima land in BDS when the board is a real-default
    fallback. Emits a UserWarning so the trace is visible."""
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))
    assert not router.was_project_loaded_from_file()
    assert not router.was_legacy_design_settings_loaded()

    with pytest.warns(UserWarning, match="DRC fallback triggered"):
        applied = apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, use_yaml=True,
        )
    assert applied is True

    got = router.get_design_rules()
    # Spot-check a couple of fields — the full mapping is exercised in
    # test_drc_config.py; this test only needs to confirm the flag
    # actually wired the YAML through.
    assert got.min_clearance_mm == pytest.approx(0.37, abs=2e-6)
    assert got.min_track_width_mm == pytest.approx(0.41, abs=2e-6)


def test_use_yaml_false_raises_on_fallback(
    orphan_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """Strict mode: ``use_yaml=False`` on a real-default fallback raises
    ``ValueError`` and does not mutate BDS."""
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))
    assert not router.was_project_loaded_from_file()
    assert not router.was_legacy_design_settings_loaded()

    before = _global_minima_snapshot(router.get_design_rules())

    with pytest.raises(ValueError, match="DRC fallback triggered"):
        apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, use_yaml=False,
        )

    after = _global_minima_snapshot(router.get_design_rules())
    assert after == before, (
        "raise path must leave BDS exactly as the loader left it; "
        f"diff: {[k for k in after if after[k] != before[k]]}"
    )


def test_use_yaml_true_without_config_path_raises(
    orphan_modern_pcb: Path
) -> None:
    """``use_yaml=True`` but ``config_path=None`` raises: there is no
    implicit default YAML, so the fallback must name its substitute
    explicitly. BDS is left untouched."""
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))
    assert not router.was_project_loaded_from_file()
    assert not router.was_legacy_design_settings_loaded()

    before = _global_minima_snapshot(router.get_design_rules())

    with pytest.raises(ValueError, match="no implicit default YAML"):
        apply_default_drc_if_fallback(router, config_path=None, use_yaml=True)

    after = _global_minima_snapshot(router.get_design_rules())
    assert after == before, (
        "missing-path raise must leave BDS exactly as the loader left it; "
        f"diff: {[k for k in after if after[k] != before[k]]}"
    )


# ---------------------------------------------------------------------------
# Non-fallback boards: flag is irrelevant
# ---------------------------------------------------------------------------

def test_use_yaml_irrelevant_when_project_loaded(
    paired_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """When the pro companion is on disk, ``was_project_loaded_from_file``
    is True → the fallback branch never runs, so both flag values must
    behave identically (no-op) and must not clobber the pro-sourced BDS.
    """
    krl = _import_krl()
    router = krl.RLRouter(str(paired_modern_pcb))
    assert router.was_project_loaded_from_file(), (
        "sibling .kicad_pro must be picked up by the loader"
    )

    baseline = _global_minima_snapshot(router.get_design_rules())

    for flag in (True, False):
        assert apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, use_yaml=flag,
        ) is False, f"expected no-op on loaded-from-file board (use_yaml={flag})"
        after = _global_minima_snapshot(router.get_design_rules())
        assert after == baseline, (
            f"BDS must not change for use_yaml={flag}; "
            f"diff: {[k for k in after if after[k] != baseline[k]]}"
        )


def test_use_yaml_irrelevant_when_legacy_loaded(
    legacy_pcb: Path, custom_drc_yaml: Path
) -> None:
    """Legacy KiCad 5 setup tokens populate BDS via the parser → the
    fallback branch must not run regardless of ``use_yaml``."""
    krl = _import_krl()
    router = krl.RLRouter(str(legacy_pcb))
    assert router.was_legacy_design_settings_loaded(), (
        "legacy crossover fixture must set the legacy flag at parse time"
    )

    baseline = _global_minima_snapshot(router.get_design_rules())

    for flag in (True, False):
        assert apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, use_yaml=flag,
        ) is False, f"expected no-op on legacy-loaded board (use_yaml={flag})"
        after = _global_minima_snapshot(router.get_design_rules())
        assert after == baseline, (
            f"BDS must not change for use_yaml={flag}; "
            f"diff: {[k for k in after if after[k] != baseline[k]]}"
        )
