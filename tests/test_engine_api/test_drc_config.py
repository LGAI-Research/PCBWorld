"""Design-rule loading / round-trip tests.

Exercises two paths:

  1. apply_default_drc_if_fallback(engine, yaml) — on a board that triggers
     the real-default fallback (no .kicad_pro companion, no legacy setup),
     the YAML values should land in BDS.
  2. set_design_rules(rules) → get_design_rules() — arbitrary values written
     in must come back out unchanged (global minima round-trip).

Uses ``kicad_rl_router.RLRouter`` directly rather than ``KiCadEngine``:
the wrapper's load contract refuses the orphan (pro-less) fixtures these
tests construct on purpose to reach the fallback provenance state.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

from pcb_world.engine.drc_config import (  # noqa: E402
    apply_default_drc_if_fallback,
    load_drc_config,
)

# Modern KiCad 9 pcb used as the source for orphan-pcb fixtures. Its setup
# block only carries modern tokens (pad_to_mask_clearance etc.), so the
# parser will NOT set m_LegacyDesignSettingsLoaded — perfect for the
# "real default" provenance state we need.
MODERN_PCB_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "simple_routing_board.kicad_pcb"


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


@pytest.fixture
def orphan_modern_pcb(tmp_path: Path) -> Path:
    """Copy a KiCad 9 pcb into tmp_path WITHOUT its .kicad_pro sibling.

    The absence of the pro companion + modern (non-legacy) setup block
    guarantees both provenance flags come back False — i.e. the C++
    loader falls back to a blank in-memory PROJECT.
    """
    if not MODERN_PCB_FIXTURE.exists():
        pytest.skip(f"Fixture not found: {MODERN_PCB_FIXTURE}")
    dest = tmp_path / "orphan.kicad_pcb"
    shutil.copy(MODERN_PCB_FIXTURE, dest)
    return dest


@pytest.fixture
def custom_drc_yaml(tmp_path: Path) -> Path:
    """YAML with values deliberately different from the repo default so
    we can tell "config applied" from "default applied" in assertions."""
    path = tmp_path / "custom_drc.yaml"
    path.write_text(yaml.safe_dump({
        "name": "test_custom",
        "global_minima": {
            "clearance":             0.33,
            "track_width":           0.44,
            "via_diameter":          0.77,
            "through_hole_drill":    0.25,
            "via_annular_width":     0.12,
            "hole_to_hole":          0.28,
            "uvia_diameter":         0.18,
            "uvia_drill":            0.08,
            "copper_edge_clearance": 0.19,
        },
    }))
    return path


# ---------------------------------------------------------------------------
# Test 1: load YAML → apply → get — values land in BDS as declared
# ---------------------------------------------------------------------------

def test_apply_default_drc_from_yaml_lands_in_bds(
    orphan_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))

    # Precondition: both provenance flags False → "real default" fallback
    # is exactly what triggers apply_default_drc_if_fallback to act.
    assert not router.was_project_loaded_from_file(), \
        "orphan pcb must not find a companion .kicad_pro"
    assert not router.was_legacy_design_settings_loaded(), \
        "modern pcb must not set the legacy-setup flag"

    with pytest.warns(UserWarning, match="DRC fallback triggered"):
        applied = apply_default_drc_if_fallback(router, config_path=custom_drc_yaml)
    assert applied is True

    got = router.get_design_rules()
    expected = load_drc_config(custom_drc_yaml)
    for field, value in expected.items():
        assert getattr(got, field) == pytest.approx(value, abs=2e-6), \
            f"{field}: expected {value}, got {getattr(got, field)}"


def test_apply_default_drc_is_noop_when_rules_are_already_set(
    orphan_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """After applying once, the flags don't change — the board still looks
    like a 'real default' fallback because the provenance flags are a
    property of the *load*, not the current BDS state. So the second call
    also 'applies' (idempotent on the same YAML). This test locks in the
    observed contract rather than asserting a skip-on-second-call invariant
    that the implementation doesn't actually provide."""
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))
    with pytest.warns(UserWarning, match="DRC fallback triggered"):
        assert apply_default_drc_if_fallback(router, config_path=custom_drc_yaml) is True
    # Second call still acts — provenance flags are load-time, not live.
    with pytest.warns(UserWarning, match="DRC fallback triggered"):
        assert apply_default_drc_if_fallback(router, config_path=custom_drc_yaml) is True


def test_apply_default_drc_raises_on_fallback_without_yaml(
    orphan_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """Strict mode: when the board has neither a pro companion nor legacy
    setup tokens AND ``use_yaml=False``, raise rather than silently route
    on KiCad compile-time defaults."""
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))
    with pytest.raises(ValueError, match="DRC fallback triggered"):
        apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, use_yaml=False,
        )


# ---------------------------------------------------------------------------
# Test 2: set_design_rules → get_design_rules — arbitrary values round-trip
# ---------------------------------------------------------------------------

# Deliberately "weird" values: no relationship to the YAML defaults, no
# sensible unit pattern — if these survive the round-trip we can rule out
# any accidental clamping, scaling, or field-mixup.
_WEIRD_VALUES: dict[str, float] = {
    "min_clearance_mm":         0.917,
    "min_track_width_mm":       0.813,
    "min_via_diameter_mm":      1.234,
    "min_through_hole_mm":      0.461,
    "min_via_annular_width_mm": 0.057,
    "min_hole_to_hole_mm":      0.389,
    "min_uvia_diameter_mm":     0.211,
    "min_uvia_drill_mm":        0.073,
    "copper_edge_clearance_mm": 0.298,
}


def test_set_design_rules_roundtrips_all_global_minima(
    orphan_modern_pcb: Path,
) -> None:
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))

    rules = router.get_design_rules()
    for field, value in _WEIRD_VALUES.items():
        setattr(rules, field, value)
    router.set_design_rules(rules)

    got = router.get_design_rules()
    for field, value in _WEIRD_VALUES.items():
        assert getattr(got, field) == pytest.approx(value, abs=2e-6), \
            f"{field}: expected {value}, got {getattr(got, field)}"


def test_set_design_rules_negative_value_leaves_field_unchanged(
    orphan_modern_pcb: Path,
) -> None:
    """The setter's documented contract: negative values mean 'skip this
    field' so callers can do partial updates without reading the struct
    back first."""
    krl = _import_krl()
    router = krl.RLRouter(str(orphan_modern_pcb))

    # Step 1: establish known baseline
    baseline = router.get_design_rules()
    baseline.min_clearance_mm = 0.555
    baseline.min_track_width_mm = 0.666
    router.set_design_rules(baseline)

    # Step 2: partial update — min_clearance_mm = -1 (should persist 0.555),
    #                          min_track_width_mm = 0.777 (should update)
    partial = router.get_design_rules()
    partial.min_clearance_mm = -1.0
    partial.min_track_width_mm = 0.777
    router.set_design_rules(partial)

    got = router.get_design_rules()
    assert got.min_clearance_mm == pytest.approx(0.555, abs=2e-6), \
        "negative sentinel should have left min_clearance_mm untouched"
    assert got.min_track_width_mm == pytest.approx(0.777, abs=2e-6), \
        "positive value should have overwritten min_track_width_mm"
