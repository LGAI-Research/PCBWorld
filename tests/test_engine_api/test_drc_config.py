"""Design-rule round-trip and load-time checks.

  * set_design_rules(rules) → get_design_rules() — arbitrary values written in come back
    out unchanged (global minima round-trip); a 0 Default class size is refused.
  * run_drc on an unloadable rules file raises.
  * verify_netclass_rules — the load-time net class consistency check.

The YAML substitution (``apply_default_drc_if_fallback``) is covered in
test_rules_source.py. Round-trip tests use ``kicad_rl_router.RLRouter`` directly: the
``KiCadEngine`` load contract refuses the orphan (pro-less) fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_engine_api.conftest import MODERN_PCB_FIXTURE


# ---------------------------------------------------------------------------
# set_design_rules → get_design_rules — arbitrary values round-trip
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
    krl, orphan_modern_pcb: Path,
) -> None:
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
    krl, orphan_modern_pcb: Path,
) -> None:
    """The setter's documented contract: negative values mean 'skip this
    field' so callers can do partial updates without reading the struct
    back first."""
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


@pytest.mark.parametrize("field", ["track_width_mm", "clearance_mm", "via_diameter_mm",
                                   "via_drill_mm"])
def test_set_design_rules_refuses_a_zero_default_class_size(field: str) -> None:
    """A Default net class routing size of 0 raises (through the engine server) before anything
    changes — the router would otherwise route at that size; negative stays 'unchanged'."""
    from pcb_world.engine.drc_config import design_rules_digest
    from tests.helpers.engine import open_engine

    eng = open_engine(str(MODERN_PCB_FIXTURE))
    try:
        before = design_rules_digest(eng.get_design_rules())
        rules = eng.get_design_rules()
        rules.min_clearance_mm = 0.123                  # would land if the call went through
        setattr(rules.default_netclass, field, 0.0)
        with pytest.raises(RuntimeError, match="Default net class size of 0 mm"):
            eng.set_design_rules(rules)
        assert design_rules_digest(eng.get_design_rules()) == before
    finally:
        eng.close()


@pytest.mark.parametrize("field", ["uvia_diameter_mm", "uvia_drill_mm"])
def test_zero_default_class_microvia_size_round_trips(field: str) -> None:
    """The router never places a microvia: a Default class microvia size of 0 (a dataset
    board, d3 0508 sympetrum, has drill 0) round-trips set_design_rules(get_design_rules());
    a 0 track width on the same rules still raises."""
    from pcb_world.engine.drc_config import design_rules_digest
    from tests.helpers.engine import open_engine

    eng = open_engine(str(MODERN_PCB_FIXTURE))
    try:
        rules = eng.get_design_rules()
        setattr(rules.default_netclass, field, 0.0)
        eng.set_design_rules(rules)
        got = eng.get_design_rules()
        assert getattr(got.default_netclass, field) == 0.0
        eng.set_design_rules(got)                       # the round trip itself
        assert design_rules_digest(eng.get_design_rules()) == design_rules_digest(got)
        got.default_netclass.track_width_mm = 0.0
        with pytest.raises(RuntimeError, match="Default net class size of 0 mm"):
            eng.set_design_rules(got)
    finally:
        eng.close()


def test_run_drc_raises_on_unloadable_rules(tmp_path: Path) -> None:
    """A rules file the DRC engine cannot parse raises with the parser's cause instead of
    a DRC on implicit rules only; the incremental baseline is dropped with it, so the
    next incremental DRC is a full one that matches a fresh full run."""
    from tests.helpers.engine import open_engine

    bad = tmp_path / "bad.kicad_dru"
    bad.write_text("(version 1)\n(rule bogus (constraint no_such_constraint))\n")
    eng = open_engine(str(MODERN_PCB_FIXTURE))
    try:
        full = len(eng.run_drc())
        with pytest.raises(RuntimeError, match=r"DRC could not load the rules .*bad\.kicad_dru"):
            eng.run_drc(str(bad))
        assert len(eng.run_drc_incremental()) == full
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# verify_netclass_rules — the load-time consistency check (no C++: a stand-in
# engine serving design rules + a project file written to tmp)
# ---------------------------------------------------------------------------

_PRO_MINIMA = {"min_track_width": 0.1, "min_clearance": 0.1,
               "min_via_diameter": 0.4, "min_through_hole_diameter": 0.2}


def _nc(name, tw=0.25, cl=0.2, vd=0.6, dr=0.3):
    from types import SimpleNamespace
    return SimpleNamespace(name=name, track_width_mm=tw, clearance_mm=cl,
                           via_diameter_mm=vd, via_drill_mm=dr)


def _rules_engine(tmp_path: Path, default, others=(), pro_minima=None):
    """Stand-in engine: board minima = ``_PRO_MINIMA``; ``pro_minima`` (default: the same)
    is what the .kicad_pro declares under board.design_settings.rules."""
    import json
    from types import SimpleNamespace
    pro = tmp_path / "b.kicad_pro"
    rules_json = dict(_PRO_MINIMA if pro_minima is None else pro_minima)
    pro.write_text(json.dumps({"board": {"design_settings": {"rules": rules_json}}}))
    rules = SimpleNamespace(
        default_netclass=default, netclasses=list(others),
        min_track_width_mm=0.1, min_clearance_mm=0.1,
        min_via_diameter_mm=0.4, min_through_hole_mm=0.2)
    return SimpleNamespace(get_design_rules=lambda: rules,
                           was_project_loaded_from_file=lambda: True,
                           get_project_path=lambda: str(pro))


def _verify(engine) -> None:
    from pcb_world.engine.drc_config import verify_netclass_rules
    verify_netclass_rules(engine, engine.get_design_rules())


def test_verify_netclass_rules_accepts_a_consistent_board(tmp_path: Path) -> None:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _verify(_rules_engine(tmp_path, _nc("Default"), [_nc("Power", tw=1.0)]))


@pytest.mark.parametrize("missing", sorted(_PRO_MINIMA))
def test_verify_netclass_rules_refuses_a_project_without_a_board_minimum(
    tmp_path: Path, missing: str,
) -> None:
    """KiCad would substitute its built-in default for the missing minimum."""
    minima = {k: v for k, v in _PRO_MINIMA.items() if k != missing}
    with pytest.raises(ValueError, match=f"declares no board minimum '{missing}'"):
        _verify(_rules_engine(tmp_path, _nc("Default"), pro_minima=minima))


def test_verify_netclass_rules_refuses_an_unset_default_class_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Default net class leaves via_drill_mm unset"):
        _verify(_rules_engine(tmp_path, _nc("Default", dr=-1.0)))


@pytest.mark.parametrize("field", ["tw", "cl", "vd", "dr"])
def test_verify_netclass_rules_refuses_a_zero_default_class_size(tmp_path: Path, field: str) -> None:
    """A 0 Default size is refused at load, as set_design_rules refuses it — also on a
    board whose minima are 0 (no 'below the board minimum' to catch it)."""
    from pcb_world.engine.drc_config import NETCLASS_RULE_FIELDS
    engine = _rules_engine(tmp_path, _nc("Default", **{field: 0.0}))
    rules = engine.get_design_rules()
    for _attr, floor_attr, _pro_key in NETCLASS_RULE_FIELDS:
        setattr(rules, floor_attr, 0.0)
    with pytest.raises(ValueError, match="Default net class .* is 0"):
        _verify(engine)


def test_verify_netclass_rules_warns_on_an_unset_other_class_field(tmp_path: Path) -> None:
    """A non-Default class field left unset inherits the Default value (KiCad's resolution)."""
    with pytest.warns(UserWarning, match="'Power' leaves track_width_mm unset"):
        _verify(_rules_engine(tmp_path, _nc("Default"), [_nc("Power", tw=-1.0)]))


def test_verify_netclass_rules_refuses_a_class_below_the_board_minimum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'Power' via_diameter_mm=0.3 is below the board minimum"):
        _verify(_rules_engine(tmp_path, _nc("Default"), [_nc("Power", vd=0.3)]))
