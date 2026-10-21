"""netclass_defaults.py (tools/datagen/pcbench_prep): diff-pair fields a KiCad 5 board never set take the
class's own track width / clearance instead of the KiCad defaults the KiCad 9 round trip writes.

The engine test pins why it matters: KiCad's implicit net-class rules route a net it detects as a
differential pair by name (``D+``/``D-``) at ``diff_pair_width``. With the round-trip default (0.2 mm)
a ``D+`` net of a 0.3 mm class was routed at 0.2 mm (found on D3-B, e.g. 0322_USBtin's /D+, /D-).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from configs.loader.schema import EnvConfig
from pcb_world.engine import engine_available

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import make_guide  # noqa: E402
import netclass_defaults as nd  # noqa: E402

JUMANJI_DIR = (Path(__file__).resolve().parent
               / "fixtures" / "jumanji" / "synth_1L_grid10_5net_v15" / "test")

LEGACY = """(kicad_pcb (version 20171130) (host pcbnew "(5.1.12)")
  (net_class Default "This is the default net class."
    (clearance 0.2)
    (trace_width 0.3)
    (via_dia 0.6)
    (via_drill 0.3)
    (uvia_dia 0.3)
    (uvia_drill 0.1)
    (add_net "D+")
  )
  (net_class Fast ""
    (clearance 0.15)
    (trace_width 0.25)
    (diff_pair_width 0.12)
    (diff_pair_gap 0.1)
    (add_net "USB+")
  )
  (net_class Same ""
    (clearance 0.2)
    (trace_width 0.157)
    (diff_pair_width 0.157)
    (diff_pair_gap 0.15)
    (add_net "X+")
  )
  (net_class GapOnly ""
    (clearance 0.3)
    (trace_width 0.3)
    (diff_pair_width 0.2)
    (diff_pair_gap 0.15)
    (add_net "Y+")
  )
)"""


def _cls(name, **kw):
    base = {"name": name, "track_width": 0.3, "clearance": 0.2,
            "diff_pair_width": 0.2, "diff_pair_gap": 0.25, "diff_pair_via_gap": 0.25}
    base.update(kw)
    return base


def test_parse_legacy_net_classes_reads_each_block():
    legacy, net_to_class = nd.parse_legacy_net_classes(LEGACY)
    assert set(legacy) == {"Default", "Fast", "Same", "GapOnly"}
    assert net_to_class == {"D+": "Default", "USB+": "Fast", "X+": "Same", "Y+": "GapOnly"}
    assert "diff_pair_width" not in legacy["Default"] and legacy["Default"]["trace_width"] == 0.3
    assert (legacy["Fast"]["diff_pair_width"], legacy["Fast"]["diff_pair_gap"]) == (0.12, 0.1)
    # make_guide's own reader is the same parser restricted to its six keys
    params, nets = make_guide.extract_pcb_net_class_blocks(LEGACY)
    assert nets == net_to_class and "diff_pair_width" not in params["Fast"] and params["Fast"]["trace_width"] == 0.25


def test_designer_set_only_for_a_value_of_its_own():
    legacy = nd.parse_legacy_net_classes(LEGACY)[0]
    assert nd.designer_set(legacy["Fast"], "diff_pair_width") and nd.designer_set(legacy["Fast"], "diff_pair_gap")
    assert not nd.designer_set(legacy["Default"], "diff_pair_gap")       # absent
    assert not nd.designer_set(legacy["Same"], "diff_pair_width")        # equal to the class value
    # KiCad 5 writes both fields when either differs from its default: the untouched width (0.2) follows
    assert not nd.designer_set(legacy["GapOnly"], "diff_pair_width")
    assert nd.designer_set(legacy["GapOnly"], "diff_pair_gap")


def test_fill_unset_diff_pair_takes_the_class_values_and_keeps_the_designers():
    pro = {"net_settings": {"classes": [
        _cls("Default"),
        _cls("Fast", track_width=0.25, clearance=0.15, diff_pair_width=0.12, diff_pair_gap=0.1),
        _cls("guide_W0.508", track_width=0.508, clearance=0.2),   # no legacy block: judged by Default's
        _cls("Same", track_width=0.2, diff_pair_width=0.157, diff_pair_gap=0.15),
        _cls("GapOnly", clearance=0.3, diff_pair_width=0.2, diff_pair_gap=0.15),
    ]}}
    legacy = nd.parse_legacy_net_classes(LEGACY)[0]
    filled = nd.fill_unset_diff_pair(pro, legacy)
    by = {c["name"]: c for c in pro["net_settings"]["classes"]}
    assert (by["Default"]["diff_pair_width"], by["Default"]["diff_pair_gap"]) == (0.3, 0.2)
    assert (by["guide_W0.508"]["diff_pair_width"], by["guide_W0.508"]["diff_pair_gap"]) == (0.508, 0.2)
    assert (by["Fast"]["diff_pair_width"], by["Fast"]["diff_pair_gap"]) == (0.12, 0.1)   # designer's
    assert (by["Same"]["diff_pair_width"], by["Same"]["diff_pair_gap"]) == (0.2, 0.15)  # width follows
    assert (by["GapOnly"]["diff_pair_width"], by["GapOnly"]["diff_pair_gap"]) == (0.3, 0.15)
    assert by["Default"]["diff_pair_via_gap"] == 0.25                                   # not read by single routing
    assert sorted(filled) == ["Default.diff_pair_gap", "Default.diff_pair_width", "GapOnly.diff_pair_width",
                              "Same.diff_pair_width", "guide_W0.508.diff_pair_gap", "guide_W0.508.diff_pair_width"]
    assert nd.fill_unset_diff_pair(pro, legacy) == []   # idempotent


def test_make_guide_classes_follow_the_guide_width_unless_default_carries_a_designer_value():
    """Guide classes are named Default / guide_W* and copy the Default class (assign_class_names)."""
    groups = {(0.3, ()): ["N1"], (0.508, ()): ["D+", "D-"]}
    info = {(0.3, ()): ("Default", {}), (0.508, ()): ("guide_W0.508", {})}
    follows = {"net_settings": {"classes": [_cls("Default", diff_pair_width=0.3, diff_pair_gap=0.2)]}}
    by = {c["name"]: c for c in make_guide.build_guide_pro(follows, groups, info)["net_settings"]["classes"]}
    assert (by["guide_W0.508"]["diff_pair_width"], by["guide_W0.508"]["diff_pair_gap"]) == (0.508, 0.2)
    designer = {"net_settings": {"classes": [_cls("Default", diff_pair_width=0.12, diff_pair_gap=0.1)]}}
    by = {c["name"]: c for c in make_guide.build_guide_pro(designer, groups, info)["net_settings"]["classes"]}
    assert (by["guide_W0.508"]["diff_pair_width"], by["guide_W0.508"]["diff_pair_gap"]) == (0.12, 0.1)


@pytest.mark.parametrize("value", [".", "nan", "inf", "0.3mm", "-0.1"])
def test_a_legacy_value_that_is_not_a_finite_number_fails_loudly(value):
    with pytest.raises(ValueError):
        nd.parse_legacy_net_classes(f'(net_class Default "d"\n  (clearance {value})\n  (trace_width 0.3)\n)')


def test_a_description_does_not_shadow_a_field():
    legacy = nd.parse_legacy_net_classes('(net_class Power "Power nets (clearance 0.3mm)"\n  (clearance 0.25)\n)')[0]
    assert legacy["Power"]["clearance"] == 0.25


def test_fill_unset_diff_pair_refuses_a_project_of_the_wrong_shape():
    with pytest.raises(ValueError, match="named objects"):
        nd.fill_unset_diff_pair({"net_settings": {"classes": [{"name": ["x"]}]}}, {})


def test_convert_sample_fails_one_board_whose_metadata_cannot_be_copied(tmp_path, monkeypatch):
    import convert_v9

    src = tmp_path / "PCBs" / "b"
    src.mkdir(parents=True)
    (src / "processed.kicad_pcb").write_text(LEGACY)
    (src / "metadata.json").write_text("{}")
    helper = tmp_path / "helper.py"
    helper.write_text("import sys, shutil, json\nshutil.copy(sys.argv[1], sys.argv[2])\n"
                      "json.dump({'net_settings': {'classes': [{'name': 'Default', 'track_width': 0.3, 'clearance': 0.2}]}},"
                      " open(sys.argv[2][:-len('.kicad_pcb')] + '.kicad_pro', 'w'))\n")
    monkeypatch.setattr(convert_v9.kicad_tools, "pcbnew_python", lambda: sys.executable)

    real_copy = convert_v9.shutil.copy

    def no_space(src, dst, *a, **k):
        if str(src).endswith("metadata.json"):
            raise OSError(28, "No space left on device")
        return real_copy(src, dst, *a, **k)
    monkeypatch.setattr(convert_v9.shutil, "copy", no_space)
    r = convert_v9.convert_sample("b", tmp_path / "PCBs", tmp_path / "out", helper)
    assert r["status"] == "failed" and r["error"].startswith("post-processing:") and "No space" in r["error"]
    assert not (tmp_path / "out" / "b").exists()


def test_convert_sample_fails_one_board_on_a_bad_project(tmp_path, monkeypatch):
    """A board whose post-processing raises is reported failed; it does not abort the run."""
    import convert_v9

    src = tmp_path / "PCBs" / "b"
    src.mkdir(parents=True)
    (src / "processed.kicad_pcb").write_text(LEGACY)
    helper = tmp_path / "helper.py"
    helper.write_text("import sys, shutil\nshutil.copy(sys.argv[1], sys.argv[2])\n"
                      "open(sys.argv[2][:-len('.kicad_pcb')] + '.kicad_pro', 'w').write('{not json')\n")
    monkeypatch.setattr(convert_v9.kicad_tools, "pcbnew_python", lambda: sys.executable)
    r = convert_v9.convert_sample("b", tmp_path / "PCBs", tmp_path / "out", helper)
    assert r["status"] == "failed" and "post-processing" in r["error"]
    assert not (tmp_path / "out" / "b").exists()
    # valid JSON of the wrong shape is bad input too, not a crash of the run
    helper.write_text("import sys, shutil\nshutil.copy(sys.argv[1], sys.argv[2])\n"
                      "open(sys.argv[2][:-len('.kicad_pcb')] + '.kicad_pro', 'w').write("
                      "'{\"net_settings\": {\"classes\": [\"Default\"]}}')\n")
    r = convert_v9.convert_sample("b", tmp_path / "PCBs", tmp_path / "out", helper)
    assert r["status"] == "failed" and "classes" in r["error"]


# --- through the engine: KiCad routes a name-detected diff pair at diff_pair_width ------------------

@pytest.fixture(scope="module")
def dp_board(tmp_path_factory):
    """Jumanji board whose NET2 / NET3 are renamed D+ / D- (a pair by KiCad's name rule)."""
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")
    if not (JUMANJI_DIR / "board_00000.kicad_pcb").exists():
        pytest.skip(f"jumanji fixture missing: {JUMANJI_DIR}")
    tmp = tmp_path_factory.mktemp("dp_board")
    text = (JUMANJI_DIR / "board_00000.kicad_pcb").read_text()
    (tmp / "b.kicad_pcb").write_text(text.replace('"NET2"', '"D+"').replace('"NET3"', '"D-"'))
    pro = json.loads((JUMANJI_DIR / "board_00000.kicad_pro").read_text())
    pro["net_settings"]["classes"] = [dict(pro["net_settings"]["classes"][0],
                                           **_cls("Default", via_diameter=0.6, via_drill=0.3))]
    pro["board"]["design_settings"]["rules"].update(
        {"min_clearance": 0.1, "min_track_width": 0.1, "min_via_diameter": 0.4, "min_through_hole_diameter": 0.2})
    return tmp, pro


def _routed_width(tmp: Path, pro: dict, name: str, net_id: int) -> float:
    from pcb_world.core.env import PCBWorld

    d = tmp / name
    d.mkdir()
    shutil.copy(tmp / "b.kicad_pcb", d / "b.kicad_pcb")
    (d / "b.kicad_pro").write_text(json.dumps(pro, indent=2))
    env = PCBWorld(str(d / "b.kicad_pcb"), EnvConfig.with_overrides(max_steps=20))
    try:
        obs, _ = env.reset()
        pad = next(iter(obs["board_static"]["nets"][f"net_{net_id}"]["pads"].values()))
        x, y = pad["center"]["xy"]
        assert env._engine.start_route(x, y, pad["layer"] or 1)
        return env._engine.get_route_sizes()["track_width"]
    finally:
        env.close()


def test_round_trip_default_routes_the_pair_at_the_kicad_default(dp_board):
    """Characterises the KiCad rule the fix works around: diff_pair_width 0.2 wins on D+ only."""
    tmp, pro = dp_board
    assert _routed_width(tmp, pro, "raw_d", 2) == pytest.approx(0.2)    # D+
    assert _routed_width(tmp, pro, "raw_n", 1) == pytest.approx(0.3)    # NET1, not a pair


def test_filled_pro_routes_the_pair_at_the_class_track_width(dp_board):
    tmp, pro = dp_board
    pro = json.loads(json.dumps(pro))
    nd.fill_unset_diff_pair(pro, {"Default": {"clearance": 0.2, "trace_width": 0.3, "via_dia": 0.6, "via_drill": 0.3}})
    assert _routed_width(tmp, pro, "filled", 2) == pytest.approx(0.3)
