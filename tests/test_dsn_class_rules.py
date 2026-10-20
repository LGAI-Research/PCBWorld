"""Net-class rules on the way to the rule-based routers: the make_dsn_orp_*
scripts write them into the DSN exactly (on the file's grid, never below the
KiCad rule, each class with its own via), and the runner hands them to
FreeRouting / KRT only after checking them against the board's .kicad_pro."""
import copy
import fnmatch
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
PREP = REPO / "engine" / "pcbnew_prep"
GENERATORS = sorted(p.name for p in PREP.glob("make_dsn_orp_*.py"))
# Modules a generator imports from its own folder (plus the stubbed pcbnew): restored after each
# test so none leaks into the next test of the same xdist worker.
_PREP_MODULES = ("pcbnew", "setup_drvzero", "dsn_units", "dsn_classes")

# Net names pcbnew writes bare (3.3V, RX.1, $N1, ~RST, +5V, a leading-dash -15V) or quoted
# (a later '-', parens, braces as in the overbar ~{RESET}; quoted without escaping, so (a)\ ends in a backslash), with the
# fnmatch / argparse specials -15V and /5(**).
POWER_NETS = ["VCC", "Net-(D1-Pad1)", "3.3V", "RX.1", "$N1", "~RST", "+5V", "~{RESET}", "-15V",
              "/5(**)", "(a)\\"]
# 8 mil track, 6 mil clearance: both have a fourth decimal in mm.
# The .kicad_pro format KiCad 9 writes: membership lives in netclass_patterns only.
PRO = {"meta": {"version": 3}, "net_settings": {"meta": {"version": 4}, "classes": [
    {"name": "Default", "track_width": 0.2032, "clearance": 0.1524,
     "via_diameter": 0.6, "via_drill": 0.4},
    {"name": "Power", "track_width": 0.5, "clearance": 0.3,
     "via_diameter": 0.8, "via_drill": 0.5},
], "netclass_patterns": [{"pattern": n, "netclass": "Power"} for n in POWER_NETS]}}


def _q(name):
    """pcbnew's quoting (OUTPUTFORMATTER::GetQuoteChar)."""
    quoted = (not name or name[0] == "#" or any(c in "\t ()%{}" for c in name) or "-" in name[1:])
    return f'"{name}"' if quoted else name


_NETS = ["GND"] + POWER_NETS
# What pcbnew's ExportSpecctraDSN writes (one class, its own via names) before the rewrite.
DSN = """(pcb board
  (parser
    (string_quote ")
    (space_in_quoted_tokens on)
  )
  (resolution um 10)
  (unit um)
  (structure
    (layer F.Cu
      (type signal)
      (property
        (index 0)
      )
    )
    (layer B.Cu
      (type signal)
      (property
        (index 1)
      )
    )
    (via "Via[0-1]_600:400_um" "Via[0-1]_800:500_um")
  )
  (network
""" + "".join(f"    (net {_q(n)}\n      (pins U1-{i})\n    )\n" for i, n in enumerate(_NETS)) + \
    "    (class kicad_default " + " ".join(_q(n) for n in _NETS) + """
      (circuit
        (use_via "Via[0-1]_600:400_um")
      )
      (rule
        (width 250)
        (clearance 200)
      )
    )
  )
  (wiring
  )
)
"""


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def prep(monkeypatch, tmp_path):
    """engine/pcbnew_prep importable without a KiCad build; its modules never outlive the test."""
    saved = {k: sys.modules.pop(k) for k in _PREP_MODULES if k in sys.modules}
    sys.modules["pcbnew"] = mock.MagicMock()
    # The scripts refuse to import without their data roots; setup_drvzero, which they import, needs none.
    for var in ("PCBENCH_PCB_ROOT", "PCBENCH_DSN_ROOT", "SYNTH2L_PCB_ROOT", "SYNTH2L_DSN_ROOT",
                "SYNTH1L_PCB_ROOT", "SYNTH1L_DSN_ROOT"):
        monkeypatch.setenv(var, str(tmp_path))
    monkeypatch.delenv("DRVZERO_DATA_DIR", raising=False)
    monkeypatch.syspath_prepend(str(PREP))
    yield
    for k in _PREP_MODULES:
        sys.modules.pop(k, None)
    sys.modules.update(saved)


@pytest.fixture
def units(prep):
    return _load(PREP / "dsn_units.py", "dsn_units_under_test")


@pytest.fixture
def classes(prep):
    return _load(PREP / "dsn_classes.py", "dsn_classes_under_test")


@pytest.fixture(params=GENERATORS)
def generator(request, prep):
    """One make_dsn_orp_* script with pcbnew stubbed."""
    return _load(PREP / request.param, f"gen_{request.param[:-3]}")


def test_there_is_one_generator_per_dataset_family():
    assert GENERATORS == ["make_dsn_orp_synth.py", "make_dsn_orp_synth_1l.py",
                          "make_dsn_orp_v3.py"], GENERATORS


def test_setup_drvzero_imports_without_its_data_dir_and_main_still_names_it(prep):
    mod = _load(PREP / "setup_drvzero.py", "setup_drvzero_under_test")
    assert callable(mod.remove_routing)
    with pytest.raises(SystemExit, match="DRVZERO_DATA_DIR"):
        mod.main()


# ── units ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mm, steps, token", [
    (0.25, 10, "250"),          # on the µm grid: stays a whole number
    (0.2032, 10, "203.2"),      # 8 mil: was written as 203
    (0.1524, 10, "152.4"),
    (0.20325, 10, "203.3"),     # off the 0.1 µm grid: up, never down
    (0.2032, 1, "204"),
    (0.2, 100, "200"),
    (0.20321, 100, "203.21"),
])
def test_mm_to_dsn_rounds_up_on_the_file_grid(units, mm, steps, token):
    assert units.mm_to_dsn(mm, steps) == token
    assert units.dsn_to_mm(token) >= mm - 1e-12
    assert units.rule_matches(units.dsn_to_mm(token), mm, steps)


@pytest.mark.parametrize("mm", [0.0, -0.20325, float("inf"), float("nan"), 0.0000004])
def test_mm_to_dsn_refuses_a_rule_that_is_not_a_positive_length(units, mm):
    with pytest.raises(ValueError, match="positive length"):
        units.mm_to_dsn(mm, 10)


def test_mm_to_dsn_works_on_kicad_s_whole_nm(units):
    # KiCad holds 0.2000001 mm as 200000 nm: no extra grid step for float dust.
    assert units.mm_to_dsn(0.2000001, 10) == "200"
    assert units.kicad_nm(0.0000005) == 1 and units.kicad_nm(0.2032) == 203200


def test_rule_matches_rejects_a_value_below_or_a_step_above(units):
    assert not units.rule_matches(0.203, 0.2032, 10)     # the old whole-µm value
    assert not units.rule_matches(0.2033, 0.2032, 10)
    assert units.rule_matches(0.2032, 0.2032, 10)


def test_resolution_comes_from_the_header(units):
    assert units.dsn_resolution(DSN) == 10
    with pytest.raises(ValueError, match="resolution"):
        units.dsn_resolution("(pcb board (unit um))")
    with pytest.raises(ValueError, match="not in um"):
        units.dsn_resolution("(pcb board (resolution mil 10))")
    # A rule on a 1/3 µm grid has no exact decimal token: refused, not truncated.
    with pytest.raises(ValueError, match="power of ten"):
        units.dsn_resolution("(pcb board (resolution um 3))")


# ── generators ───────────────────────────────────────────────────────

def _class_block(dsn, name):
    start = dsn.index(f"(class {name} ")
    return dsn[start:dsn.index("\n    )", start)]


def test_generator_writes_each_class_rule_and_via(generator):
    new_dsn, _ = generator.fix_dsn_classes(DSN, PRO)
    default, power = _class_block(new_dsn, "Default"), _class_block(new_dsn, "Power")
    assert "(width 203.2)" in default and "(clearance 152.4)" in default
    assert '(use_via "Via[0-1]_600:400_um")' in default
    assert "(width 500)" in power and "(clearance 300)" in power
    assert '(use_via "Via[0-1]_800:500_um")' in power
    assert "kicad_default" not in new_dsn
    assert generator.verify_dsn_constraints(new_dsn, PRO) == (True, [])


# make_dsn_orp_synth_1l's one-copper-layer export, as real pcbnew 9.0.8 output on 3 grid50 boards showed:
# no B.Cu layer and vias spanning layer 0 alone (it also drops the via padstack's B.Cu shape,
# which has no counterpart in the `DSN` fixture).
_B_CU = "    (layer B.Cu\n      (type signal)\n      (property\n        (index 1)\n      )\n    )\n"
DSN_1L = DSN.replace(_B_CU, "").replace("Via[0-1]", "Via[0-0]")


def test_a_one_layer_export_gets_its_rules_and_the_runner_reads_them(classes, tmp_path):
    assert _B_CU in DSN and "B.Cu" not in DSN_1L
    new_dsn, _ = classes.fix_dsn_classes(DSN_1L, PRO)
    assert '(use_via "Via[0-0]_600:400_um")' in _class_block(new_dsn, "Default")
    assert '(use_via "Via[0-0]_800:500_um")' in _class_block(new_dsn, "Power")
    assert classes.verify_dsn_constraints(new_dsn, PRO) == (True, [])
    got = _constraints(tmp_path, new_dsn)
    assert got["layers"] == ["F.Cu"]
    assert (got["track_width"], got["via_size"], got["via_drill"]) == (0.2032, 0.6, 0.4)


def test_generator_verify_flags_a_whole_micron_rule_and_a_foreign_via(generator):
    new_dsn, _ = generator.fix_dsn_classes(DSN, PRO)
    ok, issues = generator.verify_dsn_constraints(new_dsn.replace("(width 203.2)", "(width 203)"), PRO)
    assert not ok and any("width mismatch" in i for i in issues)
    swapped = new_dsn.replace('(use_via "Via[0-1]_800:500_um")', '(use_via "Via[0-1]_600:400_um")')
    ok, issues = generator.verify_dsn_constraints(swapped, PRO)
    assert not ok and any("[Power] use_via" in i for i in issues)


def test_generator_refuses_a_via_the_export_did_not_declare(generator):
    pro = copy.deepcopy(PRO)
    pro["net_settings"]["classes"][1]["via_diameter"] = 1.0
    with pytest.raises(ValueError, match=r"'Power'.*Via\[0-1\]_1000:500_um.*not declared"):
        generator.fix_dsn_classes(DSN, pro)


def test_generator_names_the_via_padstack_as_pcbnew_does(classes):
    # specctra_export.cpp: "Via[%d-%d]_%.6g:%.6g_um" of the µm values.
    assert classes.via_padstack_name(0.6858, 0.3302, "0-1") == "Via[0-1]_685.8:330.2_um"
    assert classes.via_padstack_name(0.965200, 0.599999, "0-3") == "Via[0-3]_965.2:599.999_um"
    assert classes.via_padstack_name(1.2, 0.6, "0-0") == "Via[0-0]_1200:600_um"
    assert classes.via_padstack_name(1.234567, 0.6, "0-1") == "Via[0-1]_1234.57:600_um"


def test_generator_quotes_names_as_pcbnew_does(classes):
    for name, token in (("+5V", "+5V"), ("-15V", "-15V"), ("a-b", '"a-b"'), ("#x", '"#x"'),
                        ("", '""'), ("50%", '"50%"'), ("{A}", '"{A}"'), ("/x/y", "/x/y")):
        assert classes._dsn_net_token(name) == token


@pytest.mark.parametrize("nets, patterns, message", [
    (["VCC", "VCC2"], [("VCC*", "Power")], "also matches \\['VCC', 'VCC2'\\]"),  # wildcard
    (["3.3V", "3x3V"], [("3.3V", "Power")], "also matches \\['3x3V'\\]"),  # regex reading
    (["D0", "D1"], [("D[0..7]", "Power")], "bus pattern"),
    (["VCC"], [("VCC", "Power"), ("VCC", "Default")], "KiCad merges"),
])
def test_generator_refuses_class_assignments_kicad_reads_more_widely(classes, nets, patterns, message):
    class_params = {"Default": {}, "Power": {}}
    with pytest.raises(ValueError, match=message):
        classes.resolve_net_to_class(nets, class_params, patterns)


def test_generator_reads_exact_names_with_special_characters(classes):
    nets = ["/5(**)", "/6(**)", "+5V", "Net-(D1-Pad1)", "/~{RST}"]
    patterns = [(n, "Power") for n in nets[::2]]
    got = classes.resolve_net_to_class(nets, {"Default": {}, "Power": {}}, patterns)
    assert got == {"/5(**)": "Power", "/6(**)": "Default", "+5V": "Power",
                   "Net-(D1-Pad1)": "Default", "/~{RST}": "Power"}


def test_generator_inherits_an_unset_field_from_default(generator):
    pro = copy.deepcopy(PRO)
    del pro["net_settings"]["classes"][1]["clearance"]
    new_dsn, _ = generator.fix_dsn_classes(DSN, pro)
    assert "(clearance 152.4)" in _class_block(new_dsn, "Power")
    del pro["net_settings"]["classes"][0]["clearance"]
    with pytest.raises(ValueError, match="'Default' has no clearance"):
        generator.fix_dsn_classes(DSN, pro)


@pytest.mark.parametrize("edit, message", [
    (lambda ns: ns["classes"].pop(0), "no Default net class"),
    (lambda ns: ns.update(netclass_assignments={"GND": ["Power"]}), "netclass_assignments"),
    (lambda ns: ns["classes"][1].update(track_width=0), "positive length"),
])
def test_generator_refuses_a_pro_it_cannot_state_exactly(generator, edit, message):
    pro = copy.deepcopy(PRO)
    edit(pro["net_settings"])
    with pytest.raises(ValueError, match=message):
        generator.fix_dsn_classes(DSN, pro)


def _run_board(gen, tmp_path, monkeypatch, pro):
    """A generator's per-board step with pcbnew stubbed to export DSN; returns (result, dsn path)."""
    def export(board, path):
        Path(path).write_text(DSN)
        return True
    gen.pcbnew.ExportSpecctraDSN.side_effect = export
    gen.pcbnew.LoadBoard.return_value.GetNetInfo.return_value.GetNetCount.return_value = 0
    for name in ("extract_pads", "extract_nets", "extract_layers"):
        monkeypatch.setattr(gen, name, lambda *a, **k: [])
    monkeypatch.setattr(gen, "build_drc_rules", lambda *a, **k: {"netclasses": {}})
    monkeypatch.setattr(gen, "build_orp", lambda *a, **k: {})
    src = tmp_path / "src" / "b"
    src.mkdir(parents=True)
    if hasattr(gen, "process_folder"):
        stem = gen.INPUT_STEM
        (src / f"{stem}.kicad_pcb").write_text("(kicad_pcb)\n")
        (src / f"{stem}.kicad_pro").write_text(json.dumps(pro))
        dsn = tmp_path / "dst" / "b" / gen.DSN_NAME
        dsn.parent.mkdir(parents=True)
        dsn.write_text("stale")
        return gen.process_folder(src, tmp_path / "dst"), dsn
    pcb = src / "board_00000.kicad_pcb"
    pcb.write_text("(kicad_pcb)\n")
    pcb.with_suffix(".kicad_pro").write_text(json.dumps(pro))
    dsn = tmp_path / "dst" / "board_00000_unrouted.dsn"
    dsn.parent.mkdir(parents=True)
    dsn.write_text("stale")
    return gen.process_board(pcb, pcb.with_suffix(".kicad_pro"), tmp_path / "dst", "board_00000"), dsn


def test_generator_board_writes_the_verified_dsn(generator, tmp_path, monkeypatch):
    res, dsn = _run_board(generator, tmp_path, monkeypatch, PRO)
    assert res["status"] == "ok", res
    assert "(width 203.2)" in dsn.read_text()
    assert not list(dsn.parent.glob("*_pcbnew.dsn"))


def test_only_the_1l_generator_exports_one_copper_layer(generator, tmp_path, monkeypatch):
    """synth_1L boards declare F.Cu + B.Cu but are routed on F.Cu alone: their DSN must
    carry one copper layer, set before the export, so FreeRouting cannot reach B.Cu."""
    res, _ = _run_board(generator, tmp_path, monkeypatch, PRO)
    assert res["status"] == "ok", res
    names = [c[0] for c in generator.pcbnew.mock_calls]
    layer_calls = [c for c in generator.pcbnew.mock_calls if c[0].endswith("SetCopperLayerCount")]
    if Path(generator.__file__).name == "make_dsn_orp_synth_1l.py":
        assert [c[1] for c in layer_calls] == [(1,)]
        assert names.index(layer_calls[0][0]) < names.index("ExportSpecctraDSN")
    else:
        assert layer_calls == []


def test_generator_board_fails_without_a_dsn_when_verify_fails(generator, tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "verify_dsn_constraints", lambda *a: (False, ["[Default] width mismatch"]))
    res, dsn = _run_board(generator, tmp_path, monkeypatch, PRO)
    assert res["status"] == "fail" and "do not match" in res["msg"]
    assert not dsn.exists() and not list(dsn.parent.glob("*_pcbnew.dsn"))


def test_generator_board_fails_without_a_dsn_on_a_bad_pro(generator, tmp_path, monkeypatch):
    pro = copy.deepcopy(PRO)
    pro["net_settings"]["classes"].pop(0)
    res, dsn = _run_board(generator, tmp_path, monkeypatch, pro)
    assert res["status"] == "fail" and "Default" in res["msg"]
    assert not dsn.exists()


def test_generator_run_exits_non_zero_on_a_failed_board(generator, tmp_path, monkeypatch):
    failed = {"folder": "b", "board": "b", "status": "fail", "msg": "x", "dsn_classes": 0,
              "dsn_issues": [], "warnings": [], "orp_nets": 0, "orp_classes": 0}
    if hasattr(generator, "process_folder"):
        (tmp_path / "src" / "b").mkdir(parents=True)
        (tmp_path / "src" / "b" / f"{generator.INPUT_STEM}.kicad_pcb").write_text("")
        monkeypatch.setattr(generator, "process_folder", lambda *a, **k: dict(failed))
        argv = ["x", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst")]
    else:
        (tmp_path / "src" / "test").mkdir(parents=True)
        (tmp_path / "src" / "test" / "board_00000.kicad_pcb").write_text("")
        monkeypatch.setattr(generator, "process_board", lambda *a, **k: dict(failed))
        argv = ["x", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst"), "--splits", "test"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        generator.main()
    assert exc.value.code == 1


# ── runner ───────────────────────────────────────────────────────────

@pytest.fixture
def written(classes):
    """The DSN the generators write for PRO."""
    return classes.fix_dsn_classes(DSN, PRO)[0]


# The board's .kicad_pcb layer table; ALIASED as KiCad writes user layer names (Top / Bottom).
PCB = '(kicad_pcb\n\t(layers\n\t\t(0 "F.Cu" signal)\n\t\t(2 "B.Cu" signal)\n\t\t(25 "Edge.Cuts" user)\n\t)\n)\n'
PCB_ALIASED = PCB.replace('"F.Cu" signal)', '"F.Cu" signal "Top")').replace('"B.Cu" signal)', '"B.Cu" signal "Bottom")')


def _files(tmp_path, dsn_text, pro=PRO):
    dsn, pro_path = tmp_path / "board.dsn", tmp_path / "board.kicad_pro"
    dsn.write_text(dsn_text)
    pro_path.write_text(json.dumps(pro))
    return dsn, pro_path


def _constraints(tmp_path, dsn_text, pro=PRO, pcb=PCB):
    from methods.baselines.rule_based.run_rule_based_routers import _parse_dsn_constraints
    pcb_path = tmp_path / "board.kicad_pcb"
    pcb_path.write_text(pcb)
    return _parse_dsn_constraints(*_files(tmp_path, dsn_text, pro), pcb_path)


def test_runner_reads_the_rules_the_generator_wrote(written, tmp_path):
    got = _constraints(tmp_path, written)
    assert got["layers"] == ["F.Cu", "B.Cu"]
    assert (got["track_width"], got["via_size"], got["via_drill"]) == (0.2032, 0.6, 0.4)
    # 561f158 takes one clearance and one via: the Default class's; that the classes differ is kept.
    assert got["clearance"] == 0.1524
    assert got["class_clearances"] == {"Default": 0.1524, "Power": 0.3}
    assert got["class_clearances_differ"] is True and got["class_vias_differ"] is True
    # Every net of the wider class, named exactly (no '3V' fragment of 3.3V).
    assert got["non_default_net_widths"] == {n: 0.5 for n in POWER_NETS}


def test_runner_reads_pcbnew_s_own_export(tmp_path):
    pro = copy.deepcopy(PRO)
    pro["net_settings"]["classes"] = [{"name": "Default", "track_width": 0.25, "clearance": 0.2,
                                       "via_diameter": 0.6, "via_drill": 0.4}]
    pro["net_settings"]["netclass_patterns"] = []
    got = _constraints(tmp_path, DSN, pro)
    assert (got["track_width"], got["clearance"], got["class_vias_differ"]) == (0.25, 0.2, False)
    assert got["class_clearances_differ"] is False and got["non_default_net_widths"] == {}


def test_runner_reads_a_decimal_via_name(classes, tmp_path):
    # pcbnew names vias with %.6g µm; 46 of 382 real boards have a decimal one.
    pro = copy.deepcopy(PRO)
    pro["net_settings"]["classes"][0].update(via_diameter=0.6858, via_drill=0.3302)
    dsn = DSN.replace('"Via[0-1]_600:400_um"', '"Via[0-1]_685.8:330.2_um"')
    got = _constraints(tmp_path, classes.fix_dsn_classes(dsn, pro)[0], pro)
    assert (got["via_size"], got["via_drill"]) == (0.6858, 0.3302)


def _drop_block(text, start_marker):
    """``text`` without the parenthesised block that starts at ``start_marker``."""
    start = text.index(start_marker)
    depth, i = 0, start
    while True:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
        if depth == 0:
            return text[:start] + text[i:]


@pytest.mark.parametrize("edit, message", [
    (lambda t: _drop_block(t, "(class Default "), "other classes than"),
    (lambda t: t.replace("(width 203.2)", "(width 20x)"), "malformed DSN"),
    (lambda t: t.replace("(type signal)", "(type power)"), "no signal layer"),
    (lambda t: t.replace('(use_via "Via[0-1]_600:400_um")', '(use_via "Via[0-1]_999:400_um")'),
     "uses via"),
    (lambda t: _drop_block(t, "(rule\n        (width 203.2)"), "'Default' has no \\(rule"),
    (lambda t: t.replace("(width 203.2)", "(width 203)"), "track_width 0.203 mm.*regenerate"),
    (lambda t: t.replace("(clearance 300)", "(clearance 250)"), "clearance 0.25 mm.*regenerate"),
    (lambda t: t.replace("(resolution um 10)", "(resolution um 3)"), "power of ten"),
])
def test_runner_refuses_a_dsn_that_does_not_carry_the_pro_rules(written, tmp_path, edit, message):
    from methods.baselines.rule_based.run_rule_based_routers import DsnRuleError
    with pytest.raises(DsnRuleError, match=message):
        _constraints(tmp_path, edit(written))


def test_runner_refuses_a_board_without_its_pro(written, tmp_path):
    from methods.baselines.rule_based.run_rule_based_routers import DsnRuleError, _parse_dsn_constraints
    dsn, _ = _files(tmp_path, written)
    with pytest.raises(DsnRuleError, match="kicad_pro"):
        _parse_dsn_constraints(dsn, tmp_path / "missing.kicad_pro", tmp_path / "board.kicad_pcb")


def test_krt_net_patterns_match_their_net_only_and_pass_argparse():
    from methods.baselines.rule_based.run_rule_based_routers import _krt_net_pattern
    from configs.loader.cli import StrictArgumentParser
    patterns = [_krt_net_pattern(n) for n in POWER_NETS]
    for name, pattern in zip(POWER_NETS, patterns):
        assert [n for n in POWER_NETS + ["-15", "/5(xy)", "/5(*)"] if fnmatch.fnmatch(n, pattern)] == [name]
    # KRT's own argparse shape: the escaped '-15V' is not taken for an option.
    ap = StrictArgumentParser()
    ap.add_argument("--power-nets", nargs="*", default=[])
    ap.add_argument("--power-nets-widths", nargs="*", type=float, default=[])
    args = ap.parse_args(["--power-nets", *patterns, "--power-nets-widths", *["0.5"] * len(patterns)])
    assert args.power_nets == patterns and len(args.power_nets_widths) == len(patterns)


def _board(tmp_path, dsn_text):
    dsn, pro = _files(tmp_path, dsn_text)
    raw = tmp_path / "board.kicad_pcb"
    raw.write_text(PCB)
    return SimpleNamespace(board_id="b", dsn=dsn, pro=pro, raw_pcb=raw)


# What KRT 561f158 prints when it takes the per-net widths of the written DSN for PRO.
_KRT_APPLIED = "Power net width assignments (11 nets):\n"
from methods.baselines.rule_based.run_rule_based_routers import _kicad_load_issue as _REAL_LOAD_ISSUE  # noqa: E402

# A routed board: the layer table plus one track.
ROUTED = PCB.rstrip().rstrip(")") + '\t(segment (start 0 0) (end 1 0) (width 0.2032) (layer "F.Cu") (net 1))\n)\n'


@pytest.fixture(autouse=True)
def _no_engine_load(monkeypatch, request):
    """KiCad's load of a routed board needs the engine; unit tests stub it (one test runs it)."""
    if request.node.name == "test_kicad_load_issue_with_the_engine":
        return
    from methods.baselines.rule_based import run_rule_based_routers as rb
    monkeypatch.setattr(rb, "_kicad_load_issue", lambda pcb, pro: None)
    if request.node.name != "test_runner_engine_check_stops_the_run":
        monkeypatch.setattr(rb, "_require_engine", lambda: None)


def _stub_krt(monkeypatch, tmp_path, run_route):
    """A krt_runner whose KRT tree is ``tmp_path`` (with a route.py) and whose run is ``run_route``."""
    (tmp_path / "route.py").write_text("")
    (tmp_path / "rust_router").mkdir(exist_ok=True)
    (tmp_path / "rust_router" / "Cargo.toml").write_text('version = "0.19.0"\n')
    monkeypatch.setitem(sys.modules, "krt_runner",
                        SimpleNamespace(get_krt_root=lambda: tmp_path, run_route=run_route,
                                        _ensure_grid_router_so=lambda: tmp_path))


def test_runner_routes_krt_with_the_checked_rules(written, tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    calls = []

    def run_route(args, **kwargs):
        calls.append((args, kwargs))
        Path(args[1]).write_text(ROUTED)
        return SimpleNamespace(stdout=_KRT_APPLIED, stderr="", returncode=0)

    _stub_krt(monkeypatch, tmp_path, run_route)
    res = rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit=None)
    assert res.status == "ok"
    assert res.krt_rules == {"clearance_mm": 0.1524,
                             "class_clearances_mm": {"Default": 0.1524, "Power": 0.3},
                             "class_clearances_differ": True,
                             "via_size_mm": 0.6, "via_drill_mm": 0.4, "class_vias_differ": True}
    args, kwargs = calls[0]
    assert "env" not in kwargs
    assert args[args.index("--clearance") + 1] == "0.1524"
    assert args[args.index("--layers") + 1:args.index("--layers") + 3] == ["F.Cu", "B.Cu"]
    nets = args[args.index("--power-nets") + 1:args.index("--power-nets-widths")]
    assert "[-]15V" in nets and "/5([*][*])" in nets and "3.3V" in nets
    # The papers' routing hyperparameters, explicit.
    for flag, value in (("--direction-preference-cost", "50"), ("--via-cost", "50"),
                        ("--via-proximity-cost", "10"), ("--turn-cost", "1000"),
                        ("--heuristic-weight", "1.9"), ("--max-ripup", "3"), ("--grid-step", "0.1"),
                        ("--max-iterations", "200000"), ("--max-probe-iterations", "5000"),
                        ("--hole-to-hole-clearance", "0.2"), ("--board-edge-clearance", "0.0")):
        assert args[args.index(flag) + 1] == value, flag
    # Only options KRT 561f158's route.py has.
    for gone in ("--metrics-json", "--net-clearances", "--fab-overrides", "--no-power-tap-neckdown"):
        assert gone not in args
    # The runner's own record of the call.
    record = json.loads((tmp_path / "b.krt_run.json").read_text())
    assert set(record) == {"args", "krt_root", "krt_commit", "krt_pinned_commit", "krt_unpinned_allowed",
                           "returncode", "wall_s", "status"}
    assert (record["args"], record["krt_commit"], record["returncode"],
            record["status"]) == (args, None, 0, "ok")
    assert record["wall_s"] == res.elapsed_routing_s


def test_runner_records_a_failed_krt_call(written, tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    _stub_krt(monkeypatch, tmp_path,
              lambda args, **_: SimpleNamespace(stdout="", stderr="bad", returncode=2))
    res = rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit=None)
    record = json.loads((tmp_path / "b.krt_run.json").read_text())
    assert (res.status, res.error, record["returncode"], record["status"]) == ("error", "krt rc=2", 2, "error")


@pytest.mark.parametrize("route", ["_route_krt", "_route_freerouting"])
def test_runner_turns_a_stale_dsn_into_an_error_row(written, tmp_path, monkeypatch, route):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    _stub_krt(monkeypatch, tmp_path, None)
    board = _board(tmp_path, written.replace("(width 203.2)", "(width 203)"))
    extra, kw = ((False, False, 1, {}), {}) if route == "_route_freerouting" else ((), {"krt_commit": None})
    res = getattr(rb, route)(board, 0, tmp_path, tmp_path, "algo", 60, *extra, **kw)
    assert res.status == "error" and res.error.startswith("DSN:") and "regenerate" in res.error


def _pro_with(edit):
    pro = copy.deepcopy(PRO)
    edit(pro["net_settings"])
    return pro


@pytest.mark.parametrize("edit, message", [
    # The pro moved RX.1 to Default after the DSN was written: stale membership.
    (lambda ns: ns["netclass_patterns"].remove({"pattern": "RX.1", "netclass": "Power"}),
     "other classes than.*'RX.1'"),
    (lambda ns: ns.update(netclass_assignments={"GND": ["Power"]}), "netclass_assignments"),
    (lambda ns: ns["netclass_patterns"].append({"pattern": "VC*", "netclass": "Power"}),
     "'VC\\*'.*also matches"),
    (lambda ns: ns["netclass_patterns"].append({"pattern": "VCC", "netclass": "Default"}),
     "'VCC' is in net classes"),
    (lambda ns: ns["netclass_patterns"].append({"pattern": "D[0..7]", "netclass": "Power"}), "bus pattern"),
    (lambda ns: ns["netclass_patterns"].append({"pattern": "GND", "netclass": "Nope"}), "undefined class"),
])
def test_runner_refuses_class_membership_it_cannot_match(written, tmp_path, edit, message):
    from methods.baselines.rule_based.run_rule_based_routers import DsnRuleError
    with pytest.raises(DsnRuleError, match=message):
        _constraints(tmp_path, written, _pro_with(edit))


def test_runner_matches_a_via_name_with_nm_detail(classes, tmp_path):
    # %.6g keeps six digits: 1.234567 mm is named 1234.57 µm, which must still match the pro.
    pro = _pro_with(lambda ns: ns["classes"][1].update(via_diameter=1.234567))
    dsn = DSN.replace('"Via[0-1]_800:500_um"', '"Via[0-1]_1234.57:500_um"')
    got = _constraints(tmp_path, classes.fix_dsn_classes(dsn, pro)[0], pro)
    assert got["class_vias_differ"] is True


def test_generator_board_leaves_no_stale_orp_or_temporaries(generator, tmp_path, monkeypatch):
    def export(board, path):
        Path(path).write_text("partial")
        return False                     # pcbnew reports a failed export
    res, dsn = _run_board(generator, tmp_path, monkeypatch, PRO)   # sets the stubs, succeeds
    assert res["status"] == "ok"
    generator.pcbnew.ExportSpecctraDSN.side_effect = export
    orp = next(dsn.parent.glob("*.orp"))
    res = (generator.process_folder(tmp_path / "src" / "b", tmp_path / "dst")
           if hasattr(generator, "process_folder") else
           generator.process_board(tmp_path / "src" / "b" / "board_00000.kicad_pcb",
                                   tmp_path / "src" / "b" / "board_00000.kicad_pro",
                                   tmp_path / "dst", "board_00000"))
    assert res["status"] == "fail" and "ExportSpecctraDSN" in res["msg"]
    assert not dsn.exists() and not orp.exists()
    assert not list(dsn.parent.glob("*_pcbnew.dsn"))
    if not hasattr(generator, "process_folder"):     # synth: the unrouted copies are temporary
        assert not list(dsn.parent.glob("*_unrouted.kicad_p*"))


def test_eval_rollout_puts_the_krt_rules_in_the_manifest_data(tmp_path, monkeypatch):
    driver = _load(REPO / "eval" / "rollout" / "rule_based.py", "eval_rollout_rule_based_under_test")
    rules = {"clearance_mm": 0.1524, "class_clearances_differ": True}
    _stub_krt(monkeypatch, tmp_path, None)
    monkeypatch.setattr(driver.rb, "_require_engine", lambda: None)  # the driver's own copy
    monkeypatch.setenv("KRT_ALLOW_UNPINNED", "1")       # the stub tree is not a git checkout
    monkeypatch.setattr(driver.rb, "_route_krt", lambda board, seed, *a, **k: driver.rb.BoardResult(
        board.board_id, seed, "error", 0.0, error="x", krt_rules=rules, warnings=["b.kicad_pro: w"]))
    board = SimpleNamespace(board_id="b", raw_pcb=tmp_path / "b.kicad_pcb", pro=tmp_path / "b.kicad_pro")
    rows, krt_rules, warnings = driver.run_one_seed(
        [board], baseline="krt", dataset="pcbench", grid=None, seed=0, seed_dir=tmp_path,
        timeout_s=1, workers=1, via_prohibit=False, angle_90=False, max_passes=1, use_gpu=False,
        deterministic=True)
    assert len(rows) == 1 and krt_rules == {"b": rules} and warnings == {"b": ["b.kicad_pro: w"]}


def test_runner_stops_without_a_krt_checkout(written, tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    monkeypatch.setitem(sys.modules, "krt_runner",
                        SimpleNamespace(get_krt_root=lambda: tmp_path / "none", run_route=None))
    with pytest.raises(rb.KrtNotFound, match="fetch_baselines.sh"):
        rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit=None)
    with pytest.raises(rb.KrtNotFound, match="fetch_baselines.sh"):
        rb.run_one_seed([], baseline="krt", dataset="pcbench", grid=None, seed=0,
                        seed_dir=tmp_path / "seed0", timeout_s=1, workers=1, do_eval=False,
                        reward_config=None, check_angle=45, via_prohibit=False, angle_90=False,
                        max_passes=1, extra_env={}, use_gpu=False)


def test_krt_runs_through_the_real_krt_runner_and_launcher(written, tmp_path, monkeypatch):
    """The real krt_runner and its launcher run route.py with the runner's arguments."""
    from methods.baselines.rule_based import run_rule_based_routers as rb
    krt_src = REPO / "methods" / "baselines" / "rule_based" / "krt" / "src"
    saved = sys.modules.pop("krt_runner", None)
    monkeypatch.syspath_prepend(str(krt_src))
    try:
        import krt_runner
        so_dir = tmp_path / "so"
        so_dir.mkdir()
        (so_dir / "grid_router.py").write_text("")
        monkeypatch.setattr(krt_runner, "_ensure_grid_router_so", lambda: so_dir)
        krt = tmp_path / "krt"
        krt.mkdir()
        (krt / "route.py").write_text(
            "import sys\n"
            f"print({_KRT_APPLIED!r})\n"
            f"open(sys.argv[2], 'w').write({ROUTED!r})\n")
        monkeypatch.setenv("KRT_ROOT", str(krt))
        res = rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit=None)
    finally:
        sys.modules.pop("krt_runner", None)
        if saved is not None:
            sys.modules["krt_runner"] = saved
    assert res.status == "ok", res.error
    assert Path(res.routed_pcb).read_text() == ROUTED


@pytest.mark.parametrize("log_text, routed, load_issue, message", [
    ("", ROUTED, None, "matched 0 of 11 per-net widths"),
    (_KRT_APPLIED, PCB, None, "added no copper"),
    (_KRT_APPLIED, ROUTED, "KiCad cannot load the routed board: RuntimeError: bad layer",
     "KiCad cannot load"),
])
def test_runner_refuses_a_krt_run_that_did_not_route_the_board(written, tmp_path, monkeypatch,
                                                                log_text, routed, load_issue, message):
    from methods.baselines.rule_based import run_rule_based_routers as rb

    def run_route(args, **_):
        Path(args[1]).write_text(routed)
        return SimpleNamespace(stdout=log_text, stderr="", returncode=0)

    _stub_krt(monkeypatch, tmp_path, run_route)
    monkeypatch.setattr(rb, "_kicad_load_issue", lambda pcb, pro: load_issue)
    res = rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit=None)
    assert res.status == "error" and message in res.error


def test_kicad_load_issue_with_the_engine(tmp_path):
    from pcb_world.engine import engine_available
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("engine not built")
    from methods.baselines.rule_based import run_rule_based_routers as rb
    fixtures = REPO / "tests" / "fixtures"
    assert rb._kicad_load_issue(fixtures / "sample_board.kicad_pcb", fixtures / "sample_board.kicad_pro") is None
    bad = tmp_path / "bad.kicad_pcb"
    bad.write_text((fixtures / "sample_board.kicad_pcb").read_text().rstrip().rstrip(")")
                   + '\t(segment (start 0 0) (end 1 0) (width 0.2) (layer "Top") (net 0))\n)\n')
    shutil.copy(fixtures / "sample_board.kicad_pro", tmp_path / "bad.kicad_pro")
    assert "KiCad cannot load" in rb._kicad_load_issue(bad, tmp_path / "bad.kicad_pro")


def test_runner_maps_dsn_layer_aliases_to_kicad_names(written, tmp_path):
    from methods.baselines.rule_based.run_rule_based_routers import DsnRuleError
    aliased = written.replace("(layer F.Cu", "(layer Top").replace("(layer B.Cu", "(layer Bottom")
    assert _constraints(tmp_path, aliased, pcb=PCB_ALIASED)["layers"] == ["F.Cu", "B.Cu"]
    with pytest.raises(DsnRuleError, match="'Top' is not a copper layer"):
        _constraints(tmp_path, aliased, pcb=PCB)


def test_ses_converter_writes_kicad_layer_names(tmp_path):
    from methods.baselines.rule_based._converters import convert_ses
    base = tmp_path / "base.kicad_pcb"
    base.write_text(PCB_ALIASED.replace("\t(layers", '\t(net 1 "VCC")\n\t(layers'))
    ses = tmp_path / "b.ses"
    ses.write_text('(session b (routes (resolution um 10) (network_out (net VCC '
                   '(wire (path Top 2032 0 0 10000 0))))))')
    out = tmp_path / "out.kicad_pcb"
    stats = convert_ses(ses, base, out)
    assert stats["tracks_written"] == 1 and '(layer "F.Cu")' in out.read_text()
    ses.write_text(ses.read_text().replace("path Top", "path Inner9"))
    with pytest.raises(ValueError, match="'Inner9' is not a copper layer"):
        convert_ses(ses, base, out)


def test_runner_hands_krt_a_board_with_no_default_class_net(classes, tmp_path):
    # Every net in Power: the DSN has no Default block; KRT's base comes from the .kicad_pro Default.
    pro = _pro_with(lambda ns: ns.update(
        netclass_patterns=[{"pattern": n, "netclass": "Power"} for n in _NETS]))
    got = _constraints(tmp_path, classes.fix_dsn_classes(DSN, pro)[0], pro)
    assert (got["track_width"], got["clearance"], got["class_clearances_differ"]) == (0.2032, 0.1524, True)
    assert got["non_default_net_widths"] == {n: 0.5 for n in _NETS}
    assert got["class_vias_differ"] is True     # Power's 0.8/0.5, KRT gets Default's 0.6/0.4


def _reason(message):
    for key in ("also matches", "bus pattern", "KiCad merges"):
        if key in message:
            return key
    return message


@pytest.mark.parametrize("nets, patterns", [
    (["VCC", "VCC2"], [("VCC*", "Power")]),
    (["3.3V", "3x3V"], [("3.3V", "Power")]),
    (["D0"], [("D[0..7]", "Power")]),
    (["/5(**)", "/6(**)", "+5V", "Net-(D1-Pad1)", "/~{RST}"],
     [("/5(**)", "Power"), ("+5V", "Power"), ("/~{RST}", "Power")]),
    (["VCC"], [("VCC", "Power"), ("VCC", "Default")]),
])
def test_generator_and_runner_read_class_assignments_alike(classes, nets, patterns):
    """The runner keeps a BSD-side copy of the generator's pattern reading: the same mapping, or
    a refusal for the same reason."""
    from methods.baselines.rule_based.run_rule_based_routers import DsnRuleError, _pro_net_classes
    pro = {"meta": {"version": 3}, "net_settings": {
        "meta": {"version": 4}, "classes": [{"name": "Default"}, {"name": "Power"}],
        "netclass_patterns": [{"pattern": p, "netclass": c} for p, c in patterns]}}
    try:
        read = classes.parse_pro_classes(pro, source="board.kicad_pro")
        want = classes.resolve_net_to_class(nets, {"Default": {}, "Power": {}}, read.patterns)
    except ValueError as exc:
        want = ("refused", _reason(str(exc)))
    try:
        got = _pro_net_classes(pro, "board.kicad_pro", nets)[0]
    except DsnRuleError as exc:
        got = ("refused", _reason(str(exc)))
    assert got == want


@pytest.mark.parametrize("dia, drill, span", [(0.6, 0.4, "0-1"), (0.6858, 0.3302, "0-1"),
                                              (1.234567, 0.599999, "0-3"), (2.5, 2.3, "0-0")])
def test_generator_and_runner_name_vias_alike(classes, dia, drill, span):
    from methods.baselines.rule_based.run_rule_based_routers import _pcbnew_via_name
    assert _pcbnew_via_name(dia, drill, span) == classes.via_padstack_name(dia, drill, span)


def test_krt_rust_cache_is_keyed_by_the_krt_version(tmp_path, monkeypatch):
    krt_src = REPO / "methods" / "baselines" / "rule_based" / "krt" / "src"
    saved = sys.modules.pop("krt_runner", None)
    monkeypatch.syspath_prepend(str(krt_src))
    try:
        import krt_runner
        (tmp_path / "rust_router").mkdir()
        (tmp_path / "rust_router" / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "0.19.0"\n')
        monkeypatch.setenv("KRT_ROOT", str(tmp_path))
        assert krt_runner._grid_router_version() == "0.19.0"
    finally:
        sys.modules.pop("krt_runner", None)
        if saved is not None:
            sys.modules["krt_runner"] = saved


def test_generator_and_runner_share_the_kicad9_format(classes):
    from methods.baselines.rule_based.run_rule_based_routers import _kicad9_pro_format
    fmt = classes.KICAD9_PRO_FORMAT
    assert _kicad9_pro_format() == (fmt["project_meta_version"], fmt["net_settings_meta_version"]) == (3, 4)


@pytest.mark.parametrize("edit, found", [
    (lambda pro: pro.pop("meta"), "(None, 4)"),
    (lambda pro: pro["net_settings"].pop("meta"), "(3, None)"),
    (lambda pro: pro["net_settings"]["meta"].update(version=3), "(3, 3)"),
    (lambda pro: pro["net_settings"]["meta"].update(version=5), "(3, 5)"),
    (lambda pro: pro["meta"].update(version=2), "(2, 4)"),
])
def test_generator_and_runner_refuse_a_pro_not_in_kicad9_format(classes, tmp_path, edit, found):
    from methods.baselines.rule_based.run_rule_based_routers import DsnRuleError
    pro = copy.deepcopy(PRO)
    edit(pro)
    message = re.escape(f"{found}, only the KiCad 9 format (3, 4) is supported")
    with pytest.raises(ValueError, match="board_7.kicad_pro.*" + message):
        classes.parse_pro_classes(pro, source="board_7.kicad_pro")
    dsn, pro_path = _files(tmp_path, classes.fix_dsn_classes(DSN, PRO)[0], pro)
    from methods.baselines.rule_based.run_rule_based_routers import _parse_dsn_constraints
    with pytest.raises(DsnRuleError, match=re.escape(str(pro_path)) + ".*" + message):
        _parse_dsn_constraints(dsn, pro_path, tmp_path / "board.kicad_pcb")


def _pro_with_listed_net():
    """PRO with RX.1 listed only in Power's `nets` (no pattern): KiCad 9 leaves it in Default."""
    pro = copy.deepcopy(PRO)
    pro["net_settings"]["netclass_patterns"].remove({"pattern": "RX.1", "netclass": "Power"})
    pro["net_settings"]["classes"][1]["nets"] = ["RX.1", "VCC"]   # VCC is also a pattern
    pro["net_settings"]["classes"][0]["nets"] = ["GND"]   # Default: ignoring it changes nothing
    return pro


def test_generator_names_board_and_class_for_a_nets_entry_kicad_9_does_not_read(classes):
    read = classes.parse_pro_classes(_pro_with_listed_net(), source="b/board.kicad_pro")
    assert len(read.warnings) == 1
    assert "b/board.kicad_pro" in read.warnings[0] and "'Power'" in read.warnings[0]
    assert "RX.1" in read.warnings[0] and "VCC" not in read.warnings[0]
    new_dsn, summary = classes.fix_dsn_classes(DSN, read)
    assert summary["warnings"] == read.warnings
    assert "RX.1" not in _class_block(new_dsn, "Power")


def test_generator_board_and_run_summary_carry_the_warnings(generator, tmp_path, monkeypatch, capsys):
    res, _ = _run_board(generator, tmp_path, monkeypatch, _pro_with_listed_net())
    assert res["status"] == "ok" and len(res["warnings"]) == 1
    assert "'Power'" in res["warnings"][0] and "1 .kicad_pro warning" in res["msg"]
    monkeypatch.setattr(generator, "process_folder" if hasattr(generator, "process_folder")
                        else "process_board", lambda *a, **k: dict(res))
    if hasattr(generator, "process_folder"):
        argv = ["x", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst2")]
        (tmp_path / "src" / "b" / f"{generator.INPUT_STEM}.kicad_pcb").write_text("")
    else:
        (tmp_path / "src" / "test").mkdir()
        (tmp_path / "src" / "test" / "board_00000.kicad_pcb").write_text("")
        argv = ["x", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst2"), "--splits", "test"]
    monkeypatch.setattr(sys, "argv", argv)
    capsys.readouterr()
    generator.main()
    out = capsys.readouterr().out
    assert ".kicad_pro warnings (1; pass --report <file> to keep all of them)" in out
    assert res["warnings"][0] in out


def test_runner_names_board_and_class_for_a_nets_entry_kicad_9_does_not_read(classes, tmp_path, caplog):
    pro = _pro_with_listed_net()
    got = _constraints(tmp_path, classes.fix_dsn_classes(DSN, pro)[0], pro)
    assert "RX.1" not in got["non_default_net_widths"]
    # The board's warnings travel with its constraints into the result (and so the manifest).
    assert len(got["warnings"]) == 1
    w = got["warnings"][0]
    assert str(tmp_path / "board.kicad_pro") in w and "'Power'" in w and "RX.1" in w and "GND" not in w



def test_runner_stops_when_krt_cannot_be_prepared(tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb

    def broken():
        raise RuntimeError("cargo not on PATH")

    (tmp_path / "route.py").write_text("")
    monkeypatch.setitem(sys.modules, "krt_runner", SimpleNamespace(
        get_krt_root=lambda: tmp_path, run_route=None, _ensure_grid_router_so=broken))
    monkeypatch.setenv("KRT_ALLOW_UNPINNED", "1")   # the stub tree is not a git checkout
    with pytest.raises(rb.KrtNotFound, match="cargo not on PATH"):
        rb.run_one_seed([], baseline="krt", dataset="pcbench", grid=None, seed=0,
                        seed_dir=tmp_path / "seed0", timeout_s=1, workers=1, do_eval=False,
                        reward_config=None, check_angle=45, via_prohibit=False, angle_90=False,
                        max_passes=1, extra_env={}, use_gpu=False)


def test_runner_stops_without_the_shared_format_file(tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    monkeypatch.setattr(rb, "_PROJECT_ROOT", tmp_path)
    rb._kicad9_pro_format.cache_clear()
    try:
        with pytest.raises(rb.RunSetupError, match="kicad9_pro_format.json.*engine/ submodule"):
            rb.run_one_seed([], baseline="freerouting", dataset="pcbench", grid=None, seed=0,
                            seed_dir=tmp_path / "seed0", timeout_s=1, workers=1, do_eval=False,
                            reward_config=None, check_angle=45, via_prohibit=False,
                            angle_90=False, max_passes=1, extra_env={}, use_gpu=False)
    finally:
        rb._kicad9_pro_format.cache_clear()


def test_runner_summary_counts_a_board_that_failed_this_run(tmp_path, monkeypatch):
    """A board that scored before but fails now: its old eval log is gone and the summary
    counts it as a failure, not as the earlier success."""
    from methods.baselines.rule_based import run_rule_based_routers as rb
    seed_dir = tmp_path / "seed0"
    logs = seed_dir / "eval" / "logs"
    logs.mkdir(parents=True)
    algo = rb.algorithm_tag("freerouting", "pcbench", grid=None)
    stale = logs / f"A_{algo}.json"
    stale.write_text(json.dumps({"success": True, "clean_pass": True}))
    other = logs / f"B_{algo}.json"           # a board of an earlier, larger run
    other.write_text(json.dumps({"success": True, "clean_pass": True}))
    (seed_dir / "eval" / "summary.json").write_text(json.dumps({"n_ok": 1}))
    monkeypatch.setattr(rb, "prepare_run", lambda baseline: None)
    monkeypatch.setattr(rb, "route_board", lambda baseline, board, seed, *a, **k: rb.BoardResult(
        board.board_id, seed, "error", 0.0, error="current route failed"))

    (row,) = rb.run_one_seed([SimpleNamespace(board_id="A")], baseline="freerouting",
                             dataset="pcbench", grid=None, seed=0, seed_dir=seed_dir, timeout_s=1,
                             workers=1, do_eval=True, reward_config=None, check_angle=45,
                             via_prohibit=False, angle_90=False, max_passes=1, extra_env={},
                             use_gpu=False)

    assert row["status"] == "error" and not stale.exists() and not other.exists()
    summary = json.loads((seed_dir / "eval" / "summary.json").read_text())
    assert (summary["n_total"], summary["n_ok"], summary["n_fail"]) == (1, 0, 1)
    assert summary["success_rate"] == 0.0 and summary["clean_pass_rate"] == 0.0
    assert summary["mean"]["routability"] == 0.0


def test_default_nets_entry_a_pattern_moves_elsewhere_is_warned(classes, tmp_path):
    """A Default `nets` entry a pattern assigns to another class disagrees with KiCad's reading."""
    pro = copy.deepcopy(PRO)
    pro["net_settings"]["classes"][0]["nets"] = ["VCC", "GND"]   # VCC: pattern → Power; GND: none
    read = classes.parse_pro_classes(pro, source="b.kicad_pro")
    assert len(read.warnings) == 1 and "'Default'" in read.warnings[0] and "VCC" in read.warnings[0]
    got = _constraints(tmp_path, classes.fix_dsn_classes(DSN, pro)[0], pro)
    assert len(got["warnings"]) == 1 and "VCC" in got["warnings"][0] and "GND" not in got["warnings"][0]


def test_run_summaries_show_the_first_and_last_20_warnings(classes, capsys, caplog):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    warnings = [f"w{i}" for i in range(50)]
    classes.print_pro_warnings([{"warnings": warnings}], report="r.json")
    out = capsys.readouterr().out
    assert "(50; all of them in r.json)" in out and "w19" in out and "w30" in out
    assert "w20" not in out and "... 10 more ..." in out
    with caplog.at_level("WARNING", logger="run_rule_based_routers"):
        rb.log_pro_warnings(warnings, where="manifest.json")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "50; all of them in manifest.json" in text and "w20" not in text and "w49" in text



def _fake_freerouting(monkeypatch, rb, ses_text):
    """subprocess.run stand-in for FreeRouting's JVM: writes ``ses_text`` to the -do path."""
    def run(cmd, **_):
        Path(cmd[cmd.index("-do") + 1]).write_text(ses_text)
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(rb.subprocess, "run", run)


_SES = ('(session b (routes (resolution um 10) (network_out (net VCC '
        '(wire (path {layer} 2032 0 0 10000 0))))))')


@pytest.mark.parametrize("ses, load_issue, status, message", [
    (_SES.format(layer="Top"), None, "ok", ""),
    (_SES.format(layer="Top").replace("net VCC", "net NOPE"), None, "error", "SES nets not on the board"),
    ('(session b (routes (resolution um 10) (network_out)))', None, "error", "added no copper"),
    (_SES.format(layer="Top"), "KiCad cannot load the routed board: x", "error", "KiCad cannot load"),
])
def test_freerouting_result_reflects_whether_the_board_was_routed(classes, tmp_path, monkeypatch,
                                                                  ses, load_issue, status, message):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    board = _board(tmp_path, DSN)
    board.raw_pcb.write_text(PCB_ALIASED.replace("\t(layers", '\t(net 1 "VCC")\n\t(layers'))
    _fake_freerouting(monkeypatch, rb, ses)
    monkeypatch.setattr(rb, "_kicad_load_issue", lambda pcb, pro: load_issue)
    pro = _pro_with_listed_net()
    board.pro.write_text(json.dumps(pro))
    board.dsn.write_text(classes.fix_dsn_classes(DSN, pro)[0])
    res = rb._route_freerouting(board, 0, tmp_path, tmp_path, "fr", 60, False, False, 1, {})
    assert res.status == status and message in res.error
    assert len(res.warnings) == 1       # the board's .kicad_pro warning travels on every path
    if status == "ok":
        assert '(layer "F.Cu")' in Path(res.routed_pcb).read_text()


def test_krt_wall_time_is_the_router_s_not_the_checks(written, tmp_path, monkeypatch):
    import time as _time
    from methods.baselines.rule_based import run_rule_based_routers as rb

    def run_route(args, **_):
        Path(args[1]).write_text(ROUTED)
        return SimpleNamespace(stdout=_KRT_APPLIED, stderr="", returncode=0)

    _stub_krt(monkeypatch, tmp_path, run_route)
    monkeypatch.setattr(rb, "_kicad_load_issue", lambda pcb, pro: _time.sleep(0.5))
    res = rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit=None)
    assert res.status == "ok" and res.elapsed_routing_s < 0.4


def test_eval_driver_stops_without_the_shared_format_file(tmp_path, monkeypatch):
    driver = _load(REPO / "eval" / "rollout" / "rule_based.py", "eval_rollout_rule_based_under_test")
    rb = driver.rb
    monkeypatch.setattr(rb, "_PROJECT_ROOT", tmp_path)
    rb._kicad9_pro_format.cache_clear()
    try:
        with pytest.raises(rb.RunSetupError, match="kicad9_pro_format.json"):
            driver.run_one_seed([], baseline="freerouting", dataset="pcbench", grid=None, seed=0,
                                seed_dir=tmp_path / "s", timeout_s=1, workers=1, via_prohibit=False,
                                angle_90=False, max_passes=1, use_gpu=False, deterministic=True)
    finally:
        rb._kicad9_pro_format.cache_clear()


def test_layer_names_refuse_an_alias_that_names_two_layers():
    from methods.baselines.rule_based._converters.ses_to_kicadpcb import pcb_layer_names
    swapped = PCB.replace('"F.Cu" signal)', '"F.Cu" signal "B.Cu")')
    with pytest.raises(ValueError, match="names both"):
        pcb_layer_names(swapped)



def test_runner_engine_check_stops_the_run(tmp_path, monkeypatch):
    import pcb_world.engine as engine
    from pcb_world.engine.provenance import RouterProvenanceError
    from methods.baselines.rule_based import run_rule_based_routers as rb

    def mismatch():
        raise RouterProvenanceError("built from other C++")

    monkeypatch.setattr(engine, "engine_available", lambda: True)
    monkeypatch.setattr(engine, "ensure_router_provenance", mismatch)
    with pytest.raises(rb.RunSetupError, match="other C\\+\\+"):
        rb._require_engine()
    monkeypatch.setattr(engine, "ensure_router_provenance", lambda: None)
    rb._require_engine()
    monkeypatch.setattr(engine, "engine_available", lambda: False)
    with pytest.raises(rb.RunSetupError, match="not built"):
        rb._require_engine()


def test_kicad_load_retries_an_engine_server_crash_once(tmp_path, monkeypatch):
    import pcb_world.engine as engine
    from pcb_world.engine.router_client import EngineServerCrashed
    from methods.baselines.rule_based import run_rule_based_routers as rb
    calls = []

    class Flaky:
        @classmethod
        def for_board_file(cls, *a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise EngineServerCrashed("engine server startup timed out")
            return cls()

        def close(self):
            pass

    monkeypatch.setattr(engine, "KiCadEngine", Flaky)
    assert _REAL_LOAD_ISSUE(tmp_path / "b.kicad_pcb", tmp_path / "b.kicad_pro") is None and len(calls) == 2


def test_krt_pin_matches_fetch_baselines():
    from methods.baselines.rule_based import run_rule_based_routers as rb
    script = (REPO / "tools" / "setup" / "fetch_baselines.sh").read_text()
    assert f'KRT_COMMIT="{rb.KRT_PINNED_COMMIT}"' in script


def test_freerouting_run_needs_its_jar(tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    monkeypatch.setattr(rb, "FREEROUTING_JAR", tmp_path / "missing.jar")
    with pytest.raises(rb.RunSetupError, match="fetch_baselines.sh"):
        rb.prepare_run("freerouting")


def test_ses_converter_reads_kicad_escaped_net_names(tmp_path):
    from methods.baselines.rule_based._converters.ses_to_kicadpcb import parse_net_map
    assert parse_net_map('(net 3 "A\\\\B")\n(net 4 "say \\"hi\\"")') == {"A\\B": 3, 'say "hi"': 4}


def test_krt_output_left_by_an_earlier_run_is_not_reused(written, tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    (tmp_path / "b_krt.kicad_pcb").write_text(ROUTED)          # stale output from an earlier run
    _stub_krt(monkeypatch, tmp_path,
              lambda args, **_: SimpleNamespace(stdout=_KRT_APPLIED, stderr="", returncode=0))
    res = rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit=None)
    assert res.status == "error" and res.error == "krt rc=0"


def test_ses_converter_keeps_non_ascii_names_and_refuses_unsized_vias(tmp_path):
    from methods.baselines.rule_based._converters import convert_ses
    base = tmp_path / "base.kicad_pcb"
    base.write_text(PCB.replace("\t(layers", '\t(net 1 "Net-(µC-Pad1)")\n\t(layers'))
    ses = tmp_path / "b.ses"
    ses.write_text('(session b (routes (resolution um 10) (network_out (net "Net-(µC-Pad1)" '
                   '(wire (path F.Cu 2032 0 0 10000 0))))))')
    out = tmp_path / "out.kicad_pcb"
    assert convert_ses(ses, base, out)["unmapped_nets"] == []
    ses.write_text(ses.read_text().replace("(wire", '(via "MyVia" 0 0) (wire'))
    with pytest.raises(ValueError, match="'MyVia' does not name its diameter and drill"):
        convert_ses(ses, base, out)


# ── run-level contract: a board's own failure is its row; setup failures stop the run ──

def test_unexpected_board_exception_becomes_that_board_s_error_row(tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb

    def boom(*a, **k):
        raise OSError("disk hiccup reading the DSN")

    monkeypatch.setattr(rb, "_route_krt", boom)
    board = SimpleNamespace(board_id="b", raw_pcb=tmp_path / "b.kicad_pcb", pro=tmp_path / "b.kicad_pro")
    r = rb._process_one_board(dict(board=board, seed=0, baseline="krt", raw_dir=tmp_path,
                                   routed_dir=tmp_path, eval_logs_dir=tmp_path, algo="krt",
                                   timeout_s=1, do_eval=False, reward_config=None, check_angle=45,
                                   via_prohibit=False, angle_90=False, max_passes=1, extra_env={},
                                   use_gpu=False, krt_commit=None))
    assert r["status"] == "error" and "OSError: disk hiccup" in r["error"]


def test_eval_driver_board_exception_becomes_that_board_s_row(tmp_path, monkeypatch):
    driver = _load(REPO / "eval" / "rollout" / "rule_based.py", "eval_rollout_rule_based_under_test")

    def boom(*a, **k):
        raise OSError("disk hiccup")

    monkeypatch.setattr(driver.rb, "_route_krt", boom)
    board = SimpleNamespace(board_id="b", raw_pcb=tmp_path / "b.kicad_pcb", pro=tmp_path / "b.kicad_pro")
    row, _, _ = driver._route_and_emit(dict(board=board, idx=0, seed=0, baseline="krt", raw_dir=tmp_path,
                                            artifacts_dir=tmp_path, algo="krt", timeout_s=1,
                                            via_prohibit=False, angle_90=False, max_passes=1,
                                            use_gpu=False, deterministic=True, krt_commit=None))
    assert row["status"] == "crashed" and "OSError: disk hiccup" in row["error"]


def _fake_java(tmp_path, monkeypatch, version_line):
    java = tmp_path / "bin" / "java"
    java.parent.mkdir()
    java.write_text(f"#!/bin/sh\necho '{version_line}' >&2\n")
    java.chmod(0o755)
    monkeypatch.setenv("PATH", f"{java.parent}:/usr/bin:/bin")


@pytest.mark.parametrize("version_line, ok", [
    ('openjdk version "21.0.2" 2024-01-16', True),
    ('openjdk version "17.0.9" 2023-10-17', False),
    ('java version "1.8.0_392"', False),
])
def test_freerouting_needs_java_21(tmp_path, monkeypatch, version_line, ok):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    jar = tmp_path / "fr.jar"
    jar.write_text("")
    monkeypatch.setattr(rb, "FREEROUTING_JAR", jar)
    _fake_java(tmp_path, monkeypatch, version_line)
    if ok:
        rb.prepare_run("freerouting")
    else:
        with pytest.raises(rb.RunSetupError, match="needs Java 21"):
            rb.prepare_run("freerouting")


def test_freerouting_without_java_says_so(tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    jar = tmp_path / "fr.jar"
    jar.write_text("")
    monkeypatch.setattr(rb, "FREEROUTING_JAR", jar)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    with pytest.raises(rb.RunSetupError, match="no java on PATH.*openjdk 21"):
        rb.prepare_run("freerouting")


@pytest.mark.parametrize("commit, allow, ok", [
    ("561f1584577ea86d99e223ab27a2cd1cdbfa8bb3", None, True),
    ("d9557ad10e8fa51fcf641749ae1cc9dca66e9648", None, False),
    ("561f1584577ea86d99e223ab27a2cd1cdbfa8bb3+dirty", None, False),
    (None, None, False),
    ("d9557ad10e8fa51fcf641749ae1cc9dca66e9648", "1", True),
])
def test_krt_must_be_the_pinned_commit_unless_allowed(tmp_path, monkeypatch, commit, allow, ok):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    _stub_krt(monkeypatch, tmp_path, None)
    monkeypatch.setattr(rb, "_krt_commit", lambda root: commit)
    if allow is None:
        monkeypatch.delenv("KRT_ALLOW_UNPINNED", raising=False)
    else:
        monkeypatch.setenv("KRT_ALLOW_UNPINNED", allow)
    if ok:
        assert rb.prepare_run("krt") == commit
    else:
        with pytest.raises(rb.RunSetupError, match="KRT_ALLOW_UNPINNED=1"):
            rb.prepare_run("krt")


def test_krt_sidecar_records_an_allowed_unpinned_tree(written, tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb

    def run_route(args, **_):
        Path(args[1]).write_text(ROUTED)
        return SimpleNamespace(stdout=_KRT_APPLIED, stderr="", returncode=0)

    _stub_krt(monkeypatch, tmp_path, run_route)
    monkeypatch.setenv("KRT_ALLOW_UNPINNED", "1")
    rb._route_krt(_board(tmp_path, written), 0, tmp_path, tmp_path, "krt", 60, krt_commit="abc")
    record = json.loads((tmp_path / "b.krt_run.json").read_text())
    assert record["krt_commit"] == "abc" and record["krt_pinned_commit"] == rb.KRT_PINNED_COMMIT
    assert record["krt_unpinned_allowed"] is True


@pytest.mark.parametrize("driver_name", ["runner", "eval"])
def test_a_setup_error_inside_a_board_still_stops_the_run(tmp_path, monkeypatch, driver_name):
    from methods.baselines.rule_based import run_rule_based_routers as runner
    driver = (_load(REPO / "eval" / "rollout" / "rule_based.py", "eval_rollout_rule_based_under_test")
              if driver_name == "eval" else None)
    rb = driver.rb if driver else runner

    def setup_failure(*a, **k):
        raise rb.RunSetupError("engine gone")

    monkeypatch.setattr(rb, "_route_krt", setup_failure)
    board = SimpleNamespace(board_id="b", raw_pcb=tmp_path / "b.kicad_pcb", pro=tmp_path / "b.kicad_pro")
    job = dict(board=board, seed=0, baseline="krt", raw_dir=tmp_path, algo="krt", timeout_s=1,
               via_prohibit=False, angle_90=False, max_passes=1, use_gpu=False, krt_commit=None)
    with pytest.raises(rb.RunSetupError, match="engine gone"):
        if driver:
            driver._route_and_emit(dict(job, idx=0, artifacts_dir=tmp_path, deterministic=True))
        else:
            runner._process_one_board(dict(job, routed_dir=tmp_path, eval_logs_dir=tmp_path, do_eval=False,
                                           reward_config=None, check_angle=45, extra_env={}))


def test_krt_commit_is_only_the_krt_tree_s_own(tmp_path):
    import subprocess as sp
    from methods.baselines.rule_based import run_rule_based_routers as rb
    outer = tmp_path / "outer"
    (outer / "krt").mkdir(parents=True)
    sp.run(["git", "init", "-q", str(outer)], check=True)
    sp.run(["git", "-C", str(outer), "-c", "user.email=a@b", "-c", "user.name=a", "-c",
            "commit.gpgsign=false", "commit", "-q", "--no-verify", "--allow-empty", "-m", "x"], check=True)
    # A KRT copy without .git inside another checkout: the enclosing repo's sha is not KRT's.
    assert rb._krt_commit(str(outer / "krt")) is None
    assert rb._krt_commit(str(outer)) is not None


def test_unreadable_krt_commit_says_so(tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    _stub_krt(monkeypatch, tmp_path, None)
    monkeypatch.setattr(rb, "_krt_commit", lambda root: None)
    monkeypatch.delenv("KRT_ALLOW_UNPINNED", raising=False)
    with pytest.raises(rb.RunSetupError, match="cannot be read.*KRT_ALLOW_UNPINNED=1"):
        rb.prepare_run("krt")


def test_java_that_cannot_run_is_a_setup_error(tmp_path, monkeypatch):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    jar = tmp_path / "fr.jar"
    jar.write_text("")
    monkeypatch.setattr(rb, "FREEROUTING_JAR", jar)
    java = tmp_path / "bin" / "java"
    java.parent.mkdir()
    java.write_text("not an executable format")
    java.chmod(0o755)
    monkeypatch.setenv("PATH", str(java.parent))
    with pytest.raises(rb.RunSetupError, match="cannot run"):
        rb.prepare_run("freerouting")


def test_engine_server_crashing_twice_is_the_board_s_result(tmp_path, monkeypatch):
    import pcb_world.engine as engine
    from pcb_world.engine.router_client import EngineServerCrashed

    calls = []

    def crash(*a, **k):
        calls.append(1)
        raise EngineServerCrashed("engine server died during construct")

    monkeypatch.setattr(engine, "KiCadEngine", SimpleNamespace(for_board_file=crash))
    assert "crashed twice" in _REAL_LOAD_ISSUE(tmp_path / "b.kicad_pcb", tmp_path / "b.kicad_pro")
    assert len(calls) == 2



def test_krt_commit_unreadable_status_is_not_reported_clean(tmp_path, monkeypatch):
    import subprocess as sp
    from methods.baselines.rule_based import run_rule_based_routers as rb
    real = sp.run

    def run(cmd, **k):
        if "status" in cmd:
            return SimpleNamespace(returncode=128, stdout="", stderr="fatal")
        return real(cmd, **k)

    sp.run(["git", "init", "-q", str(tmp_path)], check=True)
    sp.run(["git", "-C", str(tmp_path), "-c", "user.email=a@b", "-c", "user.name=a", "-c",
            "commit.gpgsign=false", "commit", "-q", "--no-verify", "--allow-empty", "-m", "x"], check=True)
    monkeypatch.setattr(rb.subprocess, "run", run)
    assert rb._krt_commit(str(tmp_path)) is None


@pytest.mark.parametrize("error", [OSError("no git"), subprocess.TimeoutExpired("git", 30)])
def test_krt_commit_is_unknown_when_git_cannot_run(tmp_path, monkeypatch, error):
    from methods.baselines.rule_based import run_rule_based_routers as rb

    def run(cmd, **k):
        raise error

    monkeypatch.setattr(rb.subprocess, "run", run)
    assert rb._krt_commit(str(tmp_path / "krt")) is None


def test_allowed_unreadable_krt_commit_is_logged_as_unknown(tmp_path, monkeypatch, caplog):
    from methods.baselines.rule_based import run_rule_based_routers as rb
    _stub_krt(monkeypatch, tmp_path, None)
    monkeypatch.setattr(rb, "_krt_commit", lambda root: None)
    monkeypatch.setenv("KRT_ALLOW_UNPINNED", "1")
    with caplog.at_level("WARNING", logger="run_rule_based_routers"):
        assert rb.prepare_run("krt") is None
    assert "cannot be read" in caplog.text and "at None" not in caplog.text
