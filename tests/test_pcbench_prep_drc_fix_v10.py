"""DRC tool failures are distinct from legitimate v10 board rejections."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import drc_fix_v10  # noqa: E402
import make_guide  # noqa: E402


@pytest.mark.parametrize("failure", ["timeout", "missing", "nonzero", "malformed"])
def test_run_drc_failures_are_never_empty_violation_pass(tmp_path, monkeypatch, failure):
    pcb = tmp_path / "board.kicad_pcb"
    monkeypatch.setattr(drc_fix_v10.kicad_tools, "kicad_cli", lambda: "kicad-cli")

    def run(*args, **kwargs):
        output = Path(args[0][args[0].index("--output") + 1])
        if failure == "timeout":
            raise drc_fix_v10.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        if failure == "malformed":
            output.write_text("{\"violations\": null}")
        elif failure == "nonzero":
            output.write_text("{\"violations\": []}")
            return SimpleNamespace(returncode=1, stdout="", stderr="tool failed")
        if failure == "missing":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(drc_fix_v10.subprocess, "run", run)
    result = drc_fix_v10.run_drc(pcb)
    assert result["status"] == "error"
    assert "error" in result


def test_guide_nonzero_with_empty_report_is_a_stage_error(tmp_path, monkeypatch):
    monkeypatch.setattr(make_guide, "_write_refill_helper", lambda: tmp_path / "helper.py")
    monkeypatch.setattr(make_guide, "_refill_zones", lambda *args: True)
    monkeypatch.setattr(make_guide.kicad_tools, "kicad_cli", lambda: "kicad-cli")

    def run(args, **kwargs):
        output = Path(args[args.index("--output") + 1])
        output.write_text("{\"violations\": []}")
        return SimpleNamespace(returncode=1, stdout="", stderr="failed")

    monkeypatch.setattr(make_guide.subprocess, "run", run)
    result = make_guide.run_drc("(kicad_pcb)", {}, tmp_path)
    assert "error" in result
    assert "status 1" in result["error"]


def test_guide_refill_failure_is_not_a_drc_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(make_guide, "_write_refill_helper", lambda: tmp_path / "helper.py")
    monkeypatch.setattr(make_guide, "_refill_zones", lambda *args: False)
    monkeypatch.setattr(make_guide.kicad_tools, "kicad_cli", lambda: "kicad-cli")
    monkeypatch.setattr(make_guide.subprocess, "run", lambda *args, **kwargs: pytest.fail("DRC must not run"))
    result = make_guide.run_drc("(kicad_pcb)", {}, tmp_path)
    assert "error" in result and "zone refill failed" in result["error"]


def test_valid_nonempty_report_is_a_drc_rejection(tmp_path, monkeypatch):
    src_root, out_root = tmp_path / "v10", tmp_path / "newdrc_v10"
    name = "source-folder"
    src = src_root / name
    src.mkdir(parents=True)
    (src / "processed_v10.kicad_pcb").write_text("(kicad_pcb)\n")
    (src / "processed_v10.kicad_pro").write_text("{}")
    monkeypatch.setattr(drc_fix_v10, "run_drc", lambda pcb: {
        "status": "ok", "returncode": 0,
        "violations": [{"type": "clearance", "description": "too close"}],
    })

    result = drc_fix_v10.process_sample(name, src_root, out_root)
    assert result["status"] == "drc_rejected"
    assert result["violation_counts"] == {"clearance": 1}
    assert not (out_root / name).exists()


def test_zero_violations_copy_v10_stems(tmp_path, monkeypatch):
    src_root, out_root = tmp_path / "v10", tmp_path / "newdrc_v10"
    name = "source-folder"
    src = src_root / name
    src.mkdir(parents=True)
    (src / "processed_v10.kicad_pcb").write_text("(kicad_pcb)\n")
    (src / "processed_v10.kicad_pro").write_text("{}")
    monkeypatch.setattr(drc_fix_v10, "run_drc", lambda pcb: {
        "status": "ok", "returncode": 0, "violations": [],
    })

    result = drc_fix_v10.process_sample(name, src_root, out_root)
    assert result["status"] == "ok"
    assert (out_root / name / "processed_v10.kicad_pcb").is_file()
    assert (out_root / name / "processed_v10.kicad_pro").is_file()


def test_historical_repair_policy_keeps_hole_clearance_floor():
    assert drc_fix_v10.SEVERITY_PATCH == {
        "solder_mask_bridge": "ignore", "drill_out_of_range": "ignore",
        "malformed_courtyard": "ignore", "courtyards_overlap": "ignore",
        "pth_inside_courtyard": "ignore", "npth_inside_courtyard": "ignore",
        "annular_width": "ignore", "copper_edge_clearance": "ignore",
        "hole_clearance": "ignore",
    }
    assert drc_fix_v10.RULES_PATCH == {
        "min_via_annular_width": 0.0,
        "min_copper_edge_clearance": 0.0,
        "min_hole_clearance": 0.0,
    }
    assert drc_fix_v10.hole_to_hole_min({"net_settings": {"classes": [
        {"name": "Default", "clearance": 0.1}]} }) == 0.1
    assert drc_fix_v10.hole_to_hole_min({"net_settings": {"classes": [
        {"name": "Default", "clearance": 0.3}]} }) == 0.25


def test_guide_parses_named_v10_nets_for_width_adjustment():
    content = (
        '(net 1 "GNDA")\n'
        '(segment\n  (start 0 0)\n  (end 1 0)\n  (width 0.4)\n  (net "GNDA")\n  (uuid "named-wide")\n)\n'
        '(segment\n  (start 1 0)\n  (end 2 0)\n  (width 0.2)\n  (net 1)\n  (uuid "numeric-narrow")\n)\n'
    )
    net_id_to_name = make_guide.extract_net_id_to_name(content)

    assert net_id_to_name == {"1": "GNDA"}
    assert make_guide.extract_net_width_range(content, net_id_to_name) == {
        "1": (0.2, 0.4),
    }
    assert make_guide.extract_uuid_to_net(content, net_id_to_name) == {
        "named-wide": "1", "numeric-narrow": "1",
    }
    narrowed = make_guide.apply_net_widths_to_tracks(
        content, {"1": 0.15}, net_id_to_name)
    assert narrowed.count("(width 0.15)") == 2

    assert make_guide.extract_net_width_range(content) == {
        "1": (0.2, 0.2),
    }
    assert make_guide.extract_uuid_to_net(content) == {
        "numeric-narrow": "1",
    }
    numeric_narrowed = make_guide.apply_net_widths_to_tracks(
        content, {"1": 0.1})
    assert numeric_narrowed.count("(width 0.1)") == 1
    assert "(width 0.4)" in numeric_narrowed


def test_guide_maps_named_nets_without_numeric_table(tmp_path, monkeypatch):
    content = """
(segment
  (start 0 0)
  (end 1 0)
  (width 0.4)
  (net "GNDA")
  (uuid "named-wide")
)
(segment
  (start 1 0)
  (end 2 0)
  (width 0.2)
  (net "GNDA")
  (uuid "named-narrow")
)
(segment
  (start 2 0)
  (end 3 0)
  (width 0.3)
  (net "0")
  (uuid "quoted-zero")
)
(segment
  (start 3 0)
  (end 4 0)
  (width 0.5)
  (net 0)
  (uuid "numeric-zero")
)
"""
    net_id_to_name = make_guide.extract_net_id_to_name(content)

    assert net_id_to_name == {
        "named:GNDA": "GNDA", "named:0": "0",
    }
    assert make_guide.extract_net_width_range(content, net_id_to_name) == {
        "named:GNDA": (0.2, 0.4), "named:0": (0.3, 0.3),
    }
    assert make_guide.extract_uuid_to_net(content, net_id_to_name) == {
        "named-wide": "named:GNDA",
        "named-narrow": "named:GNDA",
        "quoted-zero": "named:0",
        "numeric-zero": "0",
    }
    narrowed = make_guide.apply_net_widths_to_tracks(
        content, {"named:GNDA": 0.15, "named:0": 0.1}, net_id_to_name)
    assert narrowed.count("(width 0.15)") == 2
    assert narrowed.count("(width 0.1)") == 1
    assert narrowed.count("(width 0.5)") == 1

    groups = make_guide.build_groups(
        net_id_to_name, {"named:GNDA": 0.15, "named:0": 0.1},
        {}, {"Default": {}}, 0.25, [])
    assert groups[(0.1, ())] == ["0"]

    board = tmp_path / "named-only"
    board.mkdir()
    (board / "processed_v10.kicad_pcb").write_text(content)
    (board / "processed_v10.kicad_pro").write_text(
        '{"net_settings": {"classes": [{"name": "Default", '
        '"track_width": 0.25}]}}')
    monkeypatch.setattr(make_guide, "run_drc", lambda *args, **kwargs: {
        "counts": {}, "violations": [],
    })
    result = make_guide.process_board(board)
    assert result["status"] == "pass"
    assert result["n_nets_at_max"] == 2

def test_guide_balances_parentheses_in_named_net_before_next_segment():
    content = """
(segment
  (start 0 0)
  (end 1 0)
  (width 0.4)
  (net "Net-(U1-Pad2)")
  (uuid "named-wide")
)
(segment
  (start 1 0)
  (end 2 0)
  (width 0.2)
  (net "Net-(U1-Pad2)")
  (uuid "named-narrow")
)
"""
    net_id_to_name = make_guide.extract_net_id_to_name(content)
    net_id = "named:Net-(U1-Pad2)"

    assert net_id_to_name == {net_id: "Net-(U1-Pad2)"}
    assert make_guide.extract_net_width_range(content, net_id_to_name) == {
        net_id: (0.2, 0.4),
    }
    assert make_guide.extract_uuid_to_net(content, net_id_to_name) == {
        "named-wide": net_id, "named-narrow": net_id,
    }
    narrowed = make_guide.apply_net_widths_to_tracks(
        content, {net_id: 0.15}, net_id_to_name)
    assert narrowed.count("(width 0.15)") == 2


def test_tag_scanner_ignores_quoted_block_text():
    content = """
(net 1 "SIG")
(comment "fake (segment (width 9) (net 1) (uuid deadbeef)) (arc (width 8) (net 1) (uuid cafebabe)) (via (net 1) (uuid feedface))")
(segment (start 0 0) (end 1 0) (width 0.4) (net 1) (uuid "segment-real"))
(arc (start 1 0) (mid 1.5 0.5) (end 2 0) (width 0.2) (net 1) (uuid "arc-real"))
(via (at 2 0) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net 1) (uuid "via-real"))
"""
    net_id_to_name = make_guide.extract_net_id_to_name(content)

    assert make_guide.extract_net_width_range(content, net_id_to_name) == {
        "1": (0.2, 0.4),
    }
    assert make_guide.extract_uuid_to_net(content, net_id_to_name) == {
        "segment-real": "1", "arc-real": "1", "via-real": "1",
    }
    rewritten = make_guide.apply_net_widths_to_tracks(
        content, {"1": 0.15}, net_id_to_name)
    assert rewritten.count("(width 0.15)") == 2
    assert "(width 9)" in rewritten and "(width 8)" in rewritten

    unrouted = make_guide.remove_routing(content)
    assert "(segment (width 9) (net 1) (uuid deadbeef))" in unrouted
    assert "(arc (width 8) (net 1) (uuid cafebabe))" in unrouted
    assert "(via (net 1) (uuid feedface))" in unrouted
    assert "segment-real" not in unrouted
    assert "arc-real" not in unrouted
    assert "via-real" not in unrouted