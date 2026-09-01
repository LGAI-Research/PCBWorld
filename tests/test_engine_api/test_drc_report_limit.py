"""Regression guard: the engine's DRC reports every violation.

Upstream KiCad caps the number of reported violations per type
(``ERROR_LIMIT`` 199, ``EXTENDED_ERROR_LIMIT`` 499 for clearance / unconnected
items, ``drc_engine.cpp``) to keep the GUI marker list responsive. The patched
``engine/kicad-patches/kicad/pcbnew/drc/drc_engine.cpp`` lifts both to
``INT_MAX`` because headless callers (reward, eval metrics, the incremental
DRC baseline) count the list. A capped engine also misfiles shorts as
clearance violations once the shorting budget is spent.

The board is generated here: ``N_OPEN`` nets with two unconnected pads each
(> 499 ``unconnected_items``) and ``N_SHORT`` pairs of overlapping pads on
different nets (> 199 ``shorting_items``).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.engine.kicad_engine import KiCadEngine  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "sample_drc_violation.kicad_pcb"
N_OPEN = 600    # > EXTENDED_ERROR_LIMIT (499)
N_SHORT = 250   # > ERROR_LIMIT (199)
UNCONNECTED = "Missing connection between items"
SHORTING = "Items shorting two nets"


def _footprint(ref: str, x: float, y: float, net_id: int, net_name: str) -> str:
    return (
        f'\t(footprint "rl:pad" (layer "F.Cu") (at {x} {y})\n'
        f'\t\t(property "Reference" "{ref}" (at 0 0 0) (layer "F.SilkS") (hide yes)'
        f' (effects (font (size 1 1) (thickness 0.15))))\n'
        f'\t\t(property "Value" "pad" (at 0 0 0) (layer "F.Fab") (hide yes)'
        f' (effects (font (size 1 1) (thickness 0.15))))\n'
        f'\t\t(attr smd)\n'
        f'\t\t(pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net {net_id} "{net_name}"))\n'
        f'\t)\n'
    )


def _write_board(path: Path) -> None:
    src = FIXTURE.read_text(encoding="utf-8")
    head = src[: src.index('\t(net 0 "")')]   # header, layers, setup of a known-good board
    nets, fps = ['\t(net 0 "")'], []
    net_id = 1
    # Open nets: two pads 2 mm apart on a 4 mm grid, no copper between them.
    for i in range(N_OPEN):
        name = f"OPEN{i}"
        nets.append(f'\t(net {net_id} "{name}")')
        x, y = 10 + (i % 40) * 4, 10 + (i // 40) * 4
        fps.append(_footprint(f"A{i}", x, y, net_id, name))
        fps.append(_footprint(f"B{i}", x + 2, y, net_id, name))
        net_id += 1
    # Shorts: two pads of different nets at the same position.
    for i in range(N_SHORT):
        a, b = net_id, net_id + 1
        nets.append(f'\t(net {a} "SA{i}")')
        nets.append(f'\t(net {b} "SB{i}")')
        x, y = 200 + (i % 25) * 4, 10 + (i // 25) * 4
        fps.append(_footprint(f"C{i}", x, y, a, f"SA{i}"))
        fps.append(_footprint(f"D{i}", x, y, b, f"SB{i}"))
        net_id += 2
    outline = ('\t(gr_rect (start 0 0) (end 400 200) (stroke (width 0.1) (type default))'
               ' (layer "Edge.Cuts"))\n')
    path.write_text(head + "\n".join(nets) + "\n" + "".join(fps) + outline + ")\n",
                    encoding="utf-8")


@pytest.fixture
def big_board(tmp_path: Path) -> str:
    p = tmp_path / "report_limit.kicad_pcb"
    _write_board(p)
    return str(p)


def test_drc_reports_beyond_upstream_caps(big_board: str) -> None:
    eng = KiCadEngine(big_board, allow_default_rules=True)
    try:
        counts = Counter(v.error_type for v in eng.run_drc())
    finally:
        eng.close()
    assert counts[UNCONNECTED] == N_OPEN, counts
    assert counts[SHORTING] == N_SHORT, counts
    # Under the upstream caps these would read 499 and 199 (and the overflow
    # shorts would be reported as clearance violations instead).
    assert counts.get("Clearance violation", 0) == 0, counts
