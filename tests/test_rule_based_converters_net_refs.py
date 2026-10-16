"""Regressions for named and numeric KiCad net references in route converters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "methods" / "baselines" / "rule_based" / "_converters"))
import ors_to_kicadpcb  # noqa: E402
import ses_to_kicadpcb  # noqa: E402

_NET_NAME = 'NET "quoted"' + chr(92)
_NET_QUOTED = json.dumps(_NET_NAME, ensure_ascii=False)


def _write_base_pcb(path: Path, numeric_id: int | None, net_name: str = _NET_NAME) -> None:
    quoted = json.dumps(net_name, ensure_ascii=False)
    if numeric_id is None:
        declarations = f'  (net 0 "")\n  (net {quoted})'
    else:
        declarations = f'  (net 0 "")\n  (net {numeric_id} {quoted})'
    path.write_text(
        f"(kicad_pcb\n{declarations}\n)\n",
        encoding="utf-8",
    )

def _assert_net_ref(output: Path, expected_ref: str) -> None:
    assert f"(net {expected_ref})" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "numeric_id, net_name, expected_ref",
    [
        pytest.param(None, _NET_NAME, _NET_QUOTED, id="name-only"),
        pytest.param(7, _NET_NAME, "7", id="legacy-numeric"),
        pytest.param(None, "电源", json.dumps("电源", ensure_ascii=False), id="unicode-name-only"),
    ],
)
def test_ses_converter_preserves_base_net_reference(tmp_path, numeric_id, net_name, expected_ref):
    base = tmp_path / "base.kicad_pcb"
    session = tmp_path / "route.ses"
    output = tmp_path / "routed.kicad_pcb"
    quoted = json.dumps(net_name, ensure_ascii=False)
    _write_base_pcb(base, numeric_id, net_name)
    session.write_text(
        "(session\n"
        "  (routes\n"
        "    (resolution um 10)\n"
        "    (network_out\n"
        f"      (net {quoted}\n"
        "        (wire (path F.Cu 100 0 0 10000 0))\n"
        "      )\n"
        "    )\n"
        "  )\n"
        ")\n",
        encoding="utf-8",
    )

    stats = ses_to_kicadpcb.convert(session, base, output)

    _assert_net_ref(output, expected_ref)
    assert stats["tracks_written"] == 1
    assert stats["nets_mapped"] == 1
    assert stats["unmapped_nets"] == []


@pytest.mark.parametrize(
    "numeric_id, expected_ref",
    [(None, _NET_QUOTED), (7, "7")],
    ids=["name-only", "legacy-numeric"],
)
def test_ors_converter_preserves_base_net_reference(tmp_path, numeric_id, expected_ref):
    base = tmp_path / "base.kicad_pcb"
    ors = tmp_path / "route.ORS"
    output = tmp_path / "routed.kicad_pcb"
    _write_base_pcb(base, numeric_id)
    ors.write_text(json.dumps({
        "geometry": {
            "by_net": {
                _NET_NAME: {
                    "tracks": [{
                        "start": {"x": 1.0, "y": 2.0},
                        "end": {"x": 3.0, "y": 4.0},
                        "width": 0.2,
                        "layer": "F.Cu",
                    }],
                },
            },
        },
    }), encoding="utf-8")

    stats = ors_to_kicadpcb.convert(ors, base, output)

    _assert_net_ref(output, expected_ref)
    assert stats["tracks_written"] == 1
    assert stats["nets_mapped"] == 1
