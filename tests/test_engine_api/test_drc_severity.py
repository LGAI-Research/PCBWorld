"""Severity-aware DRCUtils API tests (pure Python, no compiled engine).

Covers the error/warning split added alongside the DRC state-token work:
``get_error_count``, ``get_warning_count``, ``get_error_counts_by_net``,
``get_warning_counts_by_net``, and the taxonomy classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.engine.drc import (  # noqa: E402
    DRC_SEVERITY_ERROR,
    DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
    DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS,
    DRC_SEVERITY_MODE_ERRORS_ONLY,
    DRC_SEVERITY_WARNING,
    DRCE_DANGLING_TRACK,
    DRCE_DANGLING_VIA,
    DRCE_NET_CONFLICT,
    DRCUtils,
    classify_violation_type_id,
    violation_matches_severity_mode,
)


class _V:
    """Mock DRC violation object. Matches the attr surface the helper reads."""

    def __init__(
        self, error_type, severity, nets=(), x=0.0, y=0.0, layer=1,
        error_code=0,
    ):
        self.error_code = error_code
        self.error_type = error_type
        self.message = ""
        self.x_mm = x
        self.y_mm = y
        self.layer = layer
        self.net_names = list(nets)
        self.severity = severity


def test_taxonomy_known_types():
    assert classify_violation_type_id("Clearance violation") == 0
    assert classify_violation_type_id("Track width") == 1
    assert classify_violation_type_id("Track has unconnected end") == 2
    assert classify_violation_type_id("Short circuit") == 3
    assert classify_violation_type_id("Via annular ring") == 4
    # copper edge takes precedence over clearance
    assert classify_violation_type_id("Copper edge clearance") == 5
    # unmapped falls into DEFAULT bucket
    assert classify_violation_type_id("Missing connection between items") == 6
    assert classify_violation_type_id("") == 6


def test_error_and_warning_counts():
    h = DRCUtils()
    h.update([
        _V("Clearance violation", DRC_SEVERITY_ERROR, nets=["A"]),
        _V("Clearance violation", DRC_SEVERITY_ERROR, nets=["B"]),
        _V("Track has unconnected end", DRC_SEVERITY_WARNING, nets=["A"]),
        _V("Missing connection between items", DRC_SEVERITY_WARNING, nets=[]),
    ])
    assert h.get_violation_count() == 4
    assert h.get_error_count() == 2
    assert h.get_warning_count() == 2


def test_error_counts_by_net_with_orphan():
    h = DRCUtils()
    h.update([
        _V("Clearance violation", DRC_SEVERITY_ERROR, nets=["A", "B"]),
        _V("Clearance violation", DRC_SEVERITY_ERROR, nets=["A"]),
        _V("Clearance violation", DRC_SEVERITY_ERROR, nets=[]),   # orphan
        _V("Track has unconnected end", DRC_SEVERITY_WARNING, nets=["A"]),
    ])
    errs = h.get_error_counts_by_net()
    assert errs == {"A": 2, "B": 1, "<orphan>": 1}
    warns = h.get_warning_counts_by_net()
    assert warns == {"A": 1}


def test_sort_by_severity_then_distance():
    h = DRCUtils()
    h.update([
        _V("Clearance violation", DRC_SEVERITY_WARNING, x=0.0, y=0.0),   # near warning
        _V("Clearance violation", DRC_SEVERITY_ERROR, x=10.0, y=10.0),   # far error
        _V("Clearance violation", DRC_SEVERITY_ERROR, x=1.0, y=1.0),     # near error
    ])
    sorted_ = h.get_sorted(head_xy=(0.0, 0.0), k=32)
    assert [v["severity"] for v in sorted_] == [
        DRC_SEVERITY_ERROR, DRC_SEVERITY_ERROR, DRC_SEVERITY_WARNING,
    ]
    # within error group, nearer-to-head comes first
    assert sorted_[0]["x_mm"] == 1.0
    assert sorted_[1]["x_mm"] == 10.0


def test_sort_cap_k():
    h = DRCUtils()
    h.update([_V("Clearance violation", DRC_SEVERITY_ERROR) for _ in range(40)])
    assert len(h.get_sorted(head_xy=(0.0, 0.0), k=32)) == 32
    assert len(h.get_sorted(head_xy=(0.0, 0.0), k=0)) == 0


def _promoted_sample():
    return [
        _V("Clearance violation", DRC_SEVERITY_ERROR, nets=["A"], error_code=5),
        _V(
            "Track has unconnected end", DRC_SEVERITY_WARNING, nets=["B"],
            error_code=DRCE_DANGLING_TRACK,
        ),
        _V(
            "Via is not connected or connected on only one layer",
            DRC_SEVERITY_WARNING, nets=["C"], error_code=DRCE_DANGLING_VIA,
        ),
        _V(
            "Pad net doesn't match netlist", DRC_SEVERITY_WARNING, nets=["D"],
            error_code=DRCE_NET_CONFLICT,
        ),
        # warning that is NOT promoted (connection_width, code 21)
        _V("Net connection too small", DRC_SEVERITY_WARNING, nets=["E"], error_code=21),
    ]


def test_severity_mode_predicate_errors_only():
    vs = _promoted_sample()
    matched = [v for v in vs if violation_matches_severity_mode(v, DRC_SEVERITY_MODE_ERRORS_ONLY)]
    assert [v.net_names[0] for v in matched] == ["A"]


def test_severity_mode_predicate_errors_and_promoted():
    vs = _promoted_sample()
    matched = [v for v in vs if violation_matches_severity_mode(v, DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED)]
    # Error + three promoted warnings, connection_width dropped.
    assert sorted(v.net_names[0] for v in matched) == ["A", "B", "C", "D"]


def test_severity_mode_predicate_errors_and_warnings():
    vs = _promoted_sample()
    matched = [v for v in vs if violation_matches_severity_mode(v, DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS)]
    assert len(matched) == 5


def test_helper_counts_by_mode():
    h = DRCUtils()
    h.update(_promoted_sample())
    assert h.get_count_by_severity_mode(DRC_SEVERITY_MODE_ERRORS_ONLY) == 1
    assert h.get_count_by_severity_mode(DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED) == 4
    assert h.get_count_by_severity_mode(DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS) == 5
    per_net = h.get_counts_by_net_by_severity_mode(DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED)
    assert per_net == {"A": 1, "B": 1, "C": 1, "D": 1}


def test_get_sorted_respects_severity_mode():
    h = DRCUtils()
    h.update(_promoted_sample())
    # No filter → every violation appears.
    assert len(h.get_sorted(head_xy=(0.0, 0.0), k=32)) == 5
    # Errors only → the single clearance error.
    assert len(h.get_sorted(
        head_xy=(0.0, 0.0), k=32,
        severity_mode=DRC_SEVERITY_MODE_ERRORS_ONLY,
    )) == 1
    # Errors + promoted → error + 3 promoted warnings.
    promoted = h.get_sorted(
        head_xy=(0.0, 0.0), k=32,
        severity_mode=DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
    )
    assert len(promoted) == 4
    # Error still sorts first (severity desc).
    assert promoted[0]["severity"] == DRC_SEVERITY_ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
