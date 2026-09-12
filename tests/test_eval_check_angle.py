"""Unit tests for the routing-angle check in ``eval/metrics.py``.

``_check_routing_angles`` is exercised against a fake engine whose
``get_tracks()`` returns hand-crafted segments. This pins the geometry
for both modes (45-only and 90-only) without paying the cost of
spinning up ``KiCadEngine``.

The integration path through ``evaluate_one`` is not covered here — the
engine-backed end-to-end scoring is exercised by the eval suites
(``eval.metrics.evaluate_one`` / ``eval.pipeline``).
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# The scoring kernel (compute_metrics / evaluate_one / _check_routing_angles)
# lives in eval/metrics.py; load it directly from its path.
_EVAL_PCB_PATH = (
    Path(__file__).resolve().parents[1] / "eval" / "metrics.py"
)
_spec = importlib.util.spec_from_file_location("_eval_metrics_under_test", _EVAL_PCB_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

_ALLOWED_ANGLES = _mod._ALLOWED_ANGLES
_ANGLE_TOL_DEG = _mod._ANGLE_TOL_DEG
_check_routing_angles = _mod._check_routing_angles
evaluate_one = _mod.evaluate_one

from eval.aggregation import aggregate  # noqa: E402  (nested per-sample aggregator)


@dataclass
class FakeTrack:
    """Subset of RLTrackInfo used by ``_check_routing_angles``."""

    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    layer: int = 0
    net_code: int = 1
    net_name: str = "N1"


@dataclass
class FakePad:
    x_mm: float
    y_mm: float
    layer: int = 0


@dataclass
class FakeVia:
    x_mm: float
    y_mm: float
    top_layer: int = 0
    bottom_layer: int = 31  # default: spans every copper layer


class FakeEngine:
    def __init__(
        self,
        tracks: list[FakeTrack],
        pads: list[FakePad] | None = None,
        vias: list[FakeVia] | None = None,
    ) -> None:
        self._tracks = tracks
        self._pads = pads or []
        self._vias = vias or []

    def get_tracks(self) -> list[FakeTrack]:
        return self._tracks

    def get_pads(self) -> list[FakePad]:
        return self._pads

    def get_vias(self) -> list[FakeVia]:
        return self._vias


def _segments(*pts: tuple[float, float], **kw) -> list[FakeTrack]:
    """Build a polyline of FakeTracks through the given points."""
    return [
        FakeTrack(p[0], p[1], q[0], q[1], **kw)
        for p, q in zip(pts, pts[1:])
    ]


# ---------------------------------------------------------------------------
# Heuristic geometry
# ---------------------------------------------------------------------------


def test_no_tracks_means_nothing_to_check() -> None:
    r = _check_routing_angles(FakeEngine([]), _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 0
    assert r["n_violations"] == 0
    assert r["violations"] == []
    # Self-describing fields reflect the requested allowed set.
    assert r["allowed_angles_deg"] == list(_ALLOWED_ANGLES[90])
    assert r["tolerance_deg"] == _ANGLE_TOL_DEG


def test_single_segment_has_no_corner() -> None:
    """Each endpoint has only one neighbor; no 2-track joint exists."""
    eng = FakeEngine(_segments((0, 0), (1, 0)))
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 0
    assert r["n_violations"] == 0


def test_right_angle_corner_passes_in_90_mode() -> None:
    """H + V meeting at (1, 0) → measured 90° → allowed in 90 mode."""
    eng = FakeEngine(_segments((0, 0), (1, 0), (1, 1)))
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 1
    assert r["n_violations"] == 0


def test_right_angle_corner_is_violation_in_45_mode() -> None:
    """A 90° corner is rejected under the 45-only allowed set {135, 180}."""
    eng = FakeEngine(_segments((0, 0), (1, 0), (1, 1)))
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[45])
    assert r["n_joints_checked"] == 1
    assert r["n_violations"] == 1
    assert r["violations"][0]["angle_deg"] == pytest.approx(90.0, abs=1e-3)


def test_straight_collinear_joint_passes_in_both_modes() -> None:
    """Straight (180°) is allowed regardless of mode."""
    eng = FakeEngine(_segments((0, 0), (1, 0), (2, 0)))
    for mode in (45, 90):
        r = _check_routing_angles(eng, _ALLOWED_ANGLES[mode])
        assert r["n_joints_checked"] == 1
        assert r["n_violations"] == 0, f"mode={mode}"


def test_45_miter_passes_in_45_mode() -> None:
    """H + diagonal at (1, 0) → measured 135° → allowed in 45 mode."""
    eng = FakeEngine(_segments((0, 0), (1, 0), (2, 1)))
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[45])
    assert r["n_joints_checked"] == 1
    assert r["n_violations"] == 0


def test_45_miter_is_violation_in_90_mode() -> None:
    """Same 135° miter corner is rejected under the orthogonal {90, 180} set."""
    eng = FakeEngine(_segments((0, 0), (1, 0), (2, 1)))
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 1
    assert r["n_violations"] == 1
    v = r["violations"][0]
    assert v["x_mm"] == pytest.approx(1.0)
    assert v["y_mm"] == pytest.approx(0.0)
    assert v["angle_deg"] == pytest.approx(135.0, abs=1e-3)
    assert v["layer"] == 0
    assert v["net_code"] == 1


def test_arbitrary_angle_is_violation_in_both_modes() -> None:
    """A 30° corner produces 150° — outside both {90, 180} and {135, 180}."""
    import math
    p = (1 + math.cos(math.radians(30)), math.sin(math.radians(30)))
    eng = FakeEngine(_segments((0, 0), (1, 0), p))
    for mode in (45, 90):
        r = _check_routing_angles(eng, _ALLOWED_ANGLES[mode])
        assert r["n_violations"] == 1, f"mode={mode}"
        assert r["violations"][0]["angle_deg"] == pytest.approx(150.0, abs=1e-3)


def test_tolerance_window_is_inclusive() -> None:
    """Measured 90.4° (within default 0.5° tolerance of 90) still passes 90 mode."""
    import math
    # Outward2 angle from +x is (180 - 90.4) = 89.6°.
    theta = math.radians(89.6)
    end = (1 + math.cos(theta), math.sin(theta))
    eng = FakeEngine(_segments((0, 0), (1, 0), end))
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 1
    assert r["n_violations"] == 0  # 90.4° is within 0.5° of 90°


def test_t_junction_is_skipped() -> None:
    """Joints with !=2 connected segments are not corners — skipped."""
    eng = FakeEngine([
        FakeTrack(0, 0, 1, 0),
        FakeTrack(1, 0, 1, 1),
        FakeTrack(1, 0, 1, -1),
    ])
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 0
    assert r["n_violations"] == 0


def test_different_layers_do_not_form_joint() -> None:
    """Same coord on different layers must not be treated as a corner."""
    eng = FakeEngine([
        FakeTrack(0, 0, 1, 0, layer=0),
        FakeTrack(1, 0, 2, 1, layer=1),  # would be a 135° violation if same layer
    ])
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 0
    assert r["n_violations"] == 0


def test_different_nets_do_not_form_joint() -> None:
    """Two different nets touching at a coord aren't treated as a corner."""
    eng = FakeEngine([
        FakeTrack(0, 0, 1, 0, net_code=1, net_name="N1"),
        FakeTrack(1, 0, 2, 1, net_code=2, net_name="N2"),
    ])
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 0
    assert r["n_violations"] == 0


def test_mixed_polyline_counts_each_joint() -> None:
    """One H+V (90°, OK in 90 mode) and one H+diag (135°, violation in 90 mode)."""
    eng = FakeEngine(_segments(
        (0, 0), (1, 0), (1, 1), (2, 2),  # 90° at (1,0); 135° at (1,1)
    ))
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[90])
    assert r["n_joints_checked"] == 2
    assert r["n_violations"] == 1
    assert r["violations"][0]["x_mm"] == pytest.approx(1.0)
    assert r["violations"][0]["y_mm"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Allowed-angle sets
# ---------------------------------------------------------------------------


def test_allowed_angles_table() -> None:
    """The mode-to-angle table must match documented conventions."""
    assert _ALLOWED_ANGLES[45] == (135.0, 180.0)
    assert _ALLOWED_ANGLES[90] == (90.0, 180.0)


# ---------------------------------------------------------------------------
# pad / via exclusion
# ---------------------------------------------------------------------------


def test_pad_at_joint_excludes_corner_from_check() -> None:
    """A geometric 90° corner that lies on a same-layer pad is not a
    routing choice — KiCad's track_angle DRC also skips pad joints,
    so our heuristic must too. Without exclusion this would flag a 90°
    violation under 45 mode.
    """
    eng = FakeEngine(
        tracks=_segments((0, 0), (1, 0), (1, 1)),
        pads=[FakePad(x_mm=1.0, y_mm=0.0, layer=0)],
    )
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[45])
    assert r["n_joints_checked"] == 0
    assert r["n_joints_skipped_pad_via"] == 1
    assert r["n_violations"] == 0


def test_pad_on_different_layer_does_not_exclude() -> None:
    """Pad on layer 1 must not mask a violation on layer 0."""
    eng = FakeEngine(
        tracks=_segments((0, 0), (1, 0), (1, 1), layer=0),
        pads=[FakePad(x_mm=1.0, y_mm=0.0, layer=1)],
    )
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[45])
    assert r["n_joints_checked"] == 1
    assert r["n_joints_skipped_pad_via"] == 0
    assert r["n_violations"] == 1  # the 90° corner on layer 0


def test_via_at_joint_excludes_corner_on_every_spanned_layer() -> None:
    """A through-hole via spans top..bottom — a joint at the via on any
    of those layers must be skipped."""
    eng = FakeEngine(
        tracks=_segments((0, 0), (1, 0), (1, 1), layer=0),
        vias=[FakeVia(x_mm=1.0, y_mm=0.0, top_layer=0, bottom_layer=31)],
    )
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[45])
    assert r["n_joints_checked"] == 0
    assert r["n_joints_skipped_pad_via"] == 1
    assert r["n_violations"] == 0


def test_via_outside_joint_layer_range_does_not_exclude() -> None:
    """A blind via that doesn't reach the joint's layer is not relevant."""
    eng = FakeEngine(
        tracks=_segments((0, 0), (1, 0), (1, 1), layer=0),
        vias=[FakeVia(x_mm=1.0, y_mm=0.0, top_layer=2, bottom_layer=3)],
    )
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[45])
    assert r["n_joints_checked"] == 1
    assert r["n_violations"] == 1


def test_pad_via_exclusion_does_not_double_count() -> None:
    """Joint at both a pad AND a via on the same layer counts once."""
    eng = FakeEngine(
        tracks=_segments((0, 0), (1, 0), (1, 1), layer=0),
        pads=[FakePad(x_mm=1.0, y_mm=0.0, layer=0)],
        vias=[FakeVia(x_mm=1.0, y_mm=0.0, top_layer=0, bottom_layer=31)],
    )
    r = _check_routing_angles(eng, _ALLOWED_ANGLES[45])
    assert r["n_joints_checked"] == 0
    assert r["n_joints_skipped_pad_via"] == 1  # set semantics, not 2
    assert r["n_violations"] == 0


# ---------------------------------------------------------------------------
# evaluate_one argument validation
# ---------------------------------------------------------------------------


def test_evaluate_one_rejects_bogus_check_angle(tmp_path: Path) -> None:
    """Catch the bad value before any engine work happens."""
    with pytest.raises(ValueError, match="check_angle must be 45 or 90"):
        evaluate_one(
            routed_pcb=str(tmp_path / "nonexistent.kicad_pcb"),
            pro_path=None,
            check_angle=60,
        )


# ---------------------------------------------------------------------------
# track_angle_drv aggregation across samples
# ---------------------------------------------------------------------------


def _sample(
    *,
    success: bool = True,
    routability: float = 1.0,
    drv_err: int = 0,
    drv_err_prom: int = 0,
    track_count: int = 1,
    via_count: int = 0,
    wirelength_mm: float = 1.0,
    final_potential: float = 0.0,
    angle_mode: int = 45,
    angle_count: int = 0,
) -> dict:
    """Minimal per-sample dict shaped like ``evaluate_one`` output —
    only the fields ``aggregate`` reads.

    ``clean_pass`` is derived from the same triple that
    ``evaluate_one`` uses (success AND drv_errors_and_promoted == 0
    AND track_angle_drv.count == 0) so aggregation tests match the
    real shape.
    """
    return {
        "success": success,
        "routability": routability,
        "track_count": track_count,
        "via_count": via_count,
        "wirelength_mm": wirelength_mm,
        "drv_errors_only_count": drv_err,
        "drv_errors_and_promoted_count": drv_err_prom,
        "final_potential": final_potential,
        "track_angle_drv": {
            "mode": angle_mode,
            "source": "heuristic",
            "count": angle_count,
            "violations": [],
        },
        "total_drv_count": int(drv_err_prom + angle_count),
        "clean_pass": bool(success and drv_err_prom == 0 and angle_count == 0),
        "drv_breakdown": {
            "errors_only_by_type": [],
            "errors_and_promoted_by_type": [],
        },
    }


def test_aggregate_track_angle_drv_45_mode() -> None:
    """45-mode samples: mean/stdev of count + total/samples_with_violations."""
    samples = [
        _sample(angle_mode=45, angle_count=0),
        _sample(angle_mode=45, angle_count=2),
        _sample(angle_mode=45, angle_count=4),
    ]
    agg = aggregate(samples)
    assert agg["mean"]["track_angle_drv_count"] == pytest.approx(2.0)
    assert agg["stdev"]["track_angle_drv_count"] > 0
    ta = agg["track_angle_drv"]
    assert ta["modes_seen"] == [45]
    assert ta["samples_with_violations"] == 2  # the two with count > 0
    assert ta["total_violations"] == 6


def test_aggregate_track_angle_drv_90_mode() -> None:
    """90-mode samples are aggregated identically (source field differs)."""
    samples = [
        _sample(angle_mode=90, angle_count=1),
        _sample(angle_mode=90, angle_count=3),
    ]
    agg = aggregate(samples)
    assert agg["mean"]["track_angle_drv_count"] == pytest.approx(2.0)
    ta = agg["track_angle_drv"]
    assert ta["modes_seen"] == [90]
    assert ta["total_violations"] == 4


def test_aggregate_skips_failed_samples() -> None:
    """Samples with an ``error`` key are excluded from angle stats."""
    samples = [
        _sample(angle_mode=45, angle_count=2),
        {"board": "x", "pro": None, "error": "boom"},  # failed
        _sample(angle_mode=45, angle_count=4),
    ]
    agg = aggregate(samples)
    assert agg["n_total"] == 3
    assert agg["n_ok"] == 2
    assert agg["n_fail"] == 1
    # Mean computed over the 2 ok samples only.
    assert agg["mean"]["track_angle_drv_count"] == pytest.approx(3.0)
    assert agg["track_angle_drv"]["total_violations"] == 6


def test_aggregate_handles_legacy_samples_without_track_angle_field() -> None:
    """Samples cached before track_angle_drv existed must not crash agg."""
    legacy = _sample(angle_mode=45, angle_count=0)
    legacy.pop("track_angle_drv")
    agg = aggregate([legacy])
    # Falls back to count=0; modes_seen excludes None.
    assert agg["mean"]["track_angle_drv_count"] == 0.0
    assert agg["track_angle_drv"]["modes_seen"] == []
    assert agg["track_angle_drv"]["total_violations"] == 0


# ---------------------------------------------------------------------------
# total_drv_count (DRC errors+promoted PLUS heuristic angle violations)
# ---------------------------------------------------------------------------


def test_total_drv_count_aggregate_sum_and_stdev() -> None:
    samples = [
        _sample(drv_err_prom=2, angle_count=1),   # 3
        _sample(drv_err_prom=0, angle_count=0),   # 0
        _sample(drv_err_prom=4, angle_count=2),   # 6
    ]
    agg = aggregate(samples)
    assert agg["mean"]["total_drv_count"] == pytest.approx((3 + 0 + 6) / 3)
    assert agg["stdev"]["total_drv_count"] > 0


def test_total_drv_count_legacy_fallback() -> None:
    """Cached samples without ``total_drv_count`` are reconstructed from
    ``drv_errors_and_promoted_count + track_angle_drv.count``."""
    legacy = _sample(drv_err_prom=3, angle_count=2)
    legacy.pop("total_drv_count")
    agg = aggregate([legacy])
    assert agg["mean"]["total_drv_count"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# clean_pass rate
# ---------------------------------------------------------------------------


def test_clean_pass_rate_counts_only_zero_drv_completions() -> None:
    """clean_pass_rate over ok samples = (success AND drv==0 AND angle==0)."""
    samples = [
        _sample(success=True,  drv_err_prom=0, angle_count=0),  # clean
        _sample(success=True,  drv_err_prom=2, angle_count=0),  # has DRV
        _sample(success=True,  drv_err_prom=0, angle_count=3),  # has angle
        _sample(success=False, drv_err_prom=0, angle_count=0),  # not routed
        _sample(success=True,  drv_err_prom=0, angle_count=0),  # clean
    ]
    agg = aggregate(samples)
    assert agg["success_rate"] == pytest.approx(4 / 5)
    assert agg["clean_pass_rate"] == pytest.approx(2 / 5)


def test_clean_pass_excludes_failed_samples_from_denominator() -> None:
    """``error`` samples don't pad the denominator — same as success_rate."""
    samples = [
        _sample(success=True, drv_err_prom=0, angle_count=0),
        {"board": "x", "pro": None, "error": "boom"},  # failed
    ]
    agg = aggregate(samples)
    assert agg["n_ok"] == 1
    assert agg["clean_pass_rate"] == pytest.approx(1.0)


def test_clean_pass_legacy_sample_treated_as_not_clean() -> None:
    """Cached samples without the field are treated as not clean (False)."""
    legacy = _sample(success=True, drv_err_prom=0, angle_count=0)
    legacy.pop("clean_pass")
    agg = aggregate([legacy])
    assert agg["clean_pass_rate"] == pytest.approx(0.0)


def test_clean_pass_nan_when_all_samples_failed() -> None:
    """Empty ok set → NaN (matches success_rate convention)."""
    import math
    samples = [{"board": "x", "pro": None, "error": "boom"}]
    agg = aggregate(samples)
    assert math.isnan(agg["clean_pass_rate"])
    assert math.isnan(agg["success_rate"])
