"""Contract tests for the shared runtime metric semantics.

``eval.eval_utils.runtime_metrics_from_info`` (+ the pure helpers
``success_from`` / ``clean_pass_from``) is the single derivation point for
per-rollout metric semantics across every live-env row producer (RL rollout,
LLM plan-only replay, LLM live episodes). These tests pin the canonical
definitions:

  - success   = terminated AND fully routed (give-up termination is NOT one)
  - clean_pass = success AND zero errors-only DRVs; blank "" when env-DRC off
"""

import math

from eval import eval_utils as u


def _info(**over):
    """A fully-routed episode-end info in the PCBWorld step-info vocabulary."""
    info = {
        "success": True,
        "unrouted_count": 0,
        "initial_unconnected": 5,
        "drc_errors": 0,
        "ratsnest_reduction": 1.0,
        "wirelength": 12.5,
        "via_count": 2,
        "track_count": 9,
        "drc_violations": 0,
    }
    info.update(over)
    return info


# ---------------------------------------------------------------------------
# The three spec'd contract cases
# ---------------------------------------------------------------------------


def test_full_route_zero_errors_is_success_and_clean_pass():
    m = u.runtime_metrics_from_info(_info(), terminated=True, truncated=False)
    assert m["status"] == "completed"
    assert m["success"] is True
    assert m["clean_pass"] is True
    assert m["drv_errors_only_count"] == 0.0


def test_giveup_termination_is_not_success():
    # Voluntary all-nets-closed finish (v0.11.28): terminated with unrouted>0.
    m = u.runtime_metrics_from_info(
        _info(success=False, unrouted_count=2, ratsnest_reduction=0.6),
        terminated=True, truncated=False,
    )
    assert m["terminated"] is True
    assert m["success"] is False
    assert m["clean_pass"] is False   # drc_errors==0 but not fully routed


def test_drc_off_clean_pass_is_blank():
    info = _info()
    del info["drc_errors"]
    m = u.runtime_metrics_from_info(info, terminated=True, truncated=False)
    assert m["clean_pass"] == ""      # blank -> skipped by aggregation
    assert math.isnan(m["drv_errors_only_count"])


# ---------------------------------------------------------------------------
# Derivation fallbacks (producers without a live env info)
# ---------------------------------------------------------------------------


def test_success_derived_from_unrouted_when_env_verdict_absent():
    # plan-only replay synthesizes info from a snapshot: no "success" key.
    m = u.runtime_metrics_from_info(
        {"unrouted_count": 2, "initial_unconnected": 4},
        terminated=True, truncated=False,
    )
    assert m["success"] is False
    m = u.runtime_metrics_from_info(
        {"unrouted_count": 0, "initial_unconnected": 4},
        terminated=True, truncated=False,
    )
    assert m["success"] is True


def test_ratsnest_reduction_derived_when_absent_routability_stays_nan():
    m = u.runtime_metrics_from_info(
        {"unrouted_count": 1, "initial_unconnected": 4},
        terminated=True, truncated=False,
    )
    assert m["ratsnest_reduction"] == 0.75
    # routability is a pad-group metric only the DRC eval stage fills.
    assert math.isnan(m["routability"])
    # Env-computed ratsnest_reduction wins over the derived ratio.
    m = u.runtime_metrics_from_info(
        _info(ratsnest_reduction=0.5), terminated=True, truncated=False,
    )
    assert m["ratsnest_reduction"] == 0.5


def test_full_route_with_errors_is_not_clean_pass():
    m = u.runtime_metrics_from_info(
        _info(drc_errors=3), terminated=True, truncated=False,
    )
    assert m["success"] is True
    assert m["clean_pass"] is False
    assert m["drv_errors_only_count"] == 3.0


def test_missing_keys_degrade_to_nan_and_blank():
    m = u.runtime_metrics_from_info({}, terminated=False, truncated=True)
    assert m["success"] is False
    assert m["clean_pass"] == ""
    assert math.isnan(m["unrouted_count"])
    assert math.isnan(m["ratsnest_reduction"])
    assert math.isnan(m["routability"])
    assert m["terminated"] is False and m["truncated"] is True


# ---------------------------------------------------------------------------
# Crash handling
# ---------------------------------------------------------------------------


def test_engine_crash_in_info_folds_into_crashed():
    m = u.runtime_metrics_from_info(
        _info(engine_crash=True), terminated=True, truncated=False,
    )
    assert m["status"] == "crashed"
    assert m["crashed"] is True and m["engine_crash"] is True
    assert m["success"] is False     # crash overrides the env verdict
    assert m["clean_pass"] is False  # ... and a crash is never a clean pass


def test_producer_crashed_flag():
    m = u.runtime_metrics_from_info(_info(), terminated=True, truncated=False,
                                    crashed=True)
    assert m["status"] == "crashed"
    assert m["crashed"] is True and m["engine_crash"] is False
    assert m["success"] is False
    assert m["clean_pass"] is False


def test_env_verdict_wins_over_derivation():
    # The env's episode-end verdict is authoritative when present, even when
    # it disagrees with what the counts would derive.
    m = u.runtime_metrics_from_info(
        _info(success=False, unrouted_count=0), terminated=True, truncated=False,
    )
    assert m["success"] is False
    m = u.runtime_metrics_from_info(
        _info(success=True, unrouted_count=3), terminated=False, truncated=True,
    )
    assert m["success"] is True


def test_measured_columns_pass_through():
    m = u.runtime_metrics_from_info(
        _info(drc_violations=1, final_potential=0.25, initial_potential=-1.5,
              potential_gain=1.75, error="invalid_action"),
        terminated=True, truncated=False,
    )
    assert m["wirelength_mm"] == 12.5    # env "wirelength" -> CSV "wirelength_mm"
    assert m["via_count"] == 2.0
    assert m["track_count"] == 9.0
    assert m["drc_violations"] == 1.0
    assert m["final_potential"] == 0.25
    assert m["initial_potential"] == -1.5
    assert m["potential_gain"] == 1.75
    assert m["error"] == "invalid_action"


def test_aggregation_skips_blank_clean_pass():
    # The DRC-off contract's aggregation half: blank "" cells are excluded
    # from the bool mean instead of counting as False.
    from eval.aggregation import _aggregate_value_metric

    rows = [{"clean_pass": True}, {"clean_pass": ""}, {"clean_pass": False}]
    assert _aggregate_value_metric(rows, "clean_pass") == 0.5


# ---------------------------------------------------------------------------
# Pure helpers + wire-protocol invariant
# ---------------------------------------------------------------------------


def test_success_from_contract():
    assert u.success_from(terminated=True, unrouted_count=0) is True
    assert u.success_from(terminated=True, unrouted_count=3) is False
    assert u.success_from(terminated=False, unrouted_count=0) is False
    assert u.success_from(terminated=True, unrouted_count=0, crashed=True) is False
    # Unknown routed-state count -> legacy terminated-only fallback.
    assert u.success_from(terminated=True) is True
    assert u.success_from(terminated=True, unrouted_count=math.nan) is True


def test_clean_pass_from_contract():
    assert u.clean_pass_from(True, 0) is True
    assert u.clean_pass_from(True, 2) is False
    assert u.clean_pass_from(False, 0) is False
    assert u.clean_pass_from(True, None) == ""
    assert u.clean_pass_from(True, "") == ""


def test_emitted_keys_are_canonical_columns():
    # Every derived field must land in a PER_ROLLOUT_FIELDS column, or the
    # CSV writer (extrasaction='ignore') would drop it silently.
    m = u.runtime_metrics_from_info(_info(), terminated=True, truncated=False)
    assert set(m) <= set(u.PER_ROLLOUT_FIELDS)
