"""Regression guard: eval `initial_potential` must use the SAME DRC convention
as `final_potential`.

`compute_metrics` snapshots the routed board with ``run_drc=True`` and the
bare baseline with ``run_drc=True`` too. If the bare snapshot ever drops back
to ``run_drc=False`` (the historical bug), ``potential_gain = final - initial``
silently subtracts a DRC-free initial from a DRC-included final — mixing two Φ
definitions and understating the gain by the bare-board DRC term.

The invariant we pin: on an **already-bare** board (routed state == bare state,
0 tracks/vias), `final_potential` and `initial_potential` are computed from the
identical board state, so the gain MUST be exactly 0. That equality only holds
when both endpoints share the same ``run_drc`` flag — flipping the bare-board
call to ``run_drc=False`` makes the DRC term vanish from the initial only, and
the gain becomes nonzero. The test fails in that case.

Scope: this exercises the real `compute_metrics` kernel + the real reward
config (no C++ router), so it guards the wiring *inside* `evaluate_one`. It does
NOT verify that the production dispatch passes the intended `--reward-config`
(a config-level concern best confirmed by inspecting a real run's output CSV).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from pcb_world.engine.containers import BoardMeta, RewardSnapshot

# Load the scoring kernel by path (mirrors tests/test_eval_check_angle.py — avoids
# the native double-init landmine in eval/__init__ at collection time).
_EVAL_PCB_PATH = Path(__file__).resolve().parents[1] / "eval" / "metrics.py"
_spec = importlib.util.spec_from_file_location("_eval_metrics_initpot", _EVAL_PCB_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
compute_metrics = _mod.compute_metrics


class _DrcGatedEngine:
    """Minimal engine for `compute_metrics`: a bare board (0 tracks/vias) with
    3 unrouted edges whose DRC violations only materialize when ``run_drc=True``.
    Records the ``run_drc`` flag of every snapshot call so the test can assert
    both endpoints used the same convention.
    """

    def __init__(self) -> None:
        self.run_drc_calls: list[bool] = []

    def get_reward_snapshot(self, run_drc: bool) -> RewardSnapshot:
        self.run_drc_calls.append(bool(run_drc))
        drc = bool(run_drc)
        return RewardSnapshot(
            unrouted_count=3,
            track_count=0,
            via_count=0,
            total_wirelength=0.0,
            drc_violation_count=2 if drc else 0,
            drc_violations_per_net={"N1": 1, "N2": 1} if drc else {},
            drc_error_count=2 if drc else 0,
            drc_errors_per_net={"N1": 1, "N2": 1} if drc else {},
            drc_promoted_count=2 if drc else 0,
            drc_promoted_per_net={"N1": 1, "N2": 1} if drc else {},
        )

    def get_board_meta(self) -> BoardMeta:
        return BoardMeta(net_count=2, copper_layers=2)

    # Bare board — nothing to enumerate.
    def get_tracks(self):
        return []

    def get_vias(self):
        return []

    def get_ratsnest(self):
        return []

    # Bare board: every pad its own group (2 nets, 2+3 groups -> 3 required
    # connections, matching the 3 unrouted edges). Routed state == bare state
    # here, so both compute_metrics captures see the same grouping.
    def get_pad_groups(self) -> dict[int, int]:
        return {1: 2, 2: 3}

    # Per-reset reward hook inputs (PotentialReward.bind_board) — inert for
    # drc_dense_promoted (no size-weighted ladder) but the kernel always asks.
    def get_net_names(self) -> dict[int, str]:
        return {1: "N1", 2: "N2"}

    def get_routable_nets(self) -> frozenset[int]:
        return frozenset({1, 2})

    def get_pads(self):
        return []

    def get_drc_violations(self):
        return []

    def is_routing(self) -> bool:
        return False

    def get_track_count(self) -> int:
        return 0

    def get_via_count(self) -> int:
        return 0

    def build_connectivity(self) -> None:
        pass


def test_initial_potential_uses_same_drc_convention_as_final():
    eng = _DrcGatedEngine()
    m = compute_metrics(
        eng,
        u_0=None,  # triggers the destructive reset + initial_potential capture
        reward_config_name="drc_dense_promoted",  # DRC-weighted (errors_and_promoted)
        check_angle=45,
    )

    # Every snapshot — routed AND bare baseline — used run_drc=True.
    assert eng.run_drc_calls, "no reward snapshot taken"
    assert all(eng.run_drc_calls), (
        f"a snapshot used run_drc=False: {eng.run_drc_calls} — bare-board initial "
        "must match the routed final convention"
    )

    # The DRC term is actually present in the baseline (otherwise the invariant
    # below would be vacuous): initial Φ is strictly below the unconnected-only Φ.
    unconnected_only = -1.0 * 3  # unconnected_penalty (1.0) * unrouted_count (3)
    assert m["initial_potential"] < unconnected_only - 1e-9, (
        "initial_potential does not include the bare-board DRC penalty — "
        "the baseline was snapshotted without DRC"
    )

    # Headline invariant: on an already-bare board the routed state IS the bare
    # state, so the gain must be exactly 0. Holds only if both endpoints share
    # the same run_drc convention; the historical asymmetry breaks it.
    assert m["initial_potential"] == pytest.approx(m["final_potential"])
    assert m["potential_gain"] == pytest.approx(0.0)
