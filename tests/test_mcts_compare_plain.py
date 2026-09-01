"""Unit tests for the eval-aligned plain baseline in ``mcts_compare``.

These exercise the pure control logic — ``_derive_check_angle`` and the
``run_plain`` best@k loop (count / wallclock-budget / greedy-collapse and the
``selection_key`` winner pick) — WITHOUT a policy checkpoint or the C++ engine
by stubbing the single-rollout function. The full end-to-end path (real env +
native ``.kicad_pro`` scoring) is covered by the manual MCTS comparison smoke run.
"""

from __future__ import annotations

import types

import pytest

from methods.rl_agent.policy import mcts_compare as mc


def _args(**over):
    """Minimal args namespace for the helpers under test."""
    base = dict(
        check_angle=None, corner_mode=None, seed=0, deterministic=False,
        selection_mode="final_potential",
        early_stop_finish_no_progress=0, early_stop_no_geometry_progress=0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------
# _derive_check_angle
# --------------------------------------------------------------------------

@pytest.mark.parametrize("corner,expected", [(None, 45), (0, 45), (1, 45), (2, 90), (3, 90)])
def test_derive_check_angle_from_corner(corner, expected):
    assert mc._derive_check_angle(_args(corner_mode=corner)) == expected


def test_derive_check_angle_explicit_overrides_corner():
    # explicit --check-angle wins over the corner-derived default
    assert mc._derive_check_angle(_args(corner_mode=0, check_angle=90)) == 90
    assert mc._derive_check_angle(_args(corner_mode=2, check_angle=45)) == 45


# --------------------------------------------------------------------------
# run_plain — count / time-budget / greedy-collapse / winner selection
# --------------------------------------------------------------------------

def _stub_rollouts(monkeypatch, phis, per_call_sec=0.0):
    """Patch plain_rollout_once to return canned rows (final_potential=phis[i]),
    optionally consuming ``per_call_sec`` of wallclock so the time-budget path
    is exercisable. Returns a list that records each produced rollout_idx."""
    produced: list[int] = []
    import time as _time

    def fake_once(wrapper, agent, device, *, rollout_idx, **kw):
        produced.append(rollout_idx)
        if per_call_sec:
            _time.sleep(per_call_sec)
        phi = phis[rollout_idx] if rollout_idx < len(phis) else phis[-1]
        return {"final_potential": phi, "routability": 1.0, "rollout_idx": rollout_idx}

    monkeypatch.setattr(mc, "plain_rollout_once", fake_once)
    return produced


def test_run_plain_fixed_n(monkeypatch):
    produced = _stub_rollouts(monkeypatch, phis=[0.1, 0.9, 0.5])
    winner, cnt, _elapsed = mc.run_plain(
        None, None, None, _args(), reward_cfg="r", check_angle=45, cap=10, n=3,
    )
    assert cnt == 3 and produced == [0, 1, 2]
    assert winner["final_potential"] == 0.9          # best@k by final_potential


def test_run_plain_greedy_collapses_to_one(monkeypatch):
    produced = _stub_rollouts(monkeypatch, phis=[0.3, 0.3, 0.3])
    winner, cnt, _e = mc.run_plain(
        None, None, None, _args(deterministic=True),
        reward_cfg="r", check_angle=45, cap=10, n=5,   # n ignored under greedy
    )
    assert cnt == 1 and produced == [0]
    assert winner["final_potential"] == 0.3


def test_run_plain_time_budget_runs_multiple_then_stops(monkeypatch):
    # ~5ms per rollout, 60ms budget → several rollouts, at least 2, and it stops
    # (doesn't run unbounded).
    _stub_rollouts(monkeypatch, phis=[0.0] * 100, per_call_sec=0.005)
    winner, cnt, elapsed = mc.run_plain(
        None, None, None, _args(), reward_cfg="r", check_angle=45, cap=10,
        time_budget=0.06,
    )
    assert cnt >= 2                      # budget large enough for several
    assert elapsed >= 0.06               # ran until at least the budget
    assert winner is not None


def test_run_plain_time_budget_guarantees_at_least_one(monkeypatch):
    # zero budget must still produce exactly one rollout (never zero candidates).
    _stub_rollouts(monkeypatch, phis=[0.7], per_call_sec=0.0)
    winner, cnt, _e = mc.run_plain(
        None, None, None, _args(), reward_cfg="r", check_angle=45, cap=10,
        time_budget=0.0,
    )
    assert cnt == 1 and winner["final_potential"] == 0.7
