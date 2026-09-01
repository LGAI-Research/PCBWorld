"""RL-wrapper MCTS state snapshot/restore (RL L2).

The RL policy is Markov, so the wrapper's only path-dependent per-env state is
the auto-net-select RNG and ``_start_route_xy`` (same-point masking). These
tests exercise that snapshot/restore pair in isolation (bypassing the heavy
wrapper __init__).
"""

import numpy as np
import pytest

from methods.rl_agent.wrappers.adapter import KiCadRLWrapper


@pytest.fixture
def wrap():
    w = KiCadRLWrapper.__new__(KiCadRLWrapper)
    w._rng = np.random.default_rng(123)
    w._start_route_xy = None
    return w


def test_rng_stream_and_start_reproduced(wrap):
    wrap._start_route_xy = (3.0, 4.0, 1)
    wrap._rng.random()
    wrap._rng.random()
    snap = wrap.snapshot_mcts_state()
    expected = [wrap._rng.random() for _ in range(5)]

    # explore a branch: advance the RNG further and move the start point
    [wrap._rng.random() for _ in range(10)]
    wrap._start_route_xy = (7.0, 7.0, 2)

    wrap.restore_mcts_state(snap)
    assert wrap._start_route_xy == (3.0, 4.0, 1)
    # RNG stream after restore reproduces the post-snapshot stream exactly
    assert [wrap._rng.random() for _ in range(5)] == expected


def test_snapshot_isolated_from_live_advance(wrap):
    snap = wrap.snapshot_mcts_state()
    state_before = snap["rng_state"]
    wrap._rng.random()                 # advance the live RNG
    assert snap["rng_state"] == state_before   # snapshot untouched (deepcopy)

    # the same snapshot can be restored repeatedly
    wrap.restore_mcts_state(snap)
    a = wrap._rng.random()
    wrap.restore_mcts_state(snap)
    b = wrap._rng.random()
    assert a == b


# ---------------------------------------------------------------------------
# obs / pointer-decode bundle carried in the snapshot
# ---------------------------------------------------------------------------
def _with_obs(wrap, tag):
    """Give the wrapper an obs bundle as ``_refresh_cache`` would."""
    wrap._last_obs = {"router_head": {"tag": tag}}
    wrap._sorted_net_codes = [1, 2, tag]
    wrap._cand_mm = [(float(tag), 0.0, 1)]
    wrap._cand_ctype = [tag % 3]              # CTYPE_* per _cand_mm entry
    return wrap


def test_obs_bundle_round_trips(wrap):
    """The snapshot carries the derived obs + pointer tables so a restore need
    not re-derive them (a full _get_obs + engine connectivity query +
    candidate-pool rebuild, paid once per MCTS simulation)."""
    _with_obs(wrap, 7)
    snap = wrap.snapshot_mcts_state()
    assert "obs_cache" in snap

    _with_obs(wrap, 99)                       # explore a different branch
    assert wrap.restore_mcts_state(snap) is True
    assert wrap._last_obs == {"router_head": {"tag": 7}}
    assert wrap._sorted_net_codes == [1, 2, 7]
    assert wrap._cand_mm == [(7.0, 0.0, 1)]   # pointer decode sees node 7 again
    assert wrap._cand_ctype == [7 % 3]        # ctype table travels with it


def test_obs_bundle_opt_out_and_absent(wrap):
    """``obs_cache=False`` (and a wrapper that never cached) leave the bundle
    out; the restore reports False so the caller re-derives."""
    _with_obs(wrap, 7)
    snap = wrap.snapshot_mcts_state(obs_cache=False)
    assert "obs_cache" not in snap
    assert wrap.restore_mcts_state(snap) is False

    del wrap._last_obs                        # never refreshed (pre-first-reset)
    assert "obs_cache" not in wrap.snapshot_mcts_state()
