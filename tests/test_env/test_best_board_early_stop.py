"""Opt-in PCBWorld early-stop + best-Φ-board selection (requires C++ router).

Two features gated off by default (training/existing rollouts unchanged):
  - ``early_stop_ratsnest_patience=T`` — truncate after T steps with no change in
    the unrouted-ratsnest count (connections stalled).
  - ``output_best_board=True`` — at episode end, roll the LIVE board back to the
    highest-Φ board seen this episode, so the scorer/artifact (which re-read the
    live board) get the best board rather than a later degraded one.

Driven on the 2-net scripted board: route NET1 (Φ peak), then a partial NET2
track that adds wirelength WITHOUT connecting (Φ dips). See
``test_scripted_routing.py`` for the board layout.
"""

import os

import pytest

from pcb_world.core.env import PCBWorld
from pcb_world.core.action_schema import (
    ACT_MAKE_LINE,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)

BOARD = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "two_net_multiterm_board.kicad_pcb"
)
NET1, NET2 = 1, 2
A1, B1, C1, J1 = (10.0, 10.0), (40.0, 10.0), (25.0, 5.0), (25.0, 10.0)
A2, MID2 = (10.0, 20.0), (25.0, 20.0)


def _step(env, at, x=0.0, y=0.0, layer=1, net_id=0):
    return env.step({
        "action_type": at, "x_mm": float(x), "y_mm": float(y),
        "layer": layer, "net_id": net_id, "routing_mode": 2,
    })


def _route_net1(env):
    """Fully connect NET1 (A1-B1 run + C1 tap) — the Φ peak of the episode."""
    _step(env, ACT_NET_SELECT, net_id=NET1)
    _step(env, ACT_START_ROUTE, *A1)
    _step(env, ACT_MAKE_LINE, *J1)
    _step(env, ACT_START_ROUTE, *J1)
    _step(env, ACT_MAKE_LINE, *B1)
    _step(env, ACT_START_ROUTE, *C1)
    _step(env, ACT_MAKE_LINE, *J1)          # taps the A1-B1 run → NET1 connected


def _wander_net2(env):
    """Partial NET2 track A2→midpoint: adds wirelength, connects no pad pair."""
    _step(env, ACT_NET_END)                  # release NET1 before switching nets
    _step(env, ACT_NET_SELECT, net_id=NET2)
    _step(env, ACT_START_ROUTE, *A2)
    return _step(env, ACT_MAKE_LINE, *MID2)  # partial (not to a pad) → NET2 unrouted


def _run_script(output_best_board: bool):
    """route NET1 (7) + wander NET2 (4) = 11 steps; step 11 truncates on max_steps.

    wirelength_penalty>0 (the default drc_only_dense config has it at 0) so the
    partial NET2 track genuinely lowers Φ below the post-NET1 peak.
    """
    env = PCBWorld(board_path=BOARD, max_steps=11, wirelength_penalty=0.002,
                   output_best_board=output_best_board)
    env.reset()
    _route_net1(env)
    _, _, _, truncated, info = _wander_net2(env)
    try:
        return truncated, info
    finally:
        env.close()


def test_output_best_board_rolls_back_to_peak():
    """output_best_board=True rolls the final board back to the post-NET1 peak
    (higher Φ, less wirelength) vs the degraded NET1+wander board of the plain run."""
    trunc_off, info_off = _run_script(output_best_board=False)
    trunc_on, info_on = _run_script(output_best_board=True)

    assert trunc_off and trunc_on                        # both ended on max_steps
    assert not info_off.get("output_best_board_restored", False)
    assert info_on.get("output_best_board_restored") is True

    # The restored board is the peak: strictly higher Φ and less copper than the
    # degraded final board, with the same (NET2-unrouted) connectivity.
    assert info_on["final_potential"] > info_off["final_potential"]
    assert info_on["wirelength"] < info_off["wirelength"]
    assert info_on["unrouted_count"] == info_off["unrouted_count"]
    assert info_on["final_potential"] == pytest.approx(info_on["best_potential"])


def test_ratsnest_patience_truncates_on_stall():
    """early_stop_ratsnest_patience truncates once the unrouted count stops
    changing. With no connection ever made, the setup steps alone stall it: the
    guard fires within patience+1 steps."""
    patience = 2
    env = PCBWorld(board_path=BOARD, max_steps=60,
                   early_stop_ratsnest_patience=patience)
    env.reset()
    truncated = False
    steps = 0
    # net_select / start_route / make_line-wander never reduce unrouted here.
    seq = [(ACT_NET_SELECT, 0.0, 0.0, NET1), (ACT_START_ROUTE, *A1, 0),
           (ACT_MAKE_LINE, *J1, 0), (ACT_MAKE_LINE, 20.0, 10.0, 0)]
    for at, x, y, net in seq:
        _, _, terminated, truncated, _ = _step(env, at, x, y, net_id=net)
        steps += 1
        if truncated or terminated:
            break
    env.close()
    assert truncated                                     # guard fired
    assert steps <= patience + 1                         # within patience+1 steps


def test_reset_rearms_episode_tracking():
    """``reset()`` re-arms ``_track_best_active``.

    A search driver (``RLSearchEnv.step``) flips the flag per step to keep
    throwaway simulations out of the episode-level bookkeeping and never
    restores it. Without the re-arm, the next episode on the SAME env would run
    with best-Φ / early-stop silently disabled — mcts_compare reuses one env for
    the MCTS arm and the plain arm, so the head-to-head would be skewed.
    """
    env = PCBWorld(board_path=BOARD, max_steps=60,
                   early_stop_ratsnest_patience=3, output_best_board=True)
    env.reset()
    env._track_best_active = False                       # what a search leaves behind
    env.reset()
    active = env._track_best_active
    env.close()
    assert active is True


def test_features_off_by_default():
    """Default construction leaves both features inert (no truncation from the
    guard, no best-board bookkeeping)."""
    env = PCBWorld(board_path=BOARD, max_steps=60)
    env.reset()
    assert env._ratsnest_patience == 0
    assert env._output_best_board is False
    assert env._best_ckpt is None
    _, _, _, truncated, info = _step(env, ACT_NET_SELECT, net_id=NET1)
    env.close()
    assert not truncated
    assert "output_best_board_restored" not in info
