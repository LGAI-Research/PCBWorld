"""Env-level (L1) checkpoint/restore round-trip for PCBWorld.

The engine-level C++ round-trip lives in test_checkpoint_restore.py; this covers
the Python L1 layer the env adds on top — step_count, dispatcher
(current_net_id / routing_mode), the obs-facing action_history records, and that
the full observation reproduces after restore. Branch-agnostic core (RL == LLM).
"""

import json

import pytest

from pcb_world.core.env import PCBWorld
from pcb_world.core.action_schema import (
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACT_MAKE_LINE,
)

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def _record(env):
    """Snapshot the L1 state + full obs + board for equality comparison.

    The DRC cache is a derived field that restore intentionally clears (lazily
    recomputed — design §2/§4.3), so normalize it by recomputing from the
    current board before reading obs. The restored board equals the
    checkpointed board, so a fresh DRC run matches on both sides.
    """
    env._engine.run_drc()
    return {
        "step_count": env._step_count,
        "current_net_id": env._dispatcher.current_net_id,
        "routing_mode": env._dispatcher.routing_mode,
        "action_history": list(env._action_history),
        "is_routing": env._engine.is_routing(),
        "tracks": sorted(
            (round(t.x1_mm, 6), round(t.y1_mm, 6), round(t.x2_mm, 6),
             round(t.y2_mm, 6), t.layer, t.net_code)
            for t in env._engine.get_tracks()
        ),
        "obs": json.dumps(env._get_obs(), sort_keys=True, default=str),
    }


@pytest.fixture
def env():
    # action_history_len > 1 (default is 1) so the multi-entry history
    # round-trip / edge-override assertions below have depth to check.
    e = PCBWorld(board_path=BOARD, max_steps=20, action_history_len=3)
    e.reset()
    yield e
    if e._engine is not None and e._engine.is_routing():
        e._engine.cancel_route()
    e.close()


def _net_at(env, x, y):
    """net_code of the pad nearest (x, y) — a routable anchor on a fresh board."""
    for p in env._engine.get_pads():
        if abs(p.x_mm - x) < 0.6 and abs(p.y_mm - y) < 0.6 and p.net_code > 0:
            return int(p.net_code)
    raise AssertionError(f"no pad near ({x}, {y})")


def test_l1_round_trip(env):
    """net_select + start_route -> checkpoint -> make_line -> restore reproduces
    the full L1 state (step_count, dispatcher, action_history, board, obs)."""
    net_id = _net_at(env, 25.0, 5.0)

    env.step({"action_type": ACT_NET_SELECT, "net_id": net_id})
    assert env._dispatcher.current_net_id == net_id

    env.step({"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.0, "layer": 1})
    assert env._engine.is_routing(), "start_route should open a session"

    ckpt = env.checkpoint()
    a = _record(env)
    assert a["step_count"] == 2
    assert a["current_net_id"] == net_id

    # mutate: commit a track (changes board + session + step_count + history)
    env.step({"action_type": ACT_MAKE_LINE, "x_mm": 25.0, "y_mm": 9.0, "routing_mode": 2})
    assert _record(env) != a

    env.restore(ckpt)
    b = _record(env)
    assert b == a, "L1 state did not round-trip exactly"
    env.release_checkpoint(ckpt)


def test_action_history_restored_from_checkpoint(env):
    """action_history is captured by the checkpoint (self-sufficient restore —
    no edge_action needed), newest first with the net context recorded."""
    net_id = _net_at(env, 25.0, 5.0)
    env.step({"action_type": ACT_NET_SELECT, "net_id": net_id})
    env.step({"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.0, "layer": 1})
    ckpt = env.checkpoint()
    saved = list(env._action_history)
    assert [e["action_type"] for e in saved] == [ACT_START_ROUTE, ACT_NET_SELECT]
    assert all(e["net_id"] == net_id for e in saved)

    # take another step so the history changes
    env.step({"action_type": ACT_MAKE_LINE, "x_mm": 25.0, "y_mm": 9.0, "routing_mode": 2})
    assert env._action_history[0]["action_type"] == ACT_MAKE_LINE

    env.restore(ckpt)
    assert list(env._action_history) == saved
    env.release_checkpoint(ckpt)


def test_edge_action_override(env):
    """edge_action is appended as the newest history entry on top of the
    checkpoint's history."""
    net_id = _net_at(env, 25.0, 5.0)
    env.step({"action_type": ACT_NET_SELECT, "net_id": net_id})
    ckpt = env.checkpoint()

    edge = {"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.0, "layer": 1}
    env.restore(ckpt, edge_action=edge, edge_success=True)
    assert env._action_history[0]["action_type"] == ACT_START_ROUTE
    assert env._action_history[0]["pointer_xy"] == [25.0, 5.0]
    # edge inherits the checkpoint's net context (start_route on the selected net)
    assert env._action_history[0]["net_id"] == net_id
    # the checkpoint's own history sits underneath the edge record
    assert env._action_history[1]["action_type"] == ACT_NET_SELECT
    env.release_checkpoint(ckpt)


def test_episode_closed_nets_round_trip(env):
    """A net closed AFTER a checkpoint (as an MCTS simulation exploring net_end
    would) must NOT survive a restore — else it leaks into the committed episode
    and fires ``all_nets_closed`` early. Regression for the checkpoint that
    omitted ``_episode_closed_nets`` (root cause of MCTS episodes terminating
    early, worse at higher n_sim)."""
    net_id = _net_at(env, 25.0, 5.0)
    env.step({"action_type": ACT_NET_SELECT, "net_id": net_id})
    ckpt = env.checkpoint()
    assert env._episode_closed_nets == set()

    # simulate net_ends explored PAST the checkpoint (as interior sims do)
    env._episode_closed_nets.add(net_id)
    env._episode_closed_nets.add(999)
    assert len(env._episode_closed_nets) == 2

    env.restore(ckpt)
    assert env._episode_closed_nets == set(), "closed-nets leaked across restore"

    # the restored set must be a FRESH copy — mutating it must not poison the ckpt
    env._episode_closed_nets.add(net_id)
    env.restore(ckpt)
    assert env._episode_closed_nets == set(), "restore aliased the checkpoint's set"
    env.release_checkpoint(ckpt)


def test_reward_and_wire_baseline_round_trip(env):
    """``reward_prev_state`` (the ΔΦ/potential_diff baseline) and
    ``wire_via_ref_state`` (on_net_end accumulation baseline) round-trip, so a
    committed step measures its reward from the node — not the last simulation."""
    net_id = _net_at(env, 25.0, 5.0)
    env.step({"action_type": ACT_NET_SELECT, "net_id": net_id})
    ckpt = env.checkpoint()
    saved_prev = env._reward.prev_state          # None or a RewardState at ckpt
    assert env._wire_via_ref_state is None

    # advance the baselines past the checkpoint, then inject leaked state
    env.step({"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.0, "layer": 1})
    env.step({"action_type": ACT_MAKE_LINE, "x_mm": 25.0, "y_mm": 9.0, "routing_mode": 2})
    assert env._reward.prev_state is not saved_prev, "baseline should have advanced"
    env._wire_via_ref_state = object()           # pretend a net_end flush set it

    env.restore(ckpt)
    assert env._reward.prev_state is saved_prev, "ΔΦ baseline leaked across restore"
    assert env._wire_via_ref_state is None, "wire/via baseline leaked across restore"
    env.release_checkpoint(ckpt)
