"""Incremental == full restore equivalence through the REAL PCBWorld action path.

The engine-level tests (test_checkpoint_incremental.py / test_checkpoint_restore.py)
exercise the C++ restore primitive in isolation, but reach their states with raw
engine calls the env never makes — e.g. ``delete_track_by_index`` to shrink the
board (PCBWorld has no delete action; MCTS shrinks only by restoring to a shallower
node), and ``fix_route(force_finish=False)`` open mid-route sessions (PCBWorld's
make_line/make_via always force_finish). They also bypass the L1 layer
(step_count / dispatcher / action_history).

This file validates restore on the path actually used by MCTS/PPO: a realistic
trajectory of ``PCBWorld`` actions (net_select / start_route / make_line /
make_via), checkpointing at every step, then restoring to those checkpoints in a
jumbled order (forcing both forward = adds-wiring and backward = removes-wiring
transitions). At each visit the fast ``incremental=True`` restore must match both
the trusted full-swap ``incremental=False`` restore and the state recorded when
the checkpoint was taken.
"""

import pytest

from pcb_world.core.env import PCBWorld
from pcb_world.core.action_schema import (
    ACT_NET_SELECT, ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_NET_END,
)

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def _net_at(env, x, y):
    for p in env._engine.get_pads():
        if abs(p.x_mm - x) < 0.6 and abs(p.y_mm - y) < 0.6 and p.net_code > 0:
            return int(p.net_code)
    raise AssertionError(f"no pad near ({x}, {y})")


def _rec(env):
    """Board + L1 state, normalized for equality (DRC is derived / lazily cleared)."""
    e = env._engine
    return dict(
        step=env._step_count,
        net=env._dispatcher.current_net_id,
        mode=env._dispatcher.routing_mode,
        routing=e.is_routing(),
        unrouted=e.get_unrouted_count(),
        tracks=sorted(
            (round(t.x1_mm, 5), round(t.y1_mm, 5), round(t.x2_mm, 5),
             round(t.y2_mm, 5), t.layer, t.net_code) for t in e.get_tracks()),
        vias=sorted(
            (round(v.x_mm, 5), round(v.y_mm, 5), v.net_code) for v in e.get_vias()),
    )


def _trajectory(env):
    """A realistic PCBWorld action list: net1 (line), net2 (line), net3 (via =>
    multi-segment). Built lazily so net ids resolve against the fresh board.

    ``net_end`` closes each finished net before the next ``net_select``: a
    completed net masks out everything but net_end (net_select needs
    ``has_net: false``), so this is the only legal way to move on — and it is
    exactly what the policy is trained to emit.
    """
    n1, n2, n3 = _net_at(env, 10, 10), _net_at(env, 10, 20), _net_at(env, 25, 5)
    env._engine.set_via_diameter(0.6)
    env._engine.set_via_drill(0.3)
    return [
        {"action_type": ACT_NET_SELECT, "net_id": n1},
        {"action_type": ACT_START_ROUTE, "x_mm": 10.0, "y_mm": 10.0, "layer": 1},
        {"action_type": ACT_MAKE_LINE, "x_mm": 40.0, "y_mm": 10.0, "routing_mode": 2},
        {"action_type": ACT_NET_END},
        {"action_type": ACT_NET_SELECT, "net_id": n2},
        {"action_type": ACT_START_ROUTE, "x_mm": 10.0, "y_mm": 20.0, "layer": 1},
        {"action_type": ACT_MAKE_LINE, "x_mm": 40.0, "y_mm": 20.0, "routing_mode": 2},
        {"action_type": ACT_NET_END},
        {"action_type": ACT_NET_SELECT, "net_id": n3},
        {"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.0, "layer": 1},
        {"action_type": ACT_MAKE_VIA, "x_mm": 25.0, "y_mm": 5.5, "routing_mode": 2},
        {"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.5, "layer": 2},
        {"action_type": ACT_MAKE_VIA, "x_mm": 25.0, "y_mm": 25.0, "routing_mode": 2},
    ]


@pytest.fixture
def built():
    """Run the trajectory, checkpointing + recording state after every step.
    Yields (env, [(handle, recorded_state), ...]) including the pre-action state."""
    env = PCBWorld(board_path=BOARD, max_steps=80)
    env.reset()
    states = [(env.checkpoint(), _rec(env))]
    for a in _trajectory(env):
        env.step(a)
        states.append((env.checkpoint(), _rec(env)))

    deep = states[-1][1]
    assert len(deep["vias"]) >= 1 and len(deep["tracks"]) >= 3   # via geometry present

    yield env, states

    for h, _ in states:
        env.release_checkpoint(h)
    if env._engine is not None and env._engine.is_routing():
        env._engine.cancel_route()
    env.close()


def test_incremental_equals_full_jumbled_visits(built):
    """Restore to checkpoints in a jumbled order (forward + backward transitions);
    incremental must equal full-swap and the recorded state every time."""
    env, states = built
    n = len(states)
    last = n - 1
    mid = n // 2
    # Order chosen to alternate deep<->shallow so each restore is a forward or
    # backward delta from the previous one.
    order = [0, last, 1, last, mid, 0, last, mid, 2, last, 0]
    order = [i for i in order if i < n]

    for i in order:
        h, recorded = states[i]
        env.restore(h, incremental=True)
        inc = _rec(env)
        env.restore(h, incremental=False)      # trusted oracle
        full = _rec(env)
        assert inc == full, f"incremental != full at checkpoint {i}"
        assert inc == recorded, f"restore did not reproduce recorded state {i}"


def test_step_after_forward_restore_is_safe(built):
    """A real env.step + DRC after a forward restore (shallow -> deep) is safe."""
    env, states = built
    env.restore(states[0][0], incremental=True)        # shallow (empty)
    env.restore(states[-1][0], incremental=True)       # forward to deepest
    env.step({"action_type": ACT_NET_SELECT, "net_id": _net_at(env, 10, 10)})
    env._engine.run_drc()                              # must not crash
