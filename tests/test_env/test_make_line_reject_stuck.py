"""HLEnv (PCBWorld) make_line rejects an un-drawable route end-to-end.

Uses ``diagonal_cross_board``: a 30x30 square with a 5x5 pad snug in each corner.
NET1 connects one diagonal (TL<->BR), NET2 the other (TR<->BL). Once NET1's
diagonal is drawn, NET2's diagonal can only be routed by crossing it, and the
corner-filling pads leave no room to detour around the ends — so the second
diagonal is impossible. (PCBWorld clears pre-routed copper on load, so NET1 is
routed within the episode; the block is real trace + geometry, not a fixture
artefact.)

make_line routes via ``fix_route(reject_if_stuck=True)``, so the second diagonal
is aborted:
  * ``info["action_success"]`` is False, ``info["action_class"]`` is
    ``"valid_dispatch_fail"`` — the *action* failed. (Distinct from
    ``info["success"]`` = board completion.)
  * the board is unchanged (no dangling stub), the session is recovered,
  * no reward is granted for the no-op (``potential_diff == 0``).
"""

import os

import pytest

from pcb_world.engine import engine_available

from pcb_world.core.action_schema import (
    ACT_MAKE_LINE,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "diagonal_cross_board.kicad_pcb"
)

TL, BR = (2.5, 2.5), (27.5, 27.5)   # NET1 diagonal
TR, BL = (27.5, 2.5), (2.5, 27.5)   # NET2 diagonal (blocked once NET1 is drawn)


def _skip_if_unavailable():
    if not os.path.exists(BOARD_PATH):
        pytest.skip(f"Board not found: {BOARD_PATH}")
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


@pytest.fixture
def env():
    _skip_if_unavailable()
    from pcb_world.core.env import PCBWorld

    e = PCBWorld(board_path=BOARD_PATH, max_steps=60)
    e.reset()
    yield e
    e.close()


def _step(env, action_type, x=0.0, y=0.0, layer=1, net_id=0):
    _, _, _, _, info = env.step({
        "action_type": action_type, "x_mm": float(x), "y_mm": float(y),
        "layer": layer, "net_id": net_id, "routing_mode": 2,
    })
    return info


def test_env_first_diagonal_draws_then_second_is_rejected(env):
    # First diagonal (NET1) routes fine.
    _step(env, ACT_NET_SELECT, net_id=1)
    _step(env, ACT_START_ROUTE, *TL)
    info1 = _step(env, ACT_MAKE_LINE, *BR)
    assert info1["action_success"] is True          # first diagonal drawn

    # NET1 is now complete: net_end is the ONLY legal action on it (masking
    # keeps a finished net from being re-routed), and it is what frees the
    # selector for the next net.
    assert _step(env, ACT_NET_END)["action_success"] is True

    # Second diagonal (NET2) is blocked by NET1 + corner pads -> cannot be drawn.
    _step(env, ACT_NET_SELECT, net_id=2)
    _step(env, ACT_START_ROUTE, *TR)
    tracks_before = env._engine.get_track_count()
    info2 = _step(env, ACT_MAKE_LINE, *BL)

    assert info2["action_success"] is False         # the action failed
    assert info2["action_class"] == "valid_dispatch_fail"
    assert env._engine.get_track_count() == tracks_before  # no partial stub
    assert env._engine.is_routing() is False        # session recovered
    assert info2["potential_diff"] == 0.0           # no reward for the no-op
