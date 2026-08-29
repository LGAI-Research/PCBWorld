"""Verifies rule-area keepout avoidance at the PCBWorld env level.

Locks in that the engine-level contract (tests/test_engine_api/test_keepout.py)
also holds through the env.step path: routing NET1 (P1(0,0)->P2(4,0)) with
make_line must produce a committed track that does not cross the keepout rect
[1.5,-1]~[2.5,1].

Secondary contract: the keepout rule-area is exposed in
obs(board_static.obstacles) as a polygon obstacle (shape="polygon" + pts).
Routing avoidance and obstacle exposure are separate code paths, so both are
locked in here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from pcb_world.engine import engine_available

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_keepout_board.kicad_pcb"

sys.path.insert(0, str(RL_MODULE_DIR))

from tests.helpers.geometry_helpers import Rect, segment_rect_intersect  # noqa: E402
from pcb_world.core.action_schema import (  # noqa: E402
    ACT_MAKE_LINE,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)

KEEPOUT_RECT: Rect = ((1.5, -1.0), (2.5, 1.0))
P1 = (0.0, 0.0)
P2 = (4.0, 0.0)
NET1 = 1


def _skip_if_unavailable() -> None:
    if not BOARD_PATH.exists():
        pytest.skip(f"Board not found: {BOARD_PATH}")
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


@pytest.fixture
def env():
    _skip_if_unavailable()
    from pcb_world.core.env import PCBWorld

    e = PCBWorld(board_path=str(BOARD_PATH), max_steps=60)
    e.reset()
    yield e
    e.close()


def _step(env, action_type, x=0.0, y=0.0, layer=1, net_id=0):
    mask = env.action_masks()
    assert mask[action_type], f"action {action_type} not in mask {mask.tolist()}"
    obs, _, terminated, _, info = env.step({
        "action_type": action_type, "x_mm": float(x), "y_mm": float(y),
        "layer": layer, "net_id": net_id, "routing_mode": 2,
    })
    assert info["action_success"], f"action {action_type} failed: {info}"
    return obs, terminated, info


def test_env_routing_avoids_keepout(env) -> None:
    _step(env, ACT_NET_SELECT, net_id=NET1)
    _step(env, ACT_START_ROUTE, *P1)
    _step(env, ACT_MAKE_LINE, *P2)

    segs = [
        ((t.x1_mm, t.y1_mm), (t.x2_mm, t.y2_mm))
        for t in env._engine.get_tracks()
    ]
    assert len(segs) > 0, "no track was created"
    crossings = [s for s in segs if segment_rect_intersect(s, KEEPOUT_RECT)]
    assert not crossings, f"env routing crossed the keepout: {crossings}"


def test_keepout_exposed_as_polygon_obstacle(env) -> None:
    """Checks that the keepout zone is exposed as a polygon obstacle in obs.

    It carries shape="polygon" + pts (the actual outline), and a bbox
    (center/width/height) is also kept for rect-only consumers.
    """
    obs = env.reset()
    obs0 = obs[0] if isinstance(obs, tuple) else obs
    obstacles = obs0["board_static"]["obstacles"]
    assert len(obstacles) == 1, f"expected 1 keepout obstacle, got: {obstacles}"

    o = next(iter(obstacles.values()))
    assert o["shape"] == "polygon", f"expected shape=polygon: {o['shape']!r}"
    pts = [tuple(p) for p in o["pts"]]
    assert set(pts) == {(1.5, -1.0), (2.5, -1.0), (2.5, 1.0), (1.5, 1.0)}, (
        f"keepout outline mismatch: {pts}"
    )
    # bbox is also kept (center=(2,0), 1x2mm)
    assert o["center"]["xy"] == [2.0, 0.0]
    assert (o["width"], o["height"]) == (1.0, 2.0)
