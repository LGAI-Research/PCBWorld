"""Scripted-action routing E2E (requires C++ kicad_rl_router).

Drives ``PCBWorld.step`` with fixed action scripts on
``tests/fixtures/two_net_multiterm_board.kicad_pcb`` (2 nets x 3 F.Cu SMD
pads each) and asserts the *intended board result*: every step must pass the
default mask and report ``action_success``, and the committed tracks / vias /
connectivity must match the expected geometry exactly.

Why hard asserts: the pre-existing finish coverage is all soft
(``test_routing_control.py`` "may or may not succeed",
``test_integration.py`` checks only the post-step phase), so a finish/
make_line/make_via regression that can never complete a net would pass the
suite unnoticed. This file pins the contract for all three routing actions.

Geometry note: consecutive collinear ``make_line`` commits are merged by the
router into a single board segment — the expected sets below describe the
final board, not one segment per commit.

Shape-dependence contracts (same board, same net, different action choice
-> different pinned board shapes):
  - finish from each NET1 pad: ``ROUTER::Finish`` heads for the nearest
    unconnected anchor from the trace head, so the start pad decides which
    edge closes and what path it takes.
  - make_line to a directional candidate (agent's 8-dir x 0.5mm pool) vs
    straight to the terminal pad: detour bend vs single straight segment.

Board layout (mm):
  NET1 pads (10,10) (40,10) (25,5)   — routed via make_line polyline
  NET2 pads (10,20) (40,20) (25,25)  — routed F.Cu -> via -> B.Cu -> via -> F.Cu
"""

import os

import pytest

from pcb_world.engine import engine_available

from pcb_world.core.action_schema import (
    ACT_FINISH,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "two_net_multiterm_board.kicad_pcb"
)

NET1, NET2 = 1, 2

# NET1: pads A1/B1/C1, J1 = scripted bend where C1 taps the A1-B1 run.
A1, B1, C1, J1 = (10.0, 10.0), (40.0, 10.0), (25.0, 5.0), (25.0, 10.0)
# NET2: pads A2/B2/C2, V1/V2 = scripted via positions on the y=20 run.
A2, B2, C2 = (10.0, 20.0), (40.0, 20.0), (25.0, 25.0)
V1, V2 = (25.0, 20.0), (37.0, 20.0)


def _seg(a, b):
    return frozenset([a, b])


NET1_FCU_EXPECTED = {_seg(A1, B1), _seg(C1, J1)}
NET2_FCU_EXPECTED = {_seg(A2, V1), _seg(V2, B2), _seg(C2, V1)}
NET2_BCU_EXPECTED = {_seg(V1, V2)}

# First finish per start pad (NET1). A1/B1 close their edge toward C1 —
# their unambiguous nearest anchor — with mirrored 45-degree bends.
START_PAD = {"A1": A1, "B1": B1}
FINISH_SHAPE_FROM = {
    "A1": {_seg(A1, (20.0, 10.0)), _seg((20.0, 10.0), C1)},
    "B1": {_seg(B1, (30.0, 10.0)), _seg((30.0, 10.0), C1)},
}
# C1 sits exactly equidistant from A1 and B1: the nearest-anchor tie-break
# is NOT pinnable (it flips with unrelated in-process engine history), so
# the contract is only "one of the two mirror shapes".
C1_TIE_SHAPES = (
    {_seg(C1, (15.0, 5.0)), _seg((15.0, 5.0), A1)},   # toward A1
    {_seg(C1, (35.0, 5.0)), _seg((35.0, 5.0), B1)},   # toward B1
)

# make_line target choice on the A1-B1 edge: the (+1,+1)-diagonal
# directional candidate (route head + 0.5mm) forces a detour that returns
# to the y=10 run at (11,10); the direct pad target is one straight segment.
DIR_CAND = (A1[0] + 0.5, A1[1] + 0.5)
MAKE_LINE_DIRECT_SHAPE = {_seg(A1, B1)}
MAKE_LINE_DETOUR_SHAPE = {
    _seg(A1, DIR_CAND), _seg(DIR_CAND, (11.0, 10.0)), _seg((11.0, 10.0), B1),
}


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
    """Mask-checked step that must succeed. Returns (obs, terminated, info)."""
    mask = env.action_masks()
    assert mask[action_type], (
        f"action {action_type} not available (mask={mask.tolist()})"
    )
    obs, _, terminated, _, info = env.step({
        "action_type": action_type, "x_mm": float(x), "y_mm": float(y),
        "layer": layer, "net_id": net_id, "routing_mode": 2,
    })
    assert info["action_success"], f"action {action_type} failed: {info}"
    return obs, terminated, info


def _remaining(env, net_code):
    eng = env._engine
    eng.build_connectivity()
    return sum(1 for e in eng.get_ratsnest() if e.net_code == net_code)


def _segments(env, net_code, board_layer):
    """Set of segment-endpoint pairs for one net on one board layer."""
    return {
        _seg((round(t.x1_mm, 2), round(t.y1_mm, 2)),
             (round(t.x2_mm, 2), round(t.y2_mm, 2)))
        for t in env._engine.get_tracks()
        if t.net_code == net_code and t.layer == board_layer
    }


def _route_net1_make_line(env):
    _step(env, ACT_NET_SELECT, net_id=NET1)
    _step(env, ACT_START_ROUTE, *A1)
    _step(env, ACT_MAKE_LINE, *J1)
    _step(env, ACT_START_ROUTE, *J1)
    _step(env, ACT_MAKE_LINE, *B1)
    _step(env, ACT_START_ROUTE, *C1)
    _, terminated, info = _step(env, ACT_MAKE_LINE, *J1)  # taps the A1-B1 run
    return terminated, info


def _route_net2_make_via(env):
    _step(env, ACT_NET_SELECT, net_id=NET2)
    _step(env, ACT_START_ROUTE, *A2)
    _step(env, ACT_MAKE_VIA, *V1)              # F.Cu seg + via, -> B.Cu
    _step(env, ACT_START_ROUTE, *V1, layer=2)
    _step(env, ACT_MAKE_VIA, *V2, layer=2)     # B.Cu seg + via, -> F.Cu
    _step(env, ACT_START_ROUTE, *V2)
    _step(env, ACT_MAKE_LINE, *B2)
    _step(env, ACT_START_ROUTE, *C2)
    _, terminated, info = _step(env, ACT_MAKE_LINE, *V1)  # 3rd terminal onto via 1
    return terminated, info


class TestFinishScript:
    """finish must actually complete nets (start_route -> finish, no manual moves)."""

    def test_finish_completes_both_nets(self, env):
        _step(env, ACT_NET_SELECT, net_id=NET1)
        assert _remaining(env, NET1) == 2
        _step(env, ACT_START_ROUTE, *A1)
        _, terminated, _ = _step(env, ACT_FINISH)
        assert not terminated
        assert _remaining(env, NET1) == 1
        _step(env, ACT_START_ROUTE, *C1)
        _step(env, ACT_FINISH)
        assert _remaining(env, NET1) == 0
        _step(env, ACT_NET_END)

        _step(env, ACT_NET_SELECT, net_id=NET2)
        _step(env, ACT_START_ROUTE, *A2)
        _step(env, ACT_FINISH)
        assert _remaining(env, NET2) == 1
        _step(env, ACT_START_ROUTE, *C2)
        # Last unrouted edge on the board: completing it ends the episode
        # (full-routing early termination) without waiting for net_end.
        _, terminated, info = _step(env, ACT_FINISH)
        assert _remaining(env, NET2) == 0
        assert terminated
        assert info["success"] is True
        assert info["unrouted_count"] == 0
        assert env._engine.get_track_count() > 0


class TestMakeLineScript:
    """make_line polyline must leave exactly the intended segments."""

    def test_polyline_exact_board_result(self, env):
        terminated, _ = _route_net1_make_line(env)
        assert not terminated  # NET2 still unrouted

        assert _remaining(env, NET1) == 0
        assert _segments(env, NET1, env._engine.layer_map.human_to_board(1)) \
            == NET1_FCU_EXPECTED
        assert env._engine.get_via_count() == 0
        # NET2 untouched.
        assert _remaining(env, NET2) == 2

        _step(env, ACT_NET_END)


class TestMakeViaScript:
    """make_via must place vias at the scripted points and route across layers."""

    def test_layer_switch_exact_board_result(self, env):
        terminated, _ = _route_net2_make_via(env)
        assert not terminated  # NET1 still unrouted

        assert _remaining(env, NET2) == 0
        fcu = env._engine.layer_map.human_to_board(1)
        bcu = env._engine.layer_map.human_to_board(2)
        assert _segments(env, NET2, fcu) == NET2_FCU_EXPECTED
        assert _segments(env, NET2, bcu) == NET2_BCU_EXPECTED

        vias = env._engine.get_vias()
        assert {(round(v.x_mm, 2), round(v.y_mm, 2)) for v in vias} == {V1, V2}
        for v in vias:
            assert v.net_code == NET2
            assert {v.top_layer, v.bottom_layer} == {fcu, bcu}
        # NET1 untouched.
        assert _remaining(env, NET1) == 2

        _step(env, ACT_NET_END)


class TestFinishStartPointShape:
    """Same 3-terminal net, different finish start pad -> different shape.

    Pins the exact board geometry the first finish leaves for each NET1
    start pad. If finish stopped depending on the start point (or its
    pathfinding changed), these break loudly.
    """

    @pytest.mark.parametrize("start", sorted(START_PAD))
    def test_first_finish_shape(self, env, start):
        _step(env, ACT_NET_SELECT, net_id=NET1)
        _step(env, ACT_START_ROUTE, *START_PAD[start])
        _, terminated, _ = _step(env, ACT_FINISH)
        assert not terminated
        assert _remaining(env, NET1) == 1  # exactly one edge closed
        fcu = env._engine.layer_map.human_to_board(1)
        assert _segments(env, NET1, fcu) == FINISH_SHAPE_FROM[start]

    def test_first_finish_from_C1_is_a_mirror_tie(self, env):
        """C1 is equidistant from A1/B1 — either mirror shape, never both."""
        _step(env, ACT_NET_SELECT, net_id=NET1)
        _step(env, ACT_START_ROUTE, *C1)
        _step(env, ACT_FINISH)
        assert _remaining(env, NET1) == 1
        fcu = env._engine.layer_map.human_to_board(1)
        assert _segments(env, NET1, fcu) in C1_TIE_SHAPES

    def test_start_pads_give_pairwise_distinct_shapes(self):
        shapes = list(FINISH_SHAPE_FROM.values()) + list(C1_TIE_SHAPES)
        assert all(
            a != b for i, a in enumerate(shapes) for b in shapes[i + 1:]
        ), "start-point dependence collapsed: two start pads pin the same shape"


class TestMakeLineTargetChoiceShape:
    """make_line via a directional candidate vs straight to the terminal pad
    must leave different boards (same net, same edge, same start pad)."""

    def test_terminal_target_is_single_straight_segment(self, env):
        _step(env, ACT_NET_SELECT, net_id=NET1)
        _step(env, ACT_START_ROUTE, *A1)
        _step(env, ACT_MAKE_LINE, *B1)
        assert _remaining(env, NET1) == 1
        fcu = env._engine.layer_map.human_to_board(1)
        assert _segments(env, NET1, fcu) == MAKE_LINE_DIRECT_SHAPE

    def test_directional_candidate_target_detours(self, env):
        from pcb_world.vec.candidate_pool import (
            CTYPE_PAD,
            build_directional_candidates,
            collect_raw_candidates,
        )

        _step(env, ACT_NET_SELECT, net_id=NET1)
        obs, _, _ = _step(env, ACT_START_ROUTE, *A1)

        # Both targets must exist in the agent's live candidate pool:
        # DIR_CAND among the directional candidates built from the route
        # head, B1 as a PAD candidate.
        rh = obs["router_head"]
        dir_cands = build_directional_candidates(
            tuple(rh["current_xy"]), rh["current_layer"],
        )
        assert DIR_CAND in {
            (round(x, 2), round(y, 2)) for x, y, _l, _t in dir_cands
        }
        pool = collect_raw_candidates(obs, NET1, dir_cands)
        assert B1 in {
            (round(x, 2), round(y, 2)) for x, y, _l, ct in pool
            if ct == CTYPE_PAD
        }

        _step(env, ACT_MAKE_LINE, *DIR_CAND)
        _step(env, ACT_START_ROUTE, *DIR_CAND)
        _step(env, ACT_MAKE_LINE, *B1)

        assert _remaining(env, NET1) == 1  # same connectivity as direct...
        fcu = env._engine.layer_map.human_to_board(1)
        assert _segments(env, NET1, fcu) == MAKE_LINE_DETOUR_SHAPE
        assert MAKE_LINE_DETOUR_SHAPE != MAKE_LINE_DIRECT_SHAPE  # ...new shape


class TestMixedScriptFullBoard:
    """Full scripted episode: both nets, exact final board, early termination."""

    def test_full_script_intended_board_and_termination(self, env):
        terminated, _ = _route_net1_make_line(env)
        assert not terminated
        _, terminated, _ = _step(env, ACT_NET_END)
        assert not terminated

        # Completing NET2's last edge fully routes the board: the episode
        # must terminate as a success right there (before any net_end).
        terminated, info = _route_net2_make_via(env)
        assert terminated
        assert info["success"] is True
        assert info["unrouted_count"] == 0

        eng = env._engine
        eng.build_connectivity()
        assert len(eng.get_ratsnest()) == 0
        fcu = eng.layer_map.human_to_board(1)
        bcu = eng.layer_map.human_to_board(2)
        assert _segments(env, NET1, fcu) == NET1_FCU_EXPECTED
        assert _segments(env, NET2, fcu) == NET2_FCU_EXPECTED
        assert _segments(env, NET2, bcu) == NET2_BCU_EXPECTED
        assert eng.get_via_count() == 2
