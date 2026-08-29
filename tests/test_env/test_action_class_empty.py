"""Regression tests for the ``valid_empty`` action class (requires C++ router).

PNS ``FixRoute`` reports success for dispatches that commit nothing to the
board — the env must classify those as ``valid_empty`` (invalid_action_penalty)
instead of ``valid_effective``:

1. make_line exactly retracing an existing same-net segment: ``NODE::Add``
   silently rejects the redundant segments but FixRoute still returns true.
2. make_via at the current route head: the lone pending via never lands on
   the board yet FixRoute returns true.

Contrast: a normal make_via away from the head commits segment + via and must
stay ``valid_effective`` (a committed via is progress, never a no-op — the
empty check compares via_count alongside track_count / wirelength).

Case 2's sibling is a dispatch failure rather than an empty step: aiming
make_via at a thru-hole pad centre is refused by ``pad_block_reason`` before the
router is touched, because the pad already spans every copper layer and PNS
would drop the via.

Since v0.30 the router also refuses AT COMMIT TIME: ``fixRoute`` checks that the
head still ends with a via before ``CommitRouting``, so a drop cause the
pre-check does not model can no longer leave a committed route behind an action
reported as failed. ``make_via``'s own via_count check is a safety net now, not
the mechanism. Board scalars are therefore unchanged in BOTH failure paths —
which is what the assertions below pin.
"""

import textwrap

import pytest

from pcb_world.engine import engine_available

ACT_NET_SELECT = 0
ACT_START_ROUTE = 1
ACT_MAKE_LINE = 3
ACT_MAKE_VIA = 4

# Two SMD pads on NET1 at (10,10) / (110,50), F.Cu, 130x60 board — a straight
# unobstructed corridor so the retrace is exactly reproducible.
BOARD_TEMPLATE = textwrap.dedent("""\
(kicad_pcb
  (version 20241229)
  (generator "test_action_class_empty")
  (generator_version "9.0.5")
  (general (thickness 1.6) (legacy_teardrops no))
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (44 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0) (allow_soldermask_bridges_in_footprints no))
  (net 0 "") (net 1 "NET1")
  (net_class "Default" "Default net class"
    (clearance 0.2) (trace_width 0.2)
    (via_dia 0.6) (via_drill 0.3) (uvia_dia 0.3) (uvia_drill 0.1))
  (footprint "SamplePad:FCu" (layer "F.Cu") (at 10 10)
    (uuid "00000000-0000-0000-0000-000000000001")
    (property "Reference" "P1" (at 0 -1) (layer "F.SilkS")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (property "Value" "Pad1" (at 0 1) (layer "F.Fab")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (pad "1" smd roundrect (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25)
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa01")))
  (footprint "SamplePad:FCu" (layer "F.Cu") (at 110 50)
    (uuid "00000000-0000-0000-0000-000000000002")
    (property "Reference" "P2" (at 0 -1) (layer "F.SilkS")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (property "Value" "Pad2" (at 0 1) (layer "F.Fab")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (pad "1" smd roundrect (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25)
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa02")))
  (gr_rect (start 0.0 0.0) (end 130.0 60.0)
    (stroke (width 0.15) (type solid)) (fill none) (layer "Edge.Cuts")
    (uuid "00000000-0000-0000-0000-0000000000ee")))
""")


# Same corridor, but the second pad is THRU-HOLE: a via aimed at its centre is
# redundant (the pad already spans F.Cu..B.Cu) and PNS drops it.
THT_BOARD_TEMPLATE = BOARD_TEMPLATE.replace(
    """    (pad "1" smd roundrect (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25)
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa02")))""",
    """    (pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 0.8)
      (layers "*.Cu" "*.Mask")
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa02")))""",
)


def _skip_if_no_kicad():
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


@pytest.fixture
def env(tmp_path):
    _skip_if_no_kicad()
    from pcb_world.core.env import PCBWorld

    board = tmp_path / "two_pad_net1.kicad_pcb"
    board.write_text(BOARD_TEMPLATE)
    # drc_dense_promoted (the KDD PPO per-step rule): step_penalty=0, so an empty
    # step's reward is exactly -invalid_action_penalty.
    from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH
    e = PCBWorld(
        board_path=str(board),
        reward_rule="drc_dense_promoted",
        use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
    )
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture
def tht_env(tmp_path):
    _skip_if_no_kicad()
    from pcb_world.core.env import PCBWorld
    from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH

    board = tmp_path / "tht_pad_net1.kicad_pcb"
    board.write_text(THT_BOARD_TEMPLATE)
    e = PCBWorld(
        board_path=str(board),
        reward_rule="drc_dense_promoted",
        use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
    )
    e.reset(seed=0)
    yield e
    e.close()


def _board_scalars(env):
    snap = env._engine.get_reward_snapshot(run_drc=False)
    return snap.track_count, snap.via_count, snap.total_wirelength


def _step(env, **action):
    a = {"action_type": action.pop("t"), **action}
    _, reward, _, _, info = env.step(a)
    return reward, info


def test_retrace_make_line_is_valid_empty(env):
    """Exact retrace: engine success=True, board unchanged -> valid_empty."""
    _step(env, t=ACT_NET_SELECT, net_id=1)
    _step(env, t=ACT_START_ROUTE, x_mm=10.0, y_mm=10.0, layer=1)
    _, info = _step(env, t=ACT_MAKE_LINE, x_mm=60.0, y_mm=10.0)
    assert info["action_class"] == "valid_effective"
    before = _board_scalars(env)
    assert before[0] > 0  # a real track was committed

    _step(env, t=ACT_START_ROUTE, x_mm=10.0, y_mm=10.0, layer=1)
    reward, info = _step(env, t=ACT_MAKE_LINE, x_mm=60.0, y_mm=10.0)

    assert info["action_success"] is True  # PNS FixRoute reported success
    assert info["action_class"] == "valid_empty"
    assert info["empty_action"] is True
    assert _board_scalars(env) == before  # board bit-identical
    invalid_pen = env._reward_config.invalid_action_penalty
    assert reward == pytest.approx(-invalid_pen)


def test_lone_via_at_head_reports_failure(env):
    """make_via at the route head places no via -> action_success False.

    PNS still returns true from FixRoute, but ``make_via`` now confirms the via
    landed (via_count delta), so the dispatch is reported as a failure rather
    than a success that quietly did nothing. Same reward either way — both
    ``valid_dispatch_fail`` and ``valid_empty`` pay invalid_action_penalty.
    """
    _step(env, t=ACT_NET_SELECT, net_id=1)
    _step(env, t=ACT_START_ROUTE, x_mm=10.0, y_mm=10.0, layer=1)
    _step(env, t=ACT_MAKE_LINE, x_mm=60.0, y_mm=10.0)
    before = _board_scalars(env)

    _step(env, t=ACT_START_ROUTE, x_mm=60.0, y_mm=10.0, layer=1)
    reward, info = _step(env, t=ACT_MAKE_VIA, x_mm=60.0, y_mm=10.0)

    assert info["action_success"] is False
    assert info["action_class"] == "valid_dispatch_fail"
    assert _board_scalars(env) == before  # no via actually landed
    invalid_pen = env._reward_config.invalid_action_penalty
    assert reward == pytest.approx(-invalid_pen)


def test_make_via_at_thru_hole_pad_commits_nothing(tht_env):
    """All-or-nothing: the via cannot land there, so the route is not committed.

    PNS would accept the route and silently drop the pending via (the pad is
    already the layer bridge), leaving a half-executed action. The handler
    refuses before the router is touched.
    """
    _step(tht_env, t=ACT_NET_SELECT, net_id=1)
    _step(tht_env, t=ACT_START_ROUTE, x_mm=10.0, y_mm=10.0, layer=1)
    before = _board_scalars(tht_env)

    _, info = _step(tht_env, t=ACT_MAKE_VIA, x_mm=110.0, y_mm=50.0)

    assert _board_scalars(tht_env) == before          # nothing committed
    assert info["action_success"] is False
    assert info["action_class"] == "valid_dispatch_fail"
    assert info["dispatch_info"]["rejected"] == "via_on_thru_pad"


def test_make_via_grazing_a_thru_hole_pad_commits_nothing(tht_env):
    """A via whose copper would only graze the pad is refused too.

    Pad r = 0.8 (1.6mm circle), via r = 0.3 -> the graze annulus is 0.8..1.1mm
    from the centre. A via at 0.9mm overlaps the pad without its centre being on
    it: connected by a copper sliver that KiCad's own cleaner deletes.
    """
    _step(tht_env, t=ACT_NET_SELECT, net_id=1)
    _step(tht_env, t=ACT_START_ROUTE, x_mm=10.0, y_mm=10.0, layer=1)
    before = _board_scalars(tht_env)

    _, info = _step(tht_env, t=ACT_MAKE_VIA, x_mm=109.1, y_mm=50.0)

    assert _board_scalars(tht_env) == before
    assert info["action_success"] is False
    assert info["dispatch_info"]["rejected"] == "pad_graze"


def test_make_via_clear_of_the_pad_still_works(tht_env):
    """The guard must not swallow a legitimate via near — but not on — the pad."""
    _step(tht_env, t=ACT_NET_SELECT, net_id=1)
    _step(tht_env, t=ACT_START_ROUTE, x_mm=10.0, y_mm=10.0, layer=1)
    tracks_before, vias_before, _ = _board_scalars(tht_env)

    _, info = _step(tht_env, t=ACT_MAKE_VIA, x_mm=108.0, y_mm=50.0)

    tracks_after, vias_after, _ = _board_scalars(tht_env)
    assert info["action_success"] is True
    assert vias_after == vias_before + 1
    assert tracks_after > tracks_before


def test_make_via_away_from_head_is_effective(env):
    """Normal via placement (segment + via) must not be penalized as empty."""
    _step(env, t=ACT_NET_SELECT, net_id=1)
    _step(env, t=ACT_START_ROUTE, x_mm=10.0, y_mm=10.0, layer=1)
    _, info = _step(env, t=ACT_MAKE_VIA, x_mm=30.0, y_mm=10.0)

    assert info["action_success"] is True
    assert info["action_class"] == "valid_effective"
    assert info["empty_action"] is False
    tracks, vias, wirelength = _board_scalars(env)
    assert tracks == 1 and vias == 1 and wirelength > 0
