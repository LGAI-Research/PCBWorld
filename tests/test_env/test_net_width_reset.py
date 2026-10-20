"""Router sizes come from the start item's net class on EVERY ``start_route``.

The engine imports track width / via diameter / via drill / clearance from the
net class of the item a route starts on (pcbnew's ``ImportSizes``); nothing
carries over between routing sessions. The regression this pins: a Default-class
net routed after a wide-class net inherited the previous net's width, and a 1 mm
width left over made every line out of a fine-pitch pad fail (found on d3b
0291_C-BISCUIT_buck-reg-5v: 1.5 mm guide class, then a 0.25 mm Default net
refused ~500 times).

``set_track_width`` / ``set_via_diameter`` / ``set_via_drill`` are an override
for the CURRENT session only (called after ``start_route``), and reject values
<= 0.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from configs.loader.schema import EnvConfig
from pcb_world.core.action_schema import (
    ACT_MAKE_LINE,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)
from pcb_world.engine import engine_available

JUMANJI_DIR = (Path(__file__).resolve().parent.parent
               / "fixtures" / "jumanji" / "synth_1L_grid10_5net_v15" / "test")

SIZE_KEYS = ("track_width", "via_diameter", "via_drill", "clearance")
DEFAULT = {"track_width": 0.3, "clearance": 0.2, "via_diameter": 0.6, "via_drill": 0.3}
# NET2: every size larger than Default.
WIDE = {"track_width": 1.0, "clearance": 0.25, "via_diameter": 0.9, "via_drill": 0.5}
# NET3: wider track but a SMALLER via than Default.
MIXED = {"track_width": 1.0, "clearance": 0.2, "via_diameter": 0.5, "via_drill": 0.25}
DEFAULT_NET, WIDE_NET, MIXED_NET = 1, 2, 3

NETFREE_RULE = """\
name: netfree_test
actions:
  net_select:
    when: {has_net: false, is_routing: false}
  start_route:
    when: {has_net: true, is_routing: false}
  net_end:
    when: {has_net: true, is_routing: false}
  make_line:
    when: {is_routing: true}
  make_via:
    when: {is_routing: true}
"""


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    """Jumanji board + rewritten pro: Default / Wide (NET2) / Mixed (NET3) classes, every class
    value at or above the board minimum of the same quantity."""
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")
    if not (JUMANJI_DIR / "board_00000.kicad_pcb").exists():
        pytest.skip(f"jumanji fixture missing: {JUMANJI_DIR}")
    tmp = tmp_path_factory.mktemp("net_width_reset_board")
    pcb = tmp / "b.kicad_pcb"
    shutil.copy(JUMANJI_DIR / "board_00000.kicad_pcb", pcb)

    pro = json.loads((JUMANJI_DIR / "board_00000.kicad_pro").read_text())
    base = pro["net_settings"]["classes"][0]
    pro["net_settings"]["classes"] = [
        dict(base, name="Default", **DEFAULT),
        dict(base, name="Wide", **WIDE),
        dict(base, name="Mixed", **MIXED),
    ]
    pro["net_settings"]["netclass_patterns"] = [
        {"netclass": "Wide", "pattern": "NET2"},
        {"netclass": "Mixed", "pattern": "NET3"},
    ]
    pro["board"]["design_settings"]["rules"].update({
        "min_clearance": 0.15,
        "min_track_width": 0.1,
        "min_via_diameter": 0.4,
        "min_through_hole_diameter": 0.2,
    })
    (tmp / "b.kicad_pro").write_text(json.dumps(pro, indent=2))
    return pcb


@pytest.fixture
def env(board, tmp_path):
    from pcb_world.core.env import PCBWorld

    rule = tmp_path / "netfree_test.yaml"
    rule.write_text(NETFREE_RULE)
    e = PCBWorld(str(board), EnvConfig.with_overrides(max_steps=50, masking_rule=str(rule)))
    yield e
    e.close()


@pytest.fixture
def engine(env):
    """The env's real engine after a reset, plus ``start(net)``: start a route on the first pad of
    ``net`` (pad coordinates read from the observation)."""
    obs, _ = env.reset()
    eng = env._engine

    def start(net_id: int) -> None:
        if eng.is_routing():
            eng.cancel_route()
        pad = next(iter(obs["board_static"]["nets"][f"net_{net_id}"]["pads"].values()))
        x, y = pad["center"]["xy"]
        assert eng.start_route(x, y, pad["layer"] or 1), f"start_route failed on net {net_id}"

    eng.start = start
    return eng


def _sizes(eng) -> dict:
    return {k: eng.get_route_sizes()[k] for k in SIZE_KEYS}


def _step(env, action_type, **kw):
    action = {"action_type": action_type, "x_mm": 0.0, "y_mm": 0.0, "layer": 1,
              "net_id": 0, "routing_mode": 2}
    action.update(kw)
    _, _, _, _, info = env.step(action)
    assert info["action_success"], (action, info)


def test_default_net_after_wide_net_gets_the_default_sizes_back(env):
    """Episode-level regression: route the wide net, end it, start a Default net — the router
    holds the Default class sizes, not the previous net's."""
    obs, _ = env.reset()

    def pads(net_id):
        return [(p["center"]["xy"], p["layer"] or 1)
                for p in obs["board_static"]["nets"][f"net_{net_id}"]["pads"].values()]

    (a, la), (b, lb) = pads(WIDE_NET)[:2]
    _step(env, ACT_NET_SELECT, net_id=WIDE_NET)
    _step(env, ACT_START_ROUTE, x_mm=a[0], y_mm=a[1], layer=la)
    assert _sizes(env._engine) == pytest.approx(WIDE)
    _step(env, ACT_MAKE_LINE, x_mm=b[0], y_mm=b[1], layer=lb)
    assert not env._engine.is_routing(), "the wide net's line must complete the session"
    _step(env, ACT_NET_END)

    (c, lc), _ = pads(DEFAULT_NET)[:2]
    _step(env, ACT_NET_SELECT, net_id=DEFAULT_NET)
    _step(env, ACT_START_ROUTE, x_mm=c[0], y_mm=c[1], layer=lc)
    assert _sizes(env._engine) == pytest.approx(DEFAULT)     # was: 1 mm kept from the wide net


def test_each_size_goes_back_to_default_independently(engine):
    """Wider track + smaller via class, then a Default net: each of the three sizes returns to
    Default — a leftover larger width gets lines refused, a leftover smaller via silently breaks
    the rule."""
    engine.start(MIXED_NET)
    held = _sizes(engine)
    assert held == pytest.approx(MIXED)
    assert held["track_width"] > DEFAULT["track_width"]
    assert held["via_diameter"] < DEFAULT["via_diameter"]
    assert held["via_drill"] < DEFAULT["via_drill"]

    engine.start(DEFAULT_NET)
    held = _sizes(engine)
    for k in ("track_width", "via_diameter", "via_drill"):
        assert held[k] == pytest.approx(DEFAULT[k]), k


def test_session_override_is_gone_after_the_next_start_route(engine):
    """``set_*`` after ``start_route`` applies to that session only; the next ``start_route``
    re-imports the net class values (same net and another net alike)."""
    engine.start(DEFAULT_NET)
    engine.set_track_width(0.7)
    engine.set_via_diameter(1.1)
    engine.set_via_drill(0.6)
    held = _sizes(engine)
    assert held["track_width"] == pytest.approx(0.7)
    assert held["via_diameter"] == pytest.approx(1.1)
    assert held["via_drill"] == pytest.approx(0.6)

    engine.start(DEFAULT_NET)
    assert _sizes(engine) == pytest.approx(DEFAULT)
    engine.set_track_width(0.7)
    engine.start(WIDE_NET)
    assert _sizes(engine) == pytest.approx(WIDE)


@pytest.mark.parametrize("restore", ["restore", "restore_incremental"])
def test_restore_brings_back_the_checkpointed_session_sizes(engine, restore):
    """A checkpoint taken inside a session restores that session's sizes, the ``set_*`` override
    included — re-opening the session must not re-import the net class values over it."""
    engine.start(DEFAULT_NET)
    engine.set_track_width(0.7)
    engine.set_via_drill(0.35)
    handle = engine.checkpoint()
    engine.start(WIDE_NET)
    assert _sizes(engine) == pytest.approx(WIDE)

    assert getattr(engine, restore)(handle)
    assert engine.is_routing()
    held = _sizes(engine)
    assert held["track_width"] == pytest.approx(0.7)
    assert held["via_drill"] == pytest.approx(0.35)
    assert held["via_diameter"] == pytest.approx(DEFAULT["via_diameter"])


@pytest.mark.parametrize("restore", ["restore", "restore_incremental"])
def test_restore_without_a_session_brings_back_every_size(engine, restore):
    """A checkpoint taken between sessions carries all four sizes the router holds, the
    clearance included — restoring it does not keep the clearance of a later session."""
    engine.start(WIDE_NET)
    engine.cancel_route()
    handle = engine.checkpoint()
    engine.start(DEFAULT_NET)
    engine.cancel_route()
    assert _sizes(engine) == pytest.approx(DEFAULT)

    assert getattr(engine, restore)(handle)
    assert not engine.is_routing()
    assert _sizes(engine) == pytest.approx(WIDE)


@pytest.mark.parametrize("setter", ["set_track_width", "set_via_diameter", "set_via_drill"])
def test_size_setter_outside_a_session_raises(engine, setter):
    """No session open: ``start_route`` would overwrite the value without a trace, so the
    call is an error (like ``set_via_placement``), and the router's sizes stay as they were."""
    engine.start(WIDE_NET)
    engine.cancel_route()
    with pytest.raises(RuntimeError, match="while a route is in progress"):
        getattr(engine, setter)(0.7)
    assert _sizes(engine) == pytest.approx(WIDE)


@pytest.mark.parametrize("setter", ["set_track_width", "set_via_diameter", "set_via_drill"])
@pytest.mark.parametrize("value", [0, -0.1])
def test_non_positive_size_raises(engine, setter, value):
    """A size <= 0 is an error, never a silent no-op (``set_track_width(0)`` does not mean
    "use the design rules"), and leaves the session's sizes untouched."""
    engine.start(DEFAULT_NET)
    with pytest.raises(ValueError, match="must be > 0"):
        getattr(engine, setter)(value)
    assert _sizes(engine) == pytest.approx(DEFAULT)


def test_route_item_radius_follows_the_router_in_a_session_and_the_class_outside(engine):
    """``route_item_radius_mm`` (make_line/make_via arrival and pad-graze radius): inside a
    session it is half of what the router holds — a session override included —, outside one
    half of the current net's class value."""
    engine.start(WIDE_NET)
    assert engine.route_item_radius_mm(for_via=False) == pytest.approx(WIDE["track_width"] / 2)
    assert engine.route_item_radius_mm(for_via=True) == pytest.approx(WIDE["via_diameter"] / 2)
    engine.set_via_diameter(1.3)
    assert engine.route_item_radius_mm(for_via=True) == pytest.approx(0.65)
    engine.cancel_route()
    # Outside a session there is no session net: the Default class.
    assert engine.get_current_net_code() == -1
    assert engine.route_item_radius_mm(for_via=False) == pytest.approx(DEFAULT["track_width"] / 2)
    assert engine.route_item_radius_mm(for_via=True) == pytest.approx(DEFAULT["via_diameter"] / 2)


@pytest.mark.parametrize("restore", ["restore", "restore_incremental"])
def test_restore_brings_back_via_placement(engine, restore):
    """Via placement on at checkpoint time is on again in the re-opened session."""
    engine.start(WIDE_NET)
    engine.set_via_placement(True)
    handle = engine.checkpoint()
    engine.cancel_route()
    assert getattr(engine, restore)(handle)
    assert engine.is_routing() and engine.is_placing_via()
    assert engine.get_current_net_code() == WIDE_NET
