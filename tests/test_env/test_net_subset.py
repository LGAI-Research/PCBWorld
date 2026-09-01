"""Net-subset (partial routing) — ``PCBWorld(target_nets=...)``.

Restricts the routing problem to a subset of nets: only the target nets appear
as routable targets (board_static.nets), carry ratsnest/routing_geometry, count
toward unrouted/termination, and gate DRC violations. Non-target nets' pads stay
visible as obstacles (unconnected_pads) — the router still clears from them —
but drop out of the problem definition.

Driven on the 2-net scripted board (NET1, NET2); routing helpers mirror
``test_best_board_early_stop.py``. The DRC-filter predicate is unit-tested
directly (no engine) since it is pure Python.

"""

import os
from types import SimpleNamespace

import pytest

from pcb_world.core.env import PCBWorld
from pcb_world.core.action_schema import (
    ACT_MAKE_LINE,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)
from pcb_world.engine.drc import DRCUtils
from pcb_world.engine.kicad_engine import KiCadEngine

BOARD = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "two_net_multiterm_board.kicad_pcb"
)
NET1, NET2 = 1, 2
A1, B1, C1, J1 = (10.0, 10.0), (40.0, 10.0), (25.0, 5.0), (25.0, 10.0)


def _step(env, at, x=0.0, y=0.0, layer=1, net_id=0):
    return env.step({
        "action_type": at, "x_mm": float(x), "y_mm": float(y),
        "layer": layer, "net_id": net_id, "routing_mode": 2,
    })


def _route_net1(env):
    """Fully connect NET1 (A1-B1 run + C1 tap). Returns the last step's tuple."""
    _step(env, ACT_NET_SELECT, net_id=NET1)
    _step(env, ACT_START_ROUTE, *A1)
    _step(env, ACT_MAKE_LINE, *J1)
    _step(env, ACT_START_ROUTE, *J1)
    _step(env, ACT_MAKE_LINE, *B1)
    _step(env, ACT_START_ROUTE, *C1)
    return _step(env, ACT_MAKE_LINE, *J1)   # taps the A1-B1 run → NET1 connected


# ---------------------------------------------------------------------------
# Observation filtering
# ---------------------------------------------------------------------------

def test_subset_filters_obs_json():
    env = PCBWorld(board_path=BOARD, target_nets={NET1})
    try:
        assert sorted(env.board_info.nets.keys()) == [NET1]
        assert sorted(env._routable_nets) == [NET1]
        obs, _ = env.reset(seed=0)
        assert sorted(obs["board_static"]["nets"].keys()) == ["net_1"]
        assert sorted(obs["routing_geometry"].keys()) == ["net_1"]
        # NET2's pads are demoted to obstacle-pads, not dropped entirely.
        assert len(obs["board_static"]["unconnected_pads"]) > 0
    finally:
        env.close()


def test_subset_filters_obs_indexed():
    env = PCBWorld(board_path=BOARD, obs_format="indexed", target_nets={NET1})
    try:
        obs, _ = env.reset(seed=0)
        assert [int(c) for c in obs["board_static"]["net_code"]] == [NET1]
        assert [int(c) for c in obs["routing_geometry"]["net_code"]] == [NET1]
    finally:
        env.close()


def test_subset_scopes_unrouted_count():
    """Scoped unrouted (target nets only) is strictly below the whole-board count
    on a board with an unrouted non-target net."""
    env_all = PCBWorld(board_path=BOARD)
    env_all.reset(seed=0)
    whole = env_all._engine.get_unrouted_count()
    env_all.close()

    env_sub = PCBWorld(board_path=BOARD, target_nets={NET1})
    env_sub.reset(seed=0)
    scoped = env_sub._engine.get_unrouted_count()
    env_sub.close()

    assert 0 < scoped < whole


# ---------------------------------------------------------------------------
# Termination — scoped completion
# ---------------------------------------------------------------------------

def test_subset_terminates_when_target_connected():
    """Routing only NET1 to completion terminates the episode under
    target_nets={NET1} (scoped unrouted → 0), even though NET2 stays unrouted."""
    env = PCBWorld(board_path=BOARD, max_steps=50, target_nets={NET1})
    env.reset(seed=0)
    _, _, terminated, _, info = _route_net1(env)
    env.close()
    assert terminated
    assert info["unrouted_count"] == 0


def test_whole_board_not_terminated_by_one_net():
    """Control: the same NET1-only route does NOT terminate the whole-board
    problem, because NET2 is still unrouted — isolates the scoping effect."""
    env = PCBWorld(board_path=BOARD, max_steps=50)   # no target_nets
    env.reset(seed=0)
    _, _, terminated, _, info = _route_net1(env)
    env.close()
    assert not terminated
    assert info["unrouted_count"] > 0


# ---------------------------------------------------------------------------
# Regression — target_nets=None is the legacy path
# ---------------------------------------------------------------------------

def test_target_nets_none_matches_baseline():
    """Explicit target_nets=None reproduces the no-arg observation exactly."""
    env_base = PCBWorld(board_path=BOARD)
    obs_base, _ = env_base.reset(seed=0)
    env_base.close()

    env_none = PCBWorld(board_path=BOARD, target_nets=None)
    obs_none, _ = env_none.reset(seed=0)
    env_none.close()

    assert obs_base["board_static"]["nets"].keys() == obs_none["board_static"]["nets"].keys()
    assert obs_base["routing_geometry"].keys() == obs_none["routing_geometry"].keys()
    assert len(obs_base["board_static"]["unconnected_pads"]) == \
        len(obs_none["board_static"]["unconnected_pads"])


# ---------------------------------------------------------------------------
# DRC filter (pure Python — no engine)
# ---------------------------------------------------------------------------

def _viol(net_names, severity=0x20, error_type="clearance", code=1):
    return SimpleNamespace(
        net_names=list(net_names), severity=severity, error_type=error_type,
        error_code=code, message="", x_mm=0.0, y_mm=0.0, layer=0,
    )


def test_drc_filter_keeps_target_involving_only():
    """Rule D3: keep a violation iff its net_names intersect the target set."""
    d = DRCUtils()
    d.set_target_net_names({"NET1"})
    d.update([
        _viol(["NET1", "NET2"]),   # target involved  → keep
        _viol(["NET1"]),           # target only       → keep
        _viol(["NET2", "NET3"]),   # no target         → drop
        _viol([]),                 # orphan (no net)   → drop
    ])
    assert d.get_violation_count() == 2
    kept = d.get_violation_counts_by_net()
    assert "NET1" in kept
    assert "NET3" not in kept


def test_drc_filter_none_keeps_all():
    """No filter (None) keeps every violation, including orphans (legacy)."""
    d = DRCUtils()
    d.update([_viol(["NET2"]), _viol([]), _viol(["NET1"])])
    assert d.get_violation_count() == 3


# ---------------------------------------------------------------------------
# Net-aware reset strip — keep pre-routed non-target nets (independent of lock)
# ---------------------------------------------------------------------------

def _preroute(eng, a, b):
    eng.set_routing_mode(2)
    eng.start_route(*a, 1)
    eng.fix_route(*b)
    eng.build_connectivity()


def test_reset_keeps_nontarget_routing():
    """With target_nets={2}, reset wipes only NET2's routing and KEEPS a
    pre-routed NET1 (not a target) — even though NET1 is unlocked. Keeping a net
    is decided by "not being re-routed", not by the lock flag."""
    env = PCBWorld(board_path=BOARD, target_nets={2})
    env.reset(seed=0)
    _preroute(env._engine, A1, B1)          # pre-existing NET1 track (non-target)
    assert any(t.net_code == 1 for t in env._engine.get_tracks())

    env.reset(seed=0)                        # net-aware strip: target {2} only
    tracks = env._engine.get_tracks()
    env.close()
    assert any(t.net_code == 1 for t in tracks)       # NET1 kept (unlocked, non-target)
    assert not any(t.net_code == 2 for t in tracks)   # NET2 (target) wiped


def test_reset_wipe_all_when_preserve_off():
    """preserve_nontarget_routing=False restores the bare-board reset: even a
    non-target pre-routed net is wiped."""
    env = PCBWorld(board_path=BOARD, target_nets={2},
                   preserve_nontarget_routing=False)
    env.reset(seed=0)
    _preroute(env._engine, A1, B1)
    assert any(t.net_code == 1 for t in env._engine.get_tracks())

    env.reset(seed=0)                        # bare-board strip
    tracks = env._engine.get_tracks()
    env.close()
    assert len(tracks) == 0                  # everything wiped


def test_reset_keep_nets_preserves_only_kept():
    """reset(options={'keep_nets': {c}}) wipes every net EXCEPT the kept set —
    the GUI 'keep' (preserve, don't re-route) semantics, independent of lock."""
    env = PCBWorld(board_path=BOARD)
    env.reset(seed=0)
    eng = env._engine
    eng.set_routing_mode(2)
    _preroute(eng, A1, B1)                        # NET1
    _preroute(eng, (10.0, 20.0), (40.0, 20.0))    # NET2
    assert {t.net_code for t in eng.get_tracks()} == {1, 2}

    env.reset(seed=0, options={"keep_nets": {1}})  # keep NET1 (unlocked), wipe NET2
    nets = {t.net_code for t in env._engine.get_tracks()}
    env.close()
    assert nets == {1}                             # NET1 preserved though never locked


def test_reset_keep_nets_empty_wipes_all():
    """An empty keep set (or none) wipes everything — legacy bare reset."""
    env = PCBWorld(board_path=BOARD)
    env.reset(seed=0)
    _preroute(env._engine, A1, B1)
    assert any(t.net_code == 1 for t in env._engine.get_tracks())
    env.reset(seed=0, options={"keep_nets": set()})
    tracks = env._engine.get_tracks()
    env.close()
    assert len(tracks) == 0


def test_delete_routing_of_nets_ignores_lock():
    """The engine strip primitive deletes by net, not by lock: a locked
    non-target net survives; the listed net is removed regardless of lock."""
    eng = KiCadEngine(BOARD)
    _preroute(eng, A1, B1)                    # NET1
    _preroute(eng, (10.0, 20.0), (40.0, 20.0))  # NET2
    eng.lock_net(1)                           # lock the net we will KEEP
    removed = eng.delete_routing_of_nets([2])
    eng.build_connectivity()
    nets = {t.net_code for t in eng.get_tracks()}
    eng.close()
    assert removed >= 1
    assert 1 in nets and 2 not in nets


# ---------------------------------------------------------------------------
# Keep-routing augmentation — PCBWorld(keep_routing_fraction=(lo, hi))
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def routed_board(tmp_path_factory):
    """The fixture board with BOTH nets fully routed, saved as its own file —
    the augmentation's precondition (a board file carrying complete routing)."""
    eng = KiCadEngine(BOARD)
    try:
        _preroute(eng, A1, B1)                          # NET1 run
        _preroute(eng, C1, J1)                          # NET1 tap onto the run
        _preroute(eng, (10.0, 20.0), (40.0, 20.0))      # NET2 run
        _preroute(eng, (25.0, 25.0), (25.0, 20.0))      # NET2 tap onto the run
        assert eng.get_unrouted_count() == 0
        path = str(tmp_path_factory.mktemp("keep_aug") / "routed_two_net.kicad_pcb")
        eng.save(path)
    finally:
        eng.close()
    return path


def test_keep_fraction_full_keeps_everything(routed_board):
    """f pinned to 1.0 → every net kept: the file routing survives reset, the
    kept nets seed born-closed, and reset info reports the sampled K."""
    env = PCBWorld(board_path=routed_board, keep_routing_fraction=(1.0, 1.0))
    _, info = env.reset(seed=0)
    tracks = {t.net_code for t in env._engine.get_tracks()}
    born = set(env._born_closed_nets)
    env.close()
    assert info["keep_nets"] == [1, 2]
    assert tracks == {1, 2}
    assert born == {1, 2}


def test_keep_nets_reproducible_from_ctor_seed(routed_board):
    """The K draw is reproducible from the CONSTRUCTOR seed alone.

    Training collectors call ``reset()`` with no seed, so before ``seed`` was
    wired through the env, np_random fell back to gymnasium's entropy seeding
    and K was unreproducible on every board reload. Same ctor seed -> same K;
    different ctor seeds must not all collapse to one draw.
    """
    def k_for(seed: int):
        env = PCBWorld(
            board_path=routed_board,
            keep_routing_fraction=(0.0, 1.0),
            seed=seed,
        )
        _, info = env.reset()      # unseeded reset — the training pattern
        env.close()
        return tuple(info["keep_nets"])

    assert k_for(4) == k_for(4)
    assert len({k_for(s) for s in range(10)}) > 1


def test_keep_fraction_half_resamples_from_pristine_file(routed_board):
    """f=0.5 keeps exactly one net (the other is wiped for re-routing). Every
    later reset reloads the board from file and cuts a FRESH K out of the
    pristine routing — mid-episode routing on the non-kept net is gone and
    the file's designer routing backs whatever the new K selects."""
    env = PCBWorld(board_path=routed_board, keep_routing_fraction=(0.5, 0.5))
    _, info = env.reset(seed=3)
    assert len(info["keep_nets"]) == 1
    kept = info["keep_nets"][0]
    other = 2 if kept == 1 else 1
    assert {t.net_code for t in env._engine.get_tracks()} == {kept}

    if other == 1:
        _preroute(env._engine, A1, B1)                     # "agent" routing
    else:
        _preroute(env._engine, (10.0, 20.0), (40.0, 20.0))
    _, info2 = env.reset(seed=99)   # reload-from-file + fresh K draw
    tracks2 = {t.net_code for t in env._engine.get_tracks()}
    env.close()
    assert len(info2["keep_nets"]) == 1
    assert tracks2 == set(info2["keep_nets"])


def test_keep_fraction_survives_damaged_kept_copper(routed_board):
    """R2 crash regression: an episode can shove — displace or even
    disconnect — KEPT copper while routing a neighbouring net, and reset has
    no in-place restore. The reload-per-reset contract must recover: damage
    the kept net's routing, then reset — no RuntimeError, and the new episode
    starts from the file's complete designer routing again."""
    env = PCBWorld(board_path=routed_board, keep_routing_fraction=(1.0, 1.0))
    _, info = env.reset(seed=0)
    assert info["keep_nets"] == [1, 2]
    env._engine.delete_routing_of_nets([1])     # simulate shove damage
    env._engine.build_connectivity()
    _, info2 = env.reset()                      # pre-fix: RuntimeError here
    tracks = {t.net_code for t in env._engine.get_tracks()}
    env.close()
    assert info2["keep_nets"] == [1, 2]
    assert tracks == {1, 2}


def test_keep_fraction_zero_is_from_scratch(routed_board):
    """lo=hi=0 → empty K: the whole board is wiped (pure from-scratch episode
    mixed into the augmentation range)."""
    env = PCBWorld(board_path=routed_board, keep_routing_fraction=(0.0, 0.0))
    _, info = env.reset(seed=0)
    n = env._engine.get_track_count()
    env.close()
    assert info["keep_nets"] == []
    assert n == 0


def test_keep_fraction_seeded_draw_reproducible(routed_board):
    """A seeded reset reproduces the K draw across fresh envs (rebuilds)."""
    def draw(seed):
        env = PCBWorld(board_path=routed_board, keep_routing_fraction=(0.0, 1.0))
        _, info = env.reset(seed=seed)
        k = info["keep_nets"]
        env.close()
        return k
    assert draw(7) == draw(7)


def test_keep_fraction_incomplete_board_raises():
    """The augmentation demands complete file routing: on the unrouted fixture
    every sampled kept net still has ratsnest after the strip → loud error
    naming the offending nets, instead of silently training on half-kept nets."""
    env = PCBWorld(board_path=BOARD, keep_routing_fraction=(1.0, 1.0))
    with pytest.raises(RuntimeError, match="not fully routed"):
        env.reset(seed=0)
    env.close()


def test_keep_fraction_conflicts_rejected(routed_board):
    """The env owns the keep set while the augmentation is on: net-subset mode,
    manual keep_nets resets, and runtime set_target_nets are all refused."""
    with pytest.raises(ValueError):
        PCBWorld(board_path=routed_board, target_nets={1},
                 keep_routing_fraction=(0.0, 0.5))
    env = PCBWorld(board_path=routed_board, keep_routing_fraction=(1.0, 1.0))
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"keep_nets": {1}})
    with pytest.raises(ValueError):
        env.set_target_nets({1})
    env.close()
