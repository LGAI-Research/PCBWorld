"""Config readback: the values the env and the engine actually run with.

``PCBWorld`` reads every ``EnvConfig`` field with a declared readback back from
the sink that holds it in its own form (``PCBWorld.effective_config``) and refuses
to construct when one differs from the config. The engine knobs are read from the
C++ router through ``KiCadEngine``'s getters — over the IPC proxy (default) and
in-process — so a knob dropped on the way, or a stale proxy cache, is visible.
"""
from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path

import pytest

from configs.loader.schema import EnvConfig
from pcb_world.engine import engine_available

BOARD = str(Path(__file__).resolve().parent.parent / "fixtures" / "jumanji"
            / "synth_1L_grid10_5net_v15" / "test" / "board_00000.kicad_pcb")

pytestmark = pytest.mark.skipif(not engine_available(),
                                reason="kicad_rl_router binding not built")

#: Every engine knob read back from the engine, away from its YAML value.
_ENGINE_KNOBS = dict(engine_seed=5, shove_iter_limit=9, followbranch_iter_limit=11,
                     simplify_outline=True)


def _break_followbranch_readback(monkeypatch):
    """Make the engine report a followBranch bound other than the one asked for."""
    from pcb_world.engine import KiCadEngine

    monkeypatch.setattr(KiCadEngine, "followbranch_iter_limit", property(lambda self: 1))


def test_engine_knob_getters_round_trip(engine_mode):
    from tests.helpers.engine import open_engine

    engine = open_engine(BOARD, **_ENGINE_KNOBS)
    try:
        assert {k: getattr(engine, k) for k in _ENGINE_KNOBS} == _ENGINE_KNOBS
        # The shove bound is runtime-mutable: a read after the setter sees the new
        # value (the IPC query cache must not hand back the construction value).
        engine._r.set_shove_iter_limit(17)
        assert engine.shove_iter_limit == 17
    finally:
        engine.close()


@pytest.mark.parametrize("seed", [2**31, -2**31 - 1, 1.0, "5", True])
def test_engine_seed_outside_the_cpp_int_is_refused(seed):
    """Before: a float or out-of-range seed reached the binding (or was coerced);
    now it is refused by name before a router is built."""
    from tests.helpers.engine import open_engine

    with pytest.raises(ValueError, match="engine_seed must be an int"):
        open_engine(BOARD, engine_seed=seed)


def test_effective_config_reads_every_declared_field_back(engine_mode):
    from pcb_world.core.env import PCBWorld

    cfg = EnvConfig.with_overrides(
        **_ENGINE_KNOBS, max_steps=7, action_history_len=3, keep_routing_fraction=[0.2, 0.4],
        drc_promoted_codes=[13], drc_incremental=False)
    env = PCBWorld(BOARD, cfg)
    try:
        declared = {f.name for f in fields(EnvConfig) if f.metadata["readback"] is not None}
        assert declared == {*_ENGINE_KNOBS, "action_history_len", "keep_routing_fraction",
                            "drc_promoted_codes", "drc_incremental"}
        assert env.effective_config() == {**_ENGINE_KNOBS, "action_history_len": 3,
                                          "keep_routing_fraction": (0.2, 0.4),
                                          "drc_promoted_codes": (13,), "drc_incremental": False}
    finally:
        env.close()


def test_effective_config_after_close_names_the_cause():
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(BOARD, EnvConfig.with_overrides(max_steps=5))
    env.close()
    with pytest.raises(RuntimeError, match="closed PCBWorld"):
        env.effective_config()


def test_readback_mismatch_refuses_the_env_and_releases_its_router(monkeypatch):
    """A sink holding another value than the config is an error at construction,
    and the refused env closes its engine: the next engine opens in-process while
    the error (whose traceback still references the refused env) is held — the
    one-live-router guard would refuse it if that env's router were alive."""
    from pcb_world.core.env import PCBWorld
    from tests.helpers.engine import open_engine

    monkeypatch.setenv("KICAD_ENGINE_IPC", "0")
    _break_followbranch_readback(monkeypatch)
    with pytest.raises(RuntimeError, match=r"readback mismatch.*followbranch_iter_limit") as err:
        PCBWorld(BOARD, EnvConfig.with_overrides(max_steps=5))
    open_engine(BOARD).close()
    del err


def test_keep_routing_reload_rechecks_the_engine_readbacks(monkeypatch):
    """Every reset after the first rebuilds the engine (keep-routing reload); the
    rebuilt engine's readbacks are checked again, the env-side ones are not."""
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(BOARD, EnvConfig.with_overrides(keep_routing_fraction=[0.0, 0.0]))
    try:
        env.reset(seed=0)
        env._action_history = type(env._action_history)(maxlen=99)   # env-side: not rechecked
        env.reset()
        _break_followbranch_readback(monkeypatch)
        with pytest.raises(RuntimeError, match=r"readback mismatch.*followbranch_iter_limit"):
            env.reset()
        assert env._engine is None                                  # closed on the mismatch
    finally:
        env.close()


def test_adaptive_max_steps_moves_the_env_config_with_the_cap():
    """Before: the LLM adapter set ``PCBWorld.max_steps`` alone, leaving the env's
    own config at the old cap."""
    from methods.llm_agent.wrappers.adapter import KiCadLLMWrapper

    worker = KiCadLLMWrapper(board_path=BOARD, seed=0,
                             env_config=EnvConfig.with_overrides(max_steps=5),
                             state_format="json", prompt_version="latest")
    try:
        assert worker.apply_adaptive_max_steps(13) == 13
        assert worker.env.max_steps == worker.env._env_config.max_steps == 13
        assert worker._env_config == worker.env._env_config
    finally:
        worker.env.close()


def test_router_cpp_hash_is_the_loaded_build_stamp():
    """The stamp of the router this process loads; that is this tree's C++ hash unless
    the provenance guard was told to accept a mismatched router."""
    from pcb_world.engine import engine_home, router_lib_dir
    from pcb_world.engine.provenance import (ALLOW_ENV, _read_lib_stamp, router_cpp_hash,
                                             source_hash)

    loaded = router_cpp_hash()
    assert loaded == _read_lib_stamp(router_lib_dir())
    if os.environ.get(ALLOW_ENV) != "1":
        assert loaded == source_hash(engine_home())
