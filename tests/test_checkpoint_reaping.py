"""Checkpoint memory reaping: explicit release + RAII GC backstop.

The C++ router holds the heavy clones behind an int handle; dropping the Python
``Checkpoint`` does not free them unless released. ``Checkpoint.release`` (called
explicitly or from ``__del__``) frees the handle; the C++ ``get_checkpoint_count``
diagnostic lets these tests assert the live-set actually shrinks.
"""

import gc

import pytest

from pcb_world.engine.kicad_engine import KiCadEngine
from pcb_world.core.env import PCBWorld

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def test_engine_count_and_release():
    """Engine-level: count tracks live checkpoints; release is idempotent."""
    e = KiCadEngine(BOARD)
    e.build_connectivity()
    assert e.checkpoint_count() == 0
    h1 = e.checkpoint()
    h2 = e.checkpoint()
    assert e.checkpoint_count() == 2
    e.release_checkpoint(h1)
    assert e.checkpoint_count() == 1
    e.release_checkpoint(h1)  # double release -> no-op
    assert e.checkpoint_count() == 1
    e.release_checkpoint(h2)
    assert e.checkpoint_count() == 0
    e.close()


@pytest.fixture
def env():
    e = PCBWorld(board_path=BOARD, max_steps=20)
    e.reset()
    yield e
    e.close()


def test_explicit_release(env):
    """env.release_checkpoint frees the C++ handle and is idempotent."""
    ck = env.checkpoint()
    assert env._engine.checkpoint_count() == 1
    env.release_checkpoint(ck)
    assert env._engine.checkpoint_count() == 0
    assert ck._released
    env.release_checkpoint(ck)  # idempotent
    assert env._engine.checkpoint_count() == 0


def test_gc_backstop_releases(env):
    """A checkpoint dropped without explicit release is freed on GC (__del__)."""
    ck = env.checkpoint()
    assert env._engine.checkpoint_count() == 1
    del ck
    gc.collect()
    assert env._engine.checkpoint_count() == 0


def test_many_checkpoints_reaped_on_gc(env):
    """Live-set returns to zero once a batch of checkpoints goes out of scope."""
    cks = [env.checkpoint() for _ in range(10)]
    assert env._engine.checkpoint_count() == 10
    del cks
    gc.collect()
    assert env._engine.checkpoint_count() == 0


def test_release_after_close_is_safe():
    """Releasing / GC-ing a checkpoint after the engine closed does not crash
    (the router already freed every handle on close)."""
    e = PCBWorld(board_path=BOARD, max_steps=20)
    e.reset()
    ck = e.checkpoint()
    e.close()           # router destroyed -> all handles gone
    ck.release()        # weakref guard -> safe no-op
    assert ck._released
    del ck
    gc.collect()        # __del__ after close -> safe
