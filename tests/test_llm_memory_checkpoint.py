"""LLM-layer memory snapshot/restore for MCTS (checkpoint L2).

The branch-agnostic L1 core (``PCBWorld.checkpoint/restore``) captures
board+config+session. The LLM raw-action memory is kept separate and synced at
the manager layer via ``snapshot_memory`` / ``restore_memory``. These tests
exercise that L2 pair in isolation (bypassing the heavy manager __init__).
"""

import pytest

from methods.llm_agent.training.manager import KiCadLLMRolloutManager
from methods.llm_agent.wrappers.memory import SimpleMemory


@pytest.fixture
def mgr():
    """Bare manager carrying only the per-env L2 state (no full __init__)."""
    m = KiCadLLMRolloutManager.__new__(KiCadLLMRolloutManager)
    m.memory = SimpleMemory()
    m.memory.reset(batch_size=2)
    m.memory.keys = ["action", "step"]
    m._total_steps = [0, 0]
    m._last_rejected = [None, None]
    return m


def _seed_env0(m):
    """Put env 0 into a known node state (2-step history)."""
    m.memory._data[0] = [
        {"action": "start_route 1 1", "step": 1},
        {"action": "make_line 2 2", "step": 2},
    ]
    m._total_steps[0] = 2
    m._last_rejected[0] = None


def test_snapshot_isolated_from_later_mutations(mgr):
    """A snapshot must not alias the live history — including in-place rewrites
    of earlier entries (mark_preceding_start_route_no_effect)."""
    _seed_env0(mgr)
    snap = mgr.snapshot_memory(0)

    # explore a branch past the snapshotted node
    mgr.memory._data[0].append({"action": "make_via 3 3", "step": 3})
    mgr._total_steps[0] = 3
    mgr._last_rejected[0] = {"body": "bad", "count": 2, "step": 4}
    # in-place rewrite of an earlier entry (the deepcopy reason)
    mgr.memory._data[0][0]["action"] = "[no effect] start_route 1 1"

    assert snap["memory"][0]["action"] == "start_route 1 1"
    assert len(snap["memory"]) == 2
    assert snap["total_steps"] == 2
    assert snap["last_rejected"] is None


def test_restore_brings_back_node_state(mgr):
    """restore reproduces the snapshotted history / counter / streak exactly."""
    _seed_env0(mgr)
    snap = mgr.snapshot_memory(0)

    mgr.memory._data[0].append({"action": "make_via 3 3", "step": 3})
    mgr._total_steps[0] = 3
    mgr._last_rejected[0] = {"body": "bad", "count": 2, "step": 4}

    mgr.restore_memory(0, snap)
    assert [r["action"] for r in mgr.memory._data[0]] == [
        "start_route 1 1", "make_line 2 2",
    ]
    assert mgr._total_steps[0] == 2
    assert mgr._last_rejected[0] is None


def test_repeated_restore_does_not_alias(mgr):
    """The same snapshot can be restored on repeated MCTS visits; mutating the
    live memory afterwards must not corrupt the snapshot."""
    _seed_env0(mgr)
    snap = mgr.snapshot_memory(0)

    mgr.restore_memory(0, snap)
    mgr.memory._data[0][0]["action"] = "MUTATED"
    # snapshot still pristine -> a second restore is still correct
    assert snap["memory"][0]["action"] == "start_route 1 1"

    mgr.restore_memory(0, snap)
    assert mgr.memory._data[0][0]["action"] == "start_route 1 1"


def test_other_env_untouched(mgr):
    """snapshot/restore of env 0 leaves env 1's state alone."""
    _seed_env0(mgr)
    snap = mgr.snapshot_memory(0)
    mgr.restore_memory(0, snap)
    assert mgr.memory._data[1] == []
    assert mgr._total_steps[1] == 0
    assert mgr._last_rejected[1] is None
