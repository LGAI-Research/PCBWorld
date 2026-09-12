"""``--eval-base-seed`` / ``--seed`` actually reach the env (regression).

Previously the rollout **computed the per-slot seed and only wrote it into the
output row** without passing it to ``env.reset(seed=...)``. Validation board
sampling (the env-side per-episode draw = ``keep_routing_fraction``) was
therefore unseeded, ``val/*`` was irreproducible, and best-ckpt selection was
an argmax over uncontrolled noise.

Two segments are pinned separately:

1. transport — ``VecBackend.reset_batch(indices, seeds)`` carries the seed
   through to the worker's ``env.reset(seed=...)`` (across the process
   boundary).
2. rollout — ``_run_one_batch`` passes ``base_seed + board.index*100 +
   rollout_idx`` to reset and records the **same** value in the row.

Reproducibility of the env-side seeded reset itself is pinned separately by
``tests/test_env/test_net_subset.py``
(``test_keep_fraction_seeded_draw_reproducible``).
"""
from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------------------
# 1. transport — the subprocess pool delivers the per-slot seed to the worker
# ---------------------------------------------------------------------------


class SeedRecorderEnv:
    """Minimal env that only records ``reset(seed=...)`` (no KiCad engine needed)."""

    def reset(self, *, seed=None, options=None):
        return {"seed": seed}, {}

    def close(self):
        pass


def test_reset_batch_delivers_per_slot_seed_to_worker():
    from pcb_world.vec.backends.subproc import SubprocDecoderVecEnv

    pool = SubprocDecoderVecEnv([SeedRecorderEnv for _ in range(3)])
    try:
        out = pool.reset_batch([0, 2], [11, 22])
        assert [obs["seed"] for obs, _info in out] == [11, 22]
        # Omitting seeds leaves the reset unseeded — gymnasium treats seed=None as a no-op.
        assert pool.reset_batch([1])[0][0]["seed"] is None
    finally:
        pool.close()


def test_reset_batch_rejects_length_mismatch():
    from pcb_world.vec.backends.subproc import SubprocDecoderVecEnv

    pool = SubprocDecoderVecEnv([SeedRecorderEnv for _ in range(2)])
    try:
        with pytest.raises(ValueError, match="2 seeds for 1 indices"):
            pool.reset_batch([0], [1, 2])
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# 2. rollout — _run_one_batch passes the derived seed to reset
# ---------------------------------------------------------------------------


class _RecordingPool:
    """Fake pool implementing only the surface ``_run_one_batch`` uses."""

    def __init__(self, n_nets: int = 3) -> None:
        self.reset_calls: list[tuple[list[int], list[int] | None]] = []
        self._obs = {"board_static": {"nets": {i: {} for i in range(n_nets)}}}

    def reload_boards(self, board_paths):
        self.boards = list(board_paths)

    def reset_batch(self, indices, seeds=None):
        idxs = list(indices)
        self.reset_calls.append((idxs, None if seeds is None else list(seeds)))
        return [(self._obs, {}) for _ in idxs]


def _board(index: int):
    from methods._shared.board_loader import BoardSpec

    return BoardSpec(index=index, board_id=f"b{index}", path=f"/tmp/b{index}.kicad_pcb")


def _run(pool: _RecordingPool, jobs, base_seed: int) -> list[dict[str, Any]]:
    """Rejects every slot immediately via ``skip_incompatible`` so that only the
    reset + row-recording segment runs, without the rollout loop (no policy or
    engine needed)."""
    from methods.rl_agent.rollout.transformer import _run_one_batch

    return _run_one_batch(
        pool=pool, policy=None, device=None, jobs=jobs,
        n_envs=len(jobs), max_steps=1, base_seed=base_seed,
        deterministic=False, skip_incompatible=True, max_net_slots=0,
        early_stop_finish_no_progress=0, early_stop_no_geometry_progress=0,
        save_artifacts=False, artifacts_dir=None, final_drc=False,
        reward_config="", check_angle=0,
    )


def test_run_one_batch_seeds_the_reset():
    jobs = [(_board(0), 0), (_board(3), 2)]
    pool = _RecordingPool()
    rows = _run(pool, jobs, base_seed=1000)

    expected = [1000 + 0 * 100 + 0, 1000 + 3 * 100 + 2]
    (indices, seeds), = pool.reset_calls
    assert indices == [0, 1]
    assert seeds == expected
    # The recorded seed and the applied seed must be the same value.
    assert [int(r["seed"]) for r in rows] == expected


def test_run_one_batch_seed_tracks_base_seed():
    jobs = [(_board(1), 0)]
    a = _RecordingPool()
    b = _RecordingPool()
    _run(a, jobs, base_seed=1000)
    _run(b, jobs, base_seed=7)
    assert a.reset_calls[0][1] == [1100]
    assert b.reset_calls[0][1] == [107]
