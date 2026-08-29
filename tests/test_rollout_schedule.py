"""Unit tests for the eval rollout wave scheduler (``eval.rollout.rl``):
``plan_job_schedule`` (flat-fill + size-sort) and the shared ``count_pads``
size proxy.

Pure Python — no KiCad / torch / rollout. The ``eval.rollout.rl`` import is
deliberately function-local (deferred to test execution time) for the same
reason as ``tests/test_evaluator.py``: importing the ``eval.*`` rollout modules
at collection time pulls torch/engine and reorders init enough to tip an
unrelated native double-init landmine elsewhere in the full suite.
"""
from __future__ import annotations


def _plan():
    from eval.rollout.rl import plan_job_schedule
    return plan_job_schedule


def _board(tmp_path, idx: int, n_pads: int):
    """A BoardSpec backed by a fake .kicad_pcb file with exactly ``n_pads``
    ``(pad ...`` tokens (what ``count_pads`` counts)."""
    from methods._shared.board_loader import BoardSpec

    p = tmp_path / f"board_{idx}.kicad_pcb"
    p.write_text("(kicad_pcb\n" + "(pad 1 smd)\n" * n_pads + ")\n")
    return BoardSpec(index=idx, board_id=f"b{idx}", path=str(p))


def _pad_seq(jobs):
    """Pad count seen along the scheduled job order (via count_pads)."""
    from methods._shared.board_loader import count_pads

    return [count_pads(b.path) for b, _ in jobs]


def test_count_pads_counts_pad_tokens(tmp_path):
    from methods._shared.board_loader import count_pads

    b = _board(tmp_path, 0, 7)
    assert count_pads(b.path) == 7


def test_fills_n_envs_when_not_a_multiple(tmp_path):
    # n_envs=3 is not a multiple of n_rollouts=2; the scheduler must not
    # raise and must fill all 3 slots.
    boards = [_board(tmp_path, i, 10 + i) for i in range(3)]
    jobs, wave_size = _plan()(boards, n_rollouts=2, n_envs=3)
    assert wave_size == 3
    assert len(jobs) == 3 * 2


def test_no_n_envs_ge_n_rollouts_requirement(tmp_path):
    # n_envs < n_rollouts must still just work (no ValueError); wave is
    # bounded by n_envs.
    boards = [_board(tmp_path, i, 5) for i in range(2)]
    jobs, wave_size = _plan()(boards, n_rollouts=5, n_envs=2)
    assert wave_size == 2
    assert len(jobs) == 2 * 5


def test_job_set_is_invariant_to_n_envs(tmp_path):
    boards = [_board(tmp_path, i, 100 - i) for i in range(4)]
    expected = {(b.index, r) for b in boards for r in range(3)}
    for n_envs in (1, 2, 3, 5, 12, 100):
        jobs, _ = _plan()(boards, n_rollouts=3, n_envs=n_envs)
        got = {(b.index, r) for b, r in jobs}
        assert got == expected, n_envs
        assert len(jobs) == len(expected)  # no dropped/duplicated jobs


def test_scheduled_in_ascending_pad_order(tmp_path):
    # Input order is NOT sorted; scheduler must reorder small-first.
    boards = [
        _board(tmp_path, 0, 30),
        _board(tmp_path, 1, 10),
        _board(tmp_path, 2, 20),
    ]
    jobs, _ = _plan()(boards, n_rollouts=2, n_envs=6)
    seq = _pad_seq(jobs)
    assert seq == sorted(seq)                       # non-decreasing
    assert [b.index for b, _ in jobs] == [1, 1, 2, 2, 0, 0]


def test_equal_size_ties_keep_input_order(tmp_path):
    # Stable sort: equal pad counts preserve the given board order.
    boards = [_board(tmp_path, 5, 10), _board(tmp_path, 3, 10)]
    jobs, _ = _plan()(boards, n_rollouts=2, n_envs=4)
    assert [b.index for b, _ in jobs] == [5, 5, 3, 3]


def test_boards_per_batch_caps_wave(tmp_path):
    boards = [_board(tmp_path, i, 10) for i in range(4)]
    # Legacy knob: cap wave to boards_per_batch whole boards' rollout sets.
    _, w1 = _plan()(boards, n_rollouts=2, n_envs=8, boards_per_batch=1)
    _, w2 = _plan()(boards, n_rollouts=2, n_envs=8, boards_per_batch=2)
    assert w1 == 2   # min(8, 1*2)
    assert w2 == 4   # min(8, 2*2)
    # Cap never exceeds n_envs even if asked for more.
    _, w3 = _plan()(boards, n_rollouts=2, n_envs=3, boards_per_batch=5)
    assert w3 == 3   # min(3, 5*2)


def test_wave_size_at_least_one(tmp_path):
    boards = [_board(tmp_path, 0, 10)]
    _, w = _plan()(boards, n_rollouts=1, n_envs=1)
    assert w == 1
