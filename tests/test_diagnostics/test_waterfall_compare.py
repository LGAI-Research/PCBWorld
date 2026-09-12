"""waterfall variant-compare mode — fixed row scheme / delta% / sum-closure contract.

The row scheme is pinned to the single ROWS table (shared by single and
compare modes): building the comparison table with an ad-hoc generator
instead would let the table drift between campaigns. This test guards
compare mode's presence and its closure-assert behavior.
"""
from __future__ import annotations

import json

import pytest

from tools.diagnostics.speed_profiler import waterfall


def _prof_json(scale: float = 1.0, *, break_closure: bool = False) -> dict:
    """Minimal valid prof JSON where sum-closure holds (units: ms)."""
    collect = 10_000 * scale
    kids = dict(mask_ipc_ms=500 * scale, forward_wall_ms=4_000 * scale,
                step_barrier_ms=3_000 * scale, obs_advance_ms=100 * scale,
                collector_ms=400 * scale, reset_ms=1_000 * scale,
                unbucketed_post_loop_ms=1_000 * scale)
    if break_closure:
        kids["step_barrier_ms"] = 0.0  # gap 30% -> assert must fail
    update = 20_000 * scale
    evaluate, backward, clip, step = 12_000 * scale, 6_000 * scale, 500 * scale, 1_000 * scale
    return {
        "run": {"host": "test-node", "dataset": "d2a", "n_envs": 64,
                "batch_size": 256, "n_steps": 128, "max_steps": 128,
                "warmup_iters": 1, "measured_iters": 2},
        "fingerprint": {"main": {"gpu_name": "TestGPU", "torch_version": "0.0"}},
        "phases": {"per_phase": {
            "select_boards": {"mean": 300 * scale},
            "collect": {"mean": collect},
            "compute_targets": {"mean": 200 * scale},
            "update": {"mean": update},
        }},
        "rollout_decomp": {**kids, "forward_split": {
            "walk_cpu_ms": 2_000 * scale, "gpu_event_ms": 1_500 * scale,
            "launch_sync_resid_ms": 500 * scale}},
        "update_decomp": {
            "perf_counter_ms": {"evaluate": evaluate, "backward": backward,
                                "clip": clip, "step": step},
            "gpu_active_ms": {"fwd_pass": 4_000 * scale},
        },
    }


def _write_cell(d, prof):
    d.mkdir(parents=True, exist_ok=True)
    (d / "prof_d2a_e64_b256.json").write_text(json.dumps(prof))


def test_compare_two_dirs_fixed_rows(tmp_path):
    a, b = tmp_path / "base", tmp_path / "variant"
    _write_cell(a, _prof_json(1.0))
    _write_cell(b, _prof_json(0.5))  # every entry -50%
    out = tmp_path / "cmp.html"
    waterfall.generate_compare([str(a), str(b)], str(out), ["baseline", "variant"])
    html = out.read_text()
    # Row scheme unchanged (same ROWS labels as single mode)
    for _, lab, _, _ in waterfall.ROWS:
        assert lab in html, f"row missing: {lab}"
    assert "baseline" in html and "variant" in html
    assert "-50%" in html  # delta% shown alongside


def test_compare_closure_violation_aborts(tmp_path):
    a, b = tmp_path / "base", tmp_path / "variant"
    _write_cell(a, _prof_json(1.0))
    _write_cell(b, _prof_json(1.0, break_closure=True))
    with pytest.raises(AssertionError, match="rollout closure gap"):
        waterfall.generate_compare([str(a), str(b)], str(tmp_path / "cmp.html"))
    assert not (tmp_path / "cmp.html").exists()  # no file written on failure


def test_compare_requires_common_cells(tmp_path):
    a, b = tmp_path / "base", tmp_path / "variant"
    _write_cell(a, _prof_json(1.0))
    b.mkdir()
    (b / "prof_d2b_e64_b256.json").write_text(json.dumps(_prof_json(1.0)))
    with pytest.raises(SystemExit):
        waterfall.generate_compare([str(a), str(b)], str(tmp_path / "cmp.html"))
