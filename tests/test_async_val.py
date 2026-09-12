"""--async-val: cadence-ckpt discovery protocol + async logger fan-out (no engine/GPU).

Covers the trainer<->watcher contract in methods/rl_agent/training/async_val.py
(regular policy_iter ckpts + results channel: pending -> result -> poll/consume
-> train_done) and the out-of-order scalar logging split in
methods/_shared/logger.py (TensorBoard-style sinks get true-step add_scalar;
W&B-style sinks get one merged row via add_scalars_async).
"""
import json

import torch

from methods._shared.logger import MultiLogger, _ASYNC_STEP_KEY
from methods.rl_agent.training.async_val import (
    AsyncValResults,
    pending_evals,
    _TRAIN_DONE,
)


def _save_ckpt(save_dir, iteration):
    """Regular periodic-ckpt stand-in (same unpadded naming as save_periodic_ckpt)."""
    torch.save(
        {"iteration": iteration, "policy_state_dict": {"w": torch.zeros(2)},
         "args": {"eval_every": 25}},
        save_dir / f"policy_iter_{iteration}.pt",
    )


def _result_json(r: AsyncValResults, iteration: int, scalars: dict) -> None:
    """Watcher-side result write (test stand-in for watcher_main)."""
    path = r.results_dir / f"iter_{iteration:06d}.json"
    path.write_text(json.dumps(
        {"iteration": iteration, "scalars": scalars, "overall": {}}
    ))


def test_pending_is_cadence_ckpts_without_results(tmp_path):
    r = AsyncValResults(tmp_path)
    for it in (0, 10, 25, 50, 60):  # iter-0 always present + save_freq=10-style ckpts
        _save_ckpt(tmp_path, it)
    # Only multiples of cadence(25) + iter 0 (every run snapshots it) are eval targets — 10/60 are not
    assert [n for n, _ in pending_evals(tmp_path, 25)] == [0, 25, 50]
    # Once a result exists (whether consumed or not) it drops out of pending
    _result_json(r, 0, {})
    assert [n for n, _ in pending_evals(tmp_path, 25)] == [25, 50]
    r.consume(0)
    assert [n for n, _ in pending_evals(tmp_path, 25)] == [25, 50]


def test_result_roundtrip_keeps_ckpt(tmp_path):
    r = AsyncValResults(tmp_path)
    _save_ckpt(tmp_path, 25)
    _result_json(r, 25, {"val/fp_mean_of_means": 0.5})
    results = r.poll_results()
    assert len(results) == 1 and results[0]["iteration"] == 25
    r.consume(25)
    # The result becomes a .done audit file; the regular ckpt is left in place (a resume asset — never deleted)
    assert r.poll_results() == [] and r.n_pending(25) == 0
    assert (r.results_dir / "iter_000025.json.done").exists()
    assert r.ckpt_path(25).exists()


def test_poll_results_ordered(tmp_path):
    r = AsyncValResults(tmp_path)
    for it in (50, 25):
        _save_ckpt(tmp_path, it)
        _result_json(r, it, {})
    assert [x["iteration"] for x in r.poll_results()] == [25, 50]
    # A completed-but-not-yet-consumed eval is not pending (drain is judged by result presence)
    assert r.n_pending(25) == 0


def test_train_done_marker_and_stale_reset(tmp_path):
    r = AsyncValResults(tmp_path)
    r.mark_train_done()
    assert (r.dir / _TRAIN_DONE).exists()
    # a new trainer on the same save_dir (--resume) clears the stale marker
    r2 = AsyncValResults(tmp_path)
    assert not (r2.dir / _TRAIN_DONE).exists()


class _PlainSink:
    """TensorBoard-style sink: add_scalar only."""

    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))


class _AsyncSink(_PlainSink):
    """W&B-style sink: dedicated async batch entrypoint."""

    def __init__(self):
        super().__init__()
        self.async_calls = []

    def add_scalars_async(self, scalars, step):
        self.async_calls.append((dict(scalars), step))


def test_multilogger_async_fanout():
    plain, batched = _PlainSink(), _AsyncSink()
    ml = MultiLogger([plain, batched])
    ml.add_scalars_async({"val/a": 1.0, "val/b": 2.0}, step=7)
    # plain sink: per-scalar at the TRUE (past) step
    assert sorted(plain.calls) == [("val/a", 1.0, 7), ("val/b", 2.0, 7)]
    # async-capable sink: one batched call, no add_scalar fallback
    assert batched.async_calls == [({"val/a": 1.0, "val/b": 2.0}, 7)]
    assert batched.calls == []


class _FakeWandb:
    def __init__(self):
        self.logged = []
        self.defined = []

    def define_metric(self, pattern, step_metric):
        self.defined.append((pattern, step_metric))

    def log(self, row, step=None):
        self.logged.append((dict(row), step))


def test_wandb_async_merges_into_current_step():
    from methods._shared.logger import WandbLogger

    fake = _FakeWandb()
    wb = WandbLogger(fake)
    wb.add_scalar("train/loss", 0.1, step=50)  # trainer is at iter 50
    wb.add_scalars_async({"val/fp": 0.3, "val_d3b/fp": 0.2}, step=42)

    # late result rides on the CURRENT step (W&B drops step<current writes),
    # carrying its true x in the async step field
    row, step = fake.logged[-1]
    assert step == 50
    assert row[_ASYNC_STEP_KEY] == 42
    assert row["val/fp"] == 0.3 and row["val_d3b/fp"] == 0.2
    # each prefix routed to the async x-axis exactly once
    assert sorted(fake.defined) == [
        ("val/*", _ASYNC_STEP_KEY), ("val_d3b/*", _ASYNC_STEP_KEY),
    ]
    wb.add_scalars_async({"val/fp": 0.4}, step=45)
    assert sorted(fake.defined) == [
        ("val/*", _ASYNC_STEP_KEY), ("val_d3b/*", _ASYNC_STEP_KEY),
    ]


# ---------------------------------------------------------------------------
# Teardown gating: only a COMPLETED fit may signal train_done (260825 incident
# — a crashed run left train_done behind, the watcher exited, and the
# restarted cell ran with no watcher at all)
# ---------------------------------------------------------------------------

class _StubTrainer:
    """Minimal Trainer subclass: setup/teardown no-ops, iteration behaviour injected."""

    def __init__(self, *, fail_at: int | None, iterations: int = 3):
        from methods._shared.trainer.base import Trainer

        class _T(Trainer):
            def setup(_s): pass
            def train_iteration(_s, iteration):
                if fail_at is not None and iteration == fail_at:
                    raise RuntimeError("boom")
                return {}
            def teardown(_s): pass
        self.t = _T(iterations=iterations)


def test_fit_completed_flag_true_on_normal_finish():
    st = _StubTrainer(fail_at=None)
    st.t.fit()
    assert st.t.fit_completed is True


def test_fit_completed_flag_false_on_exception():
    import pytest as _pt
    st = _StubTrainer(fail_at=2)
    with _pt.raises(RuntimeError, match="boom"):
        st.t.fit()
    assert st.t.fit_completed is False


class _FakeAsyncVal:
    def __init__(self, d): self.dir = d; self.marked = False
    def mark_train_done(self): self.marked = True


def _finish(fit_completed: bool, tmp_path):
    """Call PPOTrainer._finish_async_val on a bare stand-in (no engine/GPU)."""
    from methods.rl_agent.training.loop import PPOTrainer
    stub = type("S", (), {})()
    stub.async_val = _FakeAsyncVal(tmp_path)
    stub.fit_completed = fit_completed
    stub.drained = False
    stub._drain_async_results = lambda: setattr(stub, "drained", True)
    PPOTrainer._finish_async_val(stub)
    return stub


def test_finish_async_val_marks_done_only_when_completed(tmp_path, capsys):
    ok = _finish(True, tmp_path)
    assert ok.async_val.marked and ok.drained
    ab = _finish(False, tmp_path)
    assert not ab.async_val.marked and not ab.drained
    assert "ABORTED" in capsys.readouterr().out


def test_load_recorded_val_env(tmp_path, capsys):
    """Input to the watcher's env-records cross-check: load the record, or return
    None with a loud skip when it is absent."""
    from methods.rl_agent.training.async_val import load_recorded_val_env

    assert load_recorded_val_env(tmp_path) is None      # pre-records save_dir
    assert "cross-check" in capsys.readouterr().out

    rec = tmp_path / "env_records"
    rec.mkdir()
    (rec / "val_env.yaml").write_text("max_steps: 20\nseed: 7\n")
    assert load_recorded_val_env(tmp_path) == {"max_steps": 20, "seed": 7}
