"""Preemptive peak-VRAM budget chunking (``training/mem_budget.py``).

CPU-only, mirroring ``tests/test_oom_recovery.py``: a tiny real
``KiCadRLModel`` + mock obs drive the real planner/update/rollout code paths;
CUDA OOM is *simulated* by monkeypatching, and the capacity source is an
injected callable, so no GPU is needed.

Key correctness contracts:
  * planner-chunked update gradient == single-forward gradient (up to fp),
  * an OOM mid-minibatch (even a *backward* OOM that left partial gradients)
    restarts the minibatch on a halved budget and still yields the exact
    gradient,
  * the split rollout forward equals the whole-batch forward row-for-row.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from methods.rl_agent.algorithms import _common
from methods.rl_agent.algorithms._common import policy_update_loop
from methods.rl_agent.models.v1.net import NUM_ACTION_TYPES, KiCadRLModel
from methods.rl_agent.rollout.primitive import budgeted_forward
from methods.rl_agent.training.mem_budget import (
    SAFETY,
    MemBudgetModel,
    run_calibration,
)
from tests._mock_obs import make_mock_obs


def _model_with_coeffs(c: float, a: float, b: float, limit: float) -> MemBudgetModel:
    """A ready model with hand-set coefficients whose capacity() == limit."""
    m = MemBudgetModel(headroom_fn=lambda: limit / SAFETY)
    m.coeffs = np.array([c, a, b], dtype=np.float64)
    return m


# ---------------------------------------------------------------------------
# Fit + planner geometry (pure math, no model forward)
# ---------------------------------------------------------------------------
class TestFitAndPlan:
    def test_fit_recovers_coefficients(self):
        m = MemBudgetModel(headroom_fn=lambda: 1e12)
        c, a, b = 5e6, 100.0, 0.5
        for B in (2, 4, 8):
            for L in (100, 400):
                m.observe(B, L, c + a * B * L + b * B * L * L)
        assert m.fit() and m.ready
        assert np.allclose(m.coeffs, [c, a, b], rtol=1e-6)

    def test_fit_refuses_few_points_or_single_length(self):
        m = MemBudgetModel(headroom_fn=lambda: 1e12)
        for B in (2, 4, 8):
            m.observe(B, 100, 100.0 * B * 100)
        assert not m.fit()          # 3 points < MIN_FIT_POINTS
        m.observe(16, 100, 100.0 * 16 * 100)
        assert not m.fit()          # 4 points but a single L (collinear)
        assert not m.ready

    def test_negative_coefficient_clamped(self):
        m = MemBudgetModel(headroom_fn=lambda: 1e12)
        # y = 100*B*L - 0.01*B*L^2 -> lstsq b < 0 -> clamped to 0.
        for B in (2, 4, 8):
            for L in (100, 400):
                m.observe(B, L, 100.0 * B * L - 0.01 * B * L * L)
        assert m.fit()
        assert m.coeffs[2] == 0.0

    def test_plan_chunks_geometry(self):
        m = _model_with_coeffs(0.0, 1.0, 0.0, limit=60.0)   # cost = B * L_max
        seq_lens = [5, 50, 10, 40, 20, 30]
        chunks = m.plan_chunks(seq_lens, limit=60.0)
        # ascending greedy fill: [5,10,20] then 30/40/50 each break the budget
        assert chunks == [[0, 2, 4], [5], [3], [1]]
        # exact cover
        assert sorted(p for c in chunks for p in c) == list(range(6))

    def test_singleton_may_exceed_limit(self):
        m = _model_with_coeffs(0.0, 1.0, 0.0, limit=10.0)
        assert m.plan_chunks([100, 90], limit=10.0) == [[1], [0]]


class TestConfidenceBound:
    """capacity's confidence bound: effective safety based on q99(measured/predicted)."""

    def _seeded(self, ratio: float, n: int = 40) -> MemBudgetModel:
        m = _model_with_coeffs(0.0, 100.0, 0.0, limit=1.0)   # pred = 100*B*L
        m._headroom_fn = lambda: 1000.0
        for i in range(n):
            B, L = 2 + (i % 3), 50 + (i % 5)
            m.observe(B, L, 100.0 * B * L * ratio)
        return m

    def test_cold_start_uses_base_safety(self):
        from methods.rl_agent.training.mem_budget import SAFETY
        m = _model_with_coeffs(0.0, 100.0, 0.0, limit=1.0)
        m._headroom_fn = lambda: 1000.0
        assert m.effective_safety() == SAFETY
        for i in range(10):                       # < _RESID_MIN_N
            m.observe(2, 50, 100.0 * 2 * 50)
        assert m.effective_safety() == SAFETY

    def test_underprediction_tightens(self):
        from methods.rl_agent.training.mem_budget import SAFETY
        m = self._seeded(ratio=1.2)               # measured is 1.2x predicted
        assert abs(m.effective_safety() - SAFETY / 1.2) < 1e-9

    def test_accurate_predictions_relax_up_to_cap(self):
        from methods.rl_agent.training.mem_budget import SAFETY_MAX
        m = self._seeded(ratio=0.8)               # over-prediction (conservative) -> relaxes upward
        assert m.effective_safety() == SAFETY_MAX  # 0.8/0.8=1.0 -> cap 0.90

    def test_capacity_applies_bound(self):
        m = self._seeded(ratio=1.2)
        assert abs(m.capacity(fresh=True) - m.effective_safety() * 1000.0) < 1e-6


# ---------------------------------------------------------------------------
# Calibration (fake probe_fn; OOM tolerance)
# ---------------------------------------------------------------------------
class TestCalibration:
    def test_probe_oom_midway_still_fits(self):
        m = MemBudgetModel(headroom_fn=lambda: 1e12)
        c, a, b = 1e6, 50.0, 0.2
        seq_lens = [100, 250, 300, 120]

        def probe_fn(pos: int, B: int) -> float:
            L = seq_lens[pos]
            if B * L * L > 2e6:   # only the largest probe (B=64, L=300) OOMs
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return c + a * B * L + b * B * L * L

        assert run_calibration(m, probe_fn, seq_lens)
        assert np.allclose(m.coeffs, [c, a, b], rtol=1e-5)

    def test_single_length_group_disables_with_warning(self):
        m = MemBudgetModel(headroom_fn=lambda: 1e12)
        seq_lens = [100, 100, 100]
        with pytest.warns(RuntimeWarning, match="calibration failed"):
            ok = run_calibration(m, lambda pos, B: 1e6, seq_lens)
        assert not ok and not m.ready


# ---------------------------------------------------------------------------
# Update-path equivalence (mirrors TestOOMRecoveryEquivalence)
# ---------------------------------------------------------------------------
class TestPlannedUpdateEquivalence:
    def _varied_obs(self) -> list[dict]:
        counts = [2, 4, 3, 5, 2, 4, 18]   # 18 = outlier the planner must isolate
        return [
            make_mock_obs(
                n_nets=nn, pads_per_net=2, n_ratsnest_per_net=2,
                is_routing=(i % 2 == 0), current_net_phase=1, current_layer=1,
                n_tracks=i * 2, n_vias=i,
            )
            for i, nn in enumerate(counts)
        ]

    def _fresh_policy(self, use_critic: bool = True) -> KiCadRLModel:
        torch.manual_seed(0)
        p = KiCadRLModel(
            d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=4000,
            n_freq=4, use_critic=use_critic,
        )
        with torch.no_grad():
            for layer in p.layers:            # open ReZero gates
                layer.res_attn.alpha.fill_(0.7)
                layer.res_ff.alpha.fill_(0.5)
        return p

    def _buffer(self, obs_list, *, ppo: bool) -> dict:
        p = self._fresh_policy(use_critic=ppo)
        if ppo:
            acts, old_lp, _ = p.act_and_value(obs_list, deterministic=True)
        else:
            acts, old_lp = p.act(obs_list, deterministic=True)
        n = len(obs_list)
        rng = np.random.default_rng(0)
        buf = {
            "obs_list": obs_list,
            "actions": acts.cpu().numpy(),
            "old_log_probs": old_lp.cpu().numpy(),
            "action_masks": np.ones((n, NUM_ACTION_TYPES), dtype=bool),
            "advantages": rng.standard_normal(n).astype(np.float32),
        }
        if ppo:
            buf["returns"] = rng.standard_normal(n).astype(np.float32)
        return buf

    def _forced_chunk_model(self, policy, obs_list) -> MemBudgetModel:
        """cost = B * L_max tokens, budget = 2 * longest sample -> multi-chunk."""
        seq_lens = policy.tokenizer._walk_obs(obs_list)["seq_lens"]
        return _model_with_coeffs(0.0, 1.0, 0.0, limit=2.0 * max(seq_lens))

    def _run(self, obs_list, buffer, *, algo, mem_budget, policy=None):
        if policy is None:
            policy = self._fresh_policy(use_critic=(algo == "ppo"))
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        torch.manual_seed(123)
        metrics = policy_update_loop(
            policy, opt, buffer, torch.device("cpu"), algo=algo,
            n_epochs=1, batch_size=len(obs_list),
            normalize_advantages=(algo == "ppo"), entropy_coef=0.01,
            mem_budget=mem_budget,
        )
        grads = {
            n: p.grad.detach().clone()
            for n, p in policy.named_parameters() if p.grad is not None
        }
        return grads, metrics

    @pytest.mark.parametrize("algo", ["ppo", "grpo"])
    def test_planned_matches_fixed(self, algo):
        obs_list = self._varied_obs()
        buffer = self._buffer(obs_list, ppo=(algo == "ppo"))
        g_fixed, m_fixed = self._run(obs_list, buffer, algo=algo, mem_budget=None)
        model = self._forced_chunk_model(
            self._fresh_policy(use_critic=(algo == "ppo")), obs_list,
        )
        g_plan, m_plan = self._run(obs_list, buffer, algo=algo, mem_budget=model)
        assert m_fixed["planned_chunks_per_mb"] == 1.0
        assert m_plan["planned_chunks_per_mb"] > 1.0   # planner actually split
        assert m_plan["oom_minibatch_rate"] == 0.0     # ... without any OOM
        assert g_fixed and set(g_fixed) == set(g_plan)
        for name in g_fixed:
            assert torch.allclose(g_fixed[name], g_plan[name], atol=1e-6), name

    def test_backward_oom_restart_discards_partial_grad(self, monkeypatch):
        """A chunk that OOMs AFTER polluting .grad (the backward-OOM case the
        reactive peel cannot recover exactly) must be fully discarded: the
        minibatch restarts on a halved budget and the final gradient still
        equals the single-forward one."""
        obs_list = self._varied_obs()
        buffer = self._buffer(obs_list, ppo=True)
        g_fixed, _ = self._run(obs_list, buffer, algo="ppo", mem_budget=None)

        policy = self._fresh_policy(use_critic=True)
        model = self._forced_chunk_model(policy, obs_list)
        limits: list[float] = []
        real_plan = model.plan_chunks

        def recording_plan(seq_lens, limit=None):
            limits.append(limit)
            return real_plan(seq_lens, limit=limit)

        model.plan_chunks = recording_plan

        real_chunk = _common._accumulate_chunk
        state = {"failed": False}

        def failing_chunk(policy_, positions, acc, **kw):
            if not state["failed"]:
                state["failed"] = True
                # simulate a backward-OOM: partial garbage gradient, then OOM
                p0 = next(policy_.parameters())
                p0.grad = torch.full_like(p0, 1e6)
                raise torch.cuda.OutOfMemoryError("simulated backward OOM")
            return real_chunk(policy_, positions, acc, **kw)

        monkeypatch.setattr(_common, "_accumulate_chunk", failing_chunk)
        with pytest.warns(RuntimeWarning, match="despite planned chunking"):
            g_plan, metrics = self._run(
                obs_list, buffer, algo="ppo", mem_budget=model, policy=policy,
            )
        assert metrics["oom_events"] == 1.0
        assert len(limits) == 2 and limits[1] == limits[0] / 2.0   # transient halve
        for name in g_fixed:
            assert torch.allclose(g_fixed[name], g_plan[name], atol=1e-6), name

    def test_single_sample_chunk_oom_reraises(self, monkeypatch):
        obs_list = self._varied_obs()
        buffer = self._buffer(obs_list, ppo=True)
        policy = self._fresh_policy(use_critic=True)
        model = _model_with_coeffs(0.0, 1.0, 0.0, limit=1.0)   # all singletons

        def always_oom(policy_, positions, acc, **kw):
            raise torch.cuda.OutOfMemoryError("simulated OOM")

        monkeypatch.setattr(_common, "_accumulate_chunk", always_oom)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            self._run(obs_list, buffer, algo="ppo", mem_budget=model,
                      policy=policy)


# ---------------------------------------------------------------------------
# Rollout split-forward equivalence
# ---------------------------------------------------------------------------
class TestRolloutSplit:
    _fresh_policy = TestPlannedUpdateEquivalence._fresh_policy
    _varied_obs = TestPlannedUpdateEquivalence._varied_obs
    _forced_chunk_model = TestPlannedUpdateEquivalence._forced_chunk_model

    def _masks(self, n: int):
        mask_t = torch.ones(n, NUM_ACTION_TYPES, dtype=torch.bool)
        ptr_t = torch.full((n, 0), -1, dtype=torch.long)
        return mask_t, ptr_t

    def test_budgeted_forward_matches_whole_batch(self):
        obs_list = self._varied_obs()
        policy = self._fresh_policy(use_critic=True)
        mask_t, ptr_t = self._masks(len(obs_list))
        acts_w, lp_w, val_w = policy.act_and_value(
            obs_list, action_masks=mask_t, pointer_masks=ptr_t,
            deterministic=True,
        )
        model = self._forced_chunk_model(policy, obs_list)
        acts_b, lp_b, val_b = budgeted_forward(
            policy, obs_list, mask_t, ptr_t, None, {}, model,
            want_value=True, deterministic=True,
        )
        assert torch.equal(acts_w, acts_b)
        assert torch.allclose(lp_w, lp_b, atol=1e-6)
        assert torch.allclose(val_w, val_b, atol=1e-6)

    def test_budgeted_forward_act_path(self):
        obs_list = self._varied_obs()
        policy = self._fresh_policy(use_critic=False)
        mask_t, ptr_t = self._masks(len(obs_list))
        acts_w, lp_w = policy.act(
            obs_list, action_masks=mask_t, pointer_masks=ptr_t,
            deterministic=True,
        )
        model = self._forced_chunk_model(policy, obs_list)
        acts_b, lp_b, val_b = budgeted_forward(
            policy, obs_list, mask_t, ptr_t, None, {}, model,
            want_value=False, deterministic=True,
        )
        assert val_b is None
        assert torch.equal(acts_w, acts_b)
        assert torch.allclose(lp_w, lp_b, atol=1e-6)

    def test_act_walked_parity(self):
        """``walked=`` on the act path (what the split forward passes) must be
        bit-identical to the internal walk — mirror of the update-path
        walk-cache contract."""
        obs_list = self._varied_obs()
        policy = self._fresh_policy(use_critic=True)
        mask_t, ptr_t = self._masks(len(obs_list))
        tok = policy.tokenizer
        walked = tok._walk_obs(obs_list)
        out_plain = policy.act_and_value(
            obs_list, action_masks=mask_t, pointer_masks=ptr_t,
            deterministic=True,
        )
        out_walked = policy.act_and_value(
            obs_list, action_masks=mask_t, pointer_masks=ptr_t,
            deterministic=True, walked=walked,
        )
        for a, b in zip(out_plain, out_walked):
            assert torch.equal(a, b)
