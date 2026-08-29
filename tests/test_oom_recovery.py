"""Regression: update-time OOM auto-recovery in ``policy_update_loop``.

When the single-forward minibatch raises ``torch.cuda.OutOfMemoryError``, the
loop retries with sorted, budget-packed gradient accumulation (one optimizer
step per logical minibatch). The **key correctness property**: the recovered
(chunked) update must be identical — up to fp reassociation — to the single
forward it replaced. Attention is masked per row, so a sample's log_prob /
entropy / value are independent of how it is batched, and each chunk's loss is
summed over its samples and divided by the *full* minibatch size, so the
accumulated gradient equals the fixed-batch mean gradient.

Pure PyTorch, runs on CPU: the CUDA OOM is *simulated* by monkeypatching the
first forward to raise, which drives the real recovery path.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pcb_world.engine import engine_available
import torch

from pcb_world.engine.kicad_engine import allow_router_coexistence

from methods.rl_agent.algorithms._common import (
    _peel_accumulate,
    policy_update_loop,
)
from methods.rl_agent.models.v1.net import NUM_ACTION_TYPES, KiCadRLModel
from tests._mock_obs import make_mock_obs

_FIXTURE_BOARD = os.path.join(
    os.path.dirname(__file__), "fixtures", "simple_routing_board.kicad_pcb",
)


def _skip_if_no_env() -> None:
    if not os.path.exists(_FIXTURE_BOARD):
        pytest.skip(f"fixture board not found: {_FIXTURE_BOARD}")
    if not engine_available():   # probe only — no GPL import (import-hygiene)
        pytest.skip("kicad_rl_router not available")


def _real_ppo_buffer() -> dict:
    """A PPO rollout buffer from the REAL PCB env (real obs token layouts + all
    mask channels), via the same collect→GAE→buffer path the trainer uses.
    Needs the C++ ``kicad_rl_router``; skipped otherwise."""
    _skip_if_no_env()
    from methods.rl_agent.training.buffer import (
        compute_gae_flat,
        ppo_collector_to_buffer,
    )
    from methods.rl_agent.training.collect import collect_n_steps_ppo
    from methods.rl_agent.wrappers.factory import make_decoder_env
    from tests.helpers.env_kwargs import full_env_kwargs

    torch.manual_seed(0)
    device = torch.device("cpu")
    with allow_router_coexistence("collector list mode: 3 in-process envs"):
        envs = [make_decoder_env(_FIXTURE_BOARD, **full_env_kwargs(max_steps=24, seed=i)) for i in range(3)]
    try:
        collector_policy = KiCadRLModel(
            d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=4000,
            n_freq=4, use_critic=True,
        ).to(device)
        coll = collect_n_steps_ppo(envs, collector_policy, device, n_steps=24)
        advs, rets = compute_gae_flat(
            rewards=coll.rewards, values=coll.values,
            episode_starts=coll.episode_starts, final_values=coll.final_values,
            terminal_values=coll.terminal_values, gamma=1.0, gae_lambda=0.95,
        )
        return ppo_collector_to_buffer(coll, advs, rets)
    finally:
        for e in envs:
            e.close()


class TestPeelSplit:
    """On OOM the peel cuts the longest ``max(len//4, 1)`` samples into their own
    chunk and recurses on both sides — using only ``len(chunk)`` and the actual
    OOM (no budget/proxy). Verified by stubbing the per-chunk fwd+bwd to OOM until
    a chunk is small enough to 'fit'."""

    def test_peels_longest_quarter_until_it_fits(self, monkeypatch):
        from methods.rl_agent.algorithms import _common

        seen: list[list[int]] = []

        def fake_accumulate(policy, positions, acc, **kw):
            seen.append(list(positions))
            if len(positions) > 3:            # simulate OOM until a chunk is <= 3
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            # else: fits (stub does no real work)

        monkeypatch.setattr(_common, "_accumulate_chunk", fake_accumulate)

        order = list(range(16))               # sorted ascending; 15 = longest
        n_oom = _peel_accumulate(None, order, {})

        leaves = [c for c in seen if len(c) <= 3]   # chunks that "fit"
        # exact cover: every position processed exactly once, no overlap
        assert sorted(p for c in leaves for p in c) == list(range(16))
        # the longest sample (15) landed in a small tail chunk, not the bulk
        tail = next(c for c in leaves if 15 in c)
        assert len(tail) <= 3
        assert n_oom >= 1

    def test_single_sample_oom_is_unrecoverable(self, monkeypatch):
        from methods.rl_agent.algorithms import _common

        def always_oom(policy, positions, acc, **kw):
            raise torch.cuda.OutOfMemoryError("simulated OOM")

        monkeypatch.setattr(_common, "_accumulate_chunk", always_oom)
        # A lone sample that still OOMs -> board too big for VRAM -> re-raise.
        with pytest.raises(torch.cuda.OutOfMemoryError):
            _peel_accumulate(None, [0], {})


class TestOOMRecoveryEquivalence:
    """Forcing the recovery path (simulated OOM) must reproduce the single-
    forward gradient exactly (up to fp)."""

    def _varied_obs(self, with_outlier: bool = True) -> list[dict]:
        # Varying net counts -> varying sequence lengths; last one is an outlier
        # so the recovery genuinely sorts + isolates.
        counts = [2, 4, 3, 5, 2, 4]
        if with_outlier:
            counts.append(18)
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
            for layer in p.layers:            # open ReZero gates (exercise attn grads)
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

    def _grads(self, obs_list, buffer, *, algo, oom_forward: int | None) -> dict:
        """Run one epoch / one logical minibatch (batch_size=N) with lr=0 so
        ``.grad`` after the loop is the accumulated minibatch gradient. When
        ``oom_forward`` is set, the k-th forward raises a simulated CUDA OOM,
        driving the recovery path."""
        policy = self._fresh_policy(use_critic=(algo == "ppo"))
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)

        if oom_forward is not None:
            real = policy.evaluate_actions_and_value
            calls = {"n": 0}

            def patched(*a, **k):
                calls["n"] += 1
                if calls["n"] == oom_forward:
                    raise torch.cuda.OutOfMemoryError("simulated OOM")
                return real(*a, **k)

            policy.evaluate_actions_and_value = patched

        torch.manual_seed(123)  # identical randperm across runs
        policy_update_loop(
            policy, opt, buffer, torch.device("cpu"), algo=algo,
            n_epochs=1, batch_size=len(obs_list),
            normalize_advantages=(algo == "ppo"), entropy_coef=0.01,
        )
        return {
            n: p.grad.detach().clone()
            for n, p in policy.named_parameters() if p.grad is not None
        }

    def test_ppo_recovery_matches_fixed(self):
        obs_list = self._varied_obs()
        buffer = self._buffer(obs_list, ppo=True)
        g_fixed = self._grads(obs_list, buffer, algo="ppo", oom_forward=None)
        with pytest.warns(RuntimeWarning, match="OOM"):
            g_oom = self._grads(obs_list, buffer, algo="ppo", oom_forward=1)
        assert g_fixed and set(g_fixed) == set(g_oom)
        for name in g_fixed:
            assert torch.allclose(g_fixed[name], g_oom[name], atol=1e-6), name

    def test_grpo_recovery_matches_fixed(self):
        obs_list = self._varied_obs()
        buffer = self._buffer(obs_list, ppo=False)
        g_fixed = self._grads(obs_list, buffer, algo="grpo", oom_forward=None)
        with pytest.warns(RuntimeWarning, match="OOM"):
            g_oom = self._grads(obs_list, buffer, algo="grpo", oom_forward=1)
        assert g_fixed and set(g_fixed) == set(g_oom)
        for name in g_fixed:
            assert torch.allclose(g_fixed[name], g_oom[name], atol=1e-6), name

    def test_repeated_oom_peels_to_single_sample(self):
        """If the whole-minibatch forward AND the first peeled chunks keep
        OOM-ing, the peel keeps cutting 1/4 until chunks fit (never hangs) and the
        final (deeply-peeled) gradient still matches the single-forward one."""
        obs_list = self._varied_obs()
        buffer = self._buffer(obs_list, ppo=True)
        g_fixed = self._grads(obs_list, buffer, algo="ppo", oom_forward=None)

        # OOM for ANY chunk of >2 samples -> the whole minibatch and every peeled
        # chunk keep OOM-ing until the peel cuts them down to <=2 samples.
        policy = self._fresh_policy(use_critic=True)
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        real = policy.evaluate_actions_and_value

        def patched(obs_subset, *a, **k):
            if len(obs_subset) > 2:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return real(obs_subset, *a, **k)

        policy.evaluate_actions_and_value = patched
        torch.manual_seed(123)
        with pytest.warns(RuntimeWarning, match="OOM"):
            policy_update_loop(
                policy, opt, buffer, torch.device("cpu"), algo="ppo",
                n_epochs=1, batch_size=len(obs_list), normalize_advantages=True,
            )
        g_recovered = {
            n: p.grad.detach().clone()
            for n, p in policy.named_parameters() if p.grad is not None
        }
        for name in g_fixed:
            assert torch.allclose(g_fixed[name], g_recovered[name], atol=1e-6), name

    def _metrics(self, obs_list, buffer, *, algo, oom_forward):
        policy = self._fresh_policy(use_critic=(algo == "ppo"))
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        if oom_forward is not None:
            real = policy.evaluate_actions_and_value
            calls = {"n": 0}

            def patched(*a, **k):
                calls["n"] += 1
                if calls["n"] == oom_forward:
                    raise torch.cuda.OutOfMemoryError("simulated OOM")
                return real(*a, **k)

            policy.evaluate_actions_and_value = patched
        torch.manual_seed(123)
        return policy_update_loop(
            policy, opt, buffer, torch.device("cpu"), algo=algo,
            n_epochs=1, batch_size=len(obs_list),
            normalize_advantages=(algo == "ppo"),
        )

    def test_oom_rate_metric_reported(self):
        """policy_update_loop reports diag/oom_minibatch_rate + oom_events so the
        run can watch whether boards are outgrowing VRAM (rate -> 1.0)."""
        obs_list = self._varied_obs()
        buffer = self._buffer(obs_list, ppo=True)
        # No OOM -> rate 0.
        m0 = self._metrics(obs_list, buffer, algo="ppo", oom_forward=None)
        assert m0["oom_minibatch_rate"] == 0.0
        assert m0["oom_events"] == 0.0
        # Single minibatch (batch_size=N), OOM injected -> rate 1.0, >=1 event.
        with pytest.warns(RuntimeWarning, match="OOM"):
            m1 = self._metrics(obs_list, buffer, algo="ppo", oom_forward=1)
        assert m1["oom_minibatch_rate"] == 1.0
        assert m1["oom_events"] >= 1.0

    def test_ppo_recovery_matches_fixed_real_env(self):
        """Same equivalence, but on a REAL rollout buffer (real obs + every mask
        channel the env populates), not mock obs. Needs the C++ router."""
        buffer = _real_ppo_buffer()
        obs_list = buffer["obs_list"]
        assert len(obs_list) > 1
        g_fixed = self._grads(obs_list, buffer, algo="ppo", oom_forward=None)
        with pytest.warns(RuntimeWarning, match="OOM"):
            g_oom = self._grads(obs_list, buffer, algo="ppo", oom_forward=1)
        assert g_fixed and set(g_fixed) == set(g_oom)
        for name in g_fixed:
            assert torch.allclose(g_fixed[name], g_oom[name], atol=1e-6), name
