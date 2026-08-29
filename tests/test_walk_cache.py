"""Walk-cache equivalence — safety checks for the update path's re-tokenization
cache (flat walk + gather).

Contract (all **byte-identical**, down to list element order, dtype, and
container type):

1. ``merge_walked(batch walks)`` == calling ``_walk_obs`` directly on the
   concatenation of those obs — the path collect uses to merge per-step
   batch walks once, at rollout end, into ``walk_flat``.
2. ``gather_walked(flat, bounds, idx)`` == ``_walk_obs([obs[i] for i in idx])``
   — the path update uses to index-gather an arbitrary-order minibatch from
   the flat walk. (The walk resets per-obs state and appends in order, so
   merge = concatenation and gather = reordering contiguous per-sample
   segments.)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from pcb_world.core.indexed_obs import dict_to_arrays
from tests._mock_obs import make_mock_obs


@pytest.fixture(params=["dict", "indexed"])
def obs_format(request):
    """The cache contract must hold for both obs formats — whether the walk
    implementation is _walk_dict or _walk_indexed, the resulting walk dict
    schema is the same."""
    return request.param


def _make_obs_list(fmt: str = "dict"):
    obs_list = [
        make_mock_obs(n_nets=1, pads_per_net=1),
        make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1),
        make_mock_obs(n_nets=3, pads_per_net=1),
        make_mock_obs(n_nets=2, pads_per_net=3, n_ratsnest_per_net=2),
    ]
    if fmt == "indexed":
        return [dict_to_arrays(o) for o in obs_list]
    return obs_list


def _tok(seed: int = 0) -> BatchedStateTokenizer:
    torch.manual_seed(seed)
    tok = BatchedStateTokenizer()
    tok.eval()
    return tok


def _walk_field_equal(x, y) -> bool:
    """Exact walk-dict field equality — for ndarrays, type, dtype, shape,
    and values all must match."""
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return (
            isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
            and x.dtype == y.dtype and x.shape == y.shape
            and np.array_equal(x, y)
        )
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return (
            type(x) is type(y) and len(x) == len(y)
            and all(_walk_field_equal(a, b) for a, b in zip(x, y))
        )
    return x == y


def _assert_walk_equal(got, direct) -> None:
    assert got.keys() == direct.keys()
    for k in direct:
        assert _walk_field_equal(got[k], direct[k]), f"walk[{k}] mismatch"


def _assert_tokout_bitequal(a, b) -> None:
    assert torch.equal(a.token_embeddings, b.token_embeddings)
    assert torch.equal(a.key_padding_mask, b.key_padding_mask)
    assert torch.equal(a.seq_lens, b.seq_lens)
    assert torch.equal(a.net_indices, b.net_indices)
    assert torch.equal(a.cand_indices, b.cand_indices)
    assert a.cand_mm_list == b.cand_mm_list


def _step_merged_flat(tok, obs_list):
    """Exactly what collect does: per-step (chunk) batch walk -> merge_walked once."""
    chunks = [obs_list[0:1], obs_list[1:3], obs_list[3:4]]
    return tok.merge_walked([tok._walk_obs(c) for c in chunks])


class TestMergeWalked:
    def test_merge_batches_equals_direct_walk(self, obs_format):
        """Merging batch (B>=1) walks == walking the whole set directly
        (the collect-end path)."""
        tok = _tok()
        obs_list = _make_obs_list(obs_format)
        merged = _step_merged_flat(tok, obs_list)
        direct = tok._walk_obs(obs_list)
        _assert_walk_equal(merged, direct)  # fully identical down to element, order, dtype

    def test_schema_guard_fails_loudly(self):
        """If a key appears in the _walk_obs schema that merge doesn't know
        about, it aborts with an assert."""
        tok = _tok()
        walk = tok._walk_obs(_make_obs_list()[:2])
        walk["__new_walk_key__"] = [1]
        with pytest.raises(AssertionError, match="schema changed"):
            tok.merge_walked([walk])


class TestGatherWalked:
    """``gather_walked`` (flat walk -> arbitrary-order minibatch walk) —
    safety of the update path. The property tests are the real correctness
    safety net (independent of whether an assert fires); the guard-fires
    tests prove that bounds' two asserts actually trigger.
    """

    def test_gather_equals_direct_walk(self, obs_format):
        """The real safety net: gather output == walking the subset
        directly (down to element and type). Includes taking a subset and
        shuffling order exactly as a minibatch does."""
        tok = _tok()
        obs_list = _make_obs_list(obs_format)
        flat = tok._walk_obs(obs_list)
        bounds = tok.walk_sample_bounds(flat)
        for idx in ([2, 0, 3], [1], [3, 2, 1, 0], [0, 1, 2, 3]):
            got = tok.gather_walked(flat, bounds, idx)
            direct = tok._walk_obs([obs_list[i] for i in idx])
            _assert_walk_equal(got, direct)

    def test_gather_from_step_merged_flat(self, obs_format):
        """Exactly the production path: per-step batch walk -> merge -> gather == direct walk."""
        tok = _tok()
        obs_list = _make_obs_list(obs_format)
        flat = _step_merged_flat(tok, obs_list)
        bounds = tok.walk_sample_bounds(flat)
        for idx in ([2, 0, 3], [1, 3]):
            got = tok.gather_walked(flat, bounds, idx)
            direct = tok._walk_obs([obs_list[i] for i in idx])
            _assert_walk_equal(got, direct)

    def test_forward_gathered_bitequal(self, obs_format):
        """Bit-equal all the way to the tokenizer's output tensors
        (gather-walked path == direct path)."""
        tok = _tok()
        obs_list = _make_obs_list(obs_format)
        flat = _step_merged_flat(tok, obs_list)
        bounds = tok.walk_sample_bounds(flat)
        idx = [3, 1, 2]
        sub_obs = [obs_list[i] for i in idx]
        with torch.no_grad():
            direct = tok(sub_obs)
            cached = tok(sub_obs, walked=tok.gather_walked(flat, bounds, idx))
        _assert_tokout_bitequal(direct, cached)

    def test_bounds_sorted_guard_fires(self):
        """If obs_idx is not non-decreasing, bounds aborts loudly (proves
        the early-warning guard)."""
        tok = _tok()
        walk = tok._walk_obs(_make_obs_list("indexed"))
        # Reverse the 'net' obs_idx slot to break the sortedness invariant.
        oi = BatchedStateTokenizer._WALK_OBS_IDX_SLOT["net"]
        fields = list(walk["net"])
        fields[oi] = fields[oi][::-1]
        walk["net"] = tuple(fields)
        with pytest.raises(AssertionError, match="non-decreasing"):
            tok.walk_sample_bounds(walk)

    def test_bounds_schema_guard_fires(self):
        """If an unknown key appears in the walk, bounds aborts with an assert."""
        tok = _tok()
        walk = tok._walk_obs(_make_obs_list("indexed"))
        walk["__new_walk_key__"] = [1]
        with pytest.raises(AssertionError, match="schema changed"):
            tok.walk_sample_bounds(walk)


class TestUpdateWithoutObs:
    """DDP obs-strip contract: when walk_flat is carried along, update must
    produce the same result even without obs (obs_list=None) — pins
    end-to-end that the tokenizer ignores obs on the walked= path."""

    def _buffer(self, obs_list, policy):
        from methods.rl_agent.models.v1.net import NUM_ACTION_TYPES
        from methods.rl_agent.models.v1.spec import ACT_NET_SELECT

        n = len(obs_list)
        # Mirrors the real env's idle masking: rows with no candidate (cand)
        # token allow only net_select — act's dead-cand-ptr-row guard treats
        # "a non-net_select action on an obs with no cand" as an impossible
        # state, but an all-ones mask can produce exactly that combination
        # depending on the RNG stream, which would make this test flaky.
        walk = policy.tokenizer._walk_obs(obs_list)
        rows_with_cands = set(walk["cand"][5].tolist())
        masks = np.ones((n, NUM_ACTION_TYPES), dtype=bool)
        for i in range(n):
            if i not in rows_with_cands:
                masks[i, :] = False
                masks[i, ACT_NET_SELECT] = True
        with torch.no_grad():
            acts, old_lp, _ = policy.act_and_value(
                obs_list, action_masks=torch.from_numpy(masks),
                deterministic=True,
            )
        rng = np.random.default_rng(0)
        return {
            "obs_list": obs_list,
            "actions": acts.cpu().numpy(),
            "old_log_probs": old_lp.cpu().numpy().astype(np.float32),
            "action_masks": masks,
            "advantages": rng.standard_normal(n).astype(np.float32),
            "returns": rng.standard_normal(n).astype(np.float32),
        }

    def _run(self, strip_obs: bool):
        from methods.rl_agent.algorithms._common import policy_update_loop
        from methods.rl_agent.models.v1.net import KiCadRLModel
        from methods.rl_agent.training.ddp import _worker_payload

        torch.manual_seed(0)
        policy = KiCadRLModel(
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=2000, n_freq=4, use_critic=True,
        )
        with torch.no_grad():
            # Open the ReZero gates — a closed gate (identity) would make the comparison vacuous.
            for layer in policy.layers:
                layer.res_attn.alpha.fill_(0.7)
                layer.res_ff.alpha.fill_(0.5)
        obs_list = _make_obs_list("indexed")
        buffer = self._buffer(obs_list, policy)
        buffer["walk_flat"] = policy.tokenizer._walk_obs(obs_list)
        if strip_obs:
            buffer = _worker_payload(buffer)
            assert buffer["obs_list"] is None
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        torch.manual_seed(123)  # same permutation stream
        metrics = policy_update_loop(
            policy, opt, buffer, torch.device("cpu"), algo="ppo",
            n_epochs=1, batch_size=len(obs_list),
        )
        grads = {n_: p.grad.detach().clone()
                 for n_, p in policy.named_parameters() if p.grad is not None}
        return metrics, grads

    def test_stripped_buffer_matches_full(self):
        m_full, g_full = self._run(strip_obs=False)
        m_strip, g_strip = self._run(strip_obs=True)
        assert m_full == m_strip
        assert g_full.keys() == g_strip.keys()
        for k in g_full:
            assert torch.equal(g_full[k], g_strip[k]), k

    def test_payload_keeps_obs_without_walk(self):
        """A buffer without a walk (GRPO) is not stripped — the worker needs to re-walk it."""
        from methods.rl_agent.training.ddp import _worker_payload

        buffer = {"obs_list": [{"x": 1}], "actions": np.zeros((1, 3))}
        assert _worker_payload(buffer) is buffer
