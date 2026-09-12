"""Pointer-block masking is branch-free — equivalence with the indexing form.

``_combined_ptr_logits`` blocks candidate columns two ways (a per-row gated
block for the make_line off-layer rule, and an all-rows block for the
start_route point). Both used to read ``valid.any()`` into a Python bool before
doing the work, which syncs the device every call and forks the ``heads``
torch.compile graph on the outcome — the batch flips between "something to
block" and "nothing to block" constantly, so the region burned a recompile slot
each time and eventually hit dynamo's limit and fell back to eager.

These tests pin the replacement (:func:`_blocked_columns`) to the exact
semantics of the indexing form it replaced, including the two paths' *different*
out-of-range rules and the duplicate-index case that plain ``scatter_``
(last-write-wins) would get wrong.

No C++ dependency — pure PyTorch tests.
"""

from __future__ import annotations

import pytest
import torch

from methods.rl_agent.models.v1.net import _blocked_columns


def _reference_row_block(cand_logits, rb, gate):
    """The original row-gated form: out-of-range indices are NOT blocked."""
    out = cand_logits.clone()
    N = out.size(1)
    valid = (rb >= 0) & (rb < N) & gate.reshape(-1, 1)
    if bool(valid.any()):
        rows = (torch.arange(out.size(0)).unsqueeze(1).expand_as(rb)[valid])
        out[rows, rb[valid]] = float("-inf")
    return out


def _reference_cand_block(cand_logits, cbi):
    """The original all-rows form: out-of-range indices ARE blocked (clamped)."""
    out = cand_logits.clone()
    N = out.size(1)
    valid = cbi >= 0
    if valid.any():
        rows = torch.arange(cbi.size(0)).unsqueeze(1).expand_as(cbi)[valid]
        cols = cbi[valid].clamp(min=0, max=N - 1)
        out[rows, cols] = float("-inf")
    return out


class TestBlockedColumns:
    @pytest.mark.parametrize("K", [0, 1, 2, 5])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_row_gated_matches_indexing_form(self, K, seed):
        torch.manual_seed(seed)
        B, N = 6, 9
        logits = torch.randn(B, N)
        # -2..N+1 so out-of-range (>= N) and pad (-1) both occur
        rb = torch.randint(-2, N + 2, (B, K))
        gate = torch.rand(B) > 0.5
        valid = (rb >= 0) & (rb < N) & gate.reshape(-1, 1)
        got = logits.masked_fill(_blocked_columns(valid, rb, N), float("-inf"))
        assert torch.equal(got, _reference_row_block(logits, rb, gate))

    @pytest.mark.parametrize("K", [0, 1, 2, 5])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_all_rows_matches_indexing_form(self, K, seed):
        torch.manual_seed(seed)
        B, N = 6, 9
        logits = torch.randn(B, N)
        cbi = torch.randint(-2, N + 2, (B, K))
        valid = cbi >= 0
        got = logits.masked_fill(_blocked_columns(valid, cbi, N), float("-inf"))
        assert torch.equal(got, _reference_cand_block(logits, cbi))

    def test_duplicate_indices_cannot_clear_a_valid_entry(self):
        # Plain scatter_ is last-write-wins: an invalid duplicate landing after
        # a valid one would silently unblock the column.
        B, N = 1, 4
        idx = torch.tensor([[2, 2]])            # same column twice
        valid = torch.tensor([[True, False]])   # valid first, invalid second
        assert _blocked_columns(valid, idx, N).tolist() == [[False, False, True, False]]

    def test_nothing_valid_blocks_nothing(self):
        B, N = 3, 5
        idx = torch.full((B, 2), -1)
        valid = idx >= 0
        assert not _blocked_columns(valid, idx, N).any()

    def test_no_python_bool_read_of_a_device_tensor(self):
        # Guard against the sync/graph-fork regression coming back: the helper
        # must not branch on tensor contents. `.item()`/`bool()` on a fake
        # tensor whose sync would raise is the cheapest way to assert that.
        import methods.rl_agent.models.v1.net as net_mod
        src = __import__("inspect").getsource(net_mod._blocked_columns)
        body = src.split('"""')[-1]
        for forbidden in ("if valid.any()", "bool(", ".item()"):
            assert forbidden not in body, f"{forbidden} reintroduced"
