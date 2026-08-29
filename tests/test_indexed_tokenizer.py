"""Indexed Obs tokenizer walk — BIT-IDENTICAL parity vs the dict walk.

One ``BatchedStateTokenizer`` instance, two input formats: the legacy
nested dict vs ``dict_to_arrays()`` of the SAME obs. Every
``TokenizerOutput`` field must be ``torch.equal`` — exact, no tolerance:
the array walk gathers the same f64 mm values, normalizes with the same
op order, and casts f64→f32 once at the same place, so any difference
is a real regression of the training-reproducibility contract.

Also pins the shared candidate pool + pointer-mask helpers
(tuple-/bool-exact between formats) and the external ``_walk_obs``
seq_lens contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from methods.rl_agent.models.v1.encoding import (
    net_valid_mask,
    sorted_net_codes_from_obs,
)
from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from pcb_world.core.indexed_obs import dict_to_arrays, make_empty_indexed_obs
from pcb_world.vec.candidate_pool import collect_raw_candidates
from tests._mock_obs import make_mock_obs
from tests.test_indexed_obs import make_canonical_obs


def make_tok(seed: int = 0, **kwargs) -> BatchedStateTokenizer:
    torch.manual_seed(seed)
    tok = BatchedStateTokenizer(d_model=64, n_freq=8, **kwargs)
    tok.eval()
    return tok


def assert_bit_identical(tok, obs_list, *, action_type_weight=None):
    indexed = [dict_to_arrays(o) for o in obs_list]
    with torch.no_grad():
        r = tok(obs_list, action_type_weight=action_type_weight)
        i = tok(indexed, action_type_weight=action_type_weight)
    assert torch.equal(r.token_embeddings, i.token_embeddings), (
        "token_embeddings differ: max abs diff = "
        f"{(r.token_embeddings - i.token_embeddings).abs().max().item():.3e}"
    )
    assert torch.equal(r.key_padding_mask, i.key_padding_mask)
    assert torch.equal(r.seq_lens, i.seq_lens)
    assert torch.equal(r.net_indices, i.net_indices)
    assert torch.equal(r.cand_indices, i.cand_indices)
    assert torch.equal(r.slot_ids, i.slot_ids)
    assert r.cand_mm_list == i.cand_mm_list


class TestBitIdenticalTokens:
    def test_basic_mixed_batch(self):
        tok = make_tok()
        obs_list = [
            make_mock_obs(n_nets=1, pads_per_net=1,
                          is_routing=False, current_net_phase=0,
                          n_ratsnest_per_net=0),
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2),
            make_mock_obs(n_nets=3, pads_per_net=1,
                          is_routing=False, current_net_phase=1),
            make_mock_obs(n_nets=2, pads_per_net=3, n_ratsnest_per_net=2,
                          is_routing=True, current_net_phase=2),
        ]
        assert_bit_identical(tok, obs_list)

    def test_tracks_vias_ratsnest(self):
        tok = make_tok()
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_tracks=3, n_vias=2,
                          n_ratsnest_per_net=2, is_routing=True,
                          current_net_phase=2),
            make_mock_obs(n_nets=3, pads_per_net=2, n_tracks=1, n_vias=1,
                          n_ratsnest_per_net=1, is_routing=False,
                          current_net_phase=1),
        ]
        assert_bit_identical(tok, obs_list)

    def test_drc_violations(self):
        tok = make_tok()
        vio = [
            {"x_mm": 120.0, "y_mm": 65.0, "layer": 1, "type_id": 0,
             "severity": 0x20, "net_names": ["NET1"]},
            {"x_mm": 125.0, "y_mm": 60.0, "layer": 2, "type_id": 3,
             "severity": 0x10, "net_names": []},
            {"x_mm": 110.0, "y_mm": 62.0, "layer": 1, "type_id": 99,
             "severity": 0x20, "net_names": ["NOPE"]},
        ]
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_tracks=1,
                          is_routing=True, current_net_phase=2,
                          drc_violations=vio),
        ]
        assert_bit_identical(tok, obs_list)

    def test_aug_orthogonal_and_scale(self):
        tok = make_tok()
        base = dict(n_nets=3, pads_per_net=2, n_tracks=2, n_vias=1,
                    n_ratsnest_per_net=1, is_routing=True,
                    current_net_phase=2)
        o1 = make_mock_obs(**base)
        o1["_aug"] = {"axis_swap": True, "flip_x": -1, "flip_y": 1,
                      "nn_dx": 0.13, "nn_dy": -0.07}
        o2 = make_mock_obs(**base)
        o2["_aug"] = {"scale_x": 1.25, "scale_y": 0.8,
                      "aug_cx": 128.0, "aug_cy": 68.0,
                      "axis_swap": False, "flip_x": 1, "flip_y": -1,
                      "nn_dx": 0.0, "nn_dy": 0.05}
        o3 = make_mock_obs(**base)
        o3["_aug"] = {"slot_perm": [2, 0, 1]}
        assert_bit_identical(tok, [o1, o2, o3])

    def test_action_history_and_closed_nets(self):
        # Explicit K>1: the default is 1 (historical prev-action window), and
        # this test exercises the multi-entry encoding incl. sentinel padding.
        tok = make_tok(action_history_len=5)
        o1 = make_mock_obs(n_nets=3, pads_per_net=2, is_routing=True,
                           current_net_phase=2)
        o1["action_history"] = [
            {
                "action_type": 3, "pointer_xy": [120.0, 60.0],
                "pointer_layer": 2, "routing_mode": 2,
                "has_pointer": True, "success": True, "net_id": 2,
            },
            {
                "action_type": 1, "pointer_xy": [110.0, 55.0],
                "pointer_layer": 1, "routing_mode": -1,
                "has_pointer": True, "success": False, "net_id": 3,
            },
            {
                "action_type": 0, "pointer_xy": [0.0, 0.0],
                "pointer_layer": 0, "routing_mode": -1,
                "has_pointer": False, "success": True, "net_id": None,
            },
        ]
        o1["closed_nets"] = [2]
        o2 = make_mock_obs(n_nets=2, pads_per_net=1)
        o2["action_history"] = []
        o2["closed_nets"] = []
        w = torch.randn(16, 64)
        assert_bit_identical(tok, [o1, o2], action_type_weight=w)

    def test_canonical_env_builder_obs(self):
        """Obs produced by the REAL production builders (incl. dedup'd
        shared points, thru pads, obstacles, unconnected pads, DRC,
        action_history, closed_nets)."""
        tok = make_tok()
        obs, _, _ = make_canonical_obs()
        w = torch.randn(16, 64)
        assert_bit_identical(tok, [obs], action_type_weight=w)

    def test_sin_remaining_time_feature(self):
        tok = make_tok(time_feature="sin_remaining", time_feature_cap=256)
        obs_list = [make_mock_obs(n_nets=2, pads_per_net=2,
                                  is_routing=True, current_net_phase=2,
                                  steps_remaining=137)]
        assert_bit_identical(tok, obs_list)

    def test_empty_indexed_obs_forward(self):
        """Crash-path fallback obs must tokenize (BOARD+HEAD+3K hist+VAL+SOD)."""
        tok = make_tok()
        K = tok.vocab.action_history_len
        with torch.no_grad():
            out = tok([make_empty_indexed_obs()])
        assert int(out.seq_lens[0]) == 4 + 3 * K

    def test_walk_seq_lens_contract(self):
        """External callers (_common.py OOM sort, loop.py diagnostics)
        use tokenizer._walk_obs(obs_list)['seq_lens'] — must work for
        both formats and agree."""
        tok = make_tok()
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_tracks=1,
                          n_ratsnest_per_net=1, is_routing=True,
                          current_net_phase=2),
            make_mock_obs(n_nets=1, pads_per_net=1),
        ]
        legacy = tok._walk_obs(obs_list)["seq_lens"]
        indexed = tok._walk_obs([dict_to_arrays(o) for o in obs_list])["seq_lens"]
        assert legacy == indexed


class TestHelperEquivalence:
    """Shared candidate pool + pointer-mask helpers: format-exact."""

    def _pair(self, **kwargs):
        obs = make_mock_obs(**kwargs)
        obs["closed_nets"] = [2]
        return obs, dict_to_arrays(obs)

    def test_collect_raw_candidates_tuple_exact(self):
        obs, iobs = self._pair(n_nets=3, pads_per_net=2, n_tracks=2,
                               n_vias=1, is_routing=True,
                               current_net_phase=2)
        for net_id in (1, 2, 3, None, 99):
            extra = [(1.0, 2.0, 1, 4)]
            legacy = collect_raw_candidates(obs, net_id, extra)
            indexed = collect_raw_candidates(iobs, net_id, extra)
            assert legacy == indexed, f"net {net_id}"

    def test_thru_pad_expansion(self):
        obs = make_mock_obs(n_nets=1, pads_per_net=2, copper_layers=4)
        # Force a thru-hole sentinel on one pad.
        pad0 = obs["board_static"]["nets"]["net_1"]["pads"]["pad_0"]
        pad0["layer"] = 0
        iobs = dict_to_arrays(obs)
        assert collect_raw_candidates(obs, 1, None) \
            == collect_raw_candidates(iobs, 1, None)

    def test_sorted_net_codes(self):
        obs, iobs = self._pair(n_nets=3, pads_per_net=1)
        assert sorted_net_codes_from_obs(obs) == sorted_net_codes_from_obs(iobs)

    def test_net_valid_mask_exact(self):
        obs, iobs = self._pair(n_nets=3, pads_per_net=2,
                               n_ratsnest_per_net=2)
        codes = sorted_net_codes_from_obs(obs)
        m_legacy = net_valid_mask(sorted_net_codes=codes, last_obs=obs)
        m_indexed = net_valid_mask(sorted_net_codes=codes, last_obs=iobs)
        assert np.array_equal(m_legacy, m_indexed)
        # closed net 2 must be masked out in both.
        assert not m_legacy[codes.index(2)]

    def test_net_valid_mask_no_fallback(self):
        """With 0 valid nets, returns all-False as-is — no all-True fallback.
        If an all-False mask reaches the policy, net.py's all-(-inf)
        pointer-row guard fails loudly with context."""
        obs, iobs = self._pair(n_nets=2, pads_per_net=2,
                               n_ratsnest_per_net=0)
        codes = sorted_net_codes_from_obs(obs)
        m_legacy = net_valid_mask(sorted_net_codes=codes, last_obs=obs)
        m_indexed = net_valid_mask(sorted_net_codes=codes, last_obs=iobs)
        assert np.array_equal(m_legacy, m_indexed)
        assert not m_indexed.any()


class TestActionHistoryEncoding:
    """ACTION_HISTORY window semantics: K-invariant weights, age separation,
    net-slot binding, legacy prev-action compatibility."""

    _REC = {
        "action_type": 3, "pointer_xy": [120.0, 60.0], "pointer_layer": 1,
        "routing_mode": 2, "has_pointer": True, "success": True, "net_id": 2,
    }

    def test_k_invariant_state_dict(self):
        """Changing K adds/removes no weights — K=5 ckpt loads strict into K=8."""
        sd = make_tok(action_history_len=5).state_dict()
        tok8 = make_tok(action_history_len=8)
        tok8.load_state_dict(sd, strict=True)

    def test_token_count_is_3k(self):
        obs = make_mock_obs(n_nets=2, pads_per_net=1)
        obs["action_history"] = []
        n2 = int(make_tok(action_history_len=2)._walk_obs([obs])["seq_lens"][0])
        n4 = int(make_tok(action_history_len=4)._walk_obs([obs])["seq_lens"][0])
        assert n4 - n2 == 3 * 2

    def test_age_distinguishes_entries(self):
        """The same record at age 0 vs age 1 must embed differently (the age
        marker is the only order signal in the permutation-equivariant
        state zone) — and identically in legacy mode (no age path)."""
        w = torch.randn(16, 64)
        obs_a = make_mock_obs(n_nets=3, pads_per_net=1)
        obs_a["action_history"] = [dict(self._REC)]
        obs_b = make_mock_obs(n_nets=3, pads_per_net=1)
        obs_b["action_history"] = [
            {**self._REC, "action_type": 6, "has_pointer": False,
             "pointer_xy": [0.0, 0.0], "routing_mode": -1, "net_id": None},
            dict(self._REC),
        ]
        tok = make_tok(action_history_len=2)
        with torch.no_grad():
            ea = tok([obs_a], action_type_weight=w).token_embeddings
            eb = tok([obs_b], action_type_weight=w).token_embeddings
        # obs_a: REC at age 0; obs_b: REC at age 1. Compare the REC entry's
        # at-token embeddings across the two ages — must differ.
        seq = int(tok._walk_obs([obs_a])["seq_lens"][0])
        a_at0 = ea[0, seq - 8]   # layout: [.., hist0(3), hist1(3), VAL, SOD]
        b_at1 = eb[0, seq - 5]
        assert not torch.allclose(a_at0, b_at1)

    def test_net_slot_binding(self):
        """A history entry carries its net's slot id (all 3 tokens); None → -1."""
        obs = make_mock_obs(n_nets=3, pads_per_net=1)
        obs["action_history"] = [dict(self._REC)]           # net_id=2 -> slot 1
        tok = make_tok(action_history_len=2)
        walk = tok._walk_obs([obs])
        slots = walk["action_history"][7]
        assert slots.tolist() == [1, -1]                    # entry1 = sentinel

    def test_legacy_k_must_be_1(self):
        with pytest.raises(ValueError):
            make_tok(action_history_len=3, legacy_action_history=True)

    def test_legacy_has_no_history_params(self):
        names = [n for n, _ in make_tok(
            action_history_len=1, legacy_action_history=True,
        ).named_parameters()]
        assert not any("history_age" in n for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
