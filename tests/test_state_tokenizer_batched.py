"""Parity tests for BatchedStateTokenizer.

The batched implementation must produce **bit-equivalent** TokenizerOutput
to the per-obs StateTokenizer when both share the same vocab weights.
This is the headline correctness check for the batched tokenizer.

Tolerance: ``atol=1e-4`` for ``token_embeddings`` (loosened from float32
noise level 1e-6 because batched matmul has different reduction order
than per-obs matmul; in practice we observe ≤1.2e-6 across all tested
configurations, so the budget is comfortably met).
"""

from __future__ import annotations

import copy

import pytest
import torch

from tests.helpers.reference_tokenizer import StateTokenizer
from methods.rl_agent.models.v1.tokenizer import (
    BatchedStateTokenizer,
)
from tests._mock_obs import make_mock_obs


# ===================================================================
# Helpers
# ===================================================================
def make_pair(seed: int = 0, **kwargs) -> tuple[StateTokenizer, BatchedStateTokenizer]:
    """Build a (per-obs, batched) tokenizer pair with **identical** vocab
    weights. Uses load_state_dict to enforce bit-equality."""
    torch.manual_seed(seed)
    ref = StateTokenizer(**kwargs)
    torch.manual_seed(seed)
    bat = BatchedStateTokenizer(**kwargs)
    bat.vocab.load_state_dict(ref.vocab.state_dict())
    ref.eval()
    bat.eval()
    return ref, bat


def assert_outputs_equal(r, b, *, atol: float = 1e-4) -> None:
    """Assert all TokenizerOutput fields match between ref and batched."""
    diff = (r.token_embeddings - b.token_embeddings).abs().max().item()
    assert diff < atol, (
        f"token_embeddings max diff = {diff:.3e} (tol {atol:.0e})"
    )
    assert torch.equal(r.key_padding_mask, b.key_padding_mask), \
        "key_padding_mask mismatch"
    assert torch.equal(r.seq_lens, b.seq_lens), "seq_lens mismatch"
    assert torch.equal(r.net_indices, b.net_indices), "net_indices mismatch"
    assert torch.equal(r.cand_indices, b.cand_indices), "cand_indices mismatch"
    assert r.cand_mm_list == b.cand_mm_list, "cand_mm_list mismatch"


# ===================================================================
# Phase A: Numerical parity
# ===================================================================
class TestBatchedVsPerObsParity:
    """Each test invokes both tokenizers on the same obs_list and asserts
    every TokenizerOutput field matches."""

    # ---------- Coverage of obs combinations ----------

    def test_parity_basic_4_obs(self):
        ref, bat = make_pair(d_model=64, n_freq=8)
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
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_single_obs(self):
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [make_mock_obs(
            n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
            is_routing=True, current_net_phase=2,
        )]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_large_batch(self):
        """B=64: exercises scatter under realistic per-minibatch load."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [
            make_mock_obs(
                n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                is_routing=(i % 2 == 0),
                current_net_phase=2 if i % 2 == 0 else 0,
            )
            for i in range(64)
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_with_slot_perm(self):
        """slot_perm augmentation per obs."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [
            make_mock_obs(n_nets=3, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
            for _ in range(3)
        ]
        for i, o in enumerate(obs_list):
            o["_aug"] = {"slot_perm": [(i + 1 + k) % 3 for k in range(3)]}
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_with_disable_slot_emb(self):
        ref, bat = make_pair(d_model=64, n_freq=8, disable_slot_emb=True)
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
            for _ in range(4)
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_coord_encoding_mlp(self):
        ref, bat = make_pair(
            d_model=64, n_freq=8, coord_encoding="mlp", mlp_hidden=128,
        )
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
            for _ in range(4)
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_idle_obs_no_head_xy(self):
        """All obs idle (current_net_phase=0, head_xy=None path)."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=0,
                          is_routing=False, current_net_phase=0)
            for _ in range(4)
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_routing_obs_with_head_xy(self):
        """All obs routing (head_xy active)."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_tracks=2, n_vias=1,
                          n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
            for _ in range(4)
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_mixed_head_active(self):
        """Half obs idle, half routing (mixed has_head mask in one batch)."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = []
        for i in range(8):
            if i % 2 == 0:
                obs_list.append(make_mock_obs(
                    n_nets=2, pads_per_net=2, n_ratsnest_per_net=0,
                    is_routing=False, current_net_phase=0,
                ))
            else:
                obs_list.append(make_mock_obs(
                    n_nets=2, pads_per_net=2, n_tracks=1,
                    n_ratsnest_per_net=1,
                    is_routing=True, current_net_phase=2,
                ))
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_varying_n_nets(self):
        """Different n_nets per obs (variable seq_len, padding stress)."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [
            make_mock_obs(n_nets=k, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
            for k in [1, 2, 3, 5]
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_no_tracks_yet(self):
        """Routing started but no tracks/vias/rats yet."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2,
                          n_tracks=0, n_vias=0, n_ratsnest_per_net=0,
                          is_routing=True, current_net_phase=2)
            for _ in range(3)
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_with_drc_violations(self):
        """DRC violation tokens in obs — parity between per-obs and batched."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        drc_a = [
            {"x_mm": 105.0, "y_mm": 52.0, "layer": 1,
             "error_type": "Clearance violation", "type_id": 0,
             "severity": 0x20, "net_names": ["NET1"]},
            {"x_mm": 110.0, "y_mm": 60.0, "layer": 2,
             "error_type": "Track has unconnected end", "type_id": 2,
             "severity": 0x10, "net_names": ["NET2"]},
            {"x_mm": 120.0, "y_mm": 70.0, "layer": 1,
             "error_type": "Missing connection between items", "type_id": 6,
             "severity": 0x20, "net_names": []},  # orphan
        ]
        drc_b = [
            {"x_mm": 130.0, "y_mm": 65.0, "layer": 1,
             "error_type": "Clearance violation", "type_id": 0,
             "severity": 0x20, "net_names": ["NET1", "NET2"]},
        ]
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2,
                          drc_violations=drc_a),
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2,
                          drc_violations=drc_b),
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2,
                          drc_violations=[]),  # empty list
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)
        # Middle obs has 1 DRC token; first has 3; last has 0.
        assert r.seq_lens[0].item() - r.seq_lens[2].item() == 3
        assert r.seq_lens[1].item() - r.seq_lens[2].item() == 1

    def test_parity_with_drc_and_aug(self):
        """DRC coords go through _norm_pos → honor axis_swap / sign / shift."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        drc = [
            {"x_mm": 112.0, "y_mm": 58.0, "layer": 1,
             "error_type": "Clearance violation", "type_id": 0,
             "severity": 0x20, "net_names": ["NET1"]},
        ]
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2,
                          drc_violations=drc),
            make_mock_obs(n_nets=2, pads_per_net=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2,
                          drc_violations=drc),
        ]
        obs_list[0]["_aug"] = {
            "axis_swap": True, "flip_x": -1, "flip_y": 1,
            "nn_dx": 0.02, "nn_dy": -0.03,
        }
        obs_list[1]["_aug"] = {
            "scale_x": 1.1, "scale_y": 0.9,
            "aug_cx": 130.0, "aug_cy": 70.0,
        }
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)

    def test_parity_with_tracks_and_vias(self):
        """Active routing with non-trivial tracks + vias."""
        ref, bat = make_pair(d_model=64, n_freq=8)
        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2,
                          n_tracks=3, n_vias=2, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
            for _ in range(4)
        ]
        with torch.no_grad():
            r = ref(obs_list)
            b = bat(obs_list)
        assert_outputs_equal(r, b)


# ===================================================================
# Phase B: Gradient parity
# ===================================================================
class TestGradientParity:
    """Backprop a small loss through both tokenizers and assert all 11
    learnable parameters' gradients match. Catches scatter mistakes that
    would corrupt training silently."""

    def test_gradient_parity_through_all_projections(self):
        ref, bat = make_pair(d_model=64, n_freq=8)
        # Make trainable for the gradient check
        ref.train(); bat.train()

        obs_list = [
            make_mock_obs(n_nets=2, pads_per_net=2,
                          n_tracks=2, n_vias=1, n_ratsnest_per_net=1,
                          is_routing=True, current_net_phase=2)
            for _ in range(4)
        ]
        # Inject DRC violations so drc_proj receives gradient.
        for i, o in enumerate(obs_list):
            o["drc_violations"] = [
                {"x_mm": 105.0 + i, "y_mm": 52.0 + i, "layer": 1,
                 "error_type": "Clearance violation", "type_id": 0,
                 "severity": 0x20, "net_names": ["NET1"]},
                {"x_mm": 115.0, "y_mm": 60.0, "layer": 2,
                 "error_type": "Track has unconnected end", "type_id": 2,
                 "severity": 0x10, "net_names": []},
            ]

        # Forward + loss (sum of valid token embeddings, no padding)
        ref_out = ref(obs_list)
        bat_out = bat(obs_list)

        # Mask padded positions before summing so the gradient signal
        # only depends on tokens that actually carry slot/projection
        # contributions (matches what the policy backbone consumes).
        ref_mask = (~ref_out.key_padding_mask).unsqueeze(-1).to(
            ref_out.token_embeddings.dtype,
        )
        bat_mask = (~bat_out.key_padding_mask).unsqueeze(-1).to(
            bat_out.token_embeddings.dtype,
        )
        ref_loss = (ref_out.token_embeddings * ref_mask).sum()
        bat_loss = (bat_out.token_embeddings * bat_mask).sum()

        ref_loss.backward()
        bat_loss.backward()

        proj_names = [
            "pad_proj", "via_proj", "track_proj", "edge_proj",
            "rat_proj", "head_proj", "cand_proj", "net_proj",
            "board_proj", "endpoint_proj", "drc_proj",
        ]
        for name in proj_names:
            ref_grad = getattr(ref.vocab, name).weight.grad
            bat_grad = getattr(bat.vocab, name).weight.grad
            assert ref_grad is not None, f"ref {name}.grad is None"
            assert bat_grad is not None, f"bat {name}.grad is None"
            diff = (ref_grad - bat_grad).abs().max().item()
            assert diff < 1e-4, (
                f"{name}.weight.grad mismatch: max diff = {diff:.3e}"
            )

        # Structural embedding (used for VAL + SOD per obs).
        ref_se = ref.vocab.structural_embed.weight.grad
        bat_se = bat.vocab.structural_embed.weight.grad
        assert ref_se is not None and bat_se is not None
        diff = (ref_se - bat_se).abs().max().item()
        assert diff < 1e-4, (
            f"structural_embed.weight.grad mismatch: max diff = {diff:.3e}"
        )

        # slot_scale (scalar learnable; slot_emb_table itself is a
        # non-trainable buffer).
        ref_ss = ref.vocab.slot_scale.grad
        bat_ss = bat.vocab.slot_scale.grad
        assert ref_ss is not None and bat_ss is not None
        diff = (ref_ss - bat_ss).abs().max().item()
        assert diff < 1e-4, (
            f"slot_scale.grad mismatch: max diff = {diff:.3e}"
        )


# ===================================================================
# Through-hole (THT) pad encoding
# ===================================================================
class TestThtPadEncoding:
    """Pad uses via-style ``(layer_start, layer_end)`` features. SMD/connect
    pads collapse to ``ls == le``; thru-hole pads (parser sentinel
    ``layer == 0``) expand to ``(1, n_copper)``. The latter must:

    1. produce a non-degenerate (non-(0,0)) feature pair, so the policy net
       sees a real layer-range signal,
    2. distinguish 2-copper THT from 4-copper THT (the failure mode of the
       earlier 2-dim "span-remainder" proposal),
    3. equal what a thru-via barrel of the same span would feed in.
    """

    @staticmethod
    def _walk_pad_features(obs):
        """Run the batched tokenizer's Phase-1 walk and return
        ``(pad_ls, pad_le)`` for the given obs."""
        torch.manual_seed(0)
        bat = BatchedStateTokenizer(d_model=32, n_freq=4)
        walk = bat._walk_obs([obs])
        # walk["pad"] = (xy, wh, ls, le, head, has, obs_idx, pos, slot, shape_id)
        # The fields are np (N,2) — converted to list-of-list for row-wise ==.
        _, _, p_ls, p_le, *_ = walk["pad"]
        return p_ls.tolist(), p_le.tolist()

    def _make_obs_with_tht_pad(self, copper_layers: int) -> dict:
        """One net, one regular pad on layer 1, one THT pad (layer == 0)."""
        obs = make_mock_obs(
            n_nets=1, pads_per_net=1, n_ratsnest_per_net=0,
            copper_layers=copper_layers,
        )
        net = obs["board_static"]["nets"]["net_1"]
        # Append a thru-hole pad — parser sentinel for "spans every copper layer".
        net["pads"]["pad_tht"] = {
            "id": "Dtht",
            "center": {"id": "Ptht", "xy": [120.0, 65.0]},
            "width": 0.6, "height": 0.6,
            "layer": 0,
        }
        return obs

    def test_single_layer_pad_has_ls_equal_le(self):
        """Regular SMD pad on layer 1: ls == le == encode(1)."""
        obs = make_mock_obs(
            n_nets=1, pads_per_net=1, n_ratsnest_per_net=0, copper_layers=2,
        )
        p_ls, p_le = self._walk_pad_features(obs)
        assert len(p_ls) == 1 and len(p_le) == 1
        assert p_ls[0] == p_le[0], (
            f"single-layer pad must collapse ls == le; got {p_ls[0]} vs {p_le[0]}"
        )

    def test_tht_pad_expands_to_full_layer_range(self):
        """THT pad on a 2-layer board: ls = encode(1), le = encode(2)."""
        from methods.rl_agent.models.v1.embedding import encode_layer

        obs = self._make_obs_with_tht_pad(copper_layers=2)
        p_ls, p_le = self._walk_pad_features(obs)
        # 2 pads: regular on layer 1, THT
        assert len(p_ls) == 2

        expected_top = list(encode_layer(1, n_copper=2))
        expected_bot = list(encode_layer(2, n_copper=2))
        # The THT pad is the second appended (after the regular one).
        assert p_ls[1] == expected_top, (
            f"THT pad layer_start expected {expected_top}, got {p_ls[1]}"
        )
        assert p_le[1] == expected_bot, (
            f"THT pad layer_end expected {expected_bot}, got {p_le[1]}"
        )

    def test_tht_pad_distinguishable_across_copper_layers(self):
        """THT on 2-layer board != THT on 4-layer board.

        Guards against the degenerate 2-dim "span-remainder" encoding that
        would have collapsed both to (0, 0).
        """
        obs2 = self._make_obs_with_tht_pad(copper_layers=2)
        obs4 = self._make_obs_with_tht_pad(copper_layers=4)
        ls2, le2 = self._walk_pad_features(obs2)
        ls4, le4 = self._walk_pad_features(obs4)
        # THT pad is the second pad in each obs.
        assert le2[1] != le4[1], (
            f"THT layer_end must differ across n_copper; got {le2[1]} (2L) "
            f"vs {le4[1]} (4L)"
        )

    def test_tht_pad_matches_thru_via_encoding(self):
        """A THT pad on a 2-layer board must feed the same layer-range
        features as a thru via spanning layers 1..2 — they ARE the same
        plated barrel.
        """
        obs = self._make_obs_with_tht_pad(copper_layers=2)
        p_ls, p_le = self._walk_pad_features(obs)

        # Build a thru via over the same board and compare via_ls/le.
        obs_via = make_mock_obs(
            n_nets=1, pads_per_net=1, n_tracks=0, n_vias=1,
            n_ratsnest_per_net=0, copper_layers=2, is_routing=True,
        )
        torch.manual_seed(0)
        bat = BatchedStateTokenizer(d_model=32, n_freq=4)
        walk = bat._walk_obs([obs_via])
        _, v_ls, v_le, *_ = walk["via"]
        v_ls, v_le = v_ls.tolist(), v_le.tolist()

        # THT pad is index 1; thru via is the only via.
        assert p_ls[1] == v_ls[0], (
            f"THT pad layer_start {p_ls[1]} must match thru-via {v_ls[0]}"
        )
        assert p_le[1] == v_le[0], (
            f"THT pad layer_end {p_le[1]} must match thru-via {v_le[0]}"
        )
