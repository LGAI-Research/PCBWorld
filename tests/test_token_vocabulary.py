"""Tests for methods.rl_agent.models.v1.embedding.

These tests cover the API where each entity is fused into a single token
and per-entity Linear projections (``pad_proj`` / ``track_proj`` /
``via_proj`` / ``cand_proj`` / ``net_proj`` / ``board_proj`` /
``edge_proj`` / ``rat_proj`` / ``head_proj``) own the fusion weights.
"""

from __future__ import annotations

import math

import pytest
import torch

from methods.rl_agent.models.v1.embedding import (
    MAX_COPPER,
    NUM_CAND_TYPES,
    NUM_ENTITY_TYPES,
    NUM_STRUCTURAL_TOKENS,
    CandidateType,
    EntityType,
    FourierEncoding,
    StructuralToken,
    TokenVocabulary,
    cand_type_to_entity,
    encode_layer,
)


# ===================================================================
# encode_layer
# ===================================================================
class TestEncodeLayer:
    def test_top_layer(self):
        dt, db = encode_layer(1, 4)
        assert dt == pytest.approx(0.0)
        assert db == pytest.approx(3 / MAX_COPPER)

    def test_bottom_layer(self):
        dt, db = encode_layer(4, 4)
        assert dt == pytest.approx(3 / MAX_COPPER)
        assert db == pytest.approx(0.0)

    def test_invalid_layer_raises(self):
        with pytest.raises(ValueError):
            encode_layer(5, 4)
        with pytest.raises(ValueError):
            encode_layer(0, 4)

    def test_constant_max_copper_denominator(self):
        # Same layer on different stack thicknesses → top distance is the
        # same; bottom distance scales with stack thickness.
        d1 = encode_layer(2, 4)
        d2 = encode_layer(2, 32)
        assert d1[0] == pytest.approx(d2[0])
        assert d1[1] != d2[1]


# ===================================================================
# FourierEncoding
# ===================================================================
class TestFourierEncoding:
    def test_output_dim_2d(self):
        f = FourierEncoding(n_freq=8)
        x = torch.randn(3, 2)
        assert f(x).shape == (3, 2 * 2 * 8)
        assert f.output_dim(2) == 32

    def test_output_dim_1d(self):
        f = FourierEncoding(n_freq=16)
        x = torch.randn(3, 1)
        assert f(x).shape == (3, 1 * 2 * 16)

    def test_zero_input_specific_pattern(self):
        f = FourierEncoding(n_freq=4)
        x = torch.zeros(1, 1)
        out = f(x)
        # sin(0)=0, cos(0)=1
        sins = out[..., :4]
        coss = out[..., 4:]
        assert torch.allclose(sins, torch.zeros_like(sins))
        assert torch.allclose(coss, torch.ones_like(coss))


# ===================================================================
# Reduced StructuralToken / EntityType
# ===================================================================
class TestEnums:
    def test_structural_token_reduced(self):
        names = {s.name for s in StructuralToken}
        assert names == {"VAL", "SOD", "SEQ_PAD"}
        assert NUM_STRUCTURAL_TOKENS == 3

    def test_entity_type_complete(self):
        names = {e.name for e in EntityType}
        for must in (
            "BOARD", "EDGE", "NET", "PAD", "TRACK", "VIA", "RAT",
            "HEAD", "CAND_PAD", "CAND_TRACK_END", "CAND_VIA",
            "CAND_RAT", "CAND_DIR",
        ):
            assert must in names

    def test_cand_type_count(self):
        assert NUM_CAND_TYPES == 5

    def test_cand_type_to_entity_mapping(self):
        assert cand_type_to_entity(CandidateType.PAD_POINT.value) == int(EntityType.CAND_PAD)
        assert cand_type_to_entity(CandidateType.TRACK_ENDPOINT.value) == int(EntityType.CAND_TRACK_END)
        assert cand_type_to_entity(CandidateType.VIA_CENTER.value) == int(EntityType.CAND_VIA)
        assert cand_type_to_entity(CandidateType.RATSNEST.value) == int(EntityType.CAND_RAT)
        assert cand_type_to_entity(CandidateType.DIRECTIONAL.value) == int(EntityType.CAND_DIR)


# ===================================================================
# Vocabulary structure
# ===================================================================
def _vocab(d_model=32, n_freq=8, **kw):
    return TokenVocabulary(d_model=d_model, n_freq=n_freq, **kw)


class TestVocabularyStructure:
    def test_module_count(self):
        v = _vocab()
        assert v.entity_type_embed.num_embeddings == NUM_ENTITY_TYPES
        assert v.structural_embed.num_embeddings == NUM_STRUCTURAL_TOKENS
        # Per-entity projections all map to d_model.
        for proj_name in (
            "pad_proj", "via_proj", "track_proj", "edge_proj", "rat_proj",
            "head_proj", "cand_proj", "net_proj", "board_proj",
            "endpoint_proj",
        ):
            proj = getattr(v, proj_name)
            assert isinstance(proj, torch.nn.Linear)
            assert proj.out_features == v.d_model

    def test_slot_emb_table_shape_and_columns(self):
        """``orthogonal_`` on a (n_slots, d_model) tall matrix gives orthonormal
        COLUMNS (since n_slots > d_model)."""
        v = _vocab()
        T = v.slot_emb_table
        assert T.shape == (v.n_max_slots, v.d_model)
        # Column-wise orthonormality: T.T @ T == I_{d_model}.
        col_gram = T.T @ T
        assert torch.allclose(col_gram, torch.eye(v.d_model), atol=1e-4)

    def test_slot_scale_default_init(self):
        v = _vocab()
        assert isinstance(v.slot_scale, torch.nn.Parameter)
        assert v.slot_scale.item() == pytest.approx(0.3)


# ===================================================================
# Per-entity encoders
# ===================================================================
def _zero_pos_2d(K=1):
    return torch.zeros(K, 2)


class TestPerEntityEncoders:
    def test_encode_pad_shape(self):
        v = _vocab()
        out = v.encode_pad(
            xy=torch.randn(3, 2),
            wh=torch.rand(3, 2),
            layer_start_dt_db=torch.rand(3, 2),
            layer_end_dt_db=torch.rand(3, 2),
            head_xy=torch.randn(2),
        )
        assert out.shape == (3, v.d_model)
        assert torch.isfinite(out).all()

    def test_encode_via_shape(self):
        v = _vocab()
        out = v.encode_via(
            xy=torch.randn(2, 2),
            layer_start_dt_db=torch.rand(2, 2),
            layer_end_dt_db=torch.rand(2, 2),
            via_dia=torch.rand(2, 1),
            head_xy=None,
        )
        assert out.shape == (2, v.d_model)

    def test_encode_track_shape(self):
        v = _vocab()
        out = v.encode_track(
            xy1=torch.randn(2, 2),
            xy2=torch.randn(2, 2),
            width=torch.rand(2, 1),
            layer_dt_db=torch.rand(2, 2),
            head_xy=torch.randn(2),
        )
        assert out.shape == (2, v.d_model)

    def test_encode_edge_shape(self):
        v = _vocab()
        out = v.encode_edge(
            xy1=torch.randn(4, 2),
            xy2=torch.randn(4, 2),
            width=torch.rand(4, 1),
            xy_mid=torch.randn(4, 2),
        )
        assert out.shape == (4, v.d_model)

    def test_encode_rat_shape(self):
        v = _vocab()
        out = v.encode_rat(
            xy=torch.randn(5, 2),
            head_xy=torch.randn(2),
        )
        assert out.shape == (5, v.d_model)

    def test_encode_head_shape(self):
        v = _vocab()
        out = v.encode_head(
            xy=torch.randn(1, 2),
            layer_dt_db=torch.rand(1, 2),
            routing_mode=torch.tensor([2], dtype=torch.long),
            net_phase=torch.tensor([1], dtype=torch.long),
            step_ratio=torch.tensor([[0.5]]),
        )
        assert out.shape == (1, v.d_model)

    def test_encode_cand_shape(self):
        v = _vocab()
        out = v.encode_cand(
            cand_type_ints=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
            xy=torch.randn(5, 2),
            layer_dt_db=torch.rand(5, 2),
            head_xy=torch.randn(2),
        )
        assert out.shape == (5, v.d_model)

    def test_encode_net_shape(self):
        v = _vocab()
        out = v.encode_net(
            track_w=torch.rand(2, 1),
            clearance=torch.rand(2, 1),
            via_dia=torch.rand(2, 1),
            closed=torch.zeros(2, 1),
        )
        assert out.shape == (2, v.d_model)

    def test_encode_net_requires_closed_flag(self):
        v = _vocab()
        try:
            v.encode_net(
                track_w=torch.rand(2, 1),
                clearance=torch.rand(2, 1),
                via_dia=torch.rand(2, 1),
            )
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("encode_net must require `closed` when not legacy")

    def test_encode_net_legacy_3f_shape(self):
        from methods.rl_agent.models.v1.embedding import TokenVocabulary

        v = TokenVocabulary(d_model=32, n_freq=4, legacy_net_encoding=True)
        out = v.encode_net(
            track_w=torch.rand(2, 1),
            clearance=torch.rand(2, 1),
            via_dia=torch.rand(2, 1),
        )
        assert out.shape == (2, v.d_model)
        assert v.net_proj.in_features == 3 * v.fenc_dim

    def test_encode_board_shape(self):
        v = _vocab()
        out = v.encode_board(
            bbox_origin=torch.randn(1, 2),
            bbox_size=torch.rand(1, 2),
            n_copper=torch.tensor([[2.0]]),
        )
        assert out.shape == (1, v.d_model)

    def test_encode_drc_shape(self):
        v = _vocab()
        out = v.encode_drc(
            xy=torch.randn(3, 2),
            layer_dt_db=torch.rand(3, 2),
            type_id=torch.tensor([0, 2, 6], dtype=torch.long),
            severity_flag=torch.tensor([[1.0], [0.0], [1.0]]),
            head_xy=torch.randn(2),
        )
        assert out.shape == (3, v.d_model)

    def test_encode_drc_no_head(self):
        """has_head=0 branch produces finite output (no NaNs)."""
        v = _vocab()
        out = v.encode_drc(
            xy=torch.randn(2, 2),
            layer_dt_db=torch.rand(2, 2),
            type_id=torch.tensor([0, 1], dtype=torch.long),
            severity_flag=torch.tensor([[1.0], [0.0]]),
            head_xy=None,
        )
        assert out.shape == (2, v.d_model)
        assert torch.isfinite(out).all()

    def test_encode_drc_type_embed_differs(self):
        """Different taxonomy ids produce different outputs."""
        torch.manual_seed(0)
        v = _vocab().eval()
        xy = torch.tensor([[0.3, 0.4]])
        ld = torch.tensor([[0.0, 0.5]])
        sev = torch.tensor([[1.0]])
        head = torch.tensor([0.5, 0.5])
        a = v.encode_drc(xy, ld, torch.tensor([0], dtype=torch.long), sev, head)
        b = v.encode_drc(xy, ld, torch.tensor([3], dtype=torch.long), sev, head)
        assert not torch.allclose(a, b)

    def test_encode_drc_severity_differs(self):
        torch.manual_seed(0)
        v = _vocab().eval()
        xy = torch.tensor([[0.3, 0.4]])
        ld = torch.tensor([[0.0, 0.5]])
        tid = torch.tensor([0], dtype=torch.long)
        head = torch.tensor([0.5, 0.5])
        err = v.encode_drc(xy, ld, tid, torch.tensor([[1.0]]), head)
        warn = v.encode_drc(xy, ld, tid, torch.tensor([[0.0]]), head)
        assert not torch.allclose(err, warn)


# ===================================================================
# Symmetry / invariance properties
# ===================================================================
class TestEntitySymmetries:
    def test_track_endpoint_swap_invariant(self):
        torch.manual_seed(0)
        v = _vocab().eval()
        xy1 = torch.tensor([[0.1, 0.2]])
        xy2 = torch.tensor([[0.7, 0.4]])
        w = torch.tensor([[0.05]])
        ld = torch.tensor([[0.0, 0.5]])
        head = torch.tensor([0.5, 0.5])
        a = v.encode_track(xy1, xy2, w, ld, head)
        b = v.encode_track(xy2, xy1, w, ld, head)
        assert torch.allclose(a, b, atol=1e-6)

    def test_edge_endpoint_swap_invariant(self):
        torch.manual_seed(0)
        v = _vocab().eval()
        xy1 = torch.tensor([[0.0, 0.0]])
        xy2 = torch.tensor([[1.0, 0.0]])
        w = torch.tensor([[0.1]])
        mid = torch.tensor([[0.5, 0.0]])
        a = v.encode_edge(xy1, xy2, w, xy_mid=mid)
        b = v.encode_edge(xy2, xy1, w, xy_mid=mid)
        assert torch.allclose(a, b, atol=1e-6)

    def test_pad_no_head_vs_with_head_differ(self):
        """has_head_mask is on for one path and off for the other."""
        torch.manual_seed(0)
        v = _vocab().eval()
        xy = torch.tensor([[0.3, 0.4]])
        wh = torch.tensor([[0.5, 0.5]])
        ls = torch.tensor([[0.0, 0.5]])
        le = torch.tensor([[0.0, 0.5]])
        a = v.encode_pad(xy, wh, ls, le, head_xy=None)
        b = v.encode_pad(xy, wh, ls, le, head_xy=torch.tensor([0.0, 0.0]))
        assert not torch.allclose(a, b)


# ===================================================================
# Gradient flow
# ===================================================================
class TestGradientFlow:
    def test_pad_gradient_flows_through_all_inputs(self):
        v = _vocab()
        xy = torch.randn(2, 2, requires_grad=True)
        wh = torch.rand(2, 2, requires_grad=True)
        ls = torch.rand(2, 2, requires_grad=True)
        le = torch.rand(2, 2, requires_grad=True)
        out = v.encode_pad(xy, wh, ls, le, head_xy=torch.tensor([0.0, 0.0]))
        out.sum().backward()
        for t in (xy, wh, ls, le):
            assert t.grad is not None and t.grad.abs().sum() > 0

    def test_track_gradient_through_endpoints(self):
        v = _vocab()
        xy1 = torch.randn(2, 2, requires_grad=True)
        xy2 = torch.randn(2, 2, requires_grad=True)
        w = torch.rand(2, 1, requires_grad=True)
        ld = torch.rand(2, 2, requires_grad=True)
        out = v.encode_track(xy1, xy2, w, ld, head_xy=torch.tensor([0.0, 0.0]))
        out.sum().backward()
        for t in (xy1, xy2, w, ld):
            assert t.grad is not None and t.grad.abs().sum() > 0


# ===================================================================
# MLP coord encoding
# ===================================================================
class TestMLPCoordEncoding:
    def test_mlp_pad_works(self):
        v = TokenVocabulary(
            d_model=32, n_freq=8, coord_encoding="mlp", mlp_hidden=64,
        )
        out = v.encode_pad(
            xy=torch.randn(2, 2),
            wh=torch.rand(2, 2),
            layer_start_dt_db=torch.rand(2, 2),
            layer_end_dt_db=torch.rand(2, 2),
            head_xy=torch.tensor([0.0, 0.0]),
        )
        assert out.shape == (2, v.d_model)
        assert torch.isfinite(out).all()


# ===================================================================
# Per-entity head_xy + has_head signature parity. Verifies the (K, 2)
# head_xy + (K, 1) has_head call path is bit-equivalent to the legacy
# paths ((2,) head_xy and head_xy=None).
# ===================================================================
class TestPerEntityHeadXyParity:
    """For each encode_* that takes head_xy, prove:
      A) head_xy=(2,) (broadcast)  ==  head_xy=(K, 2) per-entity + has_head=ones
      B) head_xy=None              ==  head_xy=zeros(K, 2) + has_head=zeros
    """

    def _make_vocab(self):
        torch.manual_seed(0)
        return TokenVocabulary(d_model=32, n_freq=8)

    @staticmethod
    def _broadcast(head_xy_2: torch.Tensor, K: int) -> torch.Tensor:
        return head_xy_2.unsqueeze(0).expand(K, 2).contiguous()

    # ----- encode_pad -----------------------------------------------
    def test_encode_pad_legacy_shape_matches_per_entity(self):
        v = self._make_vocab()
        K = 5
        torch.manual_seed(1)
        xy = torch.randn(K, 2)
        wh = torch.rand(K, 2)
        ls = torch.rand(K, 2)
        le = torch.rand(K, 2)
        head = torch.tensor([0.5, -0.3])
        out_old = v.encode_pad(xy, wh, ls, le, head)
        out_new = v.encode_pad(
            xy, wh, ls, le, self._broadcast(head, K), has_head=torch.ones(K, 1),
        )
        assert torch.allclose(out_old, out_new, atol=1e-6)

    def test_encode_pad_has_head_zero_matches_legacy_none(self):
        v = self._make_vocab()
        K = 5
        torch.manual_seed(2)
        xy = torch.randn(K, 2)
        wh = torch.rand(K, 2)
        ls = torch.rand(K, 2)
        le = torch.rand(K, 2)
        out_none = v.encode_pad(xy, wh, ls, le, head_xy=None)
        out_zero = v.encode_pad(
            xy, wh, ls, le, torch.zeros(K, 2), has_head=torch.zeros(K, 1),
        )
        assert torch.allclose(out_none, out_zero, atol=1e-6)

    # ----- encode_via -----------------------------------------------
    def test_encode_via_legacy_shape_matches_per_entity(self):
        v = self._make_vocab()
        K = 4
        torch.manual_seed(3)
        xy = torch.randn(K, 2)
        ls = torch.rand(K, 2)
        le = torch.rand(K, 2)
        dia = torch.rand(K, 1)
        head = torch.tensor([-0.1, 0.2])
        out_old = v.encode_via(xy, ls, le, dia, head)
        out_new = v.encode_via(
            xy, ls, le, dia, self._broadcast(head, K),
            has_head=torch.ones(K, 1),
        )
        assert torch.allclose(out_old, out_new, atol=1e-6)

    def test_encode_via_has_head_zero_matches_legacy_none(self):
        v = self._make_vocab()
        K = 4
        torch.manual_seed(4)
        xy = torch.randn(K, 2)
        ls = torch.rand(K, 2)
        le = torch.rand(K, 2)
        dia = torch.rand(K, 1)
        out_none = v.encode_via(xy, ls, le, dia, head_xy=None)
        out_zero = v.encode_via(
            xy, ls, le, dia, torch.zeros(K, 2), has_head=torch.zeros(K, 1),
        )
        assert torch.allclose(out_none, out_zero, atol=1e-6)

    # ----- encode_track ---------------------------------------------
    def test_encode_track_legacy_shape_matches_per_entity(self):
        v = self._make_vocab()
        K = 3
        torch.manual_seed(5)
        xy1 = torch.randn(K, 2)
        xy2 = torch.randn(K, 2)
        w = torch.rand(K, 1)
        ld = torch.rand(K, 2)
        head = torch.tensor([0.7, 0.1])
        out_old = v.encode_track(xy1, xy2, w, ld, head)
        out_new = v.encode_track(
            xy1, xy2, w, ld, self._broadcast(head, K),
            has_head=torch.ones(K, 1),
        )
        assert torch.allclose(out_old, out_new, atol=1e-6)

    def test_encode_track_has_head_zero_matches_legacy_none(self):
        v = self._make_vocab()
        K = 3
        torch.manual_seed(6)
        xy1 = torch.randn(K, 2)
        xy2 = torch.randn(K, 2)
        w = torch.rand(K, 1)
        ld = torch.rand(K, 2)
        out_none = v.encode_track(xy1, xy2, w, ld, head_xy=None)
        out_zero = v.encode_track(
            xy1, xy2, w, ld, torch.zeros(K, 2), has_head=torch.zeros(K, 1),
        )
        assert torch.allclose(out_none, out_zero, atol=1e-6)

    # ----- encode_rat -----------------------------------------------
    def test_encode_rat_broadcast_head_matches_per_entity(self):
        v = self._make_vocab()
        K = 6
        torch.manual_seed(7)
        xy = torch.randn(K, 2)
        head = torch.tensor([0.0, 0.4])
        out_old = v.encode_rat(xy, head)
        out_new = v.encode_rat(
            xy, self._broadcast(head, K), has_head=torch.ones(K, 1),
        )
        assert torch.allclose(out_old, out_new, atol=1e-6)

    def test_encode_rat_has_head_zero_matches_none(self):
        v = self._make_vocab()
        K = 6
        torch.manual_seed(8)
        xy = torch.randn(K, 2)
        out_none = v.encode_rat(xy, head_xy=None)
        out_zero = v.encode_rat(
            xy, torch.zeros(K, 2), has_head=torch.zeros(K, 1),
        )
        assert torch.allclose(out_none, out_zero, atol=1e-6)

    # ----- encode_cand ----------------------------------------------
    def test_encode_cand_legacy_shape_matches_per_entity(self):
        v = self._make_vocab()
        K = 4
        torch.manual_seed(9)
        cand_types = torch.zeros(K, dtype=torch.long)  # CTYPE_PAD = 0
        xy = torch.randn(K, 2)
        ld = torch.rand(K, 2)
        head = torch.tensor([-0.2, 0.5])
        out_old = v.encode_cand(cand_types, xy, ld, head)
        out_new = v.encode_cand(
            cand_types, xy, ld, self._broadcast(head, K),
            has_head=torch.ones(K, 1),
        )
        assert torch.allclose(out_old, out_new, atol=1e-6)

    def test_encode_cand_has_head_zero_matches_legacy_none(self):
        v = self._make_vocab()
        K = 4
        torch.manual_seed(10)
        cand_types = torch.zeros(K, dtype=torch.long)
        xy = torch.randn(K, 2)
        ld = torch.rand(K, 2)
        out_none = v.encode_cand(cand_types, xy, ld, head_xy=None)
        out_zero = v.encode_cand(
            cand_types, xy, ld, torch.zeros(K, 2),
            has_head=torch.zeros(K, 1),
        )
        assert torch.allclose(out_none, out_zero, atol=1e-6)


def test_n_max_slots_configurable_and_ckpt_compat():
    """n_max_slots wiring: default 512, checkpoint-compatible; raising it to
    128 allows tokenizing boards with >64 nets."""
    import torch
    from configs.loader.schema import RLPolicyConfig
    from tests._mock_obs import make_mock_obs

    # Default value + from_checkpoint fallback (an old checkpoint with no key
    # falls back to the literal 64, independent of the YAML default).
    assert RLPolicyConfig().n_max_slots == 512
    assert RLPolicyConfig.from_checkpoint({}).n_max_slots == 64
    assert RLPolicyConfig.from_checkpoint({"n_max_slots": 128}).n_max_slots == 128

    torch.manual_seed(0)
    big = RLPolicyConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64, n_freq=4,
                         use_critic=False, disable_slot_emb=True,
                         n_max_slots=128).build(None)
    assert big.tokenizer.vocab.n_max_slots == 128
    obs = make_mock_obs(n_nets=100, pads_per_net=1, n_ratsnest_per_net=0,
                        n_edges=0, is_routing=False, current_net_phase=0)
    with torch.no_grad():
        out = big.tokenizer([obs])          # would raise "only 64 slots" ValueError if this were 64
    assert out.token_embeddings.shape[0] == 1
    # checkpoint round-trip: a 128-slot state_dict loads into a model rebuilt with the same config
    sd = big.state_dict()
    twin = RLPolicyConfig.from_checkpoint(
        {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "n_freq": 4,
         "use_critic": False, "disable_slot_emb": True, "n_max_slots": 128},
    ).build(None)
    twin.load_state_dict(sd)
