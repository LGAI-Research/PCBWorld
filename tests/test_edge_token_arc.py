"""EDGE token 3-point (arc) extension — pure torch/numpy, no C++.

Current checkpoints encode edges from 3 points: the 2 endpoints (shared
``endpoint_proj``, order-symmetric sum) plus an on-path midpoint through its
own ``edge_mid_proj`` — straight edge -> chord midpoint (degenerate arc),
board-outline arc -> the on-arc midpoint (KiCad's native 3-point form; full
circle: p1 == p2, mid = antipode). Old checkpoints are 2-point; the loader
derives ``legacy_edge_encoding`` from the PRESENCE of ``edge_mid_proj`` in the
state_dict (the shared proj shapes are identical, so — unlike
``legacy_net_encoding`` — a weight shape cannot be the signal).

Covers: encoding invariances, legacy fail-fast on arc obs, checkpoint
detection + strict load both ways, walk-buffer semantics on both obs formats,
indexed round-trip of the ``edge_mid`` table, and D4-augmentation equivalence
for arc mids.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from methods.rl_agent.models.loader import _policy_args_for_checkpoint
from methods.rl_agent.models.v1.embedding import TokenVocabulary
from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from methods.rl_agent.wrappers.augmentation import build_aug_dict
from pcb_world.core.indexed_obs import arrays_to_dict, dict_to_arrays
from tests._mock_obs import make_mock_obs


def _vocab(**kw) -> TokenVocabulary:
    v = TokenVocabulary(d_model=32, n_freq=4, **kw)
    v.eval()
    return v


# ---------------------------------------------------------------------------
# TokenVocabulary.encode_edge
# ---------------------------------------------------------------------------

class TestEncodeEdgeThreePoint:
    def test_shape(self):
        v = _vocab()
        out = v.encode_edge(
            torch.randn(4, 2), torch.randn(4, 2), torch.rand(4, 1),
            xy_mid=torch.randn(4, 2),
        )
        assert out.shape == (4, v.d_model)

    def test_endpoint_swap_invariant_with_mid(self):
        """arc(p1, mid, p2) == arc(p2, mid, p1) — reversal is the same arc."""
        torch.manual_seed(0)
        v = _vocab()
        xy1 = torch.tensor([[0.0, 0.0]])
        xy2 = torch.tensor([[1.0, 0.0]])
        mid = torch.tensor([[0.5, 0.4]])
        w = torch.tensor([[0.1]])
        a = v.encode_edge(xy1, xy2, w, xy_mid=mid)
        b = v.encode_edge(xy2, xy1, w, xy_mid=mid)
        assert torch.allclose(a, b, atol=1e-6)

    def test_mid_role_not_interchangeable_with_endpoint(self):
        """Swapping mid with an endpoint must change the embedding — the
        bulge point is a different geometric role, not a third endpoint."""
        torch.manual_seed(0)
        v = _vocab()
        xy1 = torch.tensor([[0.0, 0.0]])
        xy2 = torch.tensor([[1.0, 0.0]])
        mid = torch.tensor([[0.5, 0.4]])
        w = torch.tensor([[0.1]])
        a = v.encode_edge(xy1, xy2, w, xy_mid=mid)
        b = v.encode_edge(xy1, mid, w, xy_mid=xy2)
        assert not torch.allclose(a, b, atol=1e-4)

    def test_legacy_has_no_mid_module_and_rejects_mid(self):
        lv = _vocab(legacy_edge_encoding=True)
        assert not hasattr(lv, "edge_mid_proj")
        with pytest.raises(RuntimeError, match="legacy"):
            lv.encode_edge(
                torch.randn(2, 2), torch.randn(2, 2), torch.rand(2, 1),
                xy_mid=torch.randn(2, 2),
            )
        # 2-point path still works.
        out = lv.encode_edge(torch.randn(2, 2), torch.randn(2, 2), torch.rand(2, 1))
        assert out.shape == (2, lv.d_model)

    def test_current_requires_mid(self):
        v = _vocab()
        with pytest.raises(RuntimeError, match="xy_mid"):
            v.encode_edge(torch.randn(2, 2), torch.randn(2, 2), torch.rand(2, 1))


# ---------------------------------------------------------------------------
# Checkpoint compat (mirrors tests/test_net_token_closed_compat.py)
# ---------------------------------------------------------------------------

_ARGS = {
    "d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64,
    "max_seq_len": 512, "n_freq": 4, "coord_encoding": "fourier",
    "mlp_hidden": 16, "use_critic": True,
}


def _model(**overrides):
    from methods.rl_agent.models.v1.net import KiCadRLModel

    kw = dict(_ARGS)
    kw.update(overrides)
    return KiCadRLModel(**kw)


class TestCheckpointPresenceDetection:
    def test_legacy_2pt_checkpoint_detected_and_loads(self):
        legacy_sd = _model(legacy_edge_encoding=True).state_dict()
        assert "tokenizer.vocab.edge_mid_proj.weight" not in legacy_sd
        compat = _policy_args_for_checkpoint(dict(_ARGS), legacy_sd)
        assert compat["legacy_edge_encoding"] is True

        from configs.loader.schema import RLPolicyConfig

        rebuilt = RLPolicyConfig.from_checkpoint(compat).build()
        rebuilt.load_state_dict(legacy_sd, strict=True)

    def test_current_3pt_checkpoint_detected_and_loads(self):
        current_sd = _model(legacy_edge_encoding=False).state_dict()
        assert "tokenizer.vocab.edge_mid_proj.weight" in current_sd
        compat = _policy_args_for_checkpoint(dict(_ARGS), current_sd)
        assert compat["legacy_edge_encoding"] is False

        from configs.loader.schema import RLPolicyConfig

        rebuilt = RLPolicyConfig.from_checkpoint(compat).build()
        rebuilt.load_state_dict(current_sd, strict=True)


# ---------------------------------------------------------------------------
# Tokenizer walk semantics (dict + indexed) and legacy fail-fast
# ---------------------------------------------------------------------------

def _tok(**kw) -> BatchedStateTokenizer:
    t = BatchedStateTokenizer(d_model=32, n_freq=4, **kw)
    t.eval()
    return t


class TestWalkMidBuffers:
    def test_dict_walk_mid_and_is_arc(self):
        obs = make_mock_obs(n_edges=4, n_arc_edges=2)
        tok = _tok()
        walk = tok._walk_obs([obs])
        xy1, xy2, w, obs_idx, pos, mid, is_arc = walk["edge"]
        assert len(xy1) == 6
        assert is_arc.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        # Straight edges: normalized mid == mean of normalized endpoints
        # (normalization is affine, so the chord midpoint commutes).
        np.testing.assert_allclose(
            mid[:4], (xy1[:4] + xy2[:4]) / 2.0, atol=1e-12,
        )
        # Arc entries: mid is off-chord.
        assert np.abs(mid[4:] - (xy1[4:] + xy2[4:]) / 2.0).max() > 1e-3

    def test_indexed_walk_matches_dict_walk(self):
        obs = make_mock_obs(n_edges=4, n_arc_edges=2)
        iobs = dict_to_arrays(obs)
        tok = _tok()
        wd = tok._walk_obs([obs])
        wi = tok._walk_obs([iobs])
        for f, (a, b) in enumerate(zip(wd["edge"], wi["edge"])):
            assert a.dtype == b.dtype and a.shape == b.shape, f
            np.testing.assert_array_equal(a, b, err_msg=f"edge field {f}")

    def test_forward_ok_with_arcs(self):
        obs = make_mock_obs(n_edges=4, n_arc_edges=2)
        out = _tok().forward([obs])
        assert out.token_embeddings.shape[0] == 1

    def test_legacy_tokenizer_rejects_arc_obs(self):
        obs = make_mock_obs(n_edges=4, n_arc_edges=1)
        tok = _tok(legacy_edge_encoding=True)
        with pytest.raises(RuntimeError, match="outline_obs"):
            tok.forward([obs])

    def test_legacy_tokenizer_ok_on_straight_obs(self):
        obs = make_mock_obs(n_edges=4, n_arc_edges=0)
        out = _tok(legacy_edge_encoding=True).forward([obs])
        assert out.token_embeddings.shape[0] == 1


class TestIndexedRoundTrip:
    def test_edge_mid_table_and_reconstruction(self):
        obs = make_mock_obs(n_edges=4, n_arc_edges=2)
        iobs = dict_to_arrays(obs)
        em = iobs["board_static"]["edge_mid"]
        assert em.shape == (6,) and em.dtype == np.int64
        assert (em[:4] == -1).all() and (em[4:] >= 0).all()

        back = arrays_to_dict(iobs)
        bl = back["board_static"]["boardlines"]
        for i in range(4):
            assert "mid" not in bl[f"edge_{i}"]
        for i in (4, 5):
            src = obs["board_static"]["boardlines"][f"edge_{i}"]
            assert bl[f"edge_{i}"]["mid"]["xy"] == src["mid"]["xy"]


# ---------------------------------------------------------------------------
# LLM serializers: arc entries render as <arc>/(arc ...) with the mid point
# ---------------------------------------------------------------------------

class TestLLMSerializerArc:
    def _bs(self):
        obs = make_mock_obs(n_edges=1, n_arc_edges=1)
        return obs["board_static"]

    def test_sexpr_arc_entry(self):
        from methods.llm_agent.wrappers.state_converter import (
            _board_static_to_sexpr,
        )

        text = _board_static_to_sexpr(self._bs(), 0)
        assert "(edge E0 (p1 " in text
        assert "(arc E1 (p1 " in text and "(mid " in text
        # mid sits between p1 and p2 inside the arc form.
        arc_line = next(l for l in text.splitlines() if "(arc E1" in l)
        assert arc_line.index("(p1 ") < arc_line.index("(mid ") < arc_line.index("(p2 ")

    def test_xml_arc_entry(self):
        from methods.llm_agent.wrappers.state_converter import (
            _board_static_to_xml,
        )

        text = _board_static_to_xml(self._bs(), 0)
        assert '<edge id="E0">' in text
        assert '<arc id="E1">' in text and "<mid id=" in text
        assert "</arc>" in text

    def test_straight_only_render_unchanged(self):
        """No arc entries -> byte-identical render to the pre-arc format."""
        from methods.llm_agent.wrappers.state_converter import (
            _board_static_to_sexpr,
            _board_static_to_xml,
        )

        obs = make_mock_obs(n_edges=4, n_arc_edges=0)
        for render in (_board_static_to_sexpr, _board_static_to_xml):
            text = render(obs["board_static"], 0)
            assert "arc" not in text and "mid" not in text


# ---------------------------------------------------------------------------
# D4 augmentation: transforming the aug dict == transforming the raw points
# ---------------------------------------------------------------------------

class TestArcMidUnderD4Aug:
    """Point transforms alone keep arcs exact — no angle bookkeeping.

    Tokenizing the base obs with a D4 ``_aug`` must equal tokenizing a
    manually transformed obs (all points, mid included, reflected/swapped
    about the bbox centre) with no aug. Square bbox so axis_swap needs no
    bbox_w/h adjustment.
    """

    BBOX = (100.0, 50.0, 60.0, 60.0)

    def _walk_edge(self, obs):
        return _tok()._walk_obs([obs])["edge"]

    def _transform_points(self, obs, fn):
        bl = obs["board_static"]["boardlines"]
        for e in bl.values():
            for key in ("p1", "p2", "mid"):
                if key in e:
                    x, y = e[key]["xy"]
                    e[key]["xy"] = list(fn(x, y))
        return obs

    @pytest.mark.parametrize(
        "axis_swap,flip_x,flip_y",
        [(False, -1, 1), (False, 1, -1), (True, 1, 1), (True, -1, -1)],
    )
    def test_aug_equals_manual_transform(self, axis_swap, flip_x, flip_y):
        bx, by, bw, bh = self.BBOX
        cx, cy = bx + bw / 2, by + bh / 2

        aug = build_aug_dict(
            bbox_shifted=False, scale_x=1.0, scale_y=1.0, cx=cx, cy=cy,
            axis_swap=axis_swap, flip_x=flip_x, flip_y=flip_y,
            nn_dx=0.0, nn_dy=0.0, nn_zoom=1.0, slot_perm=None,
            directional_candidates=None,
        )
        obs_aug = make_mock_obs(n_edges=4, n_arc_edges=2, bbox=self.BBOX)
        obs_aug["_aug"] = aug

        def fn(x, y):
            dx, dy = x - cx, y - cy
            if axis_swap:
                dx, dy = dy, dx
            return cx + flip_x * dx, cy + flip_y * dy

        obs_manual = self._transform_points(
            make_mock_obs(n_edges=4, n_arc_edges=2, bbox=self.BBOX), fn,
        )

        got = self._walk_edge(obs_aug)
        want = self._walk_edge(obs_manual)
        for f in (0, 1, 5):  # xy1, xy2, mid
            np.testing.assert_allclose(
                got[f], want[f], atol=1e-12,
                err_msg=f"edge field {f} under swap={axis_swap} "
                        f"fx={flip_x} fy={flip_y}",
            )
        np.testing.assert_array_equal(got[6], want[6])  # is_arc
