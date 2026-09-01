"""aug_scale applied to potential-based reward in MLP and Decoder wrappers.

Verifies the refactor in which ``aug_scale`` multiplies Φ across every
step (not just the terminal step), while ``step_penalty`` is left
untouched and coordinate augmentation is routed through the existing
normalization paths without leaking aug_scale into the observation.

All tests drive the real ``PCBWorld`` on a fixture board — no mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pcb_world.engine.kicad_engine import allow_router_coexistence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))
sys.path.insert(0, str(PROJECT_ROOT))

BOARD = str(PROJECT_ROOT / "tests" / "fixtures" / "simple_routing_board.kicad_pcb")


# ---------------------------------------------------------------------------
# MLP wrapper helpers (reuse the test_reward_modes.py action sequence)
# ---------------------------------------------------------------------------
def _find_candidate(env, x, y, layer=None, tol=0.001):
    for i, (cx, cy, cl) in enumerate(env._candidates_mm):
        if env._candidate_mask[i] and abs(cx - x) < tol and abs(cy - y) < tol:
            if layer is None or cl == layer:
                return i
    return None


def _route_all_nets(env):
    steps = []

    def do(action):
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        steps.append((reward, terminated, truncated, info))
        return obs, reward, terminated, truncated, info

    do([0, 0])
    do([1, 0])
    do([3, 1])
    do([2, 0])

    do([0, 0])
    do([1, 0])
    via_pt = _find_candidate(env, 25.0, 5.5)
    assert via_pt is not None
    do([4, via_pt])
    bottom_pt = _find_candidate(env, 25.0, 5.5, layer=2)
    assert bottom_pt is not None
    do([1, bottom_pt])
    target_pt = _find_candidate(env, 25.0, 25.0)
    assert target_pt is not None
    do([4, target_pt])
    do([2, 0])

    do([0, 0])
    do([1, 0])
    far_pad = _find_candidate(env, 40.0, 20.0)
    assert far_pad is not None
    do([3, far_pad])
    return steps


# ---------------------------------------------------------------------------
# MLP — reward scaling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Decoder wrapper helpers
# ---------------------------------------------------------------------------
def _make_decoder_env_orthogonal(
    *,
    aug_flip: bool = False,
    aug_trans: bool = False,
    aug_rotate: bool = False,
    aug_bbox_shifted: bool = False,
    aug_zoom: bool = False,
):
    """Decoder wrapper exposing the 5-boolean aug interface."""
    from pcb_world.core.env import PCBWorld
    from methods.rl_agent.wrappers.adapter import KiCadRLWrapper

    if not Path(BOARD).exists():
        pytest.skip(f"Board fixture missing: {BOARD}")

    env = PCBWorld(
        board_path=BOARD,
        max_steps=200,
        masking_rule="strict_no_finish",
        reward_rule="shaped",
    )
    return KiCadRLWrapper(
        env, seed=42,
        aug_bbox_shifted=aug_bbox_shifted,
        aug_flip=aug_flip,
        aug_rotate=aug_rotate,
        aug_trans=aug_trans,
        aug_zoom=aug_zoom,
    )


class TestNormPostTransform:
    """Direct tests of _norm_pos / _norm_pos_edge with orthogonal axes.
    These run without the env to pin down the transform semantics."""

    def _ctx(self, **aug_kwargs):
        from methods.rl_agent.models.v1.encoding import _compute_norm_ctx
        bs = {
            "bbox_x": 0.0, "bbox_y": 0.0,
            "bbox_w": 50.0, "bbox_h": 30.0,
            "copper_layers": 2,
        }
        return _compute_norm_ctx(bs, aug_kwargs or None)

    def test_identity_defaults(self):
        from methods.rl_agent.models.v1.encoding import _norm_pos
        ctx = self._ctx()
        # Center of board (25, 15) should map to (0, 0) with no aug.
        nx, ny = _norm_pos(25.0, 15.0, ctx)
        assert nx == pytest.approx(0.0)
        assert ny == pytest.approx(0.0)
        assert ctx.flip_x == 1 and ctx.flip_y == 1
        assert ctx.nn_dx == 0.0 and ctx.nn_dy == 0.0

    def test_sign_reflection_flips_coords(self):
        from methods.rl_agent.models.v1.encoding import _norm_pos
        ctx_base = self._ctx()
        ctx_flip = self._ctx(flip_x=-1, flip_y=-1)
        # Off-center point: (35, 20) → nx,ny ≈ (0.4, 0.2)
        nx0, ny0 = _norm_pos(35.0, 20.0, ctx_base)
        nx1, ny1 = _norm_pos(35.0, 20.0, ctx_flip)
        assert nx1 == pytest.approx(-nx0)
        assert ny1 == pytest.approx(-ny0)

    def test_sign_reflection_per_axis(self):
        from methods.rl_agent.models.v1.encoding import _norm_pos
        ctx_x = self._ctx(flip_x=-1, flip_y=1)
        ctx_y = self._ctx(flip_x=1, flip_y=-1)
        nx_x, ny_x = _norm_pos(35.0, 20.0, ctx_x)
        nx_y, ny_y = _norm_pos(35.0, 20.0, ctx_y)
        # x-flip inverts x only; y-flip inverts y only.
        nx_base, ny_base = _norm_pos(35.0, 20.0, self._ctx())
        assert nx_x == pytest.approx(-nx_base) and ny_x == pytest.approx(ny_base)
        assert nx_y == pytest.approx(nx_base) and ny_y == pytest.approx(-ny_base)

    def test_nn_input_trans_adds_offset(self):
        from methods.rl_agent.models.v1.encoding import _norm_pos
        ctx_base = self._ctx()
        ctx_tr = self._ctx(nn_dx=0.15, nn_dy=-0.07)
        nx0, ny0 = _norm_pos(35.0, 20.0, ctx_base)
        nx1, ny1 = _norm_pos(35.0, 20.0, ctx_tr)
        assert nx1 == pytest.approx(nx0 + 0.15)
        assert ny1 == pytest.approx(ny0 - 0.07)

    def test_flip_then_trans_order(self):
        """Spec: nx' = flip_x * nx + nn_dx (flip first, then translate)."""
        from methods.rl_agent.models.v1.encoding import _norm_pos
        ctx = self._ctx(flip_x=-1, flip_y=1, nn_dx=0.1, nn_dy=0.05)
        ctx_base = self._ctx()
        nx0, ny0 = _norm_pos(35.0, 20.0, ctx_base)
        nx1, ny1 = _norm_pos(35.0, 20.0, ctx)
        assert nx1 == pytest.approx(-nx0 + 0.1)
        assert ny1 == pytest.approx(ny0 + 0.05)

    def test_edge_point_follows_same_post_transform(self):
        """_norm_pos_edge must apply the same post-transform as _norm_pos
        so the whole scene is translated/flipped uniformly."""
        from methods.rl_agent.models.v1.encoding import (
            _norm_pos, _norm_pos_edge,
        )
        ctx = self._ctx(flip_x=-1, flip_y=-1, nn_dx=0.2, nn_dy=-0.1)
        # Edge point at a board corner. In scheme="none", _norm_pos_edge
        # reduces to _norm_pos so they must agree.
        e_nx, e_ny = _norm_pos_edge(50.0, 30.0, ctx)
        p_nx, p_ny = _norm_pos(50.0, 30.0, ctx)
        assert e_nx == pytest.approx(p_nx)
        assert e_ny == pytest.approx(p_ny)

class TestSignReflect:
    """Wrapper-level sign_reflection injection + reward invariance."""

    def test_prob_zero_is_no_op(self):
        """aug_flip=False leaves flip_x=flip_y=+1, obs identical to
        the off-aug baseline after reset()."""
        with allow_router_coexistence("side-by-side aug comparison: 2 envs"):
            env_base = _make_decoder_env_orthogonal()
            env_flip = _make_decoder_env_orthogonal(aug_flip=False)
        obs_b, _ = env_base.reset()
        obs_f, _ = env_flip.reset()
        env_base.close(); env_flip.close()
        assert obs_b["_aug"]["flip_x"] == 1 and obs_b["_aug"]["flip_y"] == 1
        assert obs_f["_aug"]["flip_x"] == 1 and obs_f["_aug"]["flip_y"] == 1

    def test_inject_aug_contains_flip_keys(self):
        """Regardless of flag, the _aug dict always exposes flip_x/flip_y
        so downstream code can rely on presence."""
        env = _make_decoder_env_orthogonal(aug_flip=True)
        obs, _ = env.reset()
        env.close()
        assert "flip_x" in obs["_aug"] and "flip_y" in obs["_aug"]
        assert obs["_aug"]["flip_x"] in (+1, -1)
        assert obs["_aug"]["flip_y"] in (+1, -1)

    def test_set_augmentation_external_flip(self):
        """set_augmentation can override flip_x/flip_y for GRPO parity."""
        env = _make_decoder_env_orthogonal(aug_flip=True)
        env.set_augmentation(flip_x=-1, flip_y=-1)
        obs, _ = env.reset()
        env.close()
        assert obs["_aug"]["flip_x"] == -1
        assert obs["_aug"]["flip_y"] == -1

    def test_reward_invariant_under_flip(self):
        """Physical mm geometry is unchanged → env-level reward sequence
        must match between flip on/off for the same action sequence."""
        with allow_router_coexistence("side-by-side aug comparison: 2 envs"):
            env_base = _make_decoder_env_orthogonal()
            env_flip = _make_decoder_env_orthogonal()
        env_base.reset()
        env_flip.reset()
        env_flip.set_augmentation(flip_x=-1, flip_y=-1)

        actions = [
            np.array([0, 0, -1]),   # net_select
            np.array([1, 0, -1]),   # start_route
            np.array([2, -1, -1]),  # net_end
        ]
        for a in actions:
            _, r_b, *_ = env_base.step(a)
            _, r_f, *_ = env_flip.step(a)
            assert r_b == pytest.approx(r_f, abs=1e-9)
        env_base.close(); env_flip.close()

    def test_tokenizer_embedding_changes_under_flip(self):
        """Same obs with different flip values must produce different
        token embeddings (evidence the transform is actually applied)."""
        from tests.helpers.reference_tokenizer import StateTokenizer
        import torch

        env = _make_decoder_env_orthogonal()
        obs, _ = env.reset()
        env.close()

        tok = StateTokenizer(d_model=32, n_freq=4,
                             coord_encoding="fourier", mlp_hidden=8)
        tok.eval()
        device = torch.device("cpu")

        obs_a = dict(obs)
        obs_a["_aug"] = {"flip_x": 1, "flip_y": 1, "nn_dx": 0.0, "nn_dy": 0.0}
        out_a = tok._tokenize_single(obs_a, device)

        obs_b = dict(obs)
        obs_b["_aug"] = {"flip_x": -1, "flip_y": -1, "nn_dx": 0.0, "nn_dy": 0.0}
        out_b = tok._tokenize_single(obs_b, device)

        assert out_a.embeddings.shape == out_b.embeddings.shape
        assert not torch.allclose(out_a.embeddings, out_b.embeddings)


class TestNNInputTrans:
    """Wrapper-level nn_input_trans injection + reward invariance."""

    def test_range_zero_is_no_op(self):
        env = _make_decoder_env_orthogonal(aug_trans=False)
        obs, _ = env.reset()
        env.close()
        assert obs["_aug"]["nn_dx"] == pytest.approx(0.0)
        assert obs["_aug"]["nn_dy"] == pytest.approx(0.0)

    def test_samples_within_range(self):
        from methods.rl_agent.wrappers.adapter import KiCadRLWrapper
        r = KiCadRLWrapper._AUG_TRANS_RANGE
        env = _make_decoder_env_orthogonal(aug_trans=True)
        # Sample many resets and check all (dx, dy) fall in [-r, r].
        for _ in range(30):
            obs, _ = env.reset()
            assert -r <= obs["_aug"]["nn_dx"] <= r
            assert -r <= obs["_aug"]["nn_dy"] <= r
        env.close()

    def test_set_augmentation_external_trans(self):
        env = _make_decoder_env_orthogonal(aug_trans=True)
        env.set_augmentation(nn_dx=0.13, nn_dy=-0.07)
        obs, _ = env.reset()
        env.close()
        assert obs["_aug"]["nn_dx"] == pytest.approx(0.13)
        assert obs["_aug"]["nn_dy"] == pytest.approx(-0.07)

    def test_reward_invariant_under_trans(self):
        with allow_router_coexistence("side-by-side aug comparison: 2 envs"):
            env_base = _make_decoder_env_orthogonal()
            env_tr = _make_decoder_env_orthogonal()
        env_base.reset()
        env_tr.reset()
        env_tr.set_augmentation(nn_dx=0.15, nn_dy=-0.1)

        actions = [
            np.array([0, 0, -1]),
            np.array([1, 0, -1]),
            np.array([2, -1, -1]),
        ]
        for a in actions:
            _, r_b, *_ = env_base.step(a)
            _, r_t, *_ = env_tr.step(a)
            assert r_b == pytest.approx(r_t, abs=1e-9)
        env_base.close(); env_tr.close()


class TestNNZoom:
    """Wrapper-level nn_zoom (feature-space uniform scale) injection +
    exact-normalization semantics + reward invariance."""

    def test_exact_norm_scale_no_aug(self):
        """norm_scale is the exact bbox half-extent (no 1/2/5-series
        quantization): a 50x30 board maps its long axis to exactly
        [-1, 1] and its short axis to [-0.6, 0.6]."""
        from methods.rl_agent.models.v1.encoding import (
            _compute_norm_ctx, _norm_pos,
        )
        bs = {"bbox_x": 0.0, "bbox_y": 0.0,
              "bbox_w": 50.0, "bbox_h": 30.0, "copper_layers": 2}
        ctx = _compute_norm_ctx(bs, None)
        assert ctx.norm_scale == pytest.approx(25.0)
        assert _norm_pos(50.0, 30.0, ctx) == pytest.approx((1.0, 0.6))
        assert _norm_pos(0.0, 0.0, ctx) == pytest.approx((-1.0, -0.6))

    def test_zoom_scales_positions_and_dims_uniformly(self):
        """nn_zoom multiplies every normalized position AND every
        normalized dim by the same factor — a consistent scene zoom."""
        from methods.rl_agent.models.v1.encoding import (
            _compute_norm_ctx, _norm_dim, _norm_pos,
        )
        bs = {"bbox_x": 0.0, "bbox_y": 0.0,
              "bbox_w": 50.0, "bbox_h": 30.0, "copper_layers": 2}
        ctx_id = _compute_norm_ctx(bs, None)
        ctx_zm = _compute_norm_ctx(bs, {"nn_zoom": 1.1})
        nx0, ny0 = _norm_pos(35.0, 20.0, ctx_id)
        nx1, ny1 = _norm_pos(35.0, 20.0, ctx_zm)
        assert nx1 == pytest.approx(1.1 * nx0)
        assert ny1 == pytest.approx(1.1 * ny0)
        assert _norm_dim(0.25, ctx_zm) == pytest.approx(
            1.1 * _norm_dim(0.25, ctx_id))

    def test_zoom_composes_with_trans_after_scale(self):
        """nn_dx/nn_dy are normalized-frame offsets applied AFTER the
        zoomed division: nx' = zoom * nx + nn_dx."""
        from methods.rl_agent.models.v1.encoding import (
            _compute_norm_ctx, _norm_pos,
        )
        bs = {"bbox_x": 0.0, "bbox_y": 0.0,
              "bbox_w": 50.0, "bbox_h": 30.0, "copper_layers": 2}
        nx0, ny0 = _norm_pos(35.0, 20.0, _compute_norm_ctx(bs, None))
        ctx = _compute_norm_ctx(bs, {"nn_zoom": 0.9, "nn_dx": 0.1,
                                     "nn_dy": -0.05})
        nx, ny = _norm_pos(35.0, 20.0, ctx)
        assert nx == pytest.approx(0.9 * nx0 + 0.1)
        assert ny == pytest.approx(0.9 * ny0 - 0.05)

    def test_flag_off_is_identity(self):
        env = _make_decoder_env_orthogonal(aug_zoom=False)
        obs, _ = env.reset()
        env.close()
        assert obs["_aug"]["nn_zoom"] == pytest.approx(1.0)

    def test_samples_within_range(self):
        from methods.rl_agent.wrappers.adapter import KiCadRLWrapper
        r = KiCadRLWrapper._AUG_ZOOM_RANGE
        env = _make_decoder_env_orthogonal(aug_zoom=True)
        seen = []
        for _ in range(30):
            obs, _ = env.reset()
            z = obs["_aug"]["nn_zoom"]
            assert 1.0 - r <= z <= 1.0 + r
            seen.append(z)
        env.close()
        assert max(seen) - min(seen) > 0.0, "zoom never resampled"

    def test_set_augmentation_external_zoom(self):
        env = _make_decoder_env_orthogonal(aug_zoom=True)
        env.set_augmentation(nn_zoom=1.07)
        obs, _ = env.reset()
        env.close()
        assert obs["_aug"]["nn_zoom"] == pytest.approx(1.07)

    def test_reward_invariant_under_zoom(self):
        """Obs-only transform: physical mm geometry unchanged → reward
        sequence identical for the same action sequence."""
        with allow_router_coexistence("side-by-side aug comparison: 2 envs"):
            env_base = _make_decoder_env_orthogonal()
            env_zm = _make_decoder_env_orthogonal()
        env_base.reset()
        env_zm.reset()
        env_zm.set_augmentation(nn_zoom=1.1)

        actions = [
            np.array([0, 0, -1]),
            np.array([1, 0, -1]),
            np.array([2, -1, -1]),
        ]
        for a in actions:
            _, r_b, *_ = env_base.step(a)
            _, r_z, *_ = env_zm.step(a)
            assert r_b == pytest.approx(r_z, abs=1e-9)
        env_base.close(); env_zm.close()


class TestAxisSwap:
    """Wrapper-level axis_swap injection + coord/feature-pair swap."""

    def test_prob_zero_is_no_op(self):
        env = _make_decoder_env_orthogonal()
        obs, _ = env.reset()
        env.close()
        assert obs["_aug"]["axis_swap"] is False

    def test_inject_aug_contains_axis_swap_key(self):
        env = _make_decoder_env_orthogonal(aug_rotate=True)
        obs, _ = env.reset()
        env.close()
        assert "axis_swap" in obs["_aug"]
        assert isinstance(obs["_aug"]["axis_swap"], bool)

    def test_set_augmentation_external_swap(self):
        env = _make_decoder_env_orthogonal(aug_rotate=True)
        env.set_augmentation(axis_swap=True)
        obs, _ = env.reset()
        env.close()
        assert obs["_aug"]["axis_swap"] is True

    def test_norm_pos_swaps_coords(self):
        """Direct _norm_pos check: axis_swap swaps nx and ny."""
        from methods.rl_agent.models.v1.encoding import (
            _compute_norm_ctx, _norm_pos,
        )
        bs = {"bbox_x": 0.0, "bbox_y": 0.0,
              "bbox_w": 50.0, "bbox_h": 30.0, "copper_layers": 2}
        ctx_id = _compute_norm_ctx(bs, None)
        ctx_sw = _compute_norm_ctx(bs, {"axis_swap": True})
        # Off-center point
        nx0, ny0 = _norm_pos(35.0, 20.0, ctx_id)
        nxs, nys = _norm_pos(35.0, 20.0, ctx_sw)
        # Under swap: (nx0, ny0) -> (ny0, nx0).
        assert nxs == pytest.approx(ny0)
        assert nys == pytest.approx(nx0)

    def test_axis_swap_order_with_flip_and_trans(self):
        """Spec: axis_swap runs BEFORE sign_reflection and nn_input_trans.
        Expected: nx' = flip_x * (swap?[ny]:nx) + nn_dx.
        """
        from methods.rl_agent.models.v1.encoding import (
            _compute_norm_ctx, _norm_pos,
        )
        bs = {"bbox_x": 0.0, "bbox_y": 0.0,
              "bbox_w": 50.0, "bbox_h": 30.0, "copper_layers": 2}
        ctx_base = _compute_norm_ctx(bs, None)
        nx0, ny0 = _norm_pos(35.0, 20.0, ctx_base)

        ctx = _compute_norm_ctx(bs, {
            "axis_swap": True, "flip_x": -1, "flip_y": 1,
            "nn_dx": 0.1, "nn_dy": -0.05,
        })
        nx, ny = _norm_pos(35.0, 20.0, ctx)
        # After swap: (ny0, nx0). After flip_x: (-ny0, nx0). After trans:
        # (-ny0 + 0.1, nx0 - 0.05).
        assert nx == pytest.approx(-ny0 + 0.1, abs=1e-9)
        assert ny == pytest.approx(nx0 - 0.05, abs=1e-9)

    def test_maybe_swap_pair_swaps_scalars(self):
        from methods.rl_agent.models.v1.encoding import (
            _compute_norm_ctx, _maybe_swap_pair,
        )
        bs = {"bbox_x": 0.0, "bbox_y": 0.0,
              "bbox_w": 50.0, "bbox_h": 30.0, "copper_layers": 2}
        ctx_id = _compute_norm_ctx(bs, None)
        ctx_sw = _compute_norm_ctx(bs, {"axis_swap": True})
        assert _maybe_swap_pair(5.0, 3.0, ctx_id) == (5.0, 3.0)
        assert _maybe_swap_pair(5.0, 3.0, ctx_sw) == (3.0, 5.0)

    def test_tokenizer_bbox_and_pad_features_swap_together(self):
        """Under axis_swap, a rectangular board with bbox_w != bbox_h must
        produce a BOARD token whose [w, h] feature slot is swapped. Same
        for pad width/height. Verify by comparing normal vs swapped
        tokenizer outputs for the same env observation.
        """
        from tests.helpers.reference_tokenizer import StateTokenizer
        import torch

        env = _make_decoder_env_orthogonal()
        obs, _ = env.reset()
        env.close()

        bs = obs["board_static"]
        # Fixture must be non-square for this test to be meaningful.
        assert abs(bs["bbox_w"] - bs["bbox_h"]) > 0.1, (
            "test fixture bbox must be non-square to exercise wh swap"
        )

        tok = StateTokenizer(d_model=32, n_freq=4,
                             coord_encoding="fourier", mlp_hidden=8)
        tok.eval()

        obs_id = dict(obs); obs_id["_aug"] = {"axis_swap": False}
        obs_sw = dict(obs); obs_sw["_aug"] = {"axis_swap": True}
        out_id = tok._tokenize_single(obs_id, torch.device("cpu"))
        out_sw = tok._tokenize_single(obs_sw, torch.device("cpu"))

        # Shape unchanged; embeddings differ (scene geometry rotated).
        assert out_id.embeddings.shape == out_sw.embeddings.shape
        assert not torch.allclose(out_id.embeddings, out_sw.embeddings)

    def test_reward_invariant_under_axis_swap(self):
        """Physical mm geometry unchanged → reward sequence identical."""
        with allow_router_coexistence("side-by-side aug comparison: 2 envs"):
            env_base = _make_decoder_env_orthogonal()
            env_sw = _make_decoder_env_orthogonal(aug_rotate=True)
        env_base.reset(); env_sw.reset()
        env_sw.set_augmentation(axis_swap=True)
        actions = [np.array([0, 0, -1]), np.array([1, 0, -1]),
                   np.array([2, -1, -1])]
        for a in actions:
            _, rb, *_ = env_base.step(a)
            _, rs, *_ = env_sw.step(a)
            assert rb == pytest.approx(rs, abs=1e-9)
        env_base.close(); env_sw.close()


class TestOrthogonalAxesCombine:
    """All five aug axes compose and produce finite embeddings (smoke)."""

    def test_all_axes_combined(self):
        from pcb_world.core.env import PCBWorld
        from methods.rl_agent.wrappers.adapter import KiCadRLWrapper
        from tests.helpers.reference_tokenizer import StateTokenizer
        import torch

        if not Path(BOARD).exists():
            pytest.skip(f"Board fixture missing: {BOARD}")

        env = KiCadRLWrapper(
            PCBWorld(board_path=BOARD, max_steps=200,
                       masking_rule="strict_no_finish",
                       reward_rule="shaped"),
            seed=7,
            aug_bbox_shifted=True,
            aug_flip=True,
            aug_trans=True,
            aug_rotate=True,
            aug_zoom=True,
        )
        obs, _ = env.reset()
        env.close()

        # All orthogonal keys present alongside new-scheme keys.
        for k in ("scale_x", "scale_y", "aug_cx", "aug_cy",
                  "axis_swap", "flip_x", "flip_y", "nn_dx", "nn_dy",
                  "nn_zoom"):
            assert k in obs["_aug"], f"missing key {k}"

        tok = StateTokenizer(d_model=32, n_freq=4,
                             coord_encoding="fourier", mlp_hidden=8)
        tok.eval()
        out = tok._tokenize_single(obs, torch.device("cpu"))
        assert torch.isfinite(out.embeddings).all()
