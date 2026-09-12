"""Verify that the two paths producing the same "virtually augmented" board
emit numerically equivalent tokenizer features:

  path A: load a pre-transformed fixture (.kicad_pcb with edges already
          scaled around some c_aug) + aug=None
  path B: load the BASE fixture + training aug dict with the same
          (scale_x, scale_y, aug_cx, aug_cy)

This is the consistency property the new aug scheme is designed to have.
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_world.core.env import PCBWorld
from tests.helpers.reference_tokenizer import StateTokenizer
from methods.rl_agent.models.v1.encoding import (
    _compute_norm_ctx,
    _norm_pos,
    _norm_pos_edge,
)


FIX = Path(__file__).resolve().parent / "fixtures"
BASE = FIX / "simple_routing_board.kicad_pcb"


def _write_shifted(sx: float, sy: float, cxa: float, cya: float) -> Path:
    """Produce a fresh .kicad_pcb where the gr_rect's start/end corners are
    scaled by diag(sx, sy) around (cxa, cya). Pads untouched."""
    text = BASE.read_text()
    xmin, ymin, xmax, ymax = 0.0, 0.0, 50.0, 30.0
    nxmin = cxa + sx * (xmin - cxa)
    nxmax = cxa + sx * (xmax - cxa)
    nymin = cya + sy * (ymin - cya)
    nymax = cya + sy * (ymax - cya)
    if nxmin > nxmax:
        nxmin, nxmax = nxmax, nxmin
    if nymin > nymax:
        nymin, nymax = nymax, nymin

    text = re.sub(
        r"(\(gr_rect\s*\(start )(-?\d+\.?\d*)\s+(-?\d+\.?\d*)(\))",
        f"\\g<1>{nxmin:.4f} {nymin:.4f}\\g<4>",
        text, count=1,
    )
    text = re.sub(
        r"(\(end )(-?\d+\.?\d*)\s+(-?\d+\.?\d*)(\))",
        f"\\g<1>{nxmax:.4f} {nymax:.4f}\\g<4>",
        text, count=1,
    )

    out = FIX / f"_equiv_shifted_{uuid.uuid4().hex[:8]}.kicad_pcb"
    out.write_text(text)
    # Engine load contract: a board must carry its .kicad_pro sibling. The
    # shift only rewrites geometry, so the base board's rules apply as-is.
    out.with_suffix(".kicad_pro").write_bytes(
        BASE.with_suffix(".kicad_pro").read_bytes()
    )
    return out


@pytest.mark.parametrize("sx,sy,cxa,cya", [
    (1.10, 0.98, 42.9, 20.9),    # similar to generated shifted
    (0.84, 1.19, 38.1, 23.6),    # similar to generated shifted_shuffled
    (1.00, 1.00, 25.0, 15.0),    # identity (sanity)
    (0.90, 1.10, 20.0, 18.0),
])
def test_norm_ctx_equivalence(sx, sy, cxa, cya, tmp_path):
    """At the `_compute_norm_ctx` + `_norm_pos[_edge]` level, the two paths
    must yield the same normalized coordinates for both pads and edges."""
    shifted_path = _write_shifted(sx, sy, cxa, cya)
    try:
        # Path A: loaded fixture + aug=None
        e_a = PCBWorld(str(shifted_path), max_steps=1)
        obs_a, _ = e_a.reset()
        e_a.close()

        # Path B: base + aug={scale_x,...}
        e_b = PCBWorld(str(BASE), max_steps=1)
        obs_b, _ = e_b.reset()
        e_b.close()
        obs_b["_aug"] = {
            "scale_x": sx, "scale_y": sy,
            "aug_cx": cxa, "aug_cy": cya,
        }

        ctx_a = _compute_norm_ctx(obs_a["board_static"], None)
        ctx_b = _compute_norm_ctx(obs_b["board_static"], obs_b["_aug"])

        # Same effective norm_scale / cx / cy (path A's env bbox ≈ path B's
        # virtual bbox; KiCad adds a tiny stroke margin so tolerance ~0.1mm —
        # path A scales the rect then adds the margin, path B scales the
        # margin-inflated bbox, so norm_scale differs by margin·(s-1)/2).
        assert abs(ctx_a.norm_scale - ctx_b.norm_scale) < 0.05
        assert abs(ctx_a.cx - ctx_b.cx) < 0.15, f"cx: {ctx_a.cx} vs {ctx_b.cx}"
        assert abs(ctx_a.cy - ctx_b.cy) < 0.15, f"cy: {ctx_a.cy} vs {ctx_b.cy}"

        # Pad equivalence: pads are at identical physical positions in both,
        # so path A _norm_pos(pad, ctx_a) ≈ path B _norm_pos(pad, ctx_b).
        pad_positions_a = [
            p["center"]["xy"]
            for net in obs_a["board_static"]["nets"].values()
            for p in net.get("pads", {}).values()
        ]
        for px, py in pad_positions_a:
            a = _norm_pos(px, py, ctx_a)
            b = _norm_pos(px, py, ctx_b)
            assert abs(a[0] - b[0]) < 1e-3, f"pad x @ ({px},{py}): {a[0]} vs {b[0]}"
            assert abs(a[1] - b[1]) < 1e-3, f"pad y @ ({px},{py}): {a[1]} vs {b[1]}"

        # Edge equivalence: path A has edges at scaled physical positions,
        # path B has edges at original positions but _norm_pos_edge applies
        # the virtual transform internally. Normalized values should match.
        edges_a = obs_a["board_static"]["boardlines"]
        edges_b = obs_b["board_static"]["boardlines"]
        assert len(edges_a) == len(edges_b) == 4
        # Map by (rounded normalized) to align edges across the two obs,
        # since the env may order them differently. Compute both sides and
        # compare the SET of points.
        pts_a = sorted(
            round(v, 3)
            for e in edges_a.values()
            for pt in (e["p1"]["xy"], e["p2"]["xy"])
            for v in _norm_pos_edge(pt[0], pt[1], ctx_a)
        )
        pts_b = sorted(
            round(v, 3)
            for e in edges_b.values()
            for pt in (e["p1"]["xy"], e["p2"]["xy"])
            for v in _norm_pos_edge(pt[0], pt[1], ctx_b)
        )
        for a, b in zip(pts_a, pts_b):
            assert abs(a - b) < 5e-3, f"edge pt: {a} vs {b} (all_a={pts_a}, all_b={pts_b})"
    finally:
        shifted_path.unlink(missing_ok=True)
        shifted_path.with_suffix(".kicad_pro").unlink(missing_ok=True)


@pytest.mark.parametrize("sx,sy,cxa,cya", [
    (1.10, 0.98, 42.9, 20.9),
    (0.84, 1.19, 38.1, 23.6),
])
def test_tokenizer_forward_equivalence(sx, sy, cxa, cya):
    """Full StateTokenizer.forward equivalence: the token_embeddings from
    path A and path B must be numerically close."""
    shifted_path = _write_shifted(sx, sy, cxa, cya)
    try:
        torch.manual_seed(0)
        tok = StateTokenizer(d_model=64, n_freq=16).eval()

        e_a = PCBWorld(str(shifted_path), max_steps=1)
        obs_a, _ = e_a.reset(); e_a.close()
        # Strip any _aug key path A shouldn't have.
        obs_a.pop("_aug", None)

        e_b = PCBWorld(str(BASE), max_steps=1)
        obs_b, _ = e_b.reset(); e_b.close()
        obs_b["_aug"] = {
            "scale_x": sx, "scale_y": sy,
            "aug_cx": cxa, "aug_cy": cya,
        }

        with torch.no_grad():
            out_a = tok([obs_a])
            out_b = tok([obs_b])

        assert out_a.token_embeddings.shape == out_b.token_embeddings.shape, (
            f"shape mismatch: {out_a.token_embeddings.shape} vs {out_b.token_embeddings.shape}"
        )

        valid = ~out_a.key_padding_mask[0]  # (seq,) bool
        ea = out_a.token_embeddings[0][valid]
        eb = out_b.token_embeddings[0][valid]
        # Drop zero-magnitude tokens (e.g. PREV_ACTION positions emitted
        # without action_type_weight): cosine_similarity of zero vectors
        # returns 0 and skews the mean even though the tokens are
        # identical. We assert exact equality on those separately.
        nz = (ea.norm(dim=-1) > 0) & (eb.norm(dim=-1) > 0)
        assert torch.equal(ea[~nz], eb[~nz]), "zero-magnitude tokens differ"
        ea = ea[nz]
        eb = eb[nz]
        # Embedding equivalence after Fourier + MLP + LayerNorm: tiny
        # coord noise from KiCad's ~0.075 mm edge-stroke margin amplifies
        # through high-freq fourier features. Coordinate-level equivalence
        # is precisely checked by test_norm_ctx_equivalence; here we only
        # assert the aggregate features stay closely aligned.
        cos = torch.nn.functional.cosine_similarity(ea, eb, dim=-1)
        mean_cos = cos.mean().item()
        min_cos = cos.min().item()
        assert mean_cos > 0.995, (
            f"mean per-token cosine similarity: {mean_cos:.4f} "
            f"(min={min_cos:.4f}, shape={ea.shape})"
        )
        assert min_cos > 0.95, (
            f"min per-token cosine similarity: {min_cos:.4f} "
            f"(mean={mean_cos:.4f})"
        )
    finally:
        shifted_path.unlink(missing_ok=True)
        shifted_path.with_suffix(".kicad_pro").unlink(missing_ok=True)
