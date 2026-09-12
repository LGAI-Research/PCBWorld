"""Augmentation-compat checks for the DRC state token.

DRC coordinates flow through ``_norm_pos``, so the existing orthogonal
augmentation axes (axis_swap, sign_reflection, nn_input_trans) and the
bbox-scaling scheme apply automatically. These tests lock that in.

Kept separate from ``test_aug_equivalence.py`` to avoid depending on the
PCBWorld import (which pulls a Python 3.10+ union syntax file).
"""

from __future__ import annotations

import torch

from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from tests._mock_obs import make_mock_obs


def _drc_pos(tokenizer, obs) -> int:
    walk = tokenizer._walk_obs([obs])
    d_pos = walk["drc"][7]  # drc_pos list
    assert d_pos, "observation has no DRC tokens"
    return int(d_pos[0])


def _drc(x, y, layer=1, severity=0x20, type_id=0, nets=("NET1",)):
    return {
        "x_mm": x, "y_mm": y, "layer": layer,
        "error_type": "Clearance violation", "type_id": type_id,
        "severity": severity, "net_names": list(nets),
    }


def test_axis_swap_equivalent_to_swapping_mm_and_bbox():
    """axis_swap at the tokenizer boundary is equivalent to physically
    swapping (bbox_x<->bbox_y, bbox_w<->bbox_h, x_mm<->y_mm) in the obs
    (with no axis_swap flag). This verifies the DRC xy flows through
    _norm_pos identically to every other geometric token.
    """
    torch.manual_seed(0)
    tok = BatchedStateTokenizer(d_model=64, n_freq=8).eval()

    obs_aug = make_mock_obs(
        n_nets=2, pads_per_net=1, n_ratsnest_per_net=0, n_edges=0,
        is_routing=False, current_net_phase=0,
        bbox=(0.0, 0.0, 60.0, 60.0),  # symmetric bbox so center maps
        drc_violations=[_drc(x=45.0, y=15.0)],
    )
    obs_aug["_aug"] = {"axis_swap": True}

    obs_phys = make_mock_obs(
        n_nets=2, pads_per_net=1, n_ratsnest_per_net=0, n_edges=0,
        is_routing=False, current_net_phase=0,
        bbox=(0.0, 0.0, 60.0, 60.0),
        drc_violations=[_drc(x=15.0, y=45.0)],
    )
    # No axis_swap on obs_phys — physical swap is already baked in.

    with torch.no_grad():
        a = tok([obs_aug])
        b = tok([obs_phys])

    pos = _drc_pos(tok, obs_aug)
    diff = (a.token_embeddings[0, pos] - b.token_embeddings[0, pos]).abs().max().item()
    assert diff < 1e-5, f"axis_swap DRC equivalence mismatch: {diff}"


def test_sign_flip_changes_drc_token():
    """Sign reflection must propagate into the DRC token."""
    torch.manual_seed(0)
    tok = BatchedStateTokenizer(d_model=64, n_freq=8).eval()

    obs = make_mock_obs(
        n_nets=2, pads_per_net=2, n_ratsnest_per_net=0,
        is_routing=False, current_net_phase=0,
        drc_violations=[_drc(x=120.0, y=65.0)],
    )
    obs_flip = make_mock_obs(
        n_nets=2, pads_per_net=2, n_ratsnest_per_net=0,
        is_routing=False, current_net_phase=0,
        drc_violations=[_drc(x=120.0, y=65.0)],
    )
    obs_flip["_aug"] = {"flip_x": -1, "flip_y": -1}

    with torch.no_grad():
        a = tok([obs])
        b = tok([obs_flip])

    pos = _drc_pos(tok, obs)
    diff = (a.token_embeddings[0, pos] - b.token_embeddings[0, pos]).abs().max().item()
    assert diff > 1e-3, f"flip had no effect on DRC token (diff={diff})"


def test_nn_shift_changes_drc_token():
    """nn_input_trans should shift the DRC xy encoding."""
    torch.manual_seed(0)
    tok = BatchedStateTokenizer(d_model=64, n_freq=8).eval()

    obs = make_mock_obs(
        n_nets=2, pads_per_net=2, n_ratsnest_per_net=0,
        is_routing=False, current_net_phase=0,
        drc_violations=[_drc(x=110.0, y=60.0)],
    )
    obs_shift = make_mock_obs(
        n_nets=2, pads_per_net=2, n_ratsnest_per_net=0,
        is_routing=False, current_net_phase=0,
        drc_violations=[_drc(x=110.0, y=60.0)],
    )
    obs_shift["_aug"] = {"nn_dx": 0.05, "nn_dy": -0.07}

    with torch.no_grad():
        a = tok([obs])
        b = tok([obs_shift])

    pos = _drc_pos(tok, obs)
    diff = (a.token_embeddings[0, pos] - b.token_embeddings[0, pos]).abs().max().item()
    assert diff > 1e-4, f"nn_input_trans had no effect on DRC token (diff={diff})"


def test_non_copper_layer_violation_does_not_crash():
    """A DRC violation landing on a non-copper layer (layer_human > n_copper
    — e.g. silk/edge on a real board) must not crash the walk; it should be
    encoded as the (0,0) 'unknown layer' sentinel."""
    from methods.rl_agent.models.v1.encoding import _safe_encode_layer

    assert _safe_encode_layer(6, 4) == (0.0, 0.0)   # non-copper (the crashing case)
    assert _safe_encode_layer(0, 4) == (0.0, 0.0)   # existing <1 sentinel preserved
    assert _safe_encode_layer(2, 4) != (0.0, 0.0)   # in-range layers get a real encoding

    torch.manual_seed(0)
    tok = BatchedStateTokenizer(d_model=64, n_freq=8).eval()
    obs = make_mock_obs(
        n_nets=1, pads_per_net=1, n_ratsnest_per_net=0, n_edges=0,
        is_routing=False, current_net_phase=0,
        drc_violations=[_drc(x=10.0, y=10.0, layer=99)],
    )
    with torch.no_grad():
        out = tok([obs])          # passes if tokenization does not crash
    assert out.token_embeddings.shape[0] == 1
