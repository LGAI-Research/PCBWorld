"""``DistInputs`` / ``Transition`` row bundles and the model's use of them.

- the generic row ops (concat padding per field, select, to, None handling);
- ``net_valid`` presence must match the model's ``policy_net_select``;
- the net_select pointer gate: a net_select row's pointer counts in the
  log-prob/entropy iff ``policy_net_select`` (otherwise the env picks the net and
  the pointer is not part of the action), whatever the row's mode mask, at act
  time and at update time alike.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from methods.rl_agent.models.v1.net import KiCadRLModel
from methods.rl_agent.models.v1.spec import (
    ACT_NET_SELECT, MODE_LETTER_TO_INT, DistInputs, Transition,
)
from tests._mock_obs import make_mock_obs
from tests.helpers.dist_inputs import payload, permissive_dist, permissive_dist_np


def _dist(ptr_rows, off_rows, nvm_rows=None) -> DistInputs:
    payloads = [
        payload(pointer=np.asarray(p, np.int64), offlayer=np.asarray(o, np.int64),
                net_valid=np.asarray(n if n is not None else [], bool))
        for p, o, n in zip(ptr_rows, off_rows, nvm_rows or [None] * len(ptr_rows))
    ]
    return DistInputs.from_payloads(payloads, policy_net_select=nvm_rows is not None)


class TestRowOps:
    def test_concat_pads_each_ragged_field_with_its_fill(self):
        a = _dist([[1], []], [[], []], nvm_rows=[[True], [True]])
        b = _dist([[2, 3]], [[4]], nvm_rows=[[True, False, True]])
        c = DistInputs.concat([a, b])
        assert c.pointer.tolist() == [[1, -1], [-1, -1], [2, 3]]
        assert c.offlayer.tolist() == [[-1], [-1], [4]]
        assert c.net_valid.tolist() == [[True, False, False], [True, False, False],
                                        [True, False, True]]
        assert c.action.shape == (3, 7) and c.mode.shape == (3, 3)

    def test_select_keeps_rows_and_none(self):
        d = DistInputs.concat([_dist([[1], [2, 5]], [[], [3]])])
        s = d.select([1, 1])
        assert s.pointer.tolist() == [[2, 5], [2, 5]]
        assert s.net_valid is None
        t = s.to("cpu")
        assert t.pointer.dtype == torch.int32 and t.action.dtype == torch.bool

    def test_cand_index_fields_are_int32_through_the_buffer(self):
        """``pointer`` / ``offlayer`` are stored int32 (buffer memory, DDP
        pickle) from the payload stack through concat, select and ``to``."""
        d = DistInputs.concat([_dist([[1]], [[2]]), _dist([[3, 4]], [[]])])
        for v in (d, d.select([1]), d.to("cpu")):
            assert v.pointer.dtype in (np.int32, torch.int32)
            assert v.offlayer.dtype in (np.int32, torch.int32)

    def test_nested_transition_concat(self):
        row = Transition(
            dist=permissive_dist_np(1), actions=np.zeros((1, 3), np.int64),
            old_log_probs=np.zeros(1, np.float32), old_values=np.zeros(1, np.float32),
            kl_mask=None,
        )
        tr = Transition.concat([row, replace(row, dist=_dist([[7]], [[8]]))])
        assert tr.dist.pointer.tolist() == [[-1], [7]]
        assert tr.kl_mask is None and tr.actions.shape == (2, 3)

    def test_concat_refuses_partial_none_and_empty(self):
        with pytest.raises(ValueError, match="net_valid"):
            DistInputs.concat([_dist([[]], [[]]), _dist([[]], [[]], nvm_rows=[[True]])])
        with pytest.raises(ValueError, match="at least one part"):
            DistInputs.concat([])


def _model(policy_net_select: bool) -> KiCadRLModel:
    torch.manual_seed(0)
    return KiCadRLModel(
        disable_slot_emb=False,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        policy_net_select=policy_net_select,
    ).eval()


def test_net_valid_presence_must_match_policy_net_select():
    obs = [make_mock_obs(n_nets=2)]
    for pns in (False, True):
        m = _model(pns)
        wrong = replace(
            permissive_dist(m, obs),
            net_valid=None if pns else torch.ones(1, 2, dtype=torch.bool),
        )
        with pytest.raises(ValueError, match="net_valid"):
            m.act_and_value(obs, dist=wrong)


@pytest.mark.parametrize("pns", [False, True])
def test_net_select_pointer_gate(pns):
    """Teacher-forced net_select log-prob = type term + pointer term exactly
    when ``policy_net_select``, for an unrestricted and a restricted
    (force_walkaround) mode row alike."""
    m = _model(pns)
    obs = [make_mock_obs(n_nets=3)] * 2
    free = permissive_dist(m, obs)
    walk_only = torch.zeros(2, 3, dtype=torch.bool)
    walk_only[:, MODE_LETTER_TO_INT["w"]] = True
    dist = replace(free, mode=torch.stack([free.mode[0], walk_only[1]]))
    actions = torch.tensor([[ACT_NET_SELECT, 1, -1]] * 2)

    with torch.no_grad():
        lp, _, _ = m.evaluate_actions_and_value(obs, actions, dist=dist)
        f = m.factored_action_logits(obs, dist=dist)
    at = torch.log_softmax(f["at_logits"], -1)[:, ACT_NET_SELECT]
    ptr = torch.log_softmax(f["ptr_logits"][:, ACT_NET_SELECT], -1)[:, 1]
    assert torch.allclose(lp, at + ptr if pns else at, atol=1e-5)


@pytest.mark.parametrize("pns", [False, True])
def test_net_select_act_and_update_log_probs_agree(pns):
    """With net_select the only legal type, the act-time log-prob is the pointer
    term iff ``policy_net_select`` (else 0), and the update re-scores the same
    action under the stored bundle to the same value."""
    m = _model(pns)
    obs = [make_mock_obs(n_nets=3)] * 2
    free = permissive_dist(m, obs)
    only_ns = torch.zeros_like(free.action)
    only_ns[:, ACT_NET_SELECT] = True
    dist = replace(free, action=only_ns)

    with torch.no_grad():
        actions, act_lp, _ = m.act_and_value(obs, dist=dist)
        upd_lp, _, _ = m.evaluate_actions_and_value(obs, actions, dist=dist)
        f = m.factored_action_logits(obs, dist=dist)
    assert (actions[:, 0] == ACT_NET_SELECT).all()
    ptr = torch.log_softmax(f["ptr_logits"][:, ACT_NET_SELECT], -1)
    ptr = ptr.gather(1, actions[:, 1:2].clamp(min=0)).squeeze(1)
    assert torch.allclose(act_lp, ptr if pns else torch.zeros_like(act_lp), atol=1e-5)
    assert torch.allclose(upd_lp, act_lp, atol=1e-5)


def test_int32_block_lists_mask_like_int64():
    """The stored int32 block lists mask exactly like int64 ones."""
    m = _model(False)
    obs = [make_mock_obs(n_nets=2, is_routing=True, current_net_phase=1)] * 2
    blocked = torch.tensor([[0, -1], [1, 2]], dtype=torch.int32)
    d32 = replace(permissive_dist(m, obs), pointer=blocked, offlayer=blocked)
    d64 = replace(d32, pointer=blocked.long(), offlayer=blocked.long())
    with torch.no_grad():
        f32 = m.factored_action_logits(obs, dist=d32)["ptr_logits"]
        f64 = m.factored_action_logits(obs, dist=d64)["ptr_logits"]
    assert torch.equal(f32, f64)
    assert torch.isinf(f32[0, :, 0]).any()     # column 0 blocked on row 0
