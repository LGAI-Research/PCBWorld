"""Tests for methods/rl_agent/policy/rl_provider.py (the visualizer's local RL provider).

No checkpoint or model: ``KiCadRLAgent.from_checkpoint`` is replaced by a fake agent that
returns a fixed action, and the obs is the pure-Python mock from ``tests/_mock_obs.py``.

The fake agent carries a checkpoint's resolved config (``tests/helpers/ckpt_payload``): the
provider reads its inference-side knobs from it — a policy trained with wrapper-side net
picking must not have its pointer read as a net index.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from methods.rl_agent.policy import agent as agent_mod
from methods.rl_agent.models.v1.spec import NUM_ACTIONS
from methods.rl_agent.policy.rl_provider import RLPolicyProvider, _resolve_rl_ckpt_file
from pcb_world.core.action_schema import ACT_NET_SELECT
from tests._mock_obs import make_mock_obs
from tests.helpers.ckpt_payload import ckpt_config

_ALL_ACTIONS = [[True] * NUM_ACTIONS]   # one obs, every action type legal


class _FakeAgent:
    """Stands in for KiCadRLAgent: always emits net_select with pointer 1, walkaround."""

    def __init__(self, ckpt_args):
        self.model = object()
        self.device = torch.device("cpu")
        self.train_iteration = 0
        self.ckpt_args = ckpt_args

    def act(self, obs_list, **kwargs):
        return torch.tensor([[ACT_NET_SELECT, 1, 2]]), torch.zeros(1)


@pytest.fixture
def make_provider(monkeypatch, tmp_path):
    def _make(train_args: dict, generation: str = "iclr_2026_10") -> RLPolicyProvider:
        monkeypatch.setattr(
            agent_mod.KiCadRLAgent, "from_checkpoint",
            classmethod(lambda cls, path, device="auto", *, strict, deterministic=False:
                        _FakeAgent(ckpt_config(generation, **train_args))),
        )
        ckpt = tmp_path / "policy_best.pt"
        ckpt.write_bytes(b"")
        return RLPolicyProvider(SimpleNamespace(
            checkpoint_path=str(ckpt), device="cpu", temperature=0.0, strict_ckpt=False,
        ))
    return _make


@pytest.mark.parametrize("train_args, expected_body", [
    ({"policy_net_select": False}, "net_select 1"),  # wrapper picks: first net, pointer ignored
    ({"policy_net_select": True}, "net_select 2"),   # honoured -> pointer 1 indexes sorted codes
])
def test_policy_net_select_follows_the_checkpoint(make_provider, train_args, expected_body):
    provider = make_provider(train_args)
    assert provider.policy_net_select is train_args["policy_net_select"]

    obs = make_mock_obs()   # nets 1 and 2 -> sorted_net_codes == [1, 2]
    responses, token_counts = provider.generate(
        [("", "")], observations=[obs], action_masks=_ALL_ACTIONS,
    )

    assert len(responses) == len(token_counts) == 1
    assert f"<action>{expected_body}</action>" in responses[0]


@pytest.mark.parametrize("train_args,on", [
    ({"policy_net_select": True, "net_reselect": True}, True),
    ({"policy_net_select": True, "net_reselect": False}, False),
    ({"policy_net_select": True}, False),           # pre-knob checkpoint (legacy value)
])
def test_net_valid_mask_follows_the_checkpoint_net_reselect(make_provider, train_args, on):
    """The provider masks nets as the checkpoint was trained (the visualizer
    builds its PCBWorld with the same value): a closed net with ratsnest left is
    selectable only under ``net_reselect``."""
    provider = make_provider(train_args)
    assert provider.net_reselect is on
    seen = {}
    act = provider._agent.act
    provider._agent.act = lambda obs_list, **kw: (seen.update(kw), act(obs_list, **kw))[1]

    obs = make_mock_obs()             # nets 1 and 2, both with ratsnest
    obs["closed_nets"] = [2]
    provider.generate([("", "")], observations=[obs], action_masks=_ALL_ACTIONS)
    assert seen["dist"].net_valid.tolist() == [[True, on]]


@pytest.mark.parametrize("train_args,on", [
    ({"no_mask_start_point": False}, True),
    ({"no_mask_start_point": True}, False),       # trained with --no-mask-start-point
])
def test_mask_start_point_follows_the_checkpoint(make_provider, train_args, on):
    """The checkpoint stores the inverted CLI flag; the loader turns it into the
    ``mask_start_point`` field the provider's mask knobs read."""
    assert make_provider(train_args).mask_knobs.mask_start_point is on


@pytest.mark.parametrize("generation, train_args, want", [
    ("kdd_2026_05", {}, (True, False, False, False)),      # pre-knob: legacy values
    ("iclr_2026_10", {"no_mask_start_point": True, "offboard_mask": True,
                      "force_walkaround": True, "net_reselect": True}, (False, True, True, True)),
])
def test_mask_knobs_resolve_like_the_eval_env(generation, train_args, want):
    """``CkptMaskKnobs`` (shared by both serving paths) reads the checkpoint the
    way ``RLEnvConfig.from_checkpoint`` rebuilds the training env."""
    from configs.loader.schema import RLEnvConfig
    from methods.rl_agent.policy.rl_provider import CkptMaskKnobs

    ta = ckpt_config(generation, **train_args)
    k = CkptMaskKnobs.from_checkpoint(ta)
    assert (k.mask_start_point, k.offboard_mask, k.force_walkaround, k.net_reselect) == want
    cfg = RLEnvConfig.from_checkpoint(ta, max_steps=8)
    assert want == (cfg.mask_start_point, cfg.offboard_mask, cfg.force_walkaround,
                    cfg.env.net_reselect)


@pytest.mark.parametrize("generation, train_args", [
    ("kdd_2026_05", {}),                                   # pre-knob checkpoint
    ("iclr_2026_10", {"force_walkaround": True, "offboard_mask": True}),
])
def test_dist_is_the_wrappers_payload_on_a_routing_obs(make_provider, generation, train_args):
    """The bare-PCBWorld provider hands the policy the same distribution inputs
    the training wrapper would (``encoding.mask_payload``): the mode row follows
    the checkpoint's ``force_walkaround``, off-layer candidates of the routing
    head are blocked on make_line, off-board directionals follow ``offboard_mask``."""
    from methods.rl_agent.models.v1.encoding import (
        cand_raw_list_from_obs, mode_mask, offboard_pointer_indices,
        offlayer_pointer_indices,
    )

    provider = make_provider(train_args, generation)
    seen = {}
    act = provider._agent.act
    provider._agent.act = lambda obs_list, **kw: (seen.update(kw), act(obs_list, **kw))[1]
    obs = make_mock_obs(is_routing=True, current_net_phase=1, bbox=(0.0, 0.0, 30.0, 20.0))
    obs["_aug"] = {"directional_candidates": "mres8"}
    provider.generate([("", "")], observations=[obs], action_masks=_ALL_ACTIONS)

    raw = cand_raw_list_from_obs(obs)
    cand_mm = [(x, y, ly) for (x, y, ly, _c) in raw]
    dist = seen["dist"]
    fw = provider.mask_knobs.force_walkaround
    assert fw is bool(train_args.get("force_walkaround", False))
    assert dist.mode[0].tolist() == mode_mask(force_routing_mode=None, force_walkaround=fw).tolist()
    off = offlayer_pointer_indices(obs=obs, cand_mm=cand_mm).tolist()
    assert [c for c in dist.offlayer[0].tolist() if c >= 0] == off
    offb = offboard_pointer_indices(
        offboard_mask=provider.mask_knobs.offboard_mask, obs=obs, cand_mm=cand_mm,
        cand_ctype=[c for (*_xyz, c) in raw],
    ).tolist()
    assert [c for c in dist.pointer[0].tolist() if c >= 0] == offb   # no start point yet
    assert bool(offb) is provider.mask_knobs.offboard_mask


def test_a_run_dir_resolves_policy_best_then_the_latest_iter(tmp_path):
    for n in (5, 40):
        (tmp_path / f"policy_iter_{n}.pt").touch()
    assert _resolve_rl_ckpt_file(str(tmp_path)) == str(tmp_path / "policy_iter_40.pt")
    (tmp_path / "policy_best.pt").touch()
    assert _resolve_rl_ckpt_file(str(tmp_path)) == str(tmp_path / "policy_best.pt")

