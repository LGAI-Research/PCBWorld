"""BC-initialised PPO fine-tuning helpers (no engine, no GPU).

Covers ``methods.rl_agent.training.finetune`` (reference-KL estimator + its
DRC-state mask) and the trainer-side schedule semantics that
``methods.rl_agent.training.loop`` derives from ``--critic-warmup-iters``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from methods.rl_agent.training.finetune import (
    KL_LOG_RATIO_CLAMP,
    kl_k3,
    kl_mask_from_obs,
    parse_type_ids,
)


def _obs(*type_ids):
    return {"drc_violations": [{"type_id": t, "x_mm": 0.0, "y_mm": 0.0} for t in type_ids]}


def test_parse_type_ids():
    assert parse_type_ids("2,4,6") == frozenset({2, 4, 6})
    assert parse_type_ids(" 6 ,2") == frozenset({2, 6})
    assert parse_type_ids("") == frozenset()
    assert parse_type_ids(None) == frozenset()


def test_kl_mask_keeps_seen_drops_unseen():
    seen = frozenset({2, 4, 6})
    obs_list = [
        {},                    # no DRC key at all -> keep
        _obs(),                # empty list -> keep
        _obs(6, 2, 4),         # only connectivity types -> keep
        _obs(6, 0),            # a clearance (0) token -> drop
        _obs(3),               # short -> drop
        {"drc_violations": [{"x_mm": 1.0}]},   # malformed token (no type_id) -> drop
    ]
    mask = kl_mask_from_obs(obs_list, seen)
    assert mask.dtype == np.float32
    assert mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_kl_k3_properties():
    new_lp = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    # identical log-probs -> exactly zero
    assert torch.allclose(kl_k3(new_lp.clone(), new_lp), torch.zeros(4))
    # any disagreement -> strictly positive, and the analytic k3 value
    ref_lp = new_lp + torch.tensor([0.3, -0.7, 1.2, -2.0])
    k = kl_k3(ref_lp, new_lp)
    assert bool((k > 0).all())
    r = ref_lp - new_lp
    assert torch.allclose(k, torch.exp(r) - r - 1.0)
    # clamp: an extreme log-ratio saturates instead of exploding
    huge = kl_k3(new_lp + 50.0, new_lp)
    assert torch.allclose(huge, torch.full((4,), math.exp(KL_LOG_RATIO_CLAMP) - KL_LOG_RATIO_CLAMP - 1.0))


def test_kl_k3_gradient_flows_only_through_policy():
    new_lp = torch.tensor([-1.0, -2.0], requires_grad=True)
    ref_lp = torch.tensor([-1.5, -1.0])            # no grad (reference)
    kl_k3(ref_lp, new_lp).sum().backward()
    # d/dnew [exp(ref-new) - (ref-new) - 1] = 1 - exp(ref-new)
    expected = 1.0 - torch.exp(ref_lp - new_lp.detach())
    assert torch.allclose(new_lp.grad, expected)


@pytest.mark.parametrize("critic_warmup_iters,warmup_iters", [(0, 20), (10, 20), (5, 0)])
def test_lr_schedule_critic_warmup_then_actor_ramp(critic_warmup_iters, warmup_iters):
    """Mirror of ``BaseTrainer._build_lr_scheduler``: constant critic LR for the
    warm-up iterations, then the actor's linear ramp starts from zero."""
    from types import SimpleNamespace
    from methods.rl_agent.training.loop import PPOTrainer

    args = SimpleNamespace(lr=2e-5, critic_warmup_lr=1e-4, warmup_iters=warmup_iters,
                           critic_warmup_iters=critic_warmup_iters)
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=args.lr)
    trainer = PPOTrainer.__new__(PPOTrainer)     # no __init__: only the scheduler helper
    trainer.args = args
    trainer.optimizer = opt
    sched = trainer._build_lr_scheduler()
    lrs = []
    for _ in range(critic_warmup_iters + max(warmup_iters, 1) + 2):
        lrs.append(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step()
    cw = critic_warmup_iters
    for i in range(cw):
        assert lrs[i] == pytest.approx(args.critic_warmup_lr)
    if warmup_iters > 0:
        assert lrs[cw] == pytest.approx(0.0)                       # actor ramp starts at 0
        assert lrs[cw + warmup_iters // 2] == pytest.approx(args.lr * (warmup_iters // 2) / warmup_iters)
        assert lrs[cw + warmup_iters] == pytest.approx(args.lr)
    else:
        assert lrs[cw] == pytest.approx(args.lr)


def test_in_critic_warmup_window():
    from types import SimpleNamespace
    from methods.rl_agent.training.loop import PPOTrainer

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.args = SimpleNamespace(critic_warmup_iters=3)
    assert [trainer._in_critic_warmup(i) for i in (1, 2, 3, 4)] == [True, True, True, False]
    trainer.args = SimpleNamespace(critic_warmup_iters=0)
    assert not trainer._in_critic_warmup(1)


# ---------------------------------------------------------------------------
# End-to-end through policy_update_loop on a tiny CPU policy (mock obs).
# ---------------------------------------------------------------------------
def _fresh_policy(seed: int = 0):
    from methods.rl_agent.models.v1.net import KiCadRLModel
    torch.manual_seed(seed)
    p = KiCadRLModel(disable_slot_emb=False, policy_net_select=False, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=4000,
                     n_freq=4, use_critic=True)
    with torch.no_grad():
        for layer in p.layers:
            layer.res_attn.alpha.fill_(0.7)
            layer.res_ff.alpha.fill_(0.5)
    return p


def _mock_buffer(policy, n_obs: int = 6):
    from tests._mock_obs import make_mock_obs
    from tests.helpers.dist_inputs import permissive_buffer, permissive_dist
    obs_list = [
        make_mock_obs(n_nets=2 + (i % 3), pads_per_net=2, n_ratsnest_per_net=2,
                      is_routing=(i % 2 == 0), current_net_phase=1, current_layer=1,
                      n_tracks=i * 2, n_vias=i)
        for i in range(n_obs)
    ]
    acts, old_lp, _ = policy.act_and_value(
        obs_list, dist=permissive_dist(policy, obs_list), deterministic=True,
    )
    return permissive_buffer(obs_list, acts, old_lp)


def _run_update(policy, buffer, **kw):
    from methods.rl_agent.algorithms._common import policy_update_loop
    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    return policy_update_loop(
        policy, opt, buffer, torch.device("cpu"), algo="ppo", n_epochs=1,
        batch_size=len(buffer["obs_list"]), normalize_advantages=True, **kw,
    )


def test_actor_frozen_updates_only_critic_head():
    policy = _fresh_policy()
    policy.detach_critic = True                      # as the trainer forces during warm-up
    buffer = _mock_buffer(policy)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    metrics = _run_update(policy, buffer, actor_frozen=True)
    moved = [n for n, p in policy.named_parameters() if not torch.equal(before[n], p)]
    assert moved, "critic head must learn during warm-up"
    assert all("critic" in n for n in moved), f"non-critic params moved: {moved[:5]}"
    # the actor terms are still reported (diagnostics), the loss is the value term only
    assert metrics["value_loss"] > 0.0
    assert metrics["loss"] == pytest.approx(0.5 * metrics["value_loss"], rel=1e-5)


def test_reference_kl_zero_at_init_and_masked():
    import copy
    policy = _fresh_policy()
    ref = copy.deepcopy(policy).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    buffer = _mock_buffer(policy)
    # identical reference -> k3 KL is exactly 0 on the first update, loss unchanged
    plain = _run_update(copy.deepcopy(policy), dict(buffer))
    with_kl = _run_update(copy.deepcopy(policy), dict(buffer), ref_policy=ref, kl_ref_coef=0.5)
    assert with_kl["kl_ref"] == pytest.approx(0.0, abs=1e-6)
    assert with_kl["kl_ref_mask_frac"] == pytest.approx(1.0)
    assert with_kl["loss"] == pytest.approx(plain["loss"], rel=1e-5)
    # a divergent reference -> positive KL; an all-zero mask removes it again
    ref2 = _fresh_policy(seed=1).eval()
    for prm in ref2.parameters():
        prm.requires_grad_(False)
    pos = _run_update(copy.deepcopy(policy), dict(buffer), ref_policy=ref2, kl_ref_coef=0.5)
    assert pos["kl_ref"] > 0.0
    masked = dict(buffer, transition=dataclasses.replace(
        buffer["transition"], kl_mask=np.zeros(len(buffer["obs_list"]), dtype=np.float32),
    ))
    zero = _run_update(copy.deepcopy(policy), masked, ref_policy=ref2, kl_ref_coef=0.5)
    assert zero["kl_ref"] == pytest.approx(0.0, abs=1e-6)
    assert zero["kl_ref_mask_frac"] == pytest.approx(0.0)
    assert zero["loss"] == pytest.approx(plain["loss"], rel=1e-5)


def test_ev_gate_extends_warmup_until_streak_then_caps():
    """--critic-warmup-min-ev: the critic phase runs past --critic-warmup-iters until
    EV >= min_ev for `streak` consecutive iterations; the LR schedule's actor ramp
    starts where the phase actually ended; a cap ends it regardless."""
    from types import SimpleNamespace
    from methods.rl_agent.training.loop import PPOTrainer

    def drive(evs, **cfg):
        t = PPOTrainer.__new__(PPOTrainer)
        base = dict(critic_warmup_iters=2, critic_warmup_min_ev=0.9,
                    critic_warmup_ev_streak=2, critic_warmup_max_iters=6)
        base.update(cfg)
        t.args = SimpleNamespace(**base)
        frozen = []
        for it, ev in enumerate(evs, start=1):
            t._actor_frozen = t._in_critic_warmup(it)
            frozen.append(t._actor_frozen)
            t._note_critic_warmup_ev(it, {"explained_variance": ev})
        return frozen, getattr(t, "_critic_warmup_end", None), getattr(t, "_critic_gate_passed", False)

    # EV climbs: 0.1, 0.5, 0.95, 0.97 -> streak of 2 at iter 4 -> actor from iter 5
    frozen, end, passed = drive([0.1, 0.5, 0.95, 0.97, 0.98, 0.98, 0.98])
    assert frozen == [True, True, True, True, False, False, False]
    assert end == 4 and passed
    # a dip breaks the streak: 0.95, 0.5, 0.95, 0.96 -> passes at iter 6 (cap = 6 too)
    frozen, end, passed = drive([0.1, 0.5, 0.95, 0.5, 0.95, 0.96, 0.9])
    assert frozen == [True, True, True, True, True, True, False]
    assert end == 6
    # never reaches the gate -> the cap (6) ends the phase, flagged as not passed
    frozen, end, passed = drive([0.1] * 8)
    assert frozen == [True] * 6 + [False, False]
    assert end == 6 and not passed
    # gate off (min_ev 0) -> plain fixed length
    frozen, end, passed = drive([0.1] * 4, critic_warmup_min_ev=0.0)
    assert frozen == [True, True, False, False] and end == 2


# ---------------------------------------------------------------------------
# --resume: the LR and board schedules continue where the checkpoint stopped.
# ---------------------------------------------------------------------------
def _schedule_trainer(tmp_path, *, n_envs=4, critic_warmup_iters=0, n_boards=10,
                      split_name="split.json", ds_name="boards", ds_link=None,
                      boards_dataset_dir=None, **gate):
    import json
    from types import SimpleNamespace
    from methods._shared.board_scheduler import BoardScheduler, BoardSchedulerConfig
    from methods.rl_agent.training.loop import PPOTrainer

    ds = tmp_path / ds_name
    ds.mkdir(exist_ok=True)
    ids = [f"board_{i:05d}" for i in range(n_boards)]
    for bid in ids:
        (ds / f"{bid}.kicad_pcb").touch()
    listed = ds
    if ds_link:                                   # the same directory through a symlink
        listed = tmp_path / ds_link
        if not listed.exists():
            listed.symlink_to(ds)
    split = tmp_path / split_name
    split.write_text(json.dumps({"dataset_dirs": {"train": str(listed)}, "easy": {"train": ids}}))
    args = SimpleNamespace(lr=2e-5, critic_warmup_lr=1e-4, warmup_iters=5,
                           critic_warmup_iters=critic_warmup_iters, n_envs=n_envs, seed=42,
                           boards_order="per_env_epoch", boards_json=str(split),
                           boards_difficulty="easy", boards_split="train",
                           boards_dataset_dir=boards_dataset_dir, **gate)
    t = PPOTrainer.__new__(PPOTrainer)
    t.args = args
    t.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=args.lr)
    t.lr_scheduler = t._build_lr_scheduler()
    t.scheduler = BoardScheduler(BoardSchedulerConfig(
        mode="per_env_epoch", boards_json=str(split), seed=args.seed + 7919,
        dataset_dir=boards_dataset_dir))
    t.boards = t.scheduler.paths
    return t


def _run_iters(t, n):
    """Schedule side effects of ``n`` iterations: (LR used, boards dealt) per iteration."""
    out = []
    for _ in range(n):
        lr = t.optimizer.param_groups[0]["lr"]
        out.append((lr, t.scheduler.next_indices(t._board_pick()[0])))
        t.lr_scheduler.step()
    return out


@pytest.mark.parametrize("saved_state", [True, False], ids=["schedules", "legacy"])
def test_resume_continues_lr_and_board_order(tmp_path, saved_state):
    # 3 iterations x 4 boards crosses the 10-board epoch; iteration 3 is inside the LR ramp
    ref = _schedule_trainer(tmp_path)
    _run_iters(ref, 3)
    ckpt = {"iteration": 3, "args": vars(ref.args)}
    if saved_state:
        ckpt["schedules"] = ref._schedule_state()
    expected = _run_iters(ref, 4)

    resumed = _schedule_trainer(tmp_path)
    resumed._restore_schedules(ckpt, ckpt["args"], completed=3)
    assert _run_iters(resumed, 4) == expected
    assert expected[0][0] == pytest.approx(2e-5 * 3 / 5)   # mid-ramp, not restarted at 0


def test_resume_restores_critic_warmup_end(tmp_path):
    ref = _schedule_trainer(tmp_path, critic_warmup_iters=2)
    ref._critic_warmup_end, ref._critic_gate_passed = 4, True   # EV gate ended the phase late
    _run_iters(ref, 6)
    ckpt = {"iteration": 6, "args": vars(ref.args), "schedules": ref._schedule_state()}
    expected = _run_iters(ref, 3)

    resumed = _schedule_trainer(tmp_path, critic_warmup_iters=2)
    resumed._restore_schedules(ckpt, ckpt["args"], completed=6)
    assert resumed._critic_warmup_end == 4 and resumed._critic_gate_passed
    assert not resumed._in_critic_warmup(7)
    assert _run_iters(resumed, 3) == expected


def test_resume_legacy_ckpt_with_other_board_draws_warns(tmp_path, capsys):
    ref = _schedule_trainer(tmp_path, n_envs=4)
    ckpt = {"iteration": 3, "args": vars(ref.args)}           # no saved schedules
    resumed = _schedule_trainer(tmp_path, n_envs=8)
    resumed._restore_schedules(ckpt, ckpt["args"], completed=3)
    assert "board order restarts" in capsys.readouterr().out
    assert resumed.scheduler.iter_idx == -1                    # left untouched
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(2e-5 * 3 / 5)


@pytest.mark.parametrize("ds_name,replayed", [("boards", True), ("boards_elsewhere", False)])
def test_resume_legacy_ckpt_split_file_moved(tmp_path, capsys, ds_name, replayed):
    """A regenerated split file with the same ids and dataset dir still replays;
    the same ids under another dataset dir are other boards and do not."""
    ref = _schedule_trainer(tmp_path)
    _run_iters(ref, 3)
    ckpt = {"iteration": 3, "args": dict(vars(ref.args))}
    expected = _run_iters(ref, 4)

    resumed = _schedule_trainer(tmp_path, split_name="regenerated.json", ds_name=ds_name)
    resumed._restore_schedules(ckpt, ckpt["args"], completed=3)
    if replayed:
        assert _run_iters(resumed, 4) == expected
    else:
        assert "board order restarts" in capsys.readouterr().out
        assert resumed.scheduler.iter_idx == -1


@pytest.mark.parametrize("other", [
    dict(n_boards=6, split_name="short.json"),               # fewer boards: the saved order would index past the list
    dict(ds_name="boards_elsewhere", split_name="moved.json"),  # same ids and length, other boards
])
def test_resume_saved_order_needs_the_same_board_list(tmp_path, capsys, other):
    ref = _schedule_trainer(tmp_path)
    _run_iters(ref, 3)
    ckpt = {"iteration": 3, "args": dict(vars(ref.args)), "schedules": ref._schedule_state()}

    resumed = _schedule_trainer(tmp_path, **other)
    resumed._restore_schedules(ckpt, ckpt["args"], completed=3)
    assert "board order restarts" in capsys.readouterr().out
    assert resumed.scheduler.iter_idx == -1                     # left untouched
    _run_iters(resumed, 4)                                      # and still draws in range


def test_resume_saved_order_through_a_symlinked_dataset_dir(tmp_path):
    """Another split file naming the same boards through a symlink is the same board list."""
    ref = _schedule_trainer(tmp_path)
    _run_iters(ref, 3)
    ckpt = {"iteration": 3, "args": dict(vars(ref.args)), "schedules": ref._schedule_state()}
    expected = _run_iters(ref, 4)

    resumed = _schedule_trainer(tmp_path, split_name="linked.json", ds_link="boards_link")
    resumed._restore_schedules(ckpt, ckpt["args"], completed=3)
    assert _run_iters(resumed, 4) == expected


@pytest.mark.parametrize("override,replayed", [("boards", True), ("boards_other", False)])
def test_resume_legacy_ckpt_dataset_dir_override(tmp_path, capsys, override, replayed):
    """A legacy checkpoint (no saved schedules) replays under --boards-dataset-dir only when
    that dir is the one its split file named."""
    ref = _schedule_trainer(tmp_path)
    _run_iters(ref, 3)
    ckpt = {"iteration": 3, "args": dict(vars(ref.args))}
    expected = _run_iters(ref, 4)

    _schedule_trainer(tmp_path, ds_name="boards_other", split_name="other.json")   # same ids, other dir
    resumed = _schedule_trainer(tmp_path, boards_dataset_dir=str(tmp_path / override))
    resumed._restore_schedules(ckpt, ckpt["args"], completed=3)
    if replayed:
        assert _run_iters(resumed, 4) == expected
    else:
        assert "board order restarts" in capsys.readouterr().out
        assert resumed.scheduler.iter_idx == -1


def _drive_gate(t, first, evs):
    """Iterations ``first``.. of the EV gate with the given EVs; the actor-frozen flag per iteration."""
    frozen = []
    for it, ev in enumerate(evs, start=first):
        t._actor_frozen = t._in_critic_warmup(it)
        frozen.append(t._actor_frozen)
        t._note_critic_warmup_ev(it, {"explained_variance": ev})
    return frozen


def test_resume_mid_ev_gate_keeps_the_streak(tmp_path, capsys):
    """Stopped while the EV gate extends the warm-up with 2 of 3 streak iterations in hand:
    the resumed run starts the actor where the uninterrupted run does."""
    gate = dict(critic_warmup_iters=2, critic_warmup_min_ev=0.9, critic_warmup_ev_streak=3,
                critic_warmup_max_iters=10)
    evs = [0.1, 0.2, 0.3, 0.95, 0.96, 0.97, 0.98, 0.98]
    ref = _schedule_trainer(tmp_path, **gate)
    uninterrupted = _drive_gate(ref, 1, evs)
    assert uninterrupted == [True] * 6 + [False, False]          # gate passes at iter 6

    first = _schedule_trainer(tmp_path, **gate)
    _drive_gate(first, 1, evs[:5])                                # stop after iter 5 (streak 2 of 3)
    ckpt = {"iteration": 5, "args": dict(vars(first.args)), "schedules": first._schedule_state()}
    resumed = _schedule_trainer(tmp_path, **gate)
    resumed._restore_schedules(ckpt, ckpt["args"], completed=5)
    assert _drive_gate(resumed, 6, evs[5:]) == uninterrupted[5:]

    assert len(ckpt["schedules"]["ev_history"]) == 3              # only what the gate reads

    del ckpt["schedules"]["ev_history"]                           # written before the history was saved
    for completed in (5, 1):                                      # iter 1 already feeds the gate at iter 2..3
        again = _schedule_trainer(tmp_path, **gate)
        again._restore_schedules(dict(ckpt, iteration=completed), ckpt["args"], completed=completed)
        assert "no EV history" in capsys.readouterr().out


def test_resume_saved_order_without_digest_checks_the_range(tmp_path, capsys):
    """A saved order from before the digest existed, on a split file rewritten in place with
    fewer boards, restarts the order instead of indexing past the list."""
    ref = _schedule_trainer(tmp_path)
    _run_iters(ref, 3)
    ckpt = {"iteration": 3, "args": dict(vars(ref.args)), "schedules": ref._schedule_state()}
    del ckpt["schedules"]["board_list_digest"]

    resumed = _schedule_trainer(tmp_path, n_boards=6)            # same split path and args, fewer ids
    resumed._restore_schedules(ckpt, ckpt["args"], completed=3)
    assert "indexes past" in capsys.readouterr().out
    assert resumed.scheduler.iter_idx == -1
    _run_iters(resumed, 4)


def test_kl_reference_of_the_init_ckpt_uses_the_runs_arch_config(tmp_path):
    """The KL reference built from the init checkpoint tokenizes like pi: the run's resolved
    arch config (here obstacle_obs taken from the command line over the checkpoint's estimate),
    the init checkpoint's weights."""
    from types import SimpleNamespace

    from configs.loader.schema import RLPolicyConfig
    from methods.rl_agent.training.loop import PPOTrainer
    from tests.helpers.ckpt_payload import ckpt_config, ckpt_payload

    values = dict(ckpt_config(drop=("obstacle_obs",)))          # estimate: obstacle_obs False
    assert values["obstacle_obs"] is False
    model = RLPolicyConfig.from_checkpoint(values).build("cpu")
    payload = ckpt_payload(drop=("obstacle_obs",))
    payload["policy_state_dict"] = model.state_dict()
    path = tmp_path / "init.pt"
    torch.save(payload, path)
    t = PPOTrainer.__new__(PPOTrainer)
    t.device = torch.device("cpu")
    t.args = SimpleNamespace(**{**values, "obstacle_obs": True}, kl_ref_coef=0.1, kl_ref_ckpt=None,
                             init_ckpt=str(path), update_gpus=1, strict_ckpt=False,
                             kl_ref_mask_unseen_drc=False, kl_ref_seen_drc_types="")
    t._build_ref_policy()
    assert t.ref_policy.tokenizer.obstacle_obs is True
    ref_state = t.ref_policy.state_dict()
    assert all(torch.equal(ref_state[k], v) for k, v in model.state_dict().items())
