"""DDP update equivalence: N ranks x B/N shards == single process x B.

Adoption condition for multi-GPU PPO update (DDP): with the global batch held
fixed and split evenly across ranks, per-rank shard-MEAN loss + DDP's
rank-averaged gradient must equal the single-process full-batch mean gradient
(within reduction-order tolerance), so training is numerically equivalent to
single-GPU regardless of world size.

Pinned here at two levels, both with a tiny KiCadRLModel (d32/l2) and
synthetic inputs (embs injected — no tokenizer, no C++), update-shaped pass
(state pass with K/V cache + incremental action decode + backward), the same
machinery pattern as tests/test_incremental_decode.py:

  * one-step gradient equivalence (fp32, 1e-5);
  * 5-step parameter equivalence with Adam (optimizer included).

CPU + gloo backend via torch.multiprocessing.spawn — runs anywhere, no GPU.
An NCCL variant runs the same checks on 2 CUDA devices when present
and skips elsewhere.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from methods.rl_agent.algorithms._common import policy_update_loop
from methods.rl_agent.models.v1.net import NUM_ACTION_TYPES, KiCadRLModel
from methods.rl_agent.training.ddp import DDPCtx
from tests._mock_obs import make_mock_obs

D_MODEL, N_HEADS, N_LAYERS, D_FF = 32, 4, 2, 64
B, L = 8, 24
WORLD = 2
N_STEPS = 5
LR = 1e-3


def _tiny_model(device: str) -> KiCadRLModel:
    """ReZero gates opened — an ungated model would make the comparison
    vacuous (identity passthrough). Same pattern as test_incremental_decode."""
    torch.manual_seed(0)
    m = KiCadRLModel(
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        max_seq_len=2000, n_freq=4, use_critic=True, same_net_bias=True,
    ).to(device)
    with torch.no_grad():
        m.same_net_bias.alpha.copy_(torch.tensor([-0.3, 0.7, -0.5, 1.1]))
        for layer in m.layers:
            layer.res_attn.alpha.fill_(0.7)
            layer.res_ff.alpha.fill_(0.5)
    return m


class _UpdateShim(torch.nn.Module):
    """Update pass as a forward() so DDP's gradient hooks engage."""

    def __init__(self, m: KiCadRLModel):
        super().__init__()
        self.m = m

    def forward(self, embs, kpm, slot_ids, e1):
        Lx = embs.size(1)
        H1, cache = self.m._run_transformer(
            embs, Lx, kpm, slot_ids=slot_ids, return_cache=True)
        point = H1[:, 3:4] + 0.3
        h_new, _ = self.m._decode_appended(cache, torch.cat([e1, point], 1))
        # per-sample scalar -> batch MEAN: shard means + DDP rank averaging
        # == full-batch mean (equal shards) — the equivalence being pinned.
        per = (H1[:, 2].pow(2).sum(-1) + h_new.pow(2).sum((-1, -2))
               + H1[:, Lx - 1].sum(-1))
        return per.mean()


def _make_batch(step: int):
    """Global batch, fixed seed per step — ranks slice their own shard so
    every world size consumes identical data."""
    g = torch.Generator().manual_seed(1000 + step)
    embs = torch.randn(B, L, D_MODEL, generator=g)
    kpm = torch.zeros(B, L, dtype=torch.bool)
    kpm[:, int(L * 0.75):] = True  # padding rows exercise the masks
    slot_ids = torch.randint(-1, 6, (B, L), generator=g)
    e1 = torch.randn(B, 1, D_MODEL, generator=g)
    return embs, kpm, slot_ids, e1


def _shard(tensors, rank: int, world: int, device: str):
    n = B // world
    return [t[rank * n:(rank + 1) * n].to(device) for t in tensors]


def _run_steps(model, m, device, rank=0, world=1):
    """N_STEPS Adam updates; returns (step-0 grads, final params) on CPU."""
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    grads = None
    for step in range(N_STEPS):
        batch = _shard(_make_batch(step), rank, world, device)
        loss = model(*batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            grads = {n: p.grad.detach().cpu().clone()
                     for n, p in m.named_parameters() if p.grad is not None}
        opt.step()
    params = {n: p.detach().cpu().clone() for n, p in m.named_parameters()}
    return grads, params


def _ddp_worker(rank: int, backend: str, rdv_file: str, out_dir: str):
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(
        backend, init_method=f"file://{rdv_file}", rank=rank,
        world_size=WORLD)
    m = _tiny_model(device)
    ddp = DDP(_UpdateShim(m),
              device_ids=[rank] if backend == "nccl" else None,
              static_graph=True)  # unused params (heads etc.) are static
    grads, params = _run_steps(ddp, m, device, rank=rank, world=WORLD)
    if rank == 0:
        torch.save({"grads": grads, "params": params},
                   f"{out_dir}/ddp_{backend}.pt")
    dist.destroy_process_group()


def _reference(device: str = "cpu"):
    m = _tiny_model(device)
    return _run_steps(_UpdateShim(m), m, device)


def _spawn_and_load(backend: str, tmp_dir) -> dict:
    mp.spawn(_ddp_worker,
             args=(backend, str(tmp_dir / f"rdv_{backend}"), str(tmp_dir)),
             nprocs=WORLD, join=True)
    return torch.load(tmp_dir / f"ddp_{backend}.pt", weights_only=True)


def _assert_grads_match(ddp_grads, ref_grads):
    assert set(ddp_grads) == set(ref_grads)  # same used-parameter set
    for name, g_ref in ref_grads.items():
        assert torch.allclose(ddp_grads[name], g_ref, atol=1e-5, rtol=1e-5), \
            f"grad mismatch {name}: max|d|=" \
            f"{(ddp_grads[name] - g_ref).abs().max().item():.3e}"


def _assert_params_match(ddp_params, ref_params):
    # DDP (NCCL 2-rank) and the single-GPU reference differ in reduction
    # order, so a float-level deviation between them is expected, and its
    # magnitude depends on the input distribution — deterministic
    # max|d| can reach ~4.84e-5. This test's goal is to catch a real DDP
    # wiring bug (>>1e-3), so atol is calibrated to 1e-4.
    for name, p_ref in ref_params.items():
        assert torch.allclose(ddp_params[name], p_ref, atol=1e-4, rtol=1e-4), \
            f"param mismatch {name}: max|d|=" \
            f"{(ddp_params[name] - p_ref).abs().max().item():.3e}"


# ---------------------------------------------------------------------------
# CPU + gloo — runs everywhere
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gloo_result(tmp_path_factory):
    return _spawn_and_load("gloo", tmp_path_factory.mktemp("ddp_gloo"))


def test_grad_equivalence_gloo(gloo_result):
    ref_grads, _ = _reference()
    _assert_grads_match(gloo_result["grads"], ref_grads)


def test_param_equivalence_5step_gloo(gloo_result):
    _, ref_params = _reference()
    _assert_params_match(gloo_result["params"], ref_params)


# ---------------------------------------------------------------------------
# CUDA + NCCL — hosts with >=2 CUDA devices only, skips elsewhere
# ---------------------------------------------------------------------------
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs >=2 CUDA devices")
def test_grad_and_param_equivalence_nccl(tmp_path):
    result = _spawn_and_load("nccl", tmp_path)
    ref_grads, ref_params = _reference("cuda:0")
    _assert_grads_match(result["grads"], ref_grads)
    _assert_params_match(result["params"], ref_params)


# ===========================================================================
# Production path: the REAL ``policy_update_loop`` driven through a DDPCtx
# (methods/rl_agent/training/ddp.py — manual grad allreduce). The model-pass
# tests above pin the *principle*; these pin the *integration*, deliberately
# stepping on the production-only traps:
#   * advantage normalization ON — must use the FULL minibatch, not the shard;
#   * remainder minibatches that do NOT divide by world size (uneven shards,
#     including a rank with an EMPTY shard) — covered by the sum/global-size
#     loss scale + used-flag grad materialization.
# ===========================================================================
_LOOP_NET_COUNTS = [2, 4, 3, 5, 2, 4, 3, 6, 2, 5, 4]   # varied seq lens


def _loop_buffer(n: int) -> dict:
    """Deterministic synthetic PPO buffer over ``n`` mock observations —
    rebuilt identically on every rank (same seeds, pure CPU)."""
    obs_list = [
        make_mock_obs(
            n_nets=c, pads_per_net=2, n_ratsnest_per_net=2,
            is_routing=(i % 2 == 0), current_net_phase=1,
            n_tracks=i % 3, n_vias=i % 2,
        )
        for i, c in enumerate(_LOOP_NET_COUNTS[:n])
    ]
    policy = _tiny_model("cpu")
    with torch.no_grad():
        acts, old_lp, _ = policy.act_and_value(obs_list, deterministic=True)
    rng = np.random.default_rng(0)
    return {
        "obs_list": obs_list,
        "actions": acts.cpu().numpy(),
        "old_log_probs": old_lp.cpu().numpy().astype(np.float32),
        "action_masks": np.ones((n, NUM_ACTION_TYPES), dtype=bool),
        "advantages": rng.standard_normal(n).astype(np.float32),
        "returns": rng.standard_normal(n).astype(np.float32),
    }


def _loop_run(n: int, device: str, ddp: DDPCtx | None, *,
              n_epochs: int, batch_size: int, zero_lr: bool):
    """One ``policy_update_loop`` call from identical init; returns
    (grads, params) on CPU. ``zero_lr`` (SGD lr=0) leaves the last (= only,
    when batch_size=n) minibatch's post-sync gradient in ``p.grad``."""
    buffer = _loop_buffer(n)
    policy = _tiny_model(device)
    opt = (torch.optim.SGD(policy.parameters(), lr=0.0) if zero_lr
           else torch.optim.Adam(policy.parameters(), lr=LR))
    torch.manual_seed(123)   # rank-0 perm stream == single-process perm stream
    policy_update_loop(
        policy, opt, buffer, torch.device(device), algo="ppo",
        n_epochs=n_epochs, batch_size=batch_size,
        normalize_advantages=True, entropy_coef=0.01, ddp=ddp,
    )
    grads = {name: p.grad.detach().cpu().clone()
             for name, p in policy.named_parameters() if p.grad is not None}
    params = {name: p.detach().cpu().clone()
              for name, p in policy.named_parameters()}
    return grads, params


# (case kwargs, result key) — one spawn runs all three:
#   grad:      single minibatch, 1-step gradient (SGD lr=0);
#   remainder: N=11 / batch 4 → last minibatch 3, odd vs WORLD=2 (shards 2/1);
#   empty:     N=9 / batch 4 → last minibatch 1 → rank 1's shard is EMPTY
#              (also hits the numel<=1 normalization skip, full-mb condition).
_LOOP_CASES = {
    "grads": dict(n=8, n_epochs=1, batch_size=8, zero_lr=True),
    "params_remainder": dict(n=11, n_epochs=N_STEPS, batch_size=4, zero_lr=False),
    "params_empty_shard": dict(n=9, n_epochs=2, batch_size=4, zero_lr=False),
}


def _loop_ddp_worker(rank: int, backend: str, rdv_file: str, out_dir: str):
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(
        backend, init_method=f"file://{rdv_file}", rank=rank,
        world_size=WORLD)
    ctx = DDPCtx(rank, WORLD, device)
    result = {}
    for key, case in _LOOP_CASES.items():
        grads, params = _loop_run(case["n"], device, ctx,
                                  n_epochs=case["n_epochs"],
                                  batch_size=case["batch_size"],
                                  zero_lr=case["zero_lr"])
        result[key] = {"grads": grads, "params": params}
    if rank == 0:
        torch.save(result, f"{out_dir}/loop_{backend}.pt")
    dist.destroy_process_group()


def _loop_spawn_and_load(backend: str, tmp_dir) -> dict:
    mp.spawn(_loop_ddp_worker,
             args=(backend, str(tmp_dir / f"loop_rdv_{backend}"), str(tmp_dir)),
             nprocs=WORLD, join=True)
    return torch.load(tmp_dir / f"loop_{backend}.pt", weights_only=True)


@pytest.fixture(scope="module")
def loop_gloo_result(tmp_path_factory):
    return _loop_spawn_and_load(
        "gloo", tmp_path_factory.mktemp("ddp_loop_gloo"))


def test_update_loop_grad_equivalence_gloo(loop_gloo_result):
    ref_grads, _ = _loop_run(**_LOOP_CASES["grads"], device="cpu", ddp=None)
    # identical key set: never-used params keep grad=None on BOTH sides
    # (used-flag materialization — optimizer skip semantics preserved)
    _assert_grads_match(loop_gloo_result["grads"]["grads"], ref_grads)


@pytest.mark.parametrize("case", ["params_remainder", "params_empty_shard"])
def test_update_loop_param_equivalence_gloo(loop_gloo_result, case):
    _, ref_params = _loop_run(**_LOOP_CASES[case], device="cpu", ddp=None)
    _assert_params_match(loop_gloo_result[case]["params"], ref_params)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs >=2 CUDA devices")
def test_update_loop_equivalence_nccl(tmp_path):
    result = _loop_spawn_and_load("nccl", tmp_path)
    for case in ("grads", "params_remainder", "params_empty_shard"):
        ref_grads, ref_params = _loop_run(
            **_LOOP_CASES[case], device="cuda:0", ddp=None)
        if case == "grads":
            _assert_grads_match(result[case]["grads"], ref_grads)
        else:
            _assert_params_match(result[case]["params"], ref_params)
