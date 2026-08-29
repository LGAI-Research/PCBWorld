"""Algorithm-neutral minibatch update loop shared by PPO and GRPO.

The per-algorithm update lives in ``algorithms/{ppo,grpo}.py``; this module
holds the *shared* scaffold of that step — teacher-forcing re-eval +
clipped-ratio policy loss + optimizer step — and dispatches the per-minibatch
loss assembly to :func:`methods.rl_agent.algorithms.ppo.ppo_update_step` /
:func:`methods.rl_agent.algorithms.grpo.grpo_update_step`.

Memory safety (OOM auto-recovery). The default path runs the whole logical
minibatch in a single fwd+bwd. If that
raises ``torch.cuda.OutOfMemoryError`` (large boards → long sequences → the
PPO-update attention blows past VRAM), the minibatch is retried with **sorted
1/4-peel gradient accumulation**: the samples are sorted by sequence length, a
chunk is tried in one fwd+bwd, and *on OOM* its longest ``max(len//4, 1)``
samples are peeled into their own chunk and each side recurses — the peeled tail
shrinks (1/4 → 1/16 …) until it fits, while the shorter remainder is tried whole
(so the cheap bulk goes in one). Each chunk's loss is summed over its samples and
divided by the *full* minibatch size, so the accumulated gradient equals the
single-forward mean gradient (identical up to fp reassociation; attention is
masked per row, so a sample's log_prob/entropy/value are independent of how it is
batched). The split uses **only ``len(chunk)`` and the actual OOM signal** — no
memory proxy, no budget/threshold constant that a different transformer size
would invalidate. ``diag/oom_minibatch_rate`` (fraction of minibatches that
needed peeling) is the durable signal that boards are outgrowing VRAM.

Grad-safety note: a chunk that OOMs in the *forward* (the update-attention peak
— the dominant case) commits no gradient before the exception unwinds, so
recursing on the split is exact. A backward-OOM (rare: backward ≈ forward peak)
would leave a partial gradient; the equivalence tests cover the forward case.
See ``tests/test_oom_recovery.py``.

Preemptive planning (opt-in, ``mem_budget=``). When a calibrated
:class:`~methods.rl_agent.training.mem_budget.MemBudgetModel` is passed, each
minibatch is *pre-split* into predicted-peak-fitting chunks BEFORE the forward
(``_planned_minibatch_step``) — same ``_accumulate_chunk`` math, so the
gradient stays identical up to fp reassociation, but no forward is wasted on
an OOM retry. The reactive path above is both the ``mem_budget=None``
default and the conceptual backstop: an OOM despite planning
discards the partial gradient (``zero_grad``) and reruns the whole minibatch
with a transiently halved budget — which, unlike the reactive peel, is exact
even for a backward-OOM. See ``tests/test_mem_budget.py``.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn

from methods.rl_agent.algorithms.grpo import grpo_update_step
from methods.rl_agent.algorithms.ppo import ppo_update_step

if TYPE_CHECKING:
    from methods.rl_agent.models.v1.net import KiCadRLModel
    from methods.rl_agent.training.buffer import FlatBuffer
    from methods.rl_agent.training.ddp import DDPCtx
    from methods.rl_agent.training.mem_budget import MemBudgetModel

Algo = Literal["grpo", "ppo"]


def policy_update_loop(
    policy: KiCadRLModel,
    optimizer: torch.optim.Optimizer,
    buffer: FlatBuffer,
    device: torch.device,
    *,
    algo: Algo,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    entropy_norm: bool = False,
    n_epochs: int = 4,
    batch_size: int = 64,
    max_grad_norm: float = 0.5,
    vf_coef: float = 0.5,
    normalize_advantages: bool = False,
    mem_budget: "MemBudgetModel | None" = None,
    ddp: "DDPCtx | None" = None,
) -> dict[str, float]:
    """PPO-style minibatch update loop, shared by GRPO and PPO.

    GRPO and PPO differ only in:
        - Advantages: GRPO = per-episode group-relative scalar broadcasted
          to every step (no value model). PPO = per-step GAE.
        - Loss: PPO adds a value-loss term ``vf_coef * MSE(new_v, return)``.

    Advantage normalization (when ``normalize_advantages=True``) is done
    **per minibatch** (before any peel), matching SB3 PPO convention.

    Each logical minibatch runs as one fwd+bwd; on ``torch.cuda.OutOfMemoryError``
    it retries with sorted 1/4-peel gradient accumulation (see module docstring) —
    one optimizer step per minibatch, result identical up to fp reassociation.

    ``mem_budget``: a calibrated (``ready``) MemBudgetModel switches minibatch
    execution to preemptive budget-planned chunking (see module docstring);
    ``None`` (default) selects the reactive path above.

    ``ddp``: a :class:`~methods.rl_agent.training.ddp.DDPCtx` switches every
    minibatch to the rank-sharded step (``_ddp_shard_step``): rank-0 perm
    broadcast, FULL-minibatch advantage normalization, shard loss scaled
    ``sum / global_mb_size``, and one gradient SUM-allreduce inserted right
    before ``clip_grad_norm_`` — numerically equivalent to ``ddp=None`` on the
    whole batch (tests/test_ddp_equivalence.py). Every rank must call this
    with the identical buffer and kwargs. ``None`` (default) is the
    single-process path.
    """
    if algo == "ppo" and not policy.use_critic:
        raise ValueError("PPO requires policy.use_critic=True")
    if ddp is not None and mem_budget is not None:
        # DDP already shards VRAM 1/world; per-rank budget calibration for the
        # planner is not implemented (also asserted trainer-side at
        # --update-gpus parse).
        raise ValueError("ddp update does not support mem_budget")

    # DDP obs-strip: workers may receive obs_list=None when walk_flat is
    # carried — the walked= update path never reads obs (tokenizer ignores
    # obs_list when walked= is given).
    obs_list = buffer.get("obs_list")
    actions_np = buffer["actions"]
    old_lp_np = buffer["old_log_probs"]
    masks_np = buffer["action_masks"]
    N = len(obs_list) if obs_list is not None else buffer["walk_flat"]["B"]
    # pointer_masks is optional: a buffer without it gets a (N, 0) array (no
    # masking applied). A 1-D array is promoted to (N, 1) so the policy still
    # blocks the single recorded cand.
    ptr_masks_np = buffer.get(
        "pointer_masks", np.full((N, 0), -1, dtype=np.int64),
    )
    if ptr_masks_np.ndim == 1:
        ptr_masks_np = ptr_masks_np.reshape(-1, 1)
    # mode_masks is optional: absent means no routing-mode masking.
    mode_masks_np: np.ndarray | None = buffer.get("mode_masks")
    # Policy-driven net_select only: None on the env-driven path.
    nvm_np: np.ndarray | None = buffer.get("net_valid_masks")
    adv_np = buffer["advantages"]

    if N == 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }

    actions_t = torch.as_tensor(actions_np, dtype=torch.long, device=device)
    old_lp_t = torch.as_tensor(old_lp_np, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
    ptr_masks_t = torch.as_tensor(ptr_masks_np, dtype=torch.long, device=device)
    # Only create mode_masks tensor when masking is active (not all-True).
    mode_masks_t: torch.Tensor | None = None
    if mode_masks_np is not None and not mode_masks_np.all():
        mode_masks_t = torch.as_tensor(
            mode_masks_np, dtype=torch.bool, device=device,
        )
    nvm_t: torch.Tensor | None = None
    allow_net_select_lp = bool(
        getattr(policy, "policy_net_select", False) and nvm_np is not None,
    )
    if nvm_np is not None:
        nvm_t = torch.as_tensor(nvm_np, dtype=torch.bool, device=device)
    adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=device)

    if algo == "ppo":
        ret_t = torch.as_tensor(
            buffer["returns"], dtype=torch.float32, device=device,
        )

    policy.train()

    # Walk cache: walk (the CPU part of tokenization) is a pure function of the
    # raw obs, and obs is immutable for this update, so this reuses the flat
    # batched walk collect already performed (buffer["walk_flat"], row-aligned
    # with obs_list) via an index-gather per minibatch (no re-walk on entry, no
    # per-sample split/merge). If absent (GRPO), falls back to one batched walk
    # here. The gather result is byte-identical to walking that subset directly
    # (tests/test_walk_cache.py), so nothing downstream changes. The OOM-peel's
    # seq_len sort also comes free from this.
    tokenizer = policy.tokenizer
    walk_flat = buffer.get("walk_flat")
    if walk_flat is None:
        assert obs_list is not None, "buffer needs obs_list or walk_flat"
        walk_flat = tokenizer.walk_timed(obs_list)
    walk_bounds = tokenizer.walk_sample_bounds(walk_flat)
    seq_lens_all = list(walk_flat["seq_lens"])
    oom_events = 0        # total OOM catches this update (a peel may add several)
    n_mb_ooming = 0       # minibatches that hit >=1 OOM -> diag/oom_minibatch_rate
    use_planner = mem_budget is not None and mem_budget.ready
    planned_chunks_sum = 0   # per-minibatch chunk counts (1 on the reactive path)

    sums = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_updates = 0

    for _epoch in range(n_epochs):
        # DDP: rank-0 perm broadcast (no reliance on seed lockstep); rank 0
        # consumes its RNG stream exactly like the single-process path.
        perm = (
            ddp.broadcast_perm(N, device) if ddp is not None
            else torch.randperm(N, device=device)
        )
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            idx_list = idx.cpu().tolist()

            # On the walked= path the tokenizer ignores obs — obs-strip buffers
            # (DDP workers) fill this with a placeholder (None, so misuse fails
            # loudly).
            mb_obs = ([obs_list[i] for i in idx_list] if obs_list is not None
                      else [None] * len(idx_list))

            def mb_walk(positions=None, _idx=idx_list):
                """Walk dict for this minibatch (or its ``positions`` subset) —
                an index-gather from the flat walk (drop-in for per-minibatch
                re-walking)."""
                sel = _idx if positions is None else [_idx[p] for p in positions]
                return tokenizer.gather_walked(walk_flat, walk_bounds, sel)

            mb_act = actions_t[idx]
            mb_old_lp = old_lp_t[idx]
            mb_masks = masks_t[idx]
            mb_ptr_masks = ptr_masks_t[idx]
            mb_mode_masks = mode_masks_t[idx] if mode_masks_t is not None else None
            mb_nvm = nvm_t[idx] if nvm_t is not None else None
            mb_adv = adv_t[idx]

            if normalize_advantages and mb_adv.numel() > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            mb_ret = ret_t[idx] if algo == "ppo" else None
            mb_size = len(idx_list)
            mb_ctx = dict(
                mb_obs=mb_obs, mb_walk=mb_walk, mb_act=mb_act,
                mb_old_lp=mb_old_lp,
                mb_masks=mb_masks, mb_ptr_masks=mb_ptr_masks,
                mb_mode_masks=mb_mode_masks, mb_nvm=mb_nvm, mb_adv=mb_adv,
                mb_ret=mb_ret, mb_size=mb_size,
            )

            mb_ooming = False
            if ddp is not None:
                planned_chunks_sum += 1
                stats, n_oom_mb = _ddp_shard_step(
                    policy, optimizer, device, ddp=ddp,
                    mb_seq=[seq_lens_all[i] for i in idx_list],
                    algo=algo, clip_eps=clip_eps, entropy_coef=entropy_coef,
                    vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    allow_net_select_lp=allow_net_select_lp,
                    entropy_norm=entropy_norm, **mb_ctx,
                )
                if n_oom_mb:
                    oom_events += n_oom_mb
                    mb_ooming = True
            elif use_planner:
                # Preemptive path: split by predicted peak BEFORE the forward;
                # an OOM here restarts the minibatch on a halved budget.
                stats, n_chunks, n_oom_mb = _planned_minibatch_step(
                    policy, optimizer, device, mem_budget=mem_budget,
                    mb_seq=[seq_lens_all[i] for i in idx_list],
                    algo=algo, clip_eps=clip_eps, entropy_coef=entropy_coef,
                    vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    allow_net_select_lp=allow_net_select_lp,
                    entropy_norm=entropy_norm, **mb_ctx,
                )
                planned_chunks_sum += n_chunks
                if n_oom_mb:
                    oom_events += n_oom_mb
                    mb_ooming = True
            else:
                planned_chunks_sum += 1
                stats, n_oom_mb = _reactive_peel_step(
                    policy, optimizer, device,
                    mb_seq=[seq_lens_all[i] for i in idx_list],
                    algo=algo, clip_eps=clip_eps, entropy_coef=entropy_coef,
                    vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    allow_net_select_lp=allow_net_select_lp,
                    entropy_norm=entropy_norm, **mb_ctx,
                )
                if n_oom_mb:
                    oom_events += n_oom_mb
                    mb_ooming = True
                    if oom_events <= 6:
                        warnings.warn(
                            f"CUDA OOM in PPO update (mb_size={mb_size}); retrying "
                            f"with sorted 1/4-peel gradient accumulation "
                            f"(oom_events={oom_events}).",
                            RuntimeWarning, stacklevel=2,
                        )

            if mb_ooming:
                n_mb_ooming += 1
            sums["loss"] += stats["loss"]
            sums["policy_loss"] += stats["policy_loss"]
            sums["value_loss"] += stats["value_loss"]
            sums["entropy"] += stats["entropy"]
            n_updates += 1

    denom = max(n_updates, 1)
    oom_mb_denom = denom
    if ddp is not None:
        # Logging stats: single end-of-update allreduce of the rank-partial
        # sums (each minibatch stat is a sum/global_mb_size shard partial, so
        # the SUM is the exact global value). OOM counters become rank-
        # minibatch totals; the rate stays in [0, 1] over rank-minibatch units.
        reduced = ddp.allreduce_sums({
            **sums,
            "n_mb_ooming": float(n_mb_ooming),
            "oom_events": float(oom_events),
        })
        sums = {k: reduced[k] for k in sums}
        n_mb_ooming = reduced["n_mb_ooming"]
        oom_events = reduced["oom_events"]
        oom_mb_denom = denom * ddp.world_size
    result = {k: v / denom for k, v in sums.items()}
    # Diagnostics: fraction of minibatches that needed OOM peeling. Near 1.0 means
    # every minibatch OOMs even at nominal batch — boards outgrew VRAM.
    result["oom_minibatch_rate"] = n_mb_ooming / oom_mb_denom
    result["oom_events"] = float(oom_events)
    # Mean planned chunks per minibatch (1.0 = every minibatch fit whole).
    result["planned_chunks_per_mb"] = planned_chunks_sum / denom
    if mem_budget is not None:
        mem_budget.maybe_refit()   # fold this update's chunk measurements in
    return result


def _finalize_peel_stats(acc: dict, mb_size: int) -> dict[str, float]:
    """Accumulated ``_peel_accumulate``/``_accumulate_chunk`` sums -> per-minibatch
    loss stats. Loss-family entries are already sum/``mb_size`` partials (sum to
    the full-minibatch mean); entropy is a raw sum that normalizes here."""
    return {
        "loss": acc["loss"], "policy_loss": acc["policy_loss"],
        "value_loss": acc["value_loss"],
        "entropy": acc["entropy_sum"] / mb_size,
    }


def _reactive_peel_step(
    policy, optimizer, device, *, mb_seq, algo, clip_eps, entropy_coef,
    vf_coef, max_grad_norm, allow_net_select_lp, entropy_norm=False, **mb_ctx,
) -> tuple[dict[str, float], int]:
    """Default single-process minibatch step (a sibling of ``_ddp_shard_step``
    / ``_planned_minibatch_step`` — same dispatch shape): whole minibatch in
    one fwd+bwd; on CUDA OOM, retry reactively with the sorted 1/4-peel
    gradient accumulation.

    Returns ``(stats, n_oom)`` — ``n_oom`` is 0 on the no-OOM fast path, else
    1 + the peel's internal OOM count.
    """
    try:
        # no-OOM path: whole minibatch in one fwd+bwd.
        stats = _fixed_batch_step(
            policy, optimizer, device, algo=algo, clip_eps=clip_eps,
            entropy_coef=entropy_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            allow_net_select_lp=allow_net_select_lp,
            entropy_norm=entropy_norm, **mb_ctx,
        )
        return stats, 0
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mb_size = mb_ctx["mb_size"]
        # sort minibatch-local positions ascending by sequence length
        order = sorted(range(mb_size), key=lambda p: mb_seq[p])
        optimizer.zero_grad(set_to_none=True)   # drop the failed step's grad
        acc = {"loss": 0.0, "policy_loss": 0.0,
               "value_loss": 0.0, "entropy_sum": 0.0}
        n_oom = _peel_accumulate(
            policy, order, acc, device=device, algo=algo,
            clip_eps=clip_eps, entropy_coef=entropy_coef, vf_coef=vf_coef,
            allow_net_select_lp=allow_net_select_lp,
            entropy_norm=entropy_norm, **mb_ctx,
        )
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        return _finalize_peel_stats(acc, mb_size), 1 + n_oom


def _ddp_shard_step(
    policy, optimizer, device, *, ddp, mb_seq, algo, clip_eps, entropy_coef,
    vf_coef, max_grad_norm, allow_net_select_lp, entropy_norm=False, **mb_ctx,
) -> tuple[dict[str, float], int]:
    """Rank-sharded minibatch step (manual DDP; exactly one grad allreduce).

    Every rank holds the FULL minibatch tensors (broadcast buffer + rank-0
    perm; advantage normalization already ran on the full minibatch) and
    processes only its strided shard through the ``_accumulate_chunk``
    sum/``mb_size`` math — the SUM-allreduced gradient therefore equals the
    single-process full-batch mean gradient even when a remainder minibatch
    does not divide by world size (an empty shard contributes zeros but still
    joins the allreduce + step, keeping optimizer state in lockstep).

    ``sync_grads`` runs BEFORE ``clip_grad_norm_`` so every rank clips the
    identical global gradient (clip-then-sync would break equivalence). A CUDA
    OOM inside the shard falls back to the rank-local sorted 1/4-peel —
    per-rank chunk structure is free to differ because the sync point is the
    single fixed allreduce per minibatch (why the DDP module wrapper is not
    used; see methods/rl_agent/training/ddp.py).

    Returns ``(stats, n_oom)``; stats are rank-partial sums, finalized by one
    allreduce at the end of ``policy_update_loop``.
    """
    mb_size = mb_ctx["mb_size"]
    shard = ddp.shard_positions(mb_size)
    shard.sort(key=lambda p: mb_seq[p])   # ascending seq_len: peel-ready order
    optimizer.zero_grad(set_to_none=True)
    acc = {"loss": 0.0, "policy_loss": 0.0,
           "value_loss": 0.0, "entropy_sum": 0.0}
    n_oom = 0
    if shard:
        n_oom = _peel_accumulate(
            policy, shard, acc, device=device, algo=algo, clip_eps=clip_eps,
            entropy_coef=entropy_coef, vf_coef=vf_coef,
            allow_net_select_lp=allow_net_select_lp,
            entropy_norm=entropy_norm, **mb_ctx,
        )
    ddp.sync_grads(policy)
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return _finalize_peel_stats(acc, mb_size), n_oom


def _planned_minibatch_step(
    policy, optimizer, device, *, mem_budget, mb_seq, algo, clip_eps,
    entropy_coef, vf_coef, max_grad_norm, allow_net_select_lp,
    entropy_norm=False, **mb_ctx,
) -> tuple[dict[str, float], int, int]:
    """Budget-planned minibatch step: split BEFORE the forward, restart on OOM.

    ``plan_chunks`` partitions the minibatch by predicted peak; a single chunk
    degenerates to ``_fixed_batch_step`` (identical to the reactive no-OOM
    path), otherwise each chunk runs through ``_accumulate_chunk`` (same
    sum/``mb_size`` math — gradient equals the single forward up to fp
    reassociation) with its real peak fed back via ``mem_budget.observe``.

    OOM backstop (exact for forward- AND backward-OOM): discard the partial
    gradient, halve the budget *for this minibatch only*, replan and rerun the
    whole minibatch from scratch. Recovery runs OUTSIDE the except handler —
    the in-flight exception's traceback pins the failed forward's frames
    (graph + activations), which only release when the handler exits. A
    single-sample chunk that still OOMs re-raises (board too big for VRAM),
    matching the reactive peel.

    Returns ``(stats, n_chunks, n_oom)``.
    """
    mb_size = mb_ctx["mb_size"]
    measure = torch.cuda.is_available()
    if measure:
        from methods.rl_agent.training import mem_budget as _mb_mod
    limit = mem_budget.capacity()
    n_oom = 0
    while True:
        chunks = mem_budget.plan_chunks(mb_seq, limit=limit)
        try:
            if len(chunks) == 1:
                stats = _fixed_batch_step(
                    policy, optimizer, device, algo=algo, clip_eps=clip_eps,
                    entropy_coef=entropy_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    allow_net_select_lp=allow_net_select_lp,
                    entropy_norm=entropy_norm, **mb_ctx,
                )
                return stats, 1, n_oom
            optimizer.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "policy_loss": 0.0,
                   "value_loss": 0.0, "entropy_sum": 0.0}
            for chunk in chunks:
                if measure:
                    base = _mb_mod.begin_measured_region()
                _accumulate_chunk(
                    policy, chunk, acc, device=device, algo=algo,
                    clip_eps=clip_eps, entropy_coef=entropy_coef,
                    vf_coef=vf_coef,
                    allow_net_select_lp=allow_net_select_lp,
                    entropy_norm=entropy_norm, **mb_ctx,
                )
                if measure:
                    mem_budget.observe(
                        len(chunk), max(mb_seq[p] for p in chunk),
                        _mb_mod.end_measured_region(base),
                    )
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            return _finalize_peel_stats(acc, mb_size), len(chunks), n_oom
        except torch.cuda.OutOfMemoryError:
            if all(len(c) == 1 for c in chunks):
                raise   # a single sample OOMs -> board too big for VRAM
            # fall through: recover outside the handler (see docstring)
        n_oom += 1
        optimizer.zero_grad(set_to_none=True)   # discard any partial gradient
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        limit /= 2.0   # transient — next minibatch replans at full capacity
        if n_oom <= 3:
            warnings.warn(
                f"CUDA OOM despite planned chunking "
                f"(mb_size={mb_size}, chunks={len(chunks)}); restarting the "
                f"minibatch on a halved budget (n_oom={n_oom}).",
                RuntimeWarning, stacklevel=2,
            )


def _fixed_batch_step(
    policy, optimizer, device, *, algo, clip_eps, entropy_coef, vf_coef,
    max_grad_norm, allow_net_select_lp, entropy_norm=False,
    mb_obs, mb_walk, mb_act, mb_old_lp, mb_masks, mb_ptr_masks, mb_mode_masks,
    mb_nvm, mb_adv, mb_ret, mb_size,
) -> dict[str, float]:
    """Single-forward minibatch step (the no-OOM path).
    ``mb_walk`` is the caller's gather closure —
    ``mb_walk()`` returns this minibatch's walk dict (flat-walk index-gather).
    Returns per-minibatch loss stats."""
    new_lp, entropy, new_values = policy.evaluate_actions_and_value(
        mb_obs, mb_act,
        action_masks=mb_masks, pointer_masks=mb_ptr_masks,
        mode_mask=mb_mode_masks, net_valid_mask=mb_nvm,
        allow_net_select_lp=allow_net_select_lp,
        walked=mb_walk(), entropy_norm=entropy_norm,
    )
    ratio = torch.exp(new_lp - mb_old_lp)
    surr1 = ratio * mb_adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy_loss = -entropy_coef * entropy.mean()
    if algo == "ppo":
        loss, value_loss = ppo_update_step(
            policy_loss, entropy_loss, new_values, mb_ret, vf_coef,
        )
    else:
        loss, value_loss = grpo_update_step(policy_loss, entropy_loss, device)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "loss": loss.item(), "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(), "entropy": entropy.mean().item(),
    }


def _peel_accumulate(policy, positions: list[int], acc: dict, **kw) -> int:
    """Process ``positions`` (minibatch-local, **sorted ascending by seq_len**) as
    one fwd+bwd, accumulating grad. On ``OutOfMemoryError`` peel the longest
    ``max(len//4, 1)`` samples (the tail — they sit at the end of the sorted list)
    into their own chunk and recurse on the peeled quarter, then on the shorter
    remainder. Uses only ``len(positions)`` and the actual OOM — no memory proxy
    or budget. Returns the number of OOM events (for diagnostics)."""
    try:
        _accumulate_chunk(policy, positions, acc, **kw)
        return 0
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(positions) <= 1:
            raise   # a single sample OOMs → board too big for VRAM, unrecoverable
        cut = max(len(positions) // 4, 1)
        return (
            1
            + _peel_accumulate(policy, positions[-cut:], acc, **kw)   # longest 1/4
            + _peel_accumulate(policy, positions[:-cut], acc, **kw)   # remainder
        )


def _accumulate_chunk(
    policy, positions: list[int], acc: dict, *, device, algo, clip_eps,
    entropy_coef, vf_coef, allow_net_select_lp, entropy_norm=False,
    mb_obs, mb_walk, mb_act, mb_old_lp, mb_masks, mb_ptr_masks, mb_mode_masks,
    mb_nvm, mb_adv, mb_ret, mb_size,
) -> None:
    """One chunk fwd+bwd, accumulating grad. Loss is summed over the chunk's
    samples and divided by the **full minibatch size** (``mb_size``), so summing
    the per-chunk gradients over a partition equals the single-forward mean
    gradient. ``positions`` are minibatch-local indices into the ``mb_*`` tensors."""
    sub = torch.as_tensor(positions, dtype=torch.long, device=device)
    new_lp, entropy, new_values = policy.evaluate_actions_and_value(
        [mb_obs[j] for j in positions], mb_act[sub],
        action_masks=mb_masks[sub], pointer_masks=mb_ptr_masks[sub],
        mode_mask=mb_mode_masks[sub] if mb_mode_masks is not None else None,
        net_valid_mask=mb_nvm[sub] if mb_nvm is not None else None,
        allow_net_select_lp=allow_net_select_lp,
        walked=mb_walk(positions), entropy_norm=entropy_norm,
    )
    ratio = torch.exp(new_lp - mb_old_lp[sub])
    adv = mb_adv[sub]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_term = -torch.min(surr1, surr2).sum() / mb_size
    entropy_term = -entropy_coef * entropy.sum() / mb_size
    if algo == "ppo":
        value_term = 0.5 * (new_values - mb_ret[sub]).pow(2).sum() / mb_size
        chunk_loss = policy_term + vf_coef * value_term + entropy_term
    else:
        value_term = torch.zeros((), device=device)
        chunk_loss = policy_term + entropy_term
    chunk_loss.backward()
    acc["loss"] += chunk_loss.item()
    acc["policy_loss"] += policy_term.item()
    acc["value_loss"] += value_term.item()
    acc["entropy_sum"] += entropy.sum().item()
