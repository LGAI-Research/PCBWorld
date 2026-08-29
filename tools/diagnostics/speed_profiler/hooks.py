"""Main-proc runtime monkeypatch toggles (context managers). Zero base edits.

All of these run in the MAIN process (update + main-side rollout glue), so a
runtime monkeypatch reaches them directly — unlike the worker loop, which needs
the forkserver-preload :mod:`.worker_shim`. Each toggle restores the original on
exit; numerics/RNG are untouched (timers only).
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from tools.diagnostics.speed_profiler.instrument import CudaEventAccumulator

PC = time.perf_counter

# torch.profiler step callback — when the driver's tracer sets this, the
# mirrors call it once per unit step (rollout: env-step, update: minibatch).
# None means zero cost.
TRACE_CB = None


def _trace_step():
    cb = TRACE_CB
    if cb is not None:
        cb()


@contextmanager
def tokenizer_timer():
    """Enable the base tokenizer's dormant ``_BATCHED_TIMER_HOOK`` (walk[CPU] /
    h2d_encode / scatter / slot_emb) and yield its bucket. The hook already exists
    in base (methods/rl_agent/models/v1/tokenizer.py) — this only toggles it.

    ``walk`` is CPU-truthful (perf_counter around pure Python) — the only bucket
    the driver consumes (rollout forward_split.walk_cpu); h2d_encode/scatter/
    slot_emb wrap async launches so they are launch-time.
    """
    import methods.rl_agent.models.v1.tokenizer as tok
    prev = tok._BATCHED_TIMER_HOOK
    bucket: dict[str, list[float]] = {}
    tok._BATCHED_TIMER_HOOK = bucket
    try:
        yield bucket
    finally:
        tok._BATCHED_TIMER_HOOK = prev


@contextmanager
def transformer_pass_recorder(acc=None):
    """Record (B, L) of every ``KiCadRLModel._run_transformer`` call, so MFU FLOPs
    use MEASURED token counts (padded ``n_state_max`` varies per batch and flips
    with board size). Yields a list of (B, L) tuples. Rollout = 3 passes (L, L+1,
    L+2) per step; update = 2 passes (L, L+2) per minibatch.

    If a :class:`CudaEventAccumulator` ``acc`` is given, each pass is also bracketed
    with a cuda-event ``fwd_pass`` span — this is the CORRECT GPU-active (duty)
    numerator, because it excludes the CPU tokenizer walk that precedes the
    transformer (bracketing the whole ``evaluate`` call would fold the walk's
    GPU-idle gap into the span and inflate duty toward 1.0)."""
    from methods.rl_agent.models.v1.net import KiCadRLModel
    calls: list[tuple[int, int]] = []
    orig = KiCadRLModel._run_transformer

    def rec(self, embs, n_state, key_padding_mask, slot_ids=None, **kwargs):
        # **kwargs passes through return_cache (incremental decode). The tiny
        # appended-token decodes (_decode_appended) are NOT recorded — their
        # FLOPs are ~n_new/L of a pass (negligible; MFU slightly understated).
        calls.append((int(embs.shape[0]), int(embs.shape[1])))
        if acc is not None:
            with acc.span("fwd_pass"):
                return orig(self, embs, n_state, key_padding_mask,
                            slot_ids=slot_ids, **kwargs)
        return orig(self, embs, n_state, key_padding_mask,
                    slot_ids=slot_ids, **kwargs)

    KiCadRLModel._run_transformer = rec
    try:
        yield calls
    finally:
        KiCadRLModel._run_transformer = orig


@contextmanager
def update_decomp():
    """Monkeypatch ``_common._fixed_batch_step`` with a timed mirror that splits
    the minibatch step into evaluate / backward / clip / step, and brackets the
    GPU forward+backward with cuda events (GPU-truthful duty). Yields a handle
    ``{"bucket": ..., "acc": CudaEventAccumulator}``.

    perf_counter splits are launch-time for the GPU ops (evaluate/backward) — use
    ``acc`` (cuda-event) for duty; the perf_counter split still bounds CPU glue
    (clip/step include the .item() sync so the coarse total stays truthful). The
    OOM-peel path (_peel_accumulate) is NOT timed here (OOM rate 0 on L40; the
    caller records oom_minibatch_rate separately).

    ``_accumulate_chunk`` (the mem_budget planner's split-minibatch path) also
    gets a timed mirror — without it, split-heavy ON runs under-count evaluate
    and the waterfall update-closure assert fails negative (fwd cuda-events
    count every pass). Its clip/step happen in the caller and stay in the resid.

    Mirrors of methods/rl_agent/algorithms/_common.py:{_fixed_batch_step,
    _accumulate_chunk} (flat walk-cache: mb_walk = gather closure,
    walked=mb_walk()/mb_walk(positions), entropy_norm pass-through) — re-sync
    if those bodies change.
    """
    import torch
    import torch.nn as nn
    import methods.rl_agent.algorithms._common as C
    from methods.rl_agent.algorithms.ppo import ppo_update_step
    from methods.rl_agent.algorithms.grpo import grpo_update_step
    from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer

    bucket = {"evaluate_ms": [], "backward_ms": [], "clip_ms": [], "step_ms": [],
              # entry walk = the fallback batched walk (walk_timed) at
              # policy_update_loop's entry — an empty list (0 calls) when
              # collect carries walk_flat (PPO carry), 1 call when uncached
              # (GRPO/mock). Per-minibatch gather is included in evaluate wall.
              "entry_walk_ms": [],
              # DDP-only (only when --update-gpus>1; empty list on single-GPU → omitted from JSON).
              "ddp_sync_ms": [], "ddp_perm_ms": [], "ddp_bcast_ms": []}
    acc = CudaEventAccumulator()
    orig = C._fixed_batch_step

    # Entry-walk instrumentation: the only place walk_timed is called inside
    # the update phase is policy_update_loop's uncached fallback (once at
    # entry). The walk inside forward is skipped via the walked= cache, so
    # the two do not overlap.
    orig_walk_timed = BatchedStateTokenizer.walk_timed

    def timed_entry_walk(self, obs_list):
        t = PC()
        out = orig_walk_timed(self, obs_list)
        bucket["entry_walk_ms"].append((PC() - t) * 1000)
        return out

    BatchedStateTokenizer.walk_timed = timed_entry_walk

    # DDP-specific cost (when present): sync_grads (per-minibatch flat-buffer
    # allreduce), broadcast_perm (per-epoch rank sync), dispatch_update
    # (per-iter buffer broadcast). The ddp module is only loaded/used when
    # --update-gpus>1, so on single-GPU this patch stays installed but is
    # never called, leaving the buckets empty.
    _ddp_restore = []
    try:
        from methods.rl_agent.training.ddp import DDPCtx, DDPUpdateGroup
        _o_sync, _o_perm = DDPCtx.sync_grads, DDPCtx.broadcast_perm
        _o_disp = DDPUpdateGroup.dispatch_update

        def _t_sync(self, policy):
            t = PC(); _o_sync(self, policy)
            bucket["ddp_sync_ms"].append((PC() - t) * 1000)

        def _t_perm(self, n, device):
            t = PC(); o = _o_perm(self, n, device)
            bucket["ddp_perm_ms"].append((PC() - t) * 1000); return o

        def _t_disp(self, buffer, update_kwargs, lr):
            t = PC(); o = _o_disp(self, buffer, update_kwargs, lr)
            bucket["ddp_bcast_ms"].append((PC() - t) * 1000); return o

        DDPCtx.sync_grads, DDPCtx.broadcast_perm = _t_sync, _t_perm
        DDPUpdateGroup.dispatch_update = _t_disp
        _ddp_restore = [(DDPCtx, "sync_grads", _o_sync),
                        (DDPCtx, "broadcast_perm", _o_perm),
                        (DDPUpdateGroup, "dispatch_update", _o_disp)]
    except Exception:
        pass

    def timed_fixed_batch_step(
        policy, optimizer, device, *, algo, clip_eps, entropy_coef, vf_coef,
        max_grad_norm, allow_net_select_lp, entropy_norm=False,
        mb_obs, mb_walk, mb_act, mb_old_lp, mb_masks, mb_ptr_masks,
        mb_mode_masks, mb_nvm, mb_adv, mb_ret, mb_size,
    ):
        # No acc.span here: evaluate includes the CPU tokenizer merge/encode.
        # The GPU-active fwd time is captured by the transformer_pass_recorder's
        # per-_run_transformer 'fwd_pass' spans (same acc). perf_counter here is
        # the CPU-inclusive forward WALL (merge + encode + transformer).
        t = PC()
        new_lp, entropy, new_values = policy.evaluate_actions_and_value(
            mb_obs, mb_act, action_masks=mb_masks, pointer_masks=mb_ptr_masks,
            mode_mask=mb_mode_masks, net_valid_mask=mb_nvm,
            allow_net_select_lp=allow_net_select_lp,
            walked=mb_walk(), entropy_norm=entropy_norm,
        )
        bucket["evaluate_ms"].append((PC() - t) * 1000)

        ratio = torch.exp(new_lp - mb_old_lp)
        surr1 = ratio * mb_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy_coef * entropy.mean()
        if algo == "ppo":
            loss, value_loss = ppo_update_step(policy_loss, entropy_loss, new_values, mb_ret, vf_coef)
        else:
            loss, value_loss = grpo_update_step(policy_loss, entropy_loss, device)

        optimizer.zero_grad()
        t = PC()
        with acc.span("backward"):
            loss.backward()
        bucket["backward_ms"].append((PC() - t) * 1000)
        t = PC()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        bucket["clip_ms"].append((PC() - t) * 1000)
        t = PC()
        optimizer.step()
        bucket["step_ms"].append((PC() - t) * 1000)
        _trace_step()
        return {
            "loss": loss.item(), "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(), "entropy": entropy.mean().item(),
        }

    orig_chunk = C._accumulate_chunk

    def timed_accumulate_chunk(
        policy, positions, acc_dict, *, device, algo, clip_eps, entropy_coef,
        vf_coef, allow_net_select_lp, entropy_norm=False,
        mb_obs, mb_walk, mb_act, mb_old_lp, mb_masks, mb_ptr_masks,
        mb_mode_masks, mb_nvm, mb_adv, mb_ret, mb_size,
    ):
        sub = torch.as_tensor(positions, dtype=torch.long, device=device)
        t = PC()
        new_lp, entropy, new_values = policy.evaluate_actions_and_value(
            [mb_obs[j] for j in positions], mb_act[sub],
            action_masks=mb_masks[sub], pointer_masks=mb_ptr_masks[sub],
            mode_mask=mb_mode_masks[sub] if mb_mode_masks is not None else None,
            net_valid_mask=mb_nvm[sub] if mb_nvm is not None else None,
            allow_net_select_lp=allow_net_select_lp,
            walked=mb_walk(positions), entropy_norm=entropy_norm,
        )
        bucket["evaluate_ms"].append((PC() - t) * 1000)
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
        t = PC()
        with acc.span("backward"):
            chunk_loss.backward()
        bucket["backward_ms"].append((PC() - t) * 1000)
        acc_dict["loss"] += chunk_loss.item()
        acc_dict["policy_loss"] += policy_term.item()
        acc_dict["value_loss"] += value_term.item()
        acc_dict["entropy_sum"] += entropy.sum().item()

    C._fixed_batch_step = timed_fixed_batch_step
    C._accumulate_chunk = timed_accumulate_chunk
    try:
        yield {"bucket": bucket, "acc": acc}
    finally:
        C._fixed_batch_step = orig
        C._accumulate_chunk = orig_chunk
        BatchedStateTokenizer.walk_timed = orig_walk_timed
        for cls, attr, fn in _ddp_restore:
            setattr(cls, attr, fn)


@contextmanager
def rollout_decomp():
    """Inline, sum-closed decomposition of the collect loop.

    Monkeypatches ``methods.rl_agent.rollout.primitive.iter_rollout`` with a
    timed MIRROR of the base generator (incl. the mem_budget budgeted_forward
    branch + collect-time walk hoist/``walked=`` passthrough — re-sync if
    primitive.py changes) plus a ``SubprocDecoderVecEnv.reset_batch`` timer.
    Buckets (seconds, accumulated over the phase):

      mask_ipc   gather_mask_arrays + tensorize            (inline, not a probe)
      forward    act/act_and_value call + .cpu()x3 wall     (walk+GPU+launch+sync)
      step       step_async_selective + step_wait_selective (inline step barrier)
      advance    obs_by_slot update loop
      between    gap between yields = collector python + reset (reset is timed separately)
      reset      envs.reset_batch wall

    forward's internal split comes from the concurrently-active
    tokenizer_timer (walk) + transformer_pass_recorder(acc) (GPU-event):
    launch/sync residual = forward − walk − gpu_event. Sum:
    collect ≈ mask+forward+step+advance+between. Numerics are unchanged
    (timers only; control flow is identical)."""
    import numpy as np
    import methods.rl_agent.rollout.primitive as P
    from pcb_world.vec.backends.base import VecBackend
    from pcb_world.vec.backends.subproc import SubprocDecoderVecEnv
    from methods.rl_agent.rollout.primitive import StepBatch

    bucket = {"mask_ipc": 0.0, "forward": 0.0, "step": 0.0,
              "advance": 0.0, "between": 0.0, "reset": 0.0, "n_steps": 0}
    orig_iter = P.iter_rollout
    orig_reset_batch = SubprocDecoderVecEnv.reset_batch

    def timed_reset_batch(self, indices, seeds=None):
        t = PC()
        out = orig_reset_batch(self, indices, seeds)
        bucket["reset"] += PC() - t
        return out

    def timed_iter_rollout(envs, policy, device, obs_by_slot, active, done, *,
                           want_value, deterministic=False, max_steps,
                           mem_budget=None):
        from methods.rl_agent.policy.agent import (
            KiCadRLAgent, gather_mask_arrays, mask_arrays_to_tensors,
        )
        use_vec = isinstance(envs, VecBackend)
        policy_ns = bool(getattr(policy, "policy_net_select", False))
        agent = None
        if not want_value:
            agent = (policy if isinstance(policy, KiCadRLAgent)
                     else KiCadRLAgent(policy, device=device))
            device = agent.device

        t_resume = None
        for _step in range(max_steps):
            if t_resume is not None:
                bucket["between"] += PC() - t_resume
            live = [slot for slot in active if not done.get(slot, False)]
            if not live:
                return
            obs_live = [obs_by_slot[slot] for slot in live]

            t = PC()
            masks, ptr_masks, mode_masks, nvm, off_masks = gather_mask_arrays(
                envs, live, policy_net_select=policy_ns, obs_list=obs_live,
            )
            mask_t, ptr_t, mode_t, nvm_kwargs = mask_arrays_to_tensors(
                masks, ptr_masks, mode_masks, nvm, device, off_masks=off_masks,
                mode_none_if_all_true=want_value,
            )
            bucket["mask_ipc"] += PC() - t

            t = PC()
            step_walk = None
            if want_value:
                tok = getattr(getattr(policy, "model", policy), "tokenizer", None)
                if tok is not None:
                    step_walk = tok.walk_timed(obs_live)
            if mem_budget is not None and mem_budget.ready:
                acts_t, logp_t, vals_t = P.budgeted_forward(
                    policy if want_value else agent, obs_live,
                    mask_t, ptr_t, mode_t, nvm_kwargs, mem_budget,
                    want_value=want_value, deterministic=deterministic,
                    walk=step_walk,
                )
                actions_np = acts_t.cpu().numpy()
                log_probs_np = logp_t.cpu().numpy() if want_value else None
                values_np = vals_t.cpu().numpy() if want_value else None
            elif want_value:
                walk_kw = {"walked": step_walk} if step_walk is not None else {}
                actions, log_probs, values = policy.act_and_value(
                    obs_live, action_masks=mask_t, pointer_masks=ptr_t,
                    mode_mask=mode_t, **walk_kw, **nvm_kwargs,
                )
                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()
            else:
                acts_t, _logp = agent.act(
                    obs_live, action_masks=mask_t, pointer_masks=ptr_t,
                    mode_mask=mode_t, deterministic=deterministic, **nvm_kwargs,
                )
                actions_np = acts_t.cpu().numpy()
                log_probs_np = None
                values_np = None
            bucket["forward"] += PC() - t

            t = PC()
            if use_vec:
                envs.step_async_selective(live, actions_np)
                obs_next, rewards, terms, truncs, infos = envs.step_wait_selective()
            else:
                obs_next, rew_l, term_l, trunc_l, infos = [], [], [], [], []
                for k, i in enumerate(live):
                    o, r, te, tr, info = envs[i].step(actions_np[k])
                    obs_next.append(o); rew_l.append(r)
                    term_l.append(te); trunc_l.append(tr); infos.append(info)
                rewards = np.array(rew_l, dtype=np.float64)
                terms = np.array(term_l, dtype=bool)
                truncs = np.array(trunc_l, dtype=bool)
            bucket["step"] += PC() - t

            t = PC()
            for k, slot in enumerate(live):
                obs_next[k]["board_static"] = obs_live[k]["board_static"]
                obs_by_slot[slot] = obs_next[k]
            bucket["advance"] += PC() - t
            bucket["n_steps"] += 1
            _trace_step()

            t_resume = PC()
            # walk=step_walk: carries the batched walk, same as base — without
            # this the collector's walk cache is empty and update re-walks,
            # so the profile would fail to measure walk-carry.
            yield StepBatch(
                live=live, obs=obs_live, actions=actions_np,
                log_probs=log_probs_np, values=values_np, obs_next=obs_next,
                rewards=rewards, terminateds=terms, truncateds=truncs,
                infos=infos, action_masks=masks, pointer_masks=ptr_masks,
                mode_masks=mode_masks, net_valid_masks=nvm,
                walk=step_walk,
            )
        # Tail gap: the caller's bookkeeping+reset after the last yield (max_steps
        # truncation piles the reset onto the last step) — omitting this would
        # make the collector bucket go negative.
        if t_resume is not None:
            bucket["between"] += PC() - t_resume

    P.iter_rollout = timed_iter_rollout
    SubprocDecoderVecEnv.reset_batch = timed_reset_batch
    # Also patch consumers that import iter_rollout into their own namespace:
    # collect.py (train collector) + rollout/transformer.py (EVAL path)
    import methods.rl_agent.training.collect as C
    import methods.rl_agent.rollout.transformer as T
    orig_collect_ref = C.iter_rollout
    orig_transformer_ref = T.iter_rollout
    C.iter_rollout = timed_iter_rollout
    T.iter_rollout = timed_iter_rollout
    try:
        yield bucket
    finally:
        P.iter_rollout = orig_iter
        C.iter_rollout = orig_collect_ref
        T.iter_rollout = orig_transformer_ref
        SubprocDecoderVecEnv.reset_batch = orig_reset_batch


@contextmanager
def eval_pool_spawn_timer():
    """Time eval's own pool spawns. ``eval_transformer`` builds a fresh
    ``make_decoder_env_pool(boards[0], n_envs)`` per call (and closes it after),
    so each val set pays a full n_envs-worker spawn. Patch the factory attr
    (eval/rollout/rl.py imports it at call time) and yield the list of per-call
    spawn seconds."""
    import methods.rl_agent.wrappers.factory as F
    spawns: list[float] = []
    orig = F.make_decoder_env_pool

    def timed(*a, **k):
        t = PC()
        pool = orig(*a, **k)
        spawns.append(PC() - t)
        return pool

    F.make_decoder_env_pool = timed
    try:
        yield spawns
    finally:
        F.make_decoder_env_pool = orig


def summarize_update(handle: dict) -> dict:
    """Reduce an update_decomp handle to a JSON block. ``gpu_active_ms`` (cuda-event
    ``fwd_pass`` + ``backward``) is the duty numerator; ``perf_counter_ms.evaluate``
    is the CPU-inclusive forward wall (walk + encode + transformer), so
    ``evaluate - fwd_pass`` ~= the redundant CPU tokenizer walk. Call once after the
    update phase (``acc.collect`` syncs and reads the events)."""
    bucket = handle["bucket"]
    gpu = handle["acc"].collect()["per_region_ms"]  # {'fwd_pass':.., 'backward':..}
    def _sum(k):
        xs = bucket.get(k, [])
        return round(sum(xs), 2) if xs else 0.0
    n_mb = len(bucket.get("evaluate_ms", []))
    out = {
        "n_minibatches_timed": n_mb,
        "perf_counter_ms": {k.replace("_ms", ""): _sum(k) for k in
                            ("evaluate_ms", "backward_ms", "clip_ms", "step_ms")},
        "gpu_active_ms": {k: round(v, 2) for k, v in gpu.items()},
        # entry walk — 0 with collect walk-carry (PPO), one batched call under the uncached fallback (GRPO).
        "entry_walk_ms": _sum("entry_walk_ms"),
        "note": "gpu_active_ms=cuda-event GPU (fwd_pass=transformer passes, backward); "
                "perf_counter.evaluate is CPU-inclusive forward wall (incl. per-mb walk gather); "
                "entry_walk=uncached fallback batched walk once per update (0 with collect walk-carry)",
    }
    # DDP-only (present only when there were calls): None on a single-GPU profile → shown as "—" in waterfall.
    if bucket.get("ddp_sync_ms"):
        out["ddp_ms"] = {
            "sync": _sum("ddp_sync_ms"), "perm": _sum("ddp_perm_ms"),
            "bcast": _sum("ddp_bcast_ms"),
        }
    return out
