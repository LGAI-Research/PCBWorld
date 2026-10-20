"""Shared rollout CORE primitive — the one act+step transition generator.

:func:`iter_rollout` is the single per-step loop shared by the eval rollout
(``methods/rl_agent/rollout/transformer.py``) and the training collectors
(``methods/rl_agent/training/collect.py``); ≈ SB3 ``collect_rollouts``.
Contract:

* **The caller owns ``done``** — the primitive never marks a slot done.
  Eval marks env-done + guard-triggered truncations, GRPO marks env-done,
  PPO leaves it empty (fixed n_steps, autoreset across episode boundaries).
* **``obs_by_slot`` is shared mutable state** — the primitive re-reads it at
  the top of every iteration and advances it (``obs_by_slot[slot] = obs_next``)
  *before* yielding, so an adapter may overwrite reset slots between
  iterations (PPO reset-obs re-injection protocol).
* **The distribution inputs are gathered lazily** at the top of each
  iteration (i.e. after any adapter resets) as one :class:`DistInputs` and
  **yielded** — §3.4 invariant (rollout==update): the exact act-time bundle is
  what PPO/GRPO store in their buffers and hand back to the update forward;
  it is never regenerated from board state. Eval ignores it.
* **``board_static`` dedup** — each ``obs_next`` inherits the acting obs's
  ``board_static`` reference (static within an episode; keeps PPO memory
  O(N) instead of O(T*N)). Adapters refresh the reference chain simply by
  writing reset obs into ``obs_by_slot``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch

from pcb_world.vec.backends.base import VecBackend

from methods.rl_agent.models.v1.spec import DistInputs

if TYPE_CHECKING:
    from methods.rl_agent.models.v1.net import KiCadRLModel
    from methods.rl_agent.policy.agent import KiCadRLAgent
    from methods.rl_agent.training.mem_budget import MemBudgetModel
    from methods.rl_agent.wrappers.adapter import KiCadRLWrapper

__all__ = ["StepBatch", "budgeted_forward", "iter_rollout"]


@dataclass
class StepBatch:
    """One yielded transition batch over the currently-live slots.

    All per-slot arrays/lists are aligned with ``live`` (length ``B``).
    ``log_probs``/``values`` are ``None`` on the eval path
    (``want_value=False``).
    """
    live: list[int]
    obs: list[dict]                      # act-time obs (what the forward saw)
    actions: np.ndarray                  # (B, 3) int64
    log_probs: np.ndarray | None         # (B,) float32
    values: np.ndarray | None            # (B,) float32
    obs_next: list[dict]                 # step results (board_static deduped)
    rewards: np.ndarray                  # (B,) float
    terminateds: np.ndarray              # (B,) bool
    truncateds: np.ndarray               # (B,) bool
    infos: list[dict]
    # Act-time distribution inputs, numpy (§3.4 rollout==update invariant:
    # store these, never regenerate).
    dist: DistInputs
    # Batched CPU walk over ``live`` (the tokenize pass the forward already
    # ran). Set on the collect path (``want_value=True``) so the collector can
    # cache it flat (merged once at collect end → ``walk_flat``) and let the
    # update index-gather it instead of re-walking. ``None`` on the eval path.
    walk: dict | None = None


def budgeted_forward(
    actor: "KiCadRLAgent | KiCadRLModel",
    obs_list: list[dict],
    dist: DistInputs,
    mem_budget: "MemBudgetModel",
    *,
    want_value: bool,
    deterministic: bool = False,
    walk: "dict[str, Any] | None" = None,
) -> tuple[torch.Tensor, "torch.Tensor | None", "torch.Tensor | None"]:
    """No-grad rollout forward, split into peak-VRAM-budget-fitting chunks.

    Exact by construction: outputs are per-row (attention is masked per row,
    padding is per-chunk ``max(seq_lens)``), so splitting the batch and
    scattering the chunk outputs back to their row positions equals the whole-
    batch forward. The walk (CPU tokenize) runs ONCE, **batched** — the batch
    walk both yields ``seq_lens`` for planning and doubles as the no-split
    fast path's ``walked=`` (zero extra walk vs the unsplit path); only an
    actual split pays a second, per-chunk walk. (Per-sample B=1 walks here —
    64x Python passes + merge, every rollout step — measured 3~7x the
    batched walk and +5~12% ITER in the 260713 A/B; do not revert.)
    ``dist`` is the batch's torch bundle; each chunk takes ``dist.select``.

    OOM backstop mirrors the update path: halve the budget (this call only),
    replan, retry; a single-row chunk that still OOMs re-raises.

    Returns ``(actions, log_probs, values)`` — ``values`` is None when
    ``want_value=False`` (the eval path discards it).
    """
    from methods.rl_agent.training import mem_budget as _mb_mod

    model = getattr(actor, "model", actor)   # KiCadRLAgent facade or bare model
    tokenizer = model.tokenizer
    # Reuse the caller's hoisted walk when given (collect caches it); else walk
    # here (eval path, and the mode_none re-derivation cost is the same).
    if walk is None:
        walk = tokenizer.walk_timed(obs_list)
    seq_lens = list(walk["seq_lens"])
    measure = torch.cuda.is_available()
    limit = mem_budget.capacity()
    n_oom = 0
    while True:
        chunks = mem_budget.plan_chunks(seq_lens, limit=limit)
        try:
            if len(chunks) == 1:
                # Whole batch fits: keep the original row order (identical
                # sampling stream to the unsplit path), just skip the re-walk.
                kw = dict(dist=dist, walked=walk)
                if want_value:
                    return actor.act_and_value(
                        obs_list, deterministic=deterministic, **kw,
                    )
                acts, lps = actor.act(obs_list, deterministic=deterministic, **kw)
                return acts, lps, None
            parts: list[tuple[list[int], tuple]] = []
            for chunk in chunks:
                kw = dict(
                    dist=dist.select(chunk),
                    # split path only: one extra batched walk per chunk
                    walked=tokenizer.walk_timed([obs_list[p] for p in chunk]),
                )
                obs_chunk = [obs_list[p] for p in chunk]
                if measure:
                    base = _mb_mod.begin_measured_region()
                if want_value:
                    out = actor.act_and_value(
                        obs_chunk, deterministic=deterministic, **kw,
                    )
                else:
                    out = actor.act(obs_chunk, deterministic=deterministic, **kw)
                if measure:
                    mem_budget.observe(
                        len(chunk), max(seq_lens[p] for p in chunk),
                        _mb_mod.end_measured_region(base),
                    )
                parts.append((chunk, out))
            # Scatter chunk outputs back to original row positions.
            order = [p for chunk, _ in parts for p in chunk]
            inv = torch.empty(len(obs_list), dtype=torch.long)
            inv[torch.as_tensor(order)] = torch.arange(len(order))
            cat = [torch.cat([out[j] for _, out in parts]) for j in
                   range(2 if not want_value else 3)]
            actions = cat[0][inv.to(cat[0].device)]
            log_probs = cat[1][inv.to(cat[1].device)]
            values = cat[2][inv.to(cat[2].device)] if want_value else None
            return actions, log_probs, values
        except torch.cuda.OutOfMemoryError:
            if all(len(c) == 1 for c in chunks):
                raise   # a single row OOMs -> board too big for VRAM
            # fall through: recover outside the handler (the traceback pins
            # the failed forward's frames until the except block exits)
        n_oom += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        limit /= 2.0   # transient — this forward call only
        if n_oom <= 3:
            import warnings
            warnings.warn(
                f"CUDA OOM in rollout forward despite planned chunking "
                f"(B={len(obs_list)}, chunks={len(chunks)}); retrying on a "
                f"halved budget (n_oom={n_oom}).",
                RuntimeWarning, stacklevel=2,
            )


def iter_rollout(
    envs: "VecBackend | list[KiCadRLWrapper]",
    policy: "KiCadRLAgent | KiCadRLModel",
    device: torch.device,
    obs_by_slot: dict[int, dict[str, Any]],
    active: list[int],
    done: dict[int, bool],
    *,
    want_value: bool,
    deterministic: bool = False,
    max_steps: int,
    mem_budget: "MemBudgetModel | None" = None,
) -> Iterator[StepBatch]:
    """Roll the ``live = active − done`` slots one act+step at a time.

    Yields a :class:`StepBatch` per step for up to ``max_steps`` iterations,
    stopping early when no slot is live. ``want_value=True`` uses the
    training-collection forward (``act_and_value``: grad mode untouched,
    stochastic); ``want_value=False`` uses the eval forward
    (``KiCadRLAgent.act``: no-grad, honours ``deterministic``). A calibrated ``mem_budget`` routes the
    forward through :func:`budgeted_forward` (peak-VRAM-planned splitting;
    exact). See the module docstring for the full caller contract (done
    ownership / obs re-injection / mask storage).
    """
    from methods.rl_agent.policy.agent import KiCadRLAgent, gather_dist

    use_vec = isinstance(envs, VecBackend)
    policy_ns = bool(getattr(policy, "policy_net_select", False))
    agent: "KiCadRLAgent | None" = None
    if not want_value:
        agent = (
            policy if isinstance(policy, KiCadRLAgent)
            else KiCadRLAgent(policy, device=device)
        )
        # act_from_pool parity: eval mask tensors follow the agent's own
        # device (matters only for a pre-built agent whose device differs
        # from the argument).
        device = agent.device

    for _step in range(max_steps):
        live = [slot for slot in active if not done.get(slot, False)]
        if not live:
            return
        # Re-read shared obs state (adapters may have written reset obs).
        obs_live = [obs_by_slot[slot] for slot in live]

        # --- Act-time masks (lazy: gathered after any adapter resets) ---
        # obs_live is passed so the wrapper-embedded "_masks" payload skips
        # the IPC.
        dist = gather_dist(
            envs, live, policy_net_select=policy_ns, obs_list=obs_live,
        )
        dist_t = dist.to(device)

        # --- Forward ---
        # Collect only: hoist the batched walk (the CPU tokenize pass the
        # forward runs anyway) and pass it in, so the collector can cache it
        # for the update instead of the update re-walking. Eval builds no
        # buffer, so it skips the hoist and lets the forward walk internally.
        step_walk = None
        if want_value:
            tok = getattr(getattr(policy, "model", policy), "tokenizer", None)
            if tok is not None:   # scripted/mock policies have no tokenizer
                step_walk = tok.walk_timed(obs_live)
        if mem_budget is not None and mem_budget.ready:
            # Budget-split forward (exact; see budgeted_forward). Uses the
            # same actor per path as the branches below.
            acts_t, logp_t, vals_t = budgeted_forward(
                policy if want_value else agent, obs_live, dist_t, mem_budget,
                want_value=want_value, deterministic=deterministic,
                walk=step_walk,
            )
            actions_np = acts_t.cpu().numpy()
            log_probs_np = logp_t.cpu().numpy() if want_value else None
            values_np = vals_t.cpu().numpy() if want_value else None
        elif want_value:
            # Pass the hoisted walk only when we have one — mock policies
            # (no tokenizer → step_walk None) don't accept ``walked=``.
            walk_kw = {"walked": step_walk} if step_walk is not None else {}
            actions, log_probs, values = policy.act_and_value(
                obs_live, dist=dist_t, **walk_kw,
            )
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()
        else:
            acts_t, _logp = agent.act(
                obs_live, dist=dist_t, deterministic=deterministic,
            )
            actions_np = acts_t.cpu().numpy()
            log_probs_np = None
            values_np = None

        # --- Step ---
        if use_vec:
            envs.step_async_selective(live, actions_np)
            obs_next, rewards, terms, truncs, infos = envs.step_wait_selective()
        else:
            obs_next, rew_l, term_l, trunc_l, infos = [], [], [], [], []
            for k, i in enumerate(live):
                o, r, te, tr, info = envs[i].step(actions_np[k])
                obs_next.append(o)
                rew_l.append(r)
                term_l.append(te)
                trunc_l.append(tr)
                infos.append(info)
            rewards = np.array(rew_l, dtype=np.float64)
            terms = np.array(term_l, dtype=bool)
            truncs = np.array(trunc_l, dtype=bool)

        # --- Advance shared obs (before yield, so adapter resets win) ---
        for k, slot in enumerate(live):
            # board_static is static within an episode: keep one reference
            # alive per slot instead of a fresh pickle copy per step.
            obs_next[k]["board_static"] = obs_live[k]["board_static"]
            obs_by_slot[slot] = obs_next[k]

        yield StepBatch(
            live=live,
            obs=obs_live,
            actions=actions_np,
            log_probs=log_probs_np,
            values=values_np,
            obs_next=obs_next,
            rewards=rewards,
            terminateds=terms,
            truncateds=truncs,
            infos=infos,
            dist=dist,
            walk=step_walk,
        )
