"""H1 — sync-barrier decomposition: step barrier + 4 mask barriers.

Decomposes all 5 serial barriers per rollout step:
``action_masks / start_route_pointer_indices / mode_mask / net_valid_mask`` (the
4 mask barriers) + the ``step`` barrier.

Each barrier is split into (per step, averaged):
  - ``send_ms``      main-serial send over live workers
  - ``worker_med/max`` main-observed ready time via ``mpc.wait`` + ``recv_bytes``
                       (no unpickle) — completion ORDER, not compute
  - ``straggler_ms`` = worker_max - worker_median (median-exceeding barrier tax)
  - ``unpickle_ms``  deferred batch ``pickle.loads`` under ``gc.disable`` — pure
                     main-serial deserialize, isolated from transport
  - ``worker_compute_med`` (step only) from ``info["_prof"]`` (worker_shim) — the
                     TRUE intra-worker compute, so straggler ordering and engine
                     recompute are not conflated with transport.

The two-stage recv_bytes -> deferred-loads trick is what separates "straggler
wait" from "main-serial unpickle" — the plan's headline H1 finding (unpickle
dominates on large obs). Mask-barrier decomposition is measured on an isolated
extra read-only query (action_masks/mode_mask etc. do not mutate engine state),
so it does not perturb the rollout advance.
"""
from __future__ import annotations

import gc
import multiprocessing.connection as mpc
import pickle
import statistics as st
import time

from tools.diagnostics.speed_profiler.instrument import p90

PC = time.perf_counter

_MASK_CMDS = ["action_masks", "start_route_pointer_indices", "mode_mask", "net_valid_mask"]


def _decompose(pool, live, payloads):
    """Serial-send ``payloads[s]`` to each live worker, time the barrier, and
    return the decomposition. ``payloads`` maps slot -> (cmd, data) tuple."""
    t0 = PC()
    for s in live:
        pool.remotes[s].send(payloads[s])
    send_ms = (PC() - t0) * 1000

    base = PC()
    pending = {pool.remotes[s]: s for s in live}
    ready, raw = {}, {}
    while pending:
        for c in mpc.wait(list(pending)):
            s = pending.pop(c)
            ready[s] = (PC() - base) * 1000
            raw[s] = c.recv_bytes()
    ready_wall_ms = (PC() - base) * 1000

    gc_was = gc.isenabled()
    gc.disable()
    tu = PC()
    results = {s: pickle.loads(b) for s, b in raw.items()}
    unpickle_ms = (PC() - tu) * 1000
    if gc_was:
        gc.enable()

    rt = sorted(ready.values())
    return {
        "send_ms": send_ms, "ready_wall_ms": ready_wall_ms, "unpickle_ms": unpickle_ms,
        # median = basis for the wall-time decomposition (robust to the tail
        #   — straggler is defined as the excess over median),
        # mean   = average compute/resource view (mean x N = total
        #   worker-seconds; right-skewed, so median < mean)
        "w_min": rt[0], "w_med": rt[len(rt) // 2],
        "w_mean": st.mean(rt), "w_p90": p90(rt),
        "w_max": rt[-1],
        "results": results,
    }


def barrier_probe(trainer, n_steps: int, policy_net_select: bool | None = None) -> dict:
    """Drive ``n_steps`` real rollout steps, decomposing all 5 barriers per step.

    Requires the worker_shim active (``CADAGENT_PROFILE_WORKER``) for the
    ``worker_compute`` split; without it that field is None (the send/straggler/
    unpickle split still works). Advances the rollout faithfully (reset on done).
    """
    from methods.rl_agent.policy.agent import gather_mask_arrays, mask_arrays_to_tensors

    pool, agent, device = trainer.envs, trainer.agent, trainer.device
    if policy_net_select is None:
        policy_net_select = bool(getattr(agent, "policy_net_select", False))
    n = len(pool)
    obs_by_slot = {i: o for i, o in enumerate(pool.reset_all())}
    live = list(range(n))

    mask_rows: dict[str, list[dict]] = {c: [] for c in _MASK_CMDS}
    step_rows: list[dict] = []
    worker_compute_rows: list[float] = []

    for _ in range(n_steps):
        # --- isolated mask-barrier decomposition (read-only queries) ---
        cmds = _MASK_CMDS if policy_net_select else _MASK_CMDS[:3]
        for cmd in cmds:
            d = _decompose(pool, live, {s: (cmd, None) for s in live})
            mask_rows[cmd].append(d)

        # --- real forward (correct masks) to get actions ---
        obs_live = [obs_by_slot[s] for s in live]
        # v0.31: gather_mask_arrays returns a 5th offlayer_masks array — feed it
        # through mask_arrays_to_tensors' off_masks kwarg (same as the base path).
        masks, ptr, mode, nvm, offm = gather_mask_arrays(pool, live, policy_net_select=policy_net_select)
        mt, pt, mdt, nk = mask_arrays_to_tensors(masks, ptr, mode, nvm, device,
                                                 mode_none_if_all_true=False, off_masks=offm)
        acts, _ = agent.act(obs_live, action_masks=mt, pointer_masks=pt, mode_mask=mdt, **nk)
        acts = acts.cpu().numpy()

        # --- step-barrier decomposition ---
        d = _decompose(pool, live, {s: ("step", acts[k]) for k, s in enumerate(live)})
        step_rows.append(d)

        # advance / reset (extract worker_compute; handle done)
        reset = []
        for k, s in enumerate(live):
            res = d["results"][s]
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "__error__":
                raise RuntimeError(f"worker {s}: {res[1]}")
            o, _r, term, trunc, info = res
            if isinstance(info, dict) and "_prof" in info:
                worker_compute_rows.append(info["_prof"].get("worker_compute_s", 0.0) * 1000)
            o["board_static"] = obs_by_slot[s]["board_static"]
            if term or trunc:
                reset.append(s)
            else:
                obs_by_slot[s] = o
        for s in reset:
            pool.remotes[s].send(("reset", None))
        for s in reset:
            obs_by_slot[s] = pool.remotes[s].recv()[0]

    def _agg(rows):
        m = lambda k: round(st.mean(r[k] for r in rows), 3)
        send, wwall, unp = m("send_ms"), m("ready_wall_ms"), m("unpickle_ms")
        wmed, wmax = m("w_med"), m("w_max")
        barrier = send + wwall + unp
        return {
            "barrier_wall_ms": round(barrier, 3),
            "send_ms": send, "worker_median_ms": wmed,
            "worker_mean_ms": m("w_mean"), "worker_p90_ms": m("w_p90"),
            "worker_max_ms": wmax,
            # Two straggler quantities (different counterfactuals):
            #  - idle_waste (max-mean) = average worker idle time = the lower
            #    bound on what perfect load-balancing/async could recover
            #    (work-conserving; xN = total worker-seconds lost)
            #  - straggler (max-median) = excess over the typical case = a
            #    tail-diagnosis metric (only recoverable if the tail is
            #    coherent; overstated for a heavy-board tail)
            "idle_waste_ms": round(wmax - m("w_mean"), 3),
            "straggler_ms": round(wmax - wmed, 3), "unpickle_ms": unp,
        }

    out = {
        "n_steps": n_steps, "n_envs": n,
        "step_barrier": _agg(step_rows),
        "mask_barriers": {c: _agg(mask_rows[c]) for c in mask_rows if mask_rows[c]},
        "worker_compute_median_ms": round(st.median(worker_compute_rows), 3) if worker_compute_rows else None,
        # mean = average compute (x N = total worker-seconds per step; used
        #   for async-hiding/utilization calculations)
        "worker_compute_mean_ms": round(st.mean(worker_compute_rows), 3) if worker_compute_rows else None,
        "worker_compute_p90_ms": round(p90(worker_compute_rows), 3) if worker_compute_rows else None,
        "worker_shim_active": bool(worker_compute_rows),
    }
    return out
