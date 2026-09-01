"""Multi-GPU PPO update: main-trainer + update-worker processes, manual grad sync.

* **Manual ``torch.distributed`` primitives, NOT the ``DistributedDataParallel``
  module wrapper.** The update has three execution paths
  (:mod:`methods.rl_agent.algorithms._common`): fixed-batch, budget-planned
  chunking, and the reactive OOM 1/4-peel. The latter two can run a different
  number of backwards per rank, which conflicts with the DDP module's
  forward-backward pairing/``no_sync`` bookkeeping (one rank OOM-restarting →
  hang). Manual sync meets exactly one allreduce per minibatch, so per-rank
  chunk/peel structure is free to differ.
* **Process layout**: the main process (rank 0, the trainer) does everything it
  does today plus its own shard's update; ``world_size - 1`` spawned workers
  (``cuda:1..``) only loop "receive buffer → update own shard (same code) →
  wait". Each rank owns an optimizer copy — gradients are identical after the
  allreduce, so parameters stay in lockstep.
* **Equivalence contract** (guarded by ``tests/test_ddp_equivalence.py``):
  every rank scales its shard loss as ``sum / global_mb_size`` (the
  ``_accumulate_chunk`` convention), gradients are SUM-allreduced, advantage
  normalization uses the FULL minibatch, and the permutation is generated on
  rank 0 and broadcast — so N ranks × B/N == 1 GPU × B mean gradient exactly
  (up to fp reassociation), including non-divisible remainder minibatches.

Workers never import the KiCad engine (policy construction + update math only);
``torch.multiprocessing.spawn`` uses the spawn start method, so no forked
engine state can leak in (KiCad singleton constraint).
"""
from __future__ import annotations

import os
import pickle
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Generous collective timeout: a hung/dead peer surfaces as an error instead of
# blocking forever. Workers wait in recv across a full collect+eval iteration,
# so this must exceed the worst-case iteration wall time (the d3b 2x-horizon
# first iteration with eval-at-init exceeds 10 minutes).
DDP_TIMEOUT = timedelta(hours=6)


def _broadcast_blob(obj, *, rank: int, group) -> tuple[object, int]:
    """Pickle-broadcast an arbitrary object over a gloo group.

    Manual (length + uint8 tensor) instead of ``broadcast_object_list`` so the
    byte count is observable — the handoff requires logging per-iter transfer
    size/time. Returns ``(obj, nbytes)`` on every rank.
    """
    if rank == 0:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        size = torch.tensor([len(data)], dtype=torch.int64)
        buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        dist.broadcast(size, src=0, group=group)
        dist.broadcast(buf, src=0, group=group)
        return obj, len(data)
    size = torch.zeros(1, dtype=torch.int64)
    dist.broadcast(size, src=0, group=group)
    buf = torch.empty(int(size.item()), dtype=torch.uint8)
    dist.broadcast(buf, src=0, group=group)
    return pickle.loads(buf.numpy().tobytes()), int(size.item())


class DDPCtx:
    """Rank-local handle used inside ``policy_update_loop`` (all ranks).

    ``pg`` (default process group when ``None``) carries the tensor
    collectives — nccl on CUDA, gloo on CPU (tests). ``gloo_pg`` carries
    pickled objects; on CPU it is the same group.
    """

    def __init__(self, rank: int, world_size: int, device, *, gloo_pg=None):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.pg = None            # default group
        self.gloo_pg = gloo_pg    # None -> default group (gloo backends)
        self._grad_numel: int | None = None

    def broadcast_perm(self, n: int, device) -> torch.Tensor:
        """Rank-0 ``randperm`` broadcast — equivalence must not rely on seed
        lockstep across ranks. Rank 0 consumes its RNG stream exactly like the
        single-GPU path (A/B alignment)."""
        if self.rank == 0:
            perm = torch.randperm(n, device=device)
        else:
            perm = torch.empty(n, dtype=torch.int64, device=device)
        dist.broadcast(perm, src=0, group=self.pg)
        return perm

    def shard_positions(self, mb_size: int) -> list[int]:
        """This rank's strided slice of minibatch-local positions. Any
        partition is equivalent under the sum/``global_mb_size`` loss scale;
        a remainder minibatch may leave some ranks empty — allowed."""
        return list(range(self.rank, mb_size, self.world_size))

    def sync_grads(self, policy) -> None:
        """Single SUM-allreduce of all grads via one flat fp32 buffer.

        Fixed parameter order; ``p.grad is None`` slots are sent as zeros —
        a shard may leave a head unused on one rank only (e.g. no MAKE_*
        action in the shard → mode head), and a None/tensor shape mismatch
        across ranks would otherwise hang. Per-param used-flags ride at the
        tail of the same buffer (still one allreduce), and a grad is
        materialized on the way back ONLY where some rank used the param:
        a param unused on EVERY rank keeps ``grad=None``, preserving the
        single-process optimizer skip semantics (Adam/AdamW would otherwise
        run momentum/weight-decay off a fabricated zero grad and drift from
        the single-GPU trajectory). NO division by world size: each rank's
        shard loss is already ``sum / global_mb_size``, so the SUM equals
        the full-batch mean grad.
        """
        params = [p for p in policy.parameters() if p.requires_grad]
        if self._grad_numel is None:
            self._grad_numel = sum(p.numel() for p in params)
        flat = torch.zeros(
            self._grad_numel + len(params), dtype=torch.float32,
            device=self.device,
        )
        offset = 0
        for i, p in enumerate(params):
            n = p.numel()
            if p.grad is not None:
                flat[offset : offset + n].copy_(p.grad.detach().reshape(-1))
                flat[self._grad_numel + i] = 1.0
            offset += n
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.pg)
        # one D2H sync for all flags (per-param .item() would sync each time)
        used = (flat[self._grad_numel :] > 0).cpu().tolist()
        offset = 0
        for i, p in enumerate(params):
            n = p.numel()
            if used[i]:
                g = flat[offset : offset + n].view_as(p)
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)
            offset += n

    def allreduce_sums(self, sums: dict[str, float]) -> dict[str, float]:
        """SUM-allreduce a dict of per-rank partial sums (logging stats)."""
        keys = sorted(sums)
        t = torch.tensor(
            [sums[k] for k in keys], dtype=torch.float64, device=self.device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.pg)
        return dict(zip(keys, t.cpu().tolist()))


def _init_dist(rank: int, world_size: int, rdv_file: str, device) -> DDPCtx:
    """Init the default pg (nccl on CUDA, gloo on CPU) + a gloo side group for
    pickled-object broadcasts. Called by main (rank 0) and every worker."""
    backend = "nccl" if torch.device(device).type == "cuda" else "gloo"
    dist.init_process_group(
        backend, init_method=f"file://{rdv_file}", rank=rank,
        world_size=world_size, timeout=DDP_TIMEOUT,
    )
    gloo_pg = (
        dist.new_group(backend="gloo", timeout=DDP_TIMEOUT)
        if backend == "nccl" else None
    )
    return DDPCtx(rank, world_size, device, gloo_pg=gloo_pg)


def _worker_payload(buffer):
    """The buffer as broadcast to update workers.

    With a collect-time walk (``walk_flat``) aboard, ``obs_list`` is dropped
    (set to None): the update consumes the walk via ``walked=`` and the
    tokenizer never reads obs on that path, so shipping the obs dicts —
    the bulk of the pickle — would be pure transfer cost. Without the walk
    (GRPO), the buffer goes out unchanged (workers re-walk from obs).
    """
    if buffer.get("walk_flat") is not None:
        return {**buffer, "obs_list": None}
    return buffer


def _state_to_cpu(obj):
    """Recursively move tensors in a state_dict-like structure to CPU (pickle
    of CUDA tensors would re-materialize on the sender's device index)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_state_to_cpu(v) for v in obj)
    return obj


def _build_worker_policy_and_optimizer(args, device, *, use_critic: bool):
    """Mirror of the trainer's ``_build_policy`` + AdamW construction — same
    RLPolicyConfig path and the same ``configure_speed`` flags (bf16/compile
    knobs must match rank 0 or the loss math diverges)."""
    from configs.loader.schema import RLPolicyConfig

    policy = RLPolicyConfig.from_namespace(args, use_critic=use_critic).build(device)
    bf16 = bool(getattr(args, "bf16", False))
    regions = tuple(
        r for r in getattr(args, "compile_regions", "").split(",") if r
    )
    attn = getattr(args, "attn", "sdpa")
    if bf16 or regions or attn != "sdpa":
        policy.configure_speed(
            bf16=bf16, compile_regions=regions,
            compile_mode=getattr(args, "compile_mode", "default"),
            attn=attn,
        )
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-4,
    )
    return policy, optimizer


def _worker_main(idx: int, world_size: int, args_dict: dict,
                 rdv_file: str, use_critic: bool) -> None:
    """Update-worker entrypoint (rank = idx + 1, device = ``cuda:rank``).

    Loop: receive a command blob → ("state") load state dicts /
    ("update") run ``policy_update_loop`` on its shard / ("stop") exit.
    Never touches the env or the KiCad engine.

    Immediate exception visibility: ``mp.spawn`` worker exceptions are
    normally re-raised only by the parent's ``join()``, but rank 0 hits the
    dead worker's gloo broadcast disconnect (RuntimeError) first and dies
    before reaching join — so the worker's original exception would
    otherwise be lost. This prints it to stderr and re-raises it
    immediately as it occurs. ``pcb_world.diag`` supplies the fatal-signal
    log (for native crashes) plus an exception-context dump. stderr is not
    redirected — the FATAL message must land in launch.log for Monitor to
    catch it (unlike the engine workers, there is no assert spam here to
    filter out).
    """
    from pcb_world import diag

    rank = idx + 1
    handler = diag.install_crash_handler(f"ddp{rank}", register_atexit=False)
    try:
        _worker_body(idx, world_size, args_dict, rdv_file, use_critic)
    except BaseException:
        import sys
        import traceback
        dump = diag.dump_context(
            "ddp_worker_exception", rank=rank,
            traceback=traceback.format_exc(),
        )
        print(f"\n[ddp-worker rank={rank}] FATAL — dumping primary exception immediately"
              + (f" (context: {dump})" if dump else "") + ":",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    if handler is not None:  # normal exit (stop) — mp.spawn children have no guaranteed atexit
        diag.remove_log_if_empty(*handler)


def _worker_body(idx: int, world_size: int, args_dict: dict,
                 rdv_file: str, use_critic: bool) -> None:
    from argparse import Namespace

    rank = idx + 1
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    ctx = _init_dist(rank, world_size, rdv_file, device)
    args = Namespace(**args_dict)
    policy, optimizer = _build_worker_policy_and_optimizer(
        args, device, use_critic=use_critic,
    )

    from methods.rl_agent.algorithms._common import policy_update_loop

    while True:
        msg, _ = _broadcast_blob(None, rank=rank, group=ctx.gloo_pg)
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "state":
            _, policy_sd, optim_sd = msg
            policy.load_state_dict(policy_sd)
            # load_state_dict casts/moves optimizer state to each param's device.
            optimizer.load_state_dict(optim_sd)
            continue
        _, buffer, update_kwargs, lr = msg
        # LR lockstep: rank 0 owns the scheduler; workers take the value.
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        policy_update_loop(
            policy, optimizer, buffer, device, ddp=ctx, **update_kwargs,
        )
    dist.destroy_process_group()


class DDPUpdateGroup:
    """Main-process (rank 0) handle: spawns workers, owns state/buffer
    broadcasts, and exposes ``self.ctx`` for ``policy_update_loop``.

    Construct AFTER ``_resume()`` — the initial state broadcast then covers
    both fresh init and checkpoint resume.
    """

    def __init__(self, args, policy, optimizer, device, *, use_critic: bool):
        world_size = int(args.update_gpus)
        assert world_size >= 2, "DDPUpdateGroup needs --update-gpus >= 2"
        assert torch.device(device).type == "cuda", (
            "--update-gpus > 1 requires a CUDA device"
        )
        assert torch.cuda.device_count() >= world_size, (
            f"--update-gpus {world_size} > visible CUDA devices "
            f"({torch.cuda.device_count()})"
        )
        # file-store rendezvous: path must not pre-exist.
        fd, rdv_file = tempfile.mkstemp(prefix="cadagent_ddp_rdv_")
        os.close(fd)
        os.unlink(rdv_file)
        self._rdv_file = rdv_file
        self.world_size = world_size
        print(f"  ddp-update: spawning {world_size - 1} update worker(s) "
              f"(cuda:1..{world_size - 1}, nccl, manual grad allreduce)")
        self._spawn_ctx = mp.spawn(
            _worker_main,
            args=(world_size, vars(args), rdv_file, use_critic),
            nprocs=world_size - 1, join=False,
        )
        self.ctx = _init_dist(0, world_size, rdv_file, device)
        self.broadcast_state(policy, optimizer)

    def _postmortem_dead_workers(self, why: str) -> None:
        """Right after rank 0 catches the gloo disconnect: record the dead
        worker's exitcode.

        A worker that died from a Python exception already leaves its own
        stderr + dump (via ``_worker_main``), but a SIGKILL (OOM) or
        segfault leaves nothing on the worker side (aside from the signal
        log) — only the parent can diagnose a silent death, via the
        exitcode.
        """
        from pcb_world import diag

        for i, proc in enumerate(self._spawn_ctx.processes):
            if proc is not None and not proc.is_alive() and proc.exitcode:
                diag.write_postmortem(
                    f"ddp{i + 1}", proc, why, respawn_count=0,
                )

    def broadcast_state(self, policy, optimizer) -> None:
        """Push rank-0 policy+optimizer state to every worker (init/resume)."""
        try:
            _broadcast_blob(
                ("state",
                 _state_to_cpu(policy.state_dict()),
                 _state_to_cpu(optimizer.state_dict())),
                rank=0, group=self.ctx.gloo_pg,
            )
        except RuntimeError:
            self._postmortem_dead_workers("gloo disconnect during broadcast_state")
            raise

    def dispatch_update(self, buffer, update_kwargs: dict,
                        lr: float) -> tuple[int, float]:
        """Broadcast this iteration's buffer + update kwargs to the workers.

        When the collector cached the tokenize walk, the buffer carries it as
        ONE flat batched walk (``walk_flat`` — big numpy columns, so the
        pickle rides the buffer protocol instead of crawling per-sample
        dicts) and every rank reuses it instead of re-walking
        (``policy_update_loop`` walk-cache). The walked= update path never
        reads obs (the tokenizer ignores ``obs_list`` when ``walked=`` is
        given), so the worker payload drops ``obs_list`` entirely
        (:func:`_worker_payload`) — the obs dict pickle was the bulk of the
        base broadcast. Absent the walk (GRPO), the full buffer is sent and
        each rank re-walks its own copy. Returns ``(nbytes, seconds)`` for
        the transfer-cost log.
        """
        t0 = time.perf_counter()
        try:
            _, nbytes = _broadcast_blob(
                ("update", _worker_payload(buffer), update_kwargs, lr),
                rank=0, group=self.ctx.gloo_pg,
            )
        except RuntimeError:
            self._postmortem_dead_workers("gloo disconnect during dispatch_update")
            raise
        return nbytes, time.perf_counter() - t0

    def shutdown(self) -> None:
        """Stop the workers and tear the groups down — destroy BEFORE join.

        The other order deadlocks. ``destroy_process_group`` shuts the backends
        down one at a time precisely because "ncclCommAbort() was a 'collective'
        call in some versions of NCCL" (torch's own comment in
        ``distributed_c10d``), so every rank has to be inside the teardown for
        any of them to leave it. Joining first puts rank 0 in ``join()`` waiting
        for the worker to EXIT while the worker sits in ``destroy_process_group``
        waiting for rank 0 to arrive — neither ever moves, both NCCL watchdogs
        stay alive, and the run holds its GPUs until someone sends SIGTERM
        (observed: a d2b 600-iter run still hung 4h22m after its last iteration).

        Destroying first makes the call symmetric with the worker's; the worker
        then returns from it and exits, so ``join()`` completes immediately.
        Reproduction + both orders measured: ``sandbox/notrans/ddp_deadlock_repro.py``.
        """
        _broadcast_blob(("stop",), rank=0, group=self.ctx.gloo_pg)
        dist.destroy_process_group()
        self._spawn_ctx.join()
        if os.path.exists(self._rdv_file):
            os.unlink(self._rdv_file)
