"""Driver — build the REAL PPOTrainer and run the selected measurements.

Drives the exact functions ``train_iteration`` calls (``select_boards`` ->
``collect_rollout`` -> ``compute_targets`` -> ``policy_update_loop``) so the numbers
reflect production, under a canonical representative config. One config per
process (KiCad singleton; the main proc never imports the router). Base layer is
import-only — instrumentation is main-proc monkeypatch (:mod:`.hooks`) + the
forkserver-preload :mod:`.worker_shim`.

H3 (phase attribution + update decomposition) and H1 (barrier) are the primary
measurements — they determine whether the engine is the throughput bottleneck.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PC = time.perf_counter


@dataclass
class ProfileConfig:
    dataset: str = "d2a"                 # d2a (synth 2L) | d3b (real, medium)
    n_envs: int = 64
    n_steps: int = 512
    max_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4
    seed: int = 42
    obs_format: str = "indexed"    # same default as the RL training path; json = legacy A/B
    warmup_iters: int = 1
    measured_iters: int = 2
    # 100+: straggler/idle_waste are max-statistics, so they have high variance
    # at small sample sizes (12~30) — send/unpickle/wcomp are stable even at
    # 20 steps, but the tail metrics need a larger sample.
    barrier_steps: int = 100
    gpu_index: int = 0
    host_tag: str = "l40"
    # speed-knob A/B (bf16 autocast / torch.compile regions/mode)
    bf16: bool = False
    compile_regions: str = ""          # comma list of {stack,decode,heads}
    compile_mode: str = "default"      # default | reduce-overhead | max-autotune
    # model dims (must match the run; defaults = harness config)
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    # capability toggles
    enable_util: bool = True
    enable_barrier: bool = True
    enable_update_decomp: bool = True
    # Inline decomposition of collect: mask/forward/step/advance/collector/reset +
    # rollout GPU (cuda-event) — closes the waterfall's "residual" bucket
    # (iter_rollout timed mirror).
    enable_rollout_decomp: bool = True
    # Captures a torch.profiler chrome trace during the untimed warmup iter
    # (rollout 10 steps + update 8 minibatches) — later metric questions are
    # answered by post-processing this trace instead of re-measuring.
    enable_trace: bool = False
    trace_rollout_steps: int = 10
    trace_update_mbs: int = 8
    # Eval-Rollout (3-val) phase — off by default (spawns its own n_envs pool
    # per val set; expensive). Reuses the trainer's own evaluator wiring
    # (primary --eval-split + eval2/eval3 diagnostic sets, loop.py).
    enable_eval: bool = False
    eval_n_rollouts: int = 4
    eval_board_limit: int = 8            # primary set (applied by the trainer)
    eval2_boards: str | None = None      # explicit board-list txt (no repo default)
    eval3_boards: str | None = None
    eval23_board_limit: int | None = 10  # slice diagnostic sets (None = full list)
    out: str | None = None


# ---------------------------------------------------------------------------
# Real trainer build
# ---------------------------------------------------------------------------
def build_trainer(cfg: ProfileConfig, scratch: str):
    from methods.rl_agent.training.train_ppo import build_arg_parser
    from methods.rl_agent.training.loop import PPOTrainer

    # d3.json only has train/test (no val) → branch on eval-split.
    # d3b uses only medium/TEST (10 boards, verified by the eval path) — medium/
    # train mixes in boards the env can't parse (no Edge.Cuts).
    if cfg.dataset == "d2a":
        boards_json, difficulty, boards_split, eval_split = (
            "configs/datasets/d2a.json", "easy", "train", "val")
    elif cfg.dataset == "d3b":
        boards_json, difficulty, boards_split, eval_split = (
            "configs/datasets/d3.json", "medium", "test", "test")
    else:
        raise SystemExit(f"bad dataset {cfg.dataset}")

    argv = [
        "--board", "tests/fixtures/simple_routing_board.kicad_pcb",
        "--boards-order", "per_env_epoch", "--boards-json", boards_json,
        "--use-yaml-drc-fallback", "--drc-config-path", "configs/drc/synth_2L_v2.yaml",
        "--boards-difficulty", difficulty, "--boards-split", boards_split,
        "--eval-split", eval_split, "--eval-board-limit", str(cfg.eval_board_limit),
        "--eval-n-rollouts", str(cfg.eval_n_rollouts),
        "--eval-every", "9999",
        "--iterations", "9999", "--max-steps", str(cfg.max_steps),
        "--n-envs", str(cfg.n_envs), "--n-steps", str(cfg.n_steps),
        "--n-epochs", str(cfg.n_epochs), "--batch-size", str(cfg.batch_size),
        "--lr", "1e-4", "--entropy-coef", "0.01", "--max-grad-norm", "0.5",
        "--warmup-iters", "20", "--gamma", "0.995", "--gae-lambda", "0.95",
        "--vf-coef", "0.5",
        "--reward-rule", os.path.join(_REPO_ROOT, "tests", "fixtures",
                                      "reward_rules", "reward.yaml"),
        "--masking-rule", "default",
        "--wirelength-penalty", "0.002", "--via-penalty", "0.1",
        "--wire-via-emission", "per_step",
        "--policy-net-select", "--no-drc-tokens", "--same-net-bias",
        "--disable-slot-emb", "--coord-encoding", "fourier", "--corner-mode", "45",
        "--d-model", str(cfg.d_model), "--n-heads", str(cfg.n_heads),
        "--n-layers", str(cfg.n_layers), "--d-ff", str(cfg.d_ff),
        "--device", "cuda", "--seed", str(cfg.seed),
        "--log-dir", os.path.join(scratch, "tb"),
        "--save-dir", os.path.join(scratch, "ckpt"),
        "--time-feature", "sin_remaining", "--time-feature-cap", "10000",
    ]
    # 3-val wiring: primary val drives best-ckpt; eval2/eval3 diagnostic sets
    # have no repo default — pass explicit board-list args to enable them.
    # The trainer's own _setup_evaluator builds trainer.extra_evaluators from these.
    if cfg.enable_eval:
        ev2 = cfg.eval2_boards
        ev3 = cfg.eval3_boards
        if ev2:
            argv += ["--eval2-boards", ev2, "--eval2-prefix", "val_d3b"]
        if ev3:
            argv += ["--eval3-boards", ev3, "--eval3-prefix", "val_d3a"]
    args = build_arg_parser().parse_args(argv)
    # obs_format is not exposed on the public CLI (cli_skip) — diagnostic
    # values are injected here instead.
    args.obs_format = cfg.obs_format
    return PPOTrainer(args), args


# ---------------------------------------------------------------------------
# Worker-side fingerprint (one forkserver child)
# ---------------------------------------------------------------------------
def _fp_child(conn):
    from tools.diagnostics.speed_profiler.instrument import capture_fingerprint
    conn.send(capture_fingerprint())
    conn.close()


def worker_fingerprint() -> dict:
    import multiprocessing as mp
    try:
        ctx = mp.get_context("forkserver")
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_fp_child, args=(child,))
        p.start(); child.close()
        fp = parent.recv(); p.join(timeout=15)
        return fp
    except Exception as e:
        return {"error": repr(e)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(cfg: ProfileConfig) -> dict:
    t_run0 = PC()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # --gpu-index = physical GPU index: used for both CUDA device selection
    # (multi-GPU nodes) and util sampling (NVML). An externally-set
    # CUDA_VISIBLE_DEVICES takes priority over this.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu_index))
    import torch

    from tools.diagnostics.speed_profiler import SCHEMA_VERSION
    from tools.diagnostics.speed_profiler.instrument import capture_fingerprint, write_run
    from tools.diagnostics.speed_profiler import hooks, instrument

    # Worker-side timing needs the forkserver-preload shim installed BEFORE the
    # pool spawns its first worker.
    if cfg.enable_barrier or cfg.enable_rollout_decomp:
        os.environ["CADAGENT_PROFILE_WORKER"] = "1"
        import multiprocessing as mp
        try:
            mp.set_forkserver_preload(["tools.diagnostics.speed_profiler.worker_shim"])
        except Exception:
            pass

    scratch = os.path.join(_REPO_ROOT, "var", "diagnostics", "speed_profiler",
                           f"{cfg.dataset}_e{cfg.n_envs}")
    os.makedirs(scratch, exist_ok=True)

    result: dict = {
        "run": {"host_tag": cfg.host_tag, "host": os.uname().nodename,
                "dataset": cfg.dataset, "n_envs": cfg.n_envs, "algo": "ppo",
                "n_steps": cfg.n_steps, "max_steps": cfg.max_steps,
                "batch_size": cfg.batch_size, "n_epochs": cfg.n_epochs,
                "obs_format": cfg.obs_format,
                "bf16": cfg.bf16, "compile_regions": cfg.compile_regions,
                "compile_mode": cfg.compile_mode,
                "warmup_iters": cfg.warmup_iters, "measured_iters": cfg.measured_iters},
        "fingerprint": {"main": capture_fingerprint()},
    }

    t = PC()
    trainer, targs = build_trainer(cfg, scratch)
    trainer.setup()
    # Speed-knob A/B (bf16 autocast / torch.compile regions) — policy is
    # created inside setup(), so this wiring happens right after.
    if cfg.bf16 or cfg.compile_regions:
        regions = tuple(r for r in cfg.compile_regions.split(",") if r)
        trainer.policy.configure_speed(
            bf16=cfg.bf16, compile_regions=regions,
            compile_mode=cfg.compile_mode,
        )
        print(f"[speed-knobs] bf16={cfg.bf16} compile_regions={regions} "
              f"mode={cfg.compile_mode}", flush=True)
    result["run"]["spawn_setup_s"] = round(PC() - t, 2)
    print(f"[prof] {cfg.dataset} e{cfg.n_envs} spawn+setup {result['run']['spawn_setup_s']}s", flush=True)

    result["fingerprint"]["worker"] = worker_fingerprint()

    # --- utilization sampler (pid tree = main + workers, dynamic) ---
    sampler = None
    if cfg.enable_util:
        def pid_provider():
            # Recursive children so transient EVAL pools (spawned+closed inside
            # eval_transformer) are counted too, not just the train pool.
            try:
                import psutil
                me = psutil.Process(os.getpid())
                return [me.pid] + [c.pid for c in me.children(recursive=True)]
            except Exception:
                pids = [os.getpid()]
                try:
                    pids += [p.pid for p in trainer.envs.processes if p.pid]
                except Exception:
                    pass
                return pids
        sampler = instrument.UtilSampler(cfg.gpu_index, pid_provider, dt=0.1)
        sampler.start()

    pt = instrument.PhaseTimer(sync_cuda=True)
    from methods.rl_agent.algorithms._common import policy_update_loop

    update_blocks: list[dict] = []
    rollout_blocks: list[dict] = []

    total_iters = cfg.warmup_iters + cfg.measured_iters
    for it in range(total_iters):
        if it < cfg.warmup_iters:
            # Warm iters run UNTIMED (allocator/cuDNN/NFS warm, and so pct_iter is
            # computed over measured phases only, not diluted by warm work).
            trainer.select_boards(it + 1)
            if cfg.enable_trace and it == cfg.warmup_iters - 1:
                # The trace is captured during the untimed warmup, so measured
                # iters carry zero contamination. The mirrors call prof.step()
                # once per unit-step to drive the profiler's schedule window.
                from torch.profiler import (
                    profile as _tprof, schedule as _tsched, ProfilerActivity as _TA,
                )
                tdir = os.path.dirname(cfg.out or scratch)
                def _cap(kind, n_active, fn):
                    path = os.path.join(tdir, f"trace_{kind}.json.gz")
                    with _tprof(activities=[_TA.CPU, _TA.CUDA],
                                schedule=_tsched(wait=0, warmup=2, active=n_active),
                                on_trace_ready=lambda pr: pr.export_chrome_trace(path),
                                ) as prof:
                        hooks.TRACE_CB = prof.step
                        try:
                            out = fn()
                        finally:
                            hooks.TRACE_CB = None
                    print(f"[prof] trace_{kind} -> {path}", flush=True)
                    return out
                with hooks.rollout_decomp():   # mirror supplies the step callback (bucket is discarded)
                    coll = _cap("rollout", cfg.trace_rollout_steps,
                                trainer.collect_rollout)
                buffer = trainer.compute_targets(coll)
                with hooks.update_decomp():
                    _cap("update", cfg.trace_update_mbs,
                         lambda: policy_update_loop(
                             trainer.policy, trainer.optimizer, buffer,
                             trainer.device, **trainer.update_kwargs()))
            else:
                coll = trainer.collect_rollout()
                buffer = trainer.compute_targets(coll)
                policy_update_loop(trainer.policy, trainer.optimizer, buffer,
                                   trainer.device, **trainer.update_kwargs())
            print(f"[prof] iter {it} (warm) done", flush=True)
            continue

        with pt.phase("select_boards"):
            trainer.select_boards(it + 1)
        # rollout (+ tokenize split + inline decomposition + rollout GPU cuda-event)
        if cfg.enable_rollout_decomp:
            roll_acc = instrument.CudaEventAccumulator()
            with hooks.tokenizer_timer() as tokr, hooks.rollout_decomp() as rb, \
                    hooks.transformer_pass_recorder(acc=roll_acc):
                with pt.phase("collect"):
                    coll = trainer.collect_rollout()
            gpu_roll = roll_acc.collect()["per_region_ms"].get("fwd_pass", 0.0)
            walk_roll = sum(tokr.get("walk", [])) * 1000
            rollout_blocks.append({**{k: v for k, v in rb.items()},
                                   "fwd_gpu_event_ms": gpu_roll,
                                   "walk_ms": walk_roll})
        else:
            with pt.phase("collect"):
                coll = trainer.collect_rollout()
        with pt.phase("compute_targets"):
            buffer = trainer.compute_targets(coll)

        # update (+ decomp + tokenize + cuda-event fwd_pass/backward + (B,L)).
        # Guard against a genuine VRAM-ceiling OOM (e.g. d2b + large n_envs/batch):
        # record a partial result + flag instead of crashing, so rollout/barrier
        # numbers survive and the memory ceiling is documented.
        try:
            if cfg.enable_update_decomp:
                with hooks.update_decomp() as uh, \
                        hooks.transformer_pass_recorder(acc=uh["acc"]):
                    with pt.phase("update"):
                        policy_update_loop(trainer.policy, trainer.optimizer, buffer,
                                           trainer.device, **trainer.update_kwargs())
                    update_blocks.append(hooks.summarize_update(uh))
            else:
                with pt.phase("update"):
                    policy_update_loop(trainer.policy, trainer.optimizer, buffer,
                                       trainer.device, **trainer.update_kwargs())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result["run"]["update_oom"] = True
            print(f"[prof] update OOM at iter {it} (config exceeds VRAM even after "
                  f"OOM-peel); recording partial result", flush=True)
            break
        print(f"[prof] iter {it} (meas) done", flush=True)

    result["phases"] = pt.summary(unit_ms=True)

    # --- update decomposition (median over measured iters) ---
    if update_blocks:
        result["update_decomp"] = _median_update(update_blocks)
    # --- collect inline decomposition (closes the waterfall's "residual" bucket) ---
    if rollout_blocks:
        coll_wall = result["phases"]["per_phase"].get("collect", {}).get("mean")
        result["rollout_decomp"] = _summarize_rollout(rollout_blocks, coll_wall)

    # --- H1 barrier decomposition ---
    if cfg.enable_barrier:
        from tools.diagnostics.speed_profiler.barrier import barrier_probe
        result["barrier"] = barrier_probe(trainer, cfg.barrier_steps)

    # --- Eval-Rollout: time each val set separately (3-val) ---
    if cfg.enable_eval:
        result["eval"] = _run_eval(cfg, trainer, pt)

    # --- utilization by phase ---
    if sampler is not None:
        sampler.stop(); sampler.join(timeout=3)
        result["util"] = {"scopes": ["syswide", "proctree"], "sample_dt_s": sampler.dt,
                          "per_phase": sampler.by_phase(pt.transitions)}

    try:
        trainer.envs.close()
    except Exception:
        pass

    result["run"]["total_wall_s"] = round(PC() - t_run0, 2)
    out = cfg.out or os.path.join(scratch, f"prof_{cfg.dataset}_e{cfg.n_envs}.json")
    write_run(result, out)
    print(f"[prof] wrote {out}", flush=True)
    result["_out"] = out
    return result


def _summarize_rollout(rollout_blocks: list[dict], coll_wall: float | None) -> dict:
    """Median the per-iter inline collect decomposition into the waterfall block."""
    def rmed(k):
        return statistics.median(b[k] for b in rollout_blocks)
    fwd_wall = rmed("forward") * 1000
    walk = rmed("walk_ms")
    gpu = rmed("fwd_gpu_event_ms")
    between = rmed("between") * 1000
    reset = rmed("reset") * 1000
    parts = {
        "unit": "ms/iter (measured-iter median)",
        "mask_ipc_ms": round(rmed("mask_ipc") * 1000, 1),
        "forward_wall_ms": round(fwd_wall, 1),
        "forward_split": {"walk_cpu_ms": round(walk, 1),
                          "gpu_event_ms": round(gpu, 1),
                          "launch_sync_resid_ms": round(fwd_wall - walk - gpu, 1)},
        "step_barrier_ms": round(rmed("step") * 1000, 1),
        "obs_advance_ms": round(rmed("advance") * 1000, 1),
        "collector_ms": round(between - reset, 1),
        "reset_ms": round(reset, 1),
        "n_steps": rollout_blocks[-1]["n_steps"],
    }
    ssum = (parts["mask_ipc_ms"] + fwd_wall + parts["step_barrier_ms"]
            + parts["obs_advance_ms"] + between)
    parts["sum_ms"] = round(ssum, 1)
    parts["collect_phase_ms"] = round(coll_wall, 1) if coll_wall else None
    parts["sum_closure_pct"] = round(100 * ssum / coll_wall, 1) if coll_wall else None
    # Unbucketed work outside the loop: final_values forward (collect.py:544-578)
    # + iterator glue — left as an explicit residual so the sum closes exactly
    # against collect_phase.
    if coll_wall:
        parts["unbucketed_post_loop_ms"] = round(coll_wall - ssum, 1)
    parts["note"] = ("inline iter_rollout mirror; collector=yield-gap-reset; "
                     "launch_sync_resid=forward-walk-GPUevent (approximates async overlap)")
    return parts


def _run_eval(cfg: ProfileConfig, trainer, pt) -> dict:
    """Time each val set separately (3-val), with the same inline decomposition
    buckets as collect. Reuses the trainer's own evaluators (Evaluator.run() ->
    eval_transformer, which spawns its OWN n_envs pool per call — spawn timed via
    the factory hook). Runs AFTER the barrier probe so eval pools don't pollute it.
    """
    import gc as _gc
    from tools.diagnostics.speed_profiler import hooks, instrument

    evals: list[tuple[str, object]] = []
    if trainer.evaluator is not None:
        evals.append(("val_primary", trainer.evaluator))
    evals += [(pfx, ev) for pfx, ev in trainer.extra_evaluators]
    eval_res: dict = {}
    for name, ev in evals:
        n_full = len(ev.boards)
        if name != "val_primary" and cfg.eval23_board_limit:
            ev.boards = ev.boards[: cfg.eval23_board_limit]
        n_meas = len(ev.boards)
        # Same inline decomposition as collect, per eval set (goes through the same iter_rollout)
        ev_acc = instrument.CudaEventAccumulator()
        with hooks.eval_pool_spawn_timer() as spawns, \
                hooks.tokenizer_timer() as ev_tok, hooks.rollout_decomp() as ev_rb, \
                hooks.transformer_pass_recorder(acc=ev_acc):
            with pt.phase(f"eval_{name}"):
                t0 = PC()
                try:
                    ev.run()
                    err = None
                except Exception as e:  # noqa: BLE001 — record, keep other sets
                    err = repr(e)
                wall = PC() - t0
        n_roll = n_meas * cfg.eval_n_rollouts
        row = {
            "wall_s": round(wall, 2),
            "pool_spawn_s": round(sum(spawns), 2), "n_pool_spawns": len(spawns),
            "n_boards_measured": n_meas, "n_boards_full": n_full,
            "n_rollouts_per_board": cfg.eval_n_rollouts,
            "n_rollouts_total": n_roll,
            "s_per_rollout": round(wall / max(n_roll, 1), 3),
        }
        if err:
            row["error"] = err
        # Per-set inline decomposition (same buckets as collect's rollout_decomp)
        ev_gpu = ev_acc.collect()["per_region_ms"].get("fwd_pass", 0.0)
        ev_walk = sum(ev_tok.get("walk", [])) * 1000
        ev_fw = ev_rb["forward"] * 1000
        dsum = (ev_rb["mask_ipc"] + ev_rb["forward"] + ev_rb["step"]
                + ev_rb["advance"] + ev_rb["between"]) * 1000
        row["decomp_ms"] = {
            "mask_ipc": round(ev_rb["mask_ipc"] * 1000, 1),
            "forward_wall": round(ev_fw, 1),
            "forward_split": {"walk_cpu": round(ev_walk, 1),
                              "gpu_event": round(ev_gpu, 1),
                              "launch_sync_resid": round(ev_fw - ev_walk - ev_gpu, 1)},
            "step_barrier": round(ev_rb["step"] * 1000, 1),
            "obs_advance": round(ev_rb["advance"] * 1000, 1),
            "between_bookkeeping": round(ev_rb["between"] * 1000, 1),
            "n_steps": ev_rb["n_steps"],
            "sum": round(dsum, 1),
            # Outside the loop: wave planning, pool spawn, reload_board_slot, scoring, etc.
            "unbucketed": round(wall * 1000 - dsum, 1),
            "gpu_duty_of_wall": round(ev_gpu / (wall * 1000), 3) if wall else None,
        }
        eval_res[name] = row
        print(f"[prof] eval {name}: {row}", flush=True)
        # Mirror loop._run_dual_eval: release eval's residual GPU allocation
        # before the next set / teardown.
        _gc.collect()
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass
    return eval_res


def _median_update(blocks: list[dict]) -> dict:
    """Median the per-iter update decomposition blocks into one summary."""
    keys_pc = ("evaluate", "backward", "clip", "step")
    pc = {k: round(statistics.median(b["perf_counter_ms"][k] for b in blocks), 2) for k in keys_pc}
    gpu = {}
    for k in ("fwd_pass", "backward"):
        vals = [b["gpu_active_ms"].get(k, 0.0) for b in blocks]
        gpu[k] = round(statistics.median(vals), 2)
    entry_walk = round(statistics.median(b.get("entry_walk_ms", 0.0) for b in blocks), 2)
    total = sum(pc.values())
    out = {"unit": "ms", "perf_counter_ms": pc, "gpu_active_ms": gpu,
           "entry_walk_ms": entry_walk, "total_ms": round(total, 2),
           "n_minibatches": statistics.median(b["n_minibatches_timed"] for b in blocks),
           "note": "evaluate=CPU-inclusive fwd wall (incl. per-mb walk gather); "
                   "fwd_pass/backward=cuda-event GPU; "
                   "entry_walk=uncached fallback batched walk once/update "
                   "(0 with collect walk-carry)"}
    # DDP-only (when present): median the ddp_ms field if any block carries it.
    if any("ddp_ms" in b for b in blocks):
        out["ddp_ms"] = {
            k: round(statistics.median(b.get("ddp_ms", {}).get(k, 0.0)
                                       for b in blocks), 2)
            for k in ("sync", "perm", "bcast")
        }
    return out
