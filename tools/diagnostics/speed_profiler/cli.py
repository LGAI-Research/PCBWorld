"""CLI for the training-loop speed profiler.

Two reference modes (same file, only the options differ — use these two as
the baseline for further experiments):

    # (1) full — full decomposition (rollout/update/barrier/MFU/util; ~5-15 min per run)
    python scripts/profile.py --dataset d2a --n-envs 64 --out var/.../prof.json

    # (2) barrier — detail focused on the step-barrier probe (send/w_max/idle/
    #    unpickle; spawn+probe only, no iteration, ~1-2 min per run. For quick
    #    cap A/B comparisons etc. Measures a cold state, so absolute values run
    #    lower than in-loop — comparison use only.)
    python scripts/profile.py --mode barrier --dataset d3b --n-envs 64 \
        --engine-threads 1 --gpu-index 0 --out var/.../probe.json

Uses whichever host it runs on as-is (no hardcoding); the GPU is selected via
``--gpu-index`` (physical index; applied to both CUDA_VISIBLE_DEVICES and util
sampling), and the DRC thread cap is set per-run via ``--engine-threads``
(KICAD_ENGINE_THREADS; inherits the environment if unset, engine default 1).
One config per process (KiCad singleton). Artifacts land under
``var/`` (gitignored). Requires the ``cadagent`` conda env + PYTHONPATH/LD_LIBRARY_PATH.
"""
from __future__ import annotations

import argparse
import os

from tools.diagnostics.speed_profiler.driver import ProfileConfig, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="profile", description=__doc__)
    p.add_argument("--mode", choices=["full", "light", "barrier"], default="full",
                   help="full=complete detailed breakdown (default) · light=lightweight "
                        "A/B: Collect/Update phases only (all instrumentation hooks off "
                        "→ no cuda-event sync, absolute speed close to real training) · "
                        "barrier=step-barrier probe only"
                        " (warmup/measured 0, instrumentation off except the probe)")
    p.add_argument("--engine-threads", default=None,
                   help="KiCad DRC thread-pool cap → KICAD_ENGINE_THREADS (int or"
                        " 'physical'; inherits from env if unset)")
    p.add_argument("--dataset", default="d2a",
                   choices=["d2a", "d3b"])
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--obs-format", default="indexed", choices=["json", "indexed"],
                   help="env obs format A/B — default indexed (same as the RL "
                        "training path), json = for measuring the legacy dict path")
    p.add_argument("--warmup-iters", type=int, default=1)
    p.add_argument("--measured-iters", type=int, default=2)
    p.add_argument("--barrier-steps", type=int, default=100)  # for tail (straggler/idle_waste) statistics
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--host-tag", default="l40")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    # speed-knob A/B (bf16 autocast / torch.compile regions and mode)
    p.add_argument("--bf16", action="store_true",
                   help="autocast (bfloat16) only the transformer body (stack+decode); "
                        "logits/loss/optimizer stay fp32")
    p.add_argument("--compile-regions", default="",
                   help="comma list of torch.compile regions: stack,decode,heads")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    # capability toggles (default-on for the H3/H1 spine; --no-* to disable)
    p.add_argument("--no-util", action="store_true")
    p.add_argument("--no-barrier", action="store_true")
    p.add_argument("--no-update-decomp", action="store_true")
    p.add_argument("--no-rollout-decomp", action="store_true")
    p.add_argument("--trace", action="store_true",
                   help="capture a torch.profiler chrome trace during the untimed warmup iter (rollout/update)")
    # Eval-Rollout (3-val) — opt-in (spawns its own n_envs pool per val set)
    p.add_argument("--eval", action="store_true",
                   help="measure Eval-Rollout: primary + val_d3b + val_d3a, timed per set")
    p.add_argument("--eval-rollouts", type=int, default=4)
    p.add_argument("--eval-board-limit", type=int, default=8,
                   help="primary val set board cap (trainer-applied)")
    p.add_argument("--eval2-boards", default=None,
                   help="val_d3b diagnostic board list txt (explicit; omit to skip)")
    p.add_argument("--eval3-boards", default=None,
                   help="val_d3a diagnostic board list txt (explicit; omit to skip)")
    p.add_argument("--eval23-limit", type=int, default=10,
                   help="board cap for the diagnostic sets (0 = full list)")
    p.add_argument("--out", default=None)
    p.add_argument("--waterfall", action="store_true",
                   help="after the run, regenerate waterfall.html from all prof JSONs in the out directory")
    return p


def main() -> None:
    a = build_parser().parse_args()
    if a.engine_threads is not None:
        os.environ["KICAD_ENGINE_THREADS"] = str(a.engine_threads)
    if a.mode == "light":  # light A/B preset: phase timers only (Collect/Update absolute speed)
        # All detailed decomposition/samplers off -> removes cuda-event sync
        # overhead -> wall-clock time approaches real training (A/B).
        # waterfall fills only the Collect/Update/ITER totals; sub-rows show "—".
        a.no_update_decomp = a.no_rollout_decomp = True
        a.no_util = a.no_barrier = True
    if a.mode == "barrier":  # probe-only preset: spawn+probe only, no iteration
        a.warmup_iters = a.measured_iters = 0
        a.no_update_decomp = a.no_rollout_decomp = a.no_util = True
        a.no_barrier = False
    cfg = ProfileConfig(
        dataset=a.dataset, n_envs=a.n_envs, n_steps=a.n_steps, max_steps=a.max_steps,
        batch_size=a.batch_size, n_epochs=a.n_epochs, seed=a.seed,
        obs_format=a.obs_format,
        warmup_iters=a.warmup_iters, measured_iters=a.measured_iters,
        barrier_steps=a.barrier_steps, gpu_index=a.gpu_index, host_tag=a.host_tag,
        d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers, d_ff=a.d_ff,
        bf16=a.bf16, compile_regions=a.compile_regions, compile_mode=a.compile_mode,
        enable_util=not a.no_util, enable_barrier=not a.no_barrier,
        enable_update_decomp=not a.no_update_decomp,
        enable_rollout_decomp=not a.no_rollout_decomp,
        enable_trace=a.trace,
        enable_eval=a.eval, eval_n_rollouts=a.eval_rollouts,
        eval_board_limit=a.eval_board_limit,
        eval2_boards=a.eval2_boards, eval3_boards=a.eval3_boards,
        eval23_board_limit=(a.eval23_limit or None),
        out=a.out,
    )
    result = run(cfg)
    if a.waterfall:
        from tools.diagnostics.speed_profiler.waterfall import generate
        d = os.path.dirname(os.path.abspath(result["_out"]))
        generate(d, os.path.join(d, "waterfall.html"))


if __name__ == "__main__":
    main()
