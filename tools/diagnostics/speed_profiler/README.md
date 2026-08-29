# speed_profiler — training-loop CPU/GPU bottleneck profiler

Reusable profiler for the Decoder-only PPO training loop. Drives the **real**
`PPOTrainer` code paths (`select_boards → collect_rollout → compute_targets →
policy_update_loop`) and measures where the wall-clock goes, with the
measurement-validity corrections that make the numbers trustworthy on a shared
GPU node.

## Run

```bash
conda activate cadagent
export PYTHONPATH=$(pwd):build_rl/pcbnew/python/rl:.

python scripts/profile.py --dataset d2a --n-envs 64 --waterfall       # d2a (synth 2L)
python scripts/profile.py --dataset d3b --n-envs 128 --host-tag l40   # d3b (real, medium)
python -m tools.diagnostics.speed_profiler.waterfall <data_dir>       # (re)render the table
```

In practice the only required arguments are `--dataset {d2a,d3b}` and
`--n-envs`; the rest default to a representative config (`--n-steps 512
--max-steps 256 --batch-size 256 --n-epochs 4 --warmup-iters 1
--measured-iters 2 --barrier-steps 100`).

**`--mode`** — `full` (default, every sub-decomposition) · **`light`**
(lightweight A/B: all instrumentation hooks off, phase timers only =
Collect/Update/ITER — without cuda-event syncs the **absolute speed is close to
real training**; the waterfall fills only its top 3 rows and shows "—" below) ·
`barrier` (step-barrier probe only). **Rule of thumb: use `full` for bottleneck
location and composition, `light` for absolute speed and A/B comparisons** —
instrumentation inflates absolute numbers considerably, especially in fp32
(measured: full 603 vs light 449). Individual measurement blocks toggle off with
`--no-<capability>` (`--no-barrier --no-util --no-update-decomp
--no-rollout-decomp`); `--trace` captures a chrome trace during the warmup
iteration; `--eval` measures the 3-val Eval-Rollout (opt-in, expensive because
each set spawns its own pool — the diagnostic sets need explicit board-list txt
files via `--eval2-boards` / `--eval3-boards`).

One config per process (KiCad `RLRouter` singleton). Artifacts (JSON +
`waterfall.html`) land under `var/diagnostics/speed_profiler/` (gitignored).

## What it measures

| block | what | how (validity rule) |
|---|---|---|
| `phases` | Train-Rollout / Update / targets / select_boards, %iter | perf_counter; `cuda.synchronize` at phase ENTRY only (coarse timers are already sync-truthful via per-step `.cpu()` / per-mb `.item()`). Warm iters run untimed. |
| `update_decomp` | evaluate / backward / clip / step + **entry walk** + (DDP) sync/perm/bcast | `fwd_pass`+`backward` are cuda-event GPU; the in-forward CPU tokenize is the waterfall's `up_h2d` row (= `evaluate − fwd_pass`). **`entry_walk_ms`** = `walk_samples` once per update (the largest non-scaling CPU cost). **`ddp_ms`** = `{sync,perm,bcast}` present only under `--update-gpus>1` (otherwise the waterfall renders "—"), so the row scheme is identical for single- and multi-GPU profiles. |
| `barrier` | 5 serial barriers/step (step + 4 mask) → send / worker / straggler / unpickle | `mpc.wait`+`recv_bytes` (order+transport) → deferred `gc`-disabled `pickle.loads` (isolated main-serial unpickle). `worker_compute` from `info["_prof"]` (worker_shim). |
| `util` | CPU + GPU per phase, min/mean/max/p90 — rendered as the waterfall's "utilization" section | **`proctree`** (main+worker PID tree) is primary; **`syswide`** is a shared-node contamination tripwire only. GPU via NVML (nvidia-smi fallback). |
| `fingerprint` | threads / precision / affinity / cores / gpu | captured in **both** main and a forkserver worker (they differ; the biggest comparability trap). |

## Design — zero base edits

`methods/**` and `pcb_world/**` are **not modified**:

- **Main-proc** instrumentation is runtime monkeypatch (`hooks.py`): update
  decomposition, the tokenizer's existing dormant `_BATCHED_TIMER_HOOK`, and the
  `_run_transformer` pass/L recorder.
- **Worker-side** timing rides a **forkserver-preload shim** (`worker_shim.py`):
  `mp.set_forkserver_preload(...)` before the pool spawns makes the forkserver
  server rebind `subproc._decoder_worker` to a timed mirror that stamps
  `info["_prof"]["worker_compute_s"]`. Numerics/RNG untouched (timers only).
- The timed mirrors (hooks.py / worker_shim.py) are pinned to their base
  functions by source digest in `mirror_contract.py` (static ast extraction,
  stdlib-only — the same check runs pre-push as `tools/docs/check_docs.py`
  `mirror-sync`);
  [tests/test_diagnostics/test_speed_profiler_mirrors.py](../../../tests/test_diagnostics/test_speed_profiler_mirrors.py)
  fails on base drift with a re-sync procedure in its message.

## Modules

`instrument.py` (PhaseTimer / CudaEventAccumulator / UtilSampler + fingerprint/
stats/JSON writer) · `hooks.py` (main-proc toggles) ·
`worker_shim.py` (forkserver-preload) · `barrier.py` (step-barrier probe) ·
`mirror_contract.py` (mirror↔base source-digest pins) ·
`driver.py` (build real trainer + orchestrate) · `cli.py` / `__main__.py` ·
`waterfall.py` (JSON → decomposition-table HTML, the single display surface).
Front door: `scripts/profile.py`. `util` is rendered as a waterfall section.

## Shared-node hygiene

On a shared multi-tenant GPU node, confirm before a run that the target GPU is
idle (`nvidia-smi --query-compute-apps`) and no other user is loading the CPU;
the `syswide` util scope flags mid-run contamination. Never oversubscribe a node
another user is on.
