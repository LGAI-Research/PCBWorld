"""Measurement primitives — GPU-truthful timing + phase-tagged utilization
(+ JSON schema helpers: fingerprint / stats / versioned writer).

The load-bearing measurement-validity rules (see the plan §C) are enforced here:

* **GPU time = cuda.Event, never bare perf_counter.** Sub-region GPU numbers from
  bare ``perf_counter`` measure kernel *launch*, not completion (the real time
  spills into whichever later op syncs). :class:`CudaEventAccumulator` records
  event pairs and reads ``elapsed_time`` after a *single* ``synchronize()`` per
  phase boundary — async, so it doesn't serialize the overlap it measures.
* **Coarse phase timers are already sync-truthful** (rollout drains via per-step
  ``.cpu()``, update via per-minibatch ``.item()``), so :class:`PhaseTimer` adds a
  ``cuda.synchronize()`` only at phase ENTRY (stop the previous phase's tail
  kernels leaking in) — never inside a measured region.
* **Utilization is per-process-tree, never system-wide** on a shared multi-tenant GPU pool;
  syswide is kept only as a contamination tripwire. Phase tagging is by
  transition-timestamp containment (no lockless cross-thread string read).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable

from tools.diagnostics.speed_profiler import SCHEMA_VERSION

PC = time.perf_counter


# ---------------------------------------------------------------------------
# Phase timing (coarse, sync-truthful; sync only at phase entry)
# ---------------------------------------------------------------------------
class PhaseTimer:
    """Wall-time per phase + a transition log for sample tagging.

    ``phase(name)`` is a context manager: it syncs the GPU at ENTRY (so the prior
    phase's trailing kernels are charged to the prior phase), stamps a transition,
    times the block with ``perf_counter``, and accumulates per-phase seconds. The
    transition log ``(t, name)`` lets :meth:`UtilSampler.by_phase` bucket each
    background sample by interval containment.
    """

    def __init__(self, sync_cuda: bool = True) -> None:
        self._sync = sync_cuda
        self.per_phase: dict[str, list[float]] = defaultdict(list)
        self.transitions: list[tuple[float, str]] = []
        self._torch = None
        if sync_cuda:
            try:
                import torch
                self._torch = torch if torch.cuda.is_available() else None
            except Exception:
                self._torch = None

    def _mark(self, name: str) -> float:
        if self._torch is not None:
            self._torch.cuda.synchronize()
        t = PC()
        self.transitions.append((t, name))
        return t

    @contextmanager
    def phase(self, name: str):
        t0 = self._mark(name)
        try:
            yield
        finally:
            t1 = self._mark(f"__end__:{name}")
            self.per_phase[name].append(t1 - t0)

    def summary(self, unit_ms: bool = True) -> dict:
        scale = 1000.0 if unit_ms else 1.0
        out: dict[str, dict] = {}
        totals = []
        for name, xs in self.per_phase.items():
            arr = [x * scale for x in xs]
            out[name] = {
                "mean": round(statistics.mean(arr), 3),
                "p90": round(sorted(arr)[max(0, int(round(0.9 * (len(arr) - 1))))], 3),
                "max": round(max(arr), 3),
                "n": len(arr),
            }
            totals.append(sum(xs))
        iter_total = sum(sum(xs) for xs in self.per_phase.values())
        for name in out:
            s = sum(self.per_phase[name])
            out[name]["pct_iter"] = round(100 * s / iter_total, 1) if iter_total else None
        return {"unit": "ms" if unit_ms else "s", "per_phase": out}


# ---------------------------------------------------------------------------
# GPU-truthful sub-region timing (cuda events)
# ---------------------------------------------------------------------------
class CudaEventAccumulator:
    """Accumulate GPU-active ms per named sub-region via cuda events.

    ``span(name)`` records a start/end ``cuda.Event`` pair on the default stream
    (async — no stall). :meth:`collect` does ONE ``synchronize`` then reads every
    pair's ``elapsed_time``, accumulating per-name totals and a launch-anchored
    (name, launch_perf_counter, gpu_ms) list for the timeline. On CPU/no-CUDA it
    degrades to perf_counter (labeled).
    """

    def __init__(self) -> None:
        self.pairs: list[tuple[str, object, object, float]] = []  # name, start, end, launch_t
        self.cpu_spans: list[tuple[str, float, float]] = []
        try:
            import torch
            self._torch = torch if torch.cuda.is_available() else None
        except Exception:
            self._torch = None

    @contextmanager
    def span(self, name: str):
        if self._torch is None:
            t0 = PC()
            try:
                yield
            finally:
                self.cpu_spans.append((name, t0, PC()))
            return
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        launch_t = PC()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pairs.append((name, start, end, launch_t))

    def collect(self) -> dict:
        """Sync once, then read all event pairs. Returns per-region totals (ms) +
        launch-anchored spans for the Gantt (labeled gpu-event vs cpu-perf)."""
        per: dict[str, float] = defaultdict(float)
        spans: list[dict] = []
        if self._torch is not None and self.pairs:
            self._torch.cuda.synchronize()
            for name, start, end, launch_t in self.pairs:
                ms = start.elapsed_time(end)  # device-timeline ms
                per[name] += ms
                spans.append({"region": name, "device": "gpu", "timer": "cuda-event",
                              "start_ms": round(launch_t * 1000, 4),
                              "end_ms": round(launch_t * 1000 + ms, 4)})
        for name, t0, t1 in self.cpu_spans:
            per[name] += (t1 - t0) * 1000
            spans.append({"region": name, "device": "cpu", "timer": "perf_counter",
                          "start_ms": round(t0 * 1000, 4), "end_ms": round(t1 * 1000, 4)})
        return {"per_region_ms": {k: round(v, 3) for k, v in per.items()}, "spans": spans}


# ---------------------------------------------------------------------------
# GPU sampling backend — NVML in-process preferred over nvidia-smi fork/exec
# ---------------------------------------------------------------------------
class _GpuReader:
    """util.gpu / util.memory / mem_used / power via NVML (no fork). Falls back to
    nvidia-smi only if pynvml is unavailable (a per-call fork/exec is a CPU thief
    on the very proc doing the unpickle we measure — avoid on the hot path)."""

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self._h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self._pynvml = None

    def read(self) -> dict:
        if self._h is not None:
            p = self._pynvml
            try:
                u = p.nvmlDeviceGetUtilizationRates(self._h)
                mem = p.nvmlDeviceGetMemoryInfo(self._h)
                pw = p.nvmlDeviceGetPowerUsage(self._h) / 1000.0  # mW -> W
                return {"gpu_util": float(u.gpu), "mem_util": float(u.memory),
                        "gpu_mem_mb": mem.used / 2**20, "power_w": pw}
            except Exception:
                pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                 "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
                capture_output=True, text=True, timeout=5)
            gu, mu, mem, pw = out.stdout.strip().split(",")
            return {"gpu_util": float(gu), "mem_util": float(mu),
                    "gpu_mem_mb": float(mem), "power_w": float(pw)}
        except Exception:
            nan = float("nan")
            return {"gpu_util": nan, "mem_util": nan, "gpu_mem_mb": nan, "power_w": nan}


# ---------------------------------------------------------------------------
# Utilization sampler — syswide (tripwire) + proctree (primary), phase-tagged
# ---------------------------------------------------------------------------
class UtilSampler(threading.Thread):
    """Background sampler: system-wide CPU (contamination tripwire) + per-process-
    tree CPU (the only valid number on a shared node) + GPU (NVML). Tagged to
    phases by transition-timestamp containment, not a lockless string read.

    ``pid_provider`` returns the live PID set to sum (main + forkserver workers;
    workers change per pool spawn, so it is re-queried each tick). Prefer running
    this in a SIDECAR process (see driver) so its GIL/CPU cost doesn't perturb the
    main-proc serial unpickle it is trying to measure.
    """

    def __init__(self, gpu_index: int = 0, pid_provider: Callable[[], list[int]] | None = None,
                 dt: float = 0.1, gpu_every: int = 3) -> None:
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        self.dt = dt
        self.gpu_every = gpu_every
        self.ncpu = psutil.cpu_count(logical=True)
        self.pid_provider = pid_provider or (lambda: [os.getpid()])
        self._gpu = _GpuReader(gpu_index)
        self._stop_ev = threading.Event()
        self.samples: list[dict] = []
        self._procs: dict[int, object] = {}
        psutil.cpu_percent(interval=None)  # prime syswide

    def _proctree_cores(self) -> float:
        total = 0.0
        want = set(self.pid_provider())
        for pid in want:
            p = self._procs.get(pid)
            if p is None:
                try:
                    p = self.psutil.Process(pid)
                    p.cpu_percent(interval=None)  # prime (first read ~0)
                    self._procs[pid] = p
                except Exception:
                    continue
            try:
                total += p.cpu_percent(interval=None) / 100.0
            except Exception:
                self._procs.pop(pid, None)
        for pid in list(self._procs):
            if pid not in want:
                self._procs.pop(pid, None)
        return total

    def run(self) -> None:
        i = 0
        g = self._gpu.read()
        while not self._stop_ev.is_set():
            sys_cpu = self.psutil.cpu_percent(interval=None)
            tree_cores = self._proctree_cores()
            if i % self.gpu_every == 0:
                g = self._gpu.read()
            self.samples.append({
                "t": PC(),
                "syswide_cores": sys_cpu / 100.0 * self.ncpu,
                "proctree_cores": tree_cores,
                "gpu_util": g["gpu_util"], "gpu_mem_mb": g["gpu_mem_mb"],
                "mem_util": g.get("mem_util"), "power_w": g.get("power_w"),
            })
            i += 1
            self._stop_ev.wait(self.dt)

    def stop(self) -> None:
        self._stop_ev.set()

    def by_phase(self, transitions: list[tuple[float, str]]) -> dict:
        """Bucket samples into phases by interval containment against the phase
        transition log; drop samples within ±dt of a boundary. Returns
        {phase: {scope: {resource: stats}}} with min/mean/max/p90."""
        # Build (start, end, name) intervals from transitions, ignoring __end__ marks.
        marks = [(t, n) for t, n in transitions]
        intervals = []
        for k in range(len(marks) - 1):
            t0, n0 = marks[k]; t1, _ = marks[k + 1]
            if not n0.startswith("__end__:"):
                intervals.append((t0, t1, n0))
        buckets: dict[str, list[dict]] = defaultdict(list)
        for s in self.samples:
            for t0, t1, name in intervals:
                if t0 + self.dt <= s["t"] <= t1 - self.dt:
                    buckets[name].append(s)
                    break
        out: dict[str, dict] = {}
        for name, ss in buckets.items():
            out[name] = {
                "syswide": {"cpu_cores": stats([x["syswide_cores"] for x in ss])},
                "proctree": {"cpu_cores": stats([x["proctree_cores"] for x in ss])},
                "gpu": {
                    "gpu_util": stats([x["gpu_util"] for x in ss]),
                    "gpu_mem_mb": stats([x["gpu_mem_mb"] for x in ss]),
                    "power_w": stats([x["power_w"] for x in ss if x.get("power_w") is not None]),
                },
                "n_samples": len(ss),
            }
        return out

# ---------------------------------------------------------------------------
# JSON schema helpers — env fingerprint, stats, versioned writer
# Every number is uninterpretable without a main+worker fingerprint and a
# schema_version stamp.
# ---------------------------------------------------------------------------
def p90(values: list[float]) -> float:
    """Interpolation-free nearest-rank p90 (the package-wide definition —
    barrier.py and stats() share this one implementation)."""
    xs_sorted = sorted(values)
    return xs_sorted[max(0, int(round(0.9 * (len(xs_sorted) - 1))))]


def stats(values: list[float]) -> dict[str, float | None]:
    """min / mean / max / p90 over a sample (None-safe on empty).

    Both mean AND max are always reported (a duty-cycle mean hides the burst the
    max exposes).
    """
    xs = [float(v) for v in values if v is not None and v == v]  # drop None/NaN
    if not xs:
        return {"min": None, "mean": None, "max": None, "p90": None, "n": 0}
    return {
        "min": round(min(xs), 4),
        "mean": round(statistics.mean(xs), 4),
        "max": round(max(xs), 4),
        "p90": round(p90(xs), 4),
        "n": len(xs),
    }


def capture_fingerprint() -> dict[str, Any]:
    """Record the env knobs that silently change CPU-region ms (-> duty) and the
    MFU peak selection. Call in the main proc; the worker shim captures its own
    copy inside one forkserver worker (threads/precision are set at the worker's
    torch import, not inherited from a post-fork main-proc setter).
    """
    fp: dict[str, Any] = {
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "pid": os.getpid(),
    }
    try:
        fp["affinity_size"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        fp["affinity_size"] = None
    try:
        import psutil
        fp["cores_physical"] = psutil.cpu_count(logical=False)
        fp["cores_logical"] = psutil.cpu_count(logical=True)
    except Exception:
        fp["cores_physical"] = fp["cores_logical"] = None
    try:
        import torch
        fp["torch_version"] = torch.__version__
        fp["torch_num_threads"] = torch.get_num_threads()
        fp["torch_num_interop_threads"] = torch.get_num_interop_threads()
        fp["matmul_precision"] = torch.get_float32_matmul_precision()
        fp["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        fp["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        if torch.cuda.is_available():
            fp["gpu_name"] = torch.cuda.get_device_name(0)
            fp["cuda_version"] = torch.version.cuda
        else:
            fp["gpu_name"] = None
    except Exception as e:
        fp["torch_error"] = repr(e)
    try:
        import multiprocessing as mp
        fp["start_method"] = mp.get_start_method(allow_none=True)
    except Exception:
        fp["start_method"] = None
    return fp


def write_run(result: dict[str, Any], path: str) -> str:
    """Stamp schema_version and write the run document as indented JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    doc = {"schema_version": SCHEMA_VERSION, **result}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    return path
