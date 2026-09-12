"""Peak-VRAM budget model — preemptive batch chunking for update & rollout.

Variable board sizes make the transformer's peak memory swing per batch
(padding to the batch max ``L``). Instead of *reacting* to CUDA OOM
(``algorithms/_common.py`` sorted 1/4-peel), this models the fwd(+bwd) peak as

    peak(B, L)  ~=  c  +  a * (B*L)  +  b * (B*L^2)

and splits a batch into budget-fitting chunks *before* the forward. The three
coefficients are **measured, never derived**: a few calibration probes on real
buffer samples fit them by least squares, and every planned chunk feeds its
measured peak back in (``observe`` -> ``maybe_refit``), so model-size changes,
dtype changes and the same-net dense/absorb toggle are absorbed automatically
on the next run / next few updates. Coefficients are model-dependent but
GPU-independent; capacity is the GPU-dependent side and is re-measured from
``torch.cuda.mem_get_info()`` at plan time (adapts to 24 vs 48 GB cards and to
neighbour processes on shared GPUs).

Two independent instances are expected per trainer: the *update* model
(teacher-forced fwd+bwd, activations saved) and the *rollout* model (no-grad
forward — a completely different linear coefficient).

The quadratic term is the attention kernel's workspace (the state pass feeds
SDPA a broadcast ``(B, 1, 1, L)`` mask, so no ``L^2`` tensor of our own is in
it — ``models/v1/blocks.py::padding_attn_mask``); the linear term (saved
activations) dominates below roughly L~8k at the default model size, so
neither term alone is a safe budget metric. Both coefficients are measured,
so a change on either side is absorbed by the fit.

``SAFETY`` is a module constant, deliberately not a config knob: it covers the
allocated-vs-reserved gap (fragmentation, kernel workspaces) which is neither
model- nor GPU-specific. Systematic prediction bias is corrected by the online
refit — there is no second, safety-adjusting feedback loop, and an OOM despite
planning only halves the budget *transiently* (that batch), never permanently.

The per-chunk peak measurements need ``torch.cuda.reset_peak_memory_stats()``,
which would clobber the per-iteration ``diag/gpu_mem_peak_*`` diagnostics in
``training/loop.py`` — so this module owns ALL peak-counter resets:
measurement regions fold the running maxima into a module accumulator
(``begin/end_measured_region``) and the iteration logger drains it via
``iteration_peaks_and_reset``.
"""
from __future__ import annotations

import time
import warnings
from collections import deque
from typing import Callable

import numpy as np
import torch

__all__ = [
    "SAFETY",
    "MemBudgetModel",
    "run_calibration",
    "begin_measured_region",
    "end_measured_region",
    "iteration_peaks_and_reset",
]

SAFETY = 0.8            # headroom for the allocated-peak vs reserved/fragmentation gap
MIN_FIT_POINTS = 4      # 3 coefficients + 1; fewer -> refuse to fit
# Probe B values reach toward real minibatch sizes: fitting at B<=8 and
# extrapolating 64x (to batch 512) let the ill-conditioned quadratic term blow
# up ~20x on d2a (260713 A/B). B=64 keeps the extrapolation within ~8x.
PROBE_B_VALUES = (4, 16, 64)
_RING_SIZE = 256        # observation ring buffer (drop-oldest)
_REFIT_EVERY = 16       # refit after this many new observations
# cudaMemGetInfo stalls ~45ms/call when async GPU work is in flight (measured
# 260713 dissect; idle it is ~0.02ms) — a per-step/minibatch call turns the
# planner into a device synchronizer. Capacity is therefore cached with a TTL:
# worst case one ~45ms stall per TTL window (<1%), while neighbour-process
# VRAM changes still propagate within seconds.
_CAPACITY_TTL_S = 5.0
# Confidence-bound capacity (2026-07-13): the effective safety is
# SAFETY / q99(measured/predicted) clamped to <= SAFETY_MAX — i.e. capacity
# targets P(true peak > allowed) ~ 1%. Cold start (< _RESID_MIN_N residual
# points) falls back to plain SAFETY. Measured motivation: the probe-only fit
# under-predicts ~+14% at B-extrapolation (device hit 95.3% at SAFETY 0.8),
# where this bound TIGHTENS capacity; after the online refit flips residuals
# conservative, it recovers head-room up to SAFETY_MAX. SAFETY_MAX < 1 keeps
# the margin residuals cannot see (reserved-vs-allocated gap, neighbour
# jitter between capacity refreshes).
SAFETY_MAX = 0.90
_RESID_MIN_N = 32
_RESID_Q = 99.0


# ---------------------------------------------------------------------------
# CUDA peak-counter ownership (measurement regions vs iteration diagnostics)
# ---------------------------------------------------------------------------
_folded_peaks = {"alloc": 0, "reserved": 0}


def begin_measured_region() -> int:
    """Start a per-chunk peak measurement; returns the allocated-bytes base.

    Folds the current peak counters into the iteration accumulator before
    resetting them, so ``iteration_peaks_and_reset`` still sees the true
    iteration high-water mark.
    """
    _folded_peaks["alloc"] = max(
        _folded_peaks["alloc"], torch.cuda.max_memory_allocated(),
    )
    _folded_peaks["reserved"] = max(
        _folded_peaks["reserved"], torch.cuda.max_memory_reserved(),
    )
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def end_measured_region(base: int) -> int:
    """Peak allocated bytes *above* ``base`` since ``begin_measured_region``."""
    return torch.cuda.max_memory_allocated() - base


def iteration_peaks_and_reset() -> tuple[int, int]:
    """(peak_allocated, peak_reserved) over the whole iteration, then reset.

    Drop-in for the previous ``max_memory_*()`` + ``reset_peak_memory_stats()``
    logging pattern: identical when no measured region ran this iteration.
    """
    alloc = max(_folded_peaks["alloc"], torch.cuda.max_memory_allocated())
    reserved = max(_folded_peaks["reserved"], torch.cuda.max_memory_reserved())
    _folded_peaks["alloc"] = 0
    _folded_peaks["reserved"] = 0
    torch.cuda.reset_peak_memory_stats()
    return alloc, reserved


def _cuda_headroom_bytes() -> float:
    """Bytes a new chunk may grow into: device free + our allocator's
    reserved-but-unallocated cache (invisible to ``mem_get_info`` but
    reusable by us). Device-level ``free`` reflects neighbour processes."""
    free, _total = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return float(free + reusable)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class MemBudgetModel:
    """peak(B, L) ~= c + a*(B*L) + b*(B*L^2), measured & refit online.

    Args:
        headroom_fn: injectable ``() -> bytes`` capacity source (CPU tests);
            defaults to the CUDA free+reusable measurement.
        reserve_fn: optional ``() -> bytes`` subtracted from capacity — e.g.
            AdamW state (2x params) that is lazily created on the first
            ``optimizer.step()`` and therefore invisible to ``headroom_fn``
            at calibration time.
    """

    def __init__(
        self,
        headroom_fn: Callable[[], float] | None = None,
        reserve_fn: Callable[[], float] | None = None,
    ) -> None:
        self.coeffs: np.ndarray | None = None      # (c, a, b) float64, >= 0
        self._points: deque[tuple[int, int, float]] = deque(maxlen=_RING_SIZE)
        self._n_seen = 0
        self._n_seen_at_fit = 0
        self._headroom_fn = headroom_fn or _cuda_headroom_bytes
        self._reserve_fn = reserve_fn
        self._cap_cache: float | None = None
        self._cap_t = 0.0

    # -- prediction ----------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.coeffs is not None

    def predict(self, B: int, L: int) -> float:
        """Predicted peak bytes of one fwd(+bwd) over ``B`` rows padded to ``L``."""
        c, a, b = self.coeffs
        return float(c + a * (B * L) + b * (B * L * L))

    def effective_safety(self) -> float:
        """Confidence-bound safety: ``min(SAFETY_MAX, SAFETY / q99(meas/pred))``.

        Residual ratios are recomputed against the CURRENT coefficients from
        the observation ring (so a refit immediately refreshes them); with
        fewer than ``_RESID_MIN_N`` usable points this is plain ``SAFETY``.
        """
        if not self.ready or len(self._points) < _RESID_MIN_N:
            return SAFETY
        ratios = [
            peak / p for (B, L, peak) in self._points
            if (p := self.predict(B, L)) > 0
        ]
        if len(ratios) < _RESID_MIN_N:
            return SAFETY
        q = float(np.percentile(ratios, _RESID_Q))
        if q <= 0:
            return SAFETY
        return min(SAFETY_MAX, SAFETY / q)

    def capacity(self, fresh: bool = False) -> float:
        """Allowed predicted peak for one chunk.

        Re-measures at most every ``_CAPACITY_TTL_S`` (see the constant's
        comment — cudaMemGetInfo stalls with in-flight GPU work); ``fresh=True``
        forces a re-measure. Scaled by :meth:`effective_safety` (confidence
        bound targeting ~1% under-prediction tail).
        """
        now = time.monotonic()
        if (fresh or self._cap_cache is None
                or now - self._cap_t > _CAPACITY_TTL_S):
            cap = self.effective_safety() * self._headroom_fn()
            if self._reserve_fn is not None:
                cap -= self._reserve_fn()
            self._cap_cache = cap
            self._cap_t = now
        return self._cap_cache

    # -- planning --------------------------------------------------------
    def plan_chunks(
        self, seq_lens: list[int], limit: float | None = None,
    ) -> list[list[int]]:
        """Partition positions ``0..len(seq_lens)-1`` into budget-fitting chunks.

        Greedy fill over positions sorted ascending by ``seq_lens`` (ties by
        position): a position joins the current chunk unless the chunk's
        predicted peak with it — ``predict(B+1, L_new)``; ``L_new`` is the
        chunk max because of the ascending order — exceeds ``limit``.
        A singleton chunk may exceed the limit (nothing smaller exists; the
        caller's OOM backstop covers a real failure).
        """
        if limit is None:
            limit = self.capacity()
        order = sorted(range(len(seq_lens)), key=lambda p: (seq_lens[p], p))
        chunks: list[list[int]] = []
        cur: list[int] = []
        for p in order:
            if cur and self.predict(len(cur) + 1, seq_lens[p]) > limit:
                chunks.append(cur)
                cur = []
            cur.append(p)
        if cur:
            chunks.append(cur)
        return chunks

    # -- measurement / fitting -------------------------------------------
    def observe(self, B: int, L: int, peak_bytes: float) -> None:
        """Record one measured (chunk or probe) peak."""
        self._points.append((int(B), int(L), float(peak_bytes)))
        self._n_seen += 1

    def fit(self) -> bool:
        """Least-squares fit of (c, a, b) over the ring buffer.

        Refuses (returns False, keeps old coeffs) with < MIN_FIT_POINTS points
        or without >= 2 distinct L values — with a single L the B*L and B*L^2
        columns are collinear and the fit is meaningless. Negative
        coefficients are clamped to 0 (they can only arise from noise).
        """
        pts = list(self._points)
        if len(pts) < MIN_FIT_POINTS:
            return False
        if len({L for _, L, _ in pts}) < 2:
            return False
        A = np.array(
            [[1.0, B * L, float(B) * L * L] for B, L, _ in pts],
            dtype=np.float64,
        )
        y = np.array([peak for _, _, peak in pts], dtype=np.float64)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.coeffs = np.clip(coef, 0.0, None)
        self._n_seen_at_fit = self._n_seen
        return True

    def maybe_refit(self) -> None:
        """Refit once enough new observations accumulated (call once per update)."""
        if self.ready and self._n_seen - self._n_seen_at_fit >= _REFIT_EVERY:
            self.fit()


# ---------------------------------------------------------------------------
# Startup calibration
# ---------------------------------------------------------------------------
def run_calibration(
    model: MemBudgetModel,
    probe_fn: Callable[[int, int], float],
    seq_lens: list[int],
    b_values: tuple[int, ...] = PROBE_B_VALUES,
    *,
    label: str = "mem_budget",
) -> bool:
    """Probe ``peak(B, L)`` on real samples and fit the model.

    ``probe_fn(sample_pos, B)`` runs one fwd(+bwd) on ``B`` copies of sample
    ``sample_pos`` and returns the measured peak bytes; it may raise
    ``torch.cuda.OutOfMemoryError``. Probes use the shortest and longest
    sample (the two L groups the fit needs) x ``b_values``, ordered by
    ascending cost proxy ``B*L^2`` — so a probe OOM is a capacity upper bound:
    every remaining probe would be at least as big, and the points already
    collected still fit the model.

    Returns True when the fit succeeded; on False the model stays not-ready
    (planner disabled — callers fall back to reactive OOM recovery).
    """
    if not seq_lens:
        return False
    lo = min(range(len(seq_lens)), key=seq_lens.__getitem__)
    hi = max(range(len(seq_lens)), key=seq_lens.__getitem__)
    plan = [(pos, B) for pos in dict.fromkeys((lo, hi)) for B in b_values]
    plan.sort(key=lambda t: t[1] * seq_lens[t[0]] ** 2)
    for pos, B in plan:
        try:
            peak = probe_fn(pos, B)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break   # capacity bound reached; points below it are enough
        model.observe(B, seq_lens[pos], peak)
    ok = model.fit()
    if not ok:
        warnings.warn(
            f"{label} calibration failed (<{MIN_FIT_POINTS} points or a single "
            "sequence-length group) — preemptive chunking disabled; falling "
            "back to reactive OOM recovery.",
            RuntimeWarning, stacklevel=2,
        )
    return ok
