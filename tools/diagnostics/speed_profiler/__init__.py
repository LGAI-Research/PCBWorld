"""Training-loop speed profiler.

Reusable CPU/GPU bottleneck profiler for the Decoder-only PPO training loop.
Drives the REAL ``PPOTrainer`` code paths and measures the 3 phases
(Train-Rollout / Train-Update / Eval-Rollout), their internal decomposition,
the sync-barrier tax (H1), single-worker engine CPU (H2), the iter-level
Amdahl attribution that adjudicates the "engine 10x" goal (H3), and
oversubscription (H4) — plus phase-tagged CPU/GPU utilization and speed-knob
(bf16 / torch.compile) A/B. The display layer is a single module, :mod:`.waterfall`.

Design: base layer (``methods/**``, ``pcb_world/**``) stays import-only. Main-proc
instrumentation is runtime monkeypatch; worker-side timing rides a
forkserver-preload shim (:mod:`.worker_shim`) — zero base edits.

Front door: ``python scripts/profile.py`` (or ``python -m tools.diagnostics.speed_profiler``).
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"
