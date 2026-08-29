"""VecBackend — model-agnostic transport contract for vectorized envs.

A backend manages a pool of per-env workers (one env each, in a subprocess or
a Ray actor) and exposes a uniform *transport* surface: reset, async step
dispatch/collect, board hot-swap, crash recovery, and a generic per-worker RPC
(:meth:`env_method`). Model-specific batch operations (mask stacking, GiGPO
grouping, done-skip policy, prompt/text shaping) live *above* this contract in
the branch packages — not here.

Two implementations share this ABC but NOT their transport mechanism:
- ``subproc.py`` — multiprocessing/forkserver, ``SubprocDecoderVecEnv`` (RL)
- ``ray.py``     — Ray actors, ``RayVecBackend`` (LLM)

Design choices that let both conform without breaking their callers:

1. **Async is the primitive.** ``step_async`` + ``step_wait`` (and the
   ``*_selective`` pair) are abstract; the RL collector pipelines policy
   compute against env stepping via exactly this split, and Ray implements it
   over actor futures. A synchronous :meth:`step_sync` convenience is provided
   concretely. A branch may still expose its own caller-facing ``step()`` with
   a branch-specific return shape (e.g. the LLM/verl ``(text, image, reward,
   done, info)`` tuple) — that lives in the branch class, not this ABC.

2. **Canonical step return is the gym 5-tuple** ``(obs_list, rewards,
   terminated, truncated, infos)`` with numpy ``rewards/terminated/truncated``.
   ``obs`` is whatever the worker produces (dict for RL, text for LLM) — the
   transport is obs-type-agnostic.

3. **Masks are not first-class.** A backend exposes :meth:`env_method`
   ("call method X on each worker in parallel"); the branch wraps it (e.g.
   ``wrappers/rl/mask.py`` stacks ``env_method("action_masks")``).

See design doc §11.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class VecBackend(ABC):
    """Transport contract for a pool of per-env workers."""

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    @abstractmethod
    def reset_all(self) -> list:
        """Reset every worker in parallel. Returns ``list[obs]``."""
        ...

    @abstractmethod
    def reset_batch(
        self, indices: Sequence[int], seeds: Sequence[int] | None = None,
    ) -> list:
        """Reset a subset of workers in parallel.

        ``seeds`` (optional, parallel to ``indices``) is forwarded to each
        worker's ``env.reset(seed=...)``, which seeds the gymnasium
        ``np_random`` of that env — the RNG behind per-episode env-side
        sampling (e.g. ``keep_routing_fraction``). ``None`` leaves every env
        unseeded (gymnasium keeps its current generator), the historical
        behavior. It does NOT touch the RL wrapper's own augmentation RNG.
        """
        ...

    # ------------------------------------------------------------------
    # Step — async primitive (dispatch / collect)
    # ------------------------------------------------------------------

    @abstractmethod
    def step_async(self, actions: Sequence) -> None:
        """Dispatch a step to ALL workers (non-blocking)."""
        ...

    @abstractmethod
    def step_wait(self) -> tuple[list, Any, Any, Any, list]:
        """Collect a prior :meth:`step_async`.

        Returns ``(obs_list, rewards, terminated, truncated, infos)`` with
        numpy ``rewards/terminated/truncated``.
        """
        ...

    @abstractmethod
    def step_async_selective(self, indices: Sequence[int], actions: Sequence) -> None:
        """Dispatch a step to a SUBSET of workers (GRPO / done-skip)."""
        ...

    @abstractmethod
    def step_wait_selective(self) -> tuple[list, Any, Any, Any, list]:
        """Collect a prior :meth:`step_async_selective`."""
        ...

    # Concrete synchronous convenience (portable; gym 5-tuple shape).
    def step_sync(self, actions: Sequence) -> tuple[list, Any, Any, Any, list]:
        """Blocking step of all workers: ``step_async`` then ``step_wait``."""
        self.step_async(actions)
        return self.step_wait()

    def step_sync_selective(
        self, indices: Sequence[int], actions: Sequence,
    ) -> tuple[list, Any, Any, Any, list]:
        """Blocking selective step: dispatch then wait."""
        self.step_async_selective(indices, actions)
        return self.step_wait_selective()

    # ------------------------------------------------------------------
    # Generic per-worker RPC
    # ------------------------------------------------------------------

    @abstractmethod
    def env_method(
        self, name: str, *args: Any, indices: Sequence[int] | None = None, **kwargs: Any,
    ) -> list:
        """Parallel "call ``worker.name(*args, **kwargs)`` on each worker"."""
        ...

    # ------------------------------------------------------------------
    # Board hot-swap / lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reload_boards(self, board_paths: Sequence[str]) -> None:
        """Swap each worker's board in place (no actor/process restart)."""
        ...

    @abstractmethod
    def rebuild(self, indices: Sequence[int] | None = None) -> None:
        """Respawn workers after a crash (native SIGSEGV / actor death)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Tear down all workers."""
        ...
