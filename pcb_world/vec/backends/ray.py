"""RayVecBackend — Ray-actor transport for a pool of env-adapter workers.

Model-agnostic transport layer: spawns ``env_num * group_n`` Ray actors (one
per worker), dispatches step/reset in parallel, short-circuits done workers,
hot-swaps boards, and recovers from actor death. Two worker-injection modes
let both branches share this transport:

  * ``worker_cls`` mode (LLM): a worker class is injected (e.g.
    ``KiCadLLMWrapper``) and built per group with ``board_path`` + ``seed`` +
    ``worker_ctor_kwargs``. The LLM serialization stays in the branch package;
    this module never imports it.
  * ``env_fns`` mode (RL): per-worker env factories + board factories are
    injected (mirroring ``SubprocDecoderVecEnv``), wrapped in a generic
    :class:`_RayEnvActor`. This lets the RL decoder pool run on Ray with the
    exact same closures the subprocess pool uses — env_method (and the
    ``stack_*_masks`` helpers built on it) carry the gym-wrapper surface, so
    no RL-specific methods leak into this transport.

All ``group_n`` workers in a GiGPO group share the same board + seed — required
for group-relative advantage to be meaningful. RL uses ``group_n=1`` (flat
per-worker boards).
"""

from __future__ import annotations

import functools
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import ray

from pcb_world.engine.router_client import EngineServerCrashed
from pcb_world.vec.backends.base import VecBackend


def _server_crash_kills_actor(fn):
    """Engine-IPC crash semantics = in-process semantics, by dying: a fatal
    C++ signal kills the engine-server child of this ACTOR and surfaces as
    EngineServerCrashed instead of killing the actor. Exit the actor process
    like the in-process segfault did, so callers see the same RayActorError
    and the existing ``rebuild_workers`` recovery applies — a plain raise
    would be a survivable RayTaskError, a new failure mode no caller handles.
    (Same design as the subproc worker's re-raise-to-die; see
    pcb_world/vec/backends/subproc.py.)
    """

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except EngineServerCrashed:
            traceback.print_exc()
            os._exit(1)

    return wrapped


class _RayEnvActor:
    """Generic Ray-actor wrapper around an ``env_fn``-built gym env (RL mode).

    Bridges a plain ``gym``-style env (e.g. :class:`KiCadRLWrapper`, which
    returns the 5-tuple ``(obs, reward, terminated, truncated, info)``) onto
    the worker contract :class:`RayVecBackend` expects: ``step`` returns the
    4-tuple ``(obs, reward, done, info)`` with ``terminated``/``truncated``
    stamped into ``info`` (so ``_assemble_step`` reconstructs the gym 5-tuple
    losslessly), and arbitrary wrapper methods are reachable via :meth:`call`
    (the env_fns-mode dispatch for ``env_method`` — Ray actors only expose
    methods declared on the class, so a generic dispatcher is needed instead
    of ``__getattr__``).
    """

    def __init__(
        self,
        env_fn: Callable[[], Any],
        board_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._board_factory = board_factory
        self.env = env_fn()

    @_server_crash_kills_actor
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault("terminated", bool(terminated))
        info.setdefault("truncated", bool(truncated))
        return obs, float(reward), bool(terminated or truncated), info

    @_server_crash_kills_actor
    def reset(self, seed: int | None = None):
        # seed=None keeps the historical no-arg call so non-gymnasium envs
        # (the LLM branch) stay callable.
        obs, info = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        return obs, info

    @_server_crash_kills_actor
    def reload_board(self, board_path: str, reload_seq: int = 0) -> bool:
        """Rebuild the inner env on a new board (mirrors subproc reload).

        ``reload_seq`` is the parent-owned env-rebuild counter the factory
        mixes into the per-env RNG seed (see ``advance_rng_on_reload``); 0
        keeps the factory's original seed.
        """
        if self._board_factory is None:
            raise RuntimeError("reload_board needs a board_factory")
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        self.env = None
        import gc

        gc.collect()
        self.env = self._board_factory(board_path, reload_seq)
        return True

    @_server_crash_kills_actor
    def call(self, method_name: str, *args: Any, **kwargs: Any):
        """Generic method dispatch — backs ``env_method`` in env_fns mode."""
        return getattr(self.env, method_name)(*args, **kwargs)

    def get_last_obs_dict(self):
        """Parity placeholder (RL doesn't cache obs dicts on the actor)."""
        return getattr(self.env, "get_last_obs_dict", lambda: None)()

    def apply_adaptive_max_steps(self, base_max_steps: int) -> int:
        fn = getattr(self.env, "apply_adaptive_max_steps", None)
        return fn(base_max_steps) if fn is not None else base_max_steps


def _normalize_board_paths(
    board_path: Optional[str],
    board_paths: Optional[List[str]],
    env_num: int,
) -> List[str]:
    """Resolve the ``board_path`` / ``board_paths`` shorthand into a
    per-group list of length ``env_num``.
    """
    if board_path is None and board_paths is None:
        raise ValueError("RayVecBackend requires board_path or board_paths")
    if board_path is not None and board_paths is not None:
        raise ValueError(
            "RayVecBackend accepts board_path OR board_paths, not both"
        )
    if board_paths is None:
        return [board_path] * env_num  # type: ignore[list-item]
    if len(board_paths) != env_num:
        raise ValueError(
            f"board_paths length {len(board_paths)} != env_num {env_num}"
        )
    return list(board_paths)


class RayVecBackend(VecBackend, gym.Env):
    """Ray-based vectorised env transport over an injected worker class.

    Manages ``env_num * group_n`` parallel ``worker_cls`` actors. Each worker
    actor owns one env and exposes step / reset / get_action_mask /
    reload_board / save_board / apply_adaptive_max_steps as remote-callable
    methods (see ``KiCadLLMWrapper`` for the LLM implementation).

    Args:
        worker_cls:            The (non-remote) worker class to wrap as a Ray
                               actor. Its ctor must accept ``board_path`` +
                               ``seed`` plus everything in ``worker_ctor_kwargs``.
        worker_ctor_kwargs:    Per-worker ctor kwargs (excluding board_path/seed,
                               which are injected per group).
        seed:                  Base seed (each group_idx gets seed + group_idx).
        env_num:               Number of GiGPO groups (= logical batch size).
        group_n:               Group size for GiGPO rollouts.
        resources_per_worker:  Ray resource dict (e.g. ``{"num_cpus": 1}``).
        board_paths:           Per-group board paths; length must equal env_num.
                               Worker ``i`` is bound to ``board_paths[i // group_n]``.
    """

    def __init__(
        self,
        *,
        worker_cls: Optional[type] = None,
        worker_ctor_kwargs: Optional[Dict[str, Any]] = None,
        env_fns: Optional[List[Callable[[], Any]]] = None,
        board_factories: Optional[List[Callable[..., Any]]] = None,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict[str, Any],
        board_paths: List[str],
        advance_rng_on_reload: bool = False,
    ) -> None:
        super().__init__()

        # Exactly one injection mode.
        if (worker_cls is None) == (env_fns is None):
            raise ValueError(
                "RayVecBackend requires exactly one of worker_cls / env_fns"
            )

        if not ray.is_initialized():
            ray.init()

        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self._board_paths: List[str] = list(board_paths)
        # Per-worker env-rebuild counter mixed into the RNG seed on reload /
        # respawn — see SubprocDecoderVecEnv's ``advance_rng_on_reload``.
        self._advance_rng_on_reload = bool(advance_rng_on_reload)
        self._rng_reload_seq: List[int] = [0] * self.num_processes

        # Retain everything needed to respawn a worker actor in place — used
        # by :meth:`rebuild_workers` to recover from a Ray actor death (e.g. a
        # SIGSEGV in the native KiCad router) without tearing down the whole
        # backend. ``seed + group_idx`` is the per-group seed; storing the
        # base seed keeps the group-shared-seed invariant on respawn.
        self._seed = seed

        self._env_fns_mode = env_fns is not None
        if self._env_fns_mode:
            # RL mode: per-worker env factories (seed baked into the closure,
            # exactly as SubprocDecoderVecEnv builds them). group_n is 1, so
            # there is one board per worker.
            if len(env_fns) != self.num_processes:
                raise ValueError(
                    f"env_fns length {len(env_fns)} != env_num*group_n "
                    f"{self.num_processes}"
                )
            if board_factories is not None and len(board_factories) != self.num_processes:
                raise ValueError("board_factories length must match env_fns")
            self._env_fns: List[Callable[[], Any]] = list(env_fns)
            self._board_factories: Optional[List[Callable[..., Any]]] = (
                list(board_factories) if board_factories is not None else None
            )
            self._env_worker_cls = ray.remote(**resources_per_worker)(_RayEnvActor)
            self._worker_ctor_kwargs = {}
        else:
            # LLM mode: inject a worker class built per group.
            self._env_worker_cls = ray.remote(**resources_per_worker)(worker_cls)
            self._worker_ctor_kwargs = dict(worker_ctor_kwargs or {})

        self.workers: List[Any] = [
            self._spawn_worker(i) for i in range(self.num_processes)
        ]

        self.prev_action_masks: List[Optional[List[bool]]] = [None] * self.num_processes
        self.prev_action_mask_dicts: List[Optional[Dict[str, bool]]] = [None] * self.num_processes
        # Done short-circuit: once a worker reports done, ``step`` skips the
        # Ray RPC for that index and replays the cached terminal observation
        # with reward=0.0. Avoids wasted KiCad work (esp. per-step DRC under
        # dense reward modes) while the rest of the batch finishes.
        self._done_mask: List[bool] = [False] * self.num_processes
        self._done_cache: List[Optional[Tuple[str, Dict[str, Any]]]] = [None] * self.num_processes

        # Async step state for the VecBackend step_async/step_wait pair.
        self._pending_futures: Optional[list] = None
        self._pending_async_indices: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def step(self, actions: List[Dict[str, Any]]):
        """Step all workers in parallel.

        Args:
            actions: List of action dicts, one per worker.

        Returns:
            text_obs_list:  List[str]   - serialised observations
            image_obs_list: None        - no visual rendering (placeholder)
            rewards_list:   List[float]
            dones_list:     List[bool]
            info_list:      List[dict]
        """
        assert len(actions) == self.num_processes, (
            f"Expected {self.num_processes} actions, got {len(actions)}"
        )

        # Dispatch RPCs only for still-active workers; done indices replay
        # their cached terminal obs/info with reward=0.0.
        pending_indices: List[int] = []
        pending_futures = []
        for i, action in enumerate(actions):
            if self._done_mask[i]:
                continue
            pending_indices.append(i)
            pending_futures.append(self.workers[i].step.remote(action))
        pending_results = ray.get(pending_futures) if pending_futures else []

        text_obs_list: List[Optional[str]] = [None] * self.num_processes
        rewards_list: List[float] = [0.0] * self.num_processes
        dones_list: List[bool] = [False] * self.num_processes
        info_list: List[dict] = [{} for _ in range(self.num_processes)]

        for idx, (text_obs, reward, done, info) in zip(pending_indices, pending_results):
            text_obs_list[idx] = text_obs
            rewards_list[idx] = reward
            dones_list[idx] = done
            info_list[idx] = info
            self.prev_action_masks[idx] = info.get("action_mask")
            self.prev_action_mask_dicts[idx] = info.get("action_mask_dict")
            if done:
                self._done_mask[idx] = True
                self._done_cache[idx] = (text_obs, info)

        # Fill in already-done indices from cache (no engine work performed).
        pending_set = set(pending_indices)
        for i in range(self.num_processes):
            if self._done_mask[i] and i not in pending_set:
                cached = self._done_cache[i]
                if cached is None:
                    # Shouldn't happen — done_mask is only set when we have
                    # a real cache entry — but fall back to empty payload.
                    text_obs_list[i] = ""
                    info_list[i] = {}
                else:
                    text_obs_list[i] = cached[0]
                    info_list[i] = cached[1]
                rewards_list[i] = 0.0
                dones_list[i] = True

        return text_obs_list, None, rewards_list, dones_list, info_list

    def reset(self):
        """Reset all workers in parallel.

        Returns:
            text_obs_list:  List[str]
            image_obs_list: None
            info_list:      List[dict]
        """
        futures = [worker.reset.remote() for worker in self.workers]
        results = ray.get(futures)

        text_obs_list: List[str] = []
        info_list: List[dict] = []

        for i, (text_obs, info) in enumerate(results):
            text_obs_list.append(text_obs)
            self.prev_action_masks[i] = info.get("action_mask")
            self.prev_action_mask_dicts[i] = info.get("action_mask_dict")
            info_list.append(info)

        # New episode — clear the done short-circuit state.
        self._done_mask = [False] * self.num_processes
        self._done_cache = [None] * self.num_processes

        return text_obs_list, None, info_list

    # ------------------------------------------------------------------
    # VecBackend contract — model-agnostic transport surface.
    #
    # These are the portable primitives (gym 5-tuple, async step, env_method)
    # shared with the RL SubprocDecoderVecEnv. They do NOT apply the verl-facing
    # done short-circuit (that policy lives in the branch-facing ``step`` above);
    # ``step_async`` dispatches to *every* requested worker and ``step_wait``
    # returns ``(obs, rewards, terminated, truncated, infos)``.
    # ------------------------------------------------------------------

    def reset_all(self) -> List[Any]:
        """VecBackend: reset every worker, return ``list[obs]`` only."""
        obs_list, _img, _info = self.reset()
        return obs_list

    def reset_batch(
        self, indices: Sequence[int], seeds: Sequence[int] | None = None,
    ) -> List[Tuple[Any, dict]]:
        """VecBackend: reset a subset of workers. Returns ``list[(obs, info)]``.

        ``seeds`` (parallel to ``indices``) seeds each env's gymnasium
        ``np_random``; see :meth:`VecBackend.reset_batch`.
        """
        idxs = list(indices)
        if seeds is None:
            payloads: List[Any] = [None] * len(idxs)
        else:
            payloads = [int(s) for s in seeds]
            if len(payloads) != len(idxs):
                raise ValueError(
                    f"reset_batch: {len(payloads)} seeds for {len(idxs)} indices"
                )
        futures = [
            self.workers[i].reset.remote(seed)
            for i, seed in zip(idxs, payloads)
        ]
        results = ray.get(futures)
        out: List[Tuple[Any, dict]] = []
        for i, (text_obs, info) in zip(idxs, results):
            self.prev_action_masks[i] = info.get("action_mask")
            self.prev_action_mask_dicts[i] = info.get("action_mask_dict")
            self._done_mask[i] = False
            self._done_cache[i] = None
            out.append((text_obs, info))
        return out

    def step_async(self, actions: Sequence) -> None:
        """VecBackend: dispatch a step to ALL workers (non-blocking)."""
        assert len(actions) == self.num_processes, (
            f"Expected {self.num_processes} actions, got {len(actions)}"
        )
        self._pending_async_indices = list(range(self.num_processes))
        self._pending_futures = [
            self.workers[i].step.remote(actions[i])
            for i in self._pending_async_indices
        ]

    def step_wait(self):
        """VecBackend: collect a prior :meth:`step_async`.

        Returns ``(obs_list, rewards, terminated, truncated, infos)``.
        """
        assert self._pending_futures is not None, "call step_async first"
        results = ray.get(self._pending_futures)
        idxs = self._pending_async_indices
        self._pending_futures = None
        self._pending_async_indices = None
        return self._assemble_step(idxs, results)

    def step_async_selective(self, indices: Sequence[int], actions: Sequence) -> None:
        """VecBackend: dispatch a step to a SUBSET of workers."""
        idxs = list(indices)
        self._pending_async_indices = idxs
        self._pending_futures = [
            self.workers[i].step.remote(actions[k]) for k, i in enumerate(idxs)
        ]

    def step_wait_selective(self):
        """VecBackend: collect a prior :meth:`step_async_selective`."""
        assert self._pending_futures is not None, "call step_async_selective first"
        results = ray.get(self._pending_futures)
        idxs = self._pending_async_indices
        self._pending_futures = None
        self._pending_async_indices = None
        return self._assemble_step(idxs, results)

    def _assemble_step(self, idxs: List[int], results: list):
        """Build the gym 5-tuple from per-worker ``(obs, reward, done, info)``.

        ``terminated`` / ``truncated`` are read back from the worker info
        (KiCadLLMWrapper stores both), so the collapsed ``done`` is never lossy.
        """
        obs_l: List[Any] = []
        rew_l: List[float] = []
        term_l: List[bool] = []
        trunc_l: List[bool] = []
        info_l: List[dict] = []
        for i, (text_obs, reward, done, info) in zip(idxs, results):
            obs_l.append(text_obs)
            rew_l.append(reward)
            term_l.append(bool(info.get("terminated", done)))
            trunc_l.append(bool(info.get("truncated", False)))
            info_l.append(info)
            self.prev_action_masks[i] = info.get("action_mask")
            self.prev_action_mask_dicts[i] = info.get("action_mask_dict")
        return (
            obs_l,
            np.array(rew_l, dtype=np.float64),
            np.array(term_l, dtype=bool),
            np.array(trunc_l, dtype=bool),
            info_l,
        )

    def env_method(
        self, name: str, *args: Any, indices: Sequence[int] | None = None, **kwargs: Any,
    ) -> List[Any]:
        """VecBackend: parallel ``worker.name(*args, **kwargs)`` on each worker.

        worker_cls mode dispatches the named method directly on the actor;
        env_fns mode routes through :meth:`_RayEnvActor.call` (Ray actors only
        expose methods declared on the class, so the generic env-wrapper
        surface — ``action_masks`` / ``mode_mask`` / ``save_pcb`` / ... — is
        reached via the actor's ``call`` dispatcher).
        """
        idxs = list(range(self.num_processes)) if indices is None else list(indices)
        if self._env_fns_mode:
            futures = [
                self.workers[i].call.remote(name, *args, **kwargs) for i in idxs
            ]
        else:
            futures = [
                getattr(self.workers[i], name).remote(*args, **kwargs) for i in idxs
            ]
        return list(ray.get(futures))

    def rebuild(self, indices: Sequence[int] | None = None) -> None:
        """VecBackend alias for :meth:`rebuild_workers`."""
        self.rebuild_workers(None if indices is None else list(indices))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def get_action_masks(self) -> List[Optional[List[bool]]]:
        """Per-worker boolean action masks from the last step/reset."""
        return self.prev_action_masks

    @property
    def get_action_mask_dicts(self) -> List[Optional[Dict[str, bool]]]:
        """Per-worker {action_name: bool} masks from the last step/reset."""
        return self.prev_action_mask_dicts

    def get_last_obs_dicts(self) -> List[Optional[Dict[str, Any]]]:
        """Fetch the raw obs dict from every worker's most recent step/reset.

        One Ray round-trip for all workers. Entries are ``None`` for workers
        that have not yet observed (e.g. just after ``reload_boards`` before
        the subsequent ``reset()``).
        """
        futures = [w.get_last_obs_dict.remote() for w in self.workers]
        return list(ray.get(futures))

    def apply_adaptive_max_steps(self, base_max_steps: int) -> int:
        """Apply per-worker ``max(base, pin*3 + net*2)`` and return the max.

        Each worker computes and applies its own budget locally (using its
        cached ``_last_obs_dict``) so the env's internal truncation respects
        the per-board limit; the returned max-across-workers is suitable as the
        outer rollout-loop bound.

        Call after ``reset()`` (or ``reload_boards`` + ``reset``) so the
        observation dicts reflect the current boards.
        """
        futures = [
            w.apply_adaptive_max_steps.remote(base_max_steps)
            for w in self.workers
        ]
        per_worker = list(ray.get(futures))
        return max(per_worker) if per_worker else base_max_steps

    # ------------------------------------------------------------------
    # Board hot-swap
    # ------------------------------------------------------------------

    def reload_boards(self, board_paths: List[str]) -> None:
        """Swap boards across workers without killing Ray actors.

        ``board_paths`` has length ``env_num`` — one entry per GiGPO group.
        All ``group_n`` workers of a group get the same path so the
        group-board invariant holds.
        """
        if len(board_paths) != self.env_num:
            raise ValueError(
                f"reload_boards expected {self.env_num} paths "
                f"(env_num), got {len(board_paths)}"
            )

        futures = []
        for i, worker in enumerate(self.workers):
            group_idx = i // self.group_n
            if self._advance_rng_on_reload:
                self._rng_reload_seq[i] += 1
            futures.append(worker.reload_board.remote(
                board_paths[group_idx], self._rng_reload_seq[i],
            ))
        ray.get(futures)
        self._board_paths = list(board_paths)
        # New boards mean any cached terminal obs is stale — clear so the
        # next reset/step starts from a clean slate even if the caller
        # forgets to call reset().
        self._done_mask = [False] * self.num_processes
        self._done_cache = [None] * self.num_processes

    @property
    def board_paths(self) -> List[str]:
        """Current per-group board paths (length == env_num)."""
        return list(self._board_paths)

    # ------------------------------------------------------------------
    # Snapshot I/O
    # ------------------------------------------------------------------

    def save_boards(self, output_paths: List[str]) -> List[str]:
        """Save each worker's current board state to the matching path.

        Each save emits a ``.kicad_pcb`` + companion ``.kicad_pro`` pair
        (see ``KiCadEngine.save``) — both are needed to reload the board
        with design rules intact. Useful at episode end in eval scripts.
        """
        assert len(output_paths) == self.num_processes, (
            f"Expected {self.num_processes} output paths, got {len(output_paths)}"
        )
        futures = [
            worker.save_board.remote(path)
            for worker, path in zip(self.workers, output_paths)
        ]
        return list(ray.get(futures))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _spawn_worker(self, i: int):
        """Create one worker actor for index ``i`` against its group's board.

        worker_cls mode: workers sharing a ``group_idx`` get the same seed +
        board so the GiGPO group-shared start-state invariant holds. env_fns
        mode: the closure ``env_fns[i]`` is self-contained (seed + board baked
        in by the factory, like SubprocDecoderVecEnv), wrapped in a generic
        :class:`_RayEnvActor` with its per-worker board_factory for reload.
        """
        if self._env_fns_mode:
            bf = self._board_factories[i] if self._board_factories else None
            return self._env_worker_cls.remote(
                env_fn=self._env_fns[i], board_factory=bf,
            )
        group_idx = i // self.group_n
        return self._env_worker_cls.remote(
            board_path=self._board_paths[group_idx],
            seed=self._seed + group_idx,
            **self._worker_ctor_kwargs,
        )

    def rebuild_workers(self, indices: Optional[List[int]] = None) -> None:
        """Respawn worker actors in place — recovery from a dead Ray actor.

        When a worker dies mid-episode (native router SIGSEGV, OOM, etc.),
        every subsequent RPC to it raises ``RayActorError``. Callers that
        catch such failures can rebuild the affected workers and continue
        with the next episode rather than aborting the whole run.

        ``indices`` selects which workers to respawn (default: all). The old
        actor handle is best-effort killed first; the new actor is built
        against the *current* board (``self._board_paths``), so call after
        the board is set. Done short-circuit state for the rebuilt indices is
        cleared. A subsequent ``reset()`` re-initialises the episode.
        """
        target = list(range(self.num_processes)) if indices is None else list(indices)
        for i in target:
            try:
                ray.kill(self.workers[i])
            except Exception:  # noqa: BLE001 — already-dead actor, ignore
                pass
            self.workers[i] = self._spawn_worker(i)
            self._done_mask[i] = False
            self._done_cache[i] = None
            self.prev_action_masks[i] = None
            self.prev_action_mask_dicts[i] = None

    def close(self) -> None:
        """Kill all Ray worker actors."""
        for worker in self.workers:
            ray.kill(worker)
