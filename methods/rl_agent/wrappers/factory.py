"""RL decoder env factories.

Construct the canonical RL env surface — a single :class:`KiCadRLWrapper`
(index-space pointer wrapper over :class:`PCBWorld`) or a subprocess-parallel
pool of them (:class:`SubprocDecoderVecEnv`, required by KiCad's singleton
RLRouter constraint).

Layer note: this module lives in ``envs/wrappers/rl`` and imports only
*down* (``pcb_world.core``) and *sideways within wrappers*
(``methods.rl_agent.wrappers.adapter``, ``pcb_world.vec.backends``) — never up into
``training``.

The env is described by ONE object, :class:`~configs.loader.schema.RLEnvConfig`,
carried whole into ``PCBWorld`` (its ``env``) and the wrapper
(:meth:`~configs.loader.schema.RLEnvConfig.wrapper_kwargs`), never unpacked into a
hand-written kwarg list. ``seed`` / ``policy_net_select`` are per-call (train and
eval pass different values), so they stay arguments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from configs.loader.schema import RLEnvConfig
from pcb_world.core.env import PCBWorld
from pcb_world.vec.backends.subproc import SubprocDecoderVecEnv
from methods.rl_agent.wrappers.adapter import KiCadRLWrapper


# Salt separating the env's np_random stream from the wrapper's _rng, both of
# which are derived from the same per-env ``seed``.
_ENV_RNG_SALT = 0x5EED_E17


def env_record(env_config: RLEnvConfig, *, seed: int, policy_net_select: bool) -> dict[str, Any]:
    """What an env (pool) is built with, flat — the form the val-env cross-check compares
    (``eval.rollout.rl.assert_env_kwargs_as_recorded``) against the trainer's val record."""
    return {**env_config.to_record(), "seed": seed, "policy_net_select": policy_net_select}


def make_decoder_env(
    board_path: str,
    env_config: RLEnvConfig,
    *,
    seed: int,
    policy_net_select: bool,
) -> KiCadRLWrapper:
    """Create a single :class:`KiCadRLWrapper` for decoder training.

    Note: this wrapper passes raw JSON dict observations through (no flat
    feature conversion). Group rollouts use a Python list of these wrappers
    rather than ``DummyVecEnv`` because dict obs cannot be numpy-stacked.
    """
    env = PCBWorld(
        board_path,
        env_config.env,
        # The env owns a second RNG (gymnasium ``np_random``: keep-routing draw
        # + terminal reward noise) that is entirely separate from the wrapper's
        # ``_rng`` below. Derive it from the same ``seed`` so it is reproducible
        # and advances with the pool's reload counter, but off a distinct
        # SeedSequence so the two streams never correlate.
        seed=int(
            np.random.SeedSequence([seed, _ENV_RNG_SALT]).generate_state(1)[0]
        ),
    )
    return KiCadRLWrapper(
        env,
        seed=seed,
        policy_net_select=policy_net_select,
        **env_config.wrapper_kwargs(),
    )


def make_decoder_env_pool(
    board_path: str,
    n_envs: int,
    env_config: RLEnvConfig,
    *,
    seed: int,
    policy_net_select: bool,
    advance_rng_on_reload: bool,
    start_method: str | None = None,
    backend: str = "subproc",
    resources_per_worker: dict | None = None,
):
    """Create a parallel pool of decoder environments.

    Each env runs in its own worker (required by KiCad's singleton RLRouter
    constraint). Returns a :class:`~pcb_world.vec.backends.base.VecBackend`
    — a drop-in for ``list[KiCadRLWrapper]`` in the collectors.

    ``advance_rng_on_reload``: keep each worker's random stream moving across env
    rebuilds (training reloads a board into every worker each iteration); eval
    pools leave it off so a board's rollout stays independent of board order.

    ``backend``:
      * ``"subproc"`` (default) — :class:`SubprocDecoderVecEnv` (multiprocessing).
      * ``"ray"`` — :class:`RayVecBackend` in env_fns mode (same closures, Ray
        actors). ``resources_per_worker`` (default ``{"num_cpus": 1}``) sets the
        per-actor Ray resources. Masks/step/reset go through the shared
        ``VecBackend`` surface, so the collectors are backend-agnostic.
    """
    def _make_factory(s: int):
        def _factory(board: str, reload_seq: int = 0) -> KiCadRLWrapper:
            # ``reload_seq`` = how many times this worker's env has been
            # rebuilt (board hot-swap or crash respawn). The wrapper seeds its
            # numpy Generator once at construction, so rebuilding with the same
            # ``s`` rewinds the whole per-env random stream (augmentation,
            # slot_perm, auto net-select) to its starting point — under
            # per-iteration board reloads that replays the same draws every
            # iteration. Mixing the rebuild count into the seed keeps the
            # stream moving forward across rebuilds. reload_seq == 0 keeps the
            # legacy seed exactly, so pools that never advance (eval) stay
            # bit-identical.
            seed_i = s if reload_seq == 0 else int(
                np.random.SeedSequence([s, reload_seq]).generate_state(1)[0]
            )
            return make_decoder_env(board, env_config, seed=seed_i,
                                    policy_net_select=policy_net_select)
        return _factory

    board_factories = [_make_factory(seed + i) for i in range(n_envs)]
    env_fns = [(lambda f=f: f(board_path)) for f in board_factories]
    if backend == "subproc":
        return SubprocDecoderVecEnv(
            env_fns,
            start_method=start_method,
            board_factories=board_factories,
            advance_rng_on_reload=advance_rng_on_reload,
        )
    if backend == "ray":
        # env_fns mode: same closures the subprocess pool uses, wrapped in
        # generic Ray actors. group_n=1 → one board per worker (flat pool).
        from pcb_world.vec.backends.ray import RayVecBackend

        return RayVecBackend(
            env_fns=env_fns,
            board_factories=board_factories,
            seed=seed,
            env_num=n_envs,
            group_n=1,
            resources_per_worker=resources_per_worker or {"num_cpus": 1},
            board_paths=[board_path] * n_envs,
            advance_rng_on_reload=advance_rng_on_reload,
        )
    raise ValueError(f"backend must be 'subproc' or 'ray', got {backend!r}")


__all__ = ["env_record", "make_decoder_env", "make_decoder_env_pool"]
