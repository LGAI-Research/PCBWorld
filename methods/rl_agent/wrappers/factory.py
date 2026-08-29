"""RL decoder env factories.

Construct the canonical RL env surface — a single :class:`KiCadRLWrapper`
(index-space pointer wrapper over :class:`PCBWorld`) or a subprocess-parallel
pool of them (:class:`SubprocDecoderVecEnv`, required by KiCad's singleton
RLRouter constraint).

Layer note: this module imports only *down* (``pcb_world.core``) and
*sideways within wrappers* (``methods.rl_agent.wrappers.adapter``,
``pcb_world.vec.backends``) — never up into ``training``. Training drives
these via keyword config it owns.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from configs.loader.schema import EnvConfig, RewardOverrides
from pcb_world.core.env import PCBWorld
from pcb_world.vec.backends.subproc import SubprocDecoderVecEnv
from methods.rl_agent.wrappers.adapter import KiCadRLWrapper



#: Sentinel for env-contract kwargs. The factories refuse to fill these in:
#: the required set is exactly ``RLEnvConfig.to_pool_kwargs()`` — the shared
#: surface train and eval both build — plus ``seed`` / ``policy_net_select``,
#: which both sides pass explicitly. A knob added to that surface therefore can
#: never reach an env through a silent factory default.
#: ``tests/test_env_contract.py`` pins the correspondence.
_REQ: Any = object()

#: Knobs that exist only while TRAINING (no eval counterpart), passed as one
#: explicit bundle rather than five defaulted parameters: eval simply omits
#: ``train_extras``, which means "no training layer" rather than silently
#: selecting a value. Values here are what the absence of that layer means.
_TRAIN_EXTRAS: dict[str, Any] = {
    "reward_noise_std": 0.0,
    "aug_bbox_shifted": False,
    "aug_flip": False,
    "aug_rotate": False,
    "aug_trans": False,
    "aug_zoom": False,
    # Pool-level: keep each worker's stream moving across env rebuilds. Read by
    # ``make_decoder_env_pool``; ``make_decoder_env`` resolves and ignores it
    # (one bundle, one vocabulary).
    "advance_rng_on_reload": False,
}


def _unpack_train_extras(train_extras: dict | None, *, where: str) -> dict:
    """Resolve the train-only bundle, rejecting unknown keys loudly."""
    given = dict(train_extras or {})
    unknown = sorted(set(given) - set(_TRAIN_EXTRAS))
    if unknown:
        raise TypeError(
            f"{where}: unknown key(s) in train_extras {unknown} — "
            f"allowed: {sorted(_TRAIN_EXTRAS)}"
        )
    return {k: given.get(k, v) for k, v in _TRAIN_EXTRAS.items()}


# Salt separating the env's np_random stream from the wrapper's _rng, both of
# which are derived from the same per-env ``seed``.
_ENV_RNG_SALT = 0x5EED_E17

def make_decoder_env(
    board_path: str,
    *,
    max_steps: int = _REQ,
    masking_rule: str = _REQ,
    reward_rule: str = _REQ,
    seed: int = _REQ,
    force_walkaround: bool = _REQ,
    mask_start_point: bool = _REQ,
    slot_perm: bool = _REQ,
    emit_drc_tokens: bool = _REQ,
    via_penalty: float | None = _REQ,
    wirelength_penalty: float | None = _REQ,
    drc_penalty: float | None = _REQ,
    drc_log_scale: float | None = _REQ,
    drc_log_agg_scale: float | None = _REQ,
    drc_log_offset: float | None = _REQ,
    reward_step_penalty: float | None = _REQ,
    wire_via_emission: str | None = _REQ,
    corner_mode: int = _REQ,
    policy_net_select: bool = _REQ,
    directional_candidates: str | None = _REQ,
    connectivity_filter: bool = _REQ,
    pad_graze_margin_mm: float = _REQ,
    use_yaml_drc_fallback: bool = _REQ,
    drc_config_path: str | None = _REQ,
    simplify_outline: bool = _REQ,
    obs_format: str = _REQ,
    outline_obs: str = _REQ,
    action_history_len: int = _REQ,
    net_constraint_obs: bool = _REQ,
    keep_routing_fraction: "tuple[float, float] | list[float] | None" = _REQ,
    train_extras: dict | None = None,
    **unexpected: object,  # never swallowed — rejected below
) -> KiCadRLWrapper:
    """Create a single :class:`KiCadRLWrapper` for decoder training.

    Note: this wrapper passes raw JSON dict observations through (no flat
    feature conversion). Group rollouts use a Python list of these wrappers
    rather than ``DummyVecEnv`` because dict obs cannot be numpy-stacked.
    """
    # Unknown kwargs are never swallowed: absorbing one silently (e.g.
    # directional_grid_size, whose replacement is directional_candidates
    # "grid<N>") would route with the wrong candidate set. Every unknown name
    # raises; names whose destination is known get a pointer to it instead of a
    # bare "unexpected keyword argument".
    if unexpected:
        if "directional_grid_size" in unexpected:
            raise TypeError(
                "make_decoder_env: directional_grid_size was renamed — pass "
                "directional_candidates='grid%s'" % unexpected["directional_grid_size"]
            )
        moved = sorted(set(unexpected) & set(_TRAIN_EXTRAS))
        hint = (" — %s moved into the train_extras bundle; pass them as "
                "train_extras={...}" % moved) if moved else ""
        raise TypeError("make_decoder_env: unknown kwarg %s%s" % (sorted(unexpected), hint))
    _missing = sorted(k for k, v in locals().items() if v is _REQ)
    if _missing:
        raise TypeError(
            f"make_decoder_env: missing env-contract knob(s) {_missing} — "
            "pass the dict produced by RLEnvConfig(...).to_pool_kwargs() "
            "as-is (partial kwargs are not accepted)."
        )
    _x = _unpack_train_extras(train_extras, where="make_decoder_env")
    env = PCBWorld(
        board_path=board_path,
        # The env owns a second RNG (gymnasium ``np_random``: keep-routing draw
        # + terminal reward noise) that is entirely separate from the wrapper's
        # ``_rng`` below. Derive it from the same ``seed`` so it is reproducible
        # and advances with the pool's reload counter, but off a distinct
        # SeedSequence so the two streams never correlate.
        seed=int(
            np.random.SeedSequence([seed, _ENV_RNG_SALT]).generate_state(1)[0]
        ),
        env_config=EnvConfig(
            max_steps=max_steps,
            masking_rule=masking_rule,
            reward_rule=reward_rule,
            reward_noise_std=_x["reward_noise_std"],
            emit_drc_tokens=emit_drc_tokens,
            corner_mode=corner_mode,
            use_yaml_drc_fallback=use_yaml_drc_fallback,
            drc_config_path=drc_config_path,
            simplify_outline=simplify_outline,
            obs_format=obs_format,
            outline_obs=outline_obs,
            action_history_len=action_history_len,
            net_constraint_obs=net_constraint_obs,
            keep_routing_fraction=keep_routing_fraction,
            reward=RewardOverrides(
                via_penalty=via_penalty,
                wirelength_penalty=wirelength_penalty,
                drc_penalty=drc_penalty,
                drc_log_scale=drc_log_scale,
                drc_log_agg_scale=drc_log_agg_scale,
                drc_log_offset=drc_log_offset,
                reward_step_penalty=reward_step_penalty,
                wire_via_emission=wire_via_emission,
            ),
        ),
    )
    return KiCadRLWrapper(
        env,
        seed=seed,
        force_walkaround=force_walkaround,
        mask_start_point=mask_start_point,
        aug_bbox_shifted=_x["aug_bbox_shifted"],
        aug_flip=_x["aug_flip"],
        aug_rotate=_x["aug_rotate"],
        aug_trans=_x["aug_trans"],
        aug_zoom=_x["aug_zoom"],
        slot_perm=slot_perm,
        policy_net_select=policy_net_select,
        directional_candidates=directional_candidates,
        connectivity_filter=connectivity_filter,
        pad_graze_margin_mm=pad_graze_margin_mm,
    )


def make_decoder_env_pool(
    board_path: str,
    n_envs: int,
    *,
    max_steps: int = _REQ,
    masking_rule: str = _REQ,
    reward_rule: str = _REQ,
    seed: int = _REQ,
    force_walkaround: bool = _REQ,
    mask_start_point: bool = _REQ,
    slot_perm: bool = _REQ,
    emit_drc_tokens: bool = _REQ,
    via_penalty: float | None = _REQ,
    wirelength_penalty: float | None = _REQ,
    drc_penalty: float | None = _REQ,
    drc_log_scale: float | None = _REQ,
    drc_log_agg_scale: float | None = _REQ,
    drc_log_offset: float | None = _REQ,
    reward_step_penalty: float | None = _REQ,
    wire_via_emission: str | None = _REQ,
    start_method: str | None = None,
    corner_mode: int = _REQ,
    policy_net_select: bool = _REQ,
    directional_candidates: str | None = _REQ,
    connectivity_filter: bool = _REQ,
    pad_graze_margin_mm: float = _REQ,
    use_yaml_drc_fallback: bool = _REQ,
    drc_config_path: str | None = _REQ,
    simplify_outline: bool = _REQ,
    obs_format: str = _REQ,
    outline_obs: str = _REQ,
    action_history_len: int = _REQ,
    net_constraint_obs: bool = _REQ,
    keep_routing_fraction: "tuple[float, float] | list[float] | None" = _REQ,
    backend: str = "subproc",
    resources_per_worker: dict | None = None,
    train_extras: dict | None = None,
    **unexpected: object,  # never swallowed — rejected below
):
    """Create a parallel pool of decoder environments.

    Each env runs in its own worker (required by KiCad's singleton RLRouter
    constraint). Returns a :class:`~pcb_world.vec.backends.base.VecBackend`
    — a drop-in for ``list[KiCadRLWrapper]`` in the collectors.

    ``backend``:
      * ``"subproc"`` (default) — :class:`SubprocDecoderVecEnv` (multiprocessing).
      * ``"ray"`` — :class:`RayVecBackend` in env_fns mode (same closures, Ray
        actors). ``resources_per_worker`` (default ``{"num_cpus": 1}``) sets the
        per-actor Ray resources. Masks/step/reset go through the shared
        ``VecBackend`` surface, so the collectors are backend-agnostic.
    """
    # How workers are spawned is not what the env IS, so it stays out of both
    # the completeness check and the recorded contract.
    _PROCESS_KNOBS = ("start_method", "backend", "resources_per_worker")
    _passed = {k: v for k, v in locals().items()
               if k not in ("board_path", "n_envs", "train_extras", "unexpected",
                            "_PROCESS_KNOBS", *_PROCESS_KNOBS)}
    _missing = sorted(k for k, v in _passed.items() if v is _REQ)
    if _missing:
        raise TypeError(
            f"make_decoder_env_pool: missing env-contract knob(s) {_missing} — "
            "pass the dict produced by RLEnvConfig(...).to_pool_kwargs() "
            "as-is (partial kwargs are not accepted)."
        )
    advance_rng_on_reload = _unpack_train_extras(
        train_extras, where="make_decoder_env_pool",
    )["advance_rng_on_reload"]
    if unexpected:
        # Unknown kwargs are never swallowed (see make_decoder_env): every
        # unknown name raises, and names whose destination is known get a
        # pointer to it.
        if "directional_grid_size" in unexpected:
            raise TypeError(
                "make_decoder_env_pool: directional_grid_size was renamed — pass "
                "directional_candidates='grid%s'" % unexpected["directional_grid_size"]
            )
        moved = sorted(set(unexpected) & set(_TRAIN_EXTRAS))
        hint = (" — %s moved into the train_extras bundle; pass them as "
                "train_extras={...}" % moved) if moved else ""
        raise TypeError("make_decoder_env_pool: unknown kwarg %s%s" % (sorted(unexpected), hint))

    def _make_factory(s: int):
        def _factory(board: str, reload_seq: int = 0) -> KiCadRLWrapper:
            # ``reload_seq`` = how many times this worker's env has been
            # rebuilt (board hot-swap or crash respawn). The wrapper seeds its
            # numpy Generator once at construction, so rebuilding with the same
            # ``s`` rewinds the whole per-env random stream (augmentation,
            # slot_perm, auto net-select) to its starting point — under
            # per-iteration board reloads that replays the same draws every
            # iteration. Mixing the rebuild count into the seed keeps the
            # stream moving forward across rebuilds. reload_seq == 0 uses ``s``
            # unchanged, so pools that never advance (eval) are unaffected.
            seed_i = s if reload_seq == 0 else int(
                np.random.SeedSequence([s, reload_seq]).generate_state(1)[0]
            )
            return make_decoder_env(
                board,
                max_steps=max_steps,
                masking_rule=masking_rule,
                reward_rule=reward_rule,
                seed=seed_i,
                force_walkaround=force_walkaround,
                mask_start_point=mask_start_point,
                train_extras=train_extras,
                slot_perm=slot_perm,
                emit_drc_tokens=emit_drc_tokens,
                via_penalty=via_penalty,
                wirelength_penalty=wirelength_penalty,
                drc_penalty=drc_penalty,
                drc_log_scale=drc_log_scale,
                drc_log_agg_scale=drc_log_agg_scale,
                drc_log_offset=drc_log_offset,
                reward_step_penalty=reward_step_penalty,
                wire_via_emission=wire_via_emission,
                corner_mode=corner_mode,
                policy_net_select=policy_net_select,
                directional_candidates=directional_candidates,
                connectivity_filter=connectivity_filter,
                pad_graze_margin_mm=pad_graze_margin_mm,
                use_yaml_drc_fallback=use_yaml_drc_fallback,
                drc_config_path=drc_config_path,
                simplify_outline=simplify_outline,
                obs_format=obs_format,
                outline_obs=outline_obs,
                action_history_len=action_history_len,
                net_constraint_obs=net_constraint_obs,
                keep_routing_fraction=keep_routing_fraction,
            )
        return _factory

    board_factories = [_make_factory(seed + i) for i in range(n_envs)]
    env_fns = [(lambda f=f: f(board_path)) for f in board_factories]

    # What the envs were ACTUALLY built with, snapshotted at the function's
    # entry (so it is post-default, post-resolution). Callers dump this rather
    # than the args they *meant* to pass, so the record (ckpt args,
    # config_resolved.yaml, run name) stores effect rather than intent.
    _effective = {**_passed, "train_extras": dict(train_extras or {})}

    def _stamp(pool):
        pool.effective_env_kwargs = _effective
        return pool

    if backend == "subproc":
        return _stamp(SubprocDecoderVecEnv(
            env_fns,
            start_method=start_method,
            board_factories=board_factories,
            advance_rng_on_reload=advance_rng_on_reload,
        ))
    if backend == "ray":
        # env_fns mode: same closures the subprocess pool uses, wrapped in
        # generic Ray actors. group_n=1 → one board per worker (flat pool).
        from pcb_world.vec.backends.ray import RayVecBackend

        return _stamp(RayVecBackend(
            env_fns=env_fns,
            board_factories=board_factories,
            seed=seed,
            env_num=n_envs,
            group_n=1,
            resources_per_worker=resources_per_worker or {"num_cpus": 1},
            board_paths=[board_path] * n_envs,
            advance_rng_on_reload=advance_rng_on_reload,
        ))
    raise ValueError(f"backend must be 'subproc' or 'ray', got {backend!r}")


__all__ = ["make_decoder_env", "make_decoder_env_pool"]
