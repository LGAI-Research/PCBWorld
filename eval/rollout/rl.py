"""Stage-1 RL rollout driver (thin eval-side orchestration).

Extracted from ``methods/rl_agent/rollout/transformer.py`` (the ``eval_transformer``
function): batch ``(board, rollout)`` jobs into waves, drive the per-board rollout
loop (``methods.rl_agent.rollout.transformer._run_one_batch``), flush per-rollout
rows, and optionally aggregate. The rollout *loop* itself stays in
``methods/rl_agent/rollout/transformer.py``; this module is method-agnostic scoring
orchestration (the ``eval/`` layer).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch

from configs.loader.schema import DEFAULTS
from eval import eval_utils as u
from eval.aggregation import aggregate_inline
from eval.metrics import EvalResult
from configs.loader.schema import RLEnvConfig
from methods._shared.board_loader import BoardSpec, count_pads
from methods.rl_agent.rollout.transformer import (
    _run_one_batch,
    max_net_slots_for_policy,
)


def plan_job_schedule(
    boards: list[BoardSpec],
    *,
    n_rollouts: int,
    n_envs: int,
    boards_per_batch: int | None = None,
) -> tuple[list[tuple[BoardSpec, int]], int]:
    """Order ``(board, rollout_idx)`` jobs into gap-free, straggler-minimizing waves.

    Boards are sorted by pad count (a size proxy) so each wave of ``wave_size``
    slots holds similarly-sized boards — the synchronized wave (run-until-all-
    done, see ``_run_one_batch``) then wastes less time waiting on a straggler.
    Waves fill all ``n_envs`` slots; a board's ``n_rollouts`` jobs may span wave
    boundaries, which is exact because each job's seed depends only on
    ``board.index`` and ``rollout_idx`` (not slot/wave position) and aggregation
    groups by ``board_index``.

    ``boards_per_batch`` (legacy): when given, caps a wave to whole-board
    granularity (``boards_per_batch * n_rollouts`` slots) as a memory knob; the
    default (None) fills ``n_envs`` with no gaps. Unlike the old packing there is
    no ``n_envs >= n_rollouts`` requirement — ``n_envs`` alone bounds a wave.
    """
    def _pads(b: BoardSpec) -> int:
        try:
            return count_pads(b.path)
        except OSError:
            return 0

    ordered = sorted(boards, key=_pads)  # stable: equal sizes keep input order
    jobs = [(b, r) for b in ordered for r in range(n_rollouts)]
    if boards_per_batch is not None:
        wave_size = min(n_envs, boards_per_batch * n_rollouts)
    else:
        wave_size = n_envs
    return jobs, max(1, wave_size)


def assert_env_kwargs_as_recorded(
    got: dict[str, Any], recorded: dict[str, Any],
) -> None:
    """Fail loudly when validation builds something other than what was recorded.

    Both sides are env records (``methods.rl_agent.wrappers.factory.env_record``).
    The trainer's ``run_record.val.yaml`` is written at startup by *computing*
    what validation will use; this is the sink confirming that computation the
    first time a real val pool is built. Without it the record would stay a
    plausible-looking guess — which is exactly how the knobs audited on
    2026-08-20 went unnoticed.
    """
    absent = "<absent>"
    delta = {
        key: {"actual": got.get(key, absent), "recorded": recorded.get(key, absent)}
        for key in sorted(set(got) | set(recorded))
        if got.get(key, absent) != recorded.get(key, absent)
    }
    if delta:
        raise RuntimeError(
            "val env effective values differ from the trainer's startup record "
            "(run_record.val.yaml): "
            f"{delta} — code that changes the env config outside "
            "resolve_eval_env has been added."
        )


def eval_policy_net_select(policy: Any) -> bool:
    """Whether the eval env lets ``policy`` pick nets (its own setting)."""
    return bool(getattr(policy, "policy_net_select", False))


def resolve_eval_env(base: RLEnvConfig, *, step_drc: bool, where: str,
                     sources: Mapping[str, str] | None = None, user: Sequence[Any] = ()) -> Any:
    """The env config an eval runs, resolved once (``configs.loader.resolve`` mode ``eval``).

    CKPT = ``base`` — the checkpoint's training env config (``RLEnvConfig.from_checkpoint``) or
    the training run's own (inline validation) — minus the train-only fields
    (:data:`~configs.loader.schema.EVAL_PINNED`: validation and best-ckpt selection score the
    canonical from-scratch board), which keep their defaults (off). ``sources`` are the
    checkpoint loader's per-key sources, ``where`` names ``base``. ``step_drc=False`` adds
    :data:`~configs.loader.schema.DRC_OFF` as a ``mode-default`` layer over the checkpoint and
    under ``user`` (e.g. the eval CLI's overrides — a DRC weight given there wins): only
    ``eval_override`` fields may differ from ``base``.
    Every eval-side adjustment of the env config is a layer here — the trainer records the
    result at startup (``run_record.val.yaml``) without running a rollout, and ``eval_transformer``
    builds exactly what it is given. Returns the ``RunConfig``; :func:`eval_env_config` turns
    it into the ``RLEnvConfig`` the pool is built from.
    """
    from configs.loader.resolve import Layer, ckpt_layer, dest_policies, resolve
    from configs.loader.schema import DRC_OFF, EVAL_PINNED

    drc_off = [] if step_drc else [Layer("mode-default", DRC_OFF,
                                         where={k: "step_drc=False (env DRC off)" for k in DRC_OFF})]
    return resolve(
        "eval", policies=dest_policies(RLEnvConfig()),
        defaults=[Layer("defaults", RLEnvConfig().to_record())],
        ckpt=ckpt_layer({k: v for k, v in base.to_record().items() if k not in EVAL_PINNED},
                        sources or {}, where),
        user=[*drc_off, *user])


def eval_env_config(run: Any) -> RLEnvConfig:
    """The ``RLEnvConfig`` of a :func:`resolve_eval_env` result."""
    return RLEnvConfig().with_flat(run.values)


def check_policy_env(policy: Any, env_config: RLEnvConfig) -> None:
    """Ckpt↔env obs-drift guard: a checkpoint-loaded policy carries its training-time
    ``net_constraint_obs`` (stamped by ``methods.rl_agent.models.loader._load_policy``); the
    env must match, or the NET-token tw/cl/vd channels drift from training (resolved values
    vs constant 0). A config from :func:`resolve_eval_env` over the checkpoint always
    matches (the key is ``arch``), so only a hand-assembled config can trip this; live
    training policies carry no stamp."""
    ckpt_ncobs = getattr(policy, "net_constraint_obs", None)
    if ckpt_ncobs is not None and bool(ckpt_ncobs) != bool(env_config.env.net_constraint_obs):
        raise RuntimeError(
            "net_constraint_obs mismatch: the checkpoint was trained with "
            f"net_constraint_obs={bool(ckpt_ncobs)} but the eval env is built with "
            f"net_constraint_obs={bool(env_config.env.net_constraint_obs)} — the NET-token "
            "constraint channels would silently drift from training. Build the env config "
            "with resolve_eval_env over RLEnvConfig.from_checkpoint instead of hand-assembling it.")


def eval_transformer(
    policy: Any,
    device: torch.device,
    boards: list[BoardSpec],
    *,
    env_config: RLEnvConfig,
    n_rollouts: int,
    n_envs: int,
    base_seed: int,
    max_steps: int,
    deterministic: bool = False,
    boards_per_batch: int | None = None,
    final_drc: bool = False,
    reward_config: str = DEFAULTS.reward_config,
    check_angle: int | None = None,
    selection_mode: Literal["final_potential", "posthoc_drc_aware"] | None = None,
    save_artifacts: bool = False,
    artifacts_dir: Path | None = None,
    early_stop_finish_no_progress: int = 0,
    early_stop_no_geometry_progress: int = 0,
    skip_incompatible: bool = True,
    per_rollout_flush_path: Path | None = None,
    skip_aggregation: bool = False,
    mem_budget: Any | None = None,
    expect_env_kwargs: dict[str, Any] | None = None,
) -> EvalResult:
    """Run rollouts + (optional) inline DRC scoring + aggregation.

    ``env_config`` is final (:func:`resolve_eval_env` → :func:`eval_env_config`): the pool is
    built from it unchanged. Inline DRC results are merged directly into the per_rollout rows (one row
    per board x rollout); there is no separate scoring table. ``check_angle`` is the env
    ``corner_mode`` (:func:`~eval.eval_utils.resolve_check_angle`; an explicit one may only restate it).
    """
    from methods.rl_agent.wrappers.factory import env_record, make_decoder_env_pool

    # (board, rollout) jobs packed into waves that fill all n_envs slots, boards
    # ordered small-first so a wave rarely stalls on a straggler. See
    # plan_job_schedule for why splitting a board's rollouts across waves is exact.
    jobs_all, wave_size = plan_job_schedule(
        boards, n_rollouts=n_rollouts, n_envs=n_envs,
        boards_per_batch=boards_per_batch,
    )

    check_policy_env(policy, env_config)
    policy_net_select = eval_policy_net_select(policy)
    check_angle = u.resolve_check_angle(env_config.env.corner_mode, check_angle)

    if save_artifacts and artifacts_dir is not None:
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    if expect_env_kwargs is not None:
        assert_env_kwargs_as_recorded(
            env_record(env_config, seed=base_seed, policy_net_select=policy_net_select),
            expect_env_kwargs)
    pool = make_decoder_env_pool(
        boards[0].path, n_envs, env_config, seed=base_seed,
        policy_net_select=policy_net_select, advance_rng_on_reload=False,
    )
    per_rollout: list[dict] = []
    max_net_slots = max_net_slots_for_policy(policy)

    was_training = policy.training
    policy.eval()
    try:
        for wave_start in range(0, len(jobs_all), wave_size):
            wave = jobs_all[wave_start : wave_start + wave_size]
            rows = _run_one_batch(
                pool=pool, policy=policy, device=device,
                jobs=wave,
                n_envs=n_envs, max_steps=max_steps,
                base_seed=base_seed, deterministic=deterministic,
                skip_incompatible=skip_incompatible,
                max_net_slots=max_net_slots,
                early_stop_finish_no_progress=int(early_stop_finish_no_progress),
                early_stop_no_geometry_progress=int(early_stop_no_geometry_progress),
                save_artifacts=save_artifacts, artifacts_dir=artifacts_dir,
                final_drc=final_drc, reward_config=reward_config,
                check_angle=check_angle, mem_budget=mem_budget,
            )
            per_rollout.extend(rows)
            if per_rollout_flush_path is not None:
                u.write_csv(per_rollout_flush_path, per_rollout, u.PER_ROLLOUT_FIELDS)
    finally:
        if was_training:
            policy.train()
        try:
            pool.close()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"pool close failed: {exc!r}", stacklevel=2)

    if skip_aggregation:
        per_board: list[dict] = []
        overall: dict[str, Any] = {}
    else:
        per_board, overall = aggregate_inline(
            per_rollout, boards,
            selection_mode=selection_mode,
        )
    return EvalResult(
        per_rollout=per_rollout,
        per_board=per_board,
        overall=overall,
    )
