"""Stage-1 RL rollout driver (thin eval-side orchestration).

``eval_transformer`` batches ``(board, rollout)`` jobs into waves, drives the
per-board rollout loop (``methods.rl_agent.rollout.transformer._run_one_batch``),
flushes per-rollout rows, and optionally aggregates. The rollout *loop* itself
lives in ``methods/rl_agent/rollout/transformer.py``; this module is the
method-agnostic scoring orchestration of the ``eval/`` layer.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import torch

from configs.loader.schema import DEFAULTS
from eval import eval_utils as u
from eval.aggregation import aggregate_inline
from eval.metrics import EvalResult
from methods._shared.board_loader import BoardSpec, count_pads
from methods.rl_agent.models.loader import apply_drc_off
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

    ``boards_per_batch``: when given, caps a wave to whole-board granularity
    (``boards_per_batch * n_rollouts`` slots) as a memory knob; the default
    (None) fills ``n_envs`` with no gaps. There is no ``n_envs >= n_rollouts``
    requirement — ``n_envs`` alone bounds a wave.
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

    ``env_records/val_env.yaml`` is written at trainer startup by *computing*
    what validation will use; this is the sink confirming that computation the
    first time a real val pool is built. Without it the record would stay a
    plausible-looking guess.
    """
    absent = "<absent>"
    delta = {
        key: {"actual": got.get(key, absent), "recorded": recorded.get(key, absent)}
        for key in sorted(set(got) | set(recorded))
        if got.get(key, absent) != recorded.get(key, absent)
    }
    if delta:
        raise RuntimeError(
            "val env effective values differ from the startup record "
            "(env_records/val_env.yaml): "
            f"{delta} — code that mutates env_kwargs outside "
            "resolve_eval_env_kwargs has been added."
        )


def resolve_eval_env_kwargs(
    env_kwargs: dict[str, Any], policy: Any, *, step_drc: bool,
) -> dict[str, Any]:
    """The ONE place eval adjusts the env kwargs it was handed.

    The trainer calls this at startup to learn the *exact* kwargs validation
    will use — without running a rollout — so the train/val record dump and the
    ``--expect-env-diff`` gate compare against reality rather than a guess.
    That only holds while every eval-side adjustment lives here:
    ``tests/test_env_contract.py`` fails if ``eval_transformer`` touches
    ``env_kwargs`` outside this call.
    """
    env_kwargs = dict(env_kwargs)
    env_kwargs["policy_net_select"] = bool(
        getattr(policy, "policy_net_select", False),
    )
    # Train-only augmentation must never leak into validation: val (and the
    # best-ckpt selection built on it) scores the canonical from-scratch
    # board, not a partially pre-routed one. Inline eval passes the TRAINING
    # env_kwargs (env_kwargs_from_training_args), so pin it off here.
    env_kwargs["keep_routing_fraction"] = None
    # Ckpt↔env obs-drift guard: a checkpoint-loaded policy carries its
    # training-time net_constraint_obs (stamped by
    # methods.rl_agent.models.loader._load_policy); the env must match, or the
    # NET-token tw/cl/vd channels drift from training (resolved values vs
    # constant 0 — both directions). The canonical builders
    # (env_kwargs_from_checkpoint / env_kwargs_from_training_args) inherit the
    # knob, so only hand-assembled env_kwargs can trip this; live training
    # policies (trainer inline val) carry no stamp and skip the check.
    ckpt_ncobs = getattr(policy, "net_constraint_obs", None)
    if ckpt_ncobs is not None:
        env_ncobs = bool(env_kwargs.get("net_constraint_obs", False))
        if bool(ckpt_ncobs) != env_ncobs:
            raise RuntimeError(
                "net_constraint_obs mismatch: the checkpoint was trained with "
                f"net_constraint_obs={bool(ckpt_ncobs)} but the eval env is "
                f"built with net_constraint_obs={env_ncobs} — the NET-token "
                "constraint channels would silently drift from training. "
                "Build env_kwargs from the checkpoint args "
                "(env_kwargs_from_checkpoint / RLEnvConfig.from_checkpoint) "
                "instead of hand-assembling them."
            )
    if not step_drc:
        apply_drc_off(env_kwargs)
    return env_kwargs


def eval_transformer(
    policy: Any,
    device: torch.device,
    boards: list[BoardSpec],
    *,
    env_kwargs: dict[str, Any],
    n_rollouts: int,
    n_envs: int,
    base_seed: int,
    max_steps: int,
    deterministic: bool = False,
    boards_per_batch: int | None = None,
    step_drc: bool = True,
    final_drc: bool = False,
    reward_config: str = DEFAULTS.reward_config,
    check_angle: int = DEFAULTS.check_angle,
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

    Inline DRC results are merged directly into the per_rollout rows (one row
    per board x rollout); there is no separate scoring table.
    """
    from methods.rl_agent.wrappers.factory import make_decoder_env_pool

    # (board, rollout) jobs packed into waves that fill all n_envs slots, boards
    # ordered small-first so a wave rarely stalls on a straggler. See
    # plan_job_schedule for why splitting a board's rollouts across waves is exact.
    jobs_all, wave_size = plan_job_schedule(
        boards, n_rollouts=n_rollouts, n_envs=n_envs,
        boards_per_batch=boards_per_batch,
    )

    env_kwargs = resolve_eval_env_kwargs(env_kwargs, policy, step_drc=step_drc)

    if save_artifacts and artifacts_dir is not None:
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    if expect_env_kwargs is not None:
        assert_env_kwargs_as_recorded({**env_kwargs, "seed": base_seed},
                                      expect_env_kwargs)
    pool = make_decoder_env_pool(
        boards[0].path, n_envs, seed=base_seed, **env_kwargs,
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
