"""Policy rollout loop: ``_run_one_batch`` + supporting machinery (per-board
rollout, early-stop guards, per-rollout row assembly). The per-step act+step
transition itself is the shared primitive
:func:`methods.rl_agent.rollout.primitive.iter_rollout`; the eval-side thin
driver ``eval_transformer`` lives in ``eval/rollout/rl.py``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from eval import eval_utils as u
from methods._shared.board_loader import BoardSpec
from pcb_world.core.action_schema import ACTION_NAMES, ACT_IDLE
from methods.rl_agent.rollout.primitive import iter_rollout

REPO_ROOT = Path(__file__).resolve().parents[3]   # rollout -> rl_agent -> methods -> repo root

# Selectable actions only (idle excluded); derived from canonical action_schema.
TRACE_ACTION_NAMES = ACTION_NAMES[:ACT_IDLE]


# ============================================================================
# Policy helpers
# ============================================================================


def load_policy_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any], int]:
    """Load a policy module from a .pt checkpoint."""
    from methods.rl_agent.models.loader import _load_policy
    return _load_policy(ckpt_path, device)


def max_net_slots_for_policy(policy: Any) -> int | None:
    """Return the policy's slot-table size, or None if absent."""
    vocab = getattr(getattr(policy, "tokenizer", None), "vocab", None)
    n_max_slots = getattr(vocab, "n_max_slots", None)
    return None if n_max_slots is None else int(n_max_slots)


def apply_n_max_slots_override(policy: Any, new_value: int | None) -> None:
    """Lift the checkpoint's slot cap for inference."""
    if new_value is None:
        return
    vocab = getattr(getattr(policy, "tokenizer", None), "vocab", None)
    if vocab is None or not hasattr(vocab, "n_max_slots"):
        raise RuntimeError("policy.tokenizer.vocab.n_max_slots not found")
    if not getattr(vocab, "disable_slot_emb", False):
        raise RuntimeError(
            "override_n_max_slots requires checkpoint trained with "
            "disable_slot_emb=True (otherwise slot_emb_table would be "
            "indexed out of range)"
        )
    if getattr(vocab, "slot_perm", False):
        raise RuntimeError("override_n_max_slots requires slot_perm=False")
    prev = int(vocab.n_max_slots)
    vocab.n_max_slots = int(new_value)
    print(f"[eval] n_max_slots: {prev} -> {int(new_value)}", flush=True)


def _board_net_count_from_obs(obs: dict[str, Any]) -> int:
    board_static = obs.get("board_static", {})
    if "net_code" in board_static:  # indexed_v1 tables
        return len(board_static["net_code"])
    nets = board_static.get("nets", {})
    return len(nets) if isinstance(nets, dict) else 0


# ============================================================================
# Early-stop guard
# ============================================================================


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _geometry_signature(info: dict[str, Any]) -> tuple[float, float, float, float] | None:
    vals = (
        _float_or_nan(info.get("track_count")),
        _float_or_nan(info.get("via_count")),
        _float_or_nan(info.get("wirelength")),
        _float_or_nan(info.get("ratsnest_reduction")),
    )
    if all(math.isnan(v) for v in vals):
        return None
    return vals


def _geometry_changed(
    prev: tuple[float, float, float, float] | None,
    cur: tuple[float, float, float, float] | None,
) -> bool:
    if prev is None or cur is None:
        return False
    for a, b in zip(prev, cur):
        if math.isnan(a) or math.isnan(b):
            continue
        if abs(a - b) > 1e-9:
            return True
    return False


@dataclass
class FinishNoProgressGuard:
    """Track consecutive ``finish`` attempts that failed without geometry change."""
    threshold: int = 0
    count: int = 0
    last_geometry: tuple[float, float, float, float] | None = None

    def update(
        self, *, action_name: str, action_class: str, info: dict[str, Any],
    ) -> bool:
        cur = _geometry_signature(info)
        if _geometry_changed(self.last_geometry, cur):
            self.count = 0
        if cur is not None:
            self.last_geometry = cur
        if action_name == "finish" and action_class in {
            "valid_dispatch_fail", "valid_empty",
        }:
            self.count += 1
        return self.threshold > 0 and self.count >= self.threshold


@dataclass
class NoGeometryProgressGuard:
    """Track consecutive steps with no observed geometry metric change."""
    threshold: int = 0
    count: int = 0
    last_geometry: tuple[float, float, float, float] | None = None

    def update(self, *, info: dict[str, Any]) -> bool:
        cur = _geometry_signature(info)
        if cur is None:
            return self.threshold > 0 and self.count >= self.threshold
        if self.last_geometry is None:
            self.last_geometry = cur
            return False
        if _geometry_changed(self.last_geometry, cur):
            self.count = 0
        else:
            self.count += 1
        self.last_geometry = cur
        return self.threshold > 0 and self.count >= self.threshold


# ============================================================================
# Per-rollout row construction
# ============================================================================


def _row_from_info(
    *,
    board: BoardSpec,
    rollout_idx: int,
    seed: int,
    deterministic: bool,
    info: dict[str, Any],
    reward: float,
    steps: int,
    terminated: bool,
    truncated: bool,
    artifact_path: str | None,
    rollout_finish_time: float,
    per_board_rollout_time: float,
) -> dict[str, Any]:
    """Identity + producer bookkeeping around the shared metric semantics.

    All metric definitions (success, clean_pass, routability, ...) live in
    :func:`eval.eval_utils.runtime_metrics_from_info` — nothing is derived
    here.
    """
    return {
        "board_index": int(board.index),
        "board_id": str(board.board_id),
        "board_path": str(board.path),
        "rollout_idx": int(rollout_idx),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "skip_reason": "",
        "steps": int(steps),
        "episode_return": float(reward),
        "rollout_finish_time": float(rollout_finish_time),
        "per_board_rollout_time": per_board_rollout_time,
        "artifact_path": artifact_path or "",
        **u.runtime_metrics_from_info(
            info, terminated=terminated, truncated=truncated,
        ),
    }


def _incompatible_row(
    *,
    board: BoardSpec,
    rollout_idx: int,
    seed: int,
    deterministic: bool,
    board_net_count: int,
    max_net_slots: int,
) -> dict[str, Any]:
    return u.assemble_rollout_row(
        identity={
            "board_index": int(board.index),
            "board_id": str(board.board_id),
            "board_path": str(board.path),
            "rollout_idx": int(rollout_idx),
            "seed": int(seed),
            "deterministic": bool(deterministic),
        },
        runtime={
            "status": "skipped_incompatible_board",
            "skip_reason": "board_net_count_exceeds_policy_slots",
            "error": (
                f"Board has {board_net_count} nets but policy slot table has "
                f"only {max_net_slots} slots."
            ),
        },
    )


# ============================================================================
# Per-slot finalize (DRC scoring + save_pcb + row construction)
# ============================================================================


def _finalize_slot(
    *,
    pool: Any,
    slot: int,
    board: BoardSpec,
    rollout_idx: int,
    seed: int,
    deterministic: bool,
    info: dict[str, Any],
    reward: float,
    steps: int,
    terminated: bool,
    truncated: bool,
    save_artifacts: bool,
    artifacts_dir: Path | None,
    final_drc: bool,
    reward_config: str,
    check_angle: int,
    per_board_rollout_time: float,
) -> dict[str, Any]:
    crashed = bool(info.get("engine_crash", False))
    artifact_path_str: str | None = None
    out_path: Path | None = None
    if artifacts_dir is not None and save_artifacts:
        out_path = (
            Path(artifacts_dir)
            / f"board_{board.index:05d}_rollout_{rollout_idx}.kicad_pcb"
        )

    # Inline DRC scoring: enrich this rollout's row in place (serial mode).
    scored: dict | None = None
    scored_error = ""
    drc_meta: dict[str, Any] = {}
    if final_drc and not crashed and out_path is not None:
        t0 = time.monotonic()
        try:
            scored = pool.env_method(
                "eval_inline_drc",
                reward_config_name=reward_config,
                check_angle=check_angle,
                routed_pcb=str(out_path),
                pro_path="",
                indices=[slot],
            )[0]
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb
            scored_error = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"
        drc_meta = {
            "routed_pcb": str(out_path),
            "pro_path": "",
            "eval_time_sec": round(time.monotonic() - t0, 4),
        }

    rollout_finish_time = time.time()

    if out_path is not None and not crashed:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pool.env_method("save_pcb", str(out_path), indices=[slot])
            u.copy_kicad_sidecars(out_path, board.path)
            artifact_path_str = str(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] WARN: save_pcb failed for slot {slot}: {exc!r}")

    row = u.assemble_rollout_row(
        scored,
        error=scored_error,
        runtime=_row_from_info(
            board=board, rollout_idx=rollout_idx, seed=seed,
            deterministic=deterministic, info=info, reward=reward, steps=steps,
            terminated=terminated, truncated=truncated,
            artifact_path=artifact_path_str,
            rollout_finish_time=rollout_finish_time,
            per_board_rollout_time=per_board_rollout_time,
        ),
        reward_config=reward_config,
        check_angle=check_angle,
    )
    row.update(drc_meta)
    return row


# ============================================================================
# Core rollout loop
# ============================================================================


def _run_one_batch(
    *,
    pool: Any,
    policy: Any,
    device: torch.device,
    jobs: list[tuple[BoardSpec, int]],
    n_envs: int,
    max_steps: int,
    base_seed: int,
    deterministic: bool,
    skip_incompatible: bool,
    max_net_slots: int | None,
    early_stop_finish_no_progress: int,
    early_stop_no_geometry_progress: int,
    save_artifacts: bool,
    artifacts_dir: Path | None,
    final_drc: bool,
    reward_config: str,
    check_angle: int,
    mem_budget: Any | None = None,
) -> list[dict]:
    slot_count = len(jobs)
    # Per-rollout routing time is only attributable when a single rollout owns
    # the step loop (n_envs==1); parallel waves interleave many rollouts.
    record_routing_time = n_envs == 1
    assert slot_count <= n_envs, (
        f"slot_count {slot_count} exceeds n_envs {n_envs}"
    )
    slot_boards = [job[0] for job in jobs]
    slot_rollouts = [job[1] for job in jobs]
    slot_paths = [b.path for b in slot_boards]
    padded = slot_paths + [slot_paths[0]] * (n_envs - slot_count)
    pool.reload_boards(padded)

    # Per-slot seed: derived from (base_seed, board, rollout) so a rollout is
    # reproducible regardless of which pool slot it lands in. It is both
    # RECORDED in the output rows and APPLIED at reset — ``env.reset(seed=...)``
    # seeds that env's gymnasium ``np_random``. Previously it was only recorded,
    # so the seed column labelled rows that nothing had actually seeded.
    #
    # Scope, honestly: ``np_random``'s only consumer today is the
    # keep_routing_fraction draw, which eval pins off (see rl.eval_transformer),
    # so applying the seed is currently a no-op on eval numbers. It closes the
    # gap for any future env-side per-episode sampling. The nondeterminism that
    # DOES move val/* is elsewhere — policy sampling when deterministic=False
    # (torch RNG; val_greedy is immune) and the wrapper's own ``_rng``.
    slot_seeds = [
        base_seed + board.index * 100 + rollout_idx
        for board, rollout_idx in zip(slot_boards, slot_rollouts)
    ]

    active_slots = list(range(slot_count))
    reset_out = pool.reset_batch(
        active_slots, [slot_seeds[slot] for slot in active_slots],
    )
    obs_by_slot: dict[int, dict[str, Any]] = {
        slot: reset_out[k][0] for k, slot in enumerate(active_slots)
    }

    rows: list[dict] = []

    if skip_incompatible and max_net_slots is not None:
        compatible: list[int] = []
        for slot in active_slots:
            board = slot_boards[slot]
            rollout_idx = slot_rollouts[slot]
            seed = slot_seeds[slot]
            net_count = _board_net_count_from_obs(obs_by_slot[slot])
            if net_count > max_net_slots:
                rows.append(_incompatible_row(
                    board=board, rollout_idx=rollout_idx, seed=seed,
                    deterministic=deterministic,
                    board_net_count=net_count, max_net_slots=max_net_slots,
                ))
            else:
                compatible.append(slot)
        active_slots = compatible

    if not active_slots:
        return rows

    done: dict[int, bool] = {slot: False for slot in active_slots}
    rewards: dict[int, float] = {slot: 0.0 for slot in active_slots}
    lengths: dict[int, int] = {slot: 0 for slot in active_slots}
    last_info: dict[int, dict[str, Any]] = {slot: {} for slot in active_slots}
    finish_guards: dict[int, FinishNoProgressGuard] = (
        {slot: FinishNoProgressGuard(threshold=int(early_stop_finish_no_progress))
         for slot in active_slots}
        if early_stop_finish_no_progress > 0 else {}
    )
    geometry_guards: dict[int, NoGeometryProgressGuard] = (
        {slot: NoGeometryProgressGuard(threshold=int(early_stop_no_geometry_progress))
         for slot in active_slots}
        if early_stop_no_geometry_progress > 0 else {}
    )

    route_start = time.monotonic()

    def _routing_time() -> float:
        return time.monotonic() - route_start if record_routing_time else math.nan

    for batch in iter_rollout(
        pool, policy, device, obs_by_slot, active_slots, done,
        want_value=False, deterministic=deterministic, max_steps=max_steps,
        mem_budget=mem_budget,
    ):
        acts_np = batch.actions
        for k, slot in enumerate(batch.live):
            rewards[slot] += float(batch.rewards[k])
            lengths[slot] += 1
            info = batch.infos[k]
            step_terminated = bool(batch.terminateds[k])
            step_truncated = bool(batch.truncateds[k])
            if (
                (finish_guards or geometry_guards)
                and not step_terminated and not step_truncated
            ):
                info = dict(info)
                action_name = (
                    TRACE_ACTION_NAMES[int(acts_np[k][0])]
                    if 0 <= int(acts_np[k][0]) < len(TRACE_ACTION_NAMES)
                    else "unknown"
                )
                finish_triggered = (
                    slot in finish_guards
                    and finish_guards[slot].update(
                        action_name=action_name,
                        action_class=str(info.get("action_class", "")),
                        info=info,
                    )
                )
                if slot in finish_guards:
                    info["finish_no_progress_since_progress"] = (
                        finish_guards[slot].count
                    )

                geometry_triggered = (
                    slot in geometry_guards
                    and geometry_guards[slot].update(info=info)
                )
                if slot in geometry_guards:
                    info["no_geometry_progress_since_progress"] = (
                        geometry_guards[slot].count
                    )

                if finish_triggered:
                    info["early_stop"] = True
                    info["early_stop_reason"] = "finish_no_progress"
                    info["early_stop_threshold"] = int(early_stop_finish_no_progress)
                    step_truncated = True
                elif geometry_triggered:
                    info["early_stop"] = True
                    info["early_stop_reason"] = "no_geometry_progress"
                    info["early_stop_threshold"] = int(early_stop_no_geometry_progress)
                    step_truncated = True
            last_info[slot] = info
            if step_terminated or step_truncated:
                done[slot] = True
                board = slot_boards[slot]
                rollout_idx = slot_rollouts[slot]
                seed = slot_seeds[slot]
                rows.append(_finalize_slot(
                    pool=pool, slot=slot, board=board,
                    rollout_idx=rollout_idx, seed=seed,
                    deterministic=deterministic,
                    info=info, reward=rewards[slot], steps=lengths[slot],
                    terminated=step_terminated, truncated=step_truncated,
                    save_artifacts=save_artifacts, artifacts_dir=artifacts_dir,
                    final_drc=final_drc, reward_config=reward_config,
                    check_angle=check_angle,
                    per_board_rollout_time=_routing_time(),
                ))

    for slot in active_slots:
        if done[slot]:
            continue
        board = slot_boards[slot]
        rollout_idx = slot_rollouts[slot]
        seed = slot_seeds[slot]
        rows.append(_finalize_slot(
            pool=pool, slot=slot, board=board,
            rollout_idx=rollout_idx, seed=seed,
            deterministic=deterministic,
            info=last_info[slot], reward=rewards[slot], steps=lengths[slot],
            terminated=False, truncated=True,
            save_artifacts=save_artifacts, artifacts_dir=artifacts_dir,
            final_drc=final_drc, reward_config=reward_config,
            check_angle=check_angle,
            per_board_rollout_time=_routing_time(),
        ))
    return rows
