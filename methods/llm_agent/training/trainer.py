"""Central LLM trainer — owns the loop, delegates the train step to verl.

The LLM counterpart of :class:`methods.rl_agent.training.loop.RLTrainer`: a
:class:`methods._shared.trainer.base.Trainer` subclass that drives iterations,
validation cadence (central :class:`eval.evaluator.Evaluator` -> ``val/*``),
and checkpoint cadence itself, delegating ONLY the heavy training step to the
patched verl ``RayPPOTrainer.train_step`` (one batch: gen -> agent-env rollout
-> advantage -> update). Wired in by the verl-agent ``main_ppo`` patch when
``config.env.env_name`` is cadagent; other env types keep verl's own
``fit()`` loop.

Validation flow (the RL/LLM-shared surface)::

    Trainer.validate() -> Evaluator.run(_val_rollout_fn)
        -> multi_turn_loop on val_envs (in-training policy via verl actor)
        -> KiCadLLMRolloutManager episode-end scoring -> pop_episode_rows()
        -> EvalSummary -> log_metrics(prefix="val")

so RL and LLM training log the same canonical ``val/<overall>`` +
``val/per_board/<board_id>/<key>`` tags from the same projection.

verl / torch / omegaconf are imported lazily inside methods, keeping this
module importable in the publish-side cadagent repo without verl installed.
"""
from __future__ import annotations

import os
import time
from typing import Any, List, Optional

from methods._shared.trainer.base import Trainer

__all__ = ["LLMTrainer", "TrackingLogger"]


# RL-shaped scalar tag  <-  verl metric_dict key.
#
# Mirrors the verl train-step metrics onto the same ``train/*`` / ``rollout/*``
# vocabulary the RL trainer logs (see methods/rl_agent/training/loop.py: ``_log_common`` /
# ``log_algo_metrics``), so a wandb dashboard overlays RL and LLM runs on the
# same panels. verl's native keys (actor/* critic/* episode/* perf/*) are kept
# alongside — this only ADDS aliases.
#
# Only quantities with a faithful verl source are mapped. Deliberately NOT
# mirrored (no honest equivalent — mapping them would be a false equivalence):
#   train/loss              verl has no single combined loss scalar
#   actions/<type>          verl does not track per-action-type ratios
#   diag/gpu_mem_*          verl perf/* is throughput/time, not gpu-mem
#   train/reward_norm_std   RL-only reward normalizer
_RL_KEY_FROM_VERL: dict[str, str] = {
    # train/* — optimizer + losses
    "train/policy_loss":          "actor/pg_loss",
    "train/value_loss":           "critic/vf_loss",
    "train/entropy":              "actor/entropy_loss",
    "train/explained_variance":   "critic/vf_explained_var",
    "train/learning_rate":        "actor/lr",
    "train/time_per_iter":        "perf/time_per_step",
    "train/epoch":                "training/epoch",
    # rollout/* — episode aggregates (cadagent 6-metric panel + reward/length)
    "rollout/mean_reward":        "episode/reward/mean",
    "rollout/mean_ep_length":     "episode/length/mean",
    "rollout/terminated_rate":    "episode/success_rate",
    "rollout/ratsnest_reduction_mean": "episode/ratsnest_reduction/mean",
    "rollout/drc_violations_mean": "episode/drc_violations/mean",
    "rollout/final_wirelength_mean": "episode/wirelength/mean",
    "rollout/final_via_count_mean": "episode/via_count/mean",
    "rollout/final_track_count_mean": "episode/track_count/mean",
}


def mirror_rl_keys(metrics: dict) -> dict:
    """Return RL-shaped aliases for the verl keys present in ``metrics``.

    Pure (no mutation); only emits an alias when its verl source key exists,
    so a missing metric (e.g. no critic) silently drops its alias rather than
    logging a stale/zero value.
    """
    return {
        rl_key: metrics[verl_key]
        for rl_key, verl_key in _RL_KEY_FROM_VERL.items()
        if verl_key in metrics
    }


class TrackingLogger:
    """:class:`methods._shared.logger.MetricLogger`-conforming adapter over verl Tracking.

    verl's ``Tracking`` already owns the wandb run (and TB writer), so the
    central trainer must NOT spin up :func:`methods._shared.logger.build_logger` (double
    ``wandb.init``). This adapter routes ``add_scalar`` through the existing
    Tracking sink instead; same-step calls merge into one wandb row.
    """

    def __init__(self, tracking: Any) -> None:
        self._tracking = tracking

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        try:
            self._tracking.log(data={tag: float(value)}, step=int(step))
        except Exception:  # noqa: BLE001 — never crash training on a sink
            pass


class LLMTrainer(Trainer):
    """Drive verl-agent LLM training through the central Trainer lifecycle.

    Construction takes an already-built (but not yet worker-initialised)
    patched ``RayPPOTrainer``; :meth:`setup` finishes bring-up
    (``init_workers`` / Tracking / checkpoint resume) and wires the central
    Evaluator over the val env manager. ``fit()`` then runs the base loop:
    ``train_iteration`` -> ``on_iteration_end`` (validate/save cadence).
    """

    def __init__(self, verl_trainer: Any) -> None:
        self.verl = verl_trainer
        self.config = verl_trainer.config
        # Real horizon set in setup() once dataloaders exist; base fit() reads
        # self.iterations only after setup().
        super().__init__(iterations=0)
        self._train_iter = None
        self._epoch = 0
        self._rollout_counts: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        from eval.evaluator import Evaluator

        self.verl.init_workers()

        self._tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.writer = TrackingLogger(self._tracking)

        self.verl.global_steps = 0
        self.verl._load_checkpoint()
        self.iterations = int(self.verl.total_training_steps)
        self.start_iteration = int(self.verl.global_steps) + 1

        self.evaluator = (
            Evaluator(self._val_rollout_fn, boards=self._val_boards())
            if self.verl.val_reward_fn is not None else None
        )

        if self.evaluator is not None and self.config.trainer.get(
            "val_before_train", True,
        ):
            self.validate(self.verl.global_steps)
        if self.config.trainer.get("val_only", False):
            # Collapse the loop to zero iterations: fit() goes straight from
            # the validation above to teardown.
            self.iterations = self.start_iteration - 1

    def teardown(self) -> None:
        for envs in (getattr(self.verl, "envs", None),
                     getattr(self.verl, "val_envs", None)):
            close = getattr(envs, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass
        print(f"[llm-trainer] done at step {self.verl.global_steps}.")

    # ------------------------------------------------------------------
    # Train step (the one verl delegation point)
    # ------------------------------------------------------------------

    def _next_batch(self):
        """Cycle the verl train dataloader across epochs."""
        if self._train_iter is None:
            self._train_iter = iter(self.verl.train_dataloader)
        try:
            return next(self._train_iter)
        except StopIteration:
            self._epoch += 1
            self._train_iter = iter(self.verl.train_dataloader)
            return next(self._train_iter)

    def train_iteration(self, iteration: int) -> dict:
        # verl internals (checkpoint paths, best-ckpt stamps, dump files) key
        # off global_steps; keep it in lockstep with the central iteration.
        self.verl.global_steps = iteration
        t0 = time.time()
        metrics = self.verl.train_step(self._next_batch(), epoch=self._epoch)
        # Add RL-shaped train/*·rollout/* aliases so RL and LLM runs share one
        # dashboard vocabulary; verl-native keys are kept alongside.
        metrics.update(mirror_rl_keys(metrics))
        try:
            self._tracking.log(data=metrics, step=iteration)
        except Exception:  # noqa: BLE001 — never crash training on a sink
            pass
        print(
            f"[llm-trainer] step {iteration}/{self.iterations}  "
            f"epoch={self._epoch}  ({time.time() - t0:.1f}s)"
        )
        return metrics

    def on_iteration_end(self, iteration: int, metrics: dict) -> None:
        # W5 (eval/val equivalence) — resolved: val/* is cross-branch comparable.
        #   * val scoring config is shared: the LLM manager (_resolve_eval_*) and
        #     the RL inline-eval path both default to the same configs.loader.schema
        #     RLEvalConfig (eval_reward_config / check_angle), so both branches
        #     score with the same ruler.
        #   * best-checkpoint selection uses the single canonical
        #     RLEvalConfig.best_metric_key (= val/fp_mean_of_means, max): LLM via
        #     trainer.best_metric_key (run_cadagent*.sh) below, RL via
        #     RLTrainer.on_validation — both pick "largest fp gain is best".
        is_last = iteration >= self.iterations
        test_freq = self.config.trainer.test_freq
        if (
            self.evaluator is not None
            and test_freq > 0
            and (is_last or iteration % test_freq == 0)
        ):
            eval_metrics = self.validate(iteration)
            if eval_metrics is not None:
                flat = eval_metrics.to_logger_dict(prefix="val")
                # Config-driven best checkpoint (trainer.best_metric_key picks
                # any canonical val/* tag). Mutates ``flat`` with best stamps.
                self.verl._maybe_save_best(flat)
                for key in ("val/best_score", "val/best_step"):
                    if key in flat:
                        self.writer.add_scalar(key, flat[key], iteration)
        save_freq = self.config.trainer.save_freq
        if (save_freq > 0 and (is_last or iteration % save_freq == 0)) or (
            save_freq == -1 and is_last
        ):
            self.verl._save_checkpoint()

    # ------------------------------------------------------------------
    # Validation rollout (Evaluator mode A injection)
    # ------------------------------------------------------------------

    def _val_boards(self) -> list:
        """BoardSpec list for the Evaluator, from the val manager's scheduler."""
        from methods._shared.board_loader import BoardSpec

        scheduler = getattr(self.verl.val_envs, "_board_scheduler", None)
        paths = list(scheduler.paths) if scheduler is not None else []
        return [
            BoardSpec(i, os.path.splitext(os.path.basename(p))[0], str(p))
            for i, p in enumerate(paths)
        ]

    def _val_rollout_fn(self, boards: list) -> List[dict]:
        """Run one validation pass; return canonical per-rollout rows.

        Generation goes through the in-training policy (verl actor +
        ``multi_turn_loop`` on ``val_envs``, exactly like verl's
        ``_validate``); scoring comes from the manager's episode-end hook
        (``pop_episode_rows``). Top-K generation/trajectory wandb tables are
        preserved via the shared verl helpers. ``boards`` is the Evaluator's
        own list — unseen boards are appended in place so aggregation sees
        them.
        """
        from verl import DataProto

        verl_t = self.verl
        rows_all: List[dict] = []
        val_log_groups: list = []
        all_trajectory_logs: list = []

        for test_data in verl_t.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=verl_t.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            for key in ("multi_modal_data", "raw_prompt", "tools_kwargs",
                        "env_kwargs"):
                if key in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(key)
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            test_gen_batch.meta_info = {
                "eos_token_id": verl_t.tokenizer.eos_token_id,
                "pad_token_id": verl_t.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample":
                    verl_t.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            out = verl_t.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                actor_rollout_wg=verl_t.actor_rollout_wg,
                envs=verl_t.val_envs,
                is_train=False,
            )

            # Canonical rows from the manager's episode-end scoring.
            rows_all.extend(verl_t.val_envs.pop_episode_rows())

            # Top-K tables (best-effort; reward only ranks the table).
            try:
                result = verl_t.val_reward_fn(out, return_dict=True)
                verl_t._collect_val_groups(
                    out, result["reward_tensor"], val_log_groups,
                )
                ntb = out.non_tensor_batch
                if "trajectory_logs" in ntb and "traj_uid" in ntb:
                    seen, unique_logs = set(), []
                    for uid, log in zip(ntb["traj_uid"], ntb["trajectory_logs"]):
                        if uid not in seen:
                            seen.add(uid)
                            unique_logs.append(log)
                    all_trajectory_logs.append(unique_logs)
            except Exception as exc:  # noqa: BLE001 — tables must not kill val
                print(f"[llm-trainer] val table logging skipped: {exc!r}")

        verl_t._maybe_log_val_generations(val_log_groups)
        verl_t._maybe_log_trajectory_json(all_trajectory_logs)
        return self._assign_identity(rows_all, boards)

    def _assign_identity(self, rows: List[dict], boards: list) -> List[dict]:
        """Complete row identity: ``board_index`` from the board list (rows
        carry ``board_path`` from the worker), ``rollout_idx`` by running
        count per board across the whole training run."""
        from methods._shared.board_loader import BoardSpec

        index_by_path = {b.path: b.index for b in boards}
        for row in rows:
            path = str(row.get("board_path", "") or "")
            if path not in index_by_path:
                # Scheduler surprises (e.g. per_env_random draws outside the
                # initial list) — extend the shared board list in place.
                spec = BoardSpec(
                    len(boards),
                    os.path.splitext(os.path.basename(path))[0] if path
                    else f"board_{len(boards)}",
                    path,
                )
                boards.append(spec)
                index_by_path[path] = spec.index
            bi = index_by_path[path]
            row["board_index"] = bi
            row["rollout_idx"] = self._rollout_counts.get(bi, 0)
            self._rollout_counts[bi] = row["rollout_idx"] + 1
        return rows
