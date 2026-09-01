"""RL trainers: shared core (`RLTrainer`) + per-algorithm subclasses.

`RLTrainer(Trainer)` holds everything PPO and GRPO share — env/policy/optimizer
build, checkpoint resume, board scheduling, the per-iteration skeleton
(select boards -> collect -> targets -> update -> aggregate -> log), the shared
train/rollout/actions/diag logging, validation (central `eval.evaluator`), and
checkpointing. Algorithm-specific behaviour lives in `PPOTrainer` / `GRPOTrainer`
via the abstract hooks (collect_rollout, compute_targets, update_kwargs,
aggregate_metrics, log_algo_metrics, setup_algo_state).

The heavy RL ops come from ``training.{collect,buffer}`` + ``algorithms.*``; eval
from the canonical ``eval.*`` packages; the model from ``methods.rl_agent.policy``.
"""
from __future__ import annotations

import os
import time
from abc import abstractmethod
from types import SimpleNamespace

import numpy as np
import torch

from methods.rl_agent.algorithms._common import policy_update_loop
from methods.rl_agent.wrappers.factory import make_decoder_env, make_decoder_env_pool
from methods.rl_agent.training.utils import auto_device
from methods._shared.board_scheduler import BoardScheduler, BoardSchedulerConfig
from methods._shared.logger import build_logger
from methods._shared.trainer.base import Trainer
from pcb_world.core.action_schema import ACTION_NAMES, ACT_IDLE

# Selectable actions only (idle, at index ACT_IDLE, is the LLM parse-fail
# fallback and never appears in RL traces). Derived from the canonical
# action_schema so it can't drift.
_ACTION_NAMES = ACTION_NAMES[:ACT_IDLE]


def load_eval_boards(args):
    """Primary held-out eval board list (``--eval-split``), or None when unset.

    Shared by the trainer (``_setup_eval_boards``) and the ``--async-val``
    watcher (:mod:`methods.rl_agent.training.async_val`).
    """
    if args.eval_split is None:
        return None
    from methods._shared.board_loader import load_boards_from_split_json as load_boards

    eval_boards = load_boards(
        args.boards_json, args.boards_difficulty, args.eval_split,
        dataset_dir=args.boards_dataset_dir,
    )
    if args.eval_board_limit is not None:
        eval_boards = eval_boards[: args.eval_board_limit]
    return eval_boards


def build_evaluators(args, agent, device, eval_boards, *, mem_budget=None,
                     expect_env_kwargs=None):
    """Primary + diagnostic (eval2..eval5) Evaluators over one shared rollout fn.

    The single construction point for in-training validation — used by the
    trainer's ``_setup_evaluator`` AND the ``--async-val`` watcher, so the eval
    semantics (step_drc, seeds, batching) cannot drift between them. Returns
    ``(primary_evaluator, [(prefix, evaluator), ...])``.
    """
    from pathlib import Path

    from eval.evaluator import Evaluator
    from eval.rollout.rl import eval_transformer
    from methods._shared.board_loader import load_boards_from_dir_or_list
    from methods.rl_agent.models.loader import env_kwargs_from_training_args

    def _rollout_fn(boards):
        return eval_transformer(
            agent, device, boards,
            env_kwargs=env_kwargs_from_training_args(args),
            n_rollouts=args.eval_n_rollouts, n_envs=args.n_envs,
            base_seed=args.eval_base_seed, max_steps=args.max_steps,
            boards_per_batch=args.eval_boards_per_batch,
            # Always MEASURE DRC in validation (step_drc=True) so val metrics
            # (drc_violations, final_potential) are comparable across cells.
            # The --no-drc-tokens axis only suppresses DRC *observation* tokens
            # (env_kwargs handles that via emit_drc_tokens=not no_drc_tokens);
            # it has no effect on the DRC reward computed here.
            step_drc=True, final_drc=False,
            save_artifacts=False,
            mem_budget=mem_budget,
            # Cross-check the trainer's startup record against what validation
            # actually builds — the record is a prediction until a real pool
            # confirms it.
            expect_env_kwargs=expect_env_kwargs,
        )

    evaluator = Evaluator(_rollout_fn, eval_boards)

    # Optional diagnostic eval sets (eval2..eval5): fixed board-lists (e.g.
    # real d3-b / d3-a) evaluated each cadence under <prefix>/*. Reuse
    # _rollout_fn (same agent / env_kwargs), only the board list differs;
    # do NOT drive best-ckpt. Disabled (backward-compatible) when unset.
    #
    # --eval-diag-max-steps / --eval-diag-masking-rule pin the diagnostic
    # protocol cell-independently (rules/experiments.md §6: inline eval
    # otherwise inherits the TRAIN env_kwargs, so cells varying max_steps or
    # masking would silently change their own diagnostics). Primary val and
    # val_greedy stay native — they drive/read best-ckpt under the run's own
    # protocol.
    diag_max_steps = getattr(args, "eval_diag_max_steps", None)
    diag_masking = getattr(args, "eval_diag_masking_rule", None)
    if diag_max_steps is None and diag_masking is None:
        _rollout_fn_diag = _rollout_fn
    else:
        diag_env_kwargs = env_kwargs_from_training_args(args)
        eff_max_steps = (
            int(diag_max_steps) if diag_max_steps is not None else args.max_steps
        )
        diag_env_kwargs["max_steps"] = eff_max_steps
        if diag_masking is not None:
            diag_env_kwargs["masking_rule"] = str(diag_masking)
        print(
            "  diag eval override (eval2..eval5): "
            f"max_steps={eff_max_steps} "
            f"masking_rule={diag_env_kwargs['masking_rule']}"
        )

        def _rollout_fn_diag(boards):
            return eval_transformer(
                agent, device, boards,
                env_kwargs=dict(diag_env_kwargs),
                n_rollouts=args.eval_n_rollouts, n_envs=args.n_envs,
                base_seed=args.eval_base_seed, max_steps=eff_max_steps,
                boards_per_batch=args.eval_boards_per_batch,
                step_drc=True, final_drc=False,
                save_artifacts=False,
                mem_budget=mem_budget,
            )

    extras = []
    for boards_file, pfx in (
        (args.eval2_boards, args.eval2_prefix),
        (args.eval3_boards, args.eval3_prefix),
        (getattr(args, "eval4_boards", None), getattr(args, "eval4_prefix", "val_d2b")),
        (getattr(args, "eval5_boards", None), getattr(args, "eval5_prefix", "val_d2a")),
    ):
        if not boards_file:
            continue
        bl = load_boards_from_dir_or_list(boards_list=Path(boards_file))
        extras.append((pfx, Evaluator(_rollout_fn_diag, bl)))
        print(f"  inline eval ({Path(boards_file).name}): "
              f"{len(bl)} boards -> {pfx}/*")

    # Greedy (argmax) pass on the primary val set: 1 deterministic rollout per
    # board under val_greedy/* (mean==max by construction). Read next to the
    # sampled val/*: val_greedy ≈ val/fp_mean_of_maxes → the maxes-means gap is
    # exploration noise (temperature-recoverable); val_greedy ≈
    # val/fp_mean_of_means → genuine uncertainty. Diagnostic only — best-ckpt
    # stays on the sampled val/*.
    if eval_boards and not getattr(args, "no_eval_greedy", False):
        def _rollout_fn_greedy(boards):
            return eval_transformer(
                agent, device, boards,
                env_kwargs=env_kwargs_from_training_args(args),
                n_rollouts=1, n_envs=args.n_envs,
                base_seed=args.eval_base_seed, max_steps=args.max_steps,
                deterministic=True,
                boards_per_batch=args.eval_boards_per_batch,
                step_drc=True, final_drc=False,
                save_artifacts=False,
                mem_budget=mem_budget,
            )

        extras.append(("val_greedy", Evaluator(_rollout_fn_greedy, eval_boards)))
        print(f"  inline eval (greedy argmax): {len(eval_boards)} boards -> val_greedy/*")
    return evaluator, extras


class RLTrainer(Trainer):
    """Shared decoder-policy RL training core (PPO/GRPO)."""

    # --- subclass contract ---
    ALGO: str = "rl"
    USE_CRITIC: bool = True
    ALLOWED_MISSING_KEYS: set[str] = {"prev_action", "history_age", "drc"}

    def __init__(self, args) -> None:
        super().__init__(iterations=args.iterations,
                         max_wallclock_sec=getattr(args, "max_wallclock_sec", None))
        self.args = args
        # populated in setup()
        self.device = None
        self.policy = None
        self.optimizer = None
        self.lr_scheduler = None
        self.envs = None
        self.boards = None
        self.eval_boards = None
        self._ckpt_stamp = None  # provenance + obs probe, built on first save
        # optional diagnostic eval sets [(prefix, Evaluator)] -> <prefix>/* (eval2..eval5)
        self.extra_evaluators = []
        self.eval_cadence = None
        self._corner_mode_int = 0
        # Board scheduling: a methods._shared.board_scheduler.BoardScheduler built in
        # _setup_boards owns selection/RNG/epoch state. ``multi_board`` /
        # ``per_env_board`` are derived flags read by the no-vecenv guard.
        self.scheduler = None
        self.multi_board = False
        self.per_env_board = False
        self.current_board_idx = -1
        # Peak-VRAM budget models (--mem-budget): update = fwd+bwd, rollout =
        # no-grad forward. None = feature off (reactive OOM recovery only).
        self.mem_budget_update = None
        self.mem_budget_rollout = None
        self._mem_budget_calibrated = False
        # --update-gpus > 1: DDPUpdateGroup (spawned update workers + rank-0
        # ctx). None = single-GPU update (the unchanged default path).
        self.ddp = None
        # Validation-driven best-ckpt tracking, shared by PPO + GRPO (see
        # on_validation). Init to the worst value for the configured direction
        # (RLEvalConfig.best_metric_mode).
        from configs.loader.schema import DEFAULTS
        self.best_eval_fp = (
            float("-inf") if DEFAULTS.best_metric_mode == "max" else float("inf")
        )
        # --async-val: AsyncValResults (built in _setup_evaluator). None = inline.
        self.async_val = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        args = self.args
        self._setup_boards()
        self._setup_eval_boards()
        self.device = auto_device(args.device)
        self._print_banner()
        self.setup_algo_state()  # algo pre-flight checks + algo state

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        from configs.loader.schema import corner_deg_to_code
        self._corner_mode_int = corner_deg_to_code(args.corner_mode)

        self.envs = self._build_envs()
        self.policy = self._build_policy()
        # Inference facade (Agent layer): rollout collection and validation
        # act through this; gradient updates keep the raw model (self.policy).
        from methods.rl_agent.policy.agent import KiCadRLAgent
        self.agent = KiCadRLAgent(self.policy, device=self.device)
        self._record_env_contract()
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-4,
        )
        self._setup_mem_budget()
        self.lr_scheduler = self._build_lr_scheduler()
        self._resume()
        self._setup_ddp()  # after _resume: state broadcast covers ckpt resume
        self._setup_logger()
        self._setup_evaluator()

        # Optional iter-0 eval of the initial (zero-shot) policy: logged at step 0
        # for a baseline, but excluded from best-ckpt (a random "best" is useless).
        if self.start_iteration <= 1:  # not on --resume
            # Zero-shot snapshot — iter 0 is the one iteration save_periodic_ckpt
            # never covers. Always saved (async-val watchers and post-hoc analysis
            # can reference the initial policy without a special-case writer).
            self._save_ckpt(
                os.path.join(self.args.save_dir, "policy_iter_0.pt"),
                self._train_ckpt_payload(0, {}),
            )

        # Inline-only: async mode needs no iter-0 special case — the watcher
        # always evaluates policy_iter_0.pt (saved above for every run).
        if (self.args.eval_at_init and self.evaluator is not None
                and self.async_val is None and self.start_iteration <= 1):
            print("  [eval@init] evaluating initial policy at iteration 0 ...")
            self._run_dual_eval(0, {}, select_best=False)

    def _setup_boards(self) -> None:
        args = self.args
        # Shared curriculum/sampling engine (methods._shared.board_scheduler) — same
        # selection logic the LLM branch uses. It owns board discovery
        # (resolve_board_list) + the per_env RNG/epoch state; the trainer only
        # replicates indices by its GRPO group factor and logs.
        self.scheduler = BoardScheduler(BoardSchedulerConfig(
            mode=args.boards_order,
            single_board=args.board,
            boards_json=args.boards_json,
            difficulty=args.boards_difficulty,
            split=args.boards_split,
            seed=args.seed + 7919,
            dataset_dir=args.boards_dataset_dir,
        ))
        self.boards = self.scheduler.paths
        self.initial_board = self.boards[0]
        # Derived flags still read by _build_envs (no-vecenv guard).
        self.multi_board = args.boards_order == "round_robin" and len(self.boards) > 1
        self.per_env_board = (
            args.boards_order in ("per_env_random", "per_env_epoch")
            and len(self.boards) > 1
        )

    def _setup_eval_boards(self) -> None:
        args = self.args
        self.eval_boards = None
        if args.eval_split is None:
            if getattr(args, "async_val", False):
                raise RuntimeError("--async-val requires --eval-split")
            return
        if args.boards_order not in ("round_robin", "per_env_random", "per_env_epoch"):
            raise RuntimeError("--eval-split requires a multi-board --boards-order")
        eval_boards = load_eval_boards(args)
        if not getattr(args, "async_val", False):
            # Inline eval shares the trainer's env pool; the async watcher
            # sizes its own pool (--n-envs on the watcher).
            assert args.n_envs >= args.eval_n_rollouts, (
                f"n_envs ({args.n_envs}) must be >= eval_n_rollouts "
                f"({args.eval_n_rollouts}) for inline eval"
            )
        mode = "async" if getattr(args, "async_val", False) else "inline"
        print(
            f"  {mode} eval: {len(eval_boards)} boards × "
            f"{args.eval_n_rollouts} rollouts after every full "
            f"{len(self.boards)}-iter pass"
        )
        self.eval_boards = eval_boards

    # Kwargs the TRAINING pool adds on top of the shared RLEnvConfig surface.
    # Everything else comes from ``to_pool_kwargs()`` — the SAME dict val/eval
    # builds — so a knob added there reaches training with no edit here.
    #
    # Until 2026-08-20 this method hand-listed all 28 kwargs, and five knobs
    # (action_history_len / net_constraint_obs / outline_obs / simplify_outline
    # / keep_routing_fraction) were never added to that list: training silently
    # ran on the factory defaults while val used the CLI value (obs drift, 87
    # affected runs). Keep this tuple minimal — every entry is a train/val
    # difference that has to be declared at launch (--expect-env-diff).
    #: Knobs that exist only while training — passed to the factory as one
    #: explicit ``train_extras`` bundle. Eval omits the bundle entirely, so its
    #: absence means "no training layer" rather than a silent per-knob default
    #: (factory._TRAIN_EXTRAS). Every entry here is a train/val difference and
    #: must be declared at launch via --expect-env-diff.
    _TRAIN_EXTRAS_ARGS = (
        "reward_noise_std",
        "aug_bbox_shifted", "aug_flip", "aug_rotate", "aug_trans", "aug_zoom",
    )

    def _env_kwargs(self) -> dict:
        """Training env kwargs = shared RLEnvConfig surface + train-only bundle.

        ``seed`` / ``policy_net_select`` are NOT part of the bundle: eval passes
        both explicitly too (eval.rollout.rl), so they are ordinary required
        factory arguments whose *values* may differ.
        """
        from methods.rl_agent.models.loader import env_kwargs_from_training_args

        args = self.args
        kwargs = env_kwargs_from_training_args(args)
        kwargs["seed"] = args.seed
        kwargs["policy_net_select"] = args.policy_net_select
        kwargs["train_extras"] = {
            name: getattr(args, name) for name in self._TRAIN_EXTRAS_ARGS
        }
        # Training reloads a board into every worker each iteration, which
        # rebuilds the wrapper — keep each worker's random stream moving across
        # those rebuilds instead of rewinding it to its first draws. Eval pools
        # leave this off so a board's rollout stays independent of board order,
        # which is why it belongs in the train-only bundle rather than the
        # shared surface.
        kwargs["train_extras"]["advance_rng_on_reload"] = True
        return kwargs

    def _build_envs(self):
        args = self.args
        kwargs = self._env_kwargs()
        if args.no_vecenv:
            if self.multi_board or self.per_env_board:
                raise RuntimeError(
                    "--no-vecenv is incompatible with multi-board "
                    "--boards-order (requires the subprocess pool)"
                )
            from pcb_world.engine.kicad_engine import allow_router_coexistence

            with allow_router_coexistence(
                "--no-vecenv: n_envs in-process envs by design (stepped "
                "sequentially by the collector; no subprocess pool)"
            ):
                # The pool derives per-worker seeds as ``seed + i`` internally
                # (factory.make_decoder_env_pool); do the same by hand here.
                base_seed = kwargs.pop("seed")
                return [
                    make_decoder_env(self.initial_board, seed=base_seed + i, **kwargs)
                    for i in range(args.n_envs)
                ]
        return make_decoder_env_pool(self.initial_board, args.n_envs, **kwargs)

    def _record_env_contract(self) -> None:
        """Dump what train/val envs were ACTUALLY built with, then gate the diff.

        ``config_resolved.yaml`` records the parsed CLI namespace — the run's
        *intent*. This records the *effect*: the kwargs the factory resolved for
        training (snapshotted inside it, post-default) and the kwargs validation
        will resolve for itself. The 2026-08-20 five-knob drift was invisible
        precisely because only intent was ever stored.

        The val side is computed, not guessed: ``resolve_eval_env_kwargs`` is the
        same function ``eval_transformer`` calls, so no rollout is needed to know
        what validation will build.
        """
        from dataclasses import asdict

        from configs.loader.schema import RLPolicyConfig
        from eval.rollout.rl import resolve_eval_env_kwargs
        from methods._shared.config_dump import (
            check_expected_env_diff,
            dump_env_records,
        )
        from methods.rl_agent.models.loader import env_kwargs_from_training_args

        args = self.args
        train_env = getattr(self.envs, "effective_env_kwargs", None)
        if train_env is None:            # --no-vecenv: a plain list of wrappers
            train_env = self._env_kwargs()
        val_env = resolve_eval_env_kwargs(
            env_kwargs_from_training_args(args), self.agent, step_drc=True,
        )
        # eval passes the seed as its own argument, not through the dict.
        val_env["seed"] = args.eval_base_seed
        self._recorded_val_env = dict(val_env)
        # An all-default bundle IS "no training layer", which is exactly what
        # val has — recording it as present would put train_extras in every
        # run's diff and force a declaration that says nothing. "All-default"
        # is judged from the TRAINING side: _env_kwargs always forces the
        # pool-level advance_rng_on_reload on, so comparing against the bare
        # factory bundle (advance=False) can never match and every run would
        # be back to declaring train_extras (the pre-fix state).
        train_env = dict(train_env)
        from methods.rl_agent.wrappers.factory import _TRAIN_EXTRAS
        if train_env.get("train_extras") == {
            **_TRAIN_EXTRAS, "advance_rng_on_reload": True,
        }:
            train_env.pop("train_extras")
        out, diff = dump_env_records(
            args.save_dir,
            train_env=train_env, val_env=val_env,
            policy=asdict(RLPolicyConfig.from_namespace(
                args, use_critic=self.USE_CRITIC,
            )),
        )
        print(f"[env-contract] records -> {out}")
        if diff:
            print(f"[env-contract] train/val diff: {sorted(diff)}")
        check_expected_env_diff(diff, getattr(args, "expect_env_diff", ""))

    def _build_policy(self):
        from configs.loader.schema import RLPolicyConfig

        # slot_perm augmentation samples a fixed permutation over
        # [0, N_MAX_SLOTS=64) in the wrapper, so it is incompatible with a
        # policy whose slot table was enlarged (e.g. d3b's --n-max-slots 128).
        if getattr(self.args, "slot_perm", False) and \
                getattr(self.args, "n_max_slots", 64) != 64:
            raise RuntimeError(
                "--slot-perm assumes n_max_slots=64 (fixed permutation) — "
                f"cannot be used with --n-max-slots {self.args.n_max_slots}"
            )

        policy = RLPolicyConfig.from_namespace(
            self.args, use_critic=self.USE_CRITIC,
        ).build(self.device)
        # Speed knobs (--bf16 / --compile-*): all off by default. getattr
        # guard — entrypoints without these flags (e.g. GRPO) stay fp32/eager.
        bf16 = bool(getattr(self.args, "bf16", False))
        regions = tuple(
            r for r in getattr(self.args, "compile_regions", "").split(",") if r
        )
        attn = getattr(self.args, "attn", "sdpa")
        if bf16 or regions or attn != "sdpa":
            policy.configure_speed(
                bf16=bf16, compile_regions=regions,
                compile_mode=getattr(self.args, "compile_mode", "default"),
                attn=attn,
            )
            print(f"  speed knobs: bf16={bf16} compile_regions={regions} "
                  f"mode={getattr(self.args, 'compile_mode', 'default')} "
                  f"attn={attn}")
        n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        if self.USE_CRITIC:
            n_critic = sum(p.numel() for p in policy.critic_head.parameters())
            print(f"  parameters: {n_params:,} (critic_head: {n_critic:,})")
        else:
            print(f"  parameters: {n_params:,}")
        print()
        return policy

    def _build_lr_scheduler(self):
        args = self.args
        if args.warmup_iters > 0:
            def lr_lambda(iteration: int) -> float:
                if iteration < args.warmup_iters:
                    return iteration / args.warmup_iters
                return 1.0
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return None

    def _resume(self) -> None:
        args = self.args
        self.start_iteration = 1
        if not args.resume:
            return
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=self.device)
        # obstacle_obs adds no weights, so a flag flip across resume would be
        # SILENT obs drift (tokens appear/vanish mid-run) — cross-check the
        # saved args loudly. (shape_obs needs no guard: its shape_embed key
        # makes a flip fail the strict-ish load below either way.)
        ckpt_obst = bool((ckpt.get("args") or {}).get("obstacle_obs", False))
        run_obst = bool(getattr(self.args, "obstacle_obs", False))
        if ckpt_obst != run_obst:
            raise RuntimeError(
                f"obstacle_obs mismatch on resume: checkpoint trained with "
                f"{ckpt_obst}, this run has {run_obst} — OBSTACLE tokens "
                "would silently appear/vanish mid-run. Pass the matching "
                "flag (or start fresh)."
            )
        # Pre-OBSTACLE checkpoints carry a 14-row entity-type table — pad the
        # policy weights AND the Adam moments (torch validates the latter
        # only at the first optimizer.step()). Loud on any other mismatch.
        from methods.rl_agent.models.loader import (
            pad_legacy_entity_type_rows,
            pad_legacy_optimizer_state,
        )
        pad_legacy_entity_type_rows(ckpt["policy_state_dict"], self.policy)
        if "optimizer_state_dict" in ckpt:
            pad_legacy_optimizer_state(ckpt["optimizer_state_dict"], self.policy)
        missing, unexpected = self.policy.load_state_dict(
            ckpt["policy_state_dict"], strict=False,
        )
        hard_missing = [
            k for k in missing
            if not any(tag in k.lower() for tag in self.ALLOWED_MISSING_KEYS)
        ]
        if hard_missing or unexpected:
            raise RuntimeError(
                f"Resume load mismatch: missing={hard_missing}, unexpected={unexpected}",
            )
        if missing:
            print(f"  [resume] missing keys (newly initialized): {missing}")
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                print(f"  [resume] optimizer load failed ({e}); using fresh optimizer")
        normalizer = getattr(self, "reward_normalizer", None)
        if normalizer is not None:
            if "reward_normalizer_state" in ckpt:
                normalizer.load_state_dict(ckpt["reward_normalizer_state"])
                print(f"  [resume] reward normalizer restored (std={normalizer.std:.3f})")
            else:
                print(
                    "  [resume] ckpt has no reward_normalizer_state; "
                    "starting fresh (reward scale drifts until re-converged)"
                )
        self.start_iteration = ckpt.get("iteration", 0) + 1
        counters = ckpt.get("counters") or {}
        self._episodes_total = int(counters.get("episodes_total", 0))
        self._env_steps_total = int(counters.get("env_steps_total", 0))
        print(f"  Loaded iteration {self.start_iteration - 1}, resuming from {self.start_iteration}")

    def _setup_logger(self) -> None:
        args = self.args
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        config_extras = {
            key: value
            for key, value in {
                "cadagent_run_backend": os.environ.get("CADAGENT_RUN_BACKEND"),
                "cadagent_slurm_qos": os.environ.get("CADAGENT_SLURM_QOS"),
                "cadagent_target_iterations": os.environ.get("CADAGENT_TARGET_ITERATIONS"),
                "cadagent_stage": os.environ.get("CADAGENT_STAGE"),
            }.items()
            if value is not None
        }
        self.writer = build_logger(args.log_dir, args=args, config_extras=config_extras)

    def _setup_evaluator(self) -> None:
        args = self.args
        if self.eval_boards is None:
            self.evaluator = None
            return
        # Shared with the --async-val watcher (module-level build_evaluators);
        # in async mode the trainer never calls these, but building them still
        # fail-fasts on bad eval2..eval5 board lists at startup.
        self.evaluator, self.extra_evaluators = build_evaluators(
            args, self.agent, self.device, self.eval_boards,
            mem_budget=self.mem_budget_rollout,
            expect_env_kwargs=getattr(self, "_recorded_val_env", None),
        )
        self.eval_cadence = (
            args.eval_every if args.eval_every is not None else len(self.boards)
        )
        if getattr(args, "async_val", False):
            # The watcher evaluates the REGULAR periodic ckpts in place (no
            # separate eval snapshot), so every cadence iter must have one —
            # and the watcher reads the cadence from the ckpt args.
            if args.eval_every is None:
                raise RuntimeError("--async-val requires an explicit --eval-every")
            if args.eval_every % args.save_freq != 0:
                raise RuntimeError(
                    f"--async-val needs --save-freq ({args.save_freq}) to divide "
                    f"--eval-every ({args.eval_every}) so a policy_iter ckpt "
                    f"exists at every eval cadence"
                )
            from methods.rl_agent.training.async_val import AsyncValResults
            self.async_val = AsyncValResults(args.save_dir)
            print(f"  async val: watcher evaluates policy_iter_*.pt every "
                  f"{args.eval_every} iters; results -> {self.async_val.results_dir} "
                  f"(start a watcher — see methods/rl_agent/training/async_val.py)")

    # ------------------------------------------------------------------
    # multi-GPU update (--update-gpus)
    # ------------------------------------------------------------------

    def _setup_ddp(self) -> None:
        """``--update-gpus N`` (N>1): spawn N-1 update workers on ``cuda:1..``
        and broadcast the (possibly resumed) policy/optimizer state.

        Rollout/eval/logging stay on the main process; only the PPO update is
        rank-sharded (manual grad allreduce — see
        :mod:`methods.rl_agent.training.ddp`). The ``--mem-budget`` planner is
        not supported here (no per-rank calibration); shard VRAM is already
        1/N and the reactive OOM peel stays rank-local.
        """
        if int(getattr(self.args, "update_gpus", 1)) <= 1:
            return
        if self.mem_budget_update is not None:
            raise RuntimeError(
                "--update-gpus > 1 does not support --mem-budget (phase 1); "
                "drop one of the two flags"
            )
        from methods.rl_agent.training.ddp import DDPUpdateGroup
        self.ddp = DDPUpdateGroup(
            self.args, self.policy, self.optimizer, self.device,
            use_critic=self.USE_CRITIC,
        )

    # ------------------------------------------------------------------
    # peak-VRAM budget (--mem-budget)
    # ------------------------------------------------------------------

    def _setup_mem_budget(self) -> None:
        if not getattr(self.args, "mem_budget", False):
            return
        if self.device.type != "cuda":
            print("  mem_budget: non-CUDA device — disabled")
            return
        from methods.rl_agent.training.mem_budget import MemBudgetModel
        param_bytes = float(sum(
            p.numel() * p.element_size() for p in self.policy.parameters()
        ))
        # AdamW state (exp_avg + exp_avg_sq = 2x params) appears lazily on the
        # first optimizer.step(); until then reserve it out of the capacity.
        self.mem_budget_update = MemBudgetModel(
            reserve_fn=lambda: 0.0 if self.optimizer.state else 2.0 * param_bytes,
        )
        self.mem_budget_rollout = MemBudgetModel()

    def _calibrate_mem_budget(self, buffer: dict) -> None:
        """One-time probe calibration on the first real buffer (a few fwd/bwd).

        Probes replicate one buffer sample ``B`` times through the real update
        (fwd+bwd) and rollout (no-grad) forwards and fit each model from the
        measured allocated peaks (``mem_budget.run_calibration``). A failed
        fit leaves that model not-ready: its planner stays off and the
        reactive OOM peel keeps covering, exactly as with --mem-budget off.
        """
        obs_list = buffer["obs_list"]
        if not obs_list:
            return   # empty first buffer — retry next iteration
        self._mem_budget_calibrated = True
        from methods.rl_agent.training import mem_budget as mem_budget_mod

        seq_lens = self.policy.tokenizer._walk_obs(obs_list)["seq_lens"]
        actions_np = buffer["actions"]
        masks_np = buffer["action_masks"]
        ptr_np = buffer.get("pointer_masks")
        device = self.device
        policy = self.policy

        def _tensors(pos: int, B: int):
            obs = [obs_list[pos]] * B
            act_t = torch.as_tensor(
                actions_np[pos], dtype=torch.long, device=device,
            ).unsqueeze(0).expand(B, -1).contiguous()
            am_t = torch.as_tensor(
                masks_np[pos], dtype=torch.bool, device=device,
            ).unsqueeze(0).expand(B, -1).contiguous()
            if ptr_np is not None and ptr_np.ndim == 2 and ptr_np.shape[1]:
                pm_t = torch.as_tensor(
                    ptr_np[pos], dtype=torch.long, device=device,
                ).unsqueeze(0).expand(B, -1).contiguous()
            else:
                pm_t = torch.full((B, 0), -1, dtype=torch.long, device=device)
            return obs, act_t, am_t, pm_t

        def probe_update(pos: int, B: int) -> float:
            obs, act_t, am_t, pm_t = _tensors(pos, B)
            base = mem_budget_mod.begin_measured_region()
            new_lp, entropy, values = policy.evaluate_actions_and_value(
                obs, act_t, action_masks=am_t, pointer_masks=pm_t,
            )
            # The loss *shape* is irrelevant for the peak (the real losses only
            # add (B,)-sized terms); what matters is backward through the
            # transformer stack.
            (new_lp.sum() + entropy.sum() + values.sum()).backward()
            peak = mem_budget_mod.end_measured_region(base)
            self.optimizer.zero_grad(set_to_none=True)
            return peak

        def probe_rollout(pos: int, B: int) -> float:
            obs, _act_t, am_t, pm_t = _tensors(pos, B)
            base = mem_budget_mod.begin_measured_region()
            policy.act_and_value(obs, action_masks=am_t, pointer_masks=pm_t)
            return mem_budget_mod.end_measured_region(base)

        # RNG isolation: the rollout probe SAMPLES (act_and_value) and would
        # advance the CUDA RNG stream — --mem-budget must not change the run's
        # action trajectory vs off (A/B alignment requirement).
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self._run_calibration_probes(
                probe_update, probe_rollout, seq_lens, mem_budget_mod,
            )

    def _run_calibration_probes(
        self, probe_update, probe_rollout, seq_lens, mem_budget_mod,
    ) -> None:
        # Warm-up outside measurement (cudnn/cublas workspaces, .grad alloc).
        lo = min(range(len(seq_lens)), key=seq_lens.__getitem__)
        try:
            probe_update(lo, 2)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("  mem_budget: warm-up probe OOMed — planners disabled")
            return
        for model, probe_fn, label in (
            (self.mem_budget_update, probe_update, "update"),
            (self.mem_budget_rollout, probe_rollout, "rollout"),
        ):
            if mem_budget_mod.run_calibration(
                model, probe_fn, seq_lens, label=f"mem_budget[{label}]",
            ):
                c, a, b = model.coeffs
                print(
                    f"  mem_budget[{label}] fit: c={c / 2**20:.1f}MB "
                    f"a={a:.1f}B/tok b={b:.4f}B/tok² "
                    f"capacity={model.capacity() / 2**20:.0f}MB"
                )

    # ------------------------------------------------------------------
    # board scheduling (shared)
    # ------------------------------------------------------------------

    def _board_pick(self) -> tuple[int, int]:
        """(n base picks, replicate factor). PPO: (n_envs, 1)."""
        return self.args.n_envs, 1

    def select_boards(self, iteration: int) -> None:
        sched = self.scheduler
        if sched.mode == "single" or len(self.boards) <= 1:
            return  # built once on the single board; nothing to reload

        # ``_board_pick`` returns (#unique boards this iter, GRPO replicate).
        # round_robin → one board for all; per_env_* → one per group. The
        # scheduler owns the index selection; the trainer replicates by group
        # and maps to paths.
        n_pick, replicate = self._board_pick()
        base = sched.next_indices(n_pick)
        picks = list(np.repeat(np.asarray(base), replicate)) if replicate > 1 else list(base)
        self.envs.reload_boards([self.boards[int(p)] for p in picks])
        self.current_board_idx = int(base[0])

        # Pad counts are lazy: per_env_* resolve skips the full-pool pre-scan
        # (minutes of NFS reads on 100k boards), so count only the boards this
        # iter actually uses — ``count_pads`` is lru-cached, first touch only.
        from methods._shared.board_loader import count_pads

        w = self.writer
        if sched.mode == "round_robin":
            w.add_scalar("train/board_idx", self.current_board_idx, iteration)
            w.add_scalar(
                "train/board_pad_count",
                count_pads(self.boards[self.current_board_idx]), iteration,
            )
        else:  # per_env_random / per_env_epoch
            if sched.mode == "per_env_epoch":
                w.add_scalar("train/epoch", sched.epochs_completed, iteration)
                w.add_scalar(
                    "train/epoch_queue_remaining", sched.epoch_remaining, iteration,
                )
            mean_pad = float(np.mean([count_pads(self.boards[int(p)]) for p in base]))
            w.add_scalar("train/board_pad_mean", mean_pad, iteration)
            w.add_scalar(
                "train/board_unique", len(set(int(p) for p in base)), iteration,
            )

    # ------------------------------------------------------------------
    # per-iteration skeleton
    # ------------------------------------------------------------------

    def train_iteration(self, iteration: int) -> dict:
        t0 = time.time()
        self.select_boards(iteration)
        coll = self.collect_rollout()
        buffer = self.compute_targets(coll)
        if self.mem_budget_update is not None and not self._mem_budget_calibrated:
            self._calibrate_mem_budget(buffer)
        if self.ddp is not None:
            # Same buffer + kwargs to every rank (workers set the pushed LR),
            # then rank 0 runs its own shard through the identical loop.
            update_kwargs = self.update_kwargs()
            bcast_bytes, bcast_s = self.ddp.dispatch_update(
                buffer, update_kwargs, self.optimizer.param_groups[0]["lr"],
            )
            metrics = policy_update_loop(
                self.policy, self.optimizer, buffer, self.device,
                ddp=self.ddp.ctx, **update_kwargs,
            )
        else:
            metrics = policy_update_loop(
                self.policy, self.optimizer, buffer, self.device, **self.update_kwargs(),
            )
        self.aggregate_metrics(metrics, coll, buffer)
        elapsed = time.time() - t0
        self._log_common(metrics, buffer, iteration, elapsed)
        if self.ddp is not None:
            # Buffer-transfer cost was un-measured pre-integration — keep the
            # per-iter bytes/seconds observable (phase-2 /dev/shm call basis).
            mb = bcast_bytes / (1024.0 * 1024.0)
            self.writer.add_scalar("diag/ddp_buffer_bcast_mb", mb, iteration)
            self.writer.add_scalar("diag/ddp_buffer_bcast_s", bcast_s, iteration)
            if iteration == self.start_iteration:
                print(f"  ddp-update: buffer broadcast {mb:.1f}MB in {bcast_s:.3f}s "
                      f"(world={self.ddp.world_size})")
        self.log_algo_metrics(metrics, iteration)
        if iteration % self.args.log_every == 0 or iteration == 1:
            self._print_progress(iteration, metrics, elapsed)
        return metrics

    def _log_common(self, metrics, buffer, iteration, elapsed) -> None:
        w = self.writer
        w.add_scalar("train/loss", metrics["loss"], iteration)
        w.add_scalar("train/policy_loss", metrics["policy_loss"], iteration)
        w.add_scalar("train/entropy", metrics["entropy"], iteration)

        act_types = buffer["actions"][:, 0]
        total_acts = max(len(act_types), 1)
        for ai, aname in enumerate(_ACTION_NAMES):
            ratio = float((act_types == ai).sum()) / total_acts
            w.add_scalar(f"actions/{aname}", ratio, iteration)

        for key in (
            "mean_reward", "std_reward", "mean_ep_length", "n_episodes",
            "drc_violations_mean", "drc_violations_std",
            "final_potential_mean", "final_potential_std", "final_potential_max",
            "final_unrouted_mean", "final_unrouted_std",
            "final_wirelength_mean", "final_wirelength_std",
            "final_via_count_mean", "final_via_count_std",
            "final_track_count_mean", "final_track_count_std",
            "terminated_rate", "ratsnest_reduction_mean", "ratsnest_reduction_std",
        ):
            w.add_scalar(f"rollout/{key}", metrics[key], iteration)

        # Engine-latency telemetry (set by the PPO collect path only — `in` so
        # paths without step_times, e.g. GRPO group-collect, skip cleanly).
        for key in ("step_time_mean", "step_time_p95", "step_time_max",
                    "step_time_le_10ms", "step_time_10ms_100ms",
                    "step_time_100ms_1s", "step_time_1s_10s",
                    "step_time_10s_100s", "step_time_ge_100s"):
            if key in metrics:
                w.add_scalar(f"rollout/{key}", metrics[key], iteration)
        _st_log10 = metrics.pop("_step_time_log10", None)
        if _st_log10 is not None and hasattr(w, "add_histogram"):
            w.add_histogram("rollout/step_time_log10", _st_log10, iteration)

        # Longest tokenized sequence in this iter's buffer — the padding target
        # that drives the update-time attention memory spike. Connects a spike
        # iter to its outlier (runaway) episode (pairs with diag/oom_minibatch_rate).
        # The collect-time walk (walk_flat) already carries seq_lens; only the
        # uncached path (GRPO) pays a fresh CPU walk here.
        walk_flat = buffer.get("walk_flat")
        obs_list = buffer["obs_list"]
        if walk_flat is not None:
            seq_lens = walk_flat["seq_lens"]
        elif obs_list:
            seq_lens = self.policy.tokenizer._walk_obs(obs_list)["seq_lens"]
        else:
            seq_lens = []
        if seq_lens:
            w.add_scalar("rollout/max_seq_len", max(seq_lens), iteration)

        # Cumulative counters — alternative x-axes (wandb: set as step metric).
        self._episodes_total = getattr(self, "_episodes_total", 0) + int(metrics["n_episodes"])
        self._env_steps_total = getattr(self, "_env_steps_total", 0) + int(len(act_types))
        w.add_scalar("diag/episodes_total", self._episodes_total, iteration)
        w.add_scalar("diag/env_steps_total", self._env_steps_total, iteration)
        # Cumulative worker respawns (subproc backend only; all crash causes,
        # incl. non-step-path deaths) — slope = engine crash rate.
        respawn_total = getattr(self.envs, "respawn_total", None)
        if respawn_total is not None:
            w.add_scalar("diag/worker_respawn_total", respawn_total, iteration)
        # OOM auto-recovery health: fraction of minibatches that hit >=1 CUDA OOM
        # this update (=> gradient-accumulation chunking kicked in). ~0 normally;
        # trending toward 1.0 = boards outgrowing VRAM even at nominal batch.
        w.add_scalar(
            "diag/oom_minibatch_rate", metrics.get("oom_minibatch_rate", 0.0), iteration,
        )
        w.add_scalar("diag/oom_events", metrics.get("oom_events", 0.0), iteration)
        # Preemptive chunking health (--mem-budget): mean planned chunks per
        # minibatch (1.0 = everything fit whole) + the live capacity target.
        w.add_scalar(
            "diag/planned_chunks_per_mb",
            metrics.get("planned_chunks_per_mb", 1.0), iteration,
        )
        if self.mem_budget_update is not None and self.mem_budget_update.ready:
            w.add_scalar(
                "diag/mem_budget_capacity_mb",
                self.mem_budget_update.capacity() / (1024.0 * 1024.0), iteration,
            )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            w.add_scalar(
                "train/learning_rate", self.optimizer.param_groups[0]["lr"], iteration,
            )
        w.add_scalar("train/time_per_iter", elapsed, iteration)

        if self.device.type == "cuda":
            from methods.rl_agent.training.mem_budget import (
                iteration_peaks_and_reset,
            )
            mb = 1024.0 * 1024.0
            w.add_scalar("diag/gpu_mem_reserved_mb", torch.cuda.memory_reserved() / mb, iteration)
            # Peak counters read via mem_budget's fold-aware accessor: the
            # per-chunk peak measurements reset the raw CUDA counters mid-
            # iteration, and this preserves the true iteration high-water mark
            # (identical to the raw counters when no measured region ran).
            peak_alloc, peak_reserved = iteration_peaks_and_reset()
            w.add_scalar(
                "diag/gpu_mem_peak_reserved_mb", peak_reserved / mb, iteration,
            )
            # Peak *allocated* (live tensors) — separates the true working-set
            # high-water mark from the reserved-pool afterglow that lingers in
            # gpu_mem_peak_reserved_mb. This is the attention-spike signal.
            w.add_scalar(
                "diag/gpu_mem_peak_allocated_mb", peak_alloc / mb, iteration,
            )

    # ------------------------------------------------------------------
    # post-iteration: ckpt + validation
    # ------------------------------------------------------------------

    def on_iteration_end(self, iteration: int, metrics: dict) -> None:
        self.save_last_ckpt(iteration, metrics)
        self.save_periodic_ckpt(iteration, metrics)
        self.maybe_best_ckpt(iteration, metrics)
        if self.evaluator is not None and iteration % self.eval_cadence == 0:
            if self.async_val is None:
                self._run_dual_eval(iteration, metrics, select_best=True)
            # async: nothing to write — save_periodic_ckpt already produced
            # policy_iter_<N>.pt (cadence alignment asserted in _setup_evaluator);
            # the watcher discovers it from the filesystem.
        if self.async_val is not None:
            self._consume_async_results()

    def _train_ckpt_payload(self, iteration: int, metrics: dict) -> dict:
        return {
            "iteration": iteration,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(self.args),
        }

    def save_last_ckpt(self, iteration: int, metrics: dict) -> None:
        """Overwrite ``policy_last.pt`` every iteration — the cheapest resume
        point after a crash (periodic ckpts only land every --save-freq)."""
        self._save_ckpt(
            os.path.join(self.args.save_dir, "policy_last.pt"),
            self._train_ckpt_payload(iteration, metrics),
        )

    def save_periodic_ckpt(self, iteration: int, metrics: dict) -> None:
        if iteration % self.args.save_freq != 0:
            return
        self._save_ckpt(
            os.path.join(self.args.save_dir, f"policy_iter_{iteration}.pt"),
            self._train_ckpt_payload(iteration, metrics),
        )

    def _ckpt_stamp_payload(self) -> dict:
        """Provenance + obs-semantics probe, computed once per run.

        ``provenance`` = repo version + git commit (best-effort — "unknown"
        outside a git checkout; the values are informational, never enforced).
        ``obs_probe`` = a fixed probe obs + the digest of THIS code's tokenizer
        walk over it; loaders re-encode the stored obs with their code and
        hard-error on digest mismatch (obs-semantics drift — see
        methods/rl_agent/models/v1/obs_probe.py).
        """
        if self._ckpt_stamp is None:
            from methods._shared.config_dump import collect_provenance
            from methods.rl_agent.models.v1.obs_probe import (
                build_probe_obs, probe_digest,
            )
            probe = build_probe_obs()
            self._ckpt_stamp = {
                "provenance": collect_provenance(),
                "obs_probe": {
                    "obs": probe,
                    "digest": probe_digest(self.policy.tokenizer, probe),
                },
            }
        return self._ckpt_stamp

    def _save_ckpt(self, path: str, payload: dict) -> None:
        # Reward-norm running stats (PPO only): the critic is trained on
        # rewards ÷ this std, so consumers (e.g. MCTS critic_bootstrap) need
        # it to map V back into raw Φ units.
        normalizer = getattr(self, "reward_normalizer", None)
        if normalizer is not None:
            payload["reward_normalizer_state"] = normalizer.state_dict()
        # Cumulative x-axis counters survive resume (see _log_common).
        payload["counters"] = {
            "episodes_total": getattr(self, "_episodes_total", 0),
            "env_steps_total": getattr(self, "_env_steps_total", 0),
        }
        payload.update(self._ckpt_stamp_payload())
        # Atomic write: a crash mid-save must never corrupt an existing ckpt
        # (policy_last.pt is overwritten every iteration).
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def teardown(self) -> None:
        if self.policy is None:
            return  # setup() failed before building anything
        if self.async_val is not None:
            self._finish_async_val()  # before writer.close(): results log here
        if self.ddp is not None:
            self.ddp.shutdown()
            self.ddp = None
        final_path = os.path.join(self.args.save_dir, "policy_final.pt")
        self._save_ckpt(final_path, {
            "iteration": self.args.iterations,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args": vars(self.args),
        })
        if self.writer is not None:
            self.writer.close()
        if hasattr(self.envs, "close"):
            self.envs.close()
        else:
            for env in self.envs:
                env.close()
        self._print_done()

    # ------------------------------------------------------------------
    # algorithm hooks (abstract / overridable)
    # ------------------------------------------------------------------

    @abstractmethod
    def collect_rollout(self): ...

    @abstractmethod
    def compute_targets(self, coll) -> dict: ...

    @abstractmethod
    def update_kwargs(self) -> dict: ...

    @abstractmethod
    def aggregate_metrics(self, metrics: dict, coll, buffer) -> None: ...

    @abstractmethod
    def log_algo_metrics(self, metrics: dict, iteration: int) -> None: ...

    def setup_algo_state(self) -> None:
        """Algo pre-flight checks + per-algo state init (default: none)."""

    def maybe_best_ckpt(self, iteration: int, metrics: dict) -> None:
        """Every-iteration (train-metric) best-ckpt hook; default no-op.

        Best-ckpt selection is validation-driven for both PPO and GRPO (see
        :meth:`on_validation`); neither algorithm overrides this hook."""

    def _run_dual_eval(self, iteration: int, metrics: dict, *, select_best: bool) -> None:
        """Run the primary eval (under ``val/*``; drives best-ckpt when
        ``select_best``) then the optional diagnostic sets (eval2..eval5,
        under their ``<prefix>/*``).

        Shared by the per-cadence path (``select_best=True``) and the one-shot
        iter-0 init eval (``select_best=False``). Diagnostic sets reuse the same
        ``validate`` machinery by temporarily swapping ``self.evaluator``; only
        the primary result is ever passed to :meth:`on_validation`.
        """
        eval_metrics = self.validate(iteration)  # prefix="val" (primary; unchanged)
        if select_best:
            self.on_validation(iteration, metrics, eval_metrics)
        primary = self.evaluator
        for pfx, ev in self.extra_evaluators:
            self.evaluator = ev
            try:
                self.validate(iteration, prefix=pfx)
            finally:
                self.evaluator = primary
        # Eval leaves a large residual GPU allocation (esp. on big real boards);
        # release it before training resumes so the next update doesn't OOM.
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # --async-val: result consumption (see async_val.py; no extra writes —
    # the watcher evaluates the regular policy_iter ckpts in place)
    # ------------------------------------------------------------------

    def _consume_async_results(self) -> None:
        """Log (and best-select on) any watcher results that have arrived.

        Scalars are logged at their TRUE iteration: TensorBoard takes the
        out-of-order step directly; W&B goes through add_scalars_async.
        """
        n_left = self.async_val.n_pending(self.eval_cadence)
        if n_left >= 3:
            # Positive sink verification: ckpts are queueing up with no
            # results — the watcher is probably dead/never started.
            print(f"[async-val] WARNING: {n_left} checkpoint(s) awaiting "
                  f"validation with no results arriving — is the watcher "
                  f"alive? ({self.async_val.dir})", flush=True)
        for result in self.async_val.poll_results():
            n = int(result["iteration"])
            self.writer.add_scalars_async(result["scalars"], n)
            if n > 0:  # iter-0 init eval: logged but excluded from best-ckpt
                ckpt = self.async_val.ckpt_path(n)
                self._handle_val_overall(
                    n, result["overall"],
                    lambda ckpt=ckpt: torch.load(ckpt, map_location="cpu")["policy_state_dict"],
                )
            self.async_val.consume(n)

    def _finish_async_val(self) -> None:
        """Teardown half of the --async-val contract (unit-tested standalone).

        Completed run: mark ``train_done`` and drain the pending validations.
        Aborted run (``fit_completed`` False — exception, signal): do NEITHER.
        The watcher then keeps polling this save-dir, so a relaunch into the
        same directory is picked up by the same watcher, and teardown does not
        sit in the 4h drain wait. (260825: a crashed trainer marked done, the
        watcher exited, and the relaunched cell ran with no watcher at all.)
        """
        if getattr(self, "fit_completed", False):
            self.async_val.mark_train_done()
            self._drain_async_results()
        else:
            print("[async-val] trainer ABORTED before completion — NOT marking "
                  "train_done; the watcher keeps polling this save-dir "
                  "(relaunch resumes evaluation)", flush=True)

    def _drain_async_results(self) -> None:
        """Teardown: wait for the watcher to finish the still-pending vals."""
        from methods.rl_agent.training.async_val import DRAIN_TIMEOUT_S
        t0 = time.time()
        while True:
            self._consume_async_results()
            n_left = self.async_val.n_pending(self.eval_cadence)
            if n_left == 0:
                return
            if time.time() - t0 > DRAIN_TIMEOUT_S:
                print(f"[async-val] drain timeout — {n_left} validation(s) still "
                      f"pending in {self.async_val.dir}; restart a watcher and "
                      f"re-run consume later (results/json stay on disk)")
                return
            print(f"[async-val] waiting for {n_left} pending validation(s) ...")
            time.sleep(30)

    def on_validation(self, iteration: int, metrics: dict, eval_metrics) -> None:
        """Shared eval-driven best checkpoint (PPO + GRPO).

        Saves ``policy_best.pt`` whenever the canonical val metric improves. The
        metric + direction are the single shared criterion in
        :class:`configs.loader.schema.RLEvalConfig` (``best_metric_key`` /
        ``best_metric_mode`` — default ``val/fp_mean_of_means`` max = "the
        largest fp gain is best"), the SAME criterion the LLM trainer applies via
        ``trainer.best_metric_key`` (run_cadagent*.sh), so RL and LLM select best
        checkpoints by an identical rule.

        Validation-driven by construction: a run without inline eval boards never
        calls this, so no ``policy_best.pt`` is written (GRPO does not fall back
        to a train-reward best).
        """
        if eval_metrics is None:
            return
        self._handle_val_overall(
            iteration, eval_metrics.overall, self.policy.state_dict,
        )

    def _handle_val_overall(self, iteration: int, ov: dict, state_fn) -> None:
        """Print + best-ckpt selection on a primary val ``overall`` dict.

        Shared by the inline path (:meth:`on_validation`, ``state_fn`` =
        live ``policy.state_dict``) and the async path
        (:meth:`_consume_async_results`, ``state_fn`` loads the EVALUATED
        checkpoint from the queue — the live policy is iterations ahead).
        ``state_fn`` is lazy so the async path only touches disk on improvement.
        """
        # None-tolerant formatting: an ``overall`` entry can be None (e.g. a
        # rollout that crashed over a whole board — measured 260813 A10
        # iter-10: routability_mean=None, and the format() TypeError killed a
        # 300-iter trainer outright). The summary print shows NA and carries
        # on; best-ckpt selection is filtered by the None/finite guard below,
        # as before.
        def _f(key: str, spec: str) -> str:
            v = ov.get(key)
            return format(v, spec) if isinstance(v, (int, float)) else "NA"

        print(
            f"[eval @ iter {iteration}] "
            f"fp_mean_of_means={_f('fp_mean_of_means', '+.3f')}  "
            f"fp_mean_of_maxes={_f('fp_mean_of_maxes', '+.3f')}  "
            f"rout={_f('routability_mean', '.3f')}  "
            f"wire={_f('wirelength_mean', '.1f')}mm  "
            f"vias={_f('via_count_mean', '.2f')}"
        )
        from configs.loader.schema import DEFAULTS

        # best_metric_key is a logged val/* tag; the overall dict is keyed
        # without the prefix, so strip it to index `ov`.
        key = DEFAULTS.best_metric_key.removeprefix("val/")
        score = ov.get(key)
        if score is None or not np.isfinite(score):
            return
        improved = (
            score > self.best_eval_fp if DEFAULTS.best_metric_mode == "max"
            else score < self.best_eval_fp
        )
        if improved:
            self.best_eval_fp = score
            self._save_ckpt(
                os.path.join(self.args.save_dir, "policy_best.pt"),
                {
                    "iteration": iteration,
                    "policy_state_dict": state_fn(),
                    "eval_overall": ov,
                    "args": vars(self.args),
                },
            )

    def _print_banner(self) -> None: ...

    def _print_progress(self, iteration: int, metrics: dict, elapsed: float) -> None: ...

    def _print_done(self) -> None:
        print(f"\nTraining complete. Checkpoints: {self.args.save_dir}")
        print(f"TensorBoard: tensorboard --logdir {self.args.log_dir}")


class PPOTrainer(RLTrainer):
    """SB3-style clipped-objective PPO with a critic head."""

    ALGO = "ppo"
    USE_CRITIC = True
    ALLOWED_MISSING_KEYS = {"prev_action", "history_age", "drc"}

    def __init__(self, args) -> None:
        super().__init__(args)
        self.reward_normalizer = None
        # best_eval_fp is initialized in the base RLTrainer (shared best-ckpt).

    # --- pre-flight + algo state ---
    def setup_algo_state(self) -> None:
        args = self.args
        assert args.n_steps >= args.max_steps, (
            f"n_steps ({args.n_steps}) must be >= max_steps ({args.max_steps})"
        )
        self._check_truncation_consistency()
        from methods.rl_agent.training.utils import RewardNormalizer
        self.reward_normalizer = (
            None if args.no_norm_reward
            else RewardNormalizer(
                n_envs=args.n_envs, gamma=args.gamma, clip=args.norm_reward_clip,
            )
        )

    def _check_truncation_consistency(self) -> None:
        args = self.args
        from pcb_world.core.reward_config import get_reward_config
        cfg = get_reward_config(args.reward_rule)
        trunc_mode = cfg._potential_cfg.get("truncation_mode", "none")
        bootstrap = not args.no_truncation_bootstrap
        if cfg._mode == "terminal":
            if trunc_mode == "full" and bootstrap:
                raise ValueError(
                    f"Invalid combination: reward_rule={args.reward_rule} has "
                    f"truncation_mode=full but GAE bootstrap is enabled. "
                    f"env emits Φ(s_truncate) as r_T AND critic V(s_next) "
                    f"predicts the same Φ → return target ≈ 2Φ (double count). "
                    f"Either pass --no-truncation-bootstrap or switch to a "
                    f"terminal reward config with truncation_mode=none "
                    f"(e.g. drc_sparse_promoted_ppo)."
                )
            if trunc_mode == "none" and not bootstrap:
                raise ValueError(
                    f"Invalid combination: reward_rule={args.reward_rule} has "
                    f"truncation_mode=none and --no-truncation-bootstrap is set. "
                    f"Truncated trajectories contribute 0 to Σreturn (no env-side "
                    f"Φ, no critic bootstrap) → V(s_truncate) trained to 0 "
                    f"regardless of partial Φ. Drop --no-truncation-bootstrap "
                    f"(default critic bootstrap) or switch to a terminal reward "
                    f"config with truncation_mode=full (e.g. drc_sparse_promoted_grpo)."
                )
        print(
            f"  truncation: mode={trunc_mode} bootstrap={bootstrap} "
            f"(reward_mode={cfg._mode})"
        )
        print()

    # --- rollout / targets / update ---
    def collect_rollout(self):
        from methods.rl_agent.training.collect import collect_n_steps_ppo
        return collect_n_steps_ppo(
            self.envs, self.agent, self.device,
            n_steps=self.args.n_steps,
            reward_normalizer=self.reward_normalizer,
            bootstrap_truncation=not self.args.no_truncation_bootstrap,
            mem_budget=self.mem_budget_rollout,
        )

    def compute_targets(self, coll) -> dict:
        from methods.rl_agent.training.buffer import (
            compute_gae_flat, ppo_collector_to_buffer,
        )
        advantages, returns = compute_gae_flat(
            rewards=coll.rewards, values=coll.values,
            episode_starts=coll.episode_starts, final_values=coll.final_values,
            terminal_values=coll.terminal_values,
            gamma=self.args.gamma, gae_lambda=self.args.gae_lambda,
        )
        return ppo_collector_to_buffer(coll, advantages, returns)

    def update_kwargs(self) -> dict:
        args = self.args
        return {
            "algo": "ppo",
            "clip_eps": args.clip_eps, "entropy_coef": args.entropy_coef,
            "entropy_norm": bool(getattr(args, "entropy_norm", False)),
            "n_epochs": args.n_epochs, "batch_size": args.batch_size,
            "max_grad_norm": args.max_grad_norm, "vf_coef": args.vf_coef,
            "normalize_advantages": not args.no_normalize_adv,
            "mem_budget": self.mem_budget_update,
        }

    def aggregate_metrics(self, metrics: dict, coll, buffer) -> None:
        from methods.rl_agent.training.utils import explained_variance

        def _ms(values, *, with_max=False):
            if not len(values):
                return (0.0, 0.0, 0.0) if with_max else (0.0, 0.0)
            arr = np.asarray(values, dtype=np.float64)
            if with_max:
                return float(arr.mean()), float(arr.std()), float(arr.max())
            return float(arr.mean()), float(arr.std())

        if coll.episode_rewards:
            metrics["mean_reward"] = float(np.mean(coll.episode_rewards))
            metrics["std_reward"] = float(np.std(coll.episode_rewards))
            metrics["mean_ep_length"] = float(np.mean(coll.episode_lengths))
        else:
            metrics["mean_reward"] = metrics["std_reward"] = metrics["mean_ep_length"] = 0.0

        metrics["drc_violations_mean"], metrics["drc_violations_std"] = _ms(coll.episode_drc_violations)
        (metrics["final_potential_mean"], metrics["final_potential_std"],
         metrics["final_potential_max"]) = _ms(coll.episode_final_potentials, with_max=True)
        metrics["final_unrouted_mean"], metrics["final_unrouted_std"] = _ms(coll.episode_unrouted_counts)
        metrics["final_wirelength_mean"], metrics["final_wirelength_std"] = _ms(coll.episode_wirelengths)
        metrics["final_via_count_mean"], metrics["final_via_count_std"] = _ms(coll.episode_via_counts)
        metrics["final_track_count_mean"], metrics["final_track_count_std"] = _ms(coll.episode_track_counts)
        metrics["terminated_rate"] = (
            float(np.mean(coll.episode_terminated)) if coll.episode_terminated else 0.0
        )
        metrics["ratsnest_reduction_mean"], metrics["ratsnest_reduction_std"] = _ms(coll.episode_ratsnest_reduction)
        # Per-step engine latency telemetry (valid actions only) — stall visibility.
        # Full distribution: log10(s) histogram (tb/wandb, bucket-compressed) +
        # log-decade bucket counts as scalars (cross-run comparable time series).
        _st = getattr(coll, "step_times", None) or []
        _ST_BUCKETS = (            # (key, lo_s, hi_s) — log-decade edges
            ("step_time_le_10ms", 0.0, 0.01),
            ("step_time_10ms_100ms", 0.01, 0.1),
            ("step_time_100ms_1s", 0.1, 1.0),
            ("step_time_1s_10s", 1.0, 10.0),
            ("step_time_10s_100s", 10.0, 100.0),
            ("step_time_ge_100s", 100.0, float("inf")),
        )
        if _st:
            _st_arr = np.asarray(_st, dtype=np.float64)
            metrics["step_time_mean"] = float(_st_arr.mean())
            metrics["step_time_p95"] = float(np.percentile(_st_arr, 95))
            metrics["step_time_max"] = float(_st_arr.max())
            for key, lo, hi in _ST_BUCKETS:
                metrics[key] = int(((_st_arr >= lo) & (_st_arr < hi)).sum())
            # Raw log10 values for the histogram sink — consumed (popped) by
            # _log_common; never serialized into checkpoints/W&B config.
            metrics["_step_time_log10"] = np.log10(np.maximum(_st_arr, 1e-4))
        else:
            metrics["step_time_mean"] = metrics["step_time_p95"] = metrics["step_time_max"] = 0.0
            for key, _lo, _hi in _ST_BUCKETS:
                metrics[key] = 0
        metrics["explained_variance"] = explained_variance(
            buffer["old_values"], buffer["returns"],
        )
        metrics["n_episodes"] = len(coll.episode_rewards)
        metrics["invalid_action_ratio"] = float(
            getattr(coll, "invalid_action_ratio", 0.0)
        )
        # Worker deaths during rollout (each = one contaminated -1.0 episode).
        metrics["engine_crash_count"] = int(getattr(coll, "engine_crash_count", 0))
        metrics["engine_crash_rate"] = (
            metrics["engine_crash_count"] / float(max(coll.rewards.size, 1))
        )

        # value-loss-explosion diagnostics
        raw_r, norm_r = coll.raw_rewards, coll.rewards
        rets, vals = buffer["returns"], buffer["old_values"]
        metrics["raw_reward_max"] = float(raw_r.max())
        metrics["raw_reward_min"] = float(raw_r.min())
        metrics["raw_reward_mean"] = float(raw_r.mean())
        metrics["raw_reward_std"] = float(raw_r.std())
        metrics["norm_reward_max"] = float(norm_r.max())
        metrics["norm_reward_min"] = float(norm_r.min())
        metrics["norm_reward_std"] = float(norm_r.std())
        metrics["return_max"] = float(rets.max())
        metrics["return_min"] = float(rets.min())
        metrics["return_mean"] = float(rets.mean())
        metrics["return_std"] = float(rets.std())
        metrics["value_pred_max"] = float(vals.max())
        metrics["value_pred_min"] = float(vals.min())
        metrics["value_pred_mean"] = float(vals.mean())
        metrics["value_pred_std"] = float(vals.std())

    def log_algo_metrics(self, metrics: dict, iteration: int) -> None:
        w = self.writer
        w.add_scalar("train/value_loss", metrics["value_loss"], iteration)
        w.add_scalar("train/explained_variance", metrics["explained_variance"], iteration)
        w.add_scalar(
            "rollout/invalid_action_ratio", metrics["invalid_action_ratio"], iteration,
        )
        w.add_scalar(
            "rollout/engine_crash_count", metrics["engine_crash_count"], iteration,
        )
        w.add_scalar(
            "rollout/engine_crash_rate", metrics["engine_crash_rate"], iteration,
        )
        if self.reward_normalizer is not None:
            w.add_scalar("train/reward_norm_std", self.reward_normalizer.std, iteration)
        for key in (
            "raw_reward_max", "raw_reward_min", "raw_reward_mean", "raw_reward_std",
            "norm_reward_max", "norm_reward_min", "norm_reward_std",
            "return_max", "return_min", "return_mean", "return_std",
            "value_pred_max", "value_pred_min", "value_pred_mean", "value_pred_std",
        ):
            w.add_scalar(f"diag/{key}", metrics[key], iteration)

    # best-ckpt selection is the shared RLTrainer.on_validation (val fp).

    # --- banners / progress ---
    def _print_banner(self) -> None:
        args, boards, device = self.args, self.boards, self.device
        print("PPO Training — KiCadRLModel (with critic head)")
        if self.multi_board:
            print(f"  boards: round_robin over {len(boards)} boards "
                  f"(split={args.boards_difficulty}/{args.boards_split})")
        elif self.per_env_board:
            mode = self.scheduler.mode
            extra = (
                f"~{-(-len(boards) // args.n_envs)} iters/epoch"
                if mode == "per_env_epoch" else "WITH replacement"
            )
            print(f"  boards: {mode} over {len(boards)} boards "
                  f"(split={args.boards_difficulty}/{args.boards_split}), "
                  f"sticky-within-iter, {extra}")
        else:
            print(f"  board:  {self.initial_board}")
        print(f"  device: {device}")
        print(f"  iterations: {args.iterations}, n_envs: {args.n_envs}, n_steps: {args.n_steps}")
        print(f"  lr={args.lr}, clip_eps={args.clip_eps}, "
              f"entropy_coef={args.entropy_coef}, vf_coef={args.vf_coef}")
        print(f"  gamma={args.gamma}, gae_lambda={args.gae_lambda}")
        print(f"  n_epochs={args.n_epochs}, batch_size={args.batch_size}, "
              f"max_grad_norm={args.max_grad_norm}")
        print(f"  norm_reward={not args.no_norm_reward} (clip={args.norm_reward_clip})")
        print(f"  d_model={args.d_model}, n_heads={args.n_heads}, "
              f"n_layers={args.n_layers}, d_ff={args.d_ff}")
        print(f"  reward_rule={args.reward_rule}")
        print(f"  corner_mode={args.corner_mode}deg "
              f"({'MITERED_45 (default, diagonals OK)' if args.corner_mode == 45 else 'MITERED_90 (no diagonals)'})")
        print(f"  warmup_iters={args.warmup_iters}")
        print(f"  coord_encoding={args.coord_encoding}"
              + (f", mlp_hidden={args.mlp_hidden}" if args.coord_encoding == "mlp" else ""))
        print(f"  disable_slot_emb={args.disable_slot_emb}")
        print()

    def _print_progress(self, iteration: int, metrics: dict, elapsed: float) -> None:
        print(
            f"[{iteration:5d}/{self.args.iterations}] "
            f"reward={metrics['mean_reward']:+.3f} ± {metrics['std_reward']:.3f}  "
            f"pi={metrics['policy_loss']:+.4f}  "
            f"v={metrics['value_loss']:.4f}  "
            f"ent={metrics['entropy']:.3f}  "
            f"ev={metrics['explained_variance']:+.3f}  "
            f"inv={metrics['invalid_action_ratio']:.3f}  "
            f"ep_len={metrics['mean_ep_length']:.0f}  "
            f"n_ep={metrics['n_episodes']}  "
            f"time={elapsed:.1f}s"
        )

    def _print_done(self) -> None:
        print(f"\nTraining complete. Best eval fp_mean_of_means: {self.best_eval_fp:+.3f}")
        print(f"Checkpoints: {self.args.save_dir}")
        print(f"TensorBoard: tensorboard --logdir {self.args.log_dir}")


class GRPOTrainer(RLTrainer):
    """Group-relative PPO — no critic; group-mean baseline replaces V(s).

    No fixed n_steps (full episodes per iter), no GAE/value loss/reward
    normalizer; ``--group-size`` slices the ``--n-envs`` pool into sub-groups,
    each contributing one local mean baseline.
    """

    ALGO = "grpo"
    USE_CRITIC = False
    ALLOWED_MISSING_KEYS = {"prev_action", "history_age", "drc", "same_net_bias"}

    def __init__(self, args) -> None:
        if args.n_envs % args.group_size != 0:
            raise ValueError(
                f"--n-envs ({args.n_envs}) must be divisible by "
                f"--group-size ({args.group_size})"
            )
        super().__init__(args)
        from methods.rl_agent.training.utils import RunningRewardStd

        self.n_groups = args.n_envs // args.group_size
        self.reward_std_tracker = RunningRewardStd()
        # best-ckpt is validation-driven (shared RLTrainer.on_validation,
        # val fp) — same criterion as PPO + the LLM trainer.
        self._advantages = None
        self._n_truncated_recovered = 0

    # --- board scheduling: n_groups unique boards, replicated group_size ---
    def _board_pick(self) -> tuple[int, int]:
        return self.n_groups, self.args.group_size

    # --- rollout / targets / update ---
    def collect_rollout(self):
        from methods.rl_agent.training.collect import collect_group_episodes
        (
            trajectories, terminal_rewards, terminal_drc, terminal_pot,
            terminal_unr, terminal_term, terminal_ratsnest_red, terminal_wire,
            terminal_via, terminal_track,
        ) = collect_group_episodes(
            self.envs, self.agent, self.device, max_steps=self.args.max_steps,
            mem_budget=self.mem_budget_rollout,
        )
        return SimpleNamespace(
            trajectories=trajectories, terminal_rewards=terminal_rewards,
            terminal_drc=terminal_drc, terminal_pot=terminal_pot,
            terminal_unr=terminal_unr, terminal_term=terminal_term,
            terminal_ratsnest_red=terminal_ratsnest_red, terminal_wire=terminal_wire,
            terminal_via=terminal_via, terminal_track=terminal_track,
        )

    def compute_targets(self, coll) -> dict:
        from methods.rl_agent.algorithms.grpo import compute_grpo_advantages_grouped
        from methods.rl_agent.training.buffer import flatten_group_to_buffer
        # Truncated-Φ recovery is not implemented (single-Φ accounting).
        self._n_truncated_recovered = 0
        advantages = compute_grpo_advantages_grouped(
            coll.terminal_rewards, self.args.group_size, self.reward_std_tracker,
        )
        self._advantages = advantages
        return flatten_group_to_buffer(coll.trajectories, advantages)

    def update_kwargs(self) -> dict:
        args = self.args
        return {
            "algo": "grpo",
            "clip_eps": args.clip_eps, "entropy_coef": args.entropy_coef,
            "entropy_norm": bool(getattr(args, "entropy_norm", False)),
            "n_epochs": args.n_epochs, "batch_size": args.batch_size,
            "max_grad_norm": args.max_grad_norm, "normalize_advantages": False,
            "mem_budget": self.mem_budget_update,
        }

    def aggregate_metrics(self, metrics: dict, coll, buffer) -> None:
        metrics["mean_reward"] = float(coll.terminal_rewards.mean())
        metrics["std_reward"] = float(coll.terminal_rewards.std())
        metrics["mean_ep_length"] = float(np.mean([len(t) for t in coll.trajectories]))
        metrics["drc_violations_mean"] = float(coll.terminal_drc.mean())
        metrics["drc_violations_std"] = float(coll.terminal_drc.std())
        metrics["final_potential_mean"] = float(np.nan_to_num(coll.terminal_pot).mean())
        metrics["final_potential_std"] = float(np.nan_to_num(coll.terminal_pot).std())
        metrics["final_potential_max"] = float(np.nan_to_num(coll.terminal_pot).max())
        metrics["final_unrouted_mean"] = float(coll.terminal_unr.mean())
        metrics["final_unrouted_std"] = float(coll.terminal_unr.std())
        metrics["final_wirelength_mean"] = float(coll.terminal_wire.mean())
        metrics["final_wirelength_std"] = float(coll.terminal_wire.std())
        metrics["final_via_count_mean"] = float(coll.terminal_via.mean())
        metrics["final_via_count_std"] = float(coll.terminal_via.std())
        metrics["final_track_count_mean"] = float(coll.terminal_track.mean())
        metrics["final_track_count_std"] = float(coll.terminal_track.std())
        metrics["ratsnest_reduction_mean"] = float(coll.terminal_ratsnest_red.mean())
        metrics["ratsnest_reduction_std"] = float(coll.terminal_ratsnest_red.std())
        metrics["terminated_rate"] = float(coll.terminal_term.mean())
        metrics["n_episodes"] = int(len(coll.trajectories))
        metrics["n_truncated_recovered"] = self._n_truncated_recovered

        # diagnostics: per-step raw rewards + group advantages
        all_step = np.asarray(
            [float(s["reward"]) for traj in coll.trajectories for s in traj],
            dtype=np.float64,
        )
        if all_step.size == 0:
            all_step = np.zeros(1, dtype=np.float64)
        adv = np.asarray(self._advantages, dtype=np.float64)
        metrics["raw_reward_max"] = float(all_step.max())
        metrics["raw_reward_min"] = float(all_step.min())
        metrics["raw_reward_mean"] = float(all_step.mean())
        metrics["raw_reward_std"] = float(all_step.std())
        metrics["advantage_max"] = float(adv.max())
        metrics["advantage_min"] = float(adv.min())
        metrics["advantage_mean"] = float(adv.mean())
        metrics["advantage_std"] = float(adv.std())

    def log_algo_metrics(self, metrics: dict, iteration: int) -> None:
        w = self.writer
        w.add_scalar("train/reward_norm_std", self.reward_std_tracker.std, iteration)
        w.add_scalar("train/n_truncated_recovered", metrics["n_truncated_recovered"], iteration)
        for key in ("raw_reward_max", "raw_reward_min", "raw_reward_mean", "raw_reward_std",
                    "advantage_max", "advantage_min", "advantage_mean", "advantage_std"):
            w.add_scalar(f"diag/{key}", metrics[key], iteration)

    # best-ckpt selection is the shared RLTrainer.on_validation (val fp); GRPO
    # does not keep a separate train-reward best (consistent fp-gain criterion
    # across algorithms).

    # --- banners / progress ---
    def _print_banner(self) -> None:
        args, boards, device = self.args, self.boards, self.device
        print("GRPO Training — KiCadRLModel")
        if self.multi_board:
            print(f"  boards: round_robin over {len(boards)} boards "
                  f"(split={args.boards_difficulty}/{args.boards_split})")
        elif self.per_env_board:
            mode = self.scheduler.mode
            extra = (
                f"~{-(-len(boards) // self.n_groups)} iters/epoch"
                if mode == "per_env_epoch" else "WITH replacement"
            )
            print(f"  boards: {mode} over {len(boards)} boards "
                  f"(split={args.boards_difficulty}/{args.boards_split}), "
                  f"n_groups={self.n_groups} unique boards/iter, "
                  f"sticky-within-iter, {extra}")
        else:
            print(f"  board:  {self.initial_board}")
        print(f"  device: {device}")
        print(f"  iterations: {args.iterations}, "
              f"n_envs: {args.n_envs}, group_size: {args.group_size} "
              f"(= {self.n_groups} sub-groups/iter)")
        print(f"  lr: {args.lr}, clip_eps: {args.clip_eps}, "
              f"entropy_coef: {args.entropy_coef}, batch_size: {args.batch_size}")
        print(f"  d_model: {args.d_model}, n_heads: {args.n_heads}, "
              f"n_layers: {args.n_layers}, d_ff: {args.d_ff}")
        print(f"  reward_rule: {args.reward_rule}")
        print(f"  corner_mode={args.corner_mode}deg "
              f"({'MITERED_45' if args.corner_mode == 45 else 'MITERED_90'})")
        print(f"  warmup_iters: {args.warmup_iters}")
        print(f"  coord_encoding={args.coord_encoding}"
              + (f", mlp_hidden={args.mlp_hidden}" if args.coord_encoding == "mlp" else ""))
        print(f"  disable_slot_emb={args.disable_slot_emb}, "
              f"same_net_bias={args.same_net_bias}")
        print()

    def _print_progress(self, iteration: int, metrics: dict, elapsed: float) -> None:
        print(
            f"[{iteration:5d}/{self.args.iterations}] "
            f"reward={metrics['mean_reward']:+.3f} ± {metrics['std_reward']:.3f}  "
            f"loss={metrics['loss']:.4f}  ent={metrics['entropy']:.3f}  "
            f"ep_len={metrics['mean_ep_length']:.0f}  "
            f"trunc_rec={metrics['n_truncated_recovered']}/{self.args.n_envs}  "
            f"time={elapsed:.1f}s"
        )

    def _print_done(self) -> None:
        print(f"\nTraining complete. Best eval fp_mean_of_means: {self.best_eval_fp:+.3f}")
        print(f"Checkpoints: {self.args.save_dir}")
        print(f"TensorBoard: tensorboard --logdir {self.args.log_dir}")
