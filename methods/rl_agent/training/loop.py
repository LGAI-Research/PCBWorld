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
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from configs.loader.ckpt import ckpt_entries
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


def val_env_config(args, user=()):
    """The validation env config of a training config ``args``: the training env (minus the
    train-only layer) resolved once in mode ``eval`` (``eval.rollout.rl.resolve_eval_env``),
    ``user`` layers on top. The trainer and the ``--async-val`` watcher build it once and hand
    it to :func:`build_evaluators` and the validation run record."""
    from configs.loader.schema import RLEnvConfig
    from eval.rollout.rl import resolve_eval_env

    return resolve_eval_env(RLEnvConfig.from_namespace(args), step_drc=True,
                            where="the training run's config", user=user)


def diag_env_config(args):
    """The diagnostic eval sets' (eval2..eval5) env config, or None when the run pins no
    diagnostic protocol: ``--eval-diag-max-steps`` / ``--eval-diag-masking-rule`` pin it
    cell-independently (rules/experiments.md §6: inline eval otherwise inherits the TRAIN env
    config, so cells varying max_steps or masking would silently change their own
    diagnostics) — an eval-mode override layer of :func:`val_env_config`. Primary val and
    val_greedy stay native. Shared by :func:`build_evaluators` and the queue watcher."""
    from configs.loader.resolve import Layer

    diag = {k: v for k, v in (("max_steps", args.eval_diag_max_steps),
                              ("masking_rule", args.eval_diag_masking_rule)) if v is not None}
    if not diag:
        return None
    return val_env_config(args, user=[Layer("code:methods/rl_agent/training/loop.py", diag,
                                            where={k: f"--eval-diag-{k.replace('_', '-')}" for k in diag})])


def build_evaluators(args, agent, device, eval_boards, *, val_env, mem_budget=None,
                     expect_env_kwargs=None):
    """Primary + diagnostic (eval2..eval5) Evaluators over one shared rollout fn.

    The single construction point for in-training validation — used by the
    trainer's ``_setup_evaluator`` AND the ``--async-val`` watcher, so the eval
    semantics (seeds, batching) cannot drift between them. ``val_env`` is
    :func:`val_env_config` of ``args`` (resolved once by the caller). Validation always
    MEASURES DRC (``step_drc=True`` there) so val metrics (drc_violations,
    final_potential) are comparable across cells; ``--no-drc-tokens`` only suppresses
    the DRC *observation* tokens. Returns ``(primary_evaluator, [(prefix, evaluator), ...])``.
    """
    from pathlib import Path

    from eval.evaluator import Evaluator
    from eval.rollout.rl import eval_env_config, eval_transformer
    from methods._shared.board_loader import load_boards_from_dir_or_list

    env_config = eval_env_config(val_env)

    def _rollout(boards, config, *, n_rollouts=args.eval_n_rollouts, deterministic=False, **kw):
        return eval_transformer(
            agent, device, boards, env_config=config,
            n_rollouts=n_rollouts, n_envs=args.n_envs,
            base_seed=args.eval_base_seed, max_steps=config.env.max_steps,
            deterministic=deterministic, boards_per_batch=args.eval_boards_per_batch,
            final_drc=False, save_artifacts=False, mem_budget=mem_budget, **kw)

    def _rollout_fn(boards):
        # Cross-check the trainer's startup record against what validation actually
        # builds — the record is a prediction until a real pool confirms it.
        return _rollout(boards, env_config, expect_env_kwargs=expect_env_kwargs)

    evaluator = Evaluator(_rollout_fn, eval_boards)

    # Optional diagnostic eval sets (eval2..eval5): fixed board-lists (e.g.
    # real d3-b / d3-a) evaluated each cadence under <prefix>/*. Same agent, only the
    # board list and (with --eval-diag-*) the protocol differ; they do NOT drive best-ckpt.
    diag_run = diag_env_config(args)
    if diag_run is None:
        _rollout_fn_diag = _rollout_fn
    else:
        diag_env = eval_env_config(diag_run)
        print("  diag eval override (eval2..eval5): "
              f"max_steps={diag_env.env.max_steps} masking_rule={diag_env.env.masking_rule}")

        def _rollout_fn_diag(boards):
            return _rollout(boards, diag_env)

    extras = []
    for boards_file, pfx in (
        (args.eval2_boards, args.eval2_prefix),
        (args.eval3_boards, args.eval3_prefix),
        (args.eval4_boards, args.eval4_prefix),
        (args.eval5_boards, args.eval5_prefix),
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
    if eval_boards and not args.no_eval_greedy:
        def _rollout_fn_greedy(boards):
            return _rollout(boards, env_config, n_rollouts=1, deterministic=True)

        extras.append(("val_greedy", Evaluator(_rollout_fn_greedy, eval_boards)))
        print(f"  inline eval (greedy argmax): {len(eval_boards)} boards -> val_greedy/*")
    return evaluator, extras


def _split_identity(run_args: dict):
    """(board ids, real dataset dir, board_filename) the training split of a run resolves to,
    read with that run's own ``boards_*`` args; None when the file cannot be read that way."""
    import json
    from configs.loader.paths import expand_data_path

    try:
        with open(run_args["boards_json"]) as f:
            data = json.load(f)
        split = run_args["boards_split"]
        ids = data[run_args["boards_difficulty"]][split]
        ds = run_args.get("boards_dataset_dir") or data["dataset_dirs"][split]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return ids, os.path.realpath(expand_data_path(ds)), data.get("board_filename")


def _board_list_digest(paths) -> str:
    """Digest of a resolved board list, order included. The common directory is resolved
    once (a symlinked data root and its target give the same digest); the rest is string work."""
    import hashlib

    paths = [os.path.abspath(str(p)) for p in paths]
    root = os.path.dirname(paths[0]) if len(paths) == 1 else os.path.commonpath(paths)
    cut = len(root.rstrip(os.sep) + os.sep)          # every path starts with root + sep
    h = hashlib.sha256(os.path.realpath(root).encode())
    for p in paths:
        h.update(b"\0" + p[cut:].encode())
    return h.hexdigest()[:16]


class RLTrainer(Trainer):
    """Shared decoder-policy RL training core (PPO/GRPO)."""

    # --- subclass contract ---
    ALGO: str = "rl"
    USE_CRITIC: bool = True
    # Checkpoint-load tolerance: a missing policy key is accepted when its name
    # contains one of these substrings. Subclasses inherit it or assign
    # ``RLTrainer.ALLOWED_MISSING_KEYS | {...}``; never repeat the base values. No
    # ``history_age``: the run builds the checkpoint's action-history layout (an age-less
    # checkpoint → a model without the age projection), so its keys are never missing.
    ALLOWED_MISSING_KEYS: frozenset[str] = frozenset({"prev_action", "drc"})

    def __init__(self, args) -> None:
        """``args``: the run's resolved config (``configs.loader.resolve.RunConfig``, frozen —
        ``methods.rl_agent.training.args.resolve_train_args``)."""
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
        self.val_env = None      # the validation env config (configs.loader.resolve.RunConfig)
        self._ckpt_stamp = None  # provenance + obs probe, built on first save
        self._run_record_path = None   # <save-dir>/run_record*.yaml, written first in setup()
        # optional diagnostic eval sets [(prefix, Evaluator)] -> <prefix>/* (eval2..eval5)
        self.extra_evaluators = []
        self.eval_cadence = None
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
        self._refuse_resume_overwrite()
        self._write_run_record()    # before any env / engine / model exists: survives a crash
        self.val_env = val_env_config(args)
        self.device = auto_device(args.device)
        self._print_banner()
        self.setup_algo_state()  # algo pre-flight checks + algo state

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.envs = self._build_envs()
        self._record_effective()
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
        self._init_from_ckpt()      # BC-initialised fine-tune: weights only
        self._build_ref_policy()    # frozen KL reference (off unless --kl-ref-coef)
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

    def _refuse_resume_overwrite(self) -> None:
        """A resume writes ``policy_iter_<N>.pt`` / ``policy_last.pt`` into ``save_dir`` from the
        checkpoint's iteration + 1 on: a ``save_dir`` that already holds a later checkpoint (an
        earlier checkpoint of the run resumed in place) is refused before anything is written."""
        from pathlib import Path

        args = self.args
        if not args.resume:
            return
        it, save_dir = args.ckpt_meta["iteration"], Path(args.save_dir)
        later = sorted(p.name for p in save_dir.glob("policy_iter_*.pt")
                       if p.stem.removeprefix("policy_iter_").isdigit()
                       and int(p.stem.removeprefix("policy_iter_")) > it)
        last = save_dir / "policy_last.pt"
        if last.is_file() and not last.samefile(args.resume):
            last_it = int(torch.load(last, map_location="cpu", weights_only=False)["iteration"])
            if last_it > it:
                later.append(f"policy_last.pt (iteration {last_it})")
        if later:
            raise RuntimeError(
                f"--resume {args.resume} (iteration {it}) would overwrite later checkpoints in "
                f"save_dir {save_dir}: {', '.join(later)} — pass a fresh --save-dir")

    def _write_run_record(self) -> None:
        """``<save-dir>/run_record.yaml`` (a resumed run: ``run_record.resume_iter<N>.yaml``)."""
        from configs.loader.record import board_record, run_record_filename, write_run_record

        args = self.args
        self._run_record_path = write_run_record(
            os.path.join(args.save_dir, run_record_filename(args)), args,
            boards=board_record(self.boards, board=args.board if args.boards_order == "single" else None,
                                boards_json=args.boards_json,
                                boards_dataset_dir=args.boards_dataset_dir),
            readback=True, extra={"trainer": self.ALGO})

    def _record_effective(self) -> None:
        """Fill the run record's ``effective`` column from the first training env's sinks
        (``PCBWorld.effective_config``; its constructor already raised on a mismatch)."""
        from configs.loader.record import fill_effective

        envs = self.envs
        effective = (envs[0].effective_config() if isinstance(envs, list)
                     else envs.env_method("effective_config", indices=[0])[0])
        fill_effective(self._run_record_path, self.args, effective)

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
            if args.async_val:
                raise RuntimeError("--async-val requires --eval-split")
            return
        if args.boards_order not in ("round_robin", "per_env_random", "per_env_epoch"):
            # A single-board run has no --boards-json split: the default (or the resumed
            # run's) eval_split is skipped, one asked for on this command line or recipe is an error.
            source = args.provenance["eval_split"].source
            if source.startswith(("cli", "recipe:", "code:")):
                raise RuntimeError("--eval-split requires a multi-board --boards-order")
            if args.async_val:
                raise RuntimeError("--async-val requires a multi-board --boards-order")
            print(f"  validation: off — single-board run (--boards-order {args.boards_order}); "
                  f"eval_split={args.eval_split!r} ({source}) skipped", flush=True)
            return
        try:
            eval_boards = load_eval_boards(args)
        except ValueError as e:
            source = args.provenance["eval_split"].source
            if source.startswith(("cli", "recipe:", "code:")):
                raise
            raise RuntimeError(f"{e} — eval_split={args.eval_split!r} came from {source} "
                               "(configs/rl.yaml rl_train.shared.eval_split); name a split of this "
                               "--boards-json with --eval-split, or --no-eval-split to train "
                               "without validation") from e
        if not args.async_val:
            # Inline eval shares the trainer's env pool; the async watcher
            # sizes its own pool (--n-envs on the watcher).
            assert args.n_envs >= args.eval_n_rollouts, (
                f"n_envs ({args.n_envs}) must be >= eval_n_rollouts "
                f"({args.eval_n_rollouts}) for inline eval"
            )
        mode = "async" if args.async_val else "inline"
        print(
            f"  {mode} eval: {len(eval_boards)} boards × "
            f"{args.eval_n_rollouts} rollouts after every full "
            f"{len(self.boards)}-iter pass"
        )
        self.eval_boards = eval_boards

    def _build_envs(self):
        from configs.loader.schema import RLEnvConfig

        args = self.args
        # The config validation builds plus the train-only layer (reward noise,
        # augmentation): every train-only value is a train/val difference that has
        # to be declared at launch (--expect-env-diff).
        env_config = RLEnvConfig.for_training(args)
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
                return [
                    make_decoder_env(self.initial_board, env_config, seed=args.seed + i,
                                     policy_net_select=args.policy_net_select)
                    for i in range(args.n_envs)
                ]
        # Training reloads a board into every worker each iteration, which
        # rebuilds the wrapper — keep each worker's random stream moving across
        # those rebuilds instead of rewinding it to its first draws. Eval pools
        # leave this off so a board's rollout stays independent of board order.
        return make_decoder_env_pool(
            self.initial_board, args.n_envs, env_config, seed=args.seed,
            policy_net_select=args.policy_net_select, advance_rng_on_reload=True,
        )

    def _record_env_contract(self) -> None:
        """Record the validation env, then gate its difference from the training env.

        ``run_record.yaml`` holds the training run's config; ``run_record.val.yaml`` holds
        what validation builds — ``self.val_env``, the config the evaluators build
        (:func:`val_env_config`), so no rollout is needed to know it — with the validation
        seed and ``policy_net_select``. The two env configs are compared key by key and every
        difference must be declared at launch (``--expect-env-diff``): the 2026-08-20
        five-knob drift was a silent train/val difference.
        """
        from configs.loader.record import (
            VAL_RUN_RECORD, board_record, check_expected_env_diff, env_diff, write_run_record,
        )
        from configs.loader.schema import RLEnvConfig
        from eval.rollout.rl import eval_policy_net_select

        args = self.args
        val_meta = {"seed": args.eval_base_seed, "policy_net_select": eval_policy_net_select(self.agent)}
        write_run_record(
            os.path.join(args.save_dir, VAL_RUN_RECORD), self.val_env, extra=val_meta, echo=False,
            boards=None if self.eval_boards is None else board_record(
                [b.path for b in self.eval_boards], boards_json=args.boards_json))
        self._recorded_val_env = {**self.val_env.values, **val_meta}
        diff = env_diff(RLEnvConfig.for_training(args).to_record(), self.val_env.values)
        if diff:
            print(f"[env-contract] train/val diff: {sorted(diff)}")
        check_expected_env_diff(diff, args.expect_env_diff)

    def _build_policy(self):
        from configs.loader.schema import RLPolicyConfig

        # slot_perm augmentation samples a fixed permutation over
        # [0, N_MAX_SLOTS=64) in the wrapper, so it is incompatible with a
        # policy whose slot table was enlarged (e.g. d3b's --n-max-slots 128).
        if self.args.slot_perm and self.args.n_max_slots != 64:
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
        cw = int(getattr(args, "critic_warmup_iters", 0) or 0)
        if args.warmup_iters > 0 or cw > 0:
            # Scheduler count = completed iterations. Iterations 1..cw are the
            # critic warm-up (constant --critic-warmup-lr, only the critic head
            # gets gradients); the actor's linear LR warmup starts after them.
            cw_factor = (
                float(getattr(args, "critic_warmup_lr", args.lr)) / float(args.lr)
                if args.lr > 0 else 1.0
            )

            def lr_lambda(iteration: int) -> float:
                # The EV gate may extend the critic phase past ``cw``: the
                # trainer records where it actually ended (scheduler count).
                cw_end = getattr(self, "_critic_warmup_end", cw)
                if iteration < cw_end:
                    return cw_factor
                it = iteration - cw_end
                if args.warmup_iters > 0 and it < args.warmup_iters:
                    return it / args.warmup_iters
                return 1.0
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return None

    # ------------------------------------------------------------------
    # BC-initialised fine-tuning (methods/rl_agent/training/finetune.py)
    # ------------------------------------------------------------------
    def _init_from_ckpt(self) -> None:
        """Load policy WEIGHTS from ``--init-ckpt`` (e.g. a BC ``policy_step_*.pt``).

        Unlike ``--resume``: optimizer, iteration counter, LR schedule and the
        reward normalizer start fresh; a missing ``critic_head`` (BC trains no
        critic) is left at its fresh init. ``--resume`` after this overrides it.
        The run's arch keys are resolve's (mode init-ckpt): the checkpoint's, except a value
        the checkpoint only estimates, which the command line may replace (with a warning).
        """
        path = getattr(self.args, "init_ckpt", None)
        if not path:
            return
        print(f"Initialising policy weights from: {path}")
        iteration, missing = self._load_init_weights(self.policy, path)
        print(
            f"  [init-ckpt] step={iteration}  "
            f"newly initialised: {missing or 'none'}"
        )

    def _load_init_weights(self, model, path) -> tuple:
        """Load the init checkpoint's weights into ``model`` (built from the run's arch
        config); only critic / allowed keys may be missing. Returns ``(iteration, missing)``."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        from methods.rl_agent.models.loader import pad_legacy_entity_type_rows
        state = ckpt["policy_state_dict"]
        pad_legacy_entity_type_rows(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = set(self.ALLOWED_MISSING_KEYS) | {"critic"}
        hard_missing = [
            k for k in missing if not any(tag in k.lower() for tag in allowed)
        ]
        if hard_missing or unexpected:
            raise RuntimeError(
                f"init-ckpt load mismatch: missing={hard_missing}, "
                f"unexpected={unexpected}",
            )
        return ckpt.get("iteration"), missing

    def _build_ref_policy(self) -> None:
        """Frozen reference for the KL(pi || pi_ref) penalty (``--kl-ref-coef``)."""
        self.ref_policy = None
        self._kl_seen = frozenset()
        self._kl_mask_on = False
        coef = float(getattr(self.args, "kl_ref_coef", 0.0) or 0.0)
        if coef <= 0.0:
            return
        init_ckpt = getattr(self.args, "init_ckpt", None)
        path = getattr(self.args, "kl_ref_ckpt", None) or init_ckpt
        if not path:
            raise ValueError("--kl-ref-coef needs --kl-ref-ckpt or --init-ckpt")
        if int(getattr(self.args, "update_gpus", 1)) > 1:
            raise ValueError("--kl-ref-coef is not supported with --update-gpus > 1")
        from methods.rl_agent.training.finetune import parse_type_ids
        if path == init_ckpt:
            # The init checkpoint's weights under the run's arch config — the same
            # tokenization as pi, also where resolve took a command-line value over an estimate.
            from configs.loader.schema import RLPolicyConfig
            ref = RLPolicyConfig.from_namespace(self.args, use_critic=self.USE_CRITIC).build(self.device)
            self._load_init_weights(ref, path)
        else:
            from methods.rl_agent.policy.agent import KiCadRLAgent
            ref = KiCadRLAgent.from_checkpoint(
                path, self.device, strict=self.args.strict_ckpt).model
        ref.eval()
        for prm in ref.parameters():
            prm.requires_grad_(False)
        self.ref_policy = ref
        self._kl_mask_on = bool(getattr(self.args, "kl_ref_mask_unseen_drc", False))
        self._kl_seen = parse_type_ids(getattr(self.args, "kl_ref_seen_drc_types", ""))
        print(
            f"  [kl-ref] beta={coef}  ref={path}  "
            f"mask_unseen_drc={self._kl_mask_on} seen_type_ids={sorted(self._kl_seen)}"
        )

    def _in_critic_warmup(self, iteration: int) -> bool:
        """Critic-only phase for iteration ``iteration``: the fixed
        ``--critic-warmup-iters``, optionally extended by the EV gate
        (``--critic-warmup-min-ev`` held for ``--critic-warmup-ev-streak``
        iterations, capped at ``--critic-warmup-max-iters``)."""
        args = self.args
        cw = int(getattr(args, "critic_warmup_iters", 0) or 0)
        if cw <= 0:
            return False
        if iteration <= cw:
            return True
        min_ev = float(getattr(args, "critic_warmup_min_ev", 0.0) or 0.0)
        if min_ev <= 0.0 or getattr(self, "_critic_gate_passed", False):
            return False
        cap = int(getattr(args, "critic_warmup_max_iters", 0) or 0) or 3 * cw
        if iteration > cap:
            if not getattr(self, "_critic_gate_capped", False):
                self._critic_gate_capped = True
                print(f"  [critic-warmup] EV gate {min_ev} not reached by iter {cap}; "
                      f"starting the actor phase anyway (last EVs: "
                      f"{[round(e, 3) for e in getattr(self, '_ev_history', [])[-3:]]})", flush=True)
            return False
        return True

    def _note_critic_warmup_ev(self, iteration: int, metrics: dict) -> None:
        """Feed the EV gate after an iteration; records where the critic phase ended."""
        args = self.args
        hist = getattr(self, "_ev_history", None)
        if hist is None:
            hist = self._ev_history = []
        hist.append(float(metrics.get("explained_variance", 0.0)))
        if not getattr(self, "_actor_frozen", False):
            return
        min_ev = float(getattr(args, "critic_warmup_min_ev", 0.0) or 0.0)
        streak = max(1, int(getattr(args, "critic_warmup_ev_streak", 3) or 3))
        cw = int(getattr(args, "critic_warmup_iters", 0) or 0)
        passed = (min_ev > 0.0 and iteration >= cw and len(hist) >= streak
                  and all(e >= min_ev for e in hist[-streak:]))
        cap = int(getattr(args, "critic_warmup_max_iters", 0) or 0) or 3 * cw
        ends_now = (iteration >= cw and min_ev <= 0.0) or passed or iteration >= cap
        if passed and not getattr(self, "_critic_gate_passed", False):
            self._critic_gate_passed = True
            print(f"  [critic-warmup] EV gate passed at iter {iteration} "
                  f"(EV {[round(e, 3) for e in hist[-streak:]]} >= {min_ev}); actor phase starts", flush=True)
        if ends_now and not hasattr(self, "_critic_warmup_end"):
            # scheduler count == completed iterations == this iteration; set once
            self._critic_warmup_end = iteration

    def _resume(self) -> None:
        args = self.args
        self.start_iteration = 1
        if not args.resume:
            return
        print(f"Resuming from checkpoint: {args.resume}")
        from configs.loader.ckpt import load_ckpt_config
        ckpt = torch.load(args.resume, map_location=self.device)
        # The run's arch / experiment keys already equal the checkpoint's (resolve mode
        # resume); its config here only replays the board schedule.
        ck_args, _ = load_ckpt_config(ckpt, strict=args.strict_ckpt)
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
        self._restore_schedules(ckpt, ck_args, completed=self.start_iteration - 1)
        self._restore_best_score(ckpt)
        print(f"  Loaded iteration {self.start_iteration - 1}, resuming from {self.start_iteration}")

    def _restore_best_score(self, ckpt: dict) -> None:
        """``policy_best.pt`` is the best over the whole run, across resumes. The score this
        segment must beat is the better of the resumed ckpt's ``best_val_score`` and the score of
        ``save_dir``'s ``policy_best.pt`` (``best_val_score``, else ``eval_overall``). A resume into
        a new ``save_dir`` copies the resumed run's ``policy_best.pt`` there when it comes from at
        or before the resumed iteration; one from after it (an abandoned branch) is left behind
        and best selection starts fresh. A ``policy_best.pt`` whose score cannot be recovered is
        copied to ``policy_best.unscored.pt`` with a warning and best selection starts fresh, so
        this segment's first validation overwrites it."""
        import shutil
        from configs.loader.schema import DEFAULTS

        def copy(src: str, dst: str) -> None:     # atomic, like _save_ckpt
            shutil.copy2(src, f"{dst}.tmp")
            os.replace(f"{dst}.tmp", dst)

        key = DEFAULTS.best_metric_key.removeprefix("val/")
        best = os.path.join(self.args.save_dir, "policy_best.pt")
        prev = os.path.join(os.path.dirname(os.path.realpath(self.args.resume)), "policy_best.pt")
        if not os.path.isfile(best) and os.path.isfile(prev):
            prev_iter = torch.load(prev, map_location="cpu", weights_only=False).get("iteration")
            if prev_iter is None or prev_iter > ckpt.get("iteration", 0):
                print(f"  [resume] {prev} (iteration {prev_iter}) is not from this run's lineage "
                      f"(resumed at iteration {ckpt.get('iteration', 0)}) — not copied; best "
                      f"selection starts fresh in {self.args.save_dir}", flush=True)
                return
            copy(prev, best)
            print(f"  [resume] the run's best checkpoint {prev} copied to {best}")
        scores = [] if ckpt.get("best_val_score") is None else [float(ckpt["best_val_score"])]
        if os.path.isfile(best):
            payload = torch.load(best, map_location="cpu", weights_only=False)
            score = payload.get("best_val_score")
            if score is None:
                score = (payload.get("eval_overall") or {}).get(key)
            if score is None or not np.isfinite(score):
                kept = os.path.join(self.args.save_dir, "policy_best.unscored.pt")
                if not os.path.exists(kept):    # else: copied by a segment that ended before a val
                    copy(best, kept)
                print(f"  [resume] WARNING: {best} records no val {DEFAULTS.best_metric_key} — "
                      f"kept as {kept}; this segment's first validation overwrites policy_best.pt",
                      flush=True)
                return
            scores.append(float(score))
        if scores:
            self.best_eval_fp = (max if DEFAULTS.best_metric_mode == "max" else min)(scores)
            print(f"  [resume] best val {DEFAULTS.best_metric_key} so far: {self.best_eval_fp:+.4f} "
                  f"— policy_best.pt is written only when a validation beats it")

    # Run args that fix the board scheduler's draw sequence (seed + picks per iteration)
    # and the board list it indexes into; the last two may differ when the split still
    # resolves to the same boards (_split_identity).
    _BOARD_REPLAY_KEYS = ("seed", "n_envs", "group_size", "boards_order",
                          "boards_difficulty", "boards_split", "boards_json", "boards_dataset_dir")
    _SPLIT_LOCATION_KEYS = {"boards_json", "boards_dataset_dir"}

    def _board_digest(self) -> str | None:
        if len(self.boards or []) <= 1:
            return None
        if getattr(self, "_board_digest_cache", None) is None:
            self._board_digest_cache = _board_list_digest(self.boards)   # once per run
        return self._board_digest_cache

    def _schedule_state(self) -> dict:
        """Per-iteration schedule state that setup() rebuilds from scratch."""
        return {
            "board_scheduler": self.scheduler.state_dict(),
            "board_list_digest": self._board_digest(),
            "critic_warmup_end": getattr(self, "_critic_warmup_end", None),
            "critic_gate_passed": bool(getattr(self, "_critic_gate_passed", False)),
            # the gate reads the last `streak` EVs (the cap message the last 3)
            "ev_history": list(getattr(self, "_ev_history", None) or [])[-max(self._ev_streak(), 3):],
        }

    def _ev_streak(self) -> int:
        return max(1, int(getattr(self.args, "critic_warmup_ev_streak", 3) or 3))

    def _same_board_draws(self, ck_args, state: dict | None) -> list[str]:
        """What differs between the board draws of the run whose checkpoint config is
        ``ck_args`` and this one — empty when the saved or replayed order applies to this
        run's board list. A key one side has and the other lacks differs."""
        if state is not None and state.get("board_list_digest") is not None:
            return [] if state["board_list_digest"] == self._board_digest() else ["board list"]
        run = vars(self.args)
        diff = [k for k in self._BOARD_REPLAY_KEYS
                if (k in ck_args) != (k in run) or (k in run and ck_args[k] != run[k])]
        if diff and set(diff) <= self._SPLIT_LOCATION_KEYS:
            old = _split_identity(ck_args)
            if old is not None and old == _split_identity(vars(self.args)):
                diff = []   # the same board list, reached through another split file or dir spelling
        return diff

    def _in_ev_gate_extension(self, completed: int) -> bool:
        args = self.args
        cw = int(getattr(args, "critic_warmup_iters", 0) or 0)
        return (cw > 0 and completed >= max(1, cw - self._ev_streak() + 1)   # EVs the gate at iter cw reads
                and float(getattr(args, "critic_warmup_min_ev", 0.0) or 0.0) > 0.0
                and not getattr(self, "_critic_gate_passed", False)
                and getattr(self, "_critic_warmup_end", None) is None)

    def _restore_schedules(self, ckpt: dict, ck_args, completed: int) -> None:
        """Continue the LR and board schedules after ``completed`` iterations.

        setup() builds both schedulers fresh, so without this a resumed run
        re-runs the LR warmup and re-deals boards from the start of the first
        epoch. Checkpoints without a ``schedules`` entry (written before it
        existed) are reconstructed from the iteration count. ``ck_args`` is the
        checkpoint's resolved config (``configs.loader.ckpt.load_ckpt_config``).
        """
        state = ckpt.get("schedules")
        if state is not None:
            if state.get("critic_warmup_end") is not None:
                self._critic_warmup_end = int(state["critic_warmup_end"])
            if state.get("critic_gate_passed"):
                self._critic_gate_passed = True
            if "ev_history" in state:
                self._ev_history = [float(e) for e in state["ev_history"]]
            elif self._in_ev_gate_extension(completed):
                print("  [resume] WARNING: checkpoint has no EV history; the EV gate needs a "
                      "fresh streak of --critic-warmup-ev-streak iterations from here", flush=True)
        elif int(getattr(self.args, "critic_warmup_iters", 0) or 0) > 0 and completed > 0:
            print("  [resume] WARNING: checkpoint has no critic warm-up state; the EV gate "
                  "and the LR warmup restart point assume the warm-up ended at "
                  "--critic-warmup-iters", flush=True)

        # The LR scheduler steps once per iteration, so its count == completed.
        if self.lr_scheduler is not None:
            sch = self.lr_scheduler
            sch.last_epoch = completed
            for group, base_lr, fn in zip(self.optimizer.param_groups, sch.base_lrs, sch.lr_lambdas):
                group["lr"] = base_lr * fn(completed)
            sch._last_lr = [group["lr"] for group in self.optimizer.param_groups]
            print(f"  [resume] LR schedule at iteration {completed}: lr={sch._last_lr[0]:.3g}")

        sched = self.scheduler
        if sched.mode == "single" or len(self.boards) <= 1 or completed <= 0:
            return  # select_boards never draws here
        diff = self._same_board_draws(ck_args, state)
        if diff:
            print(f"  [resume] WARNING: {diff} differ from the run that wrote the checkpoint — "
                  "board order restarts from a fresh epoch", flush=True)
            return
        if state is not None:
            order = state["board_scheduler"].get("epoch_order") or []
            if max(order, default=-1) >= len(self.boards):   # a split file rewritten in place (no digest to tell)
                print(f"  [resume] WARNING: the saved board order indexes past this run's {len(self.boards)} "
                      "boards — board order restarts from a fresh epoch", flush=True)
                return
            sched.load_state_dict(state["board_scheduler"])
            how = "restored"
        else:
            n_pick, _ = self._board_pick()
            for _ in range(completed):
                sched.next_indices(n_pick)
            how = f"replayed {completed}x{n_pick} draws"
        print(f"  [resume] board order {how}: epoch {sched.epochs_completed}, "
              f"{sched.epoch_remaining} boards left in it")

    def _setup_logger(self) -> None:
        args = self.args
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        config_extras = {
            key: value
            for key, value in {
                "pcbworld_run_backend": os.environ.get("PCBWORLD_RUN_BACKEND"),
                "pcbworld_slurm_qos": os.environ.get("PCBWORLD_SLURM_QOS"),
                "pcbworld_target_iterations": os.environ.get("PCBWORLD_TARGET_ITERATIONS"),
                "pcbworld_stage": os.environ.get("PCBWORLD_STAGE"),
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
            args, self.agent, self.device, self.eval_boards, val_env=self.val_env,
            mem_budget=self.mem_budget_rollout,
            expect_env_kwargs=getattr(self, "_recorded_val_env", None),
        )
        self.eval_cadence = (
            args.eval_every if args.eval_every is not None else len(self.boards)
        )
        if args.async_val:
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
        if not self.args.mem_budget:
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
        transition = buffer["transition"]
        device = self.device
        policy = self.policy

        def _tensors(pos: int, B: int):
            # sample ``pos`` replicated B times, with its stored bundle
            return [obs_list[pos]] * B, transition.select([pos] * B).to(device)

        def probe_update(pos: int, B: int) -> float:
            obs, rows = _tensors(pos, B)
            base = mem_budget_mod.begin_measured_region()
            new_lp, entropy, values = policy.evaluate_actions_and_value(
                obs, rows.actions, dist=rows.dist,
            )
            # The loss *shape* is irrelevant for the peak (the real losses only
            # add (B,)-sized terms); what matters is backward through the
            # transformer stack.
            (new_lp.sum() + entropy.sum() + values.sum()).backward()
            peak = mem_budget_mod.end_measured_region(base)
            self.optimizer.zero_grad(set_to_none=True)
            return peak

        def probe_rollout(pos: int, B: int) -> float:
            obs, rows = _tensors(pos, B)
            base = mem_budget_mod.begin_measured_region()
            policy.act_and_value(obs, dist=rows.dist)
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
        # Critic warm-up: value head only — detach_critic forced on (backbone
        # gets no value gradient) and the update drops the actor terms.
        warm = self._in_critic_warmup(iteration)
        if not hasattr(self, "_detach_critic_base"):
            self._detach_critic_base = bool(getattr(self.policy, "detach_critic", False))
        self._actor_frozen = warm
        self.policy.detach_critic = True if warm else self._detach_critic_base
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
        self._note_critic_warmup_ev(iteration, metrics)
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

        act_types = buffer["transition"].actions[:, 0]
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
            **ckpt_entries(self.args.ckpt_config()),
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
            from configs.loader.record import collect_provenance
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
        payload["schedules"] = self._schedule_state()
        # The best val score so far: a resume must beat it before overwriting policy_best.pt.
        payload["best_val_score"] = (
            float(self.best_eval_fp) if np.isfinite(self.best_eval_fp) else None
        )
        # The writer: a schema-2 config takes the table value only for a key its writer never
        # records (configs/ckpt_migrations.yaml `trainers:`).
        payload["trainer"] = self.ALGO
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
            **ckpt_entries(self.args.ckpt_config()),
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
        ``trainer.best_metric_key`` (run_pcbworld*.sh), so RL and LLM select best
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
                    **ckpt_entries(self.args.ckpt_config()),
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
            None if not args.norm_reward
            else RewardNormalizer(
                n_envs=args.n_envs, gamma=args.gamma, clip=args.norm_reward_clip,
            )
        )

    def _check_truncation_consistency(self) -> None:
        args = self.args
        import inspect
        from pcb_world.core.reward import PotentialReward
        from pcb_world.core.reward_config import get_reward_config
        cfg = get_reward_config(args.reward_rule)
        # Optional rule key: absent means the PotentialReward default.
        trunc_mode = cfg.rule.potential.get(
            "truncation_mode", inspect.signature(PotentialReward).parameters["truncation_mode"].default)
        bootstrap = bool(args.truncation_bootstrap)
        if cfg.mode == "terminal":
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
            f"(reward_mode={cfg.mode})"
        )
        print()

    # --- rollout / targets / update ---
    def collect_rollout(self):
        from methods.rl_agent.training.collect import collect_n_steps_ppo
        return collect_n_steps_ppo(
            self.envs, self.agent, self.device,
            n_steps=self.args.n_steps,
            reward_normalizer=self.reward_normalizer,
            bootstrap_truncation=bool(self.args.truncation_bootstrap),
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
        buffer = ppo_collector_to_buffer(coll, advantages, returns)
        # Diagnostic lambda=1 (Monte-Carlo) returns: explained variance against
        # them is not inflated by the critic's own bootstrap (train/explained_variance_mc).
        _, returns_mc = compute_gae_flat(
            rewards=coll.rewards, values=coll.values,
            episode_starts=coll.episode_starts, final_values=coll.final_values,
            terminal_values=coll.terminal_values,
            gamma=self.args.gamma, gae_lambda=1.0,
        )
        buffer["returns_mc"] = returns_mc.reshape(-1).astype(np.float32)
        if self.ref_policy is not None and self._kl_mask_on:
            from methods.rl_agent.training.finetune import kl_mask_from_obs
            buffer["transition"] = replace(
                buffer["transition"], kl_mask=kl_mask_from_obs(coll.obs_list, self._kl_seen),
            )
        return buffer

    def update_kwargs(self) -> dict:
        args = self.args
        return {
            "algo": "ppo",
            "clip_eps": args.clip_eps, "entropy_coef": args.entropy_coef,
            "entropy_norm": bool(args.entropy_norm),
            "n_epochs": args.n_epochs, "batch_size": args.batch_size,
            "max_grad_norm": args.max_grad_norm, "vf_coef": args.vf_coef,
            "normalize_advantages": bool(args.normalize_adv),
            "mem_budget": self.mem_budget_update,
            # BC-initialised fine-tuning (None/0/False = plain PPO)
            "ref_policy": self.ref_policy,
            "kl_ref_coef": float(getattr(args, "kl_ref_coef", 0.0) or 0.0),
            "actor_frozen": bool(getattr(self, "_actor_frozen", False)),
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
        old_values = buffer["transition"].old_values
        metrics["explained_variance"] = explained_variance(
            old_values, buffer["returns"],
        )
        if "returns_mc" in buffer:
            metrics["explained_variance_mc"] = explained_variance(
                old_values, buffer["returns_mc"],
            )
        metrics["critic_warmup"] = float(getattr(self, "_actor_frozen", False))
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
        rets, vals = buffer["returns"], old_values
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
        if "explained_variance_mc" in metrics:
            w.add_scalar("train/explained_variance_mc", metrics["explained_variance_mc"], iteration)
        if self.ref_policy is not None:
            w.add_scalar("train/kl_ref", metrics.get("kl_ref", 0.0), iteration)
            w.add_scalar("train/kl_ref_mask_frac", metrics.get("kl_ref_mask_frac", 0.0), iteration)
        w.add_scalar("train/critic_warmup", metrics.get("critic_warmup", 0.0), iteration)
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
        print(f"  norm_reward={bool(args.norm_reward)} (clip={args.norm_reward_clip})")
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
            f"ev_mc={metrics.get('explained_variance_mc', float('nan')):+.3f}  "
            + (f"kl={metrics.get('kl_ref', 0.0):.4f}  " if self.ref_policy is not None else "")
            + (f"[critic-warmup] " if metrics.get('critic_warmup') else "")
            +             f"inv={metrics['invalid_action_ratio']:.3f}  "
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
    ALLOWED_MISSING_KEYS = RLTrainer.ALLOWED_MISSING_KEYS | {"same_net_bias"}

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
            "entropy_norm": bool(args.entropy_norm),
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
