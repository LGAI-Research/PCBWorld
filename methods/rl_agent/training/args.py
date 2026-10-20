"""Shared argparse builder for both PPO and GRPO decoder trainers.

The flag list is *derived from the schema* (:mod:`configs.loader.schema`) via
:func:`configs.loader.cli.add_dataclass_args`, so the dataclasses are the single visible
list of every training variable and the CLI cannot drift from them:

    * :class:`RLTrainConfig`  → shared loop / optim / logging / board / inline-eval
      cadence / W&B knobs (defaults = configs/rl.yaml (rl_train:)).
    * :class:`RLEnvConfig`    → env-core + reward overrides + wrapper knobs +
      per-episode augmentation (nests :class:`EnvConfig` → :class:`RewardOverrides`).
    * :class:`RLPolicyConfig` → network architecture.

A small tail adds the handful of flags whose *variables* live in the schema
(``cli_skip``) but whose CLI *form* is bespoke — inverted bools
(``--no-drc-tokens`` / ``--no-mask-start-point`` / ``--no-vecenv`` / ``--no-eval-greedy``;
a runtime one also has its on form, since a resume starts it from the checkpoint), the
opt-in ``--wandb`` toggle, the ``45/90`` ``--corner-mode`` sugar, the required
runtime input ``--board``, and the per-algorithm ``--n-epochs`` / ``--log-dir`` /
``--save-dir`` (parameterized by the calling entrypoint).

Usage in a trainer::

    from methods.rl_agent.training.args import add_shared_args

    def build_arg_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="...")
        add_shared_args(
            p,
            n_epochs_default=10,   # or 4 for GRPO
            log_dir_default="./logs/tb/ppo_decoder",
            save_dir_default="./checkpoints/ppo_decoder",
        )
        # add algorithm-specific flags below
        return p
"""

from __future__ import annotations

import warnings
from argparse import ArgumentParser
from typing import Any, Collection

from configs.loader.knob import Policy, runtime

#: Training options that are not schema fields: the required ``--board`` (a resumed run may
#: name it by another path — the change is in the run record's chain) and the PPO speed /
#: process knobs (the trainer reads them; nothing about the trained policy).
OPTION_POLICIES = {
    "board": Policy("experiment", True, False, "--board"),
    **{dest: runtime(f"--{dest.replace('_', '-')}")
       for dest in ("bf16", "compile_regions", "compile_mode", "attn", "update_gpus", "max_wallclock_sec")},
}

#: Keys older checkpoints store that no training config has and that a resumed run does not
#: need: the pre-``--wandb`` opt-out ``no_wandb`` (W&B is the resuming command's runtime choice),
#: a BC checkpoint's ``bc`` block, and the removed ``--finetune-lr`` / ``--finetune-clip-eps``
#: (the trainer copied them into ``lr`` / ``clip_eps``, which the checkpoint stores).
IGNORED_CKPT_ARGS = frozenset({"no_wandb", "bc", "finetune_lr", "finetune_clip_eps"})


def add_shared_args(
    p: ArgumentParser,
    *,
    n_epochs_default: int = 10,
    log_dir_default: str = "./logs/tb/decoder",
    save_dir_default: str = "./checkpoints/decoder",
) -> None:
    """Register the flags shared between the PPO and GRPO decoder trainers.

    The parser is mutated in place. The caller is expected to add any
    algorithm-specific flags after calling this function.

    All defaults are sourced from the shared schema (which reads
    configs/env.yaml (env:), configs/rl.yaml (rl_policy:, rl_train:)) so the YAML is the single
    place to edit them and training stays in lock-step with the eval ckpt
    fallback.
    """
    from configs.loader.ckpt import add_strict_ckpt_arg
    from configs.loader.cli import add_dataclass_args
    from configs.loader.resolve import add_run_args
    from configs.loader.schema import (CHECKPOINT_LAYOUT_FIELDS, NO_FLAG_FIELDS, RLEnvConfig,
                                       RLPolicyConfig, RLTrainConfig)

    _RLENV = RLEnvConfig()
    _P = RLPolicyConfig()
    _T = RLTrainConfig()

    # --- Shared loop / optim / logging / board / inline-eval cadence / W&B ---
    # Generated from RLTrainConfig (wandb / vecenv / resume are cli_skip — the
    # bespoke/typed forms are added below / by the entrypoints).
    add_dataclass_args(p, _T, style="dash")
    add_strict_ckpt_arg(p)   # --resume / --init-ckpt / --kl-ref-ckpt
    add_run_args(p)          # --recipe / --expect

    # --- Environment / reward / wrapper / augmentation ---
    # --board is a required runtime input, not a config default, so it stays
    # explicit and is the one env flag with no schema field.
    p.add_argument("--board", required=True,
                   help="Path to .kicad_pcb file (ignored when "
                        "--boards-order=round_robin)")
    # Generated from RLEnvConfig (nests EnvConfig -> RewardOverrides). emit_drc_tokens
    # / corner_mode (EnvConfig) and mask_start_point (RLEnvConfig) are cli_skip —
    # their bespoke inverted/sugar forms are added just below.
    add_dataclass_args(p, _RLENV, style="dash")
    # The no-flag env fields (engine knobs, promoted constants): the namespace still carries
    # their YAML values, so the env gets them and the run record and checkpoint record them.
    p.set_defaults(**{k: getattr(_RLENV.env, k) for k in NO_FLAG_FIELDS})
    # The RL training path emits indexed obs (no flag: the YAML json is the eval/LLM format);
    # being a dest, it is resolved and recorded — scripts/profile.py A/Bs it as a code layer.
    p.set_defaults(obs_format="indexed")

    # Bespoke env CLI forms (variables live in the schema; only the flag shape is
    # custom). Order: --no-drc-tokens before --drc-tokens so the store_true sets
    # the dest default (emit by default).
    p.add_argument("--no-drc-tokens", action="store_true", default=False,
                   help="Do not emit DRC state tokens in the observation "
                        "(env returns empty drc_violations list). Reward still "
                        "includes the DRC potential — this only controls the "
                        "policy's state input.")
    p.add_argument("--drc-tokens", dest="no_drc_tokens", action="store_false",
                   help="Emit DRC state tokens. Useful to override an earlier "
                        "--no-drc-tokens in launcher-provided defaults.")
    p.add_argument("--legacy-edge-encoding", action="store_true", default=False,
                   help="Build the tokenizer with 2-point (legacy) edge tokens — "
                        "no edge_mid_proj arc-midpoint module. --resume / "
                        "--init-ckpt take it from the checkpoint (derived from "
                        "its weights); set it only for a fresh run.")
    p.add_argument("--connectivity-filter", dest="connectivity_filter",
                   action="store_true", default=True,
                   help="Connectivity candidate filter (curbs redundant same-net "
                        "loops), ON by default. While routing, every existing-copper "
                        "candidate (pad / via / track endpoint) that the route head "
                        "is ALREADY electrically connected to is dropped — the "
                        "cluster comes from the engine (KiCad CONNECTIVITY_DATA), "
                        "matched on (x, y, layer). Directional candidates are never "
                        "filtered. Use --no-connectivity-filter to restore the "
                        "legacy 'every net candidate selectable' behavior.")
    p.add_argument("--no-connectivity-filter", dest="connectivity_filter",
                   action="store_false",
                   help="Disable the connectivity filter (legacy: already-connected "
                        "copper stays a selectable target).")
    p.add_argument("--pad-graze-margin-mm", type=float,
                   default=_RLENV.pad_graze_margin_mm,
                   help="Pad-graze guard width in mm (0 = off, the default). Drops "
                        "DIRECTIONAL candidates — the only synthesised coordinates in "
                        "the pool — that land in the annulus just outside a same-net "
                        "pad's copper, where a via / track end connects by a sliver "
                        "instead of an anchor: KiCad's shape-overlap connectivity "
                        "counts that as routed, its anchor-based dangling test does "
                        "not, and its track cleaner deletes such copper. Set to the "
                        "via radius (e.g. 0.3) to make the band unaddressable. Real "
                        "geometry (pad centres, existing track endpoints) is never "
                        "filtered.")
    p.add_argument("--corner-mode", type=int, choices=[45, 90], default=45,
                   help="PNS corner constraint angle. 45 = MITERED_45 (default; "
                        "H/V/45 diagonals allowed). 90 = MITERED_90 (H/V only, no "
                        "diagonals).")
    p.add_argument("--no-mask-start-point", action="store_true", default=False,
                   help="Disable MLP-equivalent same-point masking. By default "
                        "the cand index of the current start_route is hard-masked "
                        "(logit → -inf) to prevent zero-length moves and immediate "
                        "start-point reuse. Matches KiCadHLWrapper.action_masks.")
    p.add_argument("--keep-routing-fraction", nargs=2, type=float,
                   default=_RLENV.env.keep_routing_fraction,
                   metavar=("LO", "HI"),
                   help="Keep-routing augmentation: each board load samples "
                        "f~U[LO,HI] and keeps that fraction of the nets' file "
                        "routing (uniform random nets) as the episode's initial "
                        "state; the rest is re-routed from scratch. Requires "
                        "board files that carry complete routing (d3 real "
                        "boards). Default off; eval is always off.")
    p.add_argument("--no-vecenv", action="store_true", default=False,
                   help="Disable subprocess-parallel vectorized env. Use "
                        "sequential list of envs instead (for debugging).")
    p.add_argument("--vecenv", dest="no_vecenv", action="store_false",
                   help="Use the vectorized env (the default; a --resume starts from the "
                        "checkpoint's value).")

    # --- Optimization (PPO-style clipped surrogate used by both algos) ---
    p.add_argument("--n-epochs", type=int, default=n_epochs_default,
                   help="Epochs per training iteration")

    # --- Network architecture (generated from RLPolicyConfig) ---
    # Numeric/encoding fields auto-emitted; the bool toggles + use_critic/legacy_pad
    # are cli_skip (trainer-set or added by the entrypoints).
    add_dataclass_args(p, _P, style="dash")
    # The checkpoint-layout fields have no flag: a fresh run builds the current layout, a
    # --resume / --init-ckpt run takes them from the checkpoint's weights (arch keys, pinned).
    p.set_defaults(**{k: getattr(_P, k) for k in CHECKPOINT_LAYOUT_FIELDS})

    # --- Logging / checkpointing (per-algorithm dirs, parameterized) ---
    p.add_argument("--log-dir", default=log_dir_default)
    p.add_argument("--save-dir", default=save_dir_default)

    # --- W&B master toggle (opt-in; wandb_* string knobs generated above) ---
    p.add_argument("--wandb", action="store_true",
                   help="Enable W&B logging (telemetry is OFF by default). "
                        "Needs a credential (WANDB_API_KEY or `wandb login`); "
                        "without one the run warns loudly and logs to "
                        "TensorBoard only. TensorBoard always writes regardless.")
    p.add_argument("--no-wandb", dest="wandb", action="store_false",
                   help="Turn off --wandb (off by default; a --resume starts from the "
                        "checkpoint's value).")

    # --- Greedy val toggle (inverted; schema RLTrainConfig.eval_greedy) ---
    p.add_argument("--no-eval-greedy", action="store_true",
                   help="Skip the greedy (argmax) 1-rollout val_greedy/* pass "
                        "that runs alongside the sampled val/* each cadence.")
    p.add_argument("--eval-greedy", dest="no_eval_greedy", action="store_false",
                   help="Run the greedy val_greedy/* pass (the default; a --resume starts from "
                        "the checkpoint's value).")


def train_policies(algo_config: Any) -> dict[str, Policy]:
    """The override policy of every training-parser dest (``algo_config``: ``PPOConfig()`` /
    ``GRPOConfig()``)."""
    from configs.loader.resolve import dest_policies
    from configs.loader.schema import INVERTED_DESTS, RLEnvConfig, RLPolicyConfig, RLTrainConfig

    return dest_policies(RLTrainConfig(), RLEnvConfig(), RLPolicyConfig(), algo_config,
                         inverted=INVERTED_DESTS, extra=OPTION_POLICIES)


def resolve_train_args(parser: Any, algo_config: Any, argv: list[str] | None = None, *,
                       code: Any = None, load_ckpt: bool = True) -> Any:
    """Parse a trainer command line (``None`` = ``sys.argv[1:]``) and resolve it once into the
    run's frozen :class:`~configs.loader.resolve.RunConfig` — the args a run trains with and its
    checkpoints record. Mode: ``resume`` with ``--resume`` (the checkpoint gives every key: its
    arch and experiment keys pinned, its runtime keys changeable), ``init-ckpt`` with ``--init-ckpt`` (it pins its arch keys; the optimizer
    starts from ``FINETUNE_DEFAULTS``), else ``fresh``. ``code`` is a ``code:<file>`` layer.
    ``load_ckpt=False`` resolves without the checkpoint layer (mode ``fresh``, the fine-tune
    defaults still follow ``--init-ckpt``): for a caller that supplies the weights itself
    (e.g. a feeder that pairs another run's checkpoints with this trainer config)."""
    from configs.loader.resolve import Layer, parser_defaults, resolve_parsed
    from configs.loader.schema import FINETUNE_DEFAULTS

    args = parser.parse_args(argv)
    policies = train_policies(algo_config)
    init_ckpt = args.init_ckpt if "init_ckpt" in policies else None
    mode = "fresh" if not load_ckpt else "resume" if args.resume else "init-ckpt" if init_ckpt else "fresh"
    ckpt = None if mode == "fresh" else load_ckpt_layer(args.resume or init_ckpt, strict=args.strict_ckpt)[0]
    if mode == "resume":
        warn_unknown_ckpt_args(ckpt, parser_defaults(parser, policies).values)
    mode_defaults = None if not init_ckpt or args.resume else Layer("defaults", FINETUNE_DEFAULTS, where={
        k: f"configs/rl.yaml rl_train.finetune.{k}" for k in FINETUNE_DEFAULTS})
    return resolve_parsed(parser, args, mode, policies=policies, ckpt=ckpt,
                          mode_defaults=mode_defaults, code=code)


def load_ckpt_layer(path: Any, *, strict: bool) -> tuple[Any, Any]:
    """The CKPT layer of the checkpoint at ``path`` (one CPU load; ``meta`` = its ``iteration``)
    and its ``trainer`` marker (None before the marker)."""
    import torch

    from configs.loader.ckpt import load_ckpt_config
    from configs.loader.resolve import ckpt_layer

    payload = torch.load(path, map_location="cpu", weights_only=False)
    values, sources = load_ckpt_config(payload, strict=strict)
    return (ckpt_layer(values, sources, where=str(path),
                       meta={"iteration": int(payload.get("iteration", 0))}),
            payload.get("trainer"))


def warn_unknown_ckpt_args(ckpt: Any, keys: Collection[str]) -> None:
    """Warn about the checkpoint keys the training config ``keys`` does not have — a resolve
    drops them, so a run resumed from (or validating) the checkpoint would not carry them.
    The checkpoint loader's own fields and :data:`IGNORED_CKPT_ARGS` are known."""
    from configs.loader.ckpt import CKPT_FIELDS

    unknown = sorted(set(ckpt.values) - set(keys) - CKPT_FIELDS - IGNORED_CKPT_ARGS)
    if unknown:
        warnings.warn(f"{ckpt.where[unknown[0]]}: checkpoint keys {unknown} are not options of this "
                      "trainer and are dropped — map a renamed key in configs/loader/ckpt.py, or list "
                      "an irrelevant one in IGNORED_CKPT_ARGS (methods/rl_agent/training/args.py)",
                      UserWarning, stacklevel=2)


def resolve_ckpt_train_config(ckpt_path: Any, *, strict: bool, code: Any = None) -> Any:
    """The training config a checkpoint was written with, resolved in mode ``eval`` over its
    trainer's parser (the async-val / queue watchers' namespace); ``code`` sets the
    watcher's own values (e.g. ``n_envs``) as a ``code:<file>`` layer."""
    from configs.loader.resolve import parser_defaults, resolve
    from configs.loader.schema import GRPOConfig, PPOConfig
    from methods.rl_agent.training import train_grpo, train_ppo

    ckpt, trainer = load_ckpt_layer(ckpt_path, strict=strict)
    # Checkpoints from before the trainer marker (the KDD GRPO runs) name GRPO by its group size.
    grpo = trainer == "grpo" or (trainer is None and "group_size" in ckpt.values)
    parser = (train_grpo if grpo else train_ppo).build_arg_parser()
    policies = train_policies(GRPOConfig() if grpo else PPOConfig())
    defaults = parser_defaults(parser, policies)
    warn_unknown_ckpt_args(ckpt, defaults.values)
    return resolve("eval", policies=policies, defaults=[defaults],
                   ckpt=ckpt, user=[] if code is None else [code])
