"""Shared argparse builder for both PPO and GRPO decoder trainers.

The flag list is *derived from the schema* (:mod:`configs.loader.schema`) via
:func:`configs.loader.cli.add_dataclass_args`, so the dataclasses are the single visible
list of every training variable and the CLI cannot drift from them:

    * :class:`RLTrainConfig`  → shared loop / optim / logging / board / inline-eval
      cadence / W&B knobs (defaults = configs/defaults/rl_train.yaml).
    * :class:`RLEnvConfig`    → env-core + reward overrides + wrapper knobs +
      per-episode augmentation (nests :class:`EnvConfig` → :class:`RewardOverrides`).
    * :class:`RLPolicyConfig` → network architecture.

A small tail adds the handful of flags whose *variables* live in the schema
(``cli_skip``) but whose CLI *form* is bespoke — inverted bools
(``--no-drc-tokens`` / ``--no-mask-start-point`` / ``--no-vecenv``), the
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

from argparse import ArgumentParser


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
    configs/defaults/{env,rl_policy,rl_train}.yaml) so the YAML is the single
    place to edit them and training stays in lock-step with the eval ckpt
    fallback.
    """
    from configs.loader.cli import add_dataclass_args
    from configs.loader.schema import RLEnvConfig, RLPolicyConfig, RLTrainConfig

    _RLENV = RLEnvConfig()
    _P = RLPolicyConfig()
    _T = RLTrainConfig()

    # --- Shared loop / optim / logging / board / inline-eval cadence / W&B ---
    # Generated from RLTrainConfig (wandb / vecenv / resume are cli_skip — the
    # bespoke/typed forms are added below / by the entrypoints).
    add_dataclass_args(p, _T, style="dash")

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
    p.add_argument("--policy-net-select", action="store_true", default=False,
                   help="Learn net ordering: the policy's net_select pointer is "
                        "propagated to the env and contributes to log-prob / "
                        "entropy. Invalid (already-routed) nets are masked out. "
                        "Default (off) uses the legacy random env-driven pick.")
    p.add_argument("--legacy-edge-encoding", action="store_true", default=False,
                   help="Build the tokenizer with 2-point (legacy) edge tokens — "
                        "no edge_mid_proj arc-midpoint module. Needed to "
                        "resume/finetune a pre-arc (2-point) checkpoint; the eval "
                        "loader auto-detects this from the state_dict, but the "
                        "training model-build path must be told explicitly.")
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

    # --- Optimization (PPO-style clipped surrogate used by both algos) ---
    p.add_argument("--n-epochs", type=int, default=n_epochs_default,
                   help="Epochs per training iteration")

    # --- Network architecture (generated from RLPolicyConfig) ---
    # Numeric/encoding fields auto-emitted; the bool toggles + use_critic/legacy_pad
    # are cli_skip (trainer-set or added by the entrypoints).
    add_dataclass_args(p, _P, style="dash")

    # --- Logging / checkpointing (per-algorithm dirs, parameterized) ---
    p.add_argument("--log-dir", default=log_dir_default)
    p.add_argument("--save-dir", default=save_dir_default)

    # --- W&B master toggle (opt-in; wandb_* string knobs generated above) ---
    p.add_argument("--wandb", action="store_true",
                   help="Enable W&B logging (telemetry is OFF by default). "
                        "Needs a credential (WANDB_API_KEY or `wandb login`); "
                        "without one the run warns loudly and logs to "
                        "TensorBoard only. TensorBoard always writes regardless.")

    # --- Greedy val toggle (inverted; schema RLTrainConfig.eval_greedy) ---
    p.add_argument("--no-eval-greedy", action="store_true",
                   help="Skip the greedy (argmax) 1-rollout val_greedy/* pass "
                        "that runs alongside the sampled val/* each cadence.")
