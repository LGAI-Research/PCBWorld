"""GRPO training entrypoint for the decoder-only Transformer PCB policy.

Thin CLI wrapper: builds the argument parser and delegates the training loop to
:class:`methods.rl_agent.training.loop.GRPOTrainer` (shared core in
:class:`methods.rl_agent.training.loop.RLTrainer` / :class:`methods._shared.trainer.base.Trainer`).

Intentional diffs from PPO (now encoded in ``GRPOTrainer``):
    * No critic — group-relative baseline replaces ``V(s)``; ``--group-size``
      slices the ``--n-envs`` pool into sub-groups (``n_envs % group_size == 0``).
    * No fixed n_steps — every iteration runs all envs to completion / max-steps.
    * No gamma / GAE / value loss / reward normalizer.

Usage::

    python -m methods.rl_agent.training.train_grpo \\
        --board <path> --iterations 200 --n-envs 32 --group-size 16

    tensorboard --logdir ./logs/tb/grpo_decoder
"""
from __future__ import annotations

import argparse
import logging

from methods.rl_agent.training.args import add_shared_args


def build_arg_parser() -> argparse.ArgumentParser:
    from configs.loader.schema import GRPOConfig

    _GRPO = GRPOConfig()
    p = argparse.ArgumentParser(
        description="GRPO training for the decoder-only PCB routing policy",
    )
    add_shared_args(
        p,
        n_epochs_default=_GRPO.n_epochs,
        log_dir_default=_GRPO.log_dir,
        save_dir_default=_GRPO.save_dir,
    )

    # --- GRPO-specific rollout (generated from GRPOConfig; n_epochs/log_dir/
    # save_dir are cli_skip — fed to add_shared_args above) ---
    from configs.loader.cli import add_dataclass_args
    add_dataclass_args(p, _GRPO, style="dash")

    # PPO-equivalent architecture knobs (ported so v55-winner config is
    # reachable here too).
    p.add_argument("--disable-slot-emb", action="store_true",
                   help="Ablation: zero out the slot embedding contribution "
                        "(no per-net fingerprint added to token embeddings). "
                        "embed_ln is still applied.")
    p.add_argument("--same-net-bias", action="store_true", default=False,
                   help="Add per-head learnable additive logit bias for "
                        "same-net token pairs (alpha_h * 1[slot_i == slot_j]). "
                        "alpha init 0 (ReZero-style) -> initial forward is "
                        "bit-identical to no-bias SDPA.")

    # --- Resume (preemption-safe) — ported from PPO trainer ---
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (.pt) to resume training from. "
                        "Loads policy weights, optimizer state, and continues "
                        "from the saved iteration counter.")
    return p


def main() -> None:
    from methods.rl_agent.training.loop import GRPOTrainer

    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.WARNING)
    GRPOTrainer(args).fit()


if __name__ == "__main__":
    main()
