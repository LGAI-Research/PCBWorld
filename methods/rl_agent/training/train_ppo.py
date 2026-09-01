"""PPO training entrypoint for the Decoder-Only Transformer PCB policy.

Thin CLI wrapper: builds the argument parser and delegates the training loop to
:class:`methods.rl_agent.training.loop.PPOTrainer` (shared core in
:class:`methods.rl_agent.training.loop.RLTrainer` / :class:`methods._shared.trainer.base.Trainer`).
The standard SB3-style clipped-objective PPO with a critic head, GAE, and
truncation bootstrap now lives in the trainer; see those classes for the loop.

Usage::

    python -m methods.rl_agent.training.train_ppo \\
        --board <path> --iterations 1000 --n-envs 4 --n-steps 256

    tensorboard --logdir ./logs/tb/ppo_decoder
"""
from __future__ import annotations

import argparse
import logging

from methods.rl_agent.training.args import add_shared_args


def build_arg_parser() -> argparse.ArgumentParser:
    from configs.loader.schema import PPOConfig

    _PPO = PPOConfig()
    p = argparse.ArgumentParser(
        description="PPO training for the decoder-only PCB routing policy",
    )
    add_shared_args(
        p,
        n_epochs_default=_PPO.n_epochs,
        log_dir_default=_PPO.log_dir,
        save_dir_default=_PPO.save_dir,
    )

    # --- PPO-specific rollout + GAE (generated from PPOConfig; n_epochs/
    # log_dir/save_dir are cli_skip — fed to add_shared_args above) ---
    from configs.loader.cli import add_dataclass_args
    add_dataclass_args(p, _PPO, style="dash")

    p.add_argument("--no-normalize-adv", action="store_true", default=False,
                   help="Disable per-minibatch advantage normalization (SB3 default is on)")
    p.add_argument("--no-truncation-bootstrap", action="store_true", default=False,
                   help="Disable V(s_next) bootstrap at truncation boundaries "
                        "(legacy: treat truncation as termination with V=0)")

    # Reward normalization (SB3 VecNormalize-equivalent — on by default,
    # matching MLP `train.py` which uses VecNormalize(norm_reward=True)).
    p.add_argument("--no-norm-reward", action="store_true", default=False,
                   help="Disable SB3-style discounted-return reward normalization")

    p.add_argument("--detach-critic", action="store_true", default=False,
                   help="Stop gradient from value loss to backbone "
                        "(critic learns on frozen backbone features)")

    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (.pt) to resume training from. "
                        "Loads policy weights, optimizer state, and continues "
                        "from the saved iteration counter.")

    p.add_argument("--dump-config-only", action="store_true", default=False,
                   help="Write <save-dir>/config_resolved.yaml (+ sorted stdout "
                        "echo) and exit 0 without building the env/model/engine. "
                        "Batch preflight enabler — works without a GPU or "
                        "kicad_rl_router (see tools/experiments/preflight_diff.py).")

    p.add_argument("--disable-slot-emb", action="store_true",
                   help="Ablation: zero out the slot embedding contribution "
                        "(no per-net fingerprint added to token embeddings). "
                        "embed_ln is still applied.")
    p.add_argument("--same-net-bias", action="store_true", default=False,
                   help="Add per-head learnable additive logit bias for "
                        "same-net token pairs (alpha_h * 1[slot_i == slot_j]). "
                        "alpha init 0 (ReZero-style) -> initial forward is "
                        "bit-identical to no-bias SDPA. Combines with or "
                        "replaces input-side slot_emb (use --disable-slot-emb "
                        "to disable the latter).")
    p.add_argument("--bf16", action="store_true", default=False,
                   help="Autocast (bfloat16) only the transformer body "
                        "(stack+incremental decode); logits/loss/optimizer stay "
                        "fp32. A/B-validated: equivalent convergence, ~2x faster "
                        "update, much lower peak VRAM. Wired via "
                        "KiCadRLModel.configure_speed.")
    p.add_argument("--compile-regions", default="",
                   help="Comma list of torch.compile regions {stack,decode,heads} "
                        "or alias 'efficient' (=stack,decode,heads — the adopted "
                        "combo; recommended with --bf16). Default empty = off. "
                        "cudagraphs-family modes are incompatible with the K/V "
                        "cache — see the configure_speed docstring.")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune",
                            "max-autotune-no-cudagraphs"],
                   help="torch.compile mode ('default' recommended — best in "
                        "measurements).")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flex"],
                   help="State-pass attention kernel. 'flex' = flex_attention "
                        "over a key-padding BlockMask (skips all-padding key "
                        "blocks; ~1.2-1.5x on the transformer body, best with "
                        "--bf16 --compile-regions efficient). Not bit-identical "
                        "to sdpa (bf16-level kernel differences). Wired via "
                        "KiCadRLModel.configure_speed.")
    p.add_argument("--update-gpus", type=int, default=1,
                   help="Rank-shard only the PPO update phase across N GPUs "
                        "(default 1 = current single-GPU path unchanged). "
                        "rollout/eval/logging stay on the main process (cuda:0) "
                        "alone; the N-1 workers (cuda:1..) do only updates. One "
                        "manual grad allreduce per minibatch keeps it numerically "
                        "equivalent to single-GPU (tests/test_ddp_equivalence.py). "
                        "Incompatible with --mem-budget (phase 1).")
    p.add_argument("--max-wallclock-sec", type=float, default=None,
                   help="Stop training after the iteration that first crosses "
                        "this wallclock budget (seconds). policy_last.pt is "
                        "saved every iteration, so the latest policy is always "
                        "preserved. None (default) = run all --iterations.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.WARNING)

    # Resolved-config provenance dump: BEFORE any env/model/engine build so the
    # file survives a crash. vars(args) is the same dict ckpts store as
    # payload["args"] (RLTrainer._train_ckpt_payload) — the two cannot diverge.
    from methods._shared.config_dump import (
        dump_resolved_config,
        resume_start_iter,
    )

    start_iter = resume_start_iter(args.resume) if args.resume else None
    dump_resolved_config(vars(args), args.save_dir, start_iter=start_iter)
    if args.dump_config_only:
        return

    from methods.rl_agent.training.loop import PPOTrainer

    PPOTrainer(args).fit()


if __name__ == "__main__":
    main()
