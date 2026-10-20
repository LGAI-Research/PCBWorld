# Paper-era training knobs — the KDD recipes pass these explicitly because the repository's
# shipped defaults (configs/env.yaml, configs/rl.yaml) moved on after the paper: obstacle/shape tokens,
# per-net constraint channels, arc outlines with simplification, the multi-resolution directional
# ring with the off-board mask, the connectivity candidate filter, make_line aborting when stuck,
# the sin_remaining time feature, and PPO truncation bootstrap off.
# The values below are what the paper checkpoints were trained with (the pre-knob values of
# configs/ckpt_migrations.yaml, plus the bootstrap that was on by default then). Source this file
# and append the arrays to the trainer command; PAPER_PPO_FLAGS is for the PPO trainer only.
# PAPER_DIRECTIONAL_FLAGS is the 8-direction ring of the D2-A recipes (Table 1, Figure 6); the
# D1 grid recipe (Figure 5) trains with its own `grid<N>` candidates instead.
PAPER_ENV_FLAGS=(
  --no-obstacle-obs --no-shape-obs --no-net-constraint-obs
  --outline-obs tess --no-simplify-outline
  --no-offboard-mask --no-connectivity-filter --no-reject-if-stuck
  --time-feature step_ratio
)
PAPER_DIRECTIONAL_FLAGS=(--directional-candidates none)
PAPER_PPO_FLAGS=(--truncation-bootstrap)

# Rollout + scoring of the paper's RL cells (the unified-pipeline cells of 2026-05-30/31 behind the
# published numbers: run records eval_overall_summary.json + per_rollout.csv), common to Table 1,
# the reward sweep and D1 (experiments/_lib/env.sh kdd_eval_cell). Rollout: sampled, at most 256
# steps, env seed 5600 + 100*board + r, truncated after 10 finish actions with no progress, 64
# parallel envs. Scoring: post-hoc DRC with the plain drc_dense_promoted rule (errors + the three
# promoted warnings, no wirelength/via term), the per-board rollout of largest final potential.
# Each recipe adds --n-rollouts and --check-angle (45 for the 2-layer sets, 90 for the D1 grids).
# This rule leaves out the training wirelength/via terms (D1: the jumanji reward), so
# eval/pipeline.py warns that it differs from the checkpoint's training reward.
PAPER_ROLLOUT_FLAGS=(
  --seed 5600 --max-steps 256 --early-stop-finish-no-progress 10
  --rollout-mode parallel --n-envs 64 --env-drc off --inline-drc off --save-artifacts on
)
PAPER_SCORING_FLAGS=(
  --rollout-mode parallel --n-envs 64 --reward-config drc_dense_promoted --selection-method final_potential
)
