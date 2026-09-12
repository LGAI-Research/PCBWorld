# Paper-era training knobs — the KDD recipes pass these explicitly because the repository's
# shipped defaults (configs/pcbworld.yaml) moved on after the paper: obstacle/shape tokens,
# per-net constraint channels, arc outlines with simplification, the multi-resolution directional
# ring with the off-board mask, the sin_remaining time feature, and PPO truncation bootstrap off.
# The values below are what the paper checkpoints were trained with (the same table as
# legacy_ckpt_defaults.yaml, plus the bootstrap that was on by default then). Source this file
# and append the arrays to the trainer command; PAPER_PPO_FLAGS is for the PPO trainer only.
PAPER_ENV_FLAGS=(
  --no-obstacle-obs --no-shape-obs --no-net-constraint-obs
  --outline-obs tess --no-simplify-outline
  --directional-candidates none --no-offboard-mask
  --time-feature step_ratio
)
PAPER_PPO_FLAGS=(--truncation-bootstrap)
