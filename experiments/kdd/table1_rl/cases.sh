#!/usr/bin/env bash
# Canonical public settings for Table 1 RL policy training rows.

set -euo pipefail

paper_repro_load_table1_policy_case() {
  local method="$1"

  # Legacy method-key aliases (pre paper-alignment naming) -> canonical.
  case "$method" in
    ppo_dense) method="ppo_per_step" ;;
    ppo_episodic) method="ppo_terminal" ;;
  esac

  TABLE1_METHOD="$method"
  TABLE1_SEEDS_DEFAULT="42 43 44 45"
  TABLE1_SPLIT_JSON_DEFAULT="configs/datasets/d2a.json"
  TABLE1_DRC_CONFIG_DEFAULT="configs/drc/synth_2L_v2.yaml"
  TABLE1_MAX_STEPS=256
  TABLE1_N_EPOCHS=4
  TABLE1_BATCH_SIZE=256
  TABLE1_LR="1e-4"
  TABLE1_ENTROPY_COEF="0.01"
  TABLE1_MAX_GRAD_NORM="0.5"
  TABLE1_WARMUP_ITERS=20
  TABLE1_EVAL_EVERY=20
  TABLE1_EVAL_N_ROLLOUTS=10
  TABLE1_SAVE_FREQ=10
  TABLE1_WIRELENGTH_PENALTY="0.002"
  TABLE1_VIA_PENALTY="0.1"
  TABLE1_MASKING_RULE="default"
  TABLE1_CORNER_MODE=45
  TABLE1_D_MODEL=128
  TABLE1_N_HEADS=8
  TABLE1_N_LAYERS=4
  TABLE1_D_FF=512
  TABLE1_WIRE_VIA_EMISSION="per_step"

  case "$method" in
    ppo_per_step)
      TABLE1_TRAIN_MODULE="methods.rl_agent.training.train_ppo"
      TABLE1_REWARD_RULE="drc_dense_promoted"
      TABLE1_ITERATIONS=300
      TABLE1_N_ENVS=32
      TABLE1_N_STEPS=512
      TABLE1_GAMMA="0.995"
      TABLE1_GAE_LAMBDA="0.95"
      TABLE1_VF_COEF="0.5"
      TABLE1_GROUP_SIZE=""
      ;;
    ppo_terminal)
      TABLE1_TRAIN_MODULE="methods.rl_agent.training.train_ppo"
      TABLE1_REWARD_RULE="drc_sparse_promoted_ppo"
      TABLE1_ITERATIONS=300
      TABLE1_N_ENVS=32
      TABLE1_N_STEPS=512
      TABLE1_GAMMA="0.995"
      TABLE1_GAE_LAMBDA="0.95"
      TABLE1_VF_COEF="0.5"
      TABLE1_GROUP_SIZE=""
      ;;
    grpo)
      TABLE1_TRAIN_MODULE="methods.rl_agent.training.train_grpo"
      TABLE1_REWARD_RULE="drc_sparse_promoted_grpo"
      TABLE1_ITERATIONS=1800
      TABLE1_N_ENVS=32
      TABLE1_N_STEPS=""
      TABLE1_GAMMA=""
      TABLE1_GAE_LAMBDA=""
      TABLE1_VF_COEF=""
      TABLE1_GROUP_SIZE=16
      ;;
    *) echo "unknown Table 1 method: $method" >&2; return 2 ;;
  esac
}
