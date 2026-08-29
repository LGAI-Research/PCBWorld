#!/usr/bin/env bash
# Canonical public settings for Figure 5 D1 grid-scalability runs, plus the
# missing-prerequisite preflight every D1 entrypoint runs before doing work.

set -euo pipefail

# --- Missing-prerequisite preflight -----------------------------------------
# The D1 corpus is not distributed with this repository and no shipped generator
# produces it, so each entrypoint records the inputs it cannot find and reports
# them all at once instead of failing deep inside a trainer.

D1_MISSING=()

# Records "<label>: <path>" when <path> does not exist.
d1_need() {
  local label="$1" path="$2"
  [[ -e "$path" ]] || D1_MISSING+=("${label}: ${path}")
}

# Records "<label>: <path>" unconditionally — for an input already known to be
# absent, e.g. a glob that matched nothing.
d1_absent() {
  D1_MISSING+=("${1}: ${2}")
}

# Prints every recorded absence as one notice and exits 2. No-op when nothing
# was recorded.
d1_preflight() {
  (( ${#D1_MISSING[@]} )) || return 0
  {
    echo "figure5_d1: a required D1 input is absent — nothing was run."
    printf '  missing  %s\n' "${D1_MISSING[@]}"
    echo
    echo "D1 (paper Figure 5) is the synthetic 1-layer grid sweep. Its corpus is"
    echo "NOT distributed with this repository and no generator here reproduces"
    echo "it, so Figure 5 cannot be reproduced from a fresh clone. To run this"
    echo "script, supply the paths above yourself (DATASET_ROOT=... or the"
    echo "matching command-line flag)."
    echo "Details: experiments/kdd/figure5_d1/README.md"
    echo "Dataset conventions: configs/datasets/README.md (row d1)"
  } >&2
  exit 2
}

paper_repro_t1_step_penalty_for_grid() {
  case "$1" in
    10) printf '0.03' ;;
    50) printf '0.006' ;;
    100) printf '0.003' ;;
    200) printf '0.0015' ;;
    500) printf '0.0006' ;;
    1000) printf '0.0003' ;;
    *) echo "unsupported D1 grid size: $1" >&2; return 2 ;;
  esac
}

paper_repro_load_d1_grid_case() {
  local grid="$1"
  case "$grid" in
    10|50|100|200|500) ;;
    *) echo "unsupported public D1 grid size: $grid" >&2; return 2 ;;
  esac

  T1_GRID_SIZE="$grid"
  T1_SEEDS_DEFAULT="42 43 44 45"
  T1_BASELINE_SEEDS_DEFAULT="42 43"
  T1_MAX_STEPS=256
  T1_CONNECTOR_STEP_PENALTY="$(paper_repro_t1_step_penalty_for_grid "$grid")"

  # KiCad-API Transformer PPO settings saved in staged policy_best.pt.
  T1_TRANSFORMER_REWARD_RULE="jumanji_connector_wirelength_dense"
  T1_TRANSFORMER_REWARD_STEP_PENALTY="0"
  T1_TRANSFORMER_WIRELENGTH_PENALTY="0.003"
  T1_TRANSFORMER_VIA_PENALTY="0"
  T1_TRANSFORMER_DRC_PENALTY="0"
  T1_TRANSFORMER_N_ENVS=32
  T1_TRANSFORMER_N_STEPS=512
  T1_TRANSFORMER_ITERATIONS=300
  T1_TRANSFORMER_EVAL_EVERY=20
  T1_TRANSFORMER_EVAL_N_ROLLOUTS=10
  T1_TRANSFORMER_SAVE_FREQ=10
  T1_TRANSFORMER_BATCH_SIZE=256
  T1_TRANSFORMER_LR="1e-4"
  T1_TRANSFORMER_ENTROPY_COEF="0.01"
  T1_TRANSFORMER_GAMMA="0.995"
  T1_TRANSFORMER_GAE_LAMBDA="0.95"
  T1_TRANSFORMER_VF_COEF="0.5"
  T1_TRANSFORMER_MAX_GRAD_NORM="0.5"
  T1_TRANSFORMER_WARMUP_ITERS=20
  T1_TRANSFORMER_D_MODEL=128
  T1_TRANSFORMER_N_HEADS=8
  T1_TRANSFORMER_N_LAYERS=4
  T1_TRANSFORMER_D_FF=512
  T1_TRANSFORMER_MASKING_RULE="default_no_via"
  T1_TRANSFORMER_CORNER_MODE=90

  # Jumanji/SABLE Connector-v2 baseline settings.
  T1_JUMANJI_NUM_EPOCHS=3500
  T1_JUMANJI_NUM_LEARNER_STEPS=100
  T1_JUMANJI_N_STEPS=10
  T1_JUMANJI_TOTAL_BATCH_SIZE=256
  T1_JUMANJI_LR="2e-4"
  T1_JUMANJI_ENTROPY_COEF="0.01"
  T1_JUMANJI_EVAL_EVERY=50
  T1_JUMANJI_SAVE_FREQ=50

  T1_SABLE_NUM_UPDATES=18000
  T1_SABLE_ROLLOUT_LENGTH=128
  T1_SABLE_NUM_ENVS=16
  T1_SABLE_UPDATE_BATCH_SIZE=2
  T1_SABLE_NUM_MINIBATCHES=2
  T1_SABLE_NUM_EVALUATION=32
}
