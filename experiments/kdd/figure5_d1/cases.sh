#!/usr/bin/env bash
# Canonical public settings for Figure 5 D1 grid-scalability runs.

set -euo pipefail

# --- D1 corpus: where a locally generated one lives -------------------------
# The paper corpus (synth_1L/grid<G>_5net_v15 under a staged DATASET_ROOT) is not
# distributed. `make_grid_dataset.sh <G>` builds a D1-shaped grid corpus instead,
# and these three name exactly what it writes — the stage defaults and the
# preflight notice both read them, so the recipe and the generator cannot drift.
d1_generate_cmd() { printf 'bash tools/datagen/synthetic_generator/make_grid_dataset.sh %s' "$1"; }
d1_generated_split_json() {
  printf '%s/experiments/kdd/configs/datasets/grids/10net_2pin_1layer_grid%s_v2.json' \
    "$PCBWORLD_REPO_ROOT" "$1"
}
d1_generated_boards_dir() {
  printf '%s/var/datasets/synthetic/pcb_dataset_synthetic_10net_2pin_1layer_grid%s_test' \
    "$PCBWORLD_REPO_ROOT" "$1"
}

# --- D1 input guidance ------------------------------------------------------
# d1_absent records one missing input (a sweep reports every absent cell instead
# of dying on the first); d1_preflight prints how to produce them and exits 2.
D1_MISSING=()

d1_absent() {   # <what> <path it was looked for at>
  D1_MISSING+=("$1 -> $2")
  printf '[d1] absent: %s\n[d1]   looked for: %s\n' "$1" "$2" >&2
}

d1_preflight() {
  local grids="${L1_GRIDS:-10 50 100 200 500}" g
  {
    printf '\n[d1] nothing to run — the D1 corpus is not distributed with this repository.\n'
    printf '[d1] build a D1-shaped grid corpus (train+val+test boards + split JSON):\n'
    for g in $grids; do printf '[d1]   %s\n' "$(d1_generate_cmd "$g")"; done
    printf '[d1] then re-run this stage — train reads %s\n' "$(d1_generated_split_json '<G>')"
    printf '[d1] and eval reads %s\n' "$(d1_generated_boards_dir '<G>')"
    printf '[d1] To score the paper corpus instead, point the stage at it:\n'
    printf '[d1]   BOARDS_DIR=... CKPT=... run.sh eval   |   train_transformer_ppo.sh --split-json ...\n'
    printf '[d1] Boards generated this way are NOT the paper D1 corpus (that one is 5 nets x 2 pins,\n'
    printf '[d1] seeds unpinned here) — they reproduce the protocol, not the published numbers.\n'
    printf '[d1] Details: experiments/kdd/figure5_d1/README.md\n\n'
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
