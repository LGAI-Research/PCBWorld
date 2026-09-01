#!/usr/bin/env bash
# Public D1 Jumanji A2C training entrypoint.

set -euo pipefail
source "$(cd "$(dirname "$0")/../../_lib" && pwd)/env.sh"
source "$(cd "$(dirname "$0")" && pwd)/cases.sh"
cd "$CADAGENT_REPO_ROOT"

GRID_SIZE="${GRID_SIZE:-10}"
SEED="${SEED:-42}"
TRAIN_NPZ="${TRAIN_NPZ:-${DATASET_ROOT}/synthetic/connector_v2/grid${GRID_SIZE}/train.npz}"
EVAL_NPZ="${EVAL_NPZ:-${DATASET_ROOT}/synthetic/connector_v2/grid${GRID_SIZE}/val.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_OUT}/training_logs/d1_grid/jumanji_a2c/grid${GRID_SIZE}/seed${SEED}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
RUN_NAME="${RUN_NAME:-v56_jumanji_a2c_grid${GRID_SIZE}_seed${SEED}_repro}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
PY="${BASELINE_PYTHON:-${T1_BASELINE_PYTHON:-$PYTHON_BIN}}"

usage() {
  cat <<'EOF'
Usage: train_jumanji_a2c.sh [--grid-size N] [--seed N] [--train-npz PATH]
                            [--eval-npz PATH] [--output-dir DIR]
                            [--checkpoint-dir DIR] [--dry-run] [--smoke]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grid-size) GRID_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --train-npz) TRAIN_NPZ="$2"; shift 2 ;;
    --eval-npz) EVAL_NPZ="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

paper_repro_load_d1_grid_case "$GRID_SIZE"
STEP_PENALTY="$T1_CONNECTOR_STEP_PENALTY"
NUM_EPOCHS="$T1_JUMANJI_NUM_EPOCHS"
NUM_LEARNER_STEPS="$T1_JUMANJI_NUM_LEARNER_STEPS"
EVAL_EVERY="$T1_JUMANJI_EVAL_EVERY"
SAVE_FREQ="$T1_JUMANJI_SAVE_FREQ"
if [[ "$SMOKE" == "1" ]]; then
  NUM_EPOCHS=1
  NUM_LEARNER_STEPS=1
  EVAL_EVERY=1
  SAVE_FREQ=1
fi

cmd=(
  # TODO(Phase 2): scripts/run_v56_jumanji_a2c.py + scripts/v56_connector_fixed.py were
  # pruned from this branch; restore from `develop` (see experiments/README.md) before a real run.
  "$PY" scripts/run_v56_jumanji_a2c.py
  --train-npz "$TRAIN_NPZ"
  --eval-npz "$EVAL_NPZ"
  --out-json "${OUTPUT_DIR}/metrics.json"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --run-name "$RUN_NAME"
  --seed "$SEED"
  --step-penalty "$STEP_PENALTY"
  --time-limit "$T1_MAX_STEPS"
  --num-epochs "$NUM_EPOCHS"
  --num-learner-steps-per-epoch "$NUM_LEARNER_STEPS"
  --n-steps "$T1_JUMANJI_N_STEPS"
  --total-batch-size "$T1_JUMANJI_TOTAL_BATCH_SIZE"
  --eval-total-batch-size "$T1_JUMANJI_TOTAL_BATCH_SIZE"
  --eval-every "$EVAL_EVERY"
  --save-freq "$SAVE_FREQ"
  --lr "$T1_JUMANJI_LR"
  --entropy-coef "$T1_JUMANJI_ENTROPY_COEF"
  --stochastic-eval
  --auto-resume
)

public_log "$(printf '%q ' "${cmd[@]}")"
if [[ "$DRY_RUN" != "1" ]]; then
  "${cmd[@]}"
fi
