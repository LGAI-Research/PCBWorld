#!/usr/bin/env bash
# Public D1 SABLE/Mava training entrypoint.

set -euo pipefail
source "$(cd "$(dirname "$0")/../../_lib" && pwd)/env.sh"
source "$(cd "$(dirname "$0")" && pwd)/cases.sh"
cd "$CADAGENT_REPO_ROOT"

GRID_SIZE="${GRID_SIZE:-10}"
SEED="${SEED:-42}"
TRAIN_NPZ="${TRAIN_NPZ:-${DATASET_ROOT}/synthetic/connector_v2/grid${GRID_SIZE}/train.npz}"
EVAL_NPZ="${EVAL_NPZ:-${DATASET_ROOT}/synthetic/connector_v2/grid${GRID_SIZE}/val.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_OUT}/training_logs/d1_grid/sable/grid${GRID_SIZE}/seed${SEED}}"
RUN_NAME="${RUN_NAME:-v56_sable_grid${GRID_SIZE}_seed${SEED}_repro}"
MAVA_SRC="${MAVA_SRC:-${EXPR_ROOT}/mava_source}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
PY="${BASELINE_PYTHON:-${T1_BASELINE_PYTHON:-$PYTHON_BIN}}"

usage() {
  cat <<'EOF'
Usage: train_sable.sh [--grid-size N] [--seed N] [--train-npz PATH]
                      [--eval-npz PATH] [--output-dir DIR] [--mava-src DIR]
                      [--dry-run] [--smoke]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grid-size) GRID_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --train-npz) TRAIN_NPZ="$2"; shift 2 ;;
    --eval-npz) EVAL_NPZ="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mava-src) MAVA_SRC="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

paper_repro_load_d1_grid_case "$GRID_SIZE"
STEP_PENALTY="$T1_CONNECTOR_STEP_PENALTY"
NUM_UPDATES="$T1_SABLE_NUM_UPDATES"
NUM_EVAL="$T1_SABLE_NUM_EVALUATION"
if [[ "$SMOKE" == "1" ]]; then
  NUM_UPDATES=2
  NUM_EVAL=1
fi

cmd=(
  # TODO(Phase 2): scripts/run_v56_mava_sable.py + scripts/v56_connector_fixed.py were
  # pruned from this branch; restore from `develop` (see experiments/README.md) before a real run.
  "$PY" scripts/run_v56_mava_sable.py
  --train-npz "$TRAIN_NPZ"
  --eval-npz "$EVAL_NPZ"
  --out-json "${OUTPUT_DIR}/metrics.json"
  --run-name "$RUN_NAME"
  --seed "$SEED"
  --step-penalty "$STEP_PENALTY"
  --time-limit "$T1_MAX_STEPS"
  --num-updates "$NUM_UPDATES"
  --rollout-length "$T1_SABLE_ROLLOUT_LENGTH"
  --num-envs "$T1_SABLE_NUM_ENVS"
  --update-batch-size "$T1_SABLE_UPDATE_BATCH_SIZE"
  --num-minibatches "$T1_SABLE_NUM_MINIBATCHES"
  --num-evaluation "$NUM_EVAL"
  --mava-src "$MAVA_SRC"
)
if [[ "$SMOKE" != "1" ]]; then
  cmd+=(--save-checkpoint)
fi

public_log "$(printf '%q ' "${cmd[@]}")"
if [[ "$DRY_RUN" != "1" ]]; then
  "${cmd[@]}"
fi
