#!/usr/bin/env bash
# Public D1 KiCad-API Transformer PPO training entrypoint.

set -euo pipefail
source "$(cd "$(dirname "$0")/../../_lib" && pwd)/env.sh"
source "$(cd "$(dirname "$0")" && pwd)/cases.sh"
cd "$CADAGENT_REPO_ROOT"

GRID_SIZE="${GRID_SIZE:-10}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-}"
ITERATIONS="${ITERATIONS:-}"
GPU="${GPU:-0}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SPLIT_JSON="${SPLIT_JSON:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOCAL_OUT}/training_logs/d1_grid/transformer_ppo}"
WANDB_PROJECT="${WANDB_PROJECT:-pcbworld}"
WANDB_GROUP="${WANDB_GROUP:-t1_transformer_ppo_repro}"

usage() {
  cat <<'EOF'
Usage: run_1l_grid_transformer_ppo.sh [--grid-size N] [--seed N] [--run-name NAME]
                                      [--iterations N] [--gpu N] [--split-json PATH]
                                      [--output-root DIR] [--dry-run] [--smoke]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grid-size) GRID_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --split-json) SPLIT_JSON="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="v56_l1_transformer_grid${GRID_SIZE}_seed${SEED}_repro"
fi

SPLIT_PATTERN=""
if [[ -z "$SPLIT_JSON" ]]; then
  seed_idx=$((SEED - 42))
  if (( seed_idx < 0 )); then
    seed_idx=0
  fi
  printf -v seed_ver '%02d' "$seed_idx"
  SPLIT_PATTERN="${DATASET_ROOT}/synthetic/splits/synth_1L_grid${GRID_SIZE}_*v${seed_ver}_local.json"
  SPLIT_JSON="$(find "${DATASET_ROOT}/synthetic/splits" \
    -name "synth_1L_grid${GRID_SIZE}_*v${seed_ver}_local.json" 2>/dev/null | sort | head -n1 || true)"
fi
# Preflight: the D1 split JSON follows the gitignored `_local` naming convention
# and is not distributed here.
if [[ -z "$SPLIT_JSON" || ! -f "$SPLIT_JSON" ]]; then
  d1_absent "D1 split JSON for grid=${GRID_SIZE} seed=${SEED} (or pass --split-json)" \
            "${SPLIT_JSON:-$SPLIT_PATTERN}"
  d1_preflight
fi

paper_repro_load_d1_grid_case "$GRID_SIZE"
ITERATIONS="${ITERATIONS:-$T1_TRANSFORMER_ITERATIONS}"
N_ENVS="$T1_TRANSFORMER_N_ENVS"
N_STEPS="$T1_TRANSFORMER_N_STEPS"
MAX_STEPS="$T1_MAX_STEPS"
EVAL_EVERY="$T1_TRANSFORMER_EVAL_EVERY"
EVAL_N_ROLLOUTS="$T1_TRANSFORMER_EVAL_N_ROLLOUTS"
SAVE_FREQ="$T1_TRANSFORMER_SAVE_FREQ"
if [[ "$SMOKE" == "1" ]]; then
  ITERATIONS=1
  N_ENVS=1
  N_STEPS=16
  MAX_STEPS=16
  EVAL_EVERY=1
  EVAL_N_ROLLOUTS=1
  SAVE_FREQ=1
fi

mkdir -p "${OUTPUT_ROOT}/tb" "${OUTPUT_ROOT}/checkpoints"
public_wandb_args "$RUN_NAME" "$WANDB_GROUP" "$WANDB_PROJECT" "paper_repro,d1,transformer_ppo"

cmd=(
  "$PYTHON_BIN" -u -m methods.rl_agent.training.train_ppo
  --board tests/fixtures/simple_routing_board.kicad_pcb
  --boards-order per_env_epoch
  --boards-json "$SPLIT_JSON"
  --boards-difficulty easy
  --boards-split train
  --eval-split val
  --eval-n-rollouts "$EVAL_N_ROLLOUTS"
  --eval-every "$EVAL_EVERY"
  --iterations "$ITERATIONS"
  --max-steps "$MAX_STEPS"
  --n-envs "$N_ENVS"
  --n-steps "$N_STEPS"
  --n-epochs 4
  --batch-size "$T1_TRANSFORMER_BATCH_SIZE"
  --lr "$T1_TRANSFORMER_LR"
  --entropy-coef "$T1_TRANSFORMER_ENTROPY_COEF"
  --gamma "$T1_TRANSFORMER_GAMMA"
  --gae-lambda "$T1_TRANSFORMER_GAE_LAMBDA"
  --vf-coef "$T1_TRANSFORMER_VF_COEF"
  --max-grad-norm "$T1_TRANSFORMER_MAX_GRAD_NORM"
  --warmup-iters "$T1_TRANSFORMER_WARMUP_ITERS"
  --reward-rule "$T1_TRANSFORMER_REWARD_RULE"
  --reward-step-penalty "$T1_TRANSFORMER_REWARD_STEP_PENALTY"
  --drc-penalty "$T1_TRANSFORMER_DRC_PENALTY"
  --wirelength-penalty "$T1_TRANSFORMER_WIRELENGTH_PENALTY"
  --via-penalty "$T1_TRANSFORMER_VIA_PENALTY"
  --policy-net-select
  --no-drc-tokens
  --masking-rule "$T1_TRANSFORMER_MASKING_RULE"
  --same-net-bias
  --disable-slot-emb
  --coord-encoding fourier
  --corner-mode "$T1_TRANSFORMER_CORNER_MODE"
  --directional-candidates "grid${GRID_SIZE}"
  --d-model "$T1_TRANSFORMER_D_MODEL"
  --n-heads "$T1_TRANSFORMER_N_HEADS"
  --n-layers "$T1_TRANSFORMER_N_LAYERS"
  --d-ff "$T1_TRANSFORMER_D_FF"
  --device cuda
  --seed "$SEED"
  --log-dir "${OUTPUT_ROOT}/tb"
  --save-dir "${OUTPUT_ROOT}/checkpoints"
  --save-freq "$SAVE_FREQ"
  "${PUBLIC_WANDB_ARGS[@]}"
)

public_cuda_run "${cmd[@]}"
