#!/usr/bin/env bash
# Public Table 1 RL policy training entrypoint.

set -euo pipefail
source "$(cd "$(dirname "$0")/../../_lib" && pwd)/env.sh"
source "$(cd "$(dirname "$0")" && pwd)/cases.sh"
cd "$CADAGENT_REPO_ROOT"

METHOD="${METHOD:-ppo_per_step}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
ITERATIONS="${ITERATIONS:-}"
RUN_NAME="${RUN_NAME:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SPLIT_JSON="${SPLIT_JSON:-}"
DRC_CONFIG_PATH="${DRC_CONFIG_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOCAL_OUT}/training_logs/table1_synth2l_t3a}"
WANDB_PROJECT="${WANDB_PROJECT:-pcbworld}"
WANDB_GROUP="${WANDB_GROUP:-table1_policy_repro}"

usage() {
  cat <<'EOF'
Usage: train_policy.sh --method ppo_per_step|ppo_terminal|grpo [--seed N]
                       [--gpu N] [--iterations N] [--run-name NAME]
                       [--split-json PATH] [--output-root DIR]
                       [--dry-run] [--smoke]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --split-json) SPLIT_JSON="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

paper_repro_load_table1_policy_case "$METHOD"
METHOD="$TABLE1_METHOD"  # canonical (legacy aliases normalized in cases.sh)

ITERATIONS="${ITERATIONS:-$TABLE1_ITERATIONS}"
SPLIT_JSON="${SPLIT_JSON:-$TABLE1_SPLIT_JSON_DEFAULT}"
DRC_CONFIG_PATH="${DRC_CONFIG_PATH:-$TABLE1_DRC_CONFIG_DEFAULT}"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="table1_${METHOD}_seed${SEED}_repro"
fi

N_EPOCHS="$TABLE1_N_EPOCHS"
BATCH_SIZE="$TABLE1_BATCH_SIZE"
MAX_STEPS="$TABLE1_MAX_STEPS"
EVAL_EVERY="$TABLE1_EVAL_EVERY"
EVAL_N_ROLLOUTS="$TABLE1_EVAL_N_ROLLOUTS"
EVAL_BOARD_LIMIT=0
SAVE_FREQ="$TABLE1_SAVE_FREQ"
EXTRA_ARGS=(--n-envs "$TABLE1_N_ENVS")
if [[ -n "$TABLE1_N_STEPS" ]]; then
  EXTRA_ARGS+=(--n-steps "$TABLE1_N_STEPS" --gamma "$TABLE1_GAMMA" --gae-lambda "$TABLE1_GAE_LAMBDA" --vf-coef "$TABLE1_VF_COEF")
fi
if [[ -n "$TABLE1_GROUP_SIZE" ]]; then
  EXTRA_ARGS+=(--group-size "$TABLE1_GROUP_SIZE")
fi
if [[ "$SMOKE" == "1" ]]; then
  ITERATIONS=1
  MAX_STEPS=16
  EVAL_EVERY=1
  EVAL_N_ROLLOUTS=1
  EVAL_BOARD_LIMIT=4
  SAVE_FREQ=1
  EXTRA_ARGS=(--n-envs 1 --n-steps 16 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5)
  if [[ "$METHOD" == "grpo" ]]; then
    EXTRA_ARGS=(--n-envs 1 --group-size 1)
  fi
fi

mkdir -p "${OUTPUT_ROOT}/${METHOD}/tb" "${OUTPUT_ROOT}/${METHOD}/checkpoints"
public_wandb_args "$RUN_NAME" "$WANDB_GROUP" "$WANDB_PROJECT" "paper_repro,table1,${METHOD}"

cmd=(
  "$PYTHON_BIN" -u -m "$TABLE1_TRAIN_MODULE"
  --board tests/fixtures/simple_routing_board.kicad_pcb
  --boards-order per_env_epoch
  --boards-json "$SPLIT_JSON"
  --use-yaml-drc-fallback
  --drc-config-path "$DRC_CONFIG_PATH"
  --boards-difficulty easy
  --boards-split train
  --eval-split val
  --eval-n-rollouts "$EVAL_N_ROLLOUTS"
  --eval-every "$EVAL_EVERY"
  --iterations "$ITERATIONS"
  --max-steps "$MAX_STEPS"
  --n-epochs "$N_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$TABLE1_LR"
  --entropy-coef "$TABLE1_ENTROPY_COEF"
  --max-grad-norm "$TABLE1_MAX_GRAD_NORM"
  --warmup-iters "$TABLE1_WARMUP_ITERS"
  --reward-rule "$TABLE1_REWARD_RULE"
  --wirelength-penalty "$TABLE1_WIRELENGTH_PENALTY"
  --via-penalty "$TABLE1_VIA_PENALTY"
  --policy-net-select
  --no-drc-tokens
  --wire-via-emission "$TABLE1_WIRE_VIA_EMISSION"
  --masking-rule "$TABLE1_MASKING_RULE"
  --same-net-bias
  --disable-slot-emb
  --coord-encoding fourier
  --corner-mode "$TABLE1_CORNER_MODE"
  --d-model "$TABLE1_D_MODEL"
  --n-heads "$TABLE1_N_HEADS"
  --n-layers "$TABLE1_N_LAYERS"
  --d-ff "$TABLE1_D_FF"
  --device cuda
  --seed "$SEED"
  --log-dir "${OUTPUT_ROOT}/${METHOD}/tb"
  --save-dir "${OUTPUT_ROOT}/${METHOD}/checkpoints"
  --save-freq "$SAVE_FREQ"
  "${EXTRA_ARGS[@]}"
  "${PUBLIC_WANDB_ARGS[@]}"
)

if [[ "$EVAL_BOARD_LIMIT" != "0" ]]; then
  cmd+=(--eval-board-limit "$EVAL_BOARD_LIMIT")
fi

public_cuda_run "${cmd[@]}"
