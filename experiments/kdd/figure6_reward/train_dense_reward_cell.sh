#!/usr/bin/env bash
# Public Figure 6 dense reward ablation single-cell training entrypoint.

set -euo pipefail
source "$(cd "$(dirname "$0")/../../_lib" && pwd)/env.sh"
source "$(cd "$(dirname "$0")" && pwd)/cases.sh"
cd "$CADAGENT_REPO_ROOT"

WIRE_PENALTY="${WIRE_PENALTY:-0.002}"
VIA_PENALTY="${VIA_PENALTY:-0.1}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
ITERATIONS="${ITERATIONS:-}"
RUN_NAME="${RUN_NAME:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SPLIT_JSON="${SPLIT_JSON:-}"
DRC_CONFIG_PATH="${DRC_CONFIG_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOCAL_OUT}/training_logs/reward_ablation/training}"
WANDB_PROJECT="${WANDB_PROJECT:-pcbworld}"
WANDB_GROUP="${WANDB_GROUP:-figure6_dense_reward_repro}"

usage() {
  cat <<'EOF'
Usage: train_dense_reward_cell.sh [--wirelength-penalty X] [--via-penalty X]
                                  [--seed N] [--gpu N] [--iterations N]
                                  [--run-name NAME] [--dry-run] [--smoke]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wirelength-penalty) WIRE_PENALTY="$2"; shift 2 ;;
    --via-penalty) VIA_PENALTY="$2"; shift 2 ;;
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

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="figure6_dense_wire${WIRE_PENALTY}_via${VIA_PENALTY}_seed${SEED}_repro"
fi

paper_repro_load_figure6_reward_case
ITERATIONS="${ITERATIONS:-$FIGURE6_ITERATIONS}"
SPLIT_JSON="${SPLIT_JSON:-$FIGURE6_SPLIT_JSON_DEFAULT}"
DRC_CONFIG_PATH="${DRC_CONFIG_PATH:-$FIGURE6_DRC_CONFIG_DEFAULT}"
N_ENVS="$FIGURE6_N_ENVS"
N_STEPS="$FIGURE6_N_STEPS"
MAX_STEPS="$FIGURE6_MAX_STEPS"
EVAL_EVERY="$FIGURE6_EVAL_EVERY"
EVAL_N_ROLLOUTS="$FIGURE6_EVAL_N_ROLLOUTS"
EVAL_BOARD_LIMIT=0
SAVE_FREQ="$FIGURE6_SAVE_FREQ"
if [[ "$SMOKE" == "1" ]]; then
  ITERATIONS=1
  N_ENVS=1
  N_STEPS=16
  MAX_STEPS=16
  EVAL_EVERY=1
  EVAL_N_ROLLOUTS=1
  EVAL_BOARD_LIMIT=4
  SAVE_FREQ=1
fi

mkdir -p "${OUTPUT_ROOT}/tb" "${OUTPUT_ROOT}/checkpoints"
public_wandb_args "$RUN_NAME" "$WANDB_GROUP" "$WANDB_PROJECT" "paper_repro,figure6,reward_ablation"

cmd=(
  "$PYTHON_BIN" -u -m methods.rl_agent.training.train_ppo
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
  --n-envs "$N_ENVS"
  --n-steps "$N_STEPS"
  --n-epochs "$FIGURE6_N_EPOCHS"
  --batch-size "$FIGURE6_BATCH_SIZE"
  --lr "$FIGURE6_LR"
  --entropy-coef "$FIGURE6_ENTROPY_COEF"
  --gamma "$FIGURE6_GAMMA"
  --gae-lambda "$FIGURE6_GAE_LAMBDA"
  --vf-coef "$FIGURE6_VF_COEF"
  --max-grad-norm "$FIGURE6_MAX_GRAD_NORM"
  --warmup-iters "$FIGURE6_WARMUP_ITERS"
  --reward-rule "$FIGURE6_REWARD_RULE"
  --wirelength-penalty "$WIRE_PENALTY"
  --via-penalty "$VIA_PENALTY"
  --policy-net-select
  --no-drc-tokens
  --wire-via-emission per_step
  --masking-rule "$FIGURE6_MASKING_RULE"
  --same-net-bias
  --disable-slot-emb
  --coord-encoding fourier
  --corner-mode "$FIGURE6_CORNER_MODE"
  --d-model "$FIGURE6_D_MODEL"
  --n-heads "$FIGURE6_N_HEADS"
  --n-layers "$FIGURE6_N_LAYERS"
  --d-ff "$FIGURE6_D_FF"
  --device cuda
  --seed "$SEED"
  --log-dir "${OUTPUT_ROOT}/tb"
  --save-dir "${OUTPUT_ROOT}/checkpoints"
  --save-freq "$SAVE_FREQ"
  "${PUBLIC_WANDB_ARGS[@]}"
)

if [[ "$EVAL_BOARD_LIMIT" != "0" ]]; then
  cmd+=(--eval-board-limit "$EVAL_BOARD_LIMIT")
fi

public_cuda_run "${cmd[@]}"
