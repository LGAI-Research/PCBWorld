#!/usr/bin/env bash
# Figure 5 / Figure 6c — D1 grid scalability (PCBWorld PPO vs Jumanji A2C vs SABLE).
#
# Light per-experiment orchestrator. Stages:
#   train [transformer|jumanji|sable]  — grid x seed training sweep -> local trainer shells
#   eval                               — held-out grid test rollout+DRC via eval/pipeline.py
#   figure                             — draw_figure.py --figure fig6c
#   all                                — train(transformer) + eval + figure
#
# Honors DRY_RUN=1 / SMOKE=1 / GPU=N and the L1_GRIDS / SEEDS overrides.
#
# The D1 corpus is not distributed with this repository, so every stage checks
# its inputs first and exits 2 with a notice when they are absent
# (experiments/kdd/figure5_d1/README.md).
set -euo pipefail
_self="$(cd "$(dirname "$0")" && pwd)"
source "$_self/../../_lib/env.sh"
source "$_self/cases.sh"
cd "$CADAGENT_REPO_ROOT"

GRIDS="${L1_GRIDS:-10 50 100 200 500}"
SEEDS="${SEEDS:-42 43 44 45}"
GPU="${GPU:-0}"

train() {
  local method="${1:-transformer}"
  for grid in $GRIDS; do for seed in $SEEDS; do
    # baselines (jumanji/sable) pin the GPU via CUDA_VISIBLE_DEVICES, not a --gpu flag
    local args=(--grid-size "$grid" --seed "$seed")
    [[ "$method" == transformer || "$method" == ppo ]] && args+=(--gpu "$GPU")
    [[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
    [[ "${SMOKE:-0}" == "1" ]] && args+=(--smoke)
    case "$method" in
      transformer|ppo) quickstart_run bash "$_self/train_transformer_ppo.sh" "${args[@]}" ;;
      jumanji)         CUDA_VISIBLE_DEVICES="$GPU" quickstart_run bash "$_self/train_jumanji_a2c.sh" "${args[@]}" ;;
      sable)           CUDA_VISIBLE_DEVICES="$GPU" quickstart_run bash "$_self/train_sable.sh" "${args[@]}" ;;
      *) echo "unknown train method: $method (transformer|jumanji|sable)" >&2; exit 2 ;;
    esac
  done; done
}

# Held-out grid test eval: resolve ckpt/boards under the var/ roots and run the
# canonical 3-stage eval/pipeline.py per (grid, seed). (folded from the former
# eval_transformer_1L.sh; absolute dataset defaults dropped — roots come from _lib/env.sh.)
eval_t1() {
  local n_rollouts="${N_ROLLOUTS:-5}" n_envs="${N_ENVS:-64}" mode="${ROLLOUT_MODE:-parallel}"
  local ran=0
  for grid in $GRIDS; do for seed in $SEEDS; do
    local ckpt="${CKPT:-$CKPT_ROOT/Transformer_1L/grid${grid}/seed${seed}/policy_best.pt}"
    local boards="${BOARDS_DIR:-$DATASET_ROOT/synthetic/synth_1L/grid${grid}_5net_v15/test}"
    local out="$LOCAL_OUT/rollouts/d1_grid/transformer_ppo/grid${grid}_seed${seed}"
    local skip=0
    [[ -f "$ckpt" ]]   || { d1_absent "D1 checkpoint (grid ${grid}, seed ${seed})" "$ckpt"; skip=1; }
    [[ -d "$boards" ]] || { d1_absent "D1 test boards (grid ${grid})" "$boards"; skip=1; }
    if (( skip )); then continue; fi
    ran=$((ran + 1))
    echo "[eval] grid=$grid seed=$seed -> $out"
    [[ "${DRY_RUN:-0}" == "1" ]] && { echo "[eval] --dry-run (eval/pipeline.py)"; continue; }
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval/pipeline.py \
      --ckpt "$ckpt" --boards-dir "$boards" --seed "$seed" \
      --n-rollouts "$n_rollouts" --n-envs "$n_envs" --rollout-mode "$mode" \
      --check-angle 90 --output-dir "$out"
  done; done
  # No cell had both a checkpoint and its boards -> report and fail, rather than
  # exiting 0 having done nothing.
  (( ran )) || d1_preflight
  if (( ${#D1_MISSING[@]} )); then
    printf '[eval] skipped, absent — %s\n' "${D1_MISSING[@]}" >&2
  fi
}

figure() { quickstart_python experiments/draw_figure.py --figure fig6c; }

case "${1:-all}" in
  train)  shift; train "${1:-transformer}" ;;
  eval)   eval_t1 ;;
  figure) figure ;;
  all)    train transformer; eval_t1; figure ;;
  *) echo "usage: run.sh [train [transformer|jumanji|sable] | eval | figure | all]" >&2; exit 2 ;;
esac
