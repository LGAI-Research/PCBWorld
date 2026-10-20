#!/usr/bin/env bash
# Table 1 RL rows (D2 / D3) — PCBWorld policy quality (PPO per-step / PPO terminal / GRPO).
# Paper artifacts: Table 3, Table 22, Tables 24/25.
#
# Stages:
#   train   — method x seed training sweep -> train_policy.sh (each seed under $TRAIN_ROOT/seed<S>)
#   eval    — per (task, method): roll out each seed's checkpoint, pool them into one cell
#             $EXPR_ROOT/<task>/<cell> (the paper's cells pool seeds 42-45) and score it with the
#             cells' settings (configs/paper_train_flags.sh PAPER_*_FLAGS + cases.sh TABLE1_PAPER_*)
#   figure  — draw_figure.py for table3 + table22 + table24_25
#   all     — train + eval + figure
#
# Honors DRY_RUN=1 / SMOKE=1 / GPU=N and METHODS / SEEDS / TASKS (d2a d3a d3b) overrides;
# CKPT=<file> scores one checkpoint instead of $TRAIN_ROOT/seed<S>/<method>/checkpoints/policy_best.pt.
set -euo pipefail
_self="$(cd "$(dirname "$0")" && pwd)"
source "$_self/../../_lib/env.sh"
source "$_self/cases.sh"
source "$_self/../configs/paper_train_flags.sh"
cd "$PCBWORLD_REPO_ROOT"

METHODS="${METHODS:-${METHOD:-ppo_per_step ppo_terminal grpo}}"
paper_repro_load_table1_policy_case ppo_per_step
SEEDS="${SEEDS:-$TABLE1_SEEDS_DEFAULT}"
TASKS="${TASKS:-$TABLE1_PAPER_TASKS_DEFAULT}"
GPU="${GPU:-0}"
TRAIN_ROOT="${TRAIN_ROOT:-${LOCAL_OUT}/training_logs/table1_synth2l_t3a}"

train() {
  for method in $METHODS; do for seed in $SEEDS; do
    local args=(--method "$method" --seed "$seed" --gpu "$GPU" --output-root "$TRAIN_ROOT/seed${seed}")
    [[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
    [[ "${SMOKE:-0}" == "1" ]] && args+=(--smoke)
    quickstart_run bash "$_self/train_policy.sh" "${args[@]}"
  done; done
}

# One pooled cell per (task, method) under $EXPR_ROOT/<task>/<cell> — where draw_figure.py
# reads them (var/results/kdd) — from the checkpoints of every seed in SEEDS.
eval_cells() {
  if [[ -n "${CKPT:-}" && ( "$METHODS" == *" "* || "$SEEDS" == *" "* ) ]]; then
    echo "[eval] CKPT=<file> scores one checkpoint: give a single METHODS and SEEDS" >&2
    exit 2
  fi
  local missing=()
  for task in $TASKS; do for method in $METHODS; do
    paper_repro_load_table1_policy_case "$method"
    local ckpts=() absent=() task_dir="$task"
    [[ "$task" == d3* ]] && task_dir="d3/$task"
    for seed in $SEEDS; do
      ckpts+=("${CKPT:-$TRAIN_ROOT/seed${seed}/${TABLE1_METHOD}/checkpoints/policy_best.pt}")
      [[ -f "${ckpts[-1]}" || "${DRY_RUN:-0}" == "1" ]] || absent+=("${ckpts[-1]}")
    done
    if (( ${#absent[@]} )); then missing+=("${absent[@]}"); continue; fi
    local boards="$LOCAL_OUT/rollouts/table1/${task}_boards.txt"
    kdd_task_boards_list "$task" "$boards"
    kdd_eval_cell "$EXPR_ROOT/$task_dir/$TABLE1_CELL" "$boards" "$LOCAL_OUT/rollouts/table1/$task/$TABLE1_CELL" \
      "$TABLE1_CORNER_MODE" "${ckpts[@]}" -- \
      --n-rollouts "$TABLE1_PAPER_N_ROLLOUTS" --override-n-max-slots "$TABLE1_PAPER_N_MAX_SLOTS"
  done; done
  if (( ${#missing[@]} )); then
    printf '[eval] no checkpoint (run.sh train, or CKPT=<file>): %s\n' "${missing[@]}" >&2
    exit 2
  fi
}

figure() {
  for fig in table3 table22 table24_25; do
    quickstart_python experiments/draw_figure.py --figure "$fig"
  done
}

case "${1:-all}" in
  train)  train ;;
  eval)   eval_cells ;;
  figure) figure ;;
  all)    train; eval_cells; figure ;;
  *) echo "usage: run.sh [train | eval | figure | all]" >&2; exit 2 ;;
esac
