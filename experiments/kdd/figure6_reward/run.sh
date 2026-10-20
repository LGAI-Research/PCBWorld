#!/usr/bin/env bash
# Figure 6 / Figure 8 — reward-weight ablation (wirelength x via penalty sweep).
#
# Stages:
#   train   — wire x via x seed training sweep -> train_dense_reward_cell.sh (each cell and seed
#             under $TRAIN_ROOT/w<wire>_v<via>/seed<S>)
#   eval    — per cell: roll out each seed's policy_best.pt on the D2 test boards, pool them into
#             $EXPR_ROOT/d2a/reward_w<wire>_v<via> and score it with the paper cells' settings
#             (configs/paper_train_flags.sh PAPER_*_FLAGS, ONE rollout per board, 45 degrees — the
#             figure averages every rollout, no winner selection)
#   plot    — plot_reward_factorial.py (W&B/TensorBoard marginals)
#   figure  — draw_figure.py --figure fig8 (paper sweep figure from per_rollout.csv)
#   all     — train + eval + figure
#
# Honors DRY_RUN=1 / SMOKE=1 / GPU=N and WIRES / VIAS / SEEDS overrides.
set -euo pipefail
_self="$(cd "$(dirname "$0")" && pwd)"
source "$_self/../../_lib/env.sh"
source "$_self/cases.sh"
source "$_self/../configs/paper_train_flags.sh"
cd "$PCBWORLD_REPO_ROOT"

paper_repro_load_figure6_reward_case
WIRES="${WIRES:-$FIGURE6_WIRELENGTH_GRID_DEFAULT}"
VIAS="${VIAS:-$FIGURE6_VIA_GRID_DEFAULT}"
SEEDS="${SEEDS:-$FIGURE6_SEEDS_DEFAULT}"
GPU="${GPU:-0}"
TRAIN_ROOT="${TRAIN_ROOT:-${LOCAL_OUT}/training_logs/reward_ablation/training}"

train() {
  for wire in $WIRES; do for via in $VIAS; do for seed in $SEEDS; do
    local args=(--wirelength-penalty "$wire" --via-penalty "$via" --seed "$seed" --gpu "$GPU"
                --output-root "$TRAIN_ROOT/w${wire}_v${via}/seed${seed}")
    [[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
    [[ "${SMOKE:-0}" == "1" ]] && args+=(--smoke)
    quickstart_run bash "$_self/train_dense_reward_cell.sh" "${args[@]}"
  done; done; done
}

# One pooled cell per (wire, via) under $EXPR_ROOT/d2a/reward_w<wire>_v<via> — where
# draw_figure.py reads them (var/results/kdd) — from the checkpoints of every seed in SEEDS.
eval_cells() {
  local boards="$LOCAL_OUT/rollouts/reward/d2a_boards.txt" missing=()
  for wire in $WIRES; do for via in $VIAS; do
    local cell="reward_w${wire}_v${via}" ckpts=() absent=()
    for seed in $SEEDS; do
      ckpts+=("$TRAIN_ROOT/w${wire}_v${via}/seed${seed}/checkpoints/policy_best.pt")
      [[ -f "${ckpts[-1]}" || "${DRY_RUN:-0}" == "1" ]] || absent+=("${ckpts[-1]}")
    done
    if (( ${#absent[@]} )); then missing+=("${absent[@]}"); continue; fi
    [[ -f "$boards" ]] || kdd_task_boards_list d2a "$boards"
    kdd_eval_cell "$EXPR_ROOT/d2a/$cell" "$boards" "$LOCAL_OUT/rollouts/reward/$cell" "$FIGURE6_CORNER_MODE" \
      "${ckpts[@]}" -- --n-rollouts "$FIGURE6_PAPER_N_ROLLOUTS"
  done; done
  if (( ${#missing[@]} )); then
    printf '[eval] no checkpoint (run.sh train): %s\n' "${missing[@]}" >&2
    exit 2
  fi
}

plot() {
  quickstart_python experiments/kdd/figure6_reward/plot_reward_factorial.py \
    --overleaf-root "${OUT_OVERLEAF:-${LOCAL_OUT}/figures/figure6}" \
    --tb-root "${TB_ROOT:-${EXPR_ROOT}/training_logs/reward_ablation/tensorboard_logs}" \
    --source "${SOURCE:-auto}"
}

figure() { quickstart_python experiments/draw_figure.py --figure fig8; }

case "${1:-all}" in
  train)  train ;;
  eval)   eval_cells ;;
  plot)   plot ;;
  figure) figure ;;
  all)    train; eval_cells; figure ;;
  *) echo "usage: run.sh [train | eval | plot | figure | all]" >&2; exit 2 ;;
esac
