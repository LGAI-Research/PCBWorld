#!/usr/bin/env bash
# Figure 6 / Figure 8 — reward-weight ablation (wirelength x via penalty sweep).
#
# Stages:
#   train   — wire x via x seed training sweep -> train_dense_reward_cell.sh
#   plot    — plot_reward_factorial.py (W&B/TensorBoard marginals)
#   figure  — draw_figure.py --figure fig8 (paper sweep figure from per_rollout.csv)
#   all     — train + figure
#
# Honors DRY_RUN=1 / SMOKE=1 / GPU=N and WIRES / VIAS / SEEDS overrides.
set -euo pipefail
_self="$(cd "$(dirname "$0")" && pwd)"
source "$_self/../../_lib/env.sh"
source "$_self/cases.sh"
cd "$CADAGENT_REPO_ROOT"

paper_repro_load_figure6_reward_case
WIRES="${WIRES:-$FIGURE6_WIRELENGTH_GRID_DEFAULT}"
VIAS="${VIAS:-$FIGURE6_VIA_GRID_DEFAULT}"
SEEDS="${SEEDS:-$FIGURE6_SEEDS_DEFAULT}"
GPU="${GPU:-0}"

train() {
  for wire in $WIRES; do for via in $VIAS; do for seed in $SEEDS; do
    local args=(--wirelength-penalty "$wire" --via-penalty "$via" --seed "$seed" --gpu "$GPU")
    [[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
    [[ "${SMOKE:-0}" == "1" ]] && args+=(--smoke)
    quickstart_run bash "$_self/train_dense_reward_cell.sh" "${args[@]}"
  done; done; done
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
  plot)   plot ;;
  figure) figure ;;
  all)    train; figure ;;
  *) echo "usage: run.sh [train | plot | figure | all]" >&2; exit 2 ;;
esac
