#!/usr/bin/env bash
# Table 1 RL rows (D2 / D3) — PCBWorld policy quality (PPO per-step / PPO terminal / GRPO).
# Paper artifacts: Table 3, Table 22, Tables 24/25.
#
# Stages:
#   train   — method x seed training sweep -> train_policy.sh
#   figure  — draw_figure.py for table3 + table22 + table24_25
#   all     — train + figure
#
# Rollout+eval of trained checkpoints is the shared 3-stage path (not duplicated here):
#   python eval/pipeline.py --ckpt <ckpt> --boards-dir <dir> --output-dir <cell> --skip-drc
#   python eval/pipeline.py --stages eval,aggregate --output-dir <cell> --check-angle 45
#
# Honors DRY_RUN=1 / SMOKE=1 / GPU=N and METHODS / SEEDS overrides.
set -euo pipefail
_self="$(cd "$(dirname "$0")" && pwd)"
source "$_self/../../_lib/env.sh"
source "$_self/cases.sh"
cd "$CADAGENT_REPO_ROOT"

METHODS="${METHODS:-${METHOD:-ppo_per_step ppo_terminal grpo}}"
paper_repro_load_table1_policy_case ppo_per_step
SEEDS="${SEEDS:-$TABLE1_SEEDS_DEFAULT}"
GPU="${GPU:-0}"

train() {
  for method in $METHODS; do for seed in $SEEDS; do
    local args=(--method "$method" --seed "$seed" --gpu "$GPU")
    [[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
    [[ "${SMOKE:-0}" == "1" ]] && args+=(--smoke)
    quickstart_run bash "$_self/train_policy.sh" "${args[@]}"
  done; done
}

figure() {
  for fig in table3 table22 table24_25; do
    quickstart_python experiments/draw_figure.py --figure "$fig"
  done
}

case "${1:-all}" in
  train)  train ;;
  figure) figure ;;
  all)    train; figure ;;
  *) echo "usage: run.sh [train | figure | all]" >&2; exit 2 ;;
esac
