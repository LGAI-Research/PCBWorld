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
# The D1 corpus is not distributed with this repository. Every stage checks its
# inputs first and, when they are absent, exits 2 with a notice naming the
# generator command that builds a D1-shaped corpus (cases.sh `d1_preflight`);
# BOARDS_DIR / CKPT / --split-json point a stage at the paper corpus instead
# (experiments/kdd/figure5_d1/README.md).
set -euo pipefail
_self="$(cd "$(dirname "$0")" && pwd)"
source "$_self/../../_lib/env.sh"
source "$_self/cases.sh"
source "$_self/../configs/paper_train_flags.sh"
cd "$PCBWORLD_REPO_ROOT"

GRIDS="${L1_GRIDS:-10 50 100 200 500}"
SEEDS="${SEEDS:-42 43 44 45}"
GPU="${GPU:-0}"
TRAIN_ROOT="${TRAIN_ROOT:-${LOCAL_OUT}/training_logs/d1_grid/transformer_ppo}"

train() {
  local method="${1:-transformer}"
  for grid in $GRIDS; do for seed in $SEEDS; do
    # baselines (jumanji/sable) pin the GPU via CUDA_VISIBLE_DEVICES, not a --gpu flag
    local args=(--grid-size "$grid" --seed "$seed")
    [[ "$method" == transformer || "$method" == ppo ]] && args+=(--gpu "$GPU" --output-root "$TRAIN_ROOT/grid${grid}/seed${seed}")
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

# Held-out grid test eval: per grid, roll out each seed's checkpoint, pool them into
# $EXPR_ROOT/d1/d1_grid<G>/transformer_pcbworld — where draw_figure.py reads them — and score it
# with the paper cells' settings (configs/paper_train_flags.sh PAPER_*_FLAGS, 5 rollouts, the
# 90-degree angle the grid routes with). The checkpoint defaults to the staged paper one,
# $CKPT_ROOT/Transformer_1L_300 — the run's LAST checkpoint policy_iter_300.pt (its
# policy_best.pt was picked far too early, around iteration 25); a run trained here is
# $TRAIN_ROOT/grid<G>/seed<S>/checkpoints/policy_iter_300.pt, given by CKPT_ROOT=... or CKPT=<file>.
eval_t1() {
  if [[ -n "${CKPT:-}" && "$SEEDS" == *" "* ]]; then
    echo "[eval] CKPT=<file> scores one checkpoint: give a single SEEDS" >&2
    exit 2
  fi
  local ran=0
  for grid in $GRIDS; do
    paper_repro_load_d1_grid_case "$grid"
    local boards="${BOARDS_DIR:-$(d1_generated_boards_dir "$grid")}" ckpts=() skip=0
    for seed in $SEEDS; do
      ckpts+=("${CKPT:-$CKPT_ROOT/Transformer_1L_300/grid${grid}/seed${seed}/policy_iter_300.pt}")
      [[ -f "${ckpts[-1]}" ]] || { d1_absent "D1 checkpoint (grid ${grid}, seed ${seed})" "${ckpts[-1]}"; skip=1; }
    done
    [[ -d "$boards" ]] || { d1_absent "D1 test boards (grid ${grid})" "$boards"; skip=1; }
    if (( skip )); then continue; fi
    ran=$((ran + 1))
    local list="$LOCAL_OUT/rollouts/d1_grid/grid${grid}_boards.txt"
    mkdir -p "$(dirname "$list")"
    find "$boards" -maxdepth 1 -name '*.kicad_pcb' | sort > "$list"
    kdd_eval_cell "$EXPR_ROOT/d1/d1_grid${grid}/transformer_pcbworld" "$list" \
      "$LOCAL_OUT/rollouts/d1_grid/grid${grid}" "$T1_TRANSFORMER_CORNER_MODE" "${ckpts[@]}" -- --n-rollouts 5
  done
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
