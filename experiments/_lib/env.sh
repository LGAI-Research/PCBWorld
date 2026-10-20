#!/usr/bin/env bash
# Shared environment for KDD benchmark quickstart wrappers.

set -euo pipefail

# env.sh sits 2 levels deep (experiments/_lib/) -> repo root is ../..
export PCBWORLD_REPO_ROOT="${PCBWORLD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# In-repo var/ tree (canonical): data/ckpt/results all default under var/.
export KDD_EXPERIMENTS_ROOT="${KDD_EXPERIMENTS_ROOT:-${PCBWORLD_REPO_ROOT}/var}"
# Packaged KDD-benchmark tree (datasets + published results). Not distributed with
# the repo: defaults under the dataset root when PCBWORLD_DATA_ROOT is set, otherwise
# stays empty and must be given explicitly by whoever needs it.
export KDD_BENCH_ROOT="${KDD_BENCH_ROOT:-${PCBWORLD_DATA_ROOT:+${PCBWORLD_DATA_ROOT}/KDD_benchmark}}"
# Data / checkpoint / results roots default under the in-repo var/ tree (datasets campaign-flat; results/ckpt campaign=kdd).
# To read from the packaged benchmark tree instead: DATASET_ROOT=$KDD_BENCH_ROOT/dataset (etc).
export DATASET_ROOT="${DATASET_ROOT:-${KDD_EXPERIMENTS_ROOT}/datasets}"
export CKPT_ROOT="${CKPT_ROOT:-${KDD_EXPERIMENTS_ROOT}/checkpoints/kdd}"
export EXPR_ROOT="${EXPR_ROOT:-${KDD_EXPERIMENTS_ROOT}/results/kdd}"
# WRITE root (repo-local; new outputs default here so wrappers never need write
# access to the read-only dataset root).
# To publish back to the shared staged tree, override: LOCAL_OUT=$EXPR_ROOT
export LOCAL_OUT="${LOCAL_OUT:-${PCBWORLD_REPO_ROOT}/var/outputs}"
export OVERLEAF_ROOT="${OVERLEAF_ROOT:-}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

if [[ -z "${T1_BASELINE_PYTHON:-}" ]]; then
  export T1_BASELINE_PYTHON="$PYTHON_BIN"
fi

if [[ -z "${ROUTER_BUILD_DIR:-}" ]]; then
  if [[ -d "${PCBWORLD_REPO_ROOT}/build_rl/pcbnew/python/rl" ]]; then
    export ROUTER_BUILD_DIR="${PCBWORLD_REPO_ROOT}/build_rl"
  fi
fi

if [[ -n "${ROUTER_BUILD_DIR:-}" && -d "${ROUTER_BUILD_DIR}/pcbnew/python/rl" ]]; then
  export PCBWORLD_KICAD_RL_BUILD_DIR="$ROUTER_BUILD_DIR"
  export PYTHONPATH="${PCBWORLD_REPO_ROOT}:${ROUTER_BUILD_DIR}/pcbnew/python/rl:${PYTHONPATH:-}"
  if [[ -d "${ROUTER_BUILD_DIR}/lib" ]]; then
    export LD_LIBRARY_PATH="${ROUTER_BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

# Crash diagnostics (pcb_world/diag): deliberately NOT exported here. Every
# consumer already resolves the same default without it — default_log_dir()
# falls back to <repo>/var/crashlogs, and engine server children inherit the
# parent env / resolve the same default in-process — while exporting it defeats
# tests/conftest.py's isolation guard (the suite's deliberate crash-test
# artifacts are routed to a throwaway tmp dir ONLY when the var is unset, so a
# green pytest run with env.sh sourced would litter var/crashlogs/). Run
# scripts that want per-run logs may still export KICAD_CRASH_LOG_DIR=<run_dir>/crashlogs.

quickstart_log() {
  printf '[quickstart] %s\n' "$*" >&2
}

quickstart_mkdir() {
  mkdir -p "$@"
}

quickstart_run() {
  quickstart_log "$*"
  "$@"
}

quickstart_python() {
  quickstart_run "$PYTHON_BIN" "$@"
}

# --- Training-dispatch helpers ---
# Paths above are canonical (in-repo var/).

public_log() {
  printf '[paper-repro] %s\n' "$*" >&2
}

public_step_penalty_for_grid() {
  case "$1" in
    10) printf '0.03' ;;
    50) printf '0.006' ;;
    100) printf '0.003' ;;
    200) printf '0.0015' ;;
    500) printf '0.0006' ;;
    1000) printf '0.0003' ;;
    *) echo "unsupported grid size: $1" >&2; return 2 ;;
  esac
}

public_wandb_args() {
  local run_name="$1"
  local group="$2"
  local project="${3:-${WANDB_PROJECT:-pcbworld}}"
  local tags="${4:-paper_repro}"
  PUBLIC_WANDB_ARGS=()
  # W&B is opt-in, matching the trainer default: on only when WANDB=1 is exported
  # and the W&B client itself is not disabled. Off is said explicitly — a --resume
  # starts from the checkpoint's wandb value.
  if [[ "${WANDB:-0}" != "1" || "${WANDB_MODE:-}" == "disabled" ]]; then
    PUBLIC_WANDB_ARGS+=(--no-wandb)
    return 0
  fi
  PUBLIC_WANDB_ARGS+=(--wandb --wandb-project "$project" --wandb-run-name "$run_name" --wandb-group "$group" --wandb-tags "$tags")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    PUBLIC_WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
}

public_cuda_run() {
  public_log "CUDA_VISIBLE_DEVICES=${GPU:-0} $(printf '%q ' "$@")"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${GPU:-0}" "$@"
}

# --- KDD paper-cell evaluation helpers ---
# The recipes' eval stages write their pooled cells under $EXPR_ROOT/<task>/<cell>: the tree
# experiments/draw_figure.py reads (var/results/kdd — the EXPR_ROOT default), so a recipe's figure
# stage draws what its eval stage produced. Per-checkpoint rollouts and board lists stay under
# $LOCAL_OUT. An EXPR_ROOT pointed at a read-only staged tree makes the eval stage fail.

# kdd_task_boards_list <task> <file>: write the test boards of a KDD split alias (qs_resolve_split:
# experiments/kdd/configs/quickstart/splits.json or QS_SPLITS_CONFIG; d2a | d3a | d3b | d3c, legacy
# t-names accepted) to <file>, one path per line. Every board the split lists must exist — a
# partial set is not the paper's.
kdd_task_boards_list() {
  source "$PCBWORLD_REPO_ROOT/experiments/_lib/llm_lib.sh"
  qs_resolve_split "$1"
  mkdir -p "$(dirname "$2")"
  "$PYTHON_BIN" - "$QS_BOARDS_JSON" "$QS_DIFF" "$QS_SPLIT" "$2" <<'PY'
import json, sys
from methods._shared.board_loader import load_boards_from_split_json
split_json, diff, split, out = sys.argv[1:]
listed = json.load(open(split_json))[diff][split]
boards = load_boards_from_split_json(split_json, diff, split)
if len(boards) != len(listed):
    sys.exit(f"[kdd] {split_json} {diff}.{split}: {len(listed) - len(boards)} of {len(listed)} boards are missing (warnings above)")
open(out, "w").write("".join(b.path + "\n" for b in boards))
PY
}

# kdd_eval_cell <cell_dir> <boards_list> <rollout_root> <check_angle> <ckpt>... -- <rollout flags>:
# one paper cell. eval/pipeline.py rolls out one checkpoint per run, so each checkpoint's Stage 1
# goes to <rollout_root>/<i>/<cell name> (PAPER_ROLLOUT_FLAGS + <rollout flags>), the parts are
# pooled into <cell_dir> (pool_seed_cells.py: boards, rows, run records), and the pooled cell is
# scored and aggregated (Stages 2-3, PAPER_SCORING_FLAGS). Needs paper_train_flags.sh sourced.
kdd_eval_cell() {
  local cell="$1" boards="$2" root="$3" angle="$4" ckpts=() parts=() i=0
  shift 4
  while [[ "$1" != "--" ]]; do ckpts+=("$1"); shift; done
  shift
  if [[ -e "$cell/per_rollout.csv" ]]; then
    echo "[kdd] $cell already holds a cell (per_rollout.csv) — move it away or point EXPR_ROOT elsewhere" >&2
    exit 2
  fi
  for ckpt in "${ckpts[@]}"; do
    parts+=("$root/$i/$(basename "$cell")")
    public_cuda_run "$PYTHON_BIN" -u eval/pipeline.py --stages rollout --ckpt "$ckpt" --boards-list "$boards" \
      "${PAPER_ROLLOUT_FLAGS[@]}" "$@" --check-angle "$angle" --output-dir "${parts[-1]}"
    i=$((i + 1))
  done
  quickstart_log "pool ${#parts[@]} checkpoints -> $cell"
  [[ "${DRY_RUN:-0}" == "1" ]] || "$PYTHON_BIN" experiments/_lib/pool_seed_cells.py "$cell" "${parts[@]}"
  public_cuda_run "$PYTHON_BIN" -u eval/pipeline.py --stages eval,aggregate --boards-list "$boards" \
    "${PAPER_SCORING_FLAGS[@]}" --check-angle "$angle" --output-dir "$cell"
}
