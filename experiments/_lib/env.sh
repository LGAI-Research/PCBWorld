#!/usr/bin/env bash
# Shared environment for KDD benchmark quickstart wrappers.

set -euo pipefail

# env.sh sits 2 levels deep (experiments/_lib/) -> repo root is ../..
export CADAGENT_REPO_ROOT="${CADAGENT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# In-repo var/ tree (canonical): data/ckpt/results all default under var/.
export KDD_EXPERIMENTS_ROOT="${KDD_EXPERIMENTS_ROOT:-${CADAGENT_REPO_ROOT}/var}"
# Packaged KDD-benchmark tree (datasets + published results). Not distributed with
# the repo: defaults under the dataset root when CADAGENT_DATA_ROOT is set, otherwise
# stays empty and must be given explicitly by whoever needs it.
export KDD_BENCH_ROOT="${KDD_BENCH_ROOT:-${CADAGENT_DATA_ROOT:+${CADAGENT_DATA_ROOT}/KDD_benchmark}}"
# Data / checkpoint / results roots default under the in-repo var/ tree (datasets campaign-flat; results/ckpt campaign=kdd).
# To read from the packaged benchmark tree instead: DATASET_ROOT=$KDD_BENCH_ROOT/dataset (etc).
export DATASET_ROOT="${DATASET_ROOT:-${KDD_EXPERIMENTS_ROOT}/datasets}"
export CKPT_ROOT="${CKPT_ROOT:-${KDD_EXPERIMENTS_ROOT}/checkpoints/kdd}"
export EXPR_ROOT="${EXPR_ROOT:-${KDD_EXPERIMENTS_ROOT}/results/kdd}"
# WRITE root (repo-local; new outputs default here so wrappers never need write
# access to the read-only dataset root).
# To publish back to the shared staged tree, override: LOCAL_OUT=$EXPR_ROOT
export LOCAL_OUT="${LOCAL_OUT:-${CADAGENT_REPO_ROOT}/var/outputs}"
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
  if [[ -d "${CADAGENT_REPO_ROOT}/build_rl/pcbnew/python/rl" ]]; then
    export ROUTER_BUILD_DIR="${CADAGENT_REPO_ROOT}/build_rl"
  fi
fi

if [[ -n "${ROUTER_BUILD_DIR:-}" && -d "${ROUTER_BUILD_DIR}/pcbnew/python/rl" ]]; then
  export CADAGENT_KICAD_RL_BUILD_DIR="$ROUTER_BUILD_DIR"
  export PYTHONPATH="${CADAGENT_REPO_ROOT}:${ROUTER_BUILD_DIR}/pcbnew/python/rl:${PYTHONPATH:-}"
  if [[ -d "${ROUTER_BUILD_DIR}/lib" ]]; then
    export LD_LIBRARY_PATH="${ROUTER_BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

# Crash diagnostics (pcb_world/diag): explicit default so every subprocess of a
# dispatch resolves the same dir. Run scripts may point this at <run_dir>/crashlogs.
export KICAD_CRASH_LOG_DIR="${KICAD_CRASH_LOG_DIR:-${CADAGENT_REPO_ROOT}/var/crashlogs}"

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
  # W&B is opt-in, matching the trainer default: emit flags only when WANDB=1
  # is exported, and never when the W&B client itself is disabled.
  if [[ "${WANDB:-0}" != "1" || "${WANDB_MODE:-}" == "disabled" ]]; then
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
