#!/usr/bin/env bash
# One-shot bootstrap: conda env -> submodules -> pinned baseline downloads ->
# C++ router build -> import smoke. Each role is its own script and can be run
# alone:
#   environment.yml                   the single conda env (python/toolchains + pip lock);
#                                     environment.macos.yml on Apple Silicon
#   tools/setup/fetch_baselines.sh    pinned downloads (Freerouting jar / OrthoRoute / KRT)
#   tools/setup/build_engine.sh       C++ router build (engine/build_rl_router.sh in the env)
#                                     — also the rebuild command
#
#   bash tools/setup/setup_all.sh                 # everything
#   SKIP_BASELINES=1 bash tools/setup/setup_all.sh  # RL core only
#   SKIP_ENV=1 bash tools/setup/setup_all.sh        # keep the existing env as is (no solve)
#   PCBWORLD_ENV=myenv bash tools/setup/setup_all.sh

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

ENV_NAME="${PCBWORLD_ENV:-pcbworld}"
# The engine goes to build_rl/, where the loader looks; a BUILD_DIR the shell carries is
# refused, not silently replaced (another location: BUILD_DIR=<dir> bash tools/setup/build_engine.sh).
if [[ -n "${BUILD_DIR:-}" ]]; then
  echo "setup_all.sh always builds the engine into $PWD/build_rl, but BUILD_DIR=$BUILD_DIR is set;" >&2
  echo "unset it (another location: BUILD_DIR=<dir> bash tools/setup/build_engine.sh)" >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  ENV_FILE=environment.yml ;;
  Darwin-arm64)  ENV_FILE=environment.macos.yml ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m): environment.yml is Linux x86_64," >&2
     echo "environment.macos.yml is macOS arm64 (Apple Silicon)" >&2
     exit 1 ;;
esac

# conda's classic solver does not finish on this env (over 30 minutes); conda with the
# libmamba solver (the default from conda 23.10), or the mamba of conda's base, takes a minute.
# A mamba elsewhere (its own root prefix) would put the env where `conda run -n` cannot see it.
BASE_MAMBA="$CONDA_BASE/bin/mamba"
ENV_EXISTS=0
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" && ENV_EXISTS=1
if [[ "${SKIP_ENV:-0}" == 1 ]]; then
  if [[ "$ENV_EXISTS" != 1 ]]; then
    echo "SKIP_ENV=1, but there is no conda env '$ENV_NAME' to keep" >&2
    exit 1
  fi
  echo "== [1/5] conda env '$ENV_NAME': kept as is (SKIP_ENV=1) =="
else
  if [ -x "$BASE_MAMBA" ]; then
    SOLVE="$BASE_MAMBA"
  elif [ "$(conda config --show solver 2>/dev/null)" = "solver: libmamba" ]; then
    SOLVE=conda
  else
    echo "conda $(conda --version | awk '{print $2}') does not use the libmamba solver; its classic solver" >&2
    echo "does not finish on $ENV_FILE. Upgrade conda (conda update -n base conda), or on conda" >&2
    echo "22.11+ switch it (conda install -n base conda-libmamba-solver && conda config --set solver" >&2
    echo "libmamba), or install mamba into conda's base (conda install -n base -c conda-forge mamba)" >&2
    [[ "$ENV_EXISTS" == 1 ]] && echo "(SKIP_ENV=1 keeps the existing env '$ENV_NAME' without solving)" >&2
    exit 1
  fi
  echo "== [1/5] conda env '$ENV_NAME' ($ENV_FILE, $(basename "$SOLVE")) =="
  if [[ "$ENV_EXISTS" == 1 ]]; then
    echo "env exists — updating from $ENV_FILE"
    "$SOLVE" env update -n "$ENV_NAME" -f "$ENV_FILE"
  else
    "$SOLVE" env create -n "$ENV_NAME" -f "$ENV_FILE"
  fi
fi

echo "== [2/5] git submodules =="
git submodule update --init --recursive

if [[ "${SKIP_BASELINES:-0}" == 1 ]]; then
  echo "== [3/5] baselines: skipped (SKIP_BASELINES=1) =="
else
  echo "== [3/5] baselines: pinned downloads =="
  bash tools/setup/fetch_baselines.sh
fi

echo "== [4/5] C++ router build =="
BUILD_DIR="$PWD/build_rl" PCBWORLD_ENV="$ENV_NAME" bash tools/setup/build_engine.sh

echo "== [5/5] import smoke =="
# The env's own interpreter: a PATH entry conda does not manage (an IDE-injected env,
# say) can precede the activated env and make a bare `python` someone else's.
conda run -n "$ENV_NAME" bash -c \
  'PYTHONPATH=build_rl/pcbnew/python/rl:. "$CONDA_PREFIX/bin/python" -c "import kicad_rl_router as krl; print(\"kicad_rl_router import OK\")"'

echo "setup_all: done. Next: docs/QUICKSTART.md"
