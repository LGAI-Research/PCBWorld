#!/usr/bin/env bash
# One-shot bootstrap: conda env -> submodules -> pinned baseline downloads ->
# C++ router build -> import smoke. Each role is its own script and can be run
# alone:
#   environment.yml                   the single conda env (python/toolchains + pip lock)
#   tools/setup/fetch_baselines.sh    pinned downloads (Freerouting jar / OrthoRoute / KRT)
#   engine/build_rl_router.sh         C++ router build (rsync + patch + cmake/ninja)
#
#   bash tools/setup/setup_all.sh                 # everything
#   SKIP_BASELINES=1 bash tools/setup/setup_all.sh  # RL core only
#   CADAGENT_ENV=myenv bash tools/setup/setup_all.sh

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

ENV_NAME="${CADAGENT_ENV:-cadagent}"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "== [1/5] conda env '$ENV_NAME' =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env exists — updating from environment.yml"
  conda env update -n "$ENV_NAME" -f environment.yml
else
  conda env create -n "$ENV_NAME" -f environment.yml
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
# Same contract as pcb_world/engine/router_client.py: the engine lives at
# engine/ unless PCBWORLD_ENGINE_HOME says otherwise.
ENGINE_HOME="${PCBWORLD_ENGINE_HOME:-engine}"
if [ ! -f "$ENGINE_HOME/build_rl_router.sh" ]; then
  echo "no engine at '$ENGINE_HOME' — the routing engine is a separate GPLv3" >&2
  echo "repository, pinned here as a submodule and not part of this one." >&2
  echo "Run: git submodule update --init --recursive" >&2
  echo "(or point PCBWORLD_ENGINE_HOME at an existing checkout)" >&2
  exit 1
fi
# BUILD_DIR keeps the output in this tree; the engine would otherwise place it
# inside its own checkout.
conda run -n "$ENV_NAME" env BUILD_DIR="$PWD/build_rl" bash "$ENGINE_HOME/build_rl_router.sh"

echo "== [5/5] import smoke =="
conda run -n "$ENV_NAME" bash -c \
  'PYTHONPATH=build_rl/pcbnew/python/rl:. python -c "import kicad_rl_router as krl; print(\"kicad_rl_router import OK\")"'

echo "setup_all: done. Next: docs/QUICKSTART.md"
