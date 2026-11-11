#!/usr/bin/env bash
# Build the C++ router into build_rl/ from the project conda env: step 4 of setup_all.sh, and
# the command to rebuild after a C++ change (no env update, no submodule reset). Runs
# engine/build_rl_router.sh inside the env, on Linux and macOS alike.
#
#   bash tools/setup/build_engine.sh
#   PCBWORLD_ENV=myenv bash tools/setup/build_engine.sh
#   BUILD_DIR=<dir> bash tools/setup/build_engine.sh     # another build location

set -euo pipefail
# a relative BUILD_DIR means the caller's directory, not the repository top cd'd to below
case "${BUILD_DIR:-/}" in /*) ;; *) BUILD_DIR="$PWD/$BUILD_DIR" ;; esac
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

ENV_NAME="${PCBWORLD_ENV:-pcbworld}"
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
# BUILD_DIR keeps the output in this tree; an engine checkout elsewhere would otherwise
# place it inside itself.
conda run -n "$ENV_NAME" env BUILD_DIR="${BUILD_DIR:-$PWD/build_rl}" bash "$ENGINE_HOME/build_rl_router.sh"
