#!/usr/bin/env bash
# Fetch and PIN the three external router baselines to the versions used for
# the paper's runs. Idempotent: re-running verifies pins instead of re-cloning.
#
#   bash tools/setup/fetch_baselines.sh
#   export KRT_ROOT="$PWD/external/KiCadRoutingTools"
#
# Pins:
#   Freerouting  v2.1.0 release jar (sha256-verified) -> external/freerouting/
#   OrthoRoute   git submodule external/OrthoRoute, pinned commit f45dc68
#   KRT          commit d9557ad1 -> external/KiCadRoutingTools (route.py defaults
#                match the paper's Table 18 — via_cost 50, turn_cost 1000,
#                via_proximity_cost 10, direction_preference_cost 50)

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

FREEROUTING_URL="https://github.com/freerouting/freerouting/releases/download/v2.1.0/freerouting-2.1.0.jar"
FREEROUTING_SHA256="2c07d58f75dac03782664081e7a58b41c25400d871a9fcf166a2ea6fe60d5def"
ORTHOROUTE_COMMIT="f45dc68"
KRT_REPO="https://github.com/drandyhaas/KiCadRoutingTools"
KRT_COMMIT="d9557ad10e8fa51fcf641749ae1cc9dca66e9648"

# -- Freerouting jar ---------------------------------------------------------
mkdir -p external/freerouting
jar=external/freerouting/freerouting-2.1.0.jar
if [[ ! -f "$jar" ]]; then
  curl -L -o "$jar" "$FREEROUTING_URL"
fi
echo "$FREEROUTING_SHA256  $jar" | sha256sum -c -

# -- OrthoRoute (git submodule, recorded commit == paper commit) -------------
git submodule update --init external/OrthoRoute
got=$(git -C external/OrthoRoute rev-parse --short HEAD)
if [[ "$got" != "$ORTHOROUTE_COMMIT"* && "$ORTHOROUTE_COMMIT" != "$got"* ]]; then
  echo "ERROR: external/OrthoRoute is at $got, expected $ORTHOROUTE_COMMIT" >&2
  exit 1
fi
echo "OrthoRoute pinned at $got"

# -- KRT (KiCadRoutingTools) -------------------------------------------------
if [[ ! -d external/KiCadRoutingTools/.git ]]; then
  git clone "$KRT_REPO" external/KiCadRoutingTools
fi
git -C external/KiCadRoutingTools fetch -q origin
git -C external/KiCadRoutingTools checkout -q "$KRT_COMMIT"
echo "KRT pinned at $(git -C external/KiCadRoutingTools rev-parse --short HEAD)"

cat <<'EOF'

Done. Next steps (see methods/baselines/rule_based/README.md):
  pip install -e methods/baselines/rule_based/krt
  pip install -e external/OrthoRoute
  export KRT_ROOT="$PWD/external/KiCadRoutingTools"
EOF
