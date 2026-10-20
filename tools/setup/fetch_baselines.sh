#!/usr/bin/env bash
# Fetch and PIN the external router baselines (OrthoRoute: KDD-era record only). Idempotent: re-running
# verifies pins instead of re-cloning.
#
#   bash tools/setup/fetch_baselines.sh
#   export KRT_ROOT="$PWD/external/KiCadRoutingTools"
#
# Pins:
#   Freerouting  v2.1.0 release jar (sha256-verified) -> external/freerouting/
#   OrthoRoute   git submodule external/OrthoRoute, pinned commit f45dc68
#   KRT          commit 561f158 -> external/KiCadRoutingTools: the KRT the KDD and CADET
#                papers used (upstream main, 2026-04-04). The rule-based runner still passes
#                the routing hyperparameters explicitly (KRT_PAPER_ARGS in
#                run_rule_based_routers.py).

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

FREEROUTING_URL="https://github.com/freerouting/freerouting/releases/download/v2.1.0/freerouting-2.1.0.jar"
FREEROUTING_SHA256="2c07d58f75dac03782664081e7a58b41c25400d871a9fcf166a2ea6fe60d5def"
ORTHOROUTE_COMMIT="f45dc68"
KRT_REPO="https://github.com/drandyhaas/KiCadRoutingTools"
KRT_COMMIT="561f1584577ea86d99e223ab27a2cd1cdbfa8bb3"

# -- Freerouting jar ---------------------------------------------------------
mkdir -p external/freerouting
jar=external/freerouting/freerouting-2.1.0.jar
if [[ ! -f "$jar" ]]; then
  curl -L -o "$jar" "$FREEROUTING_URL"
fi
echo "$FREEROUTING_SHA256  $jar" | sha256sum -c -

# -- OrthoRoute (git submodule; the pinned commit does not rerun the KDD logs) --
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
  export KRT_ROOT="$PWD/external/KiCadRoutingTools"
EOF
