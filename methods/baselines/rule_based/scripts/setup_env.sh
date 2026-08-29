#!/usr/bin/env bash
#
# Install everything the 3 rule-based-router baselines need into the
# active conda env. LEGACY bootstrap for a bare env (e.g. the old
# ``cadagent-classical``): the standard ``cadagent`` env from environment.yml
# already covers steps 1-3, and tools/setup/fetch_baselines.sh fetches the
# externals with pin verification. Safe to re-run — each step is a no-op if
# already installed.
#
# What this does:
#   1. conda-install openjdk (>=21) for Freerouting
#   2. conda-install rust  (KRT's grid_router.so is rebuilt locally on first
#                           krt-route invocation; we ship no pre-built copy)
#   3. pip-install cupy-cuda12x for OrthoRoute (CUDA 12.x runtime)
#   4. pip-install editable OrthoRoute (external/OrthoRoute submodule), if present
#   5. pip-install KRT runtime deps + editable krt-runner (methods/baselines/rule_based/krt)
#   6. smoke-test each tool is invokable
#
# This script does NOT install pcbnew. Pre-converting raw .kicad_pcb files
# to DSN+ORP (the format Freerouting/OrthoRoute consume) requires pcbnew
# from a KiCad install — see engine/pcbnew_prep/README.md. The unified
# runner assumes the DSN+ORP already exist alongside each board's .kicad_pcb.
#
# Usage:
#   conda activate <target env>
#   bash methods/baselines/rule_based/scripts/setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
# repo root: rule-based -> baselines -> methods -> repo root; vendored libs live under external/
REPO_ROOT="$(cd "${BASELINES_DIR}/../../.." && pwd)"
EXTERNAL_DIR="${REPO_ROOT}/external"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" = "base" ]; then
    echo "ERROR: activate a non-base conda env first (e.g. 'conda activate cadagent')" >&2
    exit 1
fi
echo "=== env: ${CONDA_DEFAULT_ENV} (${CONDA_PREFIX}) ==="

# Use the env's interpreter explicitly. ``python`` on PATH can be shadowed
# by other conda envs the shell has touched, so always resolve through
# ${CONDA_PREFIX}/bin/python and run pip as ``python -m pip``.
PY="${CONDA_PREFIX}/bin/python"
PIP=("${PY}" "-m" "pip")
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} not found" >&2
    exit 1
fi
echo "    python: ${PY} ($(${PY} --version))"

# ──────────────────────────────────────────────────────────────────
# 1. openjdk (Freerouting needs Java 21+)
# ──────────────────────────────────────────────────────────────────
if command -v java >/dev/null 2>&1 && java -version 2>&1 | grep -qE "version \"(2[1-9]|[3-9][0-9])"; then
    echo "[1/6] java: already $(java -version 2>&1 | head -1 | sed 's/.*version //') — skip"
else
    echo "[1/6] installing openjdk>=21 via conda-forge ..."
    conda install -c conda-forge -y "openjdk>=21" >/dev/null
fi

# ──────────────────────────────────────────────────────────────────
# 2. rust (KRT's Rust grid_router.so — rebuilt at first invocation
#    against the host's glibc; the upstream pre-built copy targets a
#    newer glibc than e.g. Ubuntu 20.04 ships)
# ──────────────────────────────────────────────────────────────────
if command -v cargo >/dev/null 2>&1; then
    echo "[2/6] cargo: already $(cargo --version | head -1) — skip"
else
    echo "[2/6] installing rust via conda-forge ..."
    conda install -c conda-forge -y rust >/dev/null
fi

# ──────────────────────────────────────────────────────────────────
# 3. cupy (OrthoRoute GPU path)
# ──────────────────────────────────────────────────────────────────
if "${PY}" -c "import cupy" 2>/dev/null; then
    echo "[3/6] cupy: already importable — skip"
else
    echo "[3/6] installing cupy-cuda12x via pip ..."
    "${PIP[@]}" install --quiet "cupy-cuda12x"
fi

# ──────────────────────────────────────────────────────────────────
# 4. OrthoRoute editable install
# ──────────────────────────────────────────────────────────────────
if "${PY}" -c "import orthoroute" 2>/dev/null; then
    echo "[4/6] orthoroute: already importable — skip"
elif [ ! -f "${EXTERNAL_DIR}/OrthoRoute/main.py" ]; then
    echo "[4/6] orthoroute: ${EXTERNAL_DIR}/OrthoRoute missing — skip"
    echo "      (run 'git submodule update --init external/OrthoRoute' to enable it)"
else
    echo "[4/6] pip install -e ${EXTERNAL_DIR}/OrthoRoute ..."
    "${PIP[@]}" install --quiet -e "${EXTERNAL_DIR}/OrthoRoute"
fi

# ──────────────────────────────────────────────────────────────────
# 5. KRT runtime deps + krt-runner (thin wrapper) editable install
# ──────────────────────────────────────────────────────────────────
if "${PY}" -c "import scipy, shapely" 2>/dev/null; then
    echo "[5/6] krt deps: scipy/shapely already importable — skip"
else
    echo "[5/6] installing KRT runtime deps (scipy shapely) via pip ..."
    "${PIP[@]}" install --quiet scipy shapely
fi

if "${PY}" -c "import krt_runner" 2>/dev/null; then
    echo "[5/6] krt-runner: already importable — skip"
else
    echo "[5/6] pip install -e ${BASELINES_DIR}/krt ..."
    "${PIP[@]}" install --quiet -e "${BASELINES_DIR}/krt"
fi

# ──────────────────────────────────────────────────────────────────
# 6. Smoke tests
# ──────────────────────────────────────────────────────────────────
echo "=== smoke tests ==="

# Freerouting JAR (vendored release binary → external/freerouting/, auto-downloaded)
FR_JAR="${EXTERNAL_DIR}/freerouting/freerouting-2.1.0.jar"
FR_URL="https://github.com/freerouting/freerouting/releases/download/v2.1.0/freerouting-2.1.0.jar"
if [ ! -s "${FR_JAR}" ]; then
    echo "  freerouting: jar missing — downloading to ${FR_JAR} ..."
    mkdir -p "${EXTERNAL_DIR}/freerouting"
    curl -L -o "${FR_JAR}" "${FR_URL}"
fi
if [ ! -s "${FR_JAR}" ]; then
    echo "  freerouting: ERROR — ${FR_JAR} missing or empty" >&2
    exit 1
fi
FR_JAR_SIZE=$(stat -c%s "${FR_JAR}" 2>/dev/null || stat -f%z "${FR_JAR}")
echo "  freerouting: jar ok ($((FR_JAR_SIZE / 1024 / 1024)) MB)"
java -Djava.awt.headless=true -jar "${FR_JAR}" --help 2>&1 | head -3 \
    || echo "  freerouting: warning — --help returned non-zero (some versions exit 1 on help)"

# KRT thin wrapper
KRT_ROOT_DEFAULT="${EXTERNAL_DIR}/KiCadRoutingTools"   # pinned checkout: tools/setup/fetch_baselines.sh
KRT_ROOT="${KRT_ROOT:-${KRT_ROOT_DEFAULT}}"
if [ ! -f "${KRT_ROOT}/route.py" ]; then
    echo "  krt: ERROR — ${KRT_ROOT}/route.py missing (run tools/setup/fetch_baselines.sh or set KRT_ROOT)" >&2
    exit 1
fi
echo "  krt: route.py at ${KRT_ROOT}"
"${PY}" -c "from krt_runner import get_krt_root; print('  krt: get_krt_root() ->', get_krt_root())"

# OrthoRoute
if [ -f "${EXTERNAL_DIR}/OrthoRoute/main.py" ]; then
    "${PY}" -c "import orthoroute; print('  orthoroute: imported')"
else
    echo "  orthoroute: skipped — ${EXTERNAL_DIR}/OrthoRoute missing"
fi

echo "=== all set ==="
echo
echo "Example invocations:"
echo "  python ${BASELINES_DIR}/run_rule_based_routers.py --baseline freerouting --dataset synthetic_2l --limit 5"
echo "  python ${BASELINES_DIR}/run_rule_based_routers.py --baseline krt        --dataset pcbench      --limit 5"
echo "  python ${BASELINES_DIR}/run_rule_based_routers.py --baseline orthoroute --dataset synthetic_2l --limit 5"
echo
echo "See ${BASELINES_DIR}/README.md for full CLI / env-var documentation."
