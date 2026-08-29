#!/usr/bin/env bash
# Regenerate appendix W&B validation diagnostic figures.

set -euo pipefail
source "$(dirname "$0")/../../_lib/env.sh"
cd "$CADAGENT_REPO_ROOT"

OUT_OVERLEAF="${OUT_OVERLEAF:-${LOCAL_OUT}/figures/appendix}"
AUDIT_ROOT="${AUDIT_ROOT:-${EXPR_ROOT}/training_logs/wandb_audit}"
ARGS=(--overleaf-root "$OUT_OVERLEAF" --audit-root "$AUDIT_ROOT")
if [[ "${FETCH_WANDB:-0}" != "1" ]]; then
  ARGS+=(--skip-fetch)
fi
if [[ -n "${SAMPLES:-}" ]]; then
  ARGS+=(--samples "$SAMPLES")
fi

quickstart_python experiments/kdd/appendix_diagnostics/plot_validation_curves.py "${ARGS[@]}"
