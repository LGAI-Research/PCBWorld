#!/usr/bin/env bash
# Table 2 — P@K / CP@K aggregator wrapper.
#
# Walks a Table 2 (or Table 1 (b)) experiment tree for the per-level
# overall.json artefacts produced by
# experiments/_lib/metrics/score_rollouts.py, then collapses them into
# <prefix>_long.csv / <prefix>_pivot.csv / <prefix>_summary.md under --output-dir.
#
# Usage:
#   bash experiments/kdd/table2/eval.sh \
#       --root          "$EXPR_ROOT/table2" \
#       --output-dir    "$EXPR_ROOT/table2/_report" \
#       --output-prefix table2 \
#       --k 5
#
# All flags are forwarded to experiments/kdd/llm_eval/aggregate_p_cp_at_k.py.

set -euo pipefail
source "$(dirname "$0")/../../_lib/env.sh"
cd "$CADAGENT_REPO_ROOT"

quickstart_python experiments/kdd/llm_eval/aggregate_p_cp_at_k.py "$@"
