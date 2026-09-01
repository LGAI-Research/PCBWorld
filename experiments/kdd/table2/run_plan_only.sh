#!/usr/bin/env bash
# Table 2 — API-level column.
#
# Adapter onto the LLM baseline ``run_plan_only_llm_v8_standalone.sh``
# (LLM emits one <actions>...</actions> block of CAD-API calls; replayed
# through a fresh PCBWorld, final state scored by eval.metrics).
#
# Same translation contract as run_engine_free.sh — only the underlying
# launcher differs.
#
# Usage:
#   bash experiments/kdd/table2/run_plan_only.sh \
#       --model gpt-5.4-mini --split d2a --mode zs \
#       --out "$EXPR_ROOT/table2/plan_only/gpt54mini/d2a"
#
# Flags:
#   --model <alias>           (required)
#   --split <alias>           (required)  d2a | d3a | d3b | d3c (legacy t2/t3* accepted)
#   --out <dir>               (required)  → OUT_ROOT
#   --mode zs|fs              (default zs)
#   --samples <int>           (default 5) → N_SAMPLES
#   --limit <int>             (default 0 = all)
#   --config-models <path>    override models.json
#   --config-splits <path>    override splits.json
#   --dry-run                 print resolved env + scenario and exit 0
#   -h | --help

set -euo pipefail

_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../_lib/llm_lib.sh
source "$_self_dir/../../_lib/llm_lib.sh"
qs_load_env

BASELINE_LAUNCHER="$QS_REPO_ROOT/experiments/kdd/table1_llm/baselines/run_plan_only_llm_v8_standalone.sh"

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

MODEL_ALIAS=""
SPLIT_ALIAS=""
OUT=""
MODE="zs"
SAMPLES=5
LIMIT=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL_ALIAS="$2"; shift 2 ;;
        --split)         SPLIT_ALIAS="$2"; shift 2 ;;
        --out)           OUT="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --samples)       SAMPLES="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --config-models) export QS_MODELS_CONFIG="$2"; shift 2 ;;
        --config-splits) export QS_SPLITS_CONFIG="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               qs_die "unknown argument: $1 (try --help)" 2 ;;
    esac
done

[ -n "$MODEL_ALIAS" ] || qs_die "--model required" 2
[ -n "$SPLIT_ALIAS" ] || qs_die "--split required" 2
[ -n "$OUT" ]         || qs_die "--out required" 2
case "$MODE" in zs|fs) ;; *) qs_die "--mode must be zs or fs, got: $MODE" 2 ;; esac

qs_resolve_model "$MODEL_ALIAS"
qs_resolve_split "$SPLIT_ALIAS"

case "$SPLIT_ALIAS" in
    d2a|t2)              SCENARIO="synth_${MODE}" ;;
    d3a|d3b|d3c|t3a|t3b|t3c) SCENARIO="real_${MODE}" ;;
    *)               qs_die "split '$SPLIT_ALIAS' has no api-level scenario mapping (d2a|d3a|d3b|d3c only)" 2 ;;
esac

BOARD_DIR="$(QS_BJ="$QS_BOARDS_JSON" QS_SPLIT="$QS_SPLIT" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["QS_REPO_ROOT"])
from configs.loader.paths import expand_data_path  # ${CADAGENT_DATA_ROOT} entries

cfg = json.load(open(os.environ["QS_BJ"]))
dd = cfg.get("dataset_dirs") or {}
path = dd.get(os.environ["QS_SPLIT"])
if not path:
    sys.stderr.write(f"boards_json {os.environ['QS_BJ']} has no dataset_dirs[{os.environ['QS_SPLIT']!r}]\n")
    sys.exit(1)
print(expand_data_path(path))
PY
)"

case "$SPLIT_ALIAS" in
    d2a|t2)          export SYNTH_2L_TEST_DIR="$BOARD_DIR"
                     export LIMIT_SYNTH="$LIMIT" ;;
    d3a|d3b|d3c|t3a|t3b|t3c) export REAL_BOARD_DIR="$BOARD_DIR"
                     export LIMIT_REAL="$LIMIT" ;;
esac

export API_PROVIDER="$QS_API_PROVIDER"
export API_MODEL="$QS_API_MODEL"
export N_SAMPLES="$SAMPLES"
# Drive baseline's board selection from the same boards_json as PCBWORLD —
# the launcher resolves these via board_loader.resolve_board_list so the
# API-level evaluator sees the exact same list (filtered by difficulty/split)
# as table1/llm/run.sh would for the same alias.
export BOARDS_JSON="$QS_BOARDS_JSON"
export BOARDS_DIFFICULTY="$QS_DIFF"
export BOARDS_SPLIT="$QS_SPLIT"
# Common layout — see run_engine_free.sh.
export OUT_ROOT="$OUT/plan_only"
export QS_FLAT_OUT_DIR=1

{
    echo "[qs plan_only] launcher : $BASELINE_LAUNCHER"
    echo "[qs plan_only] scenario : $SCENARIO  (split=$SPLIT_ALIAS, mode=$MODE)"
    echo "[qs plan_only] model    : $MODEL_ALIAS -> $API_PROVIDER / $API_MODEL"
    echo "[qs plan_only] data     : $BOARD_DIR  (limit=$LIMIT)"
    echo "[qs plan_only] out      : $OUT_ROOT"
} >&2

if [ "$DRY_RUN" = "1" ]; then
    echo "[qs plan_only] dry-run — not executing." >&2
    exit 0
fi

exec bash "$BASELINE_LAUNCHER" "$SCENARIO"
