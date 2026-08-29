#!/usr/bin/env bash
# Table 2 — Code-level column.
#
# Adapter onto the LLM baseline ``run_engine_free_llm_v3_standalone.sh``
# (LLM emits raw KiCad (segment ...) / (via ...) lines, patched into the
# unrouted board and scored via eval.metrics.evaluate_one).
#
# Translates this PR's CLI surface (--model alias, --split alias) into the
# env-var contract the standalone launcher already exposes:
#     API_PROVIDER, API_MODEL          — model alias config (configs/quickstart/kdd/models.json)
#     SYNTH_2L_TEST_DIR / REAL_BOARD_DIR — split alias  (configs/quickstart/kdd/splits.json)
#     OUT_ROOT                          — --out
#     N_SAMPLES / LIMIT_*               — --samples / --limit
# The standalone launcher's "scenario" arg (synth_zs / synth_fs / real_zs /
# real_fs) is picked from --split × --mode.
#
# Usage:
#   bash experiments/kdd/table2/run_engine_free.sh \
#       --model gpt-5.4-mini --split d2a --mode zs \
#       --out "$EXPR_ROOT/table2/engine_free/gpt54mini/d2a"
#
# Flags:
#   --model <alias>           (required)  configs/quickstart/kdd/models.json
#   --split <alias>           (required)  d2a | d3a (configs/quickstart/kdd/splits.json; legacy t2/t3a accepted)
#   --out <dir>               (required)  → OUT_ROOT
#   --mode zs|fs              (default zs) zero-shot or few-shot
#   --samples <int>           (default 5) → N_SAMPLES
#   --limit <int>             (default 0 = all) → LIMIT_REAL / LIMIT_SYNTH
#   --strict-angle            → STRICT_ANGLE=1
#   --config-models <path>    override models.json
#   --config-splits <path>    override splits.json
#   --dry-run                 print resolved env + scenario and exit 0
#   -h | --help

set -euo pipefail

_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../_lib/llm_lib.sh
source "$_self_dir/../../_lib/llm_lib.sh"
qs_load_env

BASELINE_LAUNCHER="$QS_REPO_ROOT/experiments/kdd/table1_llm/baselines/run_engine_free_llm_v3_standalone.sh"

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

MODEL_ALIAS=""
SPLIT_ALIAS=""
OUT=""
MODE="zs"
SAMPLES=5
LIMIT=0
STRICT_ANGLE_FLAG=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL_ALIAS="$2"; shift 2 ;;
        --split)         SPLIT_ALIAS="$2"; shift 2 ;;
        --out)           OUT="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --samples)       SAMPLES="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --strict-angle)  STRICT_ANGLE_FLAG=1; shift ;;
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

# split alias → standalone scenario (synth_zs | synth_fs | real_zs | real_fs)
case "$SPLIT_ALIAS" in
    d2a|t2)  SCENARIO="synth_${MODE}" ;;
    d3a|t3a) SCENARIO="real_${MODE}" ;;
    *)   qs_die "split '$SPLIT_ALIAS' has no code-level scenario mapping (d2a|d3a only)" 2 ;;
esac

# split JSON's dataset_dirs[split] → board dir env var that the launcher reads.
# Pull it out of the splits.json the user pointed at (boards_json field).
BOARD_DIR="$(QS_BJ="$QS_BOARDS_JSON" QS_SPLIT="$QS_SPLIT" python3 - <<'PY'
import json, os, sys
cfg = json.load(open(os.environ["QS_BJ"]))
dd = cfg.get("dataset_dirs") or {}
path = dd.get(os.environ["QS_SPLIT"])
if not path:
    sys.stderr.write(f"boards_json {os.environ['QS_BJ']} has no dataset_dirs[{os.environ['QS_SPLIT']!r}]\n")
    sys.exit(1)
print(path)
PY
)"

# Wire splits.json data dirs to the launcher's env-var contract.
case "$SPLIT_ALIAS" in
    d2a|t2)  export SYNTH_2L_TEST_DIR="$BOARD_DIR"
         export LIMIT_SYNTH="$LIMIT" ;;
    d3a|t3a) export REAL_BOARD_DIR="$BOARD_DIR"
         export LIMIT_REAL="$LIMIT" ;;
esac

export API_PROVIDER="$QS_API_PROVIDER"
export API_MODEL="$QS_API_MODEL"
export N_SAMPLES="$SAMPLES"
# Common layout — $OUT is (model, split) scope; each eval-type lives in its
# own subdir so $OUT/<eval-type>/per_board/<id>/sample_NN.kicad_pcb stays
# the single shape eval/metrics.py walks. QS_FLAT_OUT_DIR collapses the baseline's
# own <DATE_TAG>/<scenario>/ subdirs so the result tree matches.
export OUT_ROOT="$OUT/engine_free"
export QS_FLAT_OUT_DIR=1
[ "$STRICT_ANGLE_FLAG" = "1" ] && export STRICT_ANGLE=1

{
    echo "[qs engine_free] launcher : $BASELINE_LAUNCHER"
    echo "[qs engine_free] scenario : $SCENARIO  (split=$SPLIT_ALIAS, mode=$MODE)"
    echo "[qs engine_free] model    : $MODEL_ALIAS -> $API_PROVIDER / $API_MODEL"
    echo "[qs engine_free] data     : $BOARD_DIR  (limit=$LIMIT)"
    echo "[qs engine_free] out      : $OUT_ROOT"
    [ "$STRICT_ANGLE_FLAG" = "1" ] && echo "[qs engine_free] strict-angle ON"
} >&2

if [ "$DRY_RUN" = "1" ]; then
    echo "[qs engine_free] dry-run — not executing." >&2
    exit 0
fi

exec bash "$BASELINE_LAUNCHER" "$SCENARIO"
