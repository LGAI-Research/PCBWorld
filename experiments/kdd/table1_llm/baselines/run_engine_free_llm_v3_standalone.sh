#!/usr/bin/env bash
# Launcher for experiments/kdd/llm_eval/eval_engine_free_llm_v3_standalone.py — standalone v3 (no v1/v2 imports).
#
# Same {real, synth_2L} x {zero_shot, few_shot} matrix as v1/v2, but with:
#   - prompt phrased in standard EDA terms (octilinear routing — 8 directions:
#     4 cardinal + 4 diagonals at 45°), positioned between rectilinear/Manhattan
#     and fully Euclidean routing
#   - identical audit pipeline to v2 (angle_compliance_rate, success_strict)
#   - optional --strict-angle (env var STRICT_ANGLE=1) to make pass@k key
#     off success_strict instead of v1's connectivity-only success
#
# Outputs default to a separate root so v1 results are preserved:
#   eval_out/cadgen_llm_v3_standalone/<DATE_TAG>/<set>_<mode>/   (legacy on-disk root, kept)
#
# Usage:
#   bash run_engine_free_llm_v3_standalone.sh dry             # all 4 scenarios, dry-run
#   bash run_engine_free_llm_v3_standalone.sh real_zs
#   bash run_engine_free_llm_v3_standalone.sh real_fs
#   bash run_engine_free_llm_v3_standalone.sh synth_zs
#   bash run_engine_free_llm_v3_standalone.sh synth_fs
#   bash run_engine_free_llm_v3_standalone.sh all
#
#   STRICT_ANGLE=1 bash run_engine_free_llm_v3_standalone.sh synth_zs   # pass@k = strict

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"  # baselines→table1_llm→kdd→experiments→repo
# Python implementations moved to experiments/kdd/llm_eval/
PY_DIR="$REPO_ROOT/experiments/kdd/llm_eval"

cd "$REPO_ROOT"

# ── Environment bootstrap ──────────────────────────────────────────
# Conda activate (author's machine path). Guarded so other machines can
# activate cadagent externally and just source/run this script.
# CONDA_SETUP can be overridden per machine; defaults to ~/miniconda3.
_conda_setup="${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$_conda_setup" ] && [ "${CONDA_DEFAULT_ENV:-}" != "cadagent" ]; then
    # shellcheck disable=SC1090
    source "$_conda_setup"
    conda activate cadagent
fi
export PYTHONPATH="${REPO_ROOT}/build_rl/pcbnew/python/rl:${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/build_rl/lib:${LD_LIBRARY_PATH:-}"

# ── Eval set roots (mirror v1) ─────────────────────────────────────
# Defaults resolve through configs/paths.yaml (CADAGENT_DATA_ROOT overrides the
# root); each $(...) runs only when the env var is unset, and fails loudly on
# an empty data root.
_ds() { python -m configs.loader.paths resolve "$1"; }
SYNTH_2L_TEST_DIR="${SYNTH_2L_TEST_DIR:-$(_ds synth_2L_v2)/test}"
SYNTH_2L_VAL_DIR="${SYNTH_2L_VAL_DIR:-$(_ds synth_2L_v2)/val}"
REAL_BOARD_DIR="${REAL_BOARD_DIR:-$(_ds pcbench_exacad)}"
REAL_BOARD_NAME_FILTER="${REAL_BOARD_NAME_FILTER:-processed_v9_guide_v3.kicad_pcb}"

# ── Few-shot example pools ────────────────────────────────────────
REAL_FEWSHOT_POOL="${REAL_FEWSHOT_POOL:-$(_ds pcbench_exacad)}"
REAL_FEWSHOT_NAME_FILTER="${REAL_FEWSHOT_NAME_FILTER:-processed_v9_guide_v3.kicad_pcb}"
SYNTH_FEWSHOT_CACHE="${SYNTH_FEWSHOT_CACHE:-${REPO_ROOT}/cache/synth_2L_fewshot}"
SYNTH_FEWSHOT_PREP_LIMIT="${SYNTH_FEWSHOT_PREP_LIMIT:-8}"

# ── Sample sizes ──────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-5}"
N_FEWSHOT="${N_FEWSHOT:-2}"
LIMIT_REAL="${LIMIT_REAL:-0}"       # 0 = all 100 real boards (0001~0100 in the pcbench_exacad dataset)
LIMIT_SYNTH="${LIMIT_SYNTH:-128}"   # full synth_2L_v2/test set

# ── Model / sampling ──────────────────────────────────────────────
# Provider can be 'openai' (uses OPENAI_API_KEY) or 'together' (uses
# TOGETHER_API_KEY against https://api.together.xyz/v1, OpenAI-compatible
# endpoint). Together examples:
#   API_PROVIDER=together API_MODEL=Qwen/Qwen3-8B \
#       bash run_engine_free_llm_v3_standalone.sh synth_zs
API_PROVIDER="${API_PROVIDER:-openai}"
API_MODEL="${API_MODEL:-gpt-5.4-mini}"
API_BASE_URL="${API_BASE_URL:-https://api.together.xyz/v1}"   # only used when provider=together
API_CONCURRENCY="${API_CONCURRENCY:-4}"                       # parallel requests on together
THINKING="${THINKING:-auto}"                                  # auto | on | off  (Qwen3 thinking mode, together only)
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

# ── v2-specific knob ──────────────────────────────────────────────
# 0 = report angle_compliance_rate alongside v1's success/routability;
#     pass@k still keys off connectivity-only success.
# 1 = pass@k uses success_strict (success AND every segment 45°-aligned).
STRICT_ANGLE="${STRICT_ANGLE:-0}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_out/cadgen_llm_v3_standalone}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

# ──────────────────────────────────────────────────────────────────

ensure_synth_fewshot_cache() {
    if [[ -d "$SYNTH_FEWSHOT_CACHE" ]] \
       && compgen -G "${SYNTH_FEWSHOT_CACHE}/*.kicad_pcb" >/dev/null; then
        echo "  [synth fewshot cache present: $SYNTH_FEWSHOT_CACHE]"
        return
    fi
    echo "  [building synth fewshot cache via PNS engine -> $SYNTH_FEWSHOT_CACHE]"
    python -u "$PY_DIR/prepare_synth_fewshot.py" \
        "$SYNTH_2L_VAL_DIR" \
        -o "$SYNTH_FEWSHOT_CACHE" \
        --limit "$SYNTH_FEWSHOT_PREP_LIMIT"
}

run_scenario() {
    local mode="$1"          # zero_shot | few_shot
    local set_name="$2"      # real | synth_2L
    local extra_flags="${3:-}"

    local input_arg=""
    local recursive_arg=""
    local limit
    local real_files=()
    if [[ "$set_name" == "real" ]]; then
        # Resolve real boards 0001..0100 under exacad_sorted/. Each board
        # lives in a directory named "<bid>_<slug>"; we pick its
        # processed_v9_guide_v3.kicad_pcb. LIMIT_REAL caps the count
        # (default 0 = all 100).
        mapfile -t real_files < <(
            python3 - "$REPO_ROOT" "$REAL_BOARD_DIR" "$REAL_BOARD_NAME_FILTER" "$LIMIT_REAL" <<'PY'
import glob, sys
repo, exacad, fname, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out, missing = [], []
for i in range(1, 101):
    bid = f"{i:04d}"
    matches = sorted(glob.glob(f"{exacad}/{bid}_*/{fname}"))
    (out if matches else missing).append(matches[0] if matches else bid)
if missing:
    sys.stderr.write(f"[WARN] {len(missing)} board ids didn't resolve: {missing[:3]}...\n")
if limit > 0:
    out = out[:limit]
print("\n".join(out))
PY
        )
        if [[ "${#real_files[@]}" -eq 0 ]]; then
            echo "[skip] no real boards matched $REAL_BOARD_NAME_FILTER under $REAL_BOARD_DIR"
            return
        fi
        recursive_arg=""
        input_arg=""
        limit=0
    else
        input_arg="$SYNTH_2L_TEST_DIR"
        recursive_arg=""
        limit="$LIMIT_SYNTH"
    fi

    local fewshot_args=()
    if [[ "$mode" == "few_shot" ]]; then
        if [[ "$set_name" == "real" ]]; then
            fewshot_args=(
                --fewshot-pool "$REAL_FEWSHOT_POOL"
                --num-fewshot "$N_FEWSHOT"
                --fewshot-name-contains "$REAL_FEWSHOT_NAME_FILTER"
            )
        else
            ensure_synth_fewshot_cache
            fewshot_args=(
                --fewshot-pool "$SYNTH_FEWSHOT_CACHE"
                --num-fewshot "$N_FEWSHOT"
            )
        fi
    fi

    local strict_arg=()
    if [[ "$STRICT_ANGLE" == "1" ]]; then
        strict_arg=(--strict-angle)
    fi

    # QS_FLAT_OUT_DIR=1: wrapper invokes us once for one scenario; collapse
    # the timestamp + scenario subdirs so the result tree matches the
    # common layout /<eval-type>/per_board/<id>/sample_NN.*.
    local out_dir
    if [ "${QS_FLAT_OUT_DIR:-0}" = "1" ]; then
        out_dir="$OUT_ROOT"
    else
        out_dir="${OUT_ROOT}/${DATE_TAG}/${set_name}_${mode}"
    fi
    mkdir -p "$out_dir"

    echo
    echo "=========================================================="
    echo "  scenario : ${set_name} / ${mode}  (v3 octilinear, strict=$STRICT_ANGLE)"
    echo "  out      : ${out_dir}"
    echo "=========================================================="

    # Provider-specific extra flags. The Python evaluator only honors
    # --api-base-url / --api-concurrency for provider=together, so we
    # gate them here to keep openai invocations clean.
    local provider_args=(--api-provider "$API_PROVIDER")
    if [[ "$API_PROVIDER" == "together" ]]; then
        provider_args+=(--api-base-url "$API_BASE_URL")
        provider_args+=(--api-concurrency "$API_CONCURRENCY")
        provider_args+=(--enable-thinking "$THINKING")
    fi

    if [[ "$set_name" == "real" ]]; then
        python -u "$PY_DIR/eval_engine_free_llm_v3_standalone.py" \
            "${real_files[@]}" \
            -o "$out_dir" \
            --mode "$mode" \
            --num-samples "$N_SAMPLES" \
            "${provider_args[@]}" \
            --api-model "$API_MODEL" \
            --temperature "$TEMPERATURE" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            ${fewshot_args[@]+"${fewshot_args[@]}"} \
            ${strict_arg[@]+"${strict_arg[@]}"} \
            $extra_flags
    else
        python -u "$PY_DIR/eval_engine_free_llm_v3_standalone.py" \
            "$input_arg" \
            $recursive_arg \
            --limit "$limit" \
            -o "$out_dir" \
            --mode "$mode" \
            --num-samples "$N_SAMPLES" \
            "${provider_args[@]}" \
            --api-model "$API_MODEL" \
            --temperature "$TEMPERATURE" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            ${fewshot_args[@]+"${fewshot_args[@]}"} \
            ${strict_arg[@]+"${strict_arg[@]}"} \
            $extra_flags
    fi
}

cmd="${1:-all}"
case "$cmd" in
    real_zs)   run_scenario zero_shot real ;;
    real_fs)   run_scenario few_shot  real ;;
    synth_zs)  run_scenario zero_shot synth_2L ;;
    synth_fs)  run_scenario few_shot  synth_2L ;;
    all)
        run_scenario zero_shot synth_2L
        run_scenario few_shot  synth_2L
        run_scenario zero_shot real
        run_scenario few_shot  real
        ;;
    dry)
        LIMIT_SYNTH=2 LIMIT_REAL=2 run_scenario zero_shot synth_2L "--dry-run"
        LIMIT_SYNTH=2 LIMIT_REAL=2 run_scenario few_shot  synth_2L "--dry-run"
        LIMIT_SYNTH=2 LIMIT_REAL=2 run_scenario zero_shot real "--dry-run"
        LIMIT_SYNTH=2 LIMIT_REAL=2 run_scenario few_shot  real "--dry-run"
        ;;
    *)
        echo "usage: $0 {dry|real_zs|real_fs|synth_zs|synth_fs|all}"
        exit 2
        ;;
esac
