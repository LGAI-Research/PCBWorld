#!/usr/bin/env bash
# ALTERNATIVE open-loop (one-shot) sweep — for the env-rollout (kicad-bench) sweep see
# scripts/run_kicadbench_qwen_together.sh, which is the primary path.
#
# Sweep Qwen3 family on Together's serverless endpoints, measuring
# success_rate@k and routability@k for k in {1,5,10,25} (best & mean).
#
# This launcher is for the OPEN-LOOP (one-shot) generation eval scripts
# (NOTE: targets the legacy eval_{cadgen,apiseq}_llm.py names, which now live
# only in the deprecated archive — this launcher predates the standalone rewrites):
#   - scripts/eval_cadgen_llm.py   (LLM emits raw segments+vias)
#   - scripts/eval_apiseq_llm.py   (LLM emits a single CAD-API action sequence)
# Both bypass the CadagentEnvs step loop — the LLM never observes intermediate
# routing state. For the agentic kicad-bench eval (one API call per env step,
# with state/action/reward feedback), use run_kicadbench_qwen_together.sh.
#
# DEFAULT BOARD SET: all 128 synth_2L_v2/test boards (LIMIT_SYNTH=0 = all).
#
# Per board we draw N_SAMPLES (= max k) completions, then post-hoc compute
# pass@k / routability@k_best / routability@k_mean for each k in KS using
# the first k samples. Outputs land under
#   $OUT_ROOT/$DATE_TAG/<sanitized-model>/<task>/<set>_<mode>/
#
# Auth: export TOGETHER_API_KEY=... before invoking.
#
# Usage:
#   bash scripts/run_qwen_together_sweep.sh dry     # 2 boards / model, dry-run
#   bash scripts/run_qwen_together_sweep.sh main    # 8 models, both tasks
#   bash scripts/run_qwen_together_sweep.sh coder   # Coder-480B, apiseq only
#   bash scripts/run_qwen_together_sweep.sh all     # main + coder
#   MODELS="Qwen/Qwen3-4B" bash scripts/run_qwen_together_sweep.sh main
#   TASKS="apiseq" bash scripts/run_qwen_together_sweep.sh main
#   SCENARIOS="synth_zs" bash scripts/run_qwen_together_sweep.sh main

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

# ── Eval set roots (mirror run_cadgen_llm.sh / run_apiseq_llm.sh) ──
# Defaults resolve through configs/paths.yaml (CADAGENT_DATA_ROOT overrides the
# root); each $(...) runs only when the env var is unset, and fails loudly on
# an empty data root.
_ds() { python -m configs.loader.paths resolve "$1"; }
SYNTH_2L_TEST_DIR="${SYNTH_2L_TEST_DIR:-$(_ds synth_2L_v2)/test}"
SYNTH_2L_VAL_DIR="${SYNTH_2L_VAL_DIR:-$(_ds synth_2L_v2)/val}"
REAL_BOARD_DIR="${REAL_BOARD_DIR:-$(_ds pcbench_exacad)}"
REAL_BOARD_NAME_FILTER="${REAL_BOARD_NAME_FILTER:-processed_v9_guide_v3.kicad_pcb}"

REAL_FEWSHOT_POOL="${REAL_FEWSHOT_POOL:-$(_ds pcbench_exacad)}"
REAL_FEWSHOT_NAME_FILTER="${REAL_FEWSHOT_NAME_FILTER:-processed_v9_guide_v3.kicad_pcb}"
SYNTH_FEWSHOT_CACHE="${SYNTH_FEWSHOT_CACHE:-${REPO_ROOT}/cache/synth_2L_fewshot}"
SYNTH_FEWSHOT_PREP_LIMIT="${SYNTH_FEWSHOT_PREP_LIMIT:-8}"
APISEQ_FEWSHOT_CACHE="${APISEQ_FEWSHOT_CACHE:-${REPO_ROOT}/cache/apiseq_fewshot}"

# ── @k sweep ──────────────────────────────────────────────────────
KS="${KS:-1,5,10,25}"
N_SAMPLES="${N_SAMPLES:-25}"          # = max(KS)
N_FEWSHOT="${N_FEWSHOT:-2}"
LIMIT_REAL="${LIMIT_REAL:-20}"
# 0 = no limit; the synth_2L_v2/test set holds all 128 boards.
LIMIT_SYNTH="${LIMIT_SYNTH:-0}"

TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
API_CONCURRENCY="${API_CONCURRENCY:-4}"
API_BASE_URL="${API_BASE_URL:-https://api.together.xyz/v1}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_out/qwen_together_sweep}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"

# ── Model lists ────────────────────────────────────────────────────
# Main sweep — 8 dense / MoE Qwen3 chat-family checkpoints on Together
# serverless. Override by exporting MODELS="model1 model2 ...".
DEFAULT_MAIN_MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "Qwen/Qwen3-235B-A22B"
    "Qwen/Qwen3.5-397B-A17B"
)
# Coder/agent variant — only swept on the API-Seq task (which IS an
# agentic API-call generation task).
DEFAULT_CODER_MODELS=(
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
)

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MAIN_MODELS <<<"$MODELS"
else
    MAIN_MODELS=("${DEFAULT_MAIN_MODELS[@]}")
fi
if [[ -n "${CODER_MODELS:-}" ]]; then
    read -r -a CODER_MODELS_ARR <<<"$CODER_MODELS"
else
    CODER_MODELS_ARR=("${DEFAULT_CODER_MODELS[@]}")
fi

# Scenarios to run per model. Default = synth-only (all 128 boards) since
# that is the headline target. Subset / extend by exporting SCENARIOS, e.g.
#   SCENARIOS="real_zs synth_zs"
#   SCENARIOS="real_zs real_fs synth_zs synth_fs"   # full 4-quadrant sweep
DEFAULT_SCENARIOS="synth_zs synth_fs"
SCENARIOS="${SCENARIOS:-$DEFAULT_SCENARIOS}"

# Tasks: cadgen, apiseq, or both. Override with TASKS="apiseq" etc.
DEFAULT_TASKS="cadgen apiseq"
TASKS="${TASKS:-$DEFAULT_TASKS}"

# ──────────────────────────────────────────────────────────────────

require_api_key() {
    if [[ -z "${TOGETHER_API_KEY:-}" ]]; then
        echo "[ERROR] TOGETHER_API_KEY is not set. export it before running." >&2
        exit 2
    fi
}

sanitize() {
    # Replace path-hostile characters in model names: '/' -> '__', '.' -> '_'
    echo "$1" | sed -e 's|/|__|g' -e 's|\.|_|g'
}

ensure_synth_fewshot_cache() {
    if [[ -d "$SYNTH_FEWSHOT_CACHE" ]] \
       && compgen -G "${SYNTH_FEWSHOT_CACHE}/*.kicad_pcb" >/dev/null; then
        return
    fi
    echo "  [building synth fewshot cache via PNS engine -> $SYNTH_FEWSHOT_CACHE]"
    python -u "$PY_DIR/prepare_synth_fewshot.py" \
        "$SYNTH_2L_VAL_DIR" \
        -o "$SYNTH_FEWSHOT_CACHE" \
        --limit "$SYNTH_FEWSHOT_PREP_LIMIT"
}

ensure_apiseq_fewshot_cache() {
    if [[ -d "$APISEQ_FEWSHOT_CACHE" ]] \
       && compgen -G "${APISEQ_FEWSHOT_CACHE}/*.json" >/dev/null; then
        return
    fi
    echo "  [building apiseq fewshot cache -> $APISEQ_FEWSHOT_CACHE]"
    python -u "$PY_DIR/prepare_plan_only_fewshot.py" \
        "$SYNTH_2L_VAL_DIR" \
        -o "$APISEQ_FEWSHOT_CACHE" \
        --limit "$SYNTH_FEWSHOT_PREP_LIMIT"
}

run_one() {
    # $1 task = cadgen | apiseq
    # $2 set  = real | synth_2L
    # $3 mode = zero_shot | few_shot
    # $4 model
    # $5 extra flags (e.g. --dry-run)
    local task="$1" set_name="$2" mode="$3" model="$4" extra_flags="${5:-}"

    local model_tag; model_tag="$(sanitize "$model")"
    local out_dir="${OUT_ROOT}/${DATE_TAG}/${model_tag}/${task}/${set_name}_${mode}"
    mkdir -p "$out_dir"

    local script_name
    if [[ "$task" == "cadgen" ]]; then
        script_name="eval_cadgen_llm.py"
    else
        script_name="eval_apiseq_llm.py"
    fi

    # Build inputs argv
    local input_args=()
    local recursive_arg=()
    local limit_arg=()
    if [[ "$set_name" == "real" ]]; then
        mapfile -t real_files < <(
            find "$REAL_BOARD_DIR" -maxdepth 2 -type f \
                -name "$REAL_BOARD_NAME_FILTER" | sort | head -n "$LIMIT_REAL"
        )
        if [[ "${#real_files[@]}" -eq 0 ]]; then
            echo "[skip] no real boards matched $REAL_BOARD_NAME_FILTER under $REAL_BOARD_DIR"
            return
        fi
        input_args=("${real_files[@]}")
    else
        input_args=("$SYNTH_2L_TEST_DIR")
        limit_arg=(--limit "$LIMIT_SYNTH")
    fi

    # Few-shot args
    local fewshot_args=()
    if [[ "$mode" == "few_shot" ]]; then
        if [[ "$task" == "cadgen" ]]; then
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
        else
            ensure_apiseq_fewshot_cache
            fewshot_args=(
                --fewshot-pool "$APISEQ_FEWSHOT_CACHE"
                --num-fewshot "$N_FEWSHOT"
            )
        fi
    fi

    echo
    echo "=========================================================="
    echo "  task     : $task"
    echo "  scenario : ${set_name} / ${mode}"
    echo "  model    : $model"
    echo "  out      : $out_dir"
    echo "=========================================================="

    python -u "$SCRIPT_DIR/_archive/$script_name" \
        "${input_args[@]}" \
        "${limit_arg[@]}" \
        -o "$out_dir" \
        --mode "$mode" \
        --num-samples "$N_SAMPLES" \
        --ks "$KS" \
        --api-provider together \
        --api-model "$model" \
        --api-base-url "$API_BASE_URL" \
        --api-concurrency "$API_CONCURRENCY" \
        --temperature "$TEMPERATURE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        "${fewshot_args[@]}" \
        $extra_flags
}

scenario_to_args() {
    case "$1" in
        real_zs)  echo "real zero_shot" ;;
        real_fs)  echo "real few_shot" ;;
        synth_zs) echo "synth_2L zero_shot" ;;
        synth_fs) echo "synth_2L few_shot" ;;
        *) echo "[ERROR] unknown scenario: $1" >&2; exit 2 ;;
    esac
}

sweep_models() {
    local extra_flags="${1:-}"
    shift || true
    local -a tasks_arr; read -r -a tasks_arr <<<"$TASKS"
    local -a scen_arr; read -r -a scen_arr <<<"$SCENARIOS"
    local -a models_arr=("$@")

    for model in "${models_arr[@]}"; do
        for task in "${tasks_arr[@]}"; do
            for scen in "${scen_arr[@]}"; do
                read -r set_name mode <<<"$(scenario_to_args "$scen")"
                run_one "$task" "$set_name" "$mode" "$model" "$extra_flags" \
                    || echo "[WARN] $model / $task / $scen failed; continuing"
            done
        done
    done
}

cmd="${1:-main}"
case "$cmd" in
    main)
        require_api_key
        sweep_models "" "${MAIN_MODELS[@]}"
        ;;
    coder)
        # Coder/agent variant — restrict to apiseq by default.
        require_api_key
        TASKS="${TASKS_CODER:-apiseq}" \
            sweep_models "" "${CODER_MODELS_ARR[@]}"
        ;;
    all)
        require_api_key
        sweep_models "" "${MAIN_MODELS[@]}"
        TASKS="${TASKS_CODER:-apiseq}" \
            sweep_models "" "${CODER_MODELS_ARR[@]}"
        ;;
    dry)
        # Tiny smoke run — 2 boards / model, both tasks, no API.
        LIMIT_REAL=2 LIMIT_SYNTH=2 N_SAMPLES=4 KS="1,2,4" \
            sweep_models "--dry-run" "${MAIN_MODELS[0]}"
        ;;
    *)
        echo "usage: $0 {main|coder|all|dry}" >&2
        exit 2
        ;;
esac

echo
echo "=========================================================="
echo "  Done. Run aggregator next:"
echo "    python scripts/aggregate_qwen_together_sweep.py \\"
echo "      ${OUT_ROOT}/${DATE_TAG}"
echo "=========================================================="
