#!/usr/bin/env bash
# Launcher for experiments/kdd/llm_eval/eval_plan_only_llm_v8_standalone.py — standalone v8 (no v1/v2/v3 imports).
# rewritten prompt: same {API, BOARD reading, layer-choice,
# state-machine, anti-patterns} structure as earlier versions but a
# rewritten per-topology section that prefers `make_line` over `finish`,
# and Case C uses a P1' intermediate point with an explicit
# `start_route` restart after `make_via`.
#
# Few-shot examples are emitted by v1's default build_user_prompt
# (no plan injection from v4, no reordered block from v5) — they sit
# in a leading "## Examples" block above the task target board. Point
# --fewshot-pool at any cache that follows the
# {board_static, action_sequence} JSON schema, e.g.:
#     cache/synth_2L_apiseq_fewshot       (deterministic auto-router)
#     cache/synth_2L_apiseq_fewshot_llm   (LLM-generated, rotability=1.0)
#
# Writes to eval_out/apiseq_llm_v8_standalone/ (legacy on-disk root — kept so
# cached responses and v1..v7 results stay intact).
# No strict-angle knob — PNS already enforces corner_mode = MITERED_45.
#
# Usage:
#   bash run_plan_only_llm_v8_standalone.sh dry           # all 4 scenarios, dry-run
#   bash run_plan_only_llm_v8_standalone.sh real_zs       # one scenario, real api
#   bash run_plan_only_llm_v8_standalone.sh real_fs
#   bash run_plan_only_llm_v8_standalone.sh synth_zs
#   bash run_plan_only_llm_v8_standalone.sh synth_fs
#   bash run_plan_only_llm_v8_standalone.sh all

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

# ── Eval set roots (mirror run_cadgen_llm.sh defaults) ─────────────
# Defaults resolve through configs/paths.yaml (CADAGENT_DATA_ROOT overrides the
# root); each $(...) runs only when the env var is unset, and fails loudly on
# an empty data root.
_ds() { python -m configs.loader.paths resolve "$1"; }
SYNTH_2L_TEST_DIR="${SYNTH_2L_TEST_DIR:-$(_ds synth_2L_v2)/test}"
SYNTH_2L_VAL_DIR="${SYNTH_2L_VAL_DIR:-$(_ds synth_2L_v2)/val}"
REAL_BOARD_DIR="${REAL_BOARD_DIR:-$(_ds pcbench_exacad)}"
REAL_BOARD_NAME_FILTER="${REAL_BOARD_NAME_FILTER:-processed_v9_guide_v3.kicad_pcb}"

# ── Few-shot example pools ────────────────────────────────────────
# API-Seq pools live under their own cache dirs (board_static + action
# sequence JSON, not routed PCB). Real pool is built once on demand
# from the same source as run_cadgen_llm.sh's REAL_FEWSHOT_POOL.
SYNTH_APISEQ_FEWSHOT_CACHE="${SYNTH_APISEQ_FEWSHOT_CACHE:-${REPO_ROOT}/cache/synth_2L_apiseq_fewshot_llm}"
REAL_APISEQ_FEWSHOT_CACHE="${REAL_APISEQ_FEWSHOT_CACHE:-${REPO_ROOT}/cache/real_apiseq_fewshot_llm}"
APISEQ_FEWSHOT_PREP_LIMIT="${APISEQ_FEWSHOT_PREP_LIMIT:-8}"
# When building the real cache, harvest from a routed real-board pool —
# the auto-router still works on routed boards (it strips/re-routes via
# the env), so the resulting examples are deterministic regardless.
REAL_APISEQ_PREP_DIR="${REAL_APISEQ_PREP_DIR:-${REAL_BOARD_DIR}}"
REAL_APISEQ_PREP_FILTER="${REAL_APISEQ_PREP_FILTER:-$REAL_BOARD_NAME_FILTER}"

# ── Sample sizes ──────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-5}"
N_FEWSHOT="${N_FEWSHOT:-2}"
LIMIT_REAL="${LIMIT_REAL:-0}"       # 0 = all 100 real boards (0001~0100 in the pcbench_exacad dataset)
LIMIT_SYNTH="${LIMIT_SYNTH:-128}"   # full synth_2L_v2/test set (128 boards)

# ── Model / sampling ──────────────────────────────────────────────
# Provider can be 'openai' (uses OPENAI_API_KEY) or 'together' (uses
# TOGETHER_API_KEY against https://api.together.xyz/v1, OpenAI-compatible
# endpoint). Together examples:
#   API_PROVIDER=together API_MODEL=Qwen/Qwen3-8B \
#       bash run_plan_only_llm_v8_standalone.sh synth_fs
API_PROVIDER="${API_PROVIDER:-openai}"
API_MODEL="${API_MODEL:-gpt-5.4}"
API_BASE_URL="${API_BASE_URL:-https://api.together.xyz/v1}"   # only used when provider=together
API_CONCURRENCY="${API_CONCURRENCY:-4}"                       # parallel requests on together
THINKING="${THINKING:-auto}"                                  # auto | on | off  (Qwen3 thinking mode, together only)
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_out/apiseq_llm_v8_standalone}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

# ──────────────────────────────────────────────────────────────────

ensure_synth_apiseq_cache() {
    if [[ -d "$SYNTH_APISEQ_FEWSHOT_CACHE" ]] \
       && compgen -G "${SYNTH_APISEQ_FEWSHOT_CACHE}/*.json" >/dev/null; then
        echo "  [synth apiseq fewshot cache present: $SYNTH_APISEQ_FEWSHOT_CACHE]"
        return
    fi
    # v5 needs the LLM-generated cache (routability=1.0 entries). We
    # don't auto-build it because that costs API tokens — emit an
    # actionable error so the user runs the prep script themselves.
    cat >&2 <<EOF
  [ERROR] synth apiseq LLM cache missing: $SYNTH_APISEQ_FEWSHOT_CACHE
          v5 requires routability=1.0 examples generated by
          prepare_plan_only_fewshot_llm.py. Build it first:

              python experiments/kdd/llm_eval/prepare_plan_only_fewshot_llm.py \\
                  --src-dir $SYNTH_2L_VAL_DIR \\
                  -o $SYNTH_APISEQ_FEWSHOT_CACHE \\
                  --limit 16 --top-k 6 \\
                  --api-model gpt-5.4 \\
                  --routability-min 1.0 --success-only

          Or override SYNTH_APISEQ_FEWSHOT_CACHE to point at an
          existing LLM-generated cache.
EOF
    exit 2
}

ensure_real_apiseq_cache() {
    if [[ -d "$REAL_APISEQ_FEWSHOT_CACHE" ]] \
       && compgen -G "${REAL_APISEQ_FEWSHOT_CACHE}/*.json" >/dev/null; then
        echo "  [real apiseq fewshot cache present: $REAL_APISEQ_FEWSHOT_CACHE]"
        return
    fi
    # v5 expects an LLM-generated cache. Stage easy real boards into a
    # flat dir and tell the user how to populate the cache themselves.
    cat >&2 <<EOF
  [ERROR] real apiseq LLM cache missing: $REAL_APISEQ_FEWSHOT_CACHE
          v5 requires routability=1.0 examples generated by
          prepare_plan_only_fewshot_llm.py. Build it first, e.g.:

              # stage a few easy real boards into a flat dir
              mkdir -p tmp_real_easy
              for n in 0001 0002 0003; do
                  src=\$(find $REAL_APISEQ_PREP_DIR -maxdepth 2 \\
                      -path "*\$n*" -name '$REAL_APISEQ_PREP_FILTER' | head -1)
                  ln -sf "\$src" tmp_real_easy/\$(basename "\$(dirname "\$src")")_\$(basename "\$src")
                  # also link the .kicad_pro
                  pro="\${src%.kicad_pcb}.kicad_pro"
                  [ -f "\$pro" ] && ln -sf "\$pro" tmp_real_easy/\$(basename "\$(dirname "\$src")")_\$(basename "\$pro")
              done
              python experiments/kdd/llm_eval/prepare_plan_only_fewshot_llm.py \\
                  --src-dir tmp_real_easy \\
                  -o $REAL_APISEQ_FEWSHOT_CACHE \\
                  --limit 6 --top-k 3 --api-model gpt-5.4 \\
                  --routability-min 1.0 --success-only

          Or override REAL_APISEQ_FEWSHOT_CACHE.
EOF
    exit 2
}

run_scenario() {
    local mode="$1"          # zero_shot | few_shot
    local set_name="$2"      # real | synth_2L
    local extra_flags="${3:-}"

    local input_arg=""
    local recursive_arg=""
    local limit
    local real_files=()
    if [[ -n "${BOARDS_JSON:-}" ]]; then
        # BOARDS_JSON / BOARDS_DIFFICULTY / BOARDS_SPLIT take precedence over
        # the legacy 0001..0100 hardcode and the synth-dir glob. We resolve
        # through board_loader.resolve_board_list — the same utility
        # PCBWORLD (methods/llm_agent/rollout/cadagent.py) uses — so a given
        # (alias, difficulty, split) yields a byte-identical board list on
        # both eval paths. The per-board-dir layout used by d3
        # (<root>/<bid>/processed_v9_guide_v3.kicad_pcb) is picked up from
        # the boards_json's top-level "board_filename" metadata.
        mapfile -t real_files < <(
            BOARDS_JSON="$BOARDS_JSON" \
            BOARDS_DIFFICULTY="${BOARDS_DIFFICULTY:-easy}" \
            BOARDS_SPLIT="${BOARDS_SPLIT:-test}" \
            REPO_ROOT="$REPO_ROOT" \
            python3 - <<'PY'
import contextlib, os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
from methods._shared.board_loader import resolve_board_list
# resolve_board_list prints curriculum progress to stdout; we only want
# the board paths on this script's stdout so mapfile reads cleanly.
with contextlib.redirect_stdout(sys.stderr):
    paths, _ = resolve_board_list(
        boards_order="round_robin",
        single_board="",
        boards_json=os.environ["BOARDS_JSON"],
        difficulty=os.environ["BOARDS_DIFFICULTY"],
        split=os.environ["BOARDS_SPLIT"],
    )
sys.stdout.write("\n".join(paths) + ("\n" if paths else ""))
PY
        )
        local _cap
        if [[ "$set_name" == "real" ]]; then _cap="$LIMIT_REAL"; else _cap="$LIMIT_SYNTH"; fi
        if [[ "$_cap" -gt 0 && "${#real_files[@]}" -gt "$_cap" ]]; then
            real_files=("${real_files[@]:0:$_cap}")
        fi
        if [[ "${#real_files[@]}" -eq 0 ]]; then
            echo "[skip] BOARDS_JSON resolved zero boards (${BOARDS_DIFFICULTY:-easy}/${BOARDS_SPLIT:-test}) — $BOARDS_JSON"
            return
        fi
        limit=0
    elif [[ "$set_name" == "real" ]]; then
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
        limit=0
    else
        input_arg="$SYNTH_2L_TEST_DIR"
        recursive_arg=""
        limit="$LIMIT_SYNTH"
    fi

    local fewshot_args=()
    if [[ "$mode" == "few_shot" ]]; then
        if [[ "$set_name" == "real" ]]; then
            ensure_real_apiseq_cache
            fewshot_args=(
                --fewshot-pool "$REAL_APISEQ_FEWSHOT_CACHE"
                --num-fewshot "$N_FEWSHOT"
            )
        else
            ensure_synth_apiseq_cache
            fewshot_args=(
                --fewshot-pool "$SYNTH_APISEQ_FEWSHOT_CACHE"
                --num-fewshot "$N_FEWSHOT"
            )
        fi
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
    echo "  scenario : ${set_name} / ${mode}"
    echo "  out      : ${out_dir}"
    echo "=========================================================="

    # Provider-specific extra flags. The Python evaluator only honors
    # --api-base-url / --api-concurrency for provider=together.
    local provider_args=(--api-provider "$API_PROVIDER")
    if [[ "$API_PROVIDER" == "together" ]]; then
        provider_args+=(--api-base-url "$API_BASE_URL")
        provider_args+=(--api-concurrency "$API_CONCURRENCY")
        provider_args+=(--enable-thinking "$THINKING")
    fi

    # File-list invocation path is used whenever real_files[] is populated —
    # either from the legacy real-board 0001..0100 resolution or from the new
    # BOARDS_JSON-driven resolution (real OR synth).
    if [[ "${#real_files[@]}" -gt 0 ]]; then
        python -u "$PY_DIR/eval_plan_only_llm_v8_standalone.py" \
            "${real_files[@]}" \
            -o "$out_dir" \
            --mode "$mode" \
            --num-samples "$N_SAMPLES" \
            "${provider_args[@]}" \
            --api-model "$API_MODEL" \
            --temperature "$TEMPERATURE" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            ${fewshot_args[@]+"${fewshot_args[@]}"} \
            $extra_flags
    else
        python -u "$PY_DIR/eval_plan_only_llm_v8_standalone.py" \
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
