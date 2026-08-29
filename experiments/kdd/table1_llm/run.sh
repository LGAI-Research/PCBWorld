#!/usr/bin/env bash
# Table 1 (b) — LLM agents (interactive evaluation).
#
# Drives ``methods/llm_agent/rollout/cadagent.py --mode api`` on a named benchmark
# split with a named model alias. The alias → provider/model and split →
# boards-json mappings are loaded from JSON configs in
# ``configs/quickstart/kdd/`` by default; override per-call with --config-models
# / --config-splits, or per-shell via QS_MODELS_CONFIG / QS_SPLITS_CONFIG.
#
# Usage:
#   bash experiments/kdd/table1_llm/run.sh \
#       --model gpt-5.4-mini --split d2a \
#       --out "$EXPR_ROOT/table1/llm/gpt54mini/d2a"
#
# Flags:
#   --model <alias>           (required)  see configs/quickstart/kdd/models.json
#   --split <alias>           (required)  see configs/quickstart/kdd/splits.json
#   --out <dir>               (required)  rollouts dumped under $OUT/raw/
#   --config-models <path>    override models.json
#   --config-splits <path>    override splits.json
#   --rollouts <int>          --rollout_episodes (default 5)
#   --limit <int>             temporarily truncate the resolved board split (default 0 = all)
#   --seed <int>              (default 42)
#   --max-steps <int>         (default 20)
#   --prompt <vN>             --prompt_version (default v5)
#   --extra-args "<...>"      raw flags forwarded to the cadagent rollout
#   --dry-run                 print the final python invocation and exit 0
#   -h | --help

set -euo pipefail

_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../_lib/llm_lib.sh
source "$_self_dir/../../_lib/llm_lib.sh"
qs_load_env

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# Defaults (overridable via flags below).
MODEL_ALIAS=""
SPLIT_ALIAS=""
OUT=""
ROLLOUTS=5
LIMIT=0
SEED=42
MAX_STEPS=20
PROMPT="v5"
EXTRA_ARGS=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL_ALIAS="$2"; shift 2 ;;
        --split)         SPLIT_ALIAS="$2"; shift 2 ;;
        --out)           OUT="$2"; shift 2 ;;
        --config-models) export QS_MODELS_CONFIG="$2"; shift 2 ;;
        --config-splits) export QS_SPLITS_CONFIG="$2"; shift 2 ;;
        --rollouts)      ROLLOUTS="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --max-steps)     MAX_STEPS="$2"; shift 2 ;;
        --prompt)        PROMPT="$2"; shift 2 ;;
        --extra-args)    EXTRA_ARGS="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               qs_die "unknown argument: $1 (try --help)" 2 ;;
    esac
done

[ -n "$MODEL_ALIAS" ] || qs_die "--model required (see $(qs_models_cfg_path))" 2
[ -n "$SPLIT_ALIAS" ] || qs_die "--split required (see $(qs_splits_cfg_path))" 2
[ -n "$OUT" ]         || qs_die "--out required" 2

qs_resolve_model "$MODEL_ALIAS"
qs_resolve_split "$SPLIT_ALIAS"

if [ "$LIMIT" != "0" ]; then
    QS_LIMITED_BOARDS_JSON="$(mktemp "/tmp/cadagent_qs_${SPLIT_ALIAS}_XXXXXX.json")"
    QS_SRC_BJ="$QS_BOARDS_JSON" QS_DIFF="$QS_DIFF" QS_SPLIT="$QS_SPLIT" \
    QS_LIMIT="$LIMIT" QS_OUT_BJ="$QS_LIMITED_BOARDS_JSON" python3 - <<'PY'
import json, os

src = os.environ["QS_SRC_BJ"]
diff = os.environ["QS_DIFF"]
split = os.environ["QS_SPLIT"]
limit = int(os.environ["QS_LIMIT"])
out = os.environ["QS_OUT_BJ"]

with open(src) as f:
    data = json.load(f)
boards = data.get(diff, {}).get(split)
if not isinstance(boards, list):
    raise SystemExit(f"{src} has no list at {diff}.{split}")
data[diff][split] = boards[:limit]
with open(out, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
    export QS_BOARDS_JSON="$QS_LIMITED_BOARDS_JSON"
fi

# Build the python command as an array so quoting survives the dry-run echo.
py_cmd=(
    python -m methods.llm_agent.rollout.cadagent
    --mode api
    --api_provider "$QS_API_PROVIDER"
    --api_model    "$QS_API_MODEL"
    --boards_order      round_robin
    --boards_json       "$QS_BOARDS_JSON"
    --boards_difficulty "$QS_DIFF"
    --boards_split      "$QS_SPLIT"
    --rollout_episodes  "$ROLLOUTS"
    --env_num           1
    --seed              "$SEED"
    --max_steps         "$MAX_STEPS"
    --state_format      sexpr
    --prompt_version    "$PROMPT"
    --dump_dir          "$OUT/interactive/_raw"
    --silent
)
# EXTRA_ARGS is intentionally word-split (raw flags forwarded by the user).
if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    extra=( $EXTRA_ARGS )
    py_cmd+=( "${extra[@]}" )
fi

# Print a banner + the full resolved invocation so logs are self-contained.
# Redact any secret passed via --api_key in the echoed command so the banner
# (which often gets tee'd to a log file) never leaks the key.
_cmd_str="${py_cmd[*]}"
_cmd_str="$(printf '%s' "$_cmd_str" | sed -E 's/(--api_key[[:space:]]+)[^[:space:]]+/\1***REDACTED***/g')"
{
    echo "[qs] repo_root   : $QS_REPO_ROOT"
    echo "[qs] models cfg  : $(qs_models_cfg_path)"
    echo "[qs] splits cfg  : $(qs_splits_cfg_path)"
    echo "[qs] model alias : $MODEL_ALIAS -> $QS_API_PROVIDER / $QS_API_MODEL"
    echo "[qs] split alias : $SPLIT_ALIAS -> $QS_BOARDS_JSON ($QS_DIFF/$QS_SPLIT)"
    echo "[qs] out         : $OUT"
    echo "[qs] cmd         : (cd $QS_REPO_ROOT && $_cmd_str)"
} >&2

if [ "$DRY_RUN" = "1" ]; then
    echo "[qs] dry-run — not executing." >&2
    exit 0
fi

mkdir -p "$OUT/interactive/_raw"

# Side-channel meta.json so post-hoc analysis can recover what was run
# without re-parsing the python argv. Lives alongside the per_board/ tree
# inside the eval-type subdir so the common layout
# ($OUT/<eval-type>/{per_board,meta.json,result_summary.csv}) stays uniform
# across interactive / engine-free / plan-only.
QS_OUT="$OUT/interactive" \
QS_MODEL_ALIAS="$MODEL_ALIAS" QS_SPLIT_ALIAS="$SPLIT_ALIAS" \
QS_ARGV_STR="$_cmd_str" \
python3 - <<'PY'
import datetime, json, os, subprocess
out = os.environ["QS_OUT"]
os.makedirs(out, exist_ok=True)
try:
    git_rev = subprocess.check_output(
        ["git", "-C", os.environ["QS_REPO_ROOT"], "rev-parse", "HEAD"],
        text=True,
    ).strip()
except Exception:
    git_rev = None
meta = {
    "model_alias":   os.environ["QS_MODEL_ALIAS"],
    "split_alias":   os.environ["QS_SPLIT_ALIAS"],
    "api_provider":  os.environ["QS_API_PROVIDER"],
    "api_model":     os.environ["QS_API_MODEL"],
    "boards_json":   os.environ["QS_BOARDS_JSON"],
    "boards_difficulty": os.environ["QS_DIFF"],
    "boards_split":  os.environ["QS_SPLIT"],
    "argv":          os.environ["QS_ARGV_STR"],
    "git_rev":       git_rev,
    "timestamp":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

cd "$QS_REPO_ROOT"
"${py_cmd[@]}"

# Post-process: convert the cadagent rollout's flat dump ($OUT/interactive/_raw/
# <board>_episode_NN_env_NN.*) into the common per_board layout
# ($OUT/interactive/per_board/<board>/sample_NN.*) so eval/metrics.py and
# every other downstream consumer sees the same shape across interactive /
# engine-free / plan-only.
QS_OUT_PCBWORLD="$OUT/interactive" python3 - <<'PY'
import re, shutil, sys
from collections import defaultdict
from pathlib import Path

out = Path(__import__("os").environ["QS_OUT_PCBWORLD"])
raw = out / "_raw"
if not raw.is_dir():
    sys.exit(0)  # nothing to do (e.g. the cadagent rollout never wrote)

pat = re.compile(r"^(?P<board>.+?)_episode_(?P<ep>\d+)_env_(?P<env>\d+)$")
groups: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
loose: list[Path] = []          # files that don't match the pattern
for f in sorted(raw.iterdir()):
    if not f.is_file():
        continue
    m = pat.match(f.stem)
    if m:
        groups[m.group("board")].append((int(m.group("ep")), int(m.group("env")), f))
    else:
        loose.append(f)

# Per-board: sort by (ep, env), assign a stable sample_NN index, move all
# suffixes (.kicad_pcb / .kicad_pro / .kicad_prl / .json) together.
for board_id, entries in groups.items():
    entries.sort(key=lambda x: (x[0], x[1]))
    idx_map: dict[tuple[int, int], int] = {}
    for ep, env, _ in entries:
        if (ep, env) not in idx_map:
            idx_map[(ep, env)] = len(idx_map)
    dest = out / "per_board" / board_id
    dest.mkdir(parents=True, exist_ok=True)
    for ep, env, src in entries:
        idx = idx_map[(ep, env)]
        shutil.move(str(src), str(dest / f"sample_{idx:02d}{src.suffix}"))

# Loose files (result_summary.csv etc.) — keep them at the eval-type root.
for f in loose:
    shutil.move(str(f), str(out / f.name))

try:
    raw.rmdir()
except OSError:
    pass  # leave it if something unexpected remains
PY
