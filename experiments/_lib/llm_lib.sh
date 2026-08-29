#!/usr/bin/env bash
# Shared helpers for the LLM-eval wrappers (experiments/kdd/table1_llm, experiments/table2).
#
# Source this file from a wrapper:
#   source "$(dirname "${BASH_SOURCE[0]}")/../_lib/llm_lib.sh"
#
# After sourcing, the following are available:
#
#   QS_REPO_ROOT                          — absolute path to the cadagent repo.
#   qs_models_cfg_path                    — echoes the active models.json path.
#   qs_splits_cfg_path                    — echoes the active splits.json path.
#   qs_resolve_model <alias>              — exports QS_API_PROVIDER / QS_API_MODEL.
#   qs_resolve_split <alias>              — exports QS_BOARDS_JSON (abs) /
#                                           QS_DIFF / QS_SPLIT.
#   qs_die <msg> [exit_code]              — prints to stderr and exits.
#
# Override the default config paths with either:
#   --config-models <path> / --config-splits <path>   (from the wrapper CLI)
#   QS_MODELS_CONFIG=... QS_SPLITS_CONFIG=...         (from the environment)

set -euo pipefail

# Resolve repo root without hardcoding: this file lives at
# <repo>/experiments/_lib/llm_lib.sh, so the repo is two levels up.
# Fall back to ``git rev-parse`` if the directory layout is ever moved.
_qs_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if QS_REPO_ROOT="$(cd "$_qs_self_dir/../.." && pwd)" && [ -d "$QS_REPO_ROOT/methods/llm_agent" ]; then
    :
else
    QS_REPO_ROOT="$(git -C "$_qs_self_dir" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "${QS_REPO_ROOT:-}" ] || [ ! -d "$QS_REPO_ROOT/methods/llm_agent" ]; then
    echo "[qs lib] could not resolve QS_REPO_ROOT (expected <repo>/methods/llm_agent to exist)" >&2
    exit 1
fi
export QS_REPO_ROOT

qs_die() {
    local msg="${1:-error}"
    local code="${2:-1}"
    echo "[qs] $msg" >&2
    exit "$code"
}

# qs_load_env
#   Mirrors the prelude used by ./eval-api.sh:
#     1. Source $REPO_ROOT/.env if present (per-provider API keys; different
#        providers need different keys, so the .env hand-off is intentional).
#     2. Prepend KiCad C++ bindings + repo root to PYTHONPATH (per README).
#        verl-agent is *not* added: methods/llm_agent/wrappers/memory.py inlines SimpleMemory
#        precisely so the LLM eval path does not need external/verl-agent on
#        sys.path.
#     3. macOS-only OMP duplicate-init workaround (KMP_DUPLICATE_LIB_OK).
qs_load_env() {
    if [ -f "$QS_REPO_ROOT/.env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$QS_REPO_ROOT/.env"
        set +a
    fi
    export PYTHONPATH="$QS_REPO_ROOT:$QS_REPO_ROOT/build_rl/pcbnew/python/rl:${PYTHONPATH:-}"
    if [ "$(uname -s)" = "Darwin" ]; then
        export KMP_DUPLICATE_LIB_OK=TRUE
    fi
    # SSL CA bundle — conda Python on macOS often fails to locate system
    # certificates; pointing at certifi's bundle fixes urllib/httpx/openai
    # SDK ``CERTIFICATE_VERIFY_FAILED`` errors. If the user already has
    # SSL_CERT_FILE pointing at a custom CA (e.g. a corporate proxy CA),
    # we union it with certifi's Mozilla bundle so both internal and
    # external HTTPS endpoints validate.
    local _certifi
    _certifi="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
    if [ -n "$_certifi" ] && [ -f "$_certifi" ]; then
        if [ -z "${SSL_CERT_FILE:-}" ] || [ "$SSL_CERT_FILE" = "$_certifi" ]; then
            export SSL_CERT_FILE="$_certifi"
        else
            # Merge user's CA with certifi into a deterministic cached bundle.
            local _user_ca="$SSL_CERT_FILE"
            local _bundle="$HOME/.cache/cadagent/ssl-ca-bundle.pem"
            mkdir -p "$(dirname "$_bundle")"
            if [ ! -f "$_bundle" ] \
               || [ "$_user_ca" -nt "$_bundle" ] \
               || [ "$_certifi" -nt "$_bundle" ]; then
                cat "$_certifi" "$_user_ca" > "$_bundle"
            fi
            export SSL_CERT_FILE="$_bundle"
        fi
    fi
}

qs_models_cfg_path() {
    echo "${QS_MODELS_CONFIG:-$QS_REPO_ROOT/configs/quickstart/kdd/models.json}"
}

qs_splits_cfg_path() {
    echo "${QS_SPLITS_CONFIG:-$QS_REPO_ROOT/configs/quickstart/kdd/splits.json}"
}

# qs_resolve_model <alias>
#   Looks up <alias> in the active models config, then exports
#   QS_API_PROVIDER and QS_API_MODEL. Errors with the list of known aliases
#   if the alias is missing.
qs_resolve_model() {
    local alias="${1:-}"
    [ -n "$alias" ] || qs_die "qs_resolve_model: missing alias"
    local cfg
    cfg="$(qs_models_cfg_path)"
    [ -f "$cfg" ] || qs_die "models config not found: $cfg"

    # Python-side: parse JSON, fail loudly if alias unknown. Emit two lines:
    #   <api_provider>
    #   <api_model>
    local out
    if ! out="$(QS_ALIAS="$alias" QS_CFG="$cfg" python3 - <<'PY'
import json, os, sys
alias = os.environ["QS_ALIAS"]
cfg_path = os.environ["QS_CFG"]
with open(cfg_path) as f:
    cfg = json.load(f)
if alias not in cfg:
    known = " ".join(sorted(cfg.keys()))
    sys.stderr.write(
        f"unknown model alias: {alias!r} (known: {known})\n"
        f"config: {cfg_path}\n"
    )
    sys.exit(1)
entry = cfg[alias]
provider = entry.get("api_provider")
model = entry.get("api_model")
if not provider or not model:
    sys.stderr.write(f"models[{alias!r}] missing api_provider/api_model in {cfg_path}\n")
    sys.exit(1)
print(provider)
print(model)
PY
    )"; then
        exit 1
    fi
    QS_API_PROVIDER="$(printf '%s\n' "$out" | sed -n '1p')"
    QS_API_MODEL="$(printf '%s\n' "$out" | sed -n '2p')"
    export QS_API_PROVIDER QS_API_MODEL
}

# qs_resolve_split <alias>
#   Looks up <alias> in the active splits config, absolutises ``boards_json``
#   relative to QS_REPO_ROOT, and exports QS_BOARDS_JSON, QS_DIFF, QS_SPLIT.
qs_resolve_split() {
    local alias="${1:-}"
    [ -n "$alias" ] || qs_die "qs_resolve_split: missing alias"
    # Legacy t-series split aliases -> canonical d-series (paper alignment).
    case "$alias" in
        t2) alias="d2a" ;;
        t3a|t3b|t3c) alias="d${alias#t}" ;;
    esac
    local cfg
    cfg="$(qs_splits_cfg_path)"
    [ -f "$cfg" ] || qs_die "splits config not found: $cfg"

    local out
    if ! out="$(QS_ALIAS="$alias" QS_CFG="$cfg" QS_REPO_ROOT="$QS_REPO_ROOT" python3 - <<'PY'
import json, os, sys
alias = os.environ["QS_ALIAS"]
cfg_path = os.environ["QS_CFG"]
repo_root = os.environ["QS_REPO_ROOT"]
with open(cfg_path) as f:
    cfg = json.load(f)
if alias not in cfg:
    known = " ".join(sorted(cfg.keys()))
    sys.stderr.write(
        f"unknown split alias: {alias!r} (known: {known})\n"
        f"config: {cfg_path}\n"
    )
    sys.exit(1)
entry = cfg[alias]
boards_json = entry.get("boards_json")
diff = entry.get("difficulty")
split = entry.get("boards_split")
missing = [k for k, v in (("boards_json", boards_json), ("difficulty", diff), ("boards_split", split)) if not v]
if missing:
    sys.stderr.write(f"splits[{alias!r}] missing {missing} in {cfg_path}\n")
    sys.exit(1)
# Absolutise boards_json against repo root (config holds repo-relative paths).
if not os.path.isabs(boards_json):
    boards_json = os.path.join(repo_root, boards_json)
print(boards_json)
print(diff)
print(split)
PY
    )"; then
        exit 1
    fi
    QS_BOARDS_JSON="$(printf '%s\n' "$out" | sed -n '1p')"
    QS_DIFF="$(printf '%s\n' "$out" | sed -n '2p')"
    QS_SPLIT="$(printf '%s\n' "$out" | sed -n '3p')"
    export QS_BOARDS_JSON QS_DIFF QS_SPLIT
}
