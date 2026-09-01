#!/usr/bin/env bash
# Table 2 — INTERACTIVE column.
#
# By design, Table 2's interactive column reuses Table 1(b)'s LLM rollouts
# verbatim (the earlier quick-start recipe). This wrapper is just an
# alias so the Table 2 commands in the docs stay self-consistent — for any
# real divergence, edit table1/llm/run.sh.
set -euo pipefail
_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --samples)
            args+=(--rollouts "$2")
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

exec "$_self_dir/../table1/llm/run.sh" "${args[@]}"
