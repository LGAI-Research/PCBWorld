#!/usr/bin/env bash
# diff_patches.sh — show what engine/kicad-patches/ actually changes vs the
# pinned KiCad upstream (engine/kicad-python/). Both live in the engine
# submodule; initialise it first.
#
# The patch tree stores FULL vendored files (build_rl_router.sh cp's them over a
# fresh upstream checkout), so `git diff` shows a whole file as "added" and the
# real delta is invisible. This reconstructs the per-file delta against the pinned
# submodule, so a 1821-line vendored file reads as its true ~30 changed lines.
#
# Scope: only the engine/kicad-patches/kicad/ subtree, whose paths mirror
# upstream 1:1. engine/kicad-patches/rl/ holds genuinely NEW RL source (no
# upstream counterpart) — those are reviewable as normal new files via git and are
# intentionally not covered here.
#
# Usage:
#   tools/build/diff_patches.sh              # summary table (file | +added | -removed)
#   tools/build/diff_patches.sh --show       # full unified diffs to stdout
#   tools/build/diff_patches.sh --emit=DIR   # write per-file .patch files under DIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENGINE_HOME="${PCBWORLD_ENGINE_HOME:-$ROOT/engine}"
PATCH_ROOT="$ENGINE_HOME/kicad-patches/kicad"
ORIG_ROOT="$ENGINE_HOME/kicad-python/kicad"

MODE="summary"
EMIT_DIR=""
for a in "$@"; do
    case "$a" in
        --show)    MODE="show" ;;
        --emit=*)  MODE="emit"; EMIT_DIR="${a#--emit=}" ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; $d'; exit 0 ;;
        *) echo "unknown arg: $a (try --help)" >&2; exit 2 ;;
    esac
done

[ -d "$ORIG_ROOT" ] || {
    echo "ERROR: upstream not found at $ORIG_ROOT" >&2
    echo "       (the engine submodule is not checked out)" >&2
    exit 1
}
[ "$MODE" = "emit" ] && mkdir -p "$EMIT_DIR"

tot_add=0; tot_del=0; n_files=0; n_missing=0
[ "$MODE" = "summary" ] && printf "%-44s %7s %7s\n" "file (rel to kicad-patches/kicad)" "+add" "-del"

while IFS= read -r patched; do
    rel="${patched#"$PATCH_ROOT"/}"
    orig="$ORIG_ROOT/$rel"
    if [ ! -f "$orig" ]; then
        n_missing=$((n_missing + 1))
        [ "$MODE" = "summary" ] && printf "%-44s %7s %7s\n" "$rel" "NEW" "-"
        continue
    fi

    # diff is expected to be non-empty (rc=1); rc>=2 is a real error.
    d="$(diff -u "$orig" "$patched" || true)"
    add=$(printf '%s\n' "$d" | grep -c '^+' || true); add=$((add > 0 ? add - 1 : 0))  # drop +++ header
    del=$(printf '%s\n' "$d" | grep -c '^-' || true); del=$((del > 0 ? del - 1 : 0))  # drop --- header
    [ -z "$d" ] && { add=0; del=0; }
    tot_add=$((tot_add + add)); tot_del=$((tot_del + del)); n_files=$((n_files + 1))

    case "$MODE" in
        summary) printf "%-44s %7s %7s\n" "$rel" "$add" "$del" ;;
        show)    [ -n "$d" ] && { echo "===== $rel ====="; printf '%s\n\n' "$d"; } ;;
        emit)    if [ -n "$d" ]; then
                     out="$EMIT_DIR/$rel.patch"; mkdir -p "$(dirname "$out")"
                     printf '%s\n' "$d" > "$out"
                 fi ;;
    esac
done < <(find "$PATCH_ROOT" -type f | sort)

if [ "$MODE" = "summary" ]; then
    printf -- "%-44s %7s %7s\n" "----- TOTAL ($n_files forked, $n_missing new) -----" "$tot_add" "$tot_del"
elif [ "$MODE" = "emit" ]; then
    echo "wrote .patch files under $EMIT_DIR ($n_files forked files)"
fi
