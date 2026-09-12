#!/usr/bin/env bash

# Copies all files from a patch directory into its target, preserving directory structure.
# Usage: bash patcher.sh <target>
#
#   bash patcher.sh ragen        # applies RAGEN-patch      → RAGEN
#   bash patcher.sh verl-agent   # applies verl-agent-patch → verl-agent
#   bash patcher.sh all          # applies both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

apply_patch() {
    local name="$1"
    local patch_dir="$SCRIPT_DIR/${name}-patch"
    local target_dir="$SCRIPT_DIR/${name}"

    if [[ ! -d "$patch_dir" ]]; then
        echo "Error: patch directory not found: $patch_dir" >&2
        exit 1
    fi
    if [[ ! -d "$target_dir" ]]; then
        echo "Error: target directory not found: $target_dir" >&2
        exit 1
    fi

    echo "Patch source : $patch_dir"
    echo "Target       : $target_dir"
    echo ""

    while IFS= read -r -d '' src_file; do
        rel_path="${src_file#"$patch_dir/"}"
        dst_file="$target_dir/$rel_path"

        mkdir -p "$(dirname "$dst_file")"
        cp "$src_file" "$dst_file"
        echo "  updated: $rel_path"
    done < <(find "$patch_dir" -type f \
        ! -name ".DS_Store" \
        -print0)

    echo ""
    echo "Done: $name"
}

TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash patcher.sh <ragen|verl-agent|all>"
    exit 1
fi

case "$TARGET" in
    ragen)
        apply_patch "RAGEN"
        ;;
    verl-agent)
        apply_patch "verl-agent"
        ;;
    all)
        apply_patch "RAGEN"
        echo "----------------------------------------"
        apply_patch "verl-agent"
        ;;
    *)
        echo "Error: unknown target '$TARGET'. Use ragen, verl-agent, or all." >&2
        exit 1
        ;;
esac
