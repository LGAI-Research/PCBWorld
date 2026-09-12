#!/usr/bin/env bash
# Create per-run physical copies of the combined_v2 datasets so each training
# run can read/write its own .kicad_pro.lck without contending on NFS.
#
# Layout (dereferenced, real files — no symlinks):
#   pcb_dataset_10net_2pin_1layer_combined_v2__r{01..10}/   (~279MB each)
#   pcb_dataset_multi_pin_2layer_combined_v2__r{01..06}/    (~239MB each)
#
# Total disk: ~4.2GB. Idempotent: existing non-empty indexed dirs are skipped.
#
# Run:
#   bash tools/datagen/synthetic_generator/setup_indexed_datasets.sh
#
# To force re-copy: remove the target dir first (rm -rf ..._r0X).

set -euo pipefail

# Repo root: this script lives in tools/datagen/synthetic_generator/ (three levels
# below the root); override with REPO_DIR=... if the script is run from a copy.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_DIR"

SRC_1L="pcb_dataset_10net_2pin_1layer_combined_v2"
SRC_2L="pcb_dataset_multi_pin_2layer_combined_v2"

N_1L=10
N_2L=6

copy_one() {
    local src=$1 dst=$2
    if [[ -d "$dst" ]] && [[ -n "$(ls -A "$dst" 2>/dev/null)" ]]; then
        echo "[skip] $dst already exists and is non-empty"
        return 0
    fi
    mkdir -p "$dst"
    echo "[copy] $src -> $dst (dereference symlinks)"
    # -L follows symlinks, materializing real files into dst.
    cp -rL "$src"/. "$dst"/
}

echo "=== 1L copies (r01..r$(printf '%02d' $N_1L)) from $SRC_1L ==="
for i in $(seq 1 $N_1L); do
    idx=$(printf "%02d" "$i")
    copy_one "$SRC_1L" "${SRC_1L}__r${idx}"
done

echo ""
echo "=== 2L copies (r01..r$(printf '%02d' $N_2L)) from $SRC_2L ==="
for i in $(seq 1 $N_2L); do
    idx=$(printf "%02d" "$i")
    copy_one "$SRC_2L" "${SRC_2L}__r${idx}"
done

echo ""
echo "Done. Disk usage:"
du -sh "${SRC_1L}__r"* "${SRC_2L}__r"* 2>/dev/null | tail -20
