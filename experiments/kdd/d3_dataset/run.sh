#!/usr/bin/env bash
# Build configs/datasets/d3.json — the D3 (real-board) split used
# by the LLM quickstart aliases d3a / d3b / d3c.
#
# Defaults rebuild the file in-place against the canonical PCB-bench
# source (pcb_characteristics_exacad_sorted.csv + the exacad_sorted folder
# of NNNN_<name>/ board dirs). Override --csv / --sorted-dir / --out to
# point at a fork.
#
# Usage:
#   bash experiments/kdd/d3_dataset/run.sh                          # rebuild in place
#   bash experiments/kdd/d3_dataset/run.sh --out /tmp/d3.json       # dry copy
#   bash experiments/kdd/d3_dataset/run.sh -- --medium-n 20         # forward extra flags
#
# Flags (anything after `--` is forwarded verbatim to build.py):
#   --csv <path>        CSV with sample,nets,components,pins,layers
#                       (default: <sorted-dir>/pcb_characteristics_exacad_sorted.csv)
#   --sorted-dir <dir>  Directory containing NNNN_<name>/ board folders
#                       (default: $T3_SORTED_DIR or the PCB-bench mount)
#   --out <path>        Output boards-json
#                       (default: $CADAGENT_REPO_ROOT/configs/datasets/d3.json)

set -euo pipefail
source "$(dirname "$0")/../../_lib/env.sh"
cd "$CADAGENT_REPO_ROOT"

# Real-PCB corpus: T3_SORTED_DIR wins, else $CADAGENT_DATA_ROOT/pcbench/exacad_sorted.
SORTED_DIR="${T3_SORTED_DIR:-${CADAGENT_DATA_ROOT:?set CADAGENT_DATA_ROOT to your dataset root, or pass T3_SORTED_DIR}/pcbench/exacad_sorted}"
CSV=""
OUT="${OUT:-configs/datasets/d3.json}"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --csv)        CSV="$2"; shift 2 ;;
    --sorted-dir) SORTED_DIR="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --)           shift; EXTRA=("$@"); break ;;
    -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

if [ -z "$CSV" ]; then
  CSV="$SORTED_DIR/pcb_characteristics_exacad_sorted.csv"
fi

quickstart_log "csv         : $CSV"
quickstart_log "sorted-dir  : $SORTED_DIR"
quickstart_log "out         : $OUT"

quickstart_python experiments/kdd/d3_dataset/build.py \
  --csv        "$CSV" \
  --sorted-dir "$SORTED_DIR" \
  --out        "$OUT" \
  "${EXTRA[@]}"
