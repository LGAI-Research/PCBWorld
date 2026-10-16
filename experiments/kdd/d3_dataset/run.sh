#!/usr/bin/env bash
# Build configs/datasets/d3_v2.json — the live v10 D3 split used by the LLM
# quickstart aliases d3a / d3b / d3c. The historical v1 d3.json is frozen.
#
# Defaults rebuild the v10 file against the canonical PCB-bench source
# (pcb_characteristics_exacad_sorted.csv + the exacad_sorted_v10 folder of
# NNNN_<source-id>/ board dirs). Override --csv / --sorted-dir / --out to point at a fork.
#
# Usage:
#   bash experiments/kdd/d3_dataset/run.sh                          # rebuild d3_v2.json
#   bash experiments/kdd/d3_dataset/run.sh --out /tmp/d3_v2.json
#   bash experiments/kdd/d3_dataset/run.sh -- --medium-n 20         # forward extra flags
#
# Flags (anything after `--` is forwarded verbatim to build.py):
#   --csv <path>        CSV with sample,sorted_id,nets,components,pins,layers
#                       (default: <sorted-dir>/pcb_characteristics_exacad_sorted.csv)
#   --sorted-dir <dir>  Directory containing NNNN_<source-id>/ board folders
#                       (default: $PCBENCH_SORTED_OUT or the v10 PCB-bench mount)
#   --out <path>        Output boards-json (default: configs/datasets/d3_v2.json)

set -euo pipefail
source "$(dirname "$0")/../../_lib/env.sh"
cd "$PCBWORLD_REPO_ROOT"

# Real-PCB corpus: PCBENCH_SORTED_OUT wins, else the v10 pcbench_exacad dataset
# from configs/paths.yaml (PCBWORLD_DATA_ROOT overrides the root; empty root fails loudly).
SORTED_DIR="${PCBENCH_SORTED_OUT:-$("$PYTHON_BIN" -m configs.loader.paths resolve pcbench_exacad)}"
CSV=""
OUT="${OUT:-configs/datasets/d3_v2.json}"
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
