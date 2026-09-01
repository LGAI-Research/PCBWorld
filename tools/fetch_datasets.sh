#!/usr/bin/env bash
# Stage / resolve datasets through the single-source resolver (configs/paths.yaml).
#
# Policy — repeated-read jobs (training/sweep) stage a
# local real-copy (var/datasets) to dodge shared-NFS tail latency (R3); single-pass
# jobs (eval) resolve directly. This is the CLI wrapper around configs.loader.paths.
#
# Run in the `cadagent` conda env (needs PyYAML). Usage:
#   tools/fetch_datasets.sh stage   <logical_name>   # ensure local copy, print its path
#   tools/fetch_datasets.sh resolve <logical_name>   # print read path (staged if present)
#   tools/fetch_datasets.sh list                     # list known logical dataset names
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmd="${1:-}"; name="${2:-}"
run() { PYTHONPATH="$REPO:${PYTHONPATH:-}" python -c "$1" "$name"; }
case "$cmd" in
  stage)   [ -n "$name" ] || { echo "usage: $0 stage <name>" >&2; exit 2; }
           run "import sys;from configs.loader.paths import stage_dataset;print(stage_dataset(sys.argv[1]))" ;;
  resolve) [ -n "$name" ] || { echo "usage: $0 resolve <name>" >&2; exit 2; }
           run "import sys;from configs.loader.paths import resolve_dataset;print(resolve_dataset(sys.argv[1]))" ;;
  list)    PYTHONPATH="$REPO:${PYTHONPATH:-}" python -c \
             "from configs.loader.paths import load_paths;print(chr(10).join(sorted(load_paths().datasets)))" ;;
  *)       echo "usage: $0 {stage|resolve|list} [name]" >&2; exit 2 ;;
esac
