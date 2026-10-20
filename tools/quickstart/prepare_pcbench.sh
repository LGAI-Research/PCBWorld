#!/usr/bin/env bash
# Quick start: collect the open-source PCBench boards and preprocess them into the D3 benchmark set.
#
#   bash tools/quickstart/prepare_pcbench.sh [--limit N] [--workers N]
#
# One command for the five-step chain in tools/datagen/pcbench_prep/ (KiCad 5->9 conversion,
# DRC repair/filter, guide generation, difficulty sort, DSN/ORP export). Everything lives under
# PCBWORLD_DATA_ROOT: the PCBench clone at <root>/PCBench, the intermediate trees at
# <root>/pcbench_work/{v9,newdrc}, the result at <root>/pcbench/exacad_sorted/<NNNN>_<name>/
# (the layout configs/datasets/d3.json expects), and the .dsn/.orp siblings the rule-based routers
# read at <root>/pcbench/exacad_sorted_dsn/<NNNN>_<name>/. Any PCBENCH_* variable already exported wins.
# The chain runs kicad-cli and the pcbnew Python module; when the engine build lacks them this
# script adds them first (BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh).
#   --limit N     only the first N boards (a trial run); the default is all 1 182
#   --workers N   parallel workers for the conversion/repair steps (default 16)
# Every command it runs is echoed, and the resolved paths are printed at the end.
set -euo pipefail
cd "$(dirname "$0")/../.."

LIMIT=0; WORKERS=16
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
: "${PCBWORLD_DATA_ROOT:?export PCBWORLD_DATA_ROOT=<your dataset root> first (e.g. \$PWD/var/datasets)}"

run() { echo "+ $*" >&2; "$@"; }

export PCBENCH_PCBS_ROOT="${PCBENCH_PCBS_ROOT:-$PCBWORLD_DATA_ROOT/PCBench/PCBs}"
export PCBENCH_V9_ROOT="${PCBENCH_V9_ROOT:-$PCBWORLD_DATA_ROOT/pcbench_work/v9}"
export PCBENCH_NEWDRC_OUT="${PCBENCH_NEWDRC_OUT:-$PCBWORLD_DATA_ROOT/pcbench_work/newdrc}"
export PCBENCH_SORTED_OUT="${PCBENCH_SORTED_OUT:-$PCBWORLD_DATA_ROOT/pcbench/exacad_sorted}"
export PCBENCH_DSN_ROOT="${PCBENCH_DSN_ROOT:-$PCBWORLD_DATA_ROOT/pcbench/exacad_sorted_dsn}"

BUILD_DIR="${PCBWORLD_KICAD_RL_BUILD_DIR:-build_rl}"
if [ -z "${KICAD_CLI:-}" ] && [ ! -x "$BUILD_DIR/kicad/kicad-cli" ]; then
  echo "kicad-cli not in $BUILD_DIR: building it together with the pcbnew module (one-off, a few minutes)" >&2
  run env BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh
fi
if [ ! -d "$PCBENCH_PCBS_ROOT" ]; then
  run git clone --depth 1 https://github.com/PCBench/PCBench.git "$(dirname "$PCBENCH_PCBS_ROOT")"
fi

PP=tools/datagen/pcbench_prep
LIMIT_ARGS=(); [ "$LIMIT" -gt 0 ] && LIMIT_ARGS=(--limit "$LIMIT")
run python "$PP/convert_v9.py" "${LIMIT_ARGS[@]}" --workers "$WORKERS"
run python "$PP/drc_fix_v9.py" --workers "$WORKERS"
run python "$PP/make_guide.py" --base-dir "$PCBENCH_NEWDRC_OUT" --stem processed_v9 --suffix _guide_v3 --workers "$WORKERS"
run python "$PP/sort_prefix.py"
# Step 4 exits 3 when it finished and some boards are `fail` rows (the full corpus has such boards, all outside
# the D3 eval sets; the rule-based runner refuses a D3 tier missing one): that goes on. Any other failure stops here.
dsn_rc=0; run python "$PP/make_dsn_orp.py" || dsn_rc=$?
[ "$dsn_rc" = 0 ] || [ "$dsn_rc" = 3 ] || exit "$dsn_rc"

n=$(find "$PCBENCH_SORTED_OUT" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9]_*' | wc -l)
echo "[quickstart] pcbench: $n corpus folders in $PCBENCH_SORTED_OUT"
# D3 is the board list in configs/datasets/d3.json, not every corpus folder (two are excluded).
python - "$PCBENCH_SORTED_OUT" <<'PY'
import json, os, sys
d3 = json.load(open("configs/datasets/d3.json"))
pool = [b for t in ("easy", "medium", "hard") for b in d3[t]["train"]]
have = sum(os.path.isdir(os.path.join(sys.argv[1], b)) for b in pool)
print(f"[quickstart] D3 = {len(pool)} boards (configs/datasets/d3.json; {have} present here): "
      f"pools easy {len(d3['easy']['train'])} / medium {len(d3['medium']['train'])} / "
      f"hard {len(d3['hard']['train'])}; eval sets D3-A {len(d3['easy']['test'])} / "
      f"D3-B {len(d3['medium']['test'])} / D3-C {len(d3['hard']['test'])}")
PY
python -c 'import json, sys
rows = json.load(open(sys.argv[1]))
ok = sum(r["status"] == "ok" for r in rows)
print(f"[quickstart] pcbench: .dsn + .orp for {ok} of {len(rows)} sorted boards"
      + ("" if ok == len(rows) else f"; the other {len(rows) - ok} are the fail rows of step 4 above"))' "$PCBENCH_DSN_ROOT/_results.json"
for v in PCBENCH_PCBS_ROOT PCBENCH_V9_ROOT PCBENCH_NEWDRC_OUT PCBENCH_SORTED_OUT PCBENCH_DSN_ROOT; do echo "[quickstart] $v=${!v}"; done
