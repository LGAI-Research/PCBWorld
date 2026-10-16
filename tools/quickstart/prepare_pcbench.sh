#!/usr/bin/env bash
# Build the KiCad 10.0.7 PCBench D3 corpus in isolated v10 output roots.
#
#   bash tools/quickstart/prepare_pcbench.sh [--limit N] [--workers N]
#
# The source clone is pinned to dec3be75cbdef74787625f9043c7391cd473bb64.
# Existing clones are validated, never reset. All generated boards and reports
# live under PCBWORLD_DATA_ROOT; the historical v9 trees are never touched.
set -euo pipefail
PCBWORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PCBWORLD_ROOT"
PCBW_WRAPPER="$(cd "$PCBWORLD_ROOT/../.." && pwd)/scripts/pcbw"
if [ -x "$PCBW_WRAPPER" ]; then
  PYTHON_RUNNER=("$PCBW_WRAPPER" python)
else
  PYTHON_RUNNER=(python)
  echo "[quickstart] scripts/pcbw unavailable; using active Python environment" >&2
fi

LIMIT=0; WORKERS=16
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
: "${PCBWORLD_DATA_ROOT:?export PCBWORLD_DATA_ROOT=<your dataset root> first (e.g. \$PWD/var/datasets)}"

run() { echo "+ $*" >&2; "$@"; }
PIPELINE_FAILED=0
LAST_STAGE_RC=0
STAGE_COMMANDS=()
run_stage() {
  local stage="$1" rc
  shift
  echo "+ $*" >&2
  if "$@"; then
    rc=0
  else
    rc=$?
    echo "[quickstart] $stage command exited $rc; checking its report before continuing" >&2
  fi
  LAST_STAGE_RC="$rc"
  STAGE_COMMANDS+=("$stage=$rc")
}

# Missing, malformed, duplicate, or incomplete ledgers are fatal. Complete
# per-board failures are recorded, but do not prevent later stages from running.
verify_ledger() {
  local stage="$1" report="$2" input="$3" upstream="${4:-}" limit="${5:-0}" rc="$LAST_STAGE_RC"
  local failed
  failed="$("${PYTHON_RUNNER[@]}" - "$stage" "$report" "$input" "$upstream" "$limit" "$rc" <<'PY'
import json
import sys
from pathlib import Path

stage, report_name, input_name, upstream_name, limit_text, command_rc_text = sys.argv[1:]
def fatal(message):
    raise SystemExit(f"[quickstart] {stage} ledger invalid: {message}")

def read_rows(path, label):
    path = Path(path)
    if not path.is_file():
        fatal(f"{label} report missing: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fatal(f"cannot read {label} report {path}: {exc}")
    if not isinstance(rows, list):
        fatal(f"{label} report is not a JSON array: {path}")
    return rows

def row_names(rows, key, label):
    names = []
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or not isinstance(row.get(key), str)
                or not row[key] or not isinstance(row.get("status"), str)):
            fatal(f"malformed {label} row {index}")
        names.append(row[key])
    if len(names) != len(set(names)):
        fatal(f"duplicate board IDs in {label} report")
    return names

if stage == "conversion":
    input_root = Path(input_name)
    if not input_root.is_dir():
        fatal(f"raw PCBench root missing: {input_root}")
    expected = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    limit = int(limit_text)
    if limit > 0:
        expected = expected[:limit]
    key = "name"
elif stage == "drc":
    conversion_rows = read_rows(upstream_name, "conversion")
    expected = row_names(conversion_rows, "name", "conversion")
    key = "name"
elif stage in ("guide", "sort"):
    input_root = Path(input_name)
    if not input_root.is_dir():
        fatal(f"stage input directory missing: {input_root}")
    expected = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    key = "folder" if stage == "guide" else "sample"
elif stage == "dsn":
    input_root = Path(input_name)
    if not input_root.is_dir():
        fatal(f"sorted input directory missing: {input_root}")
    expected = sorted(path.name for path in input_root.iterdir()
                      if path.is_dir() and
                      (path / "processed_v10_guide_v3.kicad_pcb").is_file())
    key = "folder"
else:
    fatal(f"unknown stage {stage!r}")

if stage in ("conversion", "drc") and not expected:
    fatal("no inputs to account for")
rows = read_rows(report_name, stage)
names = row_names(rows, key, stage)
missing = sorted(set(expected) - set(names))
unexpected = sorted(set(names) - set(expected))
if missing or unexpected:
    fatal(f"input coverage mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")

allowed = {
    "conversion": {"ok"},
    "drc": {"ok", "drc_rejected"},
    "guide": {"pass", "fail"},
    "sort": {"ok", "guide_rejected"},
    "dsn": {"ok"},
}[stage]
failed = any(row["status"] not in allowed for row in rows)
if stage == "dsn":
    output_root = Path(upstream_name)
    for row in rows:
        if row["status"] == "ok":
            folder = output_root / row["folder"]
            outputs = (folder / "processed_v10_guide_v3_unrouted.dsn",
                       folder / "processed_v10_guide_v3.orp")
            if any(not path.is_file() or path.stat().st_size == 0 for path in outputs):
                failed = True

command_rc = int(command_rc_text)
no_work = not expected and not rows
all_sort_rejections = (stage == "sort" and rows and
                       all(row["status"] == "guide_rejected" for row in rows))
if command_rc != 0 and not no_work and not all_sort_rejections:
    failed = True
print("1" if failed else "0")
PY
)"
  if [ "$failed" = 1 ]; then
    PIPELINE_FAILED=1
  fi
}

PCBENCH_COMMIT=dec3be75cbdef74787625f9043c7391cd473bb64
PCBENCH_CLONE_ROOT="${PCBENCH_CLONE_ROOT:-$(dirname "${PCBENCH_PCBS_ROOT:-$PCBWORLD_DATA_ROOT/PCBench/PCBs}")}"
mkdir -p "$(dirname "$PCBENCH_CLONE_ROOT")"
PCBENCH_CLONE_ROOT="$(cd "$(dirname "$PCBENCH_CLONE_ROOT")" && pwd)/$(basename "$PCBENCH_CLONE_ROOT")"
export PCBENCH_PCBS_ROOT="${PCBENCH_PCBS_ROOT:-$PCBENCH_CLONE_ROOT/PCBs}"
export PCBENCH_V10_ROOT="${PCBENCH_V10_ROOT:-$PCBWORLD_DATA_ROOT/pcbench_work/v10}"
export PCBENCH_NEWDRC_OUT="${PCBENCH_NEWDRC_OUT:-$PCBWORLD_DATA_ROOT/pcbench_work/newdrc_v10}"
export PCBENCH_SORTED_OUT="${PCBENCH_SORTED_OUT:-$PCBWORLD_DATA_ROOT/pcbench/exacad_sorted_v10}"
export PCBENCH_DSN_OUT="${PCBENCH_DSN_OUT:-$PCBWORLD_DATA_ROOT/pcbench/exacad_sorted_dsn_v10}"
export PCBENCH_PCB_ROOT="$PCBENCH_SORTED_OUT"
export PCBENCH_DSN_ROOT="$PCBENCH_DSN_OUT"

# Refuse writes outside the data root, overlapping the source clone or v9 data,
# and any overlapping stage roots.
"${PYTHON_RUNNER[@]}" - "$PCBWORLD_DATA_ROOT" "$PCBENCH_PCBS_ROOT" "$PCBENCH_CLONE_ROOT" \
  "$PCBENCH_V10_ROOT" "$PCBENCH_NEWDRC_OUT" "$PCBENCH_SORTED_OUT" "$PCBENCH_DSN_OUT" <<'PY'
from pathlib import Path
import sys
root, raw, clone, *outputs = map(lambda value: Path(value).resolve(), sys.argv[1:])
preserved = [
    raw, clone,
    root / "pcbench_work/v9",
    root / "pcbench_work/newdrc",
    root / "pcbench/exacad_sorted",
    root / "pcbench/exacad_sorted_dsn",
]
def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents
for output in outputs:
    if output == root or root not in output.parents:
        raise SystemExit(f"refusing output outside PCBWORLD_DATA_ROOT: {output}")
    for path in preserved:
        if overlaps(output, path):
            raise SystemExit(f"refusing output {output}: overlaps preserved path {path}")
for i, output in enumerate(outputs):
    for other in outputs[i + 1:]:
        if overlaps(output, other):
            raise SystemExit(f"v10 output roots must be disjoint: {output} and {other}")
PY

# Create a fresh exact-commit clone only when the target does not exist. Never
# fetch, reset, check out, or discard changes in an existing clone.
if [ ! -e "$PCBENCH_CLONE_ROOT" ]; then
  run git clone --no-checkout https://github.com/PCBench/PCBench.git "$PCBENCH_CLONE_ROOT"
  run git -C "$PCBENCH_CLONE_ROOT" fetch --depth 1 origin "$PCBENCH_COMMIT"
  run git -C "$PCBENCH_CLONE_ROOT" checkout --detach "$PCBENCH_COMMIT"
elif ! git -C "$PCBENCH_CLONE_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "PCBench clone path exists but is not a Git checkout: $PCBENCH_CLONE_ROOT" >&2
  exit 2
fi
CLONE_HEAD="$(git -C "$PCBENCH_CLONE_ROOT" rev-parse HEAD)"
if [ "$CLONE_HEAD" != "$PCBENCH_COMMIT" ]; then
  echo "PCBench clone must be at $PCBENCH_COMMIT; found $CLONE_HEAD (preserved; not reset)" >&2
  exit 2
fi
if [ -n "$(git -C "$PCBENCH_CLONE_ROOT" status --porcelain)" ]; then
  echo "PCBench clone has local changes; preserved without modification: $PCBENCH_CLONE_ROOT" >&2
  exit 2
fi
PCBENCH_TOP="$(git -C "$PCBENCH_PCBS_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
if [ "$PCBENCH_TOP" != "$PCBENCH_CLONE_ROOT" ] || [ ! -d "$PCBENCH_PCBS_ROOT" ]; then
  echo "PCBENCH_PCBS_ROOT must be the PCBs/ directory in $PCBENCH_CLONE_ROOT; found $PCBENCH_PCBS_ROOT" >&2
  exit 2
fi

BUILD_DIR="${PCBWORLD_KICAD_RL_BUILD_DIR:-build_rl}"
if [ -n "${PCBENCH_KICAD_CLI:-}" ]; then
  ENGINE_CLI="$PCBENCH_KICAD_CLI"
else
  case "$(uname -s)" in
    Darwin) ENGINE_CLI="$BUILD_DIR/kicad/KiCad.app/Contents/MacOS/kicad-cli" ;;
    Linux) ENGINE_CLI="$BUILD_DIR/kicad/kicad-cli" ;;
    *) echo "unsupported OS for KiCad 10.0.7 prep: $(uname -s)" >&2; exit 2 ;;
  esac
  if [ ! -x "$ENGINE_CLI" ]; then
    echo "Engine kicad-cli not in $BUILD_DIR: building KiCad 10.0.7 CLI + pcbnew (one-off)" >&2
    run env BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh
  fi
fi
if [ ! -x "$ENGINE_CLI" ]; then
  echo "Engine kicad-cli is not executable: $ENGINE_CLI" >&2
  exit 2
fi
export KICAD_CLI="$ENGINE_CLI"
case "$ENGINE_CLI" in
  */Contents/MacOS/kicad-cli)
    ENGINE_BUILD_ROOT="$(cd "$(dirname "$ENGINE_CLI")/../../../.." && pwd)"
    ;;
  *) ENGINE_BUILD_ROOT="$(cd "$(dirname "$ENGINE_CLI")/.." && pwd)" ;;
esac
export PYTHONPATH="$ENGINE_BUILD_ROOT/pcbnew:$ENGINE_BUILD_ROOT/pcbnew/python/rl${PYTHONPATH:+:$PYTHONPATH}"
export KICAD_RUN_FROM_BUILD_DIR=1

PP=tools/datagen/pcbench_prep
GUIDE_REPORT="$PCBENCH_NEWDRC_OUT/_guide_results.json"
SORT_REPORT="$PCBENCH_NEWDRC_OUT/_sort_results.json"
DSN_REPORT="$PCBENCH_DSN_OUT/_run-report.json"
LIMIT_ARGS=(); [ "$LIMIT" -gt 0 ] && LIMIT_ARGS=(--limit "$LIMIT")

# Strict toolchain preflight stays fatal before any board-processing stage.
run "${PYTHON_RUNNER[@]}" "$PP/kicad_tools.py" --assert-all
PCBNEW_PYTHON="$("${PYTHON_RUNNER[@]}" - "$PP" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import kicad_tools
print(kicad_tools.pcbnew_python())
PY
)"
export PCBNEW_PYTHON
if [ -x "$PCBW_WRAPPER" ]; then
  DSN_RUNNER=("$PCBW_WRAPPER" "$PCBNEW_PYTHON")
else
  DSN_RUNNER=("$PCBNEW_PYTHON")
fi

rm -f "$PCBENCH_V10_ROOT/_results.json"
run_stage conversion "${PYTHON_RUNNER[@]}" "$PP/convert_v10.py" \
  "${LIMIT_ARGS[@]}" --workers "$WORKERS"
verify_ledger conversion "$PCBENCH_V10_ROOT/_results.json" "$PCBENCH_PCBS_ROOT" "" "$LIMIT"

rm -f "$PCBENCH_NEWDRC_OUT/_results.json"
run_stage DRC "${PYTHON_RUNNER[@]}" "$PP/drc_fix_v10.py" --workers "$WORKERS"
verify_ledger drc "$PCBENCH_NEWDRC_OUT/_results.json" "" "$PCBENCH_V10_ROOT/_results.json"

rm -f "$GUIDE_REPORT"
run_stage guide "${PYTHON_RUNNER[@]}" "$PP/make_guide.py" --base-dir "$PCBENCH_NEWDRC_OUT" \
  --stem processed_v10 --suffix _guide_v3 --workers "$WORKERS" --log "$GUIDE_REPORT"
verify_ledger guide "$GUIDE_REPORT" "$PCBENCH_NEWDRC_OUT"

rm -f "$SORT_REPORT"
run_stage sort "${PYTHON_RUNNER[@]}" "$PP/sort_prefix.py" \
  --guide-report "$GUIDE_REPORT" --report "$SORT_REPORT"
verify_ledger sort "$SORT_REPORT" "$PCBENCH_NEWDRC_OUT"
# sort_prefix normally prunes obsolete rank folders; its no-success path leaves
# prior v10 boards and CSV metadata untouched, so discard those stale artifacts.
"${PYTHON_RUNNER[@]}" - "$SORT_REPORT" "$PCBENCH_SORTED_OUT" "$PCBENCH_DSN_OUT" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not any(row.get("status") == "ok" for row in rows):
    sorted_root = Path(sys.argv[2])
    if sorted_root.is_dir():
        for path in sorted_root.iterdir():
            if path.is_dir() and re.match(r"^\d{4}_", path.name):
                shutil.rmtree(path)
    csv_path = sorted_root / "pcb_characteristics_exacad_sorted.csv"
    try:
        csv_path.unlink()
    except FileNotFoundError:
        pass
    dsn_root = Path(sys.argv[3])
    if dsn_root.is_dir():
        for path in dsn_root.iterdir():
            if path.is_dir() and re.match(r"^\d{4}_", path.name):
                shutil.rmtree(path)
PY

rm -f "$DSN_REPORT"
run_stage DSN/ORP env DRVZERO_DATA_DIR="${DRVZERO_DATA_DIR:-$PCBWORLD_DATA_ROOT}" \
  "${DSN_RUNNER[@]}" engine/pcbnew_prep/make_dsn_orp_v3.py \
  --src "$PCBENCH_SORTED_OUT" --dst "$PCBENCH_DSN_OUT" --report "$DSN_REPORT"
verify_ledger dsn "$DSN_REPORT" "$PCBENCH_SORTED_OUT" "$PCBENCH_DSN_OUT"

"${PYTHON_RUNNER[@]}" - "$PCBENCH_V10_ROOT/_results.json" "$PCBENCH_NEWDRC_OUT/_results.json" \
  "$GUIDE_REPORT" "$SORT_REPORT" "$DSN_REPORT" <<'PY'
import collections
import json
import sys
for label, path in zip(("conversion", "DRC", "guide", "sort", "DSN/ORP"), sys.argv[1:]):
    with open(path, encoding="utf-8") as stream:
        rows = json.load(stream)
    counts = collections.Counter(row.get("status", "unknown") for row in rows)
    print(f"[quickstart] {label}: {len(rows)} outcomes {dict(sorted(counts.items()))}")
PY
echo "[quickstart] stage command exits: ${STAGE_COMMANDS[*]}"
n=$(find "$PCBENCH_SORTED_OUT" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9]_*' | wc -l)
echo "[quickstart] source commit: $CLONE_HEAD"
echo "[quickstart] pcbench: $n v10 boards in $PCBENCH_SORTED_OUT"
for v in PCBENCH_PCBS_ROOT PCBENCH_V10_ROOT PCBENCH_NEWDRC_OUT PCBENCH_SORTED_OUT PCBENCH_DSN_OUT; do
  echo "[quickstart] $v=${!v}"
done
if [ "$PIPELINE_FAILED" -ne 0 ]; then
  echo "[quickstart] completed with board or stage errors; see the reports above" >&2
  exit 1
fi
