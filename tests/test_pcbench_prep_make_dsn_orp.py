"""Step 4 of the PCBench -> D3 chain (tools/datagen/pcbench_prep/make_dsn_orp.py).

The step hands the sorted corpus to the engine's converter in the pcbnew interpreter
kicad_tools resolves. Pinned here with stand-in tools (no KiCad build): the converter
reads the sorted tree as PCBENCH_PCB_ROOT, writes PCBENCH_DSN_ROOT and its report
``_results.json`` there, extra arguments reach it, and the step's exit status tells a
finished run with fail rows (3) from one that did not finish (no report), so
prepare_pcbench.sh goes on after the first and stops on the second.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STEP = REPO / "tools" / "datagen" / "pcbench_prep" / "make_dsn_orp.py"
CONVERTER = REPO / "engine" / "pcbnew_prep" / "make_dsn_orp_v3.py"

# The stand-in converter: records argv + roots, then per $CONVERTER_MODE writes the report given after
# --report and exits 0 (ok) or 1 (finished with fail rows), or exits 1 without a report (crashed).
_CONVERTER = """
import json, os, sys
argv = sys.argv[1:]
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps({"converter": os.environ["CONVERTER_PATH"], "argv": argv, "src": os.environ.get("PCBENCH_PCB_ROOT"),
                        "dst": os.environ.get("PCBENCH_DSN_ROOT")}) + "\\n")
mode = os.environ["CONVERTER_MODE"]
if mode != "crashed":
    report = argv[argv.index("--report") + 1]
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w") as f:
        json.dump([{"folder": "0001_a", "status": "ok"}] + ([{"folder": "0002_b", "status": "fail"}]
                  if mode == "fail_rows" else []), f)
sys.exit(0 if mode == "ok" else 1)
"""


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture
def tools(tmp_path):
    """A kicad-cli and a pcbnew interpreter that answer the version probes; any other call to the
    interpreter runs the stand-in converter."""
    (tmp_path / "converter.py").write_text(_CONVERTER)
    cli = _script(tmp_path / "kicad-cli", "echo 9.0.8\n")
    py = _script(tmp_path / "pcbnew-python",
                 f'[ "$1" = -c ] && {{ echo 9.0.8; exit 0; }}\nCONVERTER_PATH="$1"; export CONVERTER_PATH; shift\n'
                 f'exec {sys.executable} {tmp_path / "converter.py"} "$@"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith("PCBENCH_")}
    env.update(KICAD_CLI=str(cli), PCBNEW_PYTHON=str(py), CALLS=str(tmp_path / "calls.jsonl"),
               PCBENCH_SORTED_OUT=str(tmp_path / "sorted"), PCBENCH_DSN_ROOT=str(tmp_path / "dsn"))
    return env


def _run(env, mode, *args):
    return subprocess.run([sys.executable, str(STEP), *args], cwd=REPO, env=dict(env, CONVERTER_MODE=mode),
                          capture_output=True, text=True, timeout=60)


def _calls(env):
    path = Path(env["CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_runs_the_converter_on_the_sorted_tree_with_its_report(tools, tmp_path):
    proc = _run(tools, "ok", "--limit", "2")
    assert proc.returncode == 0, proc.stderr
    report = tmp_path / "dsn" / "_results.json"
    assert _calls(tools) == [{"converter": str(CONVERTER), "argv": ["--limit", "2", "--report", str(report)],
                              "src": str(tmp_path / "sorted"), "dst": str(tmp_path / "dsn")}]
    assert report.is_file()


def test_a_finished_run_with_fail_rows_exits_3(tools):
    proc = _run(tools, "fail_rows")
    assert proc.returncode == 3, proc.stderr


def test_a_run_that_did_not_finish_is_not_taken_for_fail_rows(tools, tmp_path):
    stale = tmp_path / "dsn" / "_results.json"   # left by an earlier, finished run
    stale.parent.mkdir()
    stale.write_text("[]")
    proc = _run(tools, "crashed")
    assert proc.returncode not in (0, 3)
    assert not stale.exists()


@pytest.mark.parametrize("missing", ["PCBENCH_SORTED_OUT", "PCBENCH_DSN_ROOT"])
def test_a_missing_root_is_named_before_the_converter_runs(tools, missing):
    del tools[missing]
    proc = _run(tools, "ok")
    assert proc.returncode not in (0, 3) and missing in proc.stderr
    assert _calls(tools) == []
