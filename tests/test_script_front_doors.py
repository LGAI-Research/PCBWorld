"""The ``scripts/`` front doors must not leave ``scripts/`` shadowing stdlib modules.

``python scripts/<x>.py`` puts ``scripts/`` at ``sys.path[0]``, so every module
name that also exists as a file in there shadows the real one for the rest of the
process. ``scripts/profile.py`` is exactly such a name: torch's dynamo reaches
``import profile`` through ``cProfile``, gets the repo's profiler CLI instead of
the stdlib module, and dies with ``module 'profile' has no attribute 'run'``.

``scripts/train.py`` already removes the directory from ``sys.path`` for this
reason; ``scripts/eval.py`` only *outranked* it with the repo root, which fixes
the ``eval`` package shadowing but not the stdlib one — plain
``python scripts/eval.py --ckpt ...`` could not load a policy at all (2026-09-02).

Each case reproduces the condition (``scripts/`` first on the path), runs only the
front door's module-level preamble, and asserts the observable consequence.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FRONT_DOORS = sorted(p.name for p in (REPO / "scripts").glob("*.py"))

_PROBE = """
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
src = open(os.path.join("scripts", {entry!r})).read()
exec(src.split("\\ndef ")[0], {{"__name__": "__front_door__", "__file__":
     os.path.join(os.getcwd(), "scripts", {entry!r})}})
import profile
assert hasattr(profile, "run"), (
    "scripts/profile.py shadows the stdlib profile module after "
    + {entry!r} + " set up sys.path")
print("OK")
"""


@pytest.mark.parametrize("entry", FRONT_DOORS)
def test_front_door_does_not_shadow_stdlib_profile(entry):
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(entry=entry)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"{entry}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
