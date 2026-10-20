"""The repo's own Python files, for source-scanning (AST) guard tests."""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def tracked_py(root: Path = REPO) -> list[Path]:
    if not (root / ".git").exists():
        pytest.skip("needs a git checkout to list tracked files (e.g. not a source zip)")
    # Tracked plus new not-yet-added files (a moved file is scanned before `git add`).
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
                         cwd=root, capture_output=True, text=True, check=True).stdout.split()
    # Third-party code and superseded trees are not ours to change; a tracked file
    # deleted in the working tree (mid-move) has nothing to scan.
    return [root / f for f in out
            if not f.startswith("external/") and "deprecated" not in Path(f).parts
            and (root / f).exists()]


def calls_named(path: Path, names) -> Iterator[ast.Call]:
    """Calls in ``path`` whose callee is spelled ``name(...)`` or ``<expr>.name(...)``
    for a ``name`` in ``names`` — by the literal name only (an alias is not seen)."""
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Call):
            func = node.func
            if (func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)) in names:
                yield node
