"""The source-scanning helpers the AST guard tests share (``tests/helpers/tracked_files.py``)."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.helpers.tracked_files import calls_named, tracked_py


@pytest.mark.skipif(shutil.which("git") is None, reason="needs the git executable")
def test_listing_covers_new_files_skips_deleted_ones_and_skips_without_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    # Isolated from the user's global config (signing, hooks).
    git = ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
           "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    (repo / "kept.py").write_text("x = 1\n")
    (repo / "moved.py").write_text("y = 2\n")
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "c"], cwd=repo, check=True)
    (repo / "moved.py").unlink()
    (repo / "new.py").write_text("z = 3\n")
    assert sorted(tracked_py(repo)) == [repo / "kept.py", repo / "new.py"]

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(pytest.skip.Exception):
        tracked_py(plain)


def test_calls_named_matches_plain_and_attribute_callees_only(tmp_path):
    src = tmp_path / "calls.py"
    src.write_text("f(1)\nmod.f(2)\na.b.f(3)\ng(4)\nh = f\nh(5)\n")
    assert [c.lineno for c in calls_named(src, {"f"})] == [1, 2, 3]
