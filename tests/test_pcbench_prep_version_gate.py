"""The pinned KiCad gate rejects prereleases and accepts release build suffixes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datagen" / "pcbench_prep"))
import kicad_tools  # noqa: E402


@pytest.mark.parametrize(("version", "expected"), [
    ("10.0.7", True),
    ("10.0.7-ubuntu24.04.1", True),
    ("10.0.7+build", True),
    ("10.0.7rc1", False),
    ("10.0.6", False),
    ("11.0.0", False),
])
def test_only_kicad_1007_release_is_pinned(version, expected):
    assert kicad_tools._version_is_pinned(version) is expected
