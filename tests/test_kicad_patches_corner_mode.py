"""Regression guard for the MITERED_90 consistency patches.

KiCad PNS does not have a single chokepoint that enforces the configured
corner mode — each function in the routing pipeline has to opt in. We
maintain three patches under engine/kicad-patches/kicad/pcbnew/router/ that fill
the gaps the upstream code leaves open. If any of these silently drops
out of the build tree (e.g. someone overwrites the patch with stock
upstream, or removes the cp from build_rl_router.sh), 45° corners start
leaking back into 90°-mode routes.

These tests are cheap and catch that class of regression early — they do
not require a built kicad_rl_router.so, just file-content inspection.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = REPO_ROOT / "engine" / "kicad-patches" / "kicad" / "pcbnew" / "router"
BUILD_SCRIPT = REPO_ROOT / "engine" / "build_rl_router.sh"

# Each entry: (patched filename, list of substrings that MUST appear in the
# patched copy). Markers are content unique to our fix that wouldn't be
# present in stock upstream — if any is missing, the file likely got reset
# to the unpatched version.
PATCH_MARKERS: list[tuple[str, list[str]]] = [
    (
        "pns_line_placer.cpp",
        [
            # reduceTail: cornerMode forwarded to BuildInitialTrace
            "Forward the configured corner mode",
            # both BuildInitialTrace calls in reduceTail take cornerMode now
            "BuildInitialTrace( s.A, aEnd, false, cornerMode )",
            "BuildInitialTrace( new_start, aEnd, false, cornerMode )",
            # mergeHead: OBTUSE added to ForbiddenAngles in 90deg modes
            "ForbiddenAngles |= DIRECTION_45::ANG_OBTUSE",
        ],
    ),
    (
        "pns_shove.cpp",
        [
            # shoveLineToHullSet: hull -> axis-aligned bbox substitution
            "replace octagonal obstacle hulls",
            "HULL_SET hullsAxis",
            "const HULL_SET& aHulls = is90mode ? hullsAxis : aHullsIn",
        ],
    ),
    (
        "pns_optimizer.cpp",
        [
            # CornerCost: ANG_OBTUSE penalised in 90deg modes
            "RL fork fix: 45° miters (ANG_OBTUSE)",
            "if( is90mode && ang == DIRECTION_45::ANG_OBTUSE )",
        ],
    ),
]


@pytest.mark.parametrize("filename,markers", PATCH_MARKERS,
                         ids=[name for name, _ in PATCH_MARKERS])
def test_patch_present_and_has_markers(filename: str, markers: list[str]) -> None:
    path = PATCH_DIR / filename
    assert path.exists(), (
        f"Required patch file missing: {path}\n"
        "The MITERED_90 consistency fix lives in engine/kicad-patches/kicad/pcbnew/router/. "
        "If this file is gone, 90°-mode routing will silently leak 45° corners "
        "(see git log for context)."
    )
    text = path.read_text()
    missing = [m for m in markers if m not in text]
    assert not missing, (
        f"{filename} no longer contains: {missing}\n"
        "Either the patch was reverted, or the file was overwritten with "
        "stock upstream KiCad. Re-apply the patch."
    )


def test_build_script_copies_all_three_patches() -> None:
    """scripts/build_rl_router.sh must cp each patched file into the build tree."""
    assert BUILD_SCRIPT.exists()
    script = BUILD_SCRIPT.read_text()
    for filename, _ in PATCH_MARKERS:
        marker = f"kicad/pcbnew/router/{filename}"
        assert marker in script, (
            f"build_rl_router.sh no longer copies {filename} into the build tree.\n"
            "Without this cp, the patch sits in engine/kicad-patches/ but never reaches "
            "build_rl/kicad_src/ — the build silently reverts to unpatched upstream."
        )
