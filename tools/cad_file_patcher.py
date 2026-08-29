"""CLI utility: load + save .kicad_pcb files through the C++ router build.

Round-trips ``.kicad_pcb`` files through a throwaway native router
(``load_and_save_via_engine`` — deliberately NOT ``KiCadEngine``, whose
load contract refuses exactly the legacy/pro-less inputs this tool
exists to convert). Useful for:

- The ONE-SHOT legacy upgrade the engine load contract demands:
  ``KiCadEngine`` refuses KiCad 5-era boards (rules embedded in the pcb)
  and boards without a ``.kicad_pro`` — run a directory through here once
  and use the emitted pcb+pro pairs instead
- Producing a single "current-format" copy for inspection in the system
  KiCad.app (which may otherwise reject older files from a mismatched
  toolchain)
- Bulk backup-and-upgrade of a mixed-version board collection

Each save emits **both** ``<stem>.kicad_pcb`` and a companion
``<stem>.kicad_pro`` under the output path — keeping the pair together
is what preserves BDS + NetSettings across the format upgrade, since
the modern pcb writer strips those fields on save. ``--force`` / skip
logic treat the pair as one unit.

Source files are never modified. Output files are written to a separate
directory (default ``cache/patched/``); when processing a directory the
relative layout under the input root is preserved in the output root.

Usage:

    # Single file → custom output path (or output directory)
    python cad_file_patcher.py board.kicad_pcb -o out/board.kicad_pcb
    python cad_file_patcher.py board.kicad_pcb -o out/

    # Directory → mirrored output tree
    python cad_file_patcher.py pcb_dataset/ -o cache/patched/ -r

    # Multiple inputs mixed (files and directories)
    python cad_file_patcher.py a.kicad_pcb dir/ -o out/

Exit codes: 0 = all OK, 1 = one or more files failed (still continues
through the batch).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# sys.path bootstrap — adds the repo root and the built router dir so the
# script runs with just `conda activate cadagent && python
# tools/cad_file_patcher.py ...` (no manual PYTHONPATH needed). This file lives
# in tools/, so the repo root is one level up.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KICAD_RL_DIR = _REPO_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
for p in (_REPO_ROOT, _KICAD_RL_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def _iter_pcb_files(inputs: list[Path], recursive: bool,
                    name_contains: str | None = None) -> Iterable[tuple[Path, Path]]:
    """Yield ``(source_file, source_root)`` pairs for every .kicad_pcb to patch.

    ``source_root`` is the top-level input directory (or the file's own
    parent, if the input was a file) — used to compute the relative path
    for mirrored output layout.

    ``name_contains`` is a substring filter applied only during recursive
    directory traversal (e.g. ``"processed.kicad_pcb"`` to pick only
    preprocessed boards out of a mixed tree). Explicit file inputs and
    non-recursive directory inputs are left unfiltered.
    """
    for item in inputs:
        item = item.resolve()
        if item.is_file():
            if item.suffix == ".kicad_pcb":
                yield item, item.parent
            else:
                print(f"  [skip] not a .kicad_pcb: {item}", file=sys.stderr)
        elif item.is_dir():
            glob = item.rglob("*.kicad_pcb") if recursive else item.glob("*.kicad_pcb")
            for f in sorted(glob):
                if recursive and name_contains and name_contains not in f.name:
                    continue
                yield f, item
        else:
            print(f"  [skip] missing: {item}", file=sys.stderr)


def _resolve_output_path(src: Path, src_root: Path,
                         output: Path, output_is_dir: bool) -> Path:
    """Decide where the patched copy of ``src`` should land."""
    if not output_is_dir:
        # Output is an explicit file path — only valid for single-file
        # mode; the caller already enforced that.
        return output
    # Directory mode: mirror the relative layout under src_root.
    try:
        rel = src.relative_to(src_root)
    except ValueError:
        rel = Path(src.name)
    return output / rel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Round-trip .kicad_pcb files through the local KiCad "
                    "build's loader/saver to normalize them to the router's "
                    "file-format version. Source files are never modified.",
    )
    p.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more .kicad_pcb files and/or directories.",
    )
    p.add_argument(
        "-o", "--output", type=Path,
        default=_REPO_ROOT / "cache" / "patched",
        help="Output file or directory. If given as a directory, the "
             "relative layout under each input directory is mirrored. "
             "Default: cache/patched/",
    )
    p.add_argument(
        "-r", "--recursive", action="store_true",
        help="When an input is a directory, recurse into subdirectories.",
    )
    p.add_argument(
        "--name-contains", type=str, default=None, metavar="STR",
        help="Only patch files whose filename contains STR (e.g. "
             "``processed.kicad_pcb``). Applied only during recursive "
             "directory traversal; explicit file inputs are always patched.",
    )
    p.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print a line per source file (not just failures).",
    )
    p.add_argument(
        "--unrouted", action="store_true",
        help="Clear all existing tracks + vias before saving (matches "
             "PCBWorld.reset). Output filename gets a ``.unrouted`` infix: "
             "``<stem>.unrouted.kicad_pcb``.",
    )
    p.add_argument(
        "--simplify-outline", action="store_true",
        help="Rewrite tessellated micro-segment outline chains (Edge.Cuts/"
             "Margin) into native arcs/merged lines before saving — the "
             "converter form of the engine's load-time simplify pass "
             "(pcb_world/engine/outline_simplify.py). Deterministic per input.",
    )
    return p.parse_args()


def _add_unrouted_infix(path: Path) -> Path:
    """``foo.kicad_pcb`` → ``foo.unrouted.kicad_pcb``.

    Leaves an already-infixed path alone so re-runs don't stack suffixes.
    """
    if path.suffixes[-2:] == [".unrouted", ".kicad_pcb"]:
        return path
    return path.with_name(f"{path.stem}.unrouted.kicad_pcb")


def main() -> int:
    args = parse_args()

    from pcb_world.engine.utils import load_and_save_via_engine
    from tqdm import tqdm

    # Resolve output semantics. Single-file mode with an explicit file output
    # is a distinct case from directory mode.
    inputs_resolved = [Path(x).resolve() for x in args.inputs]
    single_file_with_file_output = (
        len(inputs_resolved) == 1
        and inputs_resolved[0].is_file()
        and inputs_resolved[0].suffix == ".kicad_pcb"
        and args.output.suffix == ".kicad_pcb"
    )
    output_is_dir = not single_file_with_file_output

    if output_is_dir:
        args.output.mkdir(parents=True, exist_ok=True)

    # Materialize the work list so tqdm has a total for the progress bar
    # (``_iter_pcb_files`` is a generator — count isn't known upfront).
    pairs = list(_iter_pcb_files(
        inputs_resolved, args.recursive, name_contains=args.name_contains,
    ))

    n_ok = 0
    n_fail = 0
    n_skip = 0

    bar = tqdm(
        pairs, total=len(pairs),
        desc="unrouting" if args.unrouted else "patching",
        unit="file",
    )
    for src, src_root in bar:
        # Short display for the progress bar's postfix: just the filename.
        bar.set_postfix_str(src.name, refresh=False)

        dst = _resolve_output_path(src, src_root, args.output, output_is_dir)
        if args.unrouted:
            dst = _add_unrouted_infix(dst)

        # Every patched board is a pcb + pro pair — skip only when BOTH
        # companions are already on disk, otherwise a half-patched run
        # from before would keep serving a stale / missing pro.
        dst_pro = dst.with_suffix(".kicad_pro")
        if dst.exists() and dst_pro.exists() and not args.force:
            if args.verbose:
                tqdm.write(f"  [skip] exists: {dst}  (use --force to overwrite)")
            n_skip += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            load_and_save_via_engine(
                src, dst, unroute=args.unrouted,
                simplify_outline=args.simplify_outline, verbose=args.verbose,
            )
        except Exception as exc:  # noqa: BLE001 — C++ bindings raise RuntimeError
            # Leave no half-written artifact behind — pcb AND its pro
            # companion both need cleaning up on failure so the next run
            # doesn't see a mismatched pair.
            for stray in (dst, dst_pro):
                try:
                    stray.unlink()
                except FileNotFoundError:
                    pass
            tqdm.write(
                f"  [FAIL] {src}  ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            n_fail += 1
            continue

        if args.verbose:
            tqdm.write(f"  [ok] {src}  →  {dst} (+ .kicad_pro)")
        n_ok += 1
    bar.close()

    print()
    print(f"Patched: {n_ok}  |  Failed: {n_fail}  |  Skipped (existing): {n_skip}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
