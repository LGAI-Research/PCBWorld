#!/usr/bin/env python3
"""Verify the license boundary between the two programs in this tree.

The PCBWorld environment (this repository) and the routing engine (its own
GPLv3 repository, pinned here as the ``engine/`` submodule) are separate
programs that talk over a unix socket. This script proves the boundary holds,
statically and at runtime, and prints every command it runs so the result can
be reproduced by hand.

    python tools/check_separation.py            # static checks only
    python tools/check_separation.py --runtime  # + live /proc/<pid>/maps proof
    python tools/check_separation.py --runtime --evidence-dir DIR

Checks:

  1. no-gpl-in-tree     No file tracked by this repository carries a GPL
                        licence text or SPDX tag. The engine is the only GPL
                        code and it is not distributed from here.
  2. no-combined-build  No packaging metadata declares one distributable that
                        contains both programs.
  3. no-env-import      Nothing in the engine checkout imports the environment
                        packages (``pcb_world`` / ``methods`` / ``eval``).
  4. wire-copies-match  The shared protocol module is byte-identical on both
                        sides, so neither program imports the other's copy.
  5. maps (--runtime)   With a live environment process routing a board, the
                        engine's shared library appears in the child engine
                        server's /proc/<pid>/maps and NOT in the environment
                        process's.

Checks 3-5 need the engine checked out (``git submodule update --init``);
they report SKIP, not failure, when it is absent.

The environment side of the boundary — no GPL import outside
``pcb_world/engine/`` and the native test dirs — is checked on every run of
``tools/docs/check_docs.py`` (``import-hygiene``), which is where the rule
that governs this repository's own files lives.

Exit code 0 = every check that ran passed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The engine is a separate repository, pinned as a submodule. Nothing under it
# is tracked here, so the engine-side checks read the filesystem directly.
ENGINE = Path(os.environ.get("PCBWORLD_ENGINE_HOME") or (REPO / "engine"))

ENV_PACKAGES = ("pcb_world", "methods", "eval")

# engine/kicad-python is the pinned upstream KiCad checkout — upstream code,
# not ours, and its own `tools` package is unrelated to this tree's tools/.
ENGINE_SKIP = ("kicad-python/",)

_IMPORT_RE_ENV = re.compile(
    r"^\s*(?:import|from)\s+(?:%s)\b" % "|".join(ENV_PACKAGES))

results: list[tuple[str, bool, str]] = []
skipped: list[tuple[str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def skip(name: str, detail: str) -> None:
    """A check that could not run — reported loudly, but not a failure."""
    skipped.append((name, detail))
    print(f"[SKIP] {name}: {detail}")


def engine_files(*suffixes: str) -> list[Path]:
    """Our files inside the engine checkout, upstream KiCad excluded."""
    out = []
    for p in ENGINE.rglob("*"):
        if not p.is_file() or p.suffix not in suffixes:
            continue
        rel = p.relative_to(ENGINE).as_posix()
        if rel.startswith(ENGINE_SKIP) or "__pycache__" in rel:
            continue
        out.append(p)
    return sorted(out)


# Directories that are not part of either program: pinned upstream checkouts,
# third-party submodules, build output and generated data.
_SKIP_DIRS = {
    ".git", "__pycache__", "build_rl", "var", "sandbox", "cache",
}
_SKIP_PREFIXES = (
    "engine/",                    # the engine submodule — checked separately
    "external/RAGEN/",            # third-party framework checkouts
    "external/verl-agent/",
    "external/OrthoRoute/",
)


def source_files(*patterns: str) -> list[str]:
    """Repository-relative paths matching the patterns, git or not.

    Uses ``git ls-files`` in a checkout so ignored files never leak in; falls
    back to a filesystem walk so this script also runs against an unpacked
    source archive, where the same checks must reproduce.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", *patterns],
            capture_output=True, text=True, check=True).stdout.splitlines()
        if out:
            return [f for f in out if not f.startswith(_SKIP_PREFIXES)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    files = []
    for pattern in patterns:
        for p in REPO.rglob(pattern):
            if any(part in _SKIP_DIRS for part in p.relative_to(REPO).parts):
                continue
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith(_SKIP_PREFIXES):
                continue
            files.append(rel)
    return sorted(set(files))


# Files whose bytes are data, not prose — never carry a licence header.
_BINARYISH = {".kicad_pcb", ".kicad_dru", ".kicad_pro", ".png", ".jpg", ".pdf",
              ".pt", ".npz", ".npy", ".zip", ".gz", ".pyc", ".ico", ".svg",
              ".mp4", ".gif", ".ttf", ".woff", ".woff2"}
_GPL_MARKERS = ("GNU GENERAL PUBLIC LICENSE", "GNU General Public License",
                "GNU Affero General Public License",
                "SPDX-License-Identifier: GPL", "SPDX-License-Identifier: AGPL",
                "SPDX-License-Identifier: LGPL")

# The one tracked file that MUST carry copyleft licence text: the third-party
# attribution notice. The release ships LGPL-licensed Python dependencies
# (frozendict LGPL-3.0, pycountry LGPL-2.1, soxr LGPL-2.1+), and reproducing
# their licence texts is precisely what those licences require of a
# distributor. It is an attribution document about other people's code, not a
# licence grant over anything in this repository, and it makes nothing here
# copyleft. Kept as one explicit filename, never a directory or glob, so a new
# GPL-licensed file cannot slip past this check by landing next to it.
_GPL_TEXT_ALLOWED = {"Notice.md"}


def check_no_gpl_in_tree() -> None:
    """No file distributed from this repository is GPL-licensed.

    The engine is the only GPL code in the project and it ships from its own
    repository. This check is what makes that claim auditable from here.
    """
    print("\n$ grep -rl 'GNU General Public License\\|SPDX-License-Identifier: "
          "[AL]*GPL' .")
    # This file carries the search patterns themselves, so it always matches.
    self_rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    skip_names = _GPL_TEXT_ALLOWED | {self_rel}
    targets = [f for f in source_files("*")
               if Path(f).suffix not in _BINARYISH and f not in skip_names]
    hits = []
    for rel in targets:
        p = REPO / rel
        try:
            if p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for marker in _GPL_MARKERS:
            if marker in text:
                hits.append(f"{rel}: contains {marker!r}")
                break
    for h in hits:
        print("   !", h)
    for allowed in sorted(_GPL_TEXT_ALLOWED):
        print(f"    (allowed: {allowed} — third-party attribution notice, "
              "reproduces the LGPL texts the shipped dependencies require)")
    record("no-gpl-in-tree", not hits,
           f"{len(hits)} GPL-licensed file(s) in {len(targets)} tracked files")


def check_no_env_import() -> None:
    if not ENGINE.is_dir() or not any(ENGINE.iterdir()):
        skip("no-env-import", f"engine not checked out at {ENGINE}")
        return
    print("\n$ grep -rnE --include='*.py' --exclude-dir=kicad-python "
          "'^[[:space:]]*(import|from)[[:space:]]+(pcb_world|methods|eval)\\b' "
          f"{ENGINE}")
    targets = engine_files(".py")
    hits = []
    for p in targets:
        text = p.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _IMPORT_RE_ENV.match(line):
                hits.append(f"{p}:{lineno}: {line.strip()}")
    for h in hits:
        print("   ", h)
    record("no-env-import", not hits,
           f"{len(hits)} environment imports in {len(targets)} engine files")


def check_wire_copies() -> None:
    a = REPO / "pcb_world/engine/wire.py"
    b = ENGINE / "engine_server/wire.py"
    if not b.exists():
        skip("wire-copies-match", f"engine not checked out at {ENGINE}")
        return
    print(f"\n$ sha256sum {a} {b}")
    digests = []
    for p in (a, b):
        if not p.exists():
            record("wire-copies-match", False, f"missing {p}")
            return
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        digests.append(d)
        print(f"    {d}  {p}")
    record("wire-copies-match", digests[0] == digests[1],
           "identical" if digests[0] == digests[1] else "DIVERGED")


def check_no_combined_build() -> None:
    """No packaging recipe may put both programs into one distributable.

    A packaging recipe confined to one side is fine (it can only ever produce
    a single-program artifact). What must not exist is a recipe rooted where it
    could sweep in both, or one that names the other side's tree.
    """
    print("\n$ find . -name '*pyproject.toml' -o -name 'setup.py' -o -name "
          "'setup.cfg' -o -name 'MANIFEST.in' -o -name 'Dockerfile*'")
    recipes = source_files("*pyproject.toml", "*setup.py", "*setup.cfg",
                           "*MANIFEST.in", "*Dockerfile*", "*docker-compose*")
    problems = []
    for rel in sorted(recipes):
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        has_backend = any(
            ln.strip().startswith(m) for ln in code
            for m in ("[build-system]", "[tool.setuptools", "[tool.hatch",
                      "[tool.poetry", "[tool.flit"))
        # A comment may point at the engine; only a packaging line matters.
        names_engine = any("engine/" in ln for ln in code)
        print(f"    {rel:52s} backend={'yes' if has_backend else 'no ':3s}")
        if names_engine:
            problems.append(f"{rel}: packaging line names the engine tree")
        if has_backend and rel == "pyproject.toml":
            problems.append(
                "pyproject.toml (repository root) declares a build backend — a "
                "wheel built from the root would sweep in both programs")
    for pattern in ("*.whl", "*.tar.gz"):
        for p in REPO.glob(pattern):
            problems.append(f"built artifact present at the root: {p.name}")
    for p in problems:
        print("   !", p)
    detail = (f"{len(recipes)} packaging file(s), none spanning both programs"
              if not problems else f"{len(problems)} finding(s)")
    record("no-combined-build", not problems, detail)


def check_runtime_maps(evidence_dir: Path | None) -> None:
    """Route a board for real and read both processes' memory maps."""
    if not sys.platform.startswith("linux"):
        record("maps", False, "requires Linux (/proc)")
        return
    if not ENGINE.is_dir() or not any(ENGINE.iterdir()):
        skip("maps", f"engine not checked out at {ENGINE}")
        return

    probe = REPO / "tools" / "_separation_probe.py"
    print(f"\n$ python {probe.relative_to(REPO)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run([sys.executable, str(probe)], capture_output=True,
                          text=True, env=env, cwd=str(REPO))
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, end="", file=sys.stderr)
        record("maps", False, f"probe exited {proc.returncode}")
        return
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "proc_maps.txt").write_text(proc.stdout)
        print(f"    evidence written to {evidence_dir / 'proc_maps.txt'}")
    record("maps", "SEPARATION_PROOF: OK" in proc.stdout,
           "engine library loaded only in the engine-server child")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", action="store_true",
                    help="also run the live /proc/<pid>/maps proof")
    ap.add_argument("--evidence-dir", type=Path, default=None,
                    help="write captured evidence here")
    args = ap.parse_args()

    engine_present = ENGINE.is_dir() and any(ENGINE.iterdir())
    print(f"repository: {REPO}")
    print(f"engine:     {ENGINE}"
          + ("" if engine_present else "  (not checked out)"))
    check_no_gpl_in_tree()
    check_no_combined_build()
    check_no_env_import()
    check_wire_copies()
    if args.runtime:
        check_runtime_maps(args.evidence_dir)

    failed = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 68)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed"
          + (f" — FAILED: {', '.join(failed)}" if failed else "")
          + (f" — SKIPPED: {', '.join(n for n, _ in skipped)}" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
