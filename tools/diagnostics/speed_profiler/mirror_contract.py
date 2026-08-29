"""Mirror↔base sync contract for the timed mirrors.

``hooks.py`` / ``worker_shim.py`` contain timed MIRRORS (verbatim copies of a
base function's control flow with timers added). If a base body changes, the
mirror silently measures a stale code path — so this module pins the sha256
of each base callable's source, and ``tests/test_diagnostics/
test_speed_profiler_mirrors.py`` fails loudly on drift.

Re-sync procedure when the check fails:
  1. Diff the base function against its mirror and port the change into the
     mirror (hooks.py / worker_shim.py), keeping timers/buckets intact.
  2. Refresh the digests below::

         python -m tools.diagnostics.speed_profiler.mirror_contract

     and paste the printed BASES block here.

Digests are extracted STATICALLY (ast over the module file — no import), so
this module is stdlib-only and the check runs in any env: enforced both by
``tests/test_diagnostics/test_speed_profiler_mirrors.py`` and, at push time,
by the ``mirror-sync`` check in ``tools/docs/check_docs.py`` (pre-push hook).
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# mirror location -> (base module, base qualname, sha256[:16] of base source)
BASES: dict[str, tuple[str, str, str]] = {
    "hooks.rollout_decomp": (
        "methods.rl_agent.rollout.primitive", "iter_rollout", "0e00823aa2523c05"),
    "hooks.update_decomp": (
        "methods.rl_agent.algorithms._common", "_fixed_batch_step", "7092b166ca194c45"),
    "hooks.update_decomp/chunk": (
        "methods.rl_agent.algorithms._common", "_accumulate_chunk", "a1123a1704568a84"),
    "worker_shim._decoder_worker": (
        "pcb_world.vec.backends.subproc", "_decoder_worker", "c046d07a6521ecbb"),
}


def source_digest(module_name: str, qualname: str, root: Path | None = None) -> str:
    """sha256[:16] of the base callable's source segment, read statically."""
    root = Path(root) if root is not None else _REPO_ROOT
    path = root / (module_name.replace(".", "/") + ".py")
    src = path.read_text(encoding="utf-8")
    node = ast.parse(src)
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and child.name == part):
                node = child
                break
        else:
            raise LookupError(f"{qualname!r} not found in {path}")
    seg = ast.get_source_segment(src, node)
    if not seg:
        raise LookupError(f"no source segment for {qualname!r} in {path}")
    return hashlib.sha256(seg.encode()).hexdigest()[:16]


def check(root: Path | None = None) -> list[tuple[str, str, str, str]]:
    """Return [(mirror, base, expected, actual)] for every drifted base."""
    drift = []
    for mirror, (mod, qual, expected) in BASES.items():
        actual = source_digest(mod, qual, root)
        if actual != expected:
            drift.append((mirror, f"{mod}.{qual}", expected, actual))
    return drift


if __name__ == "__main__":
    print("BASES: dict[str, tuple[str, str, str]] = {")
    for mirror, (mod, qual, _) in BASES.items():
        print(f'    "{mirror}": (\n        "{mod}", "{qual}", "{source_digest(mod, qual)}"),')
    print("}")
