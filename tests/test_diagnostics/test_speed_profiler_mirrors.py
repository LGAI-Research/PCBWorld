"""Checks that speed_profiler's timed mirrors stay in sync with their base functions.

The mirrors in hooks.py/worker_shim.py are copies of the base control flow with
timers added, so a change to a base would leave the profiler measuring a stale
path. The base source digests pinned in mirror_contract.BASES are compared against
the current sources, so drift fails loudly. (No C++ router or torch needed — the
digest is a static AST extraction using only the stdlib. The same check is enforced
pre-push through tools/docs/check_docs.py `mirror-sync`.)
"""


def test_mirror_bases_unchanged():
    from tools.diagnostics.speed_profiler.mirror_contract import BASES, check

    drift = check()
    assert not drift, (
        "speed_profiler mirror drifted from its base — re-sync the affected "
        "mirror in hooks.py/worker_shim.py to the base change, then refresh the digest "
        "(python -m tools.diagnostics.speed_profiler.mirror_contract):\n"
        + "\n".join(f"  {m}: base {b} expected {e} != actual {a}"
                    for m, b, e, a in drift)
    )
    assert len(BASES) == 4
