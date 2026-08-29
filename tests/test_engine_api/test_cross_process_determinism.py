"""Cross-process routing determinism (engine_seed / KIID seeding).

The router's ``engine_seed`` seeds KiCad's *process-global* KIID generator at
construction so that routing-created items (tracks/vias) get a reproducible UUID
stream. Those UUIDs are the obstacle/cluster sort tie-break, so a fixed seed must
make routing reproducible not just run-to-run in one process, but across
*separate OS processes* (different heap / ASLR).

A same-process check cannot see this: two engines in one interpreter share the
seeded generator. This test routes the same board+action sequence in two
independent ``subprocess`` workers and compares the full geometry+UUID
fingerprint:

  * same fixed seed  → byte-identical fingerprint (geometry AND uuids).
  * entropy seeding   → uuids diverge (control: proves the harness detects a
    difference, so the seeded-equality above is not vacuous).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
BOARD = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"

# Worker: seed the KIID generator, route a fixed multi-segment path (several
# UUIDs consumed), print a fingerprint. Run as a fresh process per invocation so
# each gets its own heap — the whole point of a cross-process check.
_WORKER = r"""
import sys, json, hashlib
import kicad_rl_router as krl

board, seed = sys.argv[1], int(sys.argv[2])
r = krl.RLRouter(board, "", seed, 250, 1000000)
r.build_connectivity()
r.set_routing_mode(krl.MODE_WALKAROUND)
assert r.start_route(0.0, 0.0, 0), "start_route failed"
for wx, wy in [(-0.5, 1.0), (0.0, 3.0)]:
    r.move(wx, wy)
    r.fix_route(wx, wy, force_finish=False)
assert r.fix_route(3.0, 5.0), "fix_route failed"

tracks = r.get_tracks()
geom = [(round(t.x1_mm, 6), round(t.y1_mm, 6), round(t.x2_mm, 6),
         round(t.y2_mm, 6), t.layer, t.net_code) for t in tracks]
full = [g + (t.uuid,) for g, t in zip(geom, tracks)]
print(json.dumps({
    "n": len(tracks),
    "geom_hash": hashlib.sha256(repr(geom).encode()).hexdigest(),
    "full_hash": hashlib.sha256(repr(full).encode()).hexdigest(),
}))
"""


def _run_worker(seed: int) -> dict:
    """Route in a fresh process (own heap) and return its fingerprint dict."""
    env = os.environ.copy()  # inherit whatever makes krl importable here
    env["PYTHONPATH"] = os.pathsep.join(
        [str(RL_MODULE_DIR), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, str(BOARD), str(seed)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"worker (seed={seed}) failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module", autouse=True)
def _require_router():
    if not (RL_MODULE_DIR / "kicad_rl_router").exists() and not list(
        RL_MODULE_DIR.glob("kicad_rl_router*")
    ):
        pytest.skip(f"kicad_rl_router not built at {RL_MODULE_DIR}")
    if not BOARD.exists():
        pytest.skip(f"Board not found: {BOARD}")


def test_fixed_seed_is_cross_process_identical():
    """Same fixed engine_seed in two separate processes → identical geometry+UUIDs."""
    a = _run_worker(77)
    b = _run_worker(77)
    assert a["n"] > 0
    assert a["full_hash"] == b["full_hash"], (
        "cross-process divergence with a fixed seed:\n"
        f"  proc A full_hash={a['full_hash']}\n"
        f"  proc B full_hash={b['full_hash']}"
    )


def test_entropy_seed_diverges_cross_process():
    """Control: entropy seeding (-1) → UUIDs differ across processes.

    Geometry may coincide on this simple board, but the routed items' UUIDs are
    drawn from an entropy-seeded generator, so the full fingerprint must differ —
    confirming the seeded-equality test above is not passing vacuously.
    """
    a = _run_worker(-1)
    b = _run_worker(-1)
    assert a["full_hash"] != b["full_hash"], (
        "entropy runs produced identical UUIDs across processes — "
        "the fingerprint is not UUID-sensitive, so the harness cannot detect divergence"
    )


# Shove-mode worker: same route driven in RM_Shove. Residual address-order
# (exact-duplicate temporaries at the shove iteration budget) leaks precisely in
# shove paths that walkaround never exercises — the ITEM::Serial tie-break
# covers it, and this pins that cross-process.
_WORKER_SHOVE = _WORKER.replace("MODE_WALKAROUND", "MODE_SHOVE")


def _run_shove_worker(seed: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(RL_MODULE_DIR), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SHOVE, str(BOARD), str(seed)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"shove worker (seed={seed}) failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_shove_route_cross_process_identical():
    """RM_Shove route in three separate processes → identical geometry+UUIDs."""
    runs = [_run_shove_worker(77) for _ in range(3)]
    assert runs[0]["n"] > 0
    hashes = {r["full_hash"] for r in runs}
    assert len(hashes) == 1, (
        f"shove-mode cross-process divergence (fixed seed): {sorted(hashes)}"
    )
