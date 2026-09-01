"""Reset re-initialises the KIID/UUID generator → bit-identical episodes.

The process-global KIID generator is seeded once at engine construction and
otherwise advances monotonically as routing mints new track/via UUIDs. Without
intervention it keeps advancing ACROSS episodes (``reset()`` deletes tracks but
never re-seeds), so successive episodes on the same board draw a different UUID
stream. Because the PNS obstacle set is ordered by UUID, that stream drift
can flip tie-breaks and make otherwise-identical episodes route differently —
exactly the "flips with unrelated in-process engine history" caveat pinned in
``test_scripted_routing.py``.

``PCBWorld.reset()`` now calls ``engine.rewind_kiid_to_episode_start()`` as its
last engine step, rewinding the generator to the construction-time position so
every episode draws the SAME stream. These tests lock that contract in.
"""
import hashlib
import os

import pytest

from pcb_world.core.env import PCBWorld

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "two_net_multiterm_board.kicad_pcb"
)

# Fixed engine-level route sequence (start pad, layer, target) — identical every
# episode. The exact geometry is irrelevant; only that the sequence is fixed so
# any cross-episode difference is attributable to UUID-stream drift.
_ROUTE_SEQ = [
    (10.0, 10.0, 1, 40.0, 10.0),
    (10.0, 20.0, 1, 40.0, 20.0),
    (25.0, 5.0, 1, 25.0, 10.0),
]


def _episode_fingerprint(env: PCBWorld) -> tuple[tuple[str, ...], str]:
    """Route the fixed sequence on the freshly-reset env, return
    (sorted routed-track UUIDs, geometry hash)."""
    eng = env._engine
    for sx, sy, layer, ex, ey in _ROUTE_SEQ:
        if eng.start_route(sx, sy, layer):
            eng.fix_route(ex, ey, True)
    eng.build_connectivity()
    tracks = eng.get_tracks()  # TrackInfo.uuid rides the engine API (both transports)
    uuids = tuple(sorted(str(t.uuid) for t in tracks))
    geo = hashlib.sha1(
        "".join(
            f"{t.x1_mm:.4f},{t.y1_mm:.4f},{t.x2_mm:.4f},{t.y2_mm:.4f};" for t in tracks
        ).encode()
    ).hexdigest()
    return uuids, geo


def test_seeded_episodes_are_bit_identical() -> None:
    """Seeded engine: reset() rewinds KIID, so every episode's routed UUIDs AND
    geometry are byte-for-byte identical, with no duplicate UUIDs."""
    env = PCBWorld(board_path=BOARD_PATH, max_steps=50, engine_seed=77)

    fps = []
    for _ in range(4):
        env.reset()
        fps.append(_episode_fingerprint(env))

    uuids0, geo0 = fps[0]
    assert len(uuids0) == len(set(uuids0)), "duplicate UUIDs within an episode"
    for i, (uuids, geo) in enumerate(fps[1:], start=1):
        assert uuids == uuids0, f"episode {i} routed UUIDs drifted from episode 0"
        assert geo == geo0, f"episode {i} geometry drifted from episode 0"


def test_entropy_mode_is_not_pinned() -> None:
    """engine_seed=None (entropy) leaves the generator un-captured, so the rewind
    is a no-op and episodes keep drawing fresh UUIDs — i.e. NOT pinned. (Geometry
    may still coincide; the UUID stream is the reliable drift signal.)"""
    env = PCBWorld(board_path=BOARD_PATH, max_steps=50, engine_seed=None)

    env.reset()
    uuids_a, _ = _episode_fingerprint(env)
    env.reset()
    uuids_b, _ = _episode_fingerprint(env)

    assert uuids_a != uuids_b, "entropy-seeded episodes should not share a UUID stream"
