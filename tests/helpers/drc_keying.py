"""Shared helpers for tests/test_drc_incremental/ — KIID-keyed DRC comparison.

KIID is a stable VALUE key (not a pointer). Clearance violations are pairwise-local,
so they can be retained / scoped-recomputed by KIID; connectivity violations are
global and recomputed in full each pass.

Comparison convention:
  - clearance family (copper-clearance provider codes): compared EXACTLY by key —
    this is what incremental DRC guarantees bit-exact.
  - everything else (connectivity / per-item): compared by COUNT. The ratsnest's
    choice of which item pair represents an unconnected/dangling violation is
    nondeterministic run-to-run (verified: full vs full disagrees on the pair, same
    count), so its keys are not stable — but run_drc_incremental recomputes these via
    the same full providers as run_drc, so the count matches.
"""

BOARD = "tests/fixtures/sample_drc_violation.kicad_pcb"   # fully routed, ~224 clearance

# Copper-clearance provider codes == C++ isClearanceFamily() (drc_item.h enum,
# DRCE_FIRST=1): SHORTING_ITEMS=2, CLEARANCE=5, TRACKS_CROSSING=7,
# ZONES_INTERSECT=9, HOLE_CLEARANCE=16. The pairwise-local, scopable family.
CLR_CODES = {2, 5, 7, 9, 16}


def _key(v):
    return (v.error_code, v.layer, frozenset((v.item_a, v.item_b)))


def _items(v):
    return {v.item_a, v.item_b}


def _is_clr(v):
    return v.error_code in CLR_CODES


def _clr(viols):
    return {_key(v) for v in viols if _is_clr(v)}


def _conn_n(viols):
    return sum(1 for v in viols if not _is_clr(v))


def _assert_eq(incr, full, ctx):
    ci, cf = _clr(incr), _clr(full)
    assert ci == cf, f"{ctx}: clearance lost={len(cf - ci)} gained={len(ci - cf)}"
    assert _conn_n(incr) == _conn_n(full), \
        f"{ctx}: non-clearance count {_conn_n(incr)} != {_conn_n(full)}"


def _full(engine):
    engine.run_drc()
    return engine.get_drc_violations()


def _incr(engine):
    engine.run_drc_incremental()
    return engine.get_drc_violations()
