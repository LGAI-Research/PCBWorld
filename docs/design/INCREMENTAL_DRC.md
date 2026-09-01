# Incremental DRC — design notes

DRC is the single largest bottleneck in RL training/inference. This document
records (1) where the cost actually is, measured; (2) what does **not** help;
(3) a verified-but-rejected fast approximation; and (4) the design for an
**exact** incremental DRC that reuses KiCAD's real clearance test.

Status: **implemented** — `runDRCIncremental` landed with the 2026-06-29 `mcts`
merge (`9f0c19d6a`): KIID-signature diff + retained clearance-family store +
scoped provider pass (≈ tier A of §6 with the §4 violation store, not the §7
self-maintained index). High-level summary: the engine README;
bit-exactness guard: [tests/test_drc_incremental/](../../tests/test_drc_incremental/).
This document is kept as the **decision record** (measurements · rejected
alternatives · design rationale). Measurements 2026-06-23, macOS, conda env
`cadagent`, instrumented build then reverted.

---

## 1. Where the cost is

`run_drc()` is full-board every call. In per_step reward mode it runs before+after
every valid step ([env.py](../../pcb_world/core/env.py), `step_drc`); also at
episode end ([env.py](../../pcb_world/core/env.py)) and in eval. Measured
~89% of `env.step` wall-time.

Breakdown of one `run_drc()` (instrumented `PNS_RL_ROUTER::runDRC` +
`DRC_ENGINE::RunTests`):

| stage | pic_programmer (370 tracks) | video (7932 tracks) |
|---|---|---|
| `BuildConnectivity` | 5.6 ms | 73–85 ms |
| construct + `InitEngine` (rule compile) | **0.7 ms** | 1–11 ms |
| `DRC_CACHE_GENERATOR` (per-layer R-tree + connectivity) | 10 ms | 105–124 ms |
| **copper clearance provider** (all-pairs) | **266 ms** | 310–330 ms |
| all other providers (width/hole/via/annular/connectivity…) | ~4 ms | ~25 ms |
| **total** | **~288 ms** | **~540 ms** |

The dominant cost is the **copper clearance provider**, which is *already*
multi-threaded (`GetKiCadThreadPool()`, 11 cores here). So it is genuinely
heavy all-pairs work, not an un-parallelized accident.

---

## 2. What does NOT help (rejected "free wins")

- **Reusing the DRC engine / caching `InitEngine`.** Rule compilation is
  0.7–1 ms. Reusing the engine saves nothing. (The router already keeps a
  persistent `bds.m_DRCEngine` — see [pns_rl_router.cpp](../../engine/kicad-patches/rl/pns_rl_router.cpp).)
- **Pruning irrelevant test providers** (silk/courtyard/solder-mask/creepage/
  text…). Each provider top-level short-circuits via
  `IsErrorLimitExceeded()` when its severity is `ignore`, so the irrelevant
  ones already cost ~0. Non-clearance providers total ~4 ms (pic) / ~25 ms
  (video). Pruning saves single-digit ms.

Conclusion: the only lever that touches the dominant cost is making the
**clearance** computation incremental.

---

## 3. Rejected fast path: PNS `QueryColliding` (approximate)

A prototype `PNS_RL_ROUTER::queryClearancePNS()` derived cross-net clearance
violations from the PNS world via `NODE::QueryColliding`, instead of the DRC
sweep. The PNS rule resolver resolves clearance through the **same** engine
(`PNS_PCBNEW_RULE_RESOLVER::Clearance` → `bds.m_DRCEngine->EvalRules`, see
build tree `pcbnew/router/pns_kicad_iface.cpp:422`), so thresholds match.

Verification:

- **Recall ✅** — on `examples/figures/figure3_6pos_2layers_crossed.kicad_pcb`,
  DRC's "Tracks crossing" F1↔F2 was reproduced exactly (same net-pair).
- **Speed ✅** — pic_programmer full pass **0.2 ms vs 290 ms ≈ 1383×**
  (incremental would be faster still).
- **Precision ✗ (the dealbreaker)** — on clean pic_programmer, PNS reported
  **1** clearance hit, DRC reported **0**. The colliding pair: a net-24
  (`/pic_sockets/VCC_PIC`) track segment that starts at **JP1.pad2** vs
  **JP1.pad1**, a **custom-shape SMD jumper pad** on net-17 (`VCC`). Whether
  this is a violation depends on the exact custom-polygon distance and net-tie
  semantics — computed only by DRC's
  `shape->Collide(clearance - epsilon, &actual)` + `IsNetTieExclusion()`.
  PNS uses net-code inequality + a **hull over-approximation**, so it cannot
  match DRC exactly here, by construction.

**Decision:** the requirement is *bit-identical* to `run_drc()`. PNS
`QueryColliding` is therefore the wrong tool and the prototype was reverted.
(It remains viable only as a *fast pre-filter* if one ever wants a superset,
but it is **not** a guaranteed superset — it does not model copper zones /
graphics — so it cannot be used to filter candidates for an exact result.)

---

## 4. Exact incremental DRC — the core idea (reuse, don't reimplement)

> Which *tier* of this to build (how much KiCAD to touch) is a throughput
> decision driven by the PPO use case — see §5 (exactness) and §6 (scope).
> §7 gives the concrete recommended design (self-maintained index, no engine
> surgery).

Key invariant: **any clearance violation that appears or disappears between two
consecutive states must involve at least one changed item.** Two items that
both stayed put were already checked last step and cannot change. So we can run
the *real* DRC clearance test on only the changed ("dirty") items and patch a
persistent violation set — the result is identical to a full sweep.

### Reuse, don't reimplement

The full clearance provider is structurally:
`for each item A: query R-tree for neighbors B near A: exact per-pair test(A,B)`
(build tree `pcbnew/drc/drc_test_provider_copper_clearance.cpp`,
`testTrackClearances` ~640, `testPadClearances` ~1002, per-pair
`testSingleLayerItemAgainstItem` ~208).

The per-pair test already applies **every** exemption that PNS missed:
same-net skip, `IsNetTieExclusion()`, `SameLogicalPadAs`, layer/flash checks,
and the exact `SHAPE::Collide` geometry. Restricting only the **A-loop** to the
dirty set keeps that code path untouched → bit-identical to the subset of full
DRC. The B-side still queries the full per-layer R-tree
(`m_board->m_CopperItemRTreeCache`, `DRC_RTREE::QueryColliding`,
build tree `pcbnew/drc/drc_rtree.h`).

### The pieces

1. **Restrict the clearance provider's A-loop to a dirty set.** Inject the dirty
   item set so `testTrackClearances`/`testPadClearances` iterate only those (B
   side stays full). Needs a small patch to the vendored provider (the per-pair
   methods are `private`).
2. **Persistent violation store with UUID invalidation.** Keep last step's
   violations; drop any referencing a removed/modified item; add the newly found
   ones. Make the violation set part of the `RLCheckpoint` so MCTS restore
   brings it back in lockstep with geometry — the same trick used for tracks
   (the incremental restore that gives 40.6×).

These two are what shipped (see status). The remaining per-step floor —
`DRC_CACHE_GENERATOR` rebuilding the per-layer R-tree + connectivity on every
`RunTests` (build tree `pcbnew/drc/drc_engine.cpp:650`) — is a throughput
question, not correctness: in-engine incremental caches are tier **C** in §6
(most invasive, skipped); §7's self-maintained index reaches the same floor
without engine surgery, if it is ever needed. Per-tier costs: §6 table.

### Caveats / open items

- KiCAD has **no** existing incremental hook. `RunTests`'s `BOARD_COMMIT*
  aCommit` parameter is a dead vestige — declared, never read (build tree
  `pcbnew/drc/drc_engine.cpp:626`). This must be built.
- Requires patching vendored KiCAD DRC source (expose/refactor the private
  per-pair test, or add a friend entry point).
- Per-type error caps (`ERROR_LIMIT` 199 / `EXTENDED_ERROR_LIMIT` 499) have
  slightly different semantics under incremental accumulation; irrelevant for
  routing-scale counts but note it.
- `IsNetTieExclusion` reads footprint net-tie groups — already handled because
  we call the real per-pair test; do **not** reimplement it.

(Validation: superseded by the §7.8 ladder and the shipped guard
[tests/test_drc_incremental/](../../tests/test_drc_incremental/).)

---

## 5. PPO usage changes the exactness requirement

This will run in **PPO training**, not only MCTS. PPO per_step mode computes a
reward every step and runs many parallel envs over millions of steps, so the
per-step DRC is the throughput-critical hot path (MCTS could defer DRC to
terminal via horizon mode; per_step PPO cannot).

The per-step reward is `r = Φ(after) − Φ(before) − step_penalty` — a potential
**delta**, not an absolute. The DRC term enters Φ via `_drc_penalty`
([reward.py](../../pcb_world/core/reward.py)) with shapes `linear` / `saturating` /
`log_per_net`. This relaxes "must match exactly":

- **Linear** (`penalty × count`): a *static* discrepancy (e.g. the §3 JP1 false
  positive) appears identically in before and after → **cancels exactly in ΔΦ**.
  Per-step needs correct *deltas*, not bit-exact absolutes.
- **Saturating / log_per_net** (nonlinear): the same static offset does **not**
  fully cancel — `f(real+1) − f(real′+1) ≠ f(real) − f(real′)` when that net's
  count changes. A static error perturbs the delta whenever the affected net is
  routed.

**Consequence:** with a nonlinear shape (in use here), per-step needs **exact
deltas** → the PNS proxy (§3) is out, the reused-per-pair-test path (§4/§7) is
required. Bit-exact *absolutes* are only needed at terminal/eval, which already
runs full `run_drc()` once per episode (cheap relative to the episode).

Confirm before building:
- **Telescoping** — per_step mode reuses `prev_state` as the next before-state
  ([env.py](../../pcb_world/core/env.py)); intermediate Φ cancels. Check the
  per-step (incremental) ↔ terminal (full `run_drc`) boundary is consistent so
  the last step isn't double-counted or mismatched.
- **Other absolute-DRC consumers** — critic target, DRC state tokens
  ([token_vocabulary.py] `encode_drc`), logged metrics: these read the absolute
  count, where the delta-cancellation argument does **not** apply. Handle them
  separately (e.g. keep feeding them the incremental absolute, which is exact by
  construction here).
- **Parallel-env safety** — each env owns its own `RLRouter`/board, so a
  per-router index + violation store is naturally isolated (no shared state).

---

## 6. Scope: engine surgery is a throughput question, not correctness

Given §5, *correctness* is satisfied by reusing the real per-pair test on the
dirty set + stock terminal DRC. The remaining axis is *throughput*, which
decides how much KiCAD we touch:

| tier | touches | per-step cost | when |
|---|---|---|---|
| **A** — provider 1-method + reuse cachegen | clearance provider (1 file) | pic ~16 ms / video ~185 ms | medium boards; big-board cachegen floor too slow for PPO |
| **B** — self index + reuse per-pair test (§7) | provider per-pair *exposure* only; `DRC_ENGINE`/cache **untouched** | target single-digit ms | **recommended for PPO** |
| **C** — in-engine incremental cache (see §4 end) | `DRC_ENGINE` + `DRC_CACHE_GENERATOR` | single-digit ms | most invasive / version-fragile; avoid unless B proves insufficient |

Tier B is the sweet spot: the per-pair test takes `(itemA, shapeA, layer,
itemB)` and **does not care where `itemB` came from**. So we can replace the
obstacle source (`m_CopperItemRTreeCache`, rebuilt by cachegen each call) with
our own incrementally-maintained index — removing the cachegen floor **without**
engine surgery. Only one vendored edit remains: exposing the per-pair test.

---

## 7. Concrete design — incremental neighbour-index from the engine snapshot

Goal: a per-router obstacle index that answers "items within max-clearance of
item X on layer L", updated incrementally from the data the engine already
produces, feeding the **reused** DRC per-pair test. No `DRC_ENGINE`/cachegen
change.

### 7.1 What the index holds
Copper obstacles keyed by `KIID`, bucketed per copper layer, bbox **inflated by
`bds.m_DRCMaxClearance`** (so a bbox-overlap query never misses a real
neighbour — the same inflation `DRC_RTREE` uses):
- **static** (insert once at construction): pads, copper graphics/text, board
  edges (for edge-clearance-style hits), copper zones (see 7.7);
- **dynamic** (insert/remove as the agent routes): tracks, vias, arcs.

### 7.2 Container choice
- **Reuse KiCAD's `DRC_RTREE`** (`pcbnew/drc/drc_rtree.h`) as the container, but
  **owned and maintained by us** (`Insert` on add, `Remove` on rip-up) instead
  of letting cachegen rebuild it. Gain: identical query semantics to the
  provider (no missed neighbours), zero spatial code to write.
- Alternative: uniform spatial-hash grid (cell ≈ maxClearance + max item). Even
  simpler incremental add/remove, but we own the query geometry. Prefer
  `DRC_RTREE` for parity.

### 7.3 Change feed — reuse the snapshot / checkpoint diff
The engine already emits a per-step `BoardSnapshot` (tracks/vias/pads/ratsnest,
pybind by-ref) and the checkpoint restore already computes added/removed/modified
sets by `KIID` + `boardItemsEqual`
([pns_rl_router.cpp](../../engine/kicad-patches/rl/pns_rl_router.cpp)). Drive the
index from the **same** change set:
- **Linear rollout (PPO)** — hook the mutation points directly: `fixRoute`
  (commit → added tracks/vias), `delete_*` (removed), so the delta is known at
  the hook with no diffing.
- **Tree jumps (MCTS restore)** — diff the restored checkpoint's `KIID`
  fingerprint set (≈ the snapshot stored with the node) against the live index
  contents; apply the add/remove. O(changed).

### 7.4 Per-step protocol (shape only — not implementation)
```
on_step(added, removed):                      # dirty set from hook / diff
    for it in removed: index.Remove(it); store.invalidate_referencing(it)
    for it in added:   index.Insert(it)
    for T in (added ∪ modified):
        for layer in T.layers():
            for B in index.QueryColliding(T, layer, maxClearance):
                v = perPairTest(T, T.shape(layer), layer, B)   # REUSED DRC test
                if v: store.upsert(canonical_uuid_pair(T, B), v)
    return store.current()        # == full-DRC clearance family, maintained incrementally
```
- `perPairTest` = the reused `testSingleLayerItemAgainstItem` (emits clearance /
  shorting / hole-clearance, with all exemptions). Capture its output via a thin
  violation collector instead of the GUI marker handler.
- `store` keyed by canonical `KIID` pair → dedups T-vs-T (found from both sides)
  and gives O(1) invalidation by item.

### 7.5 Checkpoint integration
Add `m_drcViolationStore` to `RLCheckpoint` (the index can be rebuilt from the
restored tracks, or snapshotted too). Restore brings the violation set back in
lockstep with geometry — the same pattern as the track clones that give the
40.6× incremental restore.

### 7.6 Scope of edits
- **New, in our code** (per-router → parallel-safe): index, violation store,
  update hooks, the `on_step` entry point + a Python binding.
- **One vendored patch**: expose the per-pair test — a small public wrapper on
  `DRC_TEST_PROVIDER_COPPER_CLEARANCE`, or hold a constructed provider instance.
  This is the **only** DRC-source edit.
- `bds.m_DRCMaxClearance` is a scalar (compute once); net info via the existing
  incremental connectivity.

### 7.7 Exactness & caveats
- Each reported violation is the real DRC per-pair result → bit-identical values
  + every exemption (net-tie, same-logical-pad, flash). The §3 false positive
  cannot occur here because we call DRC's own test, not a hull approximation.
- Completeness rests on (a) the invariant (any changed violation involves a
  dirty item) and (b) **all static obstacles being in the index** so a new track
  near a static pad/graphic/edge is caught.
- **Zones**: copper pours need the zone path (`testItemAgainstZone` / zone
  rtree). If RL boards have no pours, skip; else run the zone sub-test for dirty
  items or index zone outlines. Decide per dataset — explicit open item.
- Hole-clearance and shorting come for free (same per-pair test).
- Per-step exactness is required for nonlinear shapes (§5); terminal stays stock
  `run_drc()`.

### 7.8 Validation ladder
1. `index.QueryColliding` neighbour set ⊇ DRC's candidate set (no missed
   neighbours) on several boards.
2. `on_step(dirty = all items)` store == full `run_drc()` clearance family,
   **bit-identical**.
3. `on_step(dirty = changed)` == full `run_drc()` at that state; measure
   per-step ms.
4. PPO smoke: per-step ΔΦ from the incremental path == ΔΦ from full `run_drc()`
   over a rollout — especially under a nonlinear shape, where static offsets do
   not cancel (§5).

---

## Appendix: environment / repro

- Env / full build: [README](../../README.md) (conda `cadagent`, module path),
  the engine's own README (`engine/README.md`: patch → build flow). Quick C++ iteration: edit
  `build_rl/kicad_src/...` (gitignored), `cd build_rl && ninja kicad_rl_router`
  (~5 s per TU); canonical sources in [kicad-patches/rl/](../../engine/kicad-patches/rl/).
- Known trap: MarkObstacles mode (`set_routing_mode(0)`) + raw `fix_route`
  segfaults — do not use it to synthesize violations. Use a board that already
  has them (e.g. the `figure3_6pos_2layers_crossed` fixture).
