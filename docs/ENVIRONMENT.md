# Environment — state, actions, reward

`PCBWorld` poses board routing as an MDP: what the agent sees, what it can do, and how a step
is scored. Enough here to read an observation, build an action, and follow the reward. The
exact contract is in the module docstrings. The paper ([PCBWorld.pdf](../PCBWorld.pdf), §3.2
and Appendices E–H) has the derivations.

## Episode

`env.reset()` loads a `.kicad_pcb` and strips its routing. Components and pads stay where they
are. Every net starts unrouted.

The agent then routes one net at a time, one `env.step(action)` per arrow:

```
net_select → start_route → make_line / make_via … (or finish) → net_end → next net
```

`env.step()` returns `terminated=True` when every net is connected, or when the agent has
released every net. It returns `truncated=True` at `max_steps`. `info["success"]` is true only
for a fully connected board.

Every track goes through KiCad's own push-and-shove router, and every step is checked by
KiCad's DRC. A track KiCad would refuse is refused here too.

## State

The state is one nested dict, shaped like the board itself: net → pad → coordinate. Values are
exact numbers in mm, not normalized. No grid, no rendered image. Design rules are checked
against exact geometry, so the agent is given exact geometry
([observation.py](../pcb_world/core/observation.py)).

The top level splits by what changes during routing:

| Key | Changes | Holds |
|---|---|---|
| `board_static` | never (set by `env.reset()`) | outline bbox, copper layer count, per-net pads, net-less obstacles, design-rule constraints |
| `routing_geometry` | every step | per-net tracks, vias, and the open ratsnest endpoints |
| `router_head` | every step | where the router is: position, layer, current net, `is_routing`, step counters |
| `drc_violations` | every step | DRC violations near the head |
| `action_history` | every step | the last few actions: type, point, net, and whether the router accepted each one |
| `closed_nets` | every step | nets already released with `net_end` |

Below that the nesting continues. Every object carries an `id` and its geometry: a pad or via
has a `center`, a track has endpoints `p1` and `p2`, and both carry a `layer` and a width. The
access path is the hierarchy, so a pad center reads as
`obs["board_static"]["nets"]["net_1"]["pads"][<pad>]["center"]["xy"]`.

Two wrappers re-encode this dict, one per encoding.

### Token wrapper

Read by the decoder-only policy. One token per object. A token is an entity-type embedding
plus a projection of that object's features. Coordinates are normalized by the board center
and scale first, so the same geometry gives the same token on any board. The types
([spec.py](../methods/rl_agent/models/v1/spec.py) `EntityType`):

| Token | One per |
|---|---|
| `BOARD` | board |
| `EDGE` | outline segment |
| `NET` | net |
| `PAD`, `TRACK`, `VIA` | pad, track, via |
| `RAT` | open ratsnest endpoint |
| `HEAD` | router head |
| `CAND_PAD`, `CAND_TRACK_END`, `CAND_VIA`, `CAND_RAT`, `CAND_DIR` | candidate point (see Actions) |
| `DRC_VIOLATION` | violation |
| `OBSTACLE` | net-less blocker (opt-in) |

### Text wrapper

Read by tool-calling LLM agents. The same dict as S-expressions, nesting and keys
unchanged. The bare fixture board, abridged:

```
(board_static
  (bbox -0.075 -0.075 50.150 30.150) (net_count 4) (copper_layers 2)
  (boardlines (edge E0 (p1 P0 0.000 0.000) (p2 P1 50.000 0.000)) ...)
  (nets
    (net 1 "NET1" (pads (pad D0 10.000 10.000 1) (pad D1 40.000 10.000 1))) ...)
  (board_constraints (clearance 0.200) (track_width 0.200) ...))
(routing_geometry
  (net 1 (tracks) (vias) (points (point Q0 10.000 10.000) (point Q1 40.000 10.000))) ...)
(router_head
  (xy 0.000 0.000) (layer -1) (net -1) (phase 0) (is_routing false) (routing_mode w))
```

Nothing is routed yet, so every `tracks` list is empty and each net's two pads are still open
`points`. Pad `D0` of net 1 sits at (10, 10) on layer 1.

## Actions

An action is a dict: an `action_type` plus the parameters that type takes. Six routing types
and one no-op ([action_schema.py](../pcb_world/core/action_schema.py)):

| `action_type` | Parameters | Does |
|---|---|---|
| `net_select` | `net_id` | route net `net_id` from now on |
| `start_route` | `x_mm`, `y_mm`, `layer` | open a route at `(x_mm, y_mm)` on `layer` — a pad of that net |
| `make_line` | `x_mm`, `y_mm`, `routing_mode` | extend the route to `(x_mm, y_mm)` on the current layer |
| `make_via` | `x_mm`, `y_mm`, `routing_mode` | extend to `(x_mm, y_mm)`, then place a via there |
| `finish` | `routing_mode` | let the router complete to the nearest open pad |
| `net_end` | — | release the current net |
| `idle` | — | no-op fallback when a text action fails to parse; masking never grants it, so a token policy cannot select it |

`routing_mode` is how KiCad treats copper in the way: `0` mark obstacles (straight line, fails
on collision), `1` shove it aside, `2` walk around it. `layer` is 1-based, 1 = top.

**Points.** `x_mm`, `y_mm` are plain mm coordinates, but an agent is not left to invent them.
Every step it is offered a candidate set of the points that matter for the current net: the
pad centers, the endpoints and via centers of the copper drawn so far, the open ratsnest
targets, and eight near neighbours a fixed short step around the router head
([candidate_pool.py](../pcb_world/vec/candidate_pool.py)). The agent picks one of them.

The token wrapper enforces this. The policy decodes a triple `(action_type, pointer into the
candidate set, routing_mode)`, fills only the slots the type uses, and a pointer cannot leave
the set. The text wrapper lists the same points but also accepts a free coordinate, which is
how it takes a detour waypoint.

**Mask.** Which actions are legal depends on the router's current state, and that is a YAML
rule in [configs/masking/](../configs/masking/) — one condition per action. `env.action_masks()`
returns the resulting boolean mask; a masked action is rejected before it reaches the router.

## Reward

The reward is potential-based. A potential Φ(s) scores the whole board:

```
Φ(s) = − unconnected(s) − f_drv(s) − λ_w · wirelength(s) − λ_v · vias(s) + completion bonuses
```

`f_drv` grows with the DRC violation count and is steepest near zero, so a clean board is worth
far more than a short one. The bonuses pay out per connected net, per clean net, and once for
a finished board. Weights and the shape of `f_drv` live in
[pcbworld_reward.yaml](../configs/reward/pcbworld_reward.yaml); the code is
[reward.py](../pcb_world/core/reward.py).

Two ways to hand it out (`mode` in that file):

- `terminal`: 0 every step, Φ(s_T) at the end.
- `per_step`: Φ(s_{t+1}) − Φ(s_t) every step.

With γ = 1 the per-step rewards telescope to Φ(s_T) − Φ(s_0), so both modes share the optimal
policy. `per_step` is the dense form for PPO. GRPO only sees the episode return, so for it the
two coincide.

PPO then trains at γ = 0.995 rather than 1 (`gamma` in
[configs/rl.yaml](../configs/rl.yaml)), so the telescoping is only approximate: a
gain made later counts slightly less than the same gain made now.

On top of that: a `step_penalty` every step, and a small penalty for an action that was masked,
failed to parse, or changed nothing.

On the last step `env.step()` puts Φ(s_T) − Φ(s_0) in `info["potential_gain"]`. The benchmark
reports the same quantity as its routing-quality metric ([docs/METRICS.md](METRICS.md)).
