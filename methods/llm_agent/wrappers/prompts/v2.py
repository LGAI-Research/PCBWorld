# --------------------- CadAgent V2 (KiCad PCB Routing) --------------------- #
#
# V2 augmentations over V1 (cadagent.py):
#  1) Action Reference — 6 actions documented with signatures and semantics
#  2) Action Masking Rules — YAML-style precondition table
#  3) Routing Modes — m/p/w each described
#  4) Few-shot example — 7-step Via Crossing walkthrough with real sexpr state
#
# Interface-compatible with V1 (drop-in via CADAGENT_PROMPT_VERSION=v2 switch
# in cadagent.py). All exported names match V1. CADAGENT_USER_PROMPT keeps the
# wschoi-branch history format `{current_step} / {valid_step_count}` to match
# rollout/cadagent.py and env_manager.py callers.
# -----------------------------------------------------------------------------

# ======================================================================
# Board state format descriptions (same as V1)
# ======================================================================

_STATE_FORMAT_DESC_XML = """\
The board state is provided in XML-like format. All floating-point values are rounded to 3 decimal places. Each element is a tag with attributes and optional children: <tag attr=val ... /> or <tag attr=val ...> <child ... /> </tag>.

## board_static — static board context (unchanged during episode)
  <board_static bbox_x=X bbox_y=Y bbox_w=W bbox_h=H scale=N net_count=N copper_layers=N> — root tag; per-attr description below
    bbox_x bbox_y bbox_w bbox_h  — board bounding box origin and size in mm
    scale                        — max(w, h), for normalization reference
    net_count                    — total number of nets
    copper_layers                — number of copper layers
  <boardlines> <edge id="..."> <p1 id="..." x=X y=Y /> <p2 id="..." x=X y=Y /> </edge> ... </boardlines>   — board outline edges; an arc outline entry appears as <arc id="..."> with an extra <mid .../> point on the arc between p1 and p2 (full circle: p1 == p2, mid = opposite point)
  <nets> <net code=K name="..."> <pads> <pad id="..." x=X y=Y layer=L /> ... </pads> </net> ... </nets>   — nets with pad center positions; layer is a copper layer number (1=Top, N=Bottom) or "th" for a through-hole pad (plated through every copper layer — start_route / make_via may land on any layer)
  <obstacles> <obs id="..." x=X y=Y size="W H" layer=L /> ... </obstacles>   — static obstacles

## routing_geometry — dynamic per-net routing state (updated every step)
  <net code=K>
    <tracks> <track id="..." layer=L> <p1 id="..." x=X y=Y /> <p2 id="..." x=X y=Y /> </track> ... </tracks>   — placed track segments
    <vias>   <via id="..." layers="start end"> <center id="..." x=X y=Y /> </via> ... </vias>                   — placed vias
    <points> <point id="..." x=X y=Y /> ... </points>                                                          — unconnected ratsnest targets; layer info not provided (infer from the target pad or nearby tracks)
  </net>

## router_head — current router cursor state (self-closing tag with these attributes)
  xy="x y"             — cursor position in mm
  layer=N              — active copper layer (1=Top)
  net=N                — currently selected net code (0=none)
  phase=N              — 0=NET_SELECT, 1=START_ROUTE, 2=ROUTING
  is_routing=true|false
  routing_mode="M"     — m=MarkObstacles, p=PushAndShove, w=Walkaround
  step="N ratio"       — current step number and step/max_steps ratio"""

_STATE_FORMAT_DESC_SEXPR = """\
The board state is provided in S-expression format. All floating-point values are rounded to 3 decimal places. Each element is enclosed in parentheses: (tag value ...) or (tag (child ...) ...).

## board_static — static board context (unchanged during episode)
  (bbox x y w h)          — board bounding box origin and size in mm
  (scale N)               — max(w, h), for normalization reference
  (net_count N)           — total number of nets
  (copper_layers N)       — number of copper layers
  (boardlines (edge ID (p1 PID x y) (p2 PID x y)) ...)  — board outline edges; an arc outline entry appears as (arc ID (p1 ...) (mid PID x y) (p2 ...)) with mid on the arc between p1 and p2 (full circle: p1 == p2, mid = opposite point)
  (nets (net CODE "NAME" (pads (pad ID x y layer) ...)) ...)   — nets with pad center positions; layer is a copper layer number (1=Top, N=Bottom) or `th` for a through-hole pad (plated through every copper layer — start_route / make_via may land on any layer)
  (obstacles (obs ID x y (size w h) (layer N)) ...)      — static obstacles

## routing_geometry — dynamic per-net routing state (updated every step)
  (net CODE
    (tracks (track ID (p1 PID x y) (p2 PID x y) (layer N)) ...)  — placed track segments
    (vias (via ID (center PID x y) (layers start end)) ...)       — placed vias
    (points (point ID x y) ...))                                  — unconnected ratsnest targets; layer info not provided (infer from the target pad or nearby tracks)

## router_head — current router cursor state
  (xy x y)             — cursor position in mm
  (layer N)            — active copper layer (1=Top)
  (net N)              — currently selected net code (0=none)
  (phase N)            — 0=NET_SELECT, 1=START_ROUTE, 2=ROUTING
  (is_routing true|false)
  (routing_mode M)     — m=MarkObstacles, p=PushAndShove, w=Walkaround
  (step N ratio)       — current step number and step/max_steps ratio"""

_STATE_FORMAT_DESCS = {
    "xml":   _STATE_FORMAT_DESC_XML,
    "sexpr": _STATE_FORMAT_DESC_SEXPR,
}


def get_state_format_desc(state_format: str = "sexpr") -> str:
    """Return the board state format description for the given format."""
    return _STATE_FORMAT_DESCS[state_format]


# ======================================================================
# Shared prompt blocks (reused across all templates)
# ======================================================================

_ROLE_AND_PRIORITY = """\
You are a PCB routing agent for KiCad. Connect unconnected pins (points in routing_geometry) within each net. Never connect pins from different nets.

# Priority
1. Complete all connections (reduce unconnected points to zero).
2. Avoid DRC violations (no crossing other nets, respect clearance).
3. Minimize total wirelength."""

_ACTION_REFERENCE = """\
# Action Reference

## Actions

1. net_select <net_id>
   Focus routing on this net. All subsequent actions (start_route, make_line,
   make_via, finish) apply to this net until another net_select is issued or
   net_end finalizes it.

2. start_route <x> <y> <layer>
   Begin routing from a pad of the selected net at (x, y) on the given layer.
   - x, y must match a pad position listed under the selected net.
   - layer: 1=Top (F.Cu), 2=Bottom (B.Cu).

3. make_line <x> <y> <mode>
   Draw a straight track segment from the current cursor position to (x, y).

4. make_via <x> <y> <mode>
   Route to (x, y), then place a via to switch layers (Top <-> Bottom).
   After the via the cursor moves to the opposite layer.
   Never place a via directly on a pad — offset at least 1mm from the pad first.

5. finish <mode>
   Auto-complete the route to the nearest unconnected pad of the current net.
   Use when the cursor is close to the target pad and the path is clear.

6. net_end
   Finalize the current net and return to the net-select phase.
   Only succeeds when all (point ...) entries of the current net are connected.

## Routing Modes (for make_line / make_via / finish)

- m = MarkObstacles — Fastest. Treats existing tracks as obstacles without
                      routing around them. Fails on collision.
- p = PushAndShove  — Pushes existing tracks of other nets aside when geometry
                      permits. Useful in dense areas.
- w = Walkaround    — Routes around existing tracks without modifying them.
                      Safest and most common choice."""

_ACTION_MASKING_RULES = """\
# Action Masking Rules

The environment exposes only the actions whose preconditions match the current
router state. Always pick an action from the "Valid Actions" list.

  net_select:   has_net = false AND is_routing = false            (phase = 0)
  start_route:  has_net = true  AND is_routing = false            (phase = 1)
  net_end:      has_net = true  AND is_routing = false
                AND all (point ...) entries of this net are gone  (phase = 1)
  make_line:    is_routing = true                                 (phase = 2)
  make_via:     is_routing = true                                 (phase = 2)
  finish:       is_routing = true                                 (phase = 2)

Flags you can read from router_head:
  - has_net:      router_head.net != 0
  - is_routing:   router_head.is_routing = true
  - fully routed: no (point ...) remaining under the active net in routing_geometry"""

_ROUTING_GUIDELINES = """\
# Routing Guidelines
- Route toward (point ...) entries in routing_geometry — these are your unconnected pad targets.
- Avoid static obstacles.
- Avoid existing tracks on the same layer in other nets in routing_geometry.
- When a path is blocked, detour through an intermediate waypoint or switch layers to bypass the obstruction.
- Never place a via directly on a pad; move away from the pad before switching layers.
- In walkaround mode, route to a waypoint first, then to the target pad — never route straight into an obstructed pad.
- Always use exact coordinates from the observation (pad positions, point targets, track endpoints). Only compute coordinates for intermediate waypoints to avoid obstacles."""

# ----------------------------------------------------------------------
# Few-Shot Example — Via Crossing on simple_routing_board (50x30mm)
#   NET1: PA1(10,10) -> PB1(40,10), already routed horizontally on layer 1
#   NET2: PC2(10,20) -> PD2(40,20), not routed
#   NET3: PE3(25,5)  -> PF3(25,25), to be routed; must cross NET1
#
# Each step shows:
#   - Observation (routing_geometry + router_head) in sexpr format
#   - Valid Actions (identical to format_valid_actions() output)
#   - Agent response (<think>...</think><action>...</action>)
# board_static is shown once at step 0, "unchanged" thereafter.
# ----------------------------------------------------------------------

_FEW_SHOT_EXAMPLE = """\
# Few-Shot Example: Via Crossing

Scenario: a 50x30mm board with 3 nets. NET1 is already routed as a horizontal
track on layer 1 (y=10). NET3 (PE3 at (25,5) to PF3 at (25,25)) must route
vertically across NET1. The agent uses two vias to detour through layer 2.

board_static (shown once — unchanged throughout the episode):
(board_static
  (bbox 0.000 0.000 50.000 30.000)
  (scale 50.000)
  (net_count 3)
  (copper_layers 2)
  (nets
    (net 1 "NET1"
      (pads (pad D0 10.000 10.000 1) (pad D1 40.000 10.000 1)))
    (net 2 "NET2"
      (pads (pad D2 10.000 20.000 1) (pad D3 40.000 20.000 1)))
    (net 3 "NET3"
      (pads (pad D4 25.000 5.000 1) (pad D5 25.000 25.000 1))))
  (obstacles))

--- Step 0 -----------------------------------------------------------------
Observation:
(routing_geometry
  (net 1
    (tracks (track T0 (p1 P0 10.000 10.000) (p2 P1 40.000 10.000) (layer 1)))
    (vias)
    (points))
  (net 2
    (tracks)
    (vias)
    (points (point Q0 10.000 20.000) (point Q1 40.000 20.000)))
  (net 3
    (tracks)
    (vias)
    (points (point Q2 25.000 5.000) (point Q3 25.000 25.000))))
(router_head
  (xy 0.000 0.000)
  (layer 1)
  (net 0)
  (phase 0)
  (is_routing false)
  (routing_mode w)
  (step 0 0.000))

Valid Actions:
net_select <net_id> — Select a net to route.  e.g. net_select 3

Response:
<think>NET3 has two unrouted points at (25,5) and (25,25). NET1's track lies at y=10 on layer 1, blocking NET3's vertical path. I will focus on NET3 first.</think>
<action>net_select 3</action>

--- Step 1 -----------------------------------------------------------------
Observation:
(routing_geometry (unchanged))
(router_head
  (xy 0.000 0.000)
  (layer 1)
  (net 3)
  (phase 1)
  (is_routing false)
  (routing_mode w)
  (step 1 0.010))

Valid Actions:
start_route <x> <y> <layer> — Begin routing from pad at (x, y) on the given layer.  e.g. start_route 25.0 40.0 0

Response:
<think>NET3 is selected; now begin routing from PE3 at (25.0, 5.0) on layer 1.</think>
<action>start_route 25.0 5.0 1</action>

--- Step 2 -----------------------------------------------------------------
Observation:
(routing_geometry
  (net 1 (tracks (track T0 ... )) (vias) (points))
  (net 2 (tracks) (vias) (points (point Q0 10.000 20.000) (point Q1 40.000 20.000)))
  (net 3
    (tracks)
    (vias)
    (points (point Q3 25.000 25.000))))
(router_head
  (xy 25.000 5.000)
  (layer 1)
  (net 3)
  (phase 2)
  (is_routing true)
  (routing_mode w)
  (step 2 0.020))

Valid Actions:
make_line <x> <y> <mode> — Draw a track from the current position to (x, y).  e.g. make_line 30.0 45.0 w
make_via <x> <y> <mode> — Draw a track to (x, y), then place a via to switch layers.  e.g. make_via 30.0 45.0 w
finish <mode> — Auto-complete the route to the nearest unconnected pad.  e.g. finish w
mode: m=MarkObstacles, p=PushAndShove, w=Walkaround

Response:
<think>Cursor is at (25,5) on layer 1. NET1's track at y=10 blocks direct downward movement, so I'll route down to (25,8) and then drop a via to switch to layer 2 before hitting NET1.</think>
<action>make_via 25.0 8.0 w</action>

--- Step 3 -----------------------------------------------------------------
Observation:
(routing_geometry
  (net 1 (tracks (track T0 ... )) (vias) (points))
  (net 2 (tracks) (vias) (points (point Q0 10.000 20.000) (point Q1 40.000 20.000)))
  (net 3
    (tracks (track T1 (p1 P2 25.000 5.000) (p2 P3 25.000 8.000) (layer 1)))
    (vias (via V0 (center P3 25.000 8.000) (layers 1 2)))
    (points (point Q3 25.000 25.000))))
(router_head
  (xy 25.000 8.000)
  (layer 2)
  (net 3)
  (phase 2)
  (is_routing true)
  (routing_mode w)
  (step 3 0.030))

Valid Actions:
make_line <x> <y> <mode> — Draw a track from the current position to (x, y).  e.g. make_line 30.0 45.0 w
make_via <x> <y> <mode> — Draw a track to (x, y), then place a via to switch layers.  e.g. make_via 30.0 45.0 w
finish <mode> — Auto-complete the route to the nearest unconnected pad.  e.g. finish w
mode: m=MarkObstacles, p=PushAndShove, w=Walkaround

Response:
<think>I'm now on layer 2 at (25,8). I can route across the NET1 crossing zone safely on layer 2. Drawing a straight segment to (25,12) clears the crossing.</think>
<action>make_line 25.0 12.0 w</action>

--- Step 4 -----------------------------------------------------------------
Observation:
(routing_geometry
  (net 1 (tracks (track T0 ... )) (vias) (points))
  (net 2 (tracks) (vias) (points (point Q0 10.000 20.000) (point Q1 40.000 20.000)))
  (net 3
    (tracks
      (track T1 (p1 P2 25.000 5.000) (p2 P3 25.000 8.000) (layer 1))
      (track T2 (p1 P3 25.000 8.000) (p2 P4 25.000 12.000) (layer 2)))
    (vias (via V0 (center P3 25.000 8.000) (layers 1 2)))
    (points (point Q3 25.000 25.000))))
(router_head
  (xy 25.000 12.000)
  (layer 2)
  (net 3)
  (phase 2)
  (is_routing true)
  (routing_mode w)
  (step 4 0.040))

Valid Actions:
make_line <x> <y> <mode> — Draw a track from the current position to (x, y).  e.g. make_line 30.0 45.0 w
make_via <x> <y> <mode> — Draw a track to (x, y), then place a via to switch layers.  e.g. make_via 30.0 45.0 w
finish <mode> — Auto-complete the route to the nearest unconnected pad.  e.g. finish w
mode: m=MarkObstacles, p=PushAndShove, w=Walkaround

Response:
<think>I'm past NET1's crossing zone at (25,12) on layer 2. The rest of the path to PF3 at (25,25) is clear on layer 1, so place a via here to return to layer 1.</think>
<action>make_via 25.0 12.0 w</action>

--- Step 5 -----------------------------------------------------------------
Observation:
(routing_geometry
  (net 1 (tracks (track T0 ... )) (vias) (points))
  (net 2 (tracks) (vias) (points (point Q0 10.000 20.000) (point Q1 40.000 20.000)))
  (net 3
    (tracks
      (track T1 (p1 P2 25.000 5.000) (p2 P3 25.000 8.000) (layer 1))
      (track T2 (p1 P3 25.000 8.000) (p2 P4 25.000 12.000) (layer 2)))
    (vias
      (via V0 (center P3 25.000 8.000) (layers 1 2))
      (via V1 (center P4 25.000 12.000) (layers 1 2)))
    (points (point Q3 25.000 25.000))))
(router_head
  (xy 25.000 12.000)
  (layer 1)
  (net 3)
  (phase 2)
  (is_routing true)
  (routing_mode w)
  (step 5 0.050))

Valid Actions:
make_line <x> <y> <mode> — Draw a track from the current position to (x, y).  e.g. make_line 30.0 45.0 w
make_via <x> <y> <mode> — Draw a track to (x, y), then place a via to switch layers.  e.g. make_via 30.0 45.0 w
finish <mode> — Auto-complete the route to the nearest unconnected pad.  e.g. finish w
mode: m=MarkObstacles, p=PushAndShove, w=Walkaround

Response:
<think>Back on layer 1 at (25,12) with a clear path to the only remaining point at (25,25). Auto-complete with finish.</think>
<action>finish w</action>

--- Step 6 -----------------------------------------------------------------
Observation:
(routing_geometry
  (net 1 (tracks (track T0 ... )) (vias) (points))
  (net 2 (tracks) (vias) (points (point Q0 10.000 20.000) (point Q1 40.000 20.000)))
  (net 3
    (tracks
      (track T1 (p1 P2 25.000 5.000) (p2 P3 25.000 8.000) (layer 1))
      (track T2 (p1 P3 25.000 8.000) (p2 P4 25.000 12.000) (layer 2))
      (track T3 (p1 P4 25.000 12.000) (p2 P5 25.000 25.000) (layer 1)))
    (vias
      (via V0 (center P3 25.000 8.000) (layers 1 2))
      (via V1 (center P4 25.000 12.000) (layers 1 2)))
    (points)))
(router_head
  (xy 25.000 25.000)
  (layer 1)
  (net 3)
  (phase 1)
  (is_routing false)
  (routing_mode w)
  (step 6 0.060))

Valid Actions:
start_route <x> <y> <layer> — Begin routing from pad at (x, y) on the given layer.  e.g. start_route 25.0 40.0 0
net_end — Mark the current net as fully routed.  e.g. net_end

Response:
<think>NET3 now has zero (point ...) entries, meaning it is fully connected. Finalize with net_end.</think>
<action>net_end</action>

--- End of Example ---------------------------------------------------------
After net_end, NET3 is finalized. The next decision loop would return to phase
0 (net_select) for the remaining nets (NET2), or end the episode if all nets
are complete."""

_RESPONSE_FORMAT = """\
# Response Format
Output exactly one <think>...</think> followed by one <action>...</action>. Never repeat think-action pairs."""

_RESPONSE_FORMAT_HIS = """\
# Response Format
Output exactly one <think>...</think> followed by one <action>...</action> with a valid action. Never repeat think-action pairs."""


# ======================================================================
# Single-message templates (all content in one user message)
# ======================================================================

CADAGENT_TEMPLATE_NO_HIS = f"""
{_ROLE_AND_PRIORITY}

{_ACTION_REFERENCE}

{_ACTION_MASKING_RULES}

{_ROUTING_GUIDELINES}

{_FEW_SHOT_EXAMPLE}

# Board State
{{state_format_desc}}

{{current_observation}}

# Valid Actions
Choose exactly one from:
{{valid_actions}}

{_RESPONSE_FORMAT}
"""

# ======================================================================
# Split templates (system + user messages for multi-turn extension)
# ======================================================================

CADAGENT_SYSTEM_PROMPT = f"""\
{_ROLE_AND_PRIORITY}

{_ACTION_REFERENCE}

{_ACTION_MASKING_RULES}

{_ROUTING_GUIDELINES}

{_FEW_SHOT_EXAMPLE}

{_RESPONSE_FORMAT}

# Board State
{{state_format_desc}}

{{board_static}}"""

CADAGENT_USER_PROMPT_NO_HIS = """\
{dynamic_observation}

# Valid Actions
Choose exactly one from:
{valid_actions}"""

# NOTE: wschoi-branch history format uses {current_step} + {valid_step_count}
# (not V2-upstream's {step_count} + {history_length}). Matches the .format()
# kwargs in rollout/cadagent.py and agent_system/environments/env_manager.py.
CADAGENT_USER_PROMPT = """\
## Step {current_step}
{dynamic_observation}

# History (step {current_step}, {valid_step_count} valid actions taken)
{action_history}

# Valid Actions
Choose exactly one from:
{valid_actions}"""

CADAGENT_TEMPLATE = f"""
{_ROLE_AND_PRIORITY}

{_ACTION_REFERENCE}

{_ACTION_MASKING_RULES}

{_ROUTING_GUIDELINES}

{_FEW_SHOT_EXAMPLE}

# Board State
{{state_format_desc}}

## Step {{current_step}}
{{current_observation}}

# History (step {{current_step}}, {{valid_step_count}} valid actions taken)
{{action_history}}

# Valid Actions
Choose exactly one from:
{{valid_actions}}

{_RESPONSE_FORMAT_HIS}
"""
