"""Plan-only (API-Seq) via LLM (no-train): open-loop CAD-API action sequence + replay.

Standalone: state extraction, API call, action parser, env replay, per-board
metrics, multi-k aggregator, together / openai providers and ``--reaggregate``
all live in this file, with the system prompt inlined into the source.

Sister script to ``eval_engine_free_llm_v3_standalone.py`` (CAD-Gen). Where
CAD-Gen asks the model for a finished board (raw segments + vias), API-Seq
asks the model for the *sequence of CAD API calls* that would produce one —
single shot, without showing intermediate routing state, masks, or candidates.

The LLM sees only:
    1. the initial board (static section: footprints, pads, nets),
    2. the 6-action API description, and
    3. (optionally) one or more (initial_state, action_sequence) examples.

Then we parse its action list, replay it through ``PCBWorld``, and
score the final routed board with the same metric definitions as
the CAD-Gen script (success / routability / DRV / wirelength).

Per-board protocol:
    1. construct ``PCBWorld``, reset → render the *initial* board state
       text (board_static portion only).
    2. build prompt (zero_shot or few_shot with action-sequence examples).
    3. sample N completions (default 5).
    4. for each:
        a. extract the ``<actions>...</actions>`` block,
        b. parse one action per line,
        c. replay actions through a fresh env episode,
        d. capture {success, routability, drv_count, ...} from the
           final reward snapshot,
        e. save the rolled-out PCB to ``sample_NN.kicad_pcb``.
    5. aggregate over N samples (pass@N, routability@N best/mean) — same
       definitions as the CAD-Gen script.

Outputs land in --output:
    output/per_board/<board_id>/sample_<i>.{kicad_pcb,kicad_pro,json,txt}
    output/per_board/<board_id>/aggregate.json
    output/summary.csv
    output/overall.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


_THIS_DIR = Path(__file__).resolve().parent.parent.parent  # llm_eval→kdd→experiments
_KICAD_RL_DIR = _THIS_DIR / "build_rl" / "pcbnew" / "python" / "rl"
for p in (_THIS_DIR, _KICAD_RL_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert PCB routing engineer driving a KiCad routing API.

You will be given the *initial* state of a PCB board (footprints, pads, nets) and must output the **complete sequence of routing API calls** that would route every net.

## API
The router exposes 6 high-level actions. Emit one per line.
    net_select <net_id>                      Select a net to start routing.
    start_route <x_mm> <y_mm> <layer>        Begin routing at a pad position.
                                             layer: 1 = F.Cu (top), 2 = B.Cu (bottom).
    make_line <x_mm> <y_mm> <mode>           Extend a track to (x,y) on the current layer.
    make_via <x_mm> <y_mm> <mode>            Extend track to (x,y), drop a via, switch layer.
                                             AFTER make_via, the current layer flips.
    finish <mode>                            Auto-complete the active route on the *CURRENT*
                                             layer to the nearest pending pad of the
                                             current net. CANNOT cross layers — if the
                                             remaining pad is on a different layer, finish
                                             will NOT reach it.
    net_end                                  Mark the active net as fully routed.
    mode is one of: m (MarkObstacles), p (PushAndShove), w (Walkaround). Use w by default.

## Reading <BOARD>
The input is KiCad-style sexpr. Inside `(nets ...)` each net lists its pads:
    (pad <id> <x> <y> <layer_tag>)
Layer tags:
    `1`  -> the pad lives on layer 1 (F.Cu, top).
    `2`  -> the pad lives on layer 2 (B.Cu, bottom).
    `th` -> through-hole, electrically present on BOTH layers; you
           may treat it as either 1 or 2 when choosing start_route's
           layer parameter or as the via-side of a make_via.

## Coordinates and layers (MUST follow exactly)
- Copy each pad's `<x>` and `<y>` token **verbatim** — same digits, same decimal places, no rounding. The router only accepts a pad's *exact* (x_mm, y_mm). Example: if the board has
    (pad D0 67.500 44.700 th)
  then output
    start_route 67.500 44.700 1
  not `start_route 67.5 44.7 1`.
- start_route MUST be issued at a pad position of the currently selected net.

## Layer-choice rule for the FIRST pad
You pick a `<layer>` argument for `start_route`. The choice determines which layer the router head sits on, which constrains every subsequent action.

Pick the start layer using THIS priority:
  1. If the first pad has tag `1` -> use 1.
  2. If the first pad has tag `2` -> use 2.
  3. If the first pad has tag `th` and at least one other pad of
     this net has tag `1` or `2` -> use that other pad's layer.
     (This avoids an unnecessary via.)
  4. If the first pad has tag `th` and all other pads also `th`
     -> use 1.

## Routing protocol (state machine — violations cause the entire net to fail)
After `net_select`, the router is in {has_net=True, is_routing=False}.
After a SUCCESSFUL `start_route`, it moves to {has_net=True, is_routing=True}.
`make_line`, `make_via`, `finish` are ONLY valid while is_routing=True.
`net_end` closes the net and returns to {has_net=False}.
Emitting any of make_line / make_via / finish / net_end before a
successful start_route causes the entire net to fail.

## Per-topology emission rules (CASE BY CASE — match the net's shape)
Resolve each pad's *effective* layer using the layer-choice rule (`th` resolves to either 1 or 2 by neighbour). Then dispatch:

### Case A — 0 or 1 pads : skip the net entirely.

### Case B — 2 pads on the SAME effective layer L
  net_select <id>
  start_route <P1.x> <P1.y> L
  make_line  <P2.x> <P2.y> <mode>   ; recommended over finish
  net_end

  (alternative: `finish <mode>` also works — auto-completes head to P2 on L)

### Case C — 2 pads on DIFFERENT layers (cross-layer)
  `finish` cannot cross layers. Pick an intermediate point P1' and go
  P1 -> P1' -> P2, using make_via at P1' to switch layers:
  net_select <id>
  start_route <P1.x> <P1.y> L1        ; L1 = P1's layer
  make_via   <P1'.x> <P1'.y> <mode>   ; switch L1 -> L2 at P1'
  start_route <P1'.x> <P1'.y> L2      ; L2 = P2's layer, restart at P1'
  make_line  <P2.x> <P2.y> <mode>     ; (or `finish <mode>`)
  net_end

### Case D — k pads (k >= 3) all on the SAME effective layer L
  net_select <id>
  start_route <P1.x> <P1.y> L
  make_line  <P2.x> <P2.y> <mode>
  start_route <P2.x> <P2.y> L
  make_line  <P3.x> <P3.y> <mode>
  ... up to P(k-1) ...
  make_line  <P(k).x> <P(k).y> <mode>   ; (or `finish <mode>`)
  net_end

## Anti-patterns (do NOT do these)
  ❌ `finish` immediately after start_route when the second pad is
     on a different layer. finish will route within the current
     layer and silently miss the cross-layer pad.
     ✓ Use make_via to land on the cross-layer pad (Case C).
  ❌ Picking a `<layer>` for start_route that no pad of this net
     lives on (e.g. layer 2 when both pads are on 1).
     ✓ Apply the layer-choice rule.
  ❌ Reformatting coordinates (5.7 instead of 5.700, dropping
     trailing zeros). The board's pad table has the exact form;
     copy it.
  ❌ Emitting `make_line <Pk.x> <Pk.y>` THEN `finish` for the
     final pad of a same-layer chain. `finish` already handles Pk
     — the explicit make_line creates a duplicate / overlapping
     segment and may break DRC.
  ❌ Forgetting `net_end` between nets. The next `net_select`
     fails if the previous net wasn't closed.
  ❌ Calling start_route at coordinates that are not a pad of the
     currently selected net (e.g. picking a pad of a different
     net by accident). The route fails silently.

## Output Format
Wrap the entire sequence in <actions>...</actions>. One action per line. No commentary inside the block. Do NOT repeat the board.
"""


_FEWSHOT_HEADER = "\n---\n\n## Examples\n"
_FEWSHOT_EXAMPLE_TPL = (
    "\n### Example {idx}\n"
    "Initial state:\n<BOARD>\n{board}\n</BOARD>\n\n"
    "Output:\n<actions>\n{actions}\n</actions>\n"
)
_TASK_TPL = (
    "\n---\n\n## Task\n\n"
    "Now produce the routing API sequence for the following board:\n\n"
    "<BOARD>\n{target}\n</BOARD>\n\n"
    "Output:\n<actions>\n"
)


# ---------------------------------------------------------------------------
# Action parser — line text → action dict (PCBWorld.step compatible)
# ---------------------------------------------------------------------------

from pcb_world.core.action_schema import MODE_LETTER_TO_INT as _MODE_LETTER_TO_INT

_ACTION_TYPE_MAP = {
    "net_select":  0,
    "start_route": 1,
    "net_end":     2,
    "make_line":   3,
    "make_via":    4,
    "finish":      5,
}

# (param_name, type) pairs used to flesh out a parsed action.
_ACTION_SCHEMA: dict[str, list[tuple[str, type]]] = {
    "net_select":  [("net_id", int)],
    "start_route": [("x_mm", float), ("y_mm", float), ("layer", int)],
    "net_end":     [],
    "make_line":   [("x_mm", float), ("y_mm", float), ("routing_mode", str)],
    "make_via":    [("x_mm", float), ("y_mm", float), ("routing_mode", str)],
    "finish":      [("routing_mode", str)],
}

_PARAM_DEFAULTS = {
    "x_mm":         0.0,
    "y_mm":         0.0,
    "layer":        1,
    "net_id":       0,
    "routing_mode": 2,    # Walkaround
}


def parse_action_line(line: str) -> dict | None:
    """Return a step-ready action dict or ``None`` if the line is unparseable.

    Robust to leading/trailing whitespace, comments after ``#``, and the
    occasional ``<action>...</action>`` envelope. The first token is the
    action name; subsequent tokens are positional parameters in
    ``_ACTION_SCHEMA`` order.
    """
    s = line.strip()
    if not s or s.startswith("#") or s.startswith(";"):
        return None
    # Strip a single <action>...</action> wrap if present.
    m = re.match(r"<action>\s*(.*?)\s*</action>", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    tokens = s.split()
    if not tokens:
        return None
    name = tokens[0]
    if name not in _ACTION_SCHEMA:
        return None
    schema = _ACTION_SCHEMA[name]
    params: dict = {}
    values = tokens[1:]
    for i, (pname, conv) in enumerate(schema):
        if i < len(values):
            v = values[i]
            try:
                if pname == "routing_mode":
                    params[pname] = _MODE_LETTER_TO_INT.get(v, _PARAM_DEFAULTS[pname])
                    # Allow numeric mode strings too.
                    if v not in _MODE_LETTER_TO_INT:
                        try:
                            params[pname] = int(v)
                        except ValueError:
                            params[pname] = _PARAM_DEFAULTS[pname]
                elif conv is int:
                    params[pname] = int(float(v))
                else:
                    params[pname] = conv(v)
            except (ValueError, TypeError):
                params[pname] = _PARAM_DEFAULTS[pname]
        else:
            params[pname] = _PARAM_DEFAULTS[pname]
    return {"action_type": _ACTION_TYPE_MAP[name], **params}


_ACTIONS_TAG_RE = re.compile(r"<actions>\s*(.*?)\s*(?:</actions>|$)", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)\n```", re.DOTALL)


def extract_action_sequence(response: str) -> list[dict]:
    """Pull the action sequence out of an LLM response.

    Tolerates: tagged ``<actions>...</actions>``, missing close tag,
    markdown fences, or just a newline-separated list with stray prose.
    Lines that don't parse as actions are silently dropped.
    """
    m = _ACTIONS_TAG_RE.search(response)
    if m:
        body = m.group(1)
    else:
        m = _FENCE_RE.search(response)
        body = m.group(1) if m else response
    out: list[dict] = []
    for line in body.splitlines():
        a = parse_action_line(line)
        if a is not None:
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Initial-state extraction
# ---------------------------------------------------------------------------

def _build_env(board_path: Path, masking_rule: str = "default", state_format: str = "sexpr"):
    """Construct a fresh PCBWorld. Caller must close it."""
    from pcb_world.core.env import PCBWorld
    from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH
    return PCBWorld(
        board_path=str(board_path),
        masking_rule=masking_rule,
        state_format=state_format,
        # Generous step budget — sequence may have many actions.
        max_steps=10_000,
        # Synth boards' .kicad_pro lacks legacy setup tokens, so the
        # engine can't seed DRC from them — fall back to configs/drc/default.yaml
        # (passed explicitly; there is no implicit default) rather than
        # refusing to construct.
        use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
    )


def initial_board_state_text(env, state_format: str = "sexpr") -> str:
    """Return only the static portion of the initial observation.

    Resets the env, formats the (post-reset) observation dict as
    sexpr/xml text, and returns the *board_static* section. The
    dynamic routing-geometry / router-head sections are dropped so
    the LLM sees only the initial board (footprints, pads, nets) —
    the API-Seq protocol's intentional information bottleneck.
    """
    from methods.llm_agent.wrappers.state_converter import (
        format_state_split_sexpr, format_state_split,
    )
    obs, _info = env.reset()
    if state_format == "sexpr":
        static, _ = format_state_split_sexpr(obs)
    else:
        static, _ = format_state_split(obs)
    return static


# ---------------------------------------------------------------------------
# Few-shot example pool — load (board_static, action_sequence) JSONs harvested
# by ``experiments/kdd/llm_eval/prepare_plan_only_fewshot.py``.
# ---------------------------------------------------------------------------

@dataclass
class FewShotExample:
    board_id: str
    board_static: str
    action_sequence: str   # newline-separated text


def load_fewshot_pool(
    pool_paths: list[Path],
    k: int,
    max_chars: int | None = 20000,
) -> list[FewShotExample]:
    """Load up to ``k`` examples from cache JSONs.

    Each JSON file under ``pool_paths`` should have keys
    ``board_static`` (str), ``action_sequence`` (str or list[str]),
    and (optional) ``board_id``.
    """
    out: list[FewShotExample] = []
    for root in pool_paths:
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(root.glob("*.json"))
        else:
            continue
        for f in files:
            if len(out) >= k:
                break
            try:
                blob = json.loads(f.read_text())
            except Exception:
                continue
            board_static = blob.get("board_static", "")
            actions = blob.get("action_sequence", "")
            if isinstance(actions, list):
                actions = "\n".join(actions)
            if not (board_static and actions):
                continue
            if max_chars is not None and len(board_static) > max_chars:
                continue
            out.append(FewShotExample(
                board_id=blob.get("board_id", f.stem),
                board_static=board_static,
                action_sequence=actions,
            ))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(target_board_static: str, examples: list[FewShotExample]) -> str:
    parts = []
    if examples:
        parts.append(_FEWSHOT_HEADER)
        for idx, ex in enumerate(examples, start=1):
            parts.append(_FEWSHOT_EXAMPLE_TPL.format(
                idx=idx,
                board=ex.board_static.strip(),
                actions=ex.action_sequence.strip(),
            ))
    parts.append(_TASK_TPL.format(target=target_board_static.strip()))
    return "".join(parts)


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None = None,
    n: int = 1,
) -> tuple[list[str], dict]:
    from openai import OpenAI
    client = OpenAI(**({"api_key": api_key} if api_key else {}))

    is_reasoning = any(model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5"))
    if is_reasoning:
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"max_completion_tokens": max_tokens}
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"max_tokens": max_tokens, "temperature": temperature}

    resp = client.chat.completions.create(
        model=model, messages=messages, n=n, **kwargs,
    )
    texts = [c.message.content or "" for c in resp.choices]
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        "total_tokens": getattr(resp.usage, "total_tokens", 0),
    }
    return texts, usage


def call_together(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None = None,
    n: int = 1,
    base_url: str = "https://api.together.xyz/v1",
    concurrency: int = 4,
    enable_thinking: str = "auto",
) -> tuple[list[str], dict]:
    """Together AI completions via the OpenAI-compatible endpoint.

    Together's serverless deployments don't reliably honor ``n>1`` on a single
    chat-completion request, so we issue ``n`` independent calls (in a thread
    pool) and aggregate.

    ``enable_thinking``: ``"auto"`` (default — server-side default kept,
    typically thinking ON for Qwen3), ``"on"`` (force-enable), or ``"off"``
    (force-disable). The off path adds ``chat_template_kwargs.enable_thinking
    = False`` via ``extra_body`` — supported by Together's vLLM backend for
    Qwen3 family models. Non-Qwen models silently ignore the kwarg.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=(api_key or os.environ.get("TOGETHER_API_KEY") or "EMPTY"),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    extra_body: dict = {}
    if enable_thinking in ("on", "off"):
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": (enable_thinking == "on"),
        }

    def _one() -> tuple[str, dict]:
        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        c = resp.choices[0]
        u = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        }
        return (c.message.content or ""), u

    texts: list[str] = [""] * n
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if n == 1 or concurrency <= 1:
        for i in range(n):
            t, u = _one()
            texts[i] = t
            for k_ in usage:
                usage[k_] += u[k_]
    else:
        with ThreadPoolExecutor(max_workers=min(concurrency, n)) as ex:
            futs = {ex.submit(_one): i for i in range(n)}
            for fut in as_completed(futs):
                i = futs[fut]
                t, u = fut.result()
                texts[i] = t
                for k_ in usage:
                    usage[k_] += u[k_]
    return texts, usage


def call_llm(provider: str, **kwargs) -> tuple[list[str], dict]:
    """Dispatch to the right provider client."""
    if provider == "openai":
        # OpenAI proper has no `enable_thinking` knob; gpt-5/o-series
        # reasoning is server-side and not togglable from the API.
        for k in ("base_url", "concurrency", "enable_thinking"):
            kwargs.pop(k, None)
        return call_openai(**kwargs)
    if provider == "together":
        return call_together(**kwargs)
    raise ValueError(f"unknown api provider: {provider!r}")


# ---------------------------------------------------------------------------
# Replay one action sequence, capture metrics
# ---------------------------------------------------------------------------

def replay_actions_and_eval(
    board_path: Path,
    actions: list[dict],
    save_pcb_path: Path | None,
) -> dict:
    """Roll a fresh env, replay ``actions``, score the final state.

    The returned metric dict uses the same field names as the CAD-Gen
    script's per-sample JSONs, so the two are directly comparable.
    """
    env = _build_env(board_path)
    try:
        env.reset()
        u_0 = env._initial_unconnected
        steps_run = 0
        steps_accepted = 0   # passed the mask + dispatched without error
        steps_rejected = 0   # mask-rejected or raised inside the engine
        last_done = False

        for action in actions:
            if last_done:
                break
            try:
                # env.step → (obs, reward, terminated, truncated, info)
                _, _, terminated, truncated, info = env.step(action)
                last_done = bool(terminated or truncated)
                # action_success captures the dispatcher outcome (False on
                # mask reject, parse fallback, or empty effect). Counting it
                # here gives us a "fraction of LLM actions actually applied"
                # signal that's far more diagnostic than success alone.
                if info.get("action_success", False):
                    steps_accepted += 1
                else:
                    steps_rejected += 1
            except Exception:
                # Engine refusal / mask reject is logged inside env; keep
                # replaying — a bad action shouldn't kill the rollout.
                steps_rejected += 1
            steps_run += 1

        # Final state snapshot.
        snap = env._engine.get_reward_snapshot(run_drc=True)
        u_t = snap.unrouted_count
        track_count = snap.track_count
        via_count = snap.via_count
        drv_count = snap.drc_violation_count
        wirelength_mm = snap.total_wirelength

        if save_pcb_path is not None:
            save_pcb_path.parent.mkdir(parents=True, exist_ok=True)
            env._engine.save(str(save_pcb_path))

        routability = 1.0 if u_0 == 0 else (u_0 - u_t) / u_0
        return {
            "success": bool(u_t == 0),
            "routability": float(routability),
            "track_count": int(track_count),
            "via_count": int(via_count),
            "drv_count": int(drv_count),
            "wirelength_mm": float(wirelength_mm),
            "extras": {
                "initial_unrouted_edges": int(u_0),
                "unrouted_edges_remaining": int(u_t),
                "steps_replayed": int(steps_run),
                "steps_accepted": int(steps_accepted),
                "steps_rejected": int(steps_rejected),
                "action_acceptance_rate": (
                    float(steps_accepted) / steps_run if steps_run else 0.0
                ),
                "actions_in_sequence": int(len(actions)),
                "drc_error_count": int(snap.drc_error_count),
                "drc_warning_count": int(snap.drc_warning_count),
                "drc_promoted_count": int(snap.drc_promoted_count),
            },
            "parse_ok": True,
            "error": "",
        }
    finally:
        # Singleton constraint — drop the env before the next board's env.
        try:
            env.close()
        except Exception:
            pass


def _failure_metrics(error: str) -> dict:
    return {
        "success": False,
        "routability": 0.0,
        "track_count": 0,
        "via_count": 0,
        "drv_count": 0,
        "wirelength_mm": 0.0,
        "extras": {},
        "parse_ok": False,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Per-board orchestration
# ---------------------------------------------------------------------------

@dataclass
class BoardResult:
    board_id: str
    source_path: str
    samples: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def successes(self) -> int:
        return sum(1 for s in self.samples if s.get("success"))

    def routability_values(self) -> list[float]:
        return [float(s.get("routability", 0.0)) for s in self.samples]

    def aggregate(self, k: int) -> dict:
        rb = self.routability_values()
        if not rb:
            best = 0.0
            mean = 0.0
        else:
            best = max(rb)
            mean = sum(rb) / len(rb)
        pass_at_k = int(self.successes > 0)
        mean_success_rate = (self.successes / self.num_samples) if self.num_samples else 0.0
        return {
            "board_id": self.board_id,
            "source_path": self.source_path,
            "num_samples": self.num_samples,
            "successes": self.successes,
            "k": k,
            "pass_at_k": pass_at_k,
            "success_at_k": pass_at_k,
            "mean_success_rate": mean_success_rate,
            "routability_at_k_best": best,
            "routability_at_k_mean": mean,
            "drv_at_k_min": min((s.get("drv_count", 0) for s in self.samples), default=0),
            "wirelength_at_k_best": min(
                (s.get("wirelength_mm", 0.0) for s in self.samples
                 if s.get("success")),
                default=0.0,
            ),
            "error": self.error,
        }


def discover_inputs(inputs: list[Path], recursive: bool) -> Iterator[tuple[Path, Path]]:
    for raw in inputs:
        item = raw.resolve()
        if item.is_file():
            if item.suffix == ".kicad_pcb":
                yield item, item.parent
            else:
                print(f"  [skip] not a .kicad_pcb: {item}", file=sys.stderr)
        elif item.is_dir():
            glob = item.rglob("*.kicad_pcb") if recursive else item.glob("*.kicad_pcb")
            for f in sorted(glob):
                yield f, item
        else:
            print(f"  [skip] missing: {item}", file=sys.stderr)


def dry_run_response_for(board_path: Path, masking_rule: str, state_format: str) -> str:
    """Replay the prep auto-router on the fly to produce a "self-known"
    answer — used in --dry-run to validate the parse + replay pipeline
    without paying for tokens.
    """
    # Lazy import to keep the dry-run path optional.
    from prepare_plan_only_fewshot import auto_route_actions  # type: ignore
    actions_text = auto_route_actions(board_path)
    return f"<actions>\n{actions_text}\n</actions>\n"


def run_one_board(
    src: Path,
    src_root: Path,
    args: argparse.Namespace,
    examples: list[FewShotExample],
    out_root: Path,
) -> BoardResult:
    rel = src.relative_to(src_root) if src.is_relative_to(src_root) else Path(src.name)
    board_id = rel.with_suffix("").as_posix().replace("/", "__")
    if "/" not in rel.as_posix() and src.parent.name:
        board_id = f"{src.parent.name}__{board_id}"
    board_dir = out_root / "per_board" / board_id
    board_dir.mkdir(parents=True, exist_ok=True)

    result = BoardResult(board_id=board_id, source_path=str(src))

    # 1. Render initial board state.
    try:
        env = _build_env(src, masking_rule=args.masking_rule, state_format=args.state_format)
        try:
            board_static = initial_board_state_text(env, args.state_format)
        finally:
            try:
                env.close()
            except Exception:
                pass
    except Exception as exc:
        result.error = f"init_failed: {type(exc).__name__}: {exc}"
        return result

    # 2. Build prompt.
    user_prompt = build_user_prompt(board_static, examples)
    (board_dir / "prompt.txt").write_text(
        "==== SYSTEM ====\n" + _SYSTEM_PROMPT + "\n==== USER ====\n" + user_prompt
    )

    # 3. Sample.
    if args.dry_run:
        try:
            response = dry_run_response_for(src, args.masking_rule, args.state_format)
            responses = [response] * args.num_samples
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception as exc:
            result.error = f"dry_run_failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()
            return result
    else:
        try:
            responses, usage = call_llm(
                provider=args.api_provider,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=args.api_model,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                api_key=args.api_key,
                n=args.num_samples,
                base_url=args.api_base_url,
                concurrency=args.api_concurrency,
                enable_thinking=args.enable_thinking,
            )
        except Exception as exc:
            result.error = f"api_failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()
            return result

    # 4. Replay each completion.
    for i, response in enumerate(responses):
        (board_dir / f"sample_{i:02d}.response.txt").write_text(response)
        actions = extract_action_sequence(response)
        sample_pcb = board_dir / f"sample_{i:02d}.kicad_pcb"
        if not actions:
            metrics = _failure_metrics(error="no_actions_parsed")
            metrics["extras"] = {"actions_in_sequence": 0}
        else:
            try:
                metrics = replay_actions_and_eval(src, actions, sample_pcb)
            except Exception as exc:
                metrics = _failure_metrics(
                    error=f"replay_failed: {type(exc).__name__}: {exc}"
                )
        metrics["sample_idx"] = i
        metrics["usage"] = usage
        with (board_dir / f"sample_{i:02d}.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        result.samples.append(metrics)

    # 5. Per-board aggregate.
    agg = result.aggregate(args.num_samples)
    with (board_dir / "aggregate.json").open("w") as f:
        json.dump(agg, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Summary + overall stats   (same output schema as the CAD-Gen script)
# ---------------------------------------------------------------------------

_SUMMARY_FIELDS = [
    "board_id",
    "source_path",
    "num_samples",
    "successes",
    "k",
    "pass_at_k",
    "mean_success_rate",
    "routability_at_k_best",
    "routability_at_k_mean",
    "drv_at_k_min",
    "wirelength_at_k_best",
    "error",
]


def write_summary(results: list[BoardResult], out_root: Path, k: int) -> Path:
    rows = [r.aggregate(k) for r in results]
    path = out_root / "summary.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k_: row.get(k_, "") for k_ in _SUMMARY_FIELDS})
    return path


def reaggregate_from_disk(out_root: Path) -> list[BoardResult]:
    per_board_root = out_root / "per_board"
    if not per_board_root.is_dir():
        raise FileNotFoundError(f"no per_board/ under {out_root}")
    results: list[BoardResult] = []
    for board_dir in sorted(per_board_root.iterdir()):
        if not board_dir.is_dir():
            continue
        board_id = board_dir.name
        sample_files = sorted(board_dir.glob("sample_*.json"))
        source_path = ""
        agg_json = board_dir / "aggregate.json"
        if agg_json.exists():
            try:
                source_path = json.loads(agg_json.read_text()).get("source_path", "")
            except Exception:
                pass
        if not sample_files:
            results.append(BoardResult(
                board_id=board_id, source_path=source_path,
                error="no_samples_found",
            ))
            continue
        samples: list[dict] = []
        for sf in sample_files:
            try:
                samples.append(json.loads(sf.read_text()))
            except Exception as exc:
                samples.append({
                    "success": False, "routability": 0.0,
                    "drv_count": 0, "wirelength_mm": 0.0,
                    "error": f"unreadable: {type(exc).__name__}: {exc}",
                })
        results.append(BoardResult(
            board_id=board_id, source_path=source_path, samples=samples,
        ))
    return results


def _per_board_at_k(samples: list[dict], k: int) -> dict:
    """Compute @k metrics from the first ``k`` samples of a board."""
    sk = samples[:k]
    if not sk:
        return {
            "pass_at_k": 0,
            "mean_success_rate": 0.0,
            "routability_at_k_best": 0.0,
            "routability_at_k_mean": 0.0,
            "drv_at_k_min": 0,
            "wirelength_at_k_best": 0.0,
        }
    successes_k = sum(1 for s in sk if s.get("success"))
    rb = [float(s.get("routability", 0.0)) for s in sk]
    return {
        "pass_at_k": int(successes_k > 0),
        "mean_success_rate": successes_k / len(sk),
        "routability_at_k_best": max(rb),
        "routability_at_k_mean": sum(rb) / len(rb),
        "drv_at_k_min": min((s.get("drv_count", 0) for s in sk), default=0),
        "wirelength_at_k_best": min(
            (s.get("wirelength_mm", 0.0) for s in sk if s.get("success")),
            default=0.0,
        ),
    }


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Codex-style unbiased pass@k from n samples with c successes."""
    if n - c < k:
        return 1.0
    if k <= 0 or n <= 0:
        return 0.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def overall_stats_multi_k(results: list[BoardResult], ks: list[int]) -> dict:
    """Per-k overall stats from a single shared sample pool per board."""
    out: dict = {"ks": ks, "per_k": {}}
    for k in ks:
        first_k_aggs: list[dict] = []
        unbiased_pk: list[float] = []
        failed = 0
        for r in results:
            if r.error or not r.samples:
                failed += 1
                continue
            n = len(r.samples)
            if n < k:
                continue
            first_k_aggs.append(_per_board_at_k(r.samples, k))
            unbiased_pk.append(_pass_at_k_unbiased(n, r.successes, k))
        if not first_k_aggs:
            out["per_k"][str(k)] = {
                "k": k, "boards_evaluated": 0, "boards_failed": failed,
            }
            continue
        pass_k = [a["pass_at_k"] for a in first_k_aggs]
        mean_succ = [a["mean_success_rate"] for a in first_k_aggs]
        best = [a["routability_at_k_best"] for a in first_k_aggs]
        mean_per_board = [a["routability_at_k_mean"] for a in first_k_aggs]
        out["per_k"][str(k)] = {
            "k": k,
            "boards_evaluated": len(first_k_aggs),
            "boards_failed": failed,
            "pass_at_k": sum(pass_k) / len(pass_k),
            "success_rate_at_k": sum(pass_k) / len(pass_k),
            "mean_success_rate": statistics.fmean(mean_succ),
            "routability_at_k_best_mean": statistics.fmean(best),
            "routability_at_k_best_std": (
                statistics.pstdev(best) if len(best) > 1 else 0.0
            ),
            "routability_at_k_mean_mean": statistics.fmean(mean_per_board),
            "routability_at_k_mean_std": (
                statistics.pstdev(mean_per_board) if len(mean_per_board) > 1 else 0.0
            ),
            "pass_at_k_unbiased": statistics.fmean(unbiased_pk),
        }
    return out


def write_multi_k_summaries(results: list[BoardResult], out_root: Path, ks: list[int]) -> None:
    for k in ks:
        rows: list[dict] = []
        for r in results:
            base = {
                "board_id": r.board_id,
                "source_path": r.source_path,
                "num_samples": r.num_samples,
                "k": k,
                "error": r.error,
            }
            if r.error or not r.samples or r.num_samples < k:
                base.update({
                    "successes": 0,
                    "pass_at_k": 0,
                    "mean_success_rate": 0.0,
                    "routability_at_k_best": 0.0,
                    "routability_at_k_mean": 0.0,
                    "drv_at_k_min": 0,
                    "wirelength_at_k_best": 0.0,
                })
            else:
                m = _per_board_at_k(r.samples, k)
                successes_k = sum(1 for s in r.samples[:k] if s.get("success"))
                base.update({"successes": successes_k, **m})
            rows.append(base)
        path = out_root / f"summary_k{k}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)


def overall_stats(results: list[BoardResult], k: int) -> dict:
    aggs = [r.aggregate(k) for r in results if not r.error]
    failed = sum(1 for r in results if r.error)
    if not aggs:
        return {"k": k, "boards_evaluated": 0, "boards_failed": failed}
    pass_k = [a["pass_at_k"] for a in aggs]
    mean_succ = [a["mean_success_rate"] for a in aggs]
    best = [a["routability_at_k_best"] for a in aggs]
    mean = [a["routability_at_k_mean"] for a in aggs]
    return {
        "k": k,
        "boards_evaluated": len(aggs),
        "boards_failed": failed,
        "pass_at_k": sum(pass_k) / len(pass_k),
        "success_rate_at_k": sum(pass_k) / len(pass_k),
        "mean_success_rate": statistics.fmean(mean_succ),
        "routability_at_k_best_mean": statistics.fmean(best),
        "routability_at_k_best_std": statistics.pstdev(best) if len(best) > 1 else 0.0,
        "routability_at_k_mean_mean": statistics.fmean(mean),
        "routability_at_k_mean_std": statistics.pstdev(mean) if len(mean) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-only API-Seq routing eval (zero-shot / few-shot).",
    )
    p.add_argument("inputs", nargs="*", type=Path, default=[],
                   help="One or more .kicad_pcb files / directories. "
                        "Not required with --reaggregate.")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    p.add_argument(
        "--fewshot-pool", type=Path, nargs="*", default=[],
        help="Directory of cached (board_static, action_sequence) JSON files "
             "produced by prepare_plan_only_fewshot.py.",
    )
    p.add_argument("--num-fewshot", type=int, default=2)
    p.add_argument("--fewshot-max-chars", type=int, default=20000)
    p.add_argument("--num-samples", "-k", type=int, default=5)
    p.add_argument(
        "--api-provider", choices=["openai", "together"], default="openai",
        help="OpenAI-compatible API provider. Together uses the same SDK at "
             "https://api.together.xyz/v1; set TOGETHER_API_KEY or pass --api-key.",
    )
    p.add_argument("--api-model", default="gpt-5.4")
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--api-base-url", default="https://api.together.xyz/v1",
        help="Base URL for --api-provider=together.",
    )
    p.add_argument(
        "--api-concurrency", type=int, default=4,
        help="Parallel requests when looping n samples on Together (default 4).",
    )
    p.add_argument(
        "--enable-thinking", choices=("auto", "on", "off"), default="auto",
        help="Qwen3-style thinking-mode toggle (Together only). "
             "'auto' = server default; 'off' adds chat_template_kwargs."
             "enable_thinking=False via extra_body so Qwen3 emits without "
             "<think>...</think> blocks. Ignored for non-Qwen models and "
             "for --api-provider=openai.",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument(
        "--ks", default="",
        help="Comma-separated k values to also report (e.g. '1,5,10,25'). "
             "max(ks) must be <= --num-samples. Per-k metrics use the first k "
             "samples per board; written to overall_multi_k.json + per-k CSVs.",
    )
    p.add_argument("--masking-rule", default="default")
    p.add_argument("--state-format", choices=["sexpr", "xml"], default="sexpr")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip the API; replay each board's auto-router-produced "
                        "action sequence as the LLM 'response' to validate "
                        "the parse+replay+eval path.")
    p.add_argument("--reaggregate", action="store_true",
                   help="Skip everything and only rebuild summary.csv + "
                        "overall.json from existing per_board/<id>/sample_*.json.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()

    if args.reaggregate:
        if not args.output.is_dir():
            print(f"[ERROR] --reaggregate requires existing output dir: {args.output}",
                  file=sys.stderr)
            return 2
        try:
            results = reaggregate_from_disk(args.output)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        if not results:
            print(f"[ERROR] no per_board/ entries under {args.output}", file=sys.stderr)
            return 2
        summary_path = write_summary(results, args.output, args.num_samples)
        overall = overall_stats(results, args.num_samples)
        overall["wall_time_sec"] = 0.0
        overall["mode"] = "reaggregate"
        overall["api_provider"] = args.api_provider
        overall["api_model"] = args.api_model
        overall["dry_run"] = False
        overall["reaggregated"] = True
        with (args.output / "overall.json").open("w") as f:
            json.dump(overall, f, indent=2)

        if args.ks:
            try:
                ks_list = sorted({int(x) for x in args.ks.split(",") if x.strip()})
            except ValueError:
                print(f"[ERROR] --ks must be comma-separated ints: {args.ks!r}",
                      file=sys.stderr)
                return 2
            multi = overall_stats_multi_k(results, ks_list)
            multi["wall_time_sec"] = 0.0
            multi["mode"] = "reaggregate"
            multi["api_provider"] = args.api_provider
            multi["api_model"] = args.api_model
            multi["num_samples"] = args.num_samples
            with (args.output / "overall_multi_k.json").open("w") as f:
                json.dump(multi, f, indent=2)
            write_multi_k_summaries(results, args.output, ks_list)

        print("=" * 60)
        print("  Reaggregated from disk")
        print("=" * 60)
        for k_, v in overall.items():
            if isinstance(v, float):
                print(f"  {k_:<32} {v:.4f}")
            else:
                print(f"  {k_:<32} {v}")
        print(f"\n  summary  -> {summary_path}")
        print(f"  overall  -> {args.output / 'overall.json'}")
        return 0

    if not args.inputs:
        print("[ERROR] inputs required (or pass --reaggregate)", file=sys.stderr)
        return 2

    if args.mode == "few_shot" and not args.fewshot_pool:
        print("[ERROR] --mode few_shot requires --fewshot-pool DIR [DIR ...]",
              file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    targets = list(discover_inputs(args.inputs, args.recursive))
    if args.limit > 0:
        targets = targets[: args.limit]
    if not targets:
        print("[ERROR] no target .kicad_pcb files found", file=sys.stderr)
        return 2

    examples: list[FewShotExample] = []
    if args.mode == "few_shot":
        examples = load_fewshot_pool(
            args.fewshot_pool,
            k=args.num_fewshot,
            max_chars=args.fewshot_max_chars,
        )
        if len(examples) < args.num_fewshot:
            print(f"[WARN] requested {args.num_fewshot} few-shot examples but "
                  f"only loaded {len(examples)}", file=sys.stderr)
        if not examples:
            print("[ERROR] few-shot pool is empty", file=sys.stderr)
            return 2

    # Make prepare_plan_only_fewshot importable for --dry-run replay.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 60)
    print("  API-Seq LLM eval")
    print("=" * 60)
    print(f"  mode             : {args.mode}")
    print(f"  api              : {args.api_provider} / {args.api_model}"
          f"{'  (DRY RUN)' if args.dry_run else ''}")
    print(f"  num samples (k)  : {args.num_samples}")
    print(f"  target boards    : {len(targets)}")
    if examples:
        print(f"  few-shot pool    : {len(examples)} examples ("
              f"{', '.join(e.board_id for e in examples)})")
    print(f"  output           : {args.output}")
    print("=" * 60)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_kw):  # type: ignore[no-redef]
            return it

    results: list[BoardResult] = []
    t0 = time.time()
    for src, src_root in tqdm(targets, desc="boards", unit="board"):
        try:
            r = run_one_board(src, src_root, args, examples, args.output)
        except Exception as exc:
            traceback.print_exc()
            r = BoardResult(
                board_id=src.stem,
                source_path=str(src),
                error=f"unhandled: {type(exc).__name__}: {exc}",
            )
        results.append(r)

    summary_path = write_summary(results, args.output, args.num_samples)
    overall = overall_stats(results, args.num_samples)
    overall["wall_time_sec"] = time.time() - t0
    overall["mode"] = args.mode
    overall["api_provider"] = args.api_provider
    overall["api_model"] = args.api_model
    overall["dry_run"] = bool(args.dry_run)
    with (args.output / "overall.json").open("w") as f:
        json.dump(overall, f, indent=2)

    print()
    print("=" * 60)
    print("  Overall")
    print("=" * 60)
    for k_, v in overall.items():
        if isinstance(v, float):
            print(f"  {k_:<32} {v:.4f}")
        else:
            print(f"  {k_:<32} {v}")
    print(f"\n  summary  -> {summary_path}")
    print(f"  overall  -> {args.output / 'overall.json'}")

    if args.ks:
        try:
            ks_list = sorted({int(x) for x in args.ks.split(",") if x.strip()})
        except ValueError:
            print(f"[ERROR] --ks must be comma-separated ints: {args.ks!r}",
                  file=sys.stderr)
            return 2
        if max(ks_list) > args.num_samples:
            print(
                f"[ERROR] --ks contains {max(ks_list)} > --num-samples "
                f"{args.num_samples}; raise --num-samples or trim --ks",
                file=sys.stderr,
            )
            return 2
        multi = overall_stats_multi_k(results, ks_list)
        multi["wall_time_sec"] = overall["wall_time_sec"]
        multi["mode"] = args.mode
        multi["api_provider"] = args.api_provider
        multi["api_model"] = args.api_model
        multi["dry_run"] = bool(args.dry_run)
        multi["num_samples"] = args.num_samples
        with (args.output / "overall_multi_k.json").open("w") as f:
            json.dump(multi, f, indent=2)
        write_multi_k_summaries(results, args.output, ks_list)

        print()
        print("=" * 60)
        print(f"  Per-k Overall  (ks = {ks_list})")
        print("=" * 60)
        print(
            f"  {'k':>3}  {'boards':>6}  {'pass@k':>8}  "
            f"{'pass@k_unb':>10}  {'rb_best':>8}  {'rb_mean':>8}"
        )
        for k_str, st in multi["per_k"].items():
            if st.get("boards_evaluated", 0) == 0:
                print(f"  {k_str:>3}  {0:>6}  (no boards with >= k samples)")
                continue
            print(
                f"  {k_str:>3}  {st['boards_evaluated']:>6}  "
                f"{st['pass_at_k']:>8.4f}  {st['pass_at_k_unbiased']:>10.4f}  "
                f"{st['routability_at_k_best_mean']:>8.4f}  "
                f"{st['routability_at_k_mean_mean']:>8.4f}"
            )
        print(f"\n  multi_k -> {args.output / 'overall_multi_k.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
