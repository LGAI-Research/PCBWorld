"""CAD-Gen v3 — standalone (no v1/v2 imports).

Single self-contained file: v1's full pipeline + v2's 45° / octilinear
angle audit + v3's octilinear prompt + --strict-angle CLI flag.

Originally implemented as eval_cadgen_llm_v3.py that chained
v3 -> v2 -> v1 via monkey-patches. This standalone bakes both the v3
system prompt and v2's audit/evaluate_candidate wrap directly into the
source so the file runs without any of the chained versions on PYTHONPATH.

Sister of eval_plan_only_llm_v8_standalone.py.

Original v1 docstring follows.

Engine-free (CAD-Gen) via LLM (no-train): open-loop routing generation + evaluation.

Asks an LLM to produce *all* tracks/vias for an unrouted PCB in a single
completion (no env stepping), then patches the LLM output back into the
board and evaluates the resulting routed PCB with the same PCBWorld
pipeline that ``eval_pcb_file.py`` uses.

Two prompting modes:
    --mode zero_shot   prompt-only, no examples
    --mode few_shot    prompt + K (input PCB, output ROUTING) examples
                       harvested from --fewshot-pool

Per-board protocol:
    1. unroute the source .kicad_pcb (delete every segment + via) using
       the C++ round-trip helper -> input.kicad_pcb (input PCB text).
    2. build the prompt (zero-shot or few-shot), pass to the chosen API.
    3. sample N completions (default 5). For each:
        a. extract the <ROUTING>...</ROUTING> block,
        b. inject the extracted segments+vias into the unrouted PCB text,
        c. write the candidate .kicad_pcb (+ companion .kicad_pro),
        d. evaluate via eval_pcb_file.evaluate -> {success, routability, ...}.
    4. aggregate over the N samples:
        success_rate@N        = 1 if any sample succeeded else 0
        routability@N (best)  = max routability across samples
        routability@N (mean)  = mean routability across samples

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
import shutil
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


_THIS_DIR = Path(__file__).resolve().parent.parent.parent  # llm_eval→paper_repro→scripts→repo
_KICAD_RL_DIR = _THIS_DIR / "build_rl" / "pcbnew" / "python" / "rl"
for p in (_THIS_DIR, _KICAD_RL_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Prompt template (verbatim from the experiment spec; zero-shot strips the
# Examples block, few-shot fills it in).
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert PCB routing engineer using KiCad PCB format.\n\n"
    "Your task is to generate valid PCB routing (tracks and vias) for a "
    "given KiCad PCB board.\n\n"
    "## Instructions\n"
    "- Analyze the given PCB layout, including components, pads, and nets.\n"
    "- Generate routing (segments and vias) that correctly connects pads "
    "belonging to the same net.\n"
    "- Ensure:\n"
    "  - No short circuits between different nets\n"
    "  - Minimal via usage unless necessary\n"
    "  - Reasonable routing paths (avoid unnecessary detours)\n"
    "  - Respect layer usage (F.Cu, B.Cu)\n\n"
    "## Routing geometry: OCTILINEAR ROUTING with the 45\u00b0-ONLY constraint (mandatory)\n"
    "Use **octilinear routing** for every `(segment ...)`. Octilinear "
    "routing is a wire/edge routing style where connections are restricted "
    "to **eight directions**: the four cardinal (horizontal, vertical) plus "
    "the four diagonals at 45\u00b0. It sits between rectilinear routing (4 "
    "directions, Manhattan-style) and fully arbitrary Euclidean routing.\n\n"
    "### The 45\u00b0-only constraint (sub-rule of octilinear)\n"
    "Whenever a segment is *not* horizontal or vertical, it MUST be a "
    "**strict 45\u00b0 (or 135\u00b0) diagonal** \u2014 i.e. exactly `|dx| == |dy|`. "
    "No other diagonal angle is permitted. The following are all VIOLATIONS:\n"
    "    30\u00b0, 60\u00b0  (e.g. (0,0) -> (10, 5.77) or (0,0) -> (5, 8.66))\n"
    "    22.5\u00b0, 67.5\u00b0  (\"half-octilinear\" angles)\n"
    "    13.16\u00b0, 26.6\u00b0  (or any other arbitrary slope)\n"
    "Even tiny rounding deviations break the rule: `(start 0 0) (end 10 9.9)` "
    "is NOT octilinear. If you intend a 45\u00b0 diagonal, the rise must equal "
    "the run exactly.\n\n"
    "### Allowed segment shapes (and only these)\n"
    "Concretely, a segment from `(start sx sy)` to `(end ex ey)` is octilinear "
    "iff one of the following holds, with `dx = ex - sx` and `dy = ey - sy`:\n"
    "    1.  dy == 0                              (East / West   \u2014 0\u00b0,   horizontal)\n"
    "    2.  dx == 0                              (North / South \u2014 90\u00b0,  vertical)\n"
    "    3.  dx ==  dy   (and both nonzero)       (NE  / SW      \u2014 45\u00b0,  diagonal)\n"
    "    4.  dx == -dy   (and both nonzero)       (NW  / SE      \u2014 135\u00b0, diagonal)\n"
    "Equivalently, every segment is horizontal, vertical, or a 45\u00b0 diagonal "
    "where |dx| == |dy|. Anything else is forbidden.\n\n"
    "### How to handle non-octilinear paths\n"
    "If the natural route between two pads is at an angle that is not in "
    "{0\u00b0, 45\u00b0, 90\u00b0, 135\u00b0}, decompose it into multiple octilinear segments "
    "joined at corners. For example, to go from (0,0) to (10,3) you might "
    "use:\n"
    "    (segment ... (start 0 0) (end 3 3) ...)   ; 45\u00b0 diagonal\n"
    "    (segment ... (start 3 3) (end 10 3) ...)  ; horizontal\n"
    "Use as many segments as needed; each one must individually satisfy "
    "the rule above.\n\n"
    "This is exactly the constraint KiCad's interactive PNS router enforces "
    "with `corner_mode = MITERED_45`.\n\n"
    "## Output Format\n"
    "- Return ONLY the routing additions in valid KiCad PCB format.\n"
    "- Do NOT repeat the full board.\n"
    "- Only include `(segment ...)` and `(via ...)` entries.\n"
)


_FEWSHOT_HEADER = "\n---\n\n## Examples\n"
_FEWSHOT_EXAMPLE_TPL = (
    "\n### Example {idx}\n"
    "Input:\n<PCB>\n{pcb}\n</PCB>\n\n"
    "Output:\n<ROUTING>\n{routing}\n</ROUTING>\n"
)
_TASK_TPL = (
    "\n---\n\n## Task\n\n"
    "Now generate routing for the following PCB:\n\n"
    "<PCB>\n{target}\n</PCB>\n\n"
    "Output:\n<ROUTING>\n"
)


# ---------------------------------------------------------------------------
# .kicad_pcb (S-expression) helpers — strip / inject routing
# ---------------------------------------------------------------------------

_BLOCK_HEADS = ("(segment", "(via ", "(via\n", "(via\t")


def _scan_balanced(text: str, start: int) -> int:
    """Return the index *just past* the `)` that closes the `(` at ``start``.

    Walks the source character-by-character, ignoring parens that appear
    inside double-quoted strings (which can legally contain `(` or `)`).
    """
    assert text[start] == "("
    depth = 0
    i = start
    in_string = False
    n = len(text)
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"unbalanced parentheses starting at index {start}")


def _is_block_head(text: str, i: int) -> bool:
    """True iff the `(` at ``i`` opens a top-level (segment/via) block.

    We only match the explicit list — any other top-level form (footprint,
    gr_rect, embedded_fonts, ...) is left alone.
    """
    # Must start with "(segment" or "(via" + delimiter
    if text.startswith("(segment", i):
        nxt = text[i + len("(segment")] if i + len("(segment") < len(text) else ""
        return nxt in (" ", "\n", "\t", "(", ")")
    if text.startswith("(via", i):
        nxt = text[i + len("(via")] if i + len("(via") < len(text) else ""
        return nxt in (" ", "\n", "\t", "(", ")")
    return False


def split_routing(pcb_text: str) -> tuple[str, str]:
    """Strip every top-level segment/via block from ``pcb_text``.

    Returns ``(stripped_text, routing_text)``. ``routing_text`` is the
    concatenated, newline-joined block of removed entries (still valid
    KiCad S-expression fragments).
    """
    out: list[str] = []
    routing: list[str] = []
    i = 0
    n = len(pcb_text)
    while i < n:
        c = pcb_text[i]
        if c == "(" and _is_block_head(pcb_text, i):
            end = _scan_balanced(pcb_text, i)
            routing.append(pcb_text[i:end])
            # Strip any trailing whitespace/newline that belonged to this block
            j = end
            while j < n and pcb_text[j] in (" ", "\t"):
                j += 1
            if j < n and pcb_text[j] == "\n":
                j += 1
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out), "\n".join(routing)


def inject_routing(pcb_text: str, routing_text: str) -> str:
    """Insert ``routing_text`` just before the file's final closing paren.

    Skips an ``(embedded_fonts ...)`` clause if present so the routing
    sits between the footprints and that trailer (matching how KiCad
    writes routed boards in practice).
    """
    n = len(pcb_text)
    last = pcb_text.rfind(")")
    if last < 0:
        raise ValueError("not a kicad_pcb file (no closing paren)")
    # Walk back over trailing whitespace.
    j = last
    while j > 0 and pcb_text[j - 1] in (" ", "\t", "\n"):
        j -= 1
    # If the last form is (embedded_fonts ...), insert before *that*.
    # Find the matching '(' for the form ending at j-1.
    end_form = j  # exclusive
    if end_form > 0 and pcb_text[end_form - 1] == ")":
        # Walk back to find the matching '('.
        depth = 1
        k = end_form - 2
        in_string = False
        while k >= 0 and depth > 0:
            c = pcb_text[k]
            if in_string:
                if c == '"' and (k == 0 or pcb_text[k - 1] != "\\"):
                    in_string = False
                k -= 1
                continue
            if c == '"':
                in_string = True
                k -= 1
                continue
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        if k >= 0 and pcb_text.startswith("(embedded_fonts", k):
            j = k
            # Walk back over the whitespace before (embedded_fonts ...).
            while j > 0 and pcb_text[j - 1] in (" ", "\t"):
                j -= 1
    # Build inserted text. Use a leading newline so we're on a fresh line,
    # and a trailing newline so the next form starts at column 0.
    insert = "\n" + routing_text.rstrip() + "\n"
    return pcb_text[:j] + insert + pcb_text[j:]


# ---------------------------------------------------------------------------
# LLM response → routing extractor
# ---------------------------------------------------------------------------

_ROUTING_TAG_RE = re.compile(r"<ROUTING>\s*(.*?)\s*(?:</ROUTING>|$)", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:kicad|kicad_pcb|sexpr|lisp)?\s*\n(.*?)\n```", re.DOTALL)


def extract_routing(response: str) -> str:
    """Pull the routing block out of an LLM response.

    Tolerates:
        - properly tagged ``<ROUTING>...</ROUTING>``
        - missing closing tag (model truncated)
        - markdown fenced blocks
        - bare segment/via lines (last-resort)
    """
    m = _ROUTING_TAG_RE.search(response)
    if m:
        body = m.group(1)
    else:
        m = _FENCE_RE.search(response)
        body = m.group(1) if m else response
    # Drop empty and comment-only lines.
    keep = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(";"):
            continue
        keep.append(line)
    return "\n".join(keep)


# ---------------------------------------------------------------------------
# Few-shot example pool — harvest (input PCB text, output routing text)
# from already-routed boards.
# ---------------------------------------------------------------------------

@dataclass
class FewShotExample:
    board_id: str
    input_pcb: str
    output_routing: str


def _pcb_files_under(root: Path, recursive: bool, name_contains: str | None = None) -> list[Path]:
    if root.is_file() and root.suffix == ".kicad_pcb":
        return [root]
    if not root.is_dir():
        return []
    glob_iter = root.rglob("*.kicad_pcb") if recursive else root.glob("*.kicad_pcb")
    files = sorted(glob_iter)
    if name_contains:
        files = [f for f in files if name_contains in f.name]
    return files


def build_fewshot_pool(
    pool_paths: list[Path],
    k: int,
    max_chars: int | None,
    name_contains: str | None = None,
) -> list[FewShotExample]:
    """Return up to ``k`` few-shot examples sourced from routed boards.

    A "routed" board is one with at least one ``(segment ...)`` block.
    For each candidate we strip the routing off the textual board to make
    the example input, and use the stripped block as the example output.

    ``max_chars``: drop examples whose input PCB exceeds this many chars
    (controls prompt blow-up).
    """
    candidates: list[Path] = []
    for p in pool_paths:
        candidates.extend(
            _pcb_files_under(p, recursive=True, name_contains=name_contains)
        )
    out: list[FewShotExample] = []
    for f in candidates:
        if len(out) >= k:
            break
        try:
            text = f.read_text()
        except Exception:
            continue
        stripped, routing = split_routing(text)
        if not routing.strip():
            continue
        if max_chars is not None and len(stripped) > max_chars:
            continue
        # Disambiguate by parent dir — many real-board pools have identical
        # filenames (processed_v9_guide_v3.kicad_pcb) across hundreds of
        # subdirs.
        bid = f"{f.parent.name}/{f.stem}"
        out.append(FewShotExample(
            board_id=bid, input_pcb=stripped, output_routing=routing,
        ))
    return out


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    target_pcb: str, examples: list[FewShotExample],
) -> str:
    parts = []
    if examples:
        parts.append(_FEWSHOT_HEADER)
        for idx, ex in enumerate(examples, start=1):
            parts.append(_FEWSHOT_EXAMPLE_TPL.format(
                idx=idx, pcb=ex.input_pcb.strip(), routing=ex.output_routing.strip(),
            ))
    parts.append(_TASK_TPL.format(target=target_pcb.strip()))
    return "".join(parts)


# ---------------------------------------------------------------------------
# OpenAI client (only provider we wire here; others trivial to extend)
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
    """Single completion request that returns ``n`` samples.

    Reasoning models (o-series, GPT-5+) use ``max_completion_tokens`` and
    ignore ``temperature``; the regular API supports both. We auto-detect.
    """
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
    chat-completion request, so we issue ``n`` independent calls (in a small
    thread pool to recover throughput) and aggregate their results. Each call
    bumps ``temperature`` slightly via Together's own RNG, so independent
    samples differ even when temperature is moderate.

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
# Dry-run stub: copy ground-truth routing from the source board (if any).
# Lets you exercise the entire patch + eval pipeline with no API cost.
# ---------------------------------------------------------------------------

def dry_run_response(target_pcb_text: str, source_routing: str) -> str:
    """Wrap ``source_routing`` in a fake ``<ROUTING>...</ROUTING>`` reply.

    If there's no ground truth routing in the source board (e.g. synth_2L
    test boards are unrouted), we fall back to the empty routing — letting
    the eval pipeline confirm the patch path still produces a valid file.
    """
    return f"<ROUTING>\n{source_routing}\n</ROUTING>\n"


# ---------------------------------------------------------------------------
# Per-sample patch + evaluate
# ---------------------------------------------------------------------------

def evaluate_candidate(
    unrouted_pcb_path: Path,
    response: str,
    output_pcb_path: Path,
) -> dict:
    """Inject LLM-generated routing into the unrouted board, write it,
    then run :func:`eval_pcb_file.evaluate` on the patched copy.

    Returns the metrics dict from ``evaluate`` plus ``parse_ok`` /
    ``routing_chars`` / ``error`` fields. On any patch failure we still
    return a metrics-shaped dict (success=False, routability=0).
    """
    # The legacy ``eval_pcb_file.evaluate`` returned a flat metrics dict
    # keyed on ``drv_count``. The current scoring backend
    # ``eval.metrics.evaluate_one`` (a) requires a ``pro_path``
    # second argument and (b) splits DRV into errors_only vs
    # errors_and_promoted. We adapt to its signature and remap the
    # field names so downstream code that expects ``drv_count`` keeps
    # working (this is also the field BoardResult.aggregate reads).
    from eval.metrics import evaluate_one as _eval_routed

    unrouted_text = unrouted_pcb_path.read_text()
    routing_text = extract_routing(response)

    # Mirror the .kicad_pro companion alongside the patched .kicad_pcb so
    # KiCadEngine's auto-paired loader keeps the BDS / NetSettings, AND
    # so we have a concrete pro path to hand evaluate_one.
    unrouted_pro = unrouted_pcb_path.with_suffix(".kicad_pro")
    output_pro = output_pcb_path.with_suffix(".kicad_pro")
    if unrouted_pro.exists():
        shutil.copyfile(unrouted_pro, output_pro)

    info = {
        "parse_ok": True,
        "routing_chars": len(routing_text),
        "error": "",
    }
    try:
        patched = inject_routing(unrouted_text, routing_text) if routing_text else unrouted_text
        output_pcb_path.write_text(patched)
    except Exception as exc:
        info["parse_ok"] = False
        info["error"] = f"inject_failed: {type(exc).__name__}: {exc}"
        return {**_failure_metrics(), **info}

    pro_arg = str(output_pro) if output_pro.exists() else None
    try:
        m = _eval_routed(str(output_pcb_path), pro_arg)
    except Exception as exc:
        info["error"] = f"eval_failed: {type(exc).__name__}: {exc}"
        return {**_failure_metrics(), **info}

    # evaluate_one's schema lacks ``drv_count`` (it splits the counter).
    # Backfill the legacy field so BoardResult.aggregate's
    # ``drv_at_k_min`` and v2's audit can read it. Use
    # ``errors_and_promoted`` — that's the broader, dataset-default
    # severity mode.
    if "drv_count" not in m:
        m = dict(m)
        m["drv_count"] = int(m.get("drv_errors_and_promoted_count", 0))

    return {**m, **info}


def _failure_metrics() -> dict:
    return {
        "success": False,
        "routability": 0.0,
        "track_count": 0,
        "via_count": 0,
        "drv_count": 0,
        "wirelength_mm": 0.0,
        "extras": {},
    }


# ---------------------------------------------------------------------------
# v2-derived 45°/octilinear audit (inlined for standalone use). Adds per-sample
# angle_compliance_rate / nonaligned_segments / success_strict alongside v1's
# success / routability metrics.
# ---------------------------------------------------------------------------

ANGLE_TOLERANCE_DEG = 0.5


def _segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the segment's angle in degrees, normalized to [0, 180)."""
    import math
    if x1 == x2 and y1 == y2:
        return 0.0
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def is_angle_compliant(angle_deg: float, tol: float = ANGLE_TOLERANCE_DEG) -> bool:
    """True iff ``angle_deg`` is within tolerance of {0, 45, 90, 135}°."""
    for target in (0.0, 45.0, 90.0, 135.0):
        d = abs(((angle_deg - target + 90.0) % 180.0) - 90.0)
        if d <= tol:
            return True
    return False


_AUDIT_NUM_RE = r"-?\d+(?:\.\d+)?"


def _scan_segment_block_audit(text: str, start: int) -> tuple[int, str]:
    """Use v1's paren scanner to return (end_index, body) for the segment block."""
    end = _scan_balanced(text, start)
    return end, text[start:end]


def parse_segments_from_pcb(pcb_text: str) -> list[dict]:
    """Walk all top-level (segment ...) blocks; extract geometry. Skips malformed."""
    out: list[dict] = []
    i = 0
    n = len(pcb_text)
    while i < n:
        c = pcb_text[i]
        if c == "(" and pcb_text.startswith("(segment", i):
            try:
                end, body = _scan_segment_block_audit(pcb_text, i)
            except ValueError:
                i += 1
                continue
            seg = _segment_body_to_dict(body)
            if seg is not None:
                out.append(seg)
            i = end
            continue
        i += 1
    return out


def _segment_body_to_dict(body: str) -> dict | None:
    """Pull start/end/layer/net from a single (segment ...) body string."""
    import re as _re
    m_start = _re.search(rf"\(start\s+({_AUDIT_NUM_RE})\s+({_AUDIT_NUM_RE})\s*\)", body)
    m_end = _re.search(rf"\(end\s+({_AUDIT_NUM_RE})\s+({_AUDIT_NUM_RE})\s*\)", body)
    if not (m_start and m_end):
        return None
    x1, y1 = float(m_start.group(1)), float(m_start.group(2))
    x2, y2 = float(m_end.group(1)), float(m_end.group(2))
    angle = _segment_angle_deg(x1, y1, x2, y2)
    layer_m = _re.search(r'\(layer\s+"([^"]+)"\s*\)', body)
    net_m = _re.search(r"\(net\s+(\d+)\s*\)", body)
    return {
        "start": [x1, y1],
        "end": [x2, y2],
        "layer": layer_m.group(1) if layer_m else "",
        "net": int(net_m.group(1)) if net_m else -1,
        "angle_deg": angle,
        "aligned": is_angle_compliant(angle),
    }


def angle_compliance_metrics(pcb_text: str, max_violations: int = 16) -> dict:
    """Summarize 45°-alignment for every segment in a routed board."""
    segs = parse_segments_from_pcb(pcb_text)
    n_total = len(segs)
    n_aligned = sum(1 for s in segs if s["aligned"])
    violations = [
        {**s, "angle_deg": round(s["angle_deg"], 4)}
        for s in segs if not s["aligned"]
    ]
    rate = (n_aligned / n_total) if n_total else 1.0
    return {
        "angle_total_segments": n_total,
        "angle_aligned_segments": n_aligned,
        "angle_compliance_rate": rate,
        "angle_all_aligned": (n_total == 0) or (n_aligned == n_total),
        "nonaligned_segments": violations[:max_violations],
        "nonaligned_truncated": max(0, len(violations) - max_violations),
    }


# Wrap v1's evaluate_candidate so every sample carries the angle audit.
_evaluate_candidate_loose = evaluate_candidate


def evaluate_candidate(
    unrouted_pcb_path: Path,
    response: str,
    output_pcb_path: Path,
) -> dict:
    """v1.evaluate_candidate + 45° angle audit on the patched board."""
    metrics = _evaluate_candidate_loose(unrouted_pcb_path, response, output_pcb_path)
    if not metrics.get("parse_ok") or not output_pcb_path.exists():
        metrics.update({
            "angle_total_segments": 0,
            "angle_aligned_segments": 0,
            "angle_compliance_rate": 0.0,
            "angle_all_aligned": False,
            "nonaligned_segments": [],
            "nonaligned_truncated": 0,
            "success_strict": False,
        })
        return metrics
    try:
        angle_info = angle_compliance_metrics(output_pcb_path.read_text())
    except Exception as exc:  # noqa: BLE001
        angle_info = {
            "angle_total_segments": 0,
            "angle_aligned_segments": 0,
            "angle_compliance_rate": 0.0,
            "angle_all_aligned": False,
            "nonaligned_segments": [],
            "nonaligned_truncated": 0,
            "angle_audit_error": f"{type(exc).__name__}: {exc}",
        }
    metrics.update(angle_info)
    metrics["success_strict"] = bool(metrics.get("success") and angle_info["angle_all_aligned"])
    return metrics


def _promote_strict_success(samples: list[dict]) -> list[dict]:
    """Replace each sample's ``success`` with ``success_strict`` (backups under
    ``success_loose`` / ``routability_loose``). Routability is demoted to 0 when
    the sample isn't 45°-compliant.
    """
    out = []
    for s in samples:
        s2 = dict(s)
        s2["success_loose"] = s.get("success", False)
        s2["routability_loose"] = s.get("routability", 0.0)
        s2["success"] = bool(s.get("success_strict", False))
        if not s.get("angle_all_aligned", False):
            s2["routability"] = 0.0
        out.append(s2)
    return out


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
        # pass@k: at least one of the k samples succeeded (the "best"
        # interpretation the user asked about). Mathematically identical
        # to ``success_at_k`` below — it's just the clearer name.
        pass_at_k = int(self.successes > 0)
        # Per-board mean success rate: how many of the k samples succeeded.
        # This is the metric you'd average if you wanted "mean of pass rates"
        # rather than "pass@k".
        mean_success_rate = (self.successes / self.num_samples) if self.num_samples else 0.0
        return {
            "board_id": self.board_id,
            "source_path": self.source_path,
            "num_samples": self.num_samples,
            "successes": self.successes,
            "k": k,
            "pass_at_k": pass_at_k,
            "success_at_k": pass_at_k,  # legacy alias (== pass_at_k)
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
    """Yield ``(file, root)`` pairs (root = directory we mirrored under)."""
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


def unroute_board(src: Path, dst: Path, verbose: bool = False) -> None:
    """Round-trip ``src`` through the C++ router stripping all routing."""
    from pcb_world.engine.utils import load_and_save_via_engine
    dst.parent.mkdir(parents=True, exist_ok=True)
    load_and_save_via_engine(src, dst, unroute=True, verbose=verbose)


def run_one_board(
    src: Path,
    src_root: Path,
    args: argparse.Namespace,
    examples: list[FewShotExample],
    out_root: Path,
) -> BoardResult:
    rel = src.relative_to(src_root) if src.is_relative_to(src_root) else Path(src.name)
    board_id = rel.with_suffix("").as_posix().replace("/", "__")
    # If the relative path is just the filename (e.g. when callers pass each
    # file explicitly so src_root == src.parent), disambiguate via parent
    # dir. Real-board collections often share the same filename across
    # hundreds of subdirs (processed_v9_guide_v3.kicad_pcb).
    if "/" not in rel.as_posix() and src.parent.name:
        board_id = f"{src.parent.name}__{board_id}"
    board_dir = out_root / "per_board" / board_id
    board_dir.mkdir(parents=True, exist_ok=True)

    result = BoardResult(board_id=board_id, source_path=str(src))

    # 1. Unroute (always — gives a clean canonical input even when the
    #    source board is already empty of routing).
    unrouted = board_dir / "input.kicad_pcb"
    try:
        unroute_board(src, unrouted, verbose=False)
    except Exception as exc:
        result.error = f"unroute_failed: {type(exc).__name__}: {exc}"
        return result

    target_pcb_text = unrouted.read_text()

    # 2. Build user prompt (system prompt is fixed).
    user_prompt = build_user_prompt(target_pcb_text, examples)
    (board_dir / "prompt.txt").write_text(
        "==== SYSTEM ====\n" + _SYSTEM_PROMPT + "\n==== USER ====\n" + user_prompt
    )

    # 3. Sample N completions.
    if args.dry_run:
        # Use the *source* routing (if any) so the eval pipeline gets a
        # realistic patch even without an API.
        _, src_routing = split_routing(src.read_text())
        responses = [dry_run_response(target_pcb_text, src_routing)] * args.num_samples
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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

    # 4. Patch + evaluate each sample.
    for i, response in enumerate(responses):
        sample_pcb = board_dir / f"sample_{i:02d}.kicad_pcb"
        (board_dir / f"sample_{i:02d}.response.txt").write_text(response)
        sample_metrics = evaluate_candidate(unrouted, response, sample_pcb)
        sample_metrics["sample_idx"] = i
        sample_metrics["usage"] = usage
        with (board_dir / f"sample_{i:02d}.json").open("w") as f:
            json.dump(sample_metrics, f, indent=2)
        result.samples.append(sample_metrics)

    # 5. Per-board aggregate.
    agg = result.aggregate(args.num_samples)
    with (board_dir / "aggregate.json").open("w") as f:
        json.dump(agg, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Summary + overall stats
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
            w.writerow({k: row.get(k, "") for k in _SUMMARY_FIELDS})
    return path


def reaggregate_from_disk(out_root: Path) -> list[BoardResult]:
    """Rebuild ``BoardResult`` objects by reading existing sample JSONs.

    Walks ``<out_root>/per_board/<board_id>/sample_*.json`` and reconstructs
    a ``BoardResult`` per board. Skips boards that have no sample files
    (treats them as failures with ``error="no_samples_found"``). Useful
    after an interrupted / partial API run, or whenever you change the
    aggregate definitions and want to refresh summary.csv / overall.json
    without re-spending API tokens.
    """
    per_board_root = out_root / "per_board"
    if not per_board_root.is_dir():
        raise FileNotFoundError(f"no per_board/ under {out_root}")

    results: list[BoardResult] = []
    for board_dir in sorted(per_board_root.iterdir()):
        if not board_dir.is_dir():
            continue
        board_id = board_dir.name
        sample_files = sorted(board_dir.glob("sample_*.json"))
        # Try to recover the original source path from a previous aggregate.json
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
                # Don't drop the whole board over one corrupt sample —
                # treat the bad file as a failed sample.
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
    """Compute @k metrics from the first ``k`` samples of a board.

    Mirrors ``BoardResult.aggregate`` but on a sample slice — used by the
    multi-k overall stats so we can report @1 / @5 / @10 / @25 from a single
    pool of samples per board.
    """
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
    """Per-k overall stats from a single shared sample pool per board.

    For each k in ``ks`` we report two flavors:
      - first-k (matches ``aggregate``): pass / best / mean over samples[:k].
      - unbiased: Codex pass@k estimator from the FULL sample count, mean'd
        across boards (interpretable as "expected fraction of boards where a
        random k-sized draw would yield at least one success").
    """
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
                # Don't fabricate: skip boards short on samples for this k.
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
            # First-k semantics (match the existing single-k overall_stats).
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
            # Unbiased Codex-style pass@k for noise reduction.
            "pass_at_k_unbiased": statistics.fmean(unbiased_pk),
        }
    return out


def write_multi_k_summaries(results: list[BoardResult], out_root: Path, ks: list[int]) -> None:
    """Write per-k summary_k{K}.csv siblings to summary.csv for analyzers."""
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
        # pass@k: fraction of boards where at least one of k samples succeeded.
        # This is the "success_rate@K best" the experiment spec asks for.
        "pass_at_k": sum(pass_k) / len(pass_k),
        # Backward-compat alias (older runs read this name).
        "success_rate_at_k": sum(pass_k) / len(pass_k),
        # Mean of per-board (successes/k) — the "average per-sample success
        # rate" companion. Equals total_successes / (boards*k).
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
        description="LLM-only CAD-Gen routing eval (zero-shot / few-shot).",
    )
    p.add_argument(
        "inputs", nargs="*", type=Path, default=[],
        help="One or more .kicad_pcb files and/or directories of target boards. "
             "Not required when --reaggregate is set.",
    )
    p.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output directory.",
    )
    p.add_argument(
        "-r", "--recursive", action="store_true",
        help="When an input is a directory, recurse into subdirectories.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="If >0, evaluate only the first N target boards.",
    )
    p.add_argument(
        "--mode", choices=["zero_shot", "few_shot"], default="zero_shot",
    )
    p.add_argument(
        "--fewshot-pool", type=Path, nargs="*", default=[],
        help="Directory (or directories) of *routed* example boards used "
             "to harvest few-shot pairs. Required when --mode=few_shot.",
    )
    p.add_argument(
        "--num-fewshot", type=int, default=2,
        help="Number of few-shot examples (default 2).",
    )
    p.add_argument(
        "--fewshot-max-chars", type=int, default=20000,
        help="Drop few-shot examples whose input PCB text exceeds this "
             "many characters (default 20000).",
    )
    p.add_argument(
        "--fewshot-name-contains", type=str, default=None,
        help="Only consider example files whose filename contains this "
             "substring (e.g. 'processed_v9_guide').",
    )
    p.add_argument(
        "--num-samples", "-k", type=int, default=5,
        help="Samples per board (the @k in success_rate@k / routability@k).",
    )
    p.add_argument(
        "--api-provider", choices=["openai", "together"], default="openai",
        help="OpenAI-compatible API provider. Together uses the same SDK at "
             "https://api.together.xyz/v1; set TOGETHER_API_KEY or pass --api-key.",
    )
    p.add_argument("--api-model", default="gpt-5.4")
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--api-base-url", default="https://api.together.xyz/v1",
        help="Base URL for --api-provider=together (default Together public endpoint).",
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
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip the API; replay each source board's own routing as the "
             "completion. Useful for validating the patch+eval pipeline.",
    )
    p.add_argument(
        "--strict-angle", action="store_true",
        help="Aggregate pass@k / success using success_strict (success AND every\n"
             "segment within 0.5\u00b0 of {0,45,90,135}\u00b0). Per-sample JSONs always carry\n"
             "both the strict and loose metrics — this only flips the aggregate.",
    )
    p.add_argument(
        "--reaggregate", action="store_true",
        help="Skip everything and only re-read per_board/<id>/sample_*.json "
             "from --output to recompute summary.csv + overall.json. "
             "Use this to cheaply refresh metric definitions or to harvest "
             "results from a partial / interrupted run without re-spending "
             "API tokens. Inputs/api flags are ignored in this mode.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --strict-angle: swap BoardResult.aggregate so its `success` reads
    # `success_strict` for the duration of this run. Per-sample JSONs on
    # disk keep both keys; only in-memory aggregation flips.
    if getattr(args, "strict_angle", False):
        _original_aggregate = BoardResult.aggregate

        def _aggregate_strict(self, k):
            backup = self.samples
            try:
                self.samples = _promote_strict_success(self.samples)
                return _original_aggregate(self, k)
            finally:
                self.samples = backup

        BoardResult.aggregate = _aggregate_strict


    args.output = args.output.resolve()

    # Reaggregate path: skip inputs, fewshot pool, and API entirely. Just
    # walk the existing per_board/ tree and recompute summary + overall.
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
        print(f"  source dir : {args.output}")
        print(f"  boards     : {len(results)} ({sum(1 for r in results if r.error)} failed)")
        print()
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
        print(
            "[ERROR] --mode few_shot requires --fewshot-pool DIR [DIR ...]",
            file=sys.stderr,
        )
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    # Resolve target boards.
    targets = list(discover_inputs(args.inputs, args.recursive))
    if args.limit > 0:
        targets = targets[: args.limit]
    if not targets:
        print("[ERROR] no target .kicad_pcb files found", file=sys.stderr)
        return 2

    # Build few-shot pool once.
    examples: list[FewShotExample] = []
    if args.mode == "few_shot":
        examples = build_fewshot_pool(
            args.fewshot_pool,
            k=args.num_fewshot,
            max_chars=args.fewshot_max_chars,
            name_contains=args.fewshot_name_contains,
        )
        if len(examples) < args.num_fewshot:
            print(
                f"[WARN] requested {args.num_fewshot} few-shot examples but "
                f"only collected {len(examples)} from "
                f"{[str(p) for p in args.fewshot_pool]}",
                file=sys.stderr,
            )
        if not examples:
            print("[ERROR] few-shot pool is empty after filtering", file=sys.stderr)
            return 2

    print("=" * 60)
    print("  CAD-Gen LLM eval")
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
        header = (
            f"  {'k':>3}  {'boards':>6}  {'pass@k':>8}  "
            f"{'pass@k_unb':>10}  {'rb_best':>8}  {'rb_mean':>8}"
        )
        print(header)
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
