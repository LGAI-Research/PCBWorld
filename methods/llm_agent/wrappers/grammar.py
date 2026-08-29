"""Guided-decoding grammars for cadagent LLM rollouts.

Co-located with ``action_converter._ACTION_SCHEMA`` so the regex stays in lock-step
with the canonical action signatures. Imported by the verl-agent vLLM rollout
to build a ``GuidedDecodingParams`` object that constrains every generated
response to exactly one ``<think>...</think><action>...</action>`` pair with
a syntactically valid action body.

Semantic mask validity (phase, net_id range, layer count, ...) is still
enforced env-side — this grammar only eliminates parse-failure rejections.

Action body shapes mirror ``methods.llm_agent.wrappers.action_converter._ACTION_SCHEMA``:

  net_select  <int>
  start_route <float> <float> <int>
  net_end
  make_line   <float> <float> <m|p|w>
  make_via    <float> <float> <m|p|w>
  finish      <m|p|w>
"""

# Inner think body: any chars except '<' (prevents nested or repeated tags).
# Length 1..400 leaves headroom for the action segment within the rollout's
# max_response_length (currently 256 tokens, ≈1k chars).
_THINK_BODY = r"[^<]{1,400}"

# Numeric tokens. Coordinates are mm with up to 3 decimals on the env side;
# integer part up to 4 digits covers boards up to ~1000mm, fraction up to 3
# digits matches the env's rounding. Optional leading '-' for safety.
_FLOAT = r"-?\d{1,4}(\.\d{1,3})?"
_INT_SMALL = r"\d{1,2}"   # layer (1..N copper layers), bounded
_INT_NETID = r"\d{1,4}"   # net_id, generously bounded
_MODE = r"[mpw]"          # MarkObstacles / PushAndShove / Walkaround

_ACTION_BODY = (
    r"(?:"
    rf"net_select {_INT_NETID}"
    rf"|start_route {_FLOAT} {_FLOAT} {_INT_SMALL}"
    r"|net_end"
    rf"|make_line {_FLOAT} {_FLOAT} {_MODE}"
    rf"|make_via {_FLOAT} {_FLOAT} {_MODE}"
    rf"|finish {_MODE}"
    r")"
)

CADAGENT_V1_REGEX = (
    rf"<think>{_THINK_BODY}</think><action>{_ACTION_BODY}</action>"
)

GRAMMARS = {
    "cadagent_v1": CADAGENT_V1_REGEX,
}


def get_grammar_regex(name: str) -> str:
    """Look up a registered grammar by name.

    Raises ``KeyError`` if ``name`` is unknown so the rollout fails fast
    instead of silently dropping the constraint.
    """
    if name not in GRAMMARS:
        raise KeyError(
            f"Unknown guided-decoding grammar '{name}'. "
            f"Available: {sorted(GRAMMARS)}"
        )
    return GRAMMARS[name]
