"""Shared helpers for tests of the rule YAMLs (reward / masking / DRC).

``ENV_PENALTIES`` / ``RULE_PENALTIES``: the penalties every reward rule states under
``potential:``, for tests that write a rule inline — values are the ones the rule loader
used to default to; ``RULE_PENALTIES`` adds ``step_penalty`` (also a ``PotentialReward``
argument). ``tracked_rule_files``: every tracked rule YAML under given git pathspecs, found by
its top-level key; ``LOADERS``: the loader of each rule kind (it refuses an incomplete tracked file).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pcb_world.core.masking import load_masking_rule
from pcb_world.core.reward_config import load_reward_config
from pcb_world.engine.drc_config import load_drc_config

REPO = Path(__file__).resolve().parents[1]

ENV_PENALTIES = {"invalid_action_penalty": 0.01, "parse_fail_penalty": 0.2, "mask_reject_penalty": 0.09}
RULE_PENALTIES = {"step_penalty": 0.01, **ENV_PENALTIES}

# Top-level key that marks a YAML as a rule file of that kind.
_RULE_KEYS = {"potential": "reward", "actions": "masking", "global_minima": "drc"}


def yaml_lines(penalties: dict[str, float]) -> str:
    """*penalties* as indented ``potential:`` lines of a rule YAML."""
    return "".join(f"  {key}: {value}\n" for key, value in penalties.items())


def tracked_rule_files(*pathspecs: str) -> list[tuple[str, Path]]:
    """``(kind, path)`` of every tracked YAML under the git *pathspecs* with a top-level rule key."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-oE", f"^({'|'.join(_RULE_KEYS)}):", "--", *pathspecs],
        capture_output=True, text=True, check=True,
    ).stdout
    found = {}
    for line in out.splitlines():
        rel, key, _ = line.rsplit(":", 2)
        if rel.endswith((".yaml", ".yml")):
            found[rel] = _RULE_KEYS[key]
    return [(kind, REPO / rel) for rel, kind in sorted(found.items())]


LOADERS = {"reward": load_reward_config, "masking": load_masking_rule, "drc": load_drc_config}

