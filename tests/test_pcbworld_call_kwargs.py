"""Every ``PCBWorld(...)`` call in the repo passes only keywords its constructor declares.

``PCBWorld`` rejects an unknown kwarg with ``TypeError``. A script that still passes a
removed name fails per board at run time, where a broad ``except`` can hide it as a
routing failure (the plan-only scripts passed ``state_format`` and crashed every board
while exiting 0). The scan catches such a call before it runs.

Limits: it sees a call only when the callee is spelled ``PCBWorld`` (an alias or a
factory is not seen), and it skips ``**kwargs`` expansions, which resolve at run time.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from pcb_world.core.env import PCBWorld
from tests.helpers.tracked_files import REPO, calls_named, tracked_py


def _declared_kwargs() -> set[str]:
    params = inspect.signature(PCBWorld.__init__).parameters.values()
    return {p.name for p in params
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def _unknown_kwargs(path: Path, declared: set[str]) -> list[tuple[int, str]]:
    # `**kwargs` expansions (arg None) are resolved at run time — not checkable here.
    return [(node.lineno, k.arg) for node in calls_named(path, {"PCBWorld"})
            for k in node.keywords if k.arg is not None and k.arg not in declared]


def test_every_pcbworld_call_uses_declared_kwargs():
    declared = _declared_kwargs()
    offenders = [f"{p.relative_to(REPO)}:{line} {kw}="
                 for p in tracked_py() for line, kw in _unknown_kwargs(p, declared)]
    assert not offenders, f"PCBWorld(...) called with kwargs its __init__ does not declare: {offenders}"


def test_scan_flags_an_undeclared_kwarg(tmp_path):
    src = tmp_path / "calls.py"
    src.write_text("env = PCBWorld(board_path='b', state_format='sexpr')\n"
                   "env = core.PCBWorld(seed=1, **extra)\n"
                   "env = pcb_world.core.env.PCBWorld(bogus=1)\n")
    assert _unknown_kwargs(src, _declared_kwargs()) == [(1, "state_format"), (3, "bogus")]
