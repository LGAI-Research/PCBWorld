"""Resolved-config provenance dump — run-start first-class artifact.

Writes ``<save_dir>/config_resolved.yaml`` at trainer startup, *before* any
env/model/engine build so the file survives a crash, from the SAME dict the
checkpoints store as ``payload["args"]`` (``vars(args)`` — see
``RLTrainer._train_ckpt_payload`` in :mod:`methods.rl_agent.training.loop`),
so the two artifacts cannot diverge. Method-agnostic (plain args dict in,
yaml out) — lives in ``methods/_shared`` so the LLM/GRPO trainers can adopt
the same dump; currently wired into ``train_ppo`` only.

On ``--resume`` the dump goes to a
separate ``config_resolved.resume_iter<N>.yaml`` instead of overwriting the
original — the post-hoc audit trail for the CLI-wins resume semantics
(which hyperparameters the run was actually restarted with).

Deliberately light: stdlib + PyYAML only, no torch / env / engine imports at
module level, so ``train_ppo --dump-config-only`` works without a GPU or
``kicad_rl_router`` (batch preflight enabler — see
``tools/experiments/preflight_diff.py``).
"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

#: provenance meta keys prepended to every dump (excluded from preflight diffs)
META_KEYS = ("_created", "_git_rev", "_version")


def _read_version() -> str:
    """README ``<!--VERSION-->`` marker (the version single source of truth)."""
    try:
        # [^<] (not [\d.]) so personal-line suffixes (v0.26.0+b16) survive.
        m = re.search(
            r"<!--VERSION-->v?([^<\s]+)<!--/VERSION-->",
            (_REPO / "README.md").read_text(),
        )
        return f"v{m.group(1)}" if m else "unknown"
    except OSError:
        return "unknown"


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def collect_provenance() -> dict:
    """Best-effort code identity for ckpt stamping: repo version + git HEAD.

    Purely informational (loaders print it, never enforce it) — failures
    outside a git checkout degrade to "unknown" instead of blocking training.
    Single definition shared with the trainer's ckpt stamp (loop.py) so the
    version source stays the README marker above.
    """
    prov: dict = {
        "repo_version": _read_version().lstrip("v"),
        "git_commit": "unknown", "git_dirty": None,
    }
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        prov["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"],
            cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return prov


def resolved_config_filename(start_iter: int | None = None) -> str:
    """``config_resolved.yaml``; on resume, a per-restart file kept alongside."""
    if start_iter is None:
        return "config_resolved.yaml"
    return f"config_resolved.resume_iter{start_iter}.yaml"


def resume_start_iter(resume_path: str) -> int:
    """First iteration the resumed run will execute (mirrors ``RLTrainer._resume``)."""
    import torch  # lazy: only the resume path needs it

    ckpt = torch.load(resume_path, map_location="cpu")
    return int(ckpt.get("iteration", 0)) + 1


def dump_resolved_config(
    args_dict: dict, save_dir: str, *, start_iter: int | None = None,
) -> Path:
    """Write the resolved-config yaml, echo it sorted to stdout, return the path.

    The stdout echo (``[config] key = value`` per line) lands at the head of a
    nohup launch.log, so the run's effective config is greppable there too.
    """
    import yaml

    payload = {
        "_created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "_git_rev": _git_rev(),
        "_version": _read_version(),
        **args_dict,
    }
    os.makedirs(save_dir, exist_ok=True)
    path = Path(save_dir) / resolved_config_filename(start_iter)
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    for key in sorted(payload):
        print(f"[config] {key} = {payload[key]}")
    print(f"[config] resolved config -> {path}")
    return path


# ---------------------------------------------------------------------------
# Env records — what the envs were ACTUALLY built with
# ---------------------------------------------------------------------------
# ``config_resolved.yaml`` above records *intent*: the parsed CLI namespace.
# That alone is not enough — a knob can read ``true`` there while the training
# envs run with the factory default, because the trainer hand-lists its kwargs
# and can skip one. These records store the *effect*: the kwarg dicts the
# factories resolved (post-default), one per side, plus their diff. A difference
# is not judged here; it is reported, and the launch declares which differences
# are intended (``--expect-env-diff``).

ENV_RECORDS_DIR = "env_records"

#: Marker for a key present on one side only (never a real kwarg value).
_ABSENT = "<absent>"

#: Differences the harness creates on purpose, so a stock run has nothing to
#: declare. ``seed``: validation is seeded from ``--eval-base-seed`` by design
#: (held-out rollouts must not replay the training stream), so the two sides
#: differ on the shipped defaults (42 vs 1000) and the declaration would carry
#: no information — the gate could never pass unaided. Only a *value*
#: difference is inherent: a side missing the key entirely still halts, since
#: that absence is how a dropped kwarg hides.
INHERENT_DIFF_KEYS = frozenset({"seed"})


def _diff(train: dict, val: dict) -> dict:
    """``{key: {"train": ..., "val": ...}}`` for every key that differs.

    A key missing on one side is reported as ``"<absent>"`` rather than being
    skipped — absence is exactly how a dropped kwarg hides.
    """
    absent = _ABSENT
    out: dict[str, dict] = {}
    for key in sorted(set(train) | set(val)):
        t, v = train.get(key, absent), val.get(key, absent)
        if t != v:
            out[key] = {"train": t, "val": v}
    return out


def dump_env_records(
    save_dir: str, *, train_env: dict, val_env: dict, policy: dict,
) -> tuple[Path, dict]:
    """Write ``<save_dir>/env_records/`` and return ``(dir, diff)``.

    ``policy.yaml`` is written once, not per side: inline validation reuses the
    very policy object training steps (see ``_build_evaluators``), so a
    train/val split of it would always be byte-identical.
    """
    import yaml

    out = Path(save_dir) / ENV_RECORDS_DIR
    out.mkdir(parents=True, exist_ok=True)
    diff = _diff(train_env, val_env)
    for name, payload in (("train_env", train_env), ("val_env", val_env),
                          ("policy", policy), ("diff", diff)):
        (out / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
        )
    return out, diff


def check_expected_env_diff(diff: dict, expected: str | None) -> None:
    """Halt unless every train/val difference was declared at launch.

    Intent belongs to the experiment, not to a list in the source tree: a run
    may legitimately want validation to differ (a deliberately harder training
    reward, say), and a global "these keys may differ" table would either
    forbid that or quietly bless a genuine mistake. So the declaration travels
    with the launch command, and the error message is the line to paste.
    ``INHERENT_DIFF_KEYS`` are exempt: the harness itself creates them, so
    demanding a declaration would only teach launchers to paste noise.
    """
    declared = {k.strip() for k in (expected or "").split(",") if k.strip()}
    unexplained = [
        k for k in sorted(set(diff) - declared)
        if not (k in INHERENT_DIFF_KEYS
                and _ABSENT not in (diff[k]["train"], diff[k]["val"]))
    ]
    if not unexplained:
        return
    rows = "\n".join(
        f"    {k:<28} train={diff[k]['train']!r}  val={diff[k]['val']!r}"
        for k in unexplained
    )
    raise SystemExit(
        "\n[env-contract] the training env and the validation env are "
        "configured differently:\n"
        f"{rows}\n\n"
        "  If the difference is intended, paste this into the launch command:\n"
        f"    --expect-env-diff {','.join(sorted(set(unexplained) | declared))}\n"
        "  If it is not, start by finding where the two paths diverge.\n"
    )
