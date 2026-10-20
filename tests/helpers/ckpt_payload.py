"""Checkpoint payloads for tests, built from the golden fixtures in ``tests/fixtures/ckpt_args/``.

Each fixture is one checkpoint generation: its stored ``args`` (config keys only, path values
reduced to their stem), the weight shapes ``configs.loader.ckpt`` derives from, its provenance,
and the expected resolution (``expected``). Tests that need a checkpoint config start from a
generation and override only what they exercise; the state_dict holds shape stand-ins, not weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "ckpt_args"
GENERATIONS = sorted(p.stem for p in FIXTURES.glob("*.json"))
CURRENT = "iclr_2026_10"


def fixture(generation: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{generation}.json").read_text())


def ckpt_payload(generation: str | dict[str, Any] = CURRENT, *, drop: tuple[str, ...] = (),
                 shapes: dict[str, list[int] | None] | None = None, **args: Any) -> dict[str, Any]:
    """A checkpoint payload of ``generation`` (a fixture name, or a fixture dict of the same
    shape — e.g. ``tests/helpers/ckpt_golden.py``'s, whose ``trainer`` marker is carried) with
    ``args`` overriding stored keys, ``drop`` removing them, and ``shapes`` replacing (list) or
    removing (None) weight entries."""
    fx = fixture(generation) if isinstance(generation, str) else generation
    stored = {k: v for k, v in fx["args"].items() if k not in drop}
    stored.update(args)
    sd_shapes = dict(fx["shapes"])
    for k, v in (shapes or {}).items():
        if v is None:
            sd_shapes.pop(k, None)
        else:
            sd_shapes[k] = v
    return {
        "args": stored,
        "policy_state_dict": {k: SimpleNamespace(shape=tuple(v)) for k, v in sd_shapes.items()},
        "provenance": fx["provenance"],
        "obs_probe": {"stub": 1} if fx["obs_probe"] else None,
        **({"trainer": fx["trainer"]} if fx.get("trainer") else {}),
    }


def quiet_load(payload: dict[str, Any], *, strict: bool = False):
    """:func:`configs.loader.ckpt.load_ckpt_config` with its legacy-fill warning silenced."""
    import warnings

    from configs.loader.ckpt import load_ckpt_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_ckpt_config(payload, strict=strict)


def ckpt_config(generation: str = CURRENT, **kwargs: Any):
    """The resolved ``values`` of :func:`ckpt_payload` (non-strict load)."""
    return quiet_load(ckpt_payload(generation, **kwargs))[0]


#: Keys ``configs.loader.ckpt`` derives from the weights (a stored value must agree).
DERIVED = ("shape_obs", "legacy_edge_encoding", "legacy_action_history", "use_critic",
           "n_max_slots", "legacy_pad_layer_encoding", "legacy_net_encoding")


def resolve_state_dict(state_dict: dict[str, Any], **args: Any):
    """Resolved config of a current-generation checkpoint holding ``state_dict`` (real
    weights of a test-sized model): the policy fields start at the ``RLPolicyConfig``
    defaults, ``args`` override them (e.g. the model's own size), and the weight-shaped
    keys are left to the loader."""
    from dataclasses import asdict

    from configs.loader.schema import RLPolicyConfig

    policy = {k: v for k, v in asdict(RLPolicyConfig()).items() if k not in DERIVED}
    payload = ckpt_payload(drop=DERIVED, **{**policy, **args})
    payload["policy_state_dict"] = state_dict
    return quiet_load(payload)[0]


def save_ckpt(path: Path, generation: str | dict[str, Any] = CURRENT, **kwargs: Any) -> Path:
    """Write :func:`ckpt_payload` to ``path`` as a loadable ``.pt`` (zero tensors of the
    fixture's shapes stand in for the weights)."""
    import torch

    payload = ckpt_payload(generation, **kwargs)
    payload["policy_state_dict"] = {k: torch.zeros(v.shape)
                                    for k, v in payload["policy_state_dict"].items()}
    torch.save(payload, path)
    return path
