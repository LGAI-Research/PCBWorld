"""Complete RL factory kwargs for tests.

``make_decoder_env`` / ``make_decoder_env_pool`` take the whole
:class:`~configs.loader.schema.RLEnvConfig` plus the per-call ``seed`` /
``policy_net_select`` — none of them has a default. Tests build the YAML config
and override only what they exercise.

Mirrored by the ``pool_kwargs`` fixture in ``tests/conftest.py`` for tests that
prefer fixture injection; both delegate here so there is one definition.
"""

from __future__ import annotations

from typing import Any


def full_env_kwargs(**overrides: Any) -> dict[str, Any]:
    """``env_config`` (``RLEnvConfig()`` with ``overrides`` applied — keys of
    ``RLEnvConfig.to_record``) plus ``seed`` (0) / ``policy_net_select`` (False)."""
    from configs.loader.schema import RLEnvConfig

    per_call = {"seed": 0, "policy_net_select": False}
    per_call.update({k: overrides.pop(k) for k in list(overrides) if k in per_call})
    return {"env_config": RLEnvConfig().with_flat(overrides), **per_call}
