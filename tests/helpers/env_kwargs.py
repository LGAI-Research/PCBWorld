"""Complete env-contract kwarg surfaces for tests.

The factories have no signature defaults for env-contract knobs (see
``methods.rl_agent.wrappers.factory._REQ``): a partial call raises instead of
silently filling values in, which is what let five knobs drift between training
and validation for months. Tests therefore build the whole surface and override
only what they exercise.

Mirrored by the ``pool_kwargs`` fixture in ``tests/conftest.py`` for tests that
prefer fixture injection; both delegate here so there is one definition.
"""

from __future__ import annotations

from typing import Any


def full_env_kwargs(**overrides: Any) -> dict[str, Any]:
    """``to_pool_kwargs()`` plus the two knobs both train and eval pass."""
    from configs.loader.schema import RLEnvConfig

    return {
        **RLEnvConfig().to_pool_kwargs(),
        "seed": 0,
        "policy_net_select": False,
        **overrides,
    }
