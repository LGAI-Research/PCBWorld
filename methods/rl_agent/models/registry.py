"""Model-variant registry — name → (codec, net) bundle.

The seam for adding a model variant: a variant is a
``models/vN/`` bundle of {spec, encoding, net} that shares vocab/head shape, and
one entry here. ``algorithm`` (PPO/GRPO) and ``masking-rule`` compose
orthogonally on top of a variant and are deliberately NOT keyed here.

Currently one variant (v1); ``v2`` (e.g. a GNN over point/line topology) would be
a second bundle + one line — see §3.1.
"""
from __future__ import annotations

from methods.rl_agent.models.v1 import encoding as _v1_encoding
from methods.rl_agent.models.v1 import net as _v1_net

VARIANTS = {
    "v1": (_v1_encoding, _v1_net),
}


def get_variant(name: str = "v1"):
    """Return the ``(codec_module, net_module)`` bundle registered under *name*."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise KeyError(
            f"unknown model variant {name!r}; known: {sorted(VARIANTS)}"
        )
