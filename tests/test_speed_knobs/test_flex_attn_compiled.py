"""flex_attention under the adopted training combo: the static-compiled stack, decode and heads
regions around flex must match the eager sdpa policy (actions exact, outputs/grads 1e-4).

Split from ``test_flex_attn.py``: the compile alone takes ~18 s, so it gets its own file and
therefore its own pytest-xdist worker under ``--dist loadfile``.
"""

from __future__ import annotations

import pytest
import torch

from tests.test_speed_knobs._helpers import assert_compile_matches_eager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="speed knobs are CUDA-only")
@pytest.mark.slow   # inductor compile
def test_policy_flex_with_efficient_regions_matches_sdpa():
    # The adopted training combo: static-compiled stack around flex.
    assert_compile_matches_eager(("stack", "decode", "heads"), attn="flex")
