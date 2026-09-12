"""torch.compile 'heads' region vs eager — one file per region so xdist
(--dist loadfile) parallelizes the inductor compile cost across workers."""

from __future__ import annotations

import pytest
import torch

from tests.test_speed_knobs._helpers import assert_compile_matches_eager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="speed knobs are CUDA-only",
)


def test_heads_matches_eager():
    assert_compile_matches_eager(("heads",))
