"""Speed-knob (bf16 autocast / torch.compile regions) equivalence guards.

Split one-file-per-compile-region so xdist ``--dist loadfile`` spreads the
inductor compile cost (~5-8s per region) across workers instead of serializing
~22s on a single worker — the former wall-clock critical path of the suite.
Shared model/batch/assertion helpers live in ``_helpers.py``.
"""
