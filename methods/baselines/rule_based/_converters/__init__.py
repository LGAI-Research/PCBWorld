"""Verbatim copies of the SES/ORS → kicad_pcb converters.

Source: the ``bench_results_kicadpcb/_scripts/`` helpers of the original
benchmark result tree (under the shared data root; not distributed with
this repo).

Both modules expose a single ``convert(src_path, base_pcb_path, out_pcb_path)
-> stats: dict`` function that injects routed track/via tokens from the
respective format into the original unrouted ``.kicad_pcb`` text.

No pcbnew / KiCad CLI dependency — pure text S-expression processing.
"""
from .ses_to_kicadpcb import convert as convert_ses  # noqa: F401
from .ors_to_kicadpcb import convert as convert_ors  # noqa: F401

__all__ = ["convert_ses", "convert_ors"]
