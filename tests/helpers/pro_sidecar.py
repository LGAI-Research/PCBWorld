"""Write a ``.kicad_pro`` sidecar for a test-synthesized ``.kicad_pcb``.

The engine's load contract refuses pro-less boards (their design rules
would silently be KiCad compile-time defaults). Test boards written from
inline s-expression templates carry no project file, so give them the
default-rules pro used across the fixture corpus — behaviourally identical
to the pre-contract era, when such boards always ran on default rules.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "simple_routing_board.kicad_pro"
)


def write_default_pro(pcb_path: str | Path) -> Path:
    """Copy the default-rules pro next to ``pcb_path`` (same stem)."""
    pro = Path(pcb_path).with_suffix(".kicad_pro")
    pro.write_bytes(_TEMPLATE.read_bytes())
    return pro


def materialize_pro_pair(rendered_pcb: str | Path) -> str:
    """Round-trip a rendered board into the pcb+pro pair the contract accepts.

    For templates that declare rules INSIDE the pcb (e.g. a ``net_class``
    block): the engine saver carries those values into a companion
    ``.kicad_pro``, exactly like the retired upgrade cache used to. A plain
    :func:`write_default_pro` would replace them with KiCad defaults and
    change the board's effective clearances. Returns the new pcb path
    (sibling file, ``_pair`` stem).
    """
    from pcb_world.engine.utils import load_and_save_via_engine

    src = Path(rendered_pcb)
    out = src.with_name(src.stem + "_pair.kicad_pcb")
    load_and_save_via_engine(src, out)
    return str(out)
