"""KiCad API layer: the sole interface to C++ bindings.

Modules outside this package should never import kicad_rl_router directly.
"""

import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RL_MODULE_DIR = _os.environ.get("CADAGENT_KICAD_RL_MODULE_DIR")
if _RL_MODULE_DIR:
    _RL_LIB_PATH = _RL_MODULE_DIR
else:
    _RL_BUILD_DIR = _os.environ.get(
        "CADAGENT_KICAD_RL_BUILD_DIR",
        _os.path.join(_PROJECT_ROOT, "build_rl"),
    )
    _RL_LIB_PATH = _os.path.join(_RL_BUILD_DIR, "pcbnew", "python", "rl")
if _os.path.isdir(_RL_LIB_PATH) and _RL_LIB_PATH not in _sys.path:
    _sys.path.insert(0, _RL_LIB_PATH)

def engine_available() -> bool:
    """True when the C++ router build is present on this host.

    Checks for the ``kicad_rl_router`` extension file WITHOUT importing it —
    the import would load the GPL shared library into this (NC) process,
    which engine-IPC mode exists to avoid. This is the availability probe
    tests must use instead of ``import kicad_rl_router`` (enforced by the
    ``import-hygiene`` check in tools/docs/check_docs.py).
    """
    import glob as _glob
    return bool(_glob.glob(_os.path.join(_RL_LIB_PATH, "kicad_rl_router*")))


from pcb_world.engine.kicad_engine import (
    KiCadEngine,
    CLEANUP_FINALIZE,
    CLEANUP_TOPOLOGY_PRESERVING,
)
from pcb_world.engine.containers import (
    BoardMeta,
    BoardSnapshot,
    CleanupResult,
    DRCResult,
    RewardSnapshot,
    RoutingSessionState,
)
from pcb_world.engine.drc import DRCUtils
from pcb_world.engine.router_client import engine_home


__all__ = [
    "KiCadEngine",
    "engine_available",
    "engine_home",
    "BoardMeta",
    "BoardSnapshot",
    "RoutingSessionState",
    "CleanupResult",
    "CLEANUP_TOPOLOGY_PRESERVING",
    "CLEANUP_FINALIZE",
    "DRCResult",
    "RewardSnapshot",
    "DRCUtils"
]
