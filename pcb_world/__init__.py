"""Shared world for all routing methods (rl · llm · baseline) — not an agent.

Layers:
- ``pcb_world.engine``    : the sole interface to the C++ KiCad bindings (KiCadEngine, singleton)
- ``pcb_world.core``      : PCBWorld (step/reset/reward, candidate-pool -> obs)
- ``pcb_world.vec``       : vectorized backends (subproc / ray) + candidate_pool · slots
- ``pcb_world.rendering`` : PCBRenderer (kicad-cli)
"""
