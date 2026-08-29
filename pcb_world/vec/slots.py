"""Net-slot contract shared by the wrapper augmentation and the policy tokenizer.

``N_MAX_SLOTS`` is the size of the slot-embedding table. It must agree between:

- the **tokenizer** (``methods.rl_agent.models.v1.embedding``), which owns a
  ``(N_MAX_SLOTS, d_model)`` slot-embedding table and tags each net's tokens
  with a slot id in ``[0, N_MAX_SLOTS)``; and
- the **slot_perm augmentation** (``methods.rl_agent.wrappers.augmentation``), which
  samples a random permutation of ``[0, N_MAX_SLOTS)`` per episode.

Because the wrapper (env layer) and the tokenizer (training layer) both need
this value, it lives here in ``pcb_world/vec`` so both import it *downward*
(same rationale as ``candidate_pool``). token_vocabulary re-exports it so its
existing import path keeps working.
"""
from __future__ import annotations

N_MAX_SLOTS = 64
