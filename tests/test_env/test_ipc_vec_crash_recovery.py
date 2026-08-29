"""Vec-pool recovery when an ENGINE SERVER child crashes (engine IPC).

Under IPC a fatal C++ signal kills the per-worker engine-server child, not
the vec worker. The worker translates ``EngineServerCrashed`` into its own
death (re-raise → pipe EOF — see ``_decoder_worker``), so the parent's
EXISTING dead-worker path must fire exactly as for an in-process segfault:
respawn + synthetic terminated step with ``info["engine_crash"]``, postmortem
accounting, and — critically — sibling envs untouched.

Own file: spawns a subprocess pool (multi-second) — keeps xdist's loadfile
scheduling honest. N/A with ``KICAD_ENGINE_IPC=0`` (crash would kill the
worker directly; that path is covered by test_crash_diag / reload tests).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from methods.rl_agent.wrappers.factory import make_decoder_env_pool
from pcb_world.core.masking import ACT_NET_SELECT, ACT_START_ROUTE
from pcb_world.engine import engine_available
from pcb_world.engine.router_client import ipc_enabled

BOARD = Path(__file__).resolve().parent.parent / "fixtures" / "simple_routing_board.kicad_pcb"

pytestmark = [
    pytest.mark.skipif(not engine_available(), reason="C++ router build not present"),
    pytest.mark.skipif(
        not ipc_enabled(),
        reason="KICAD_ENGINE_IPC=0 — no engine server, IPC crash recovery N/A",
    ),
    pytest.mark.skipif(not BOARD.exists(), reason="fixture board missing"),
]


def _server_pid_of_worker(worker_pid: int) -> int:
    out = subprocess.run(
        ["pgrep", "-P", str(worker_pid), "-f", "rl_engine_server"],
        capture_output=True, text=True,
    )
    pids = [int(x) for x in out.stdout.split()]
    assert len(pids) == 1, (
        f"expected exactly one engine server under worker {worker_pid}, got {pids}"
    )
    return pids[0]


def _wait_dead(pid: int, deadline_s: float = 10.0) -> None:
    """Wait until ``pid`` is gone OR a zombie (state Z). The SIGKILLed server
    stays a zombie until its parent — the vec worker, blocked in recv() —
    exits, so ``os.kill(pid, 0)`` alone would never see it die."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            return
        if state == "Z":
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} still alive")


def test_server_crash_respawns_worker_and_spares_siblings(
    pool_kwargs, tmp_path, monkeypatch,
):
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", str(tmp_path))  # parent postmortem
    pool = make_decoder_env_pool(
        str(BOARD), n_envs=2,
        **pool_kwargs(max_steps=30, masking_rule="default_no_finish",
                      reward_rule="drc_only_dense"),
    )
    try:
        pool.reset_all()
        # Select a net on both envs while everything is healthy: only VALID
        # actions touch the engine (idle/mask-reject short-circuit env-side),
        # so the crash must be probed with a valid engine-touching action.
        net_sel = np.array([ACT_NET_SELECT, 0, -1], dtype=np.int64)
        pool.step_async(np.stack([net_sel, net_sel]))
        _obs0, _r0, _t0, _tr0, infos0 = pool.step_wait()
        assert infos0[0]["action_success"] and infos0[1]["action_success"]

        victim = pool.processes[1]
        srv = _server_pid_of_worker(victim.pid)
        os.kill(srv, signal.SIGKILL)
        _wait_dead(srv)

        start = np.array([ACT_START_ROUTE, 0, -1], dtype=np.int64)
        pool.step_async(np.stack([start, start]))
        _obs, rews, terms, _truncs, infos = pool.step_wait()

        # Crashed env: worker died (EOF) → respawned, synthetic terminated
        # step — byte-for-byte the in-process worker-segfault contract.
        assert infos[1].get("engine_crash") is True
        assert bool(terms[1]) and rews[1] == -1.0
        assert pool.respawn_total == 1
        assert pool.processes[1].pid != victim.pid
        assert pool.processes[1].is_alive()
        assert not victim.is_alive()
        pms = list(tmp_path.glob("*_env1_postmortem.json"))
        assert len(pms) == 1
        assert json.loads(pms[0].read_text())["respawn_count"] == 1

        # Sibling env: its start_route executed normally on ITS server.
        assert "engine_crash" not in infos[0]
        assert infos[0]["action_success"] is True
        assert not bool(terms[0])

        # Both envs functional afterwards (recovery already reset env1, so
        # net_select is again the valid opener there).
        pool.step_async(np.stack([net_sel, net_sel]))
        _obs2, _r2, _t2, _tr2, infos2 = pool.step_wait()
        assert "engine_crash" not in infos2[0]
        assert "engine_crash" not in infos2[1]
        assert infos2[1]["action_success"] is True
        assert pool.respawn_total == 1        # no further respawns
    finally:
        pool.close()
