"""Tests for CrashLogger (pcb_world.diag.step_stream) and its PCBWorld wiring.

Verifies that:
1. File is created as <process stem>_e<env_id>_steps.jsonl in the diag dir
2. reset() writes a reset event; reset clears the previous episode
3. step() writes flushed pre_step (action + router state + rss_mb) / post_step pairs
4. Simulated crash leaves pre_step without post_step (the crash marker)
5. close() deletes the file (clean exit leaves no trace); killswitch disables all writes
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from pcb_world.core.env import PCBWorld
from pcb_world.core.masking import ACTION_NAMES
from pcb_world.diag.step_stream import CrashLogger, _safe_serialize


BOARD_PATH = str(
    Path(__file__).resolve().parent.parent / "fixtures" / "simple_routing_board.kicad_pcb"
)


@pytest.fixture
def crash_log_dir():
    """Temporary directory for crash logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def env(crash_log_dir, monkeypatch):
    """Create env with crash logging to temp dir (monkeypatch restores the
    session-wide diag dir the conftest set)."""
    monkeypatch.setenv("KICAD_CRASH_LOG_DIR", crash_log_dir)
    e = PCBWorld(
        board_path=BOARD_PATH,
        max_steps=200,
        masking_rule="strict_phase",
    )
    yield e
    e.close()


def _read_log(crash_log_dir: str) -> list[dict]:
    """Read all JSONL entries from the crash log directory."""
    entries = []
    for f in Path(crash_log_dir).glob("*_steps.jsonl"):
        for line in f.read_text().strip().splitlines():
            if line:
                entries.append(json.loads(line))
    return entries


class TestCrashLoggerUnit:
    """Unit tests for CrashLogger class.

    close() deletes the file (clean exit = no evidence), so every test reads
    the entries BEFORE closing.
    """

    def test_creates_file_with_env_id_suffix(self, crash_log_dir):
        logger = CrashLogger(env_id=42, log_dir=crash_log_dir)
        files = list(Path(crash_log_dir).glob("*_e42_steps.jsonl"))
        assert len(files) == 1
        logger.close()

    def test_on_reset_writes_event(self, crash_log_dir):
        logger = CrashLogger(env_id=1, log_dir=crash_log_dir)
        logger.on_reset("test_board.kicad_pcb")

        entries = _read_log(crash_log_dir)
        assert len(entries) == 1
        assert entries[0]["event"] == "reset"
        assert entries[0]["board"] == "test_board.kicad_pcb"
        logger.close()

    def test_on_reset_clears_previous(self, crash_log_dir):
        logger = CrashLogger(env_id=1, log_dir=crash_log_dir)
        logger.on_reset("board1.kicad_pcb")
        logger.on_pre_step({"action_type": 0}, {"is_routing": False})
        logger.on_post_step(True, {})

        # Second reset should clear everything
        logger.on_reset("board2.kicad_pcb")

        entries = _read_log(crash_log_dir)
        assert len(entries) == 1
        assert entries[0]["board"] == "board2.kicad_pcb"
        logger.close()

    def test_pre_post_step_pair(self, crash_log_dir):
        logger = CrashLogger(env_id=1, log_dir=crash_log_dir)
        logger.on_reset("board.kicad_pcb")

        action = {"action_type": 3, "x_mm": 15.0, "y_mm": 20.0}
        state = {"is_routing": True, "head_xy": [10.0, 10.0]}
        logger.on_pre_step(action, state)
        logger.on_post_step(True, {"unrouted_count": 2})

        entries = _read_log(crash_log_dir)
        assert len(entries) == 3  # reset + pre + post
        assert entries[1]["event"] == "pre_step"
        assert entries[1]["action"]["action_type"] == 3
        assert entries[1]["action"]["x_mm"] == 15.0
        assert entries[1]["router_head"]["is_routing"] is True
        # Evidence for OOM diagnosis. read_rss_mb() reads /proc/self/statm, so
        # the field is None off Linux (macOS dev boxes) — assert the value
        # only where the source exists, but always require the key to be
        # emitted.
        assert "rss_mb" in entries[1]
        if Path("/proc/self/statm").exists():
            assert entries[1]["rss_mb"] and entries[1]["rss_mb"] > 0
        assert entries[2]["event"] == "post_step"
        assert entries[2]["success"] is True
        logger.close()

    def test_simulated_crash_has_pre_without_post(self, crash_log_dir):
        """If crash occurs after pre_step, post_step is missing."""
        logger = CrashLogger(env_id=1, log_dir=crash_log_dir)
        logger.on_reset("board.kicad_pcb")
        logger.on_pre_step({"action_type": 5}, {"is_routing": True})
        # No post_step — simulating crash

        entries = _read_log(crash_log_dir)
        assert len(entries) == 2  # reset + pre (no post)
        assert entries[-1]["event"] == "pre_step"
        assert entries[-1]["action"]["action_type"] == 5
        logger.close()

    def test_close_deletes_file(self, crash_log_dir):
        """Clean close leaves nothing — a surviving file means an incident."""
        logger = CrashLogger(env_id=1, log_dir=crash_log_dir)
        logger.on_reset("board.kicad_pcb")
        assert len(list(Path(crash_log_dir).glob("*_steps.jsonl"))) == 1
        logger.close()
        assert list(Path(crash_log_dir).glob("*_steps.jsonl")) == []

    def test_two_envs_one_process_do_not_clobber(self, crash_log_dir):
        """Same process stem + different env ids → distinct files."""
        a = CrashLogger(env_id=11, log_dir=crash_log_dir)
        b = CrashLogger(env_id=22, log_dir=crash_log_dir)
        a.on_reset("a.kicad_pcb")
        b.on_reset("b.kicad_pcb")
        assert a._path != b._path
        assert json.loads(Path(a._path).read_text())["board"] == "a.kicad_pcb"
        assert json.loads(Path(b._path).read_text())["board"] == "b.kicad_pcb"
        a.close()
        b.close()

    def test_killswitch_disables_all_writes(self, crash_log_dir, monkeypatch):
        monkeypatch.setenv("KICAD_CRASH_DIAG", "0")
        logger = CrashLogger(env_id=1, log_dir=crash_log_dir)
        logger.on_reset("board.kicad_pcb")
        logger.on_pre_step({"action_type": 0}, {})
        logger.on_post_step(True, {})
        logger.close()
        assert os.listdir(crash_log_dir) == []


class TestCrashLoggerIntegration:
    """Integration tests with actual PCBWorld."""

    def test_reset_creates_log(self, env, crash_log_dir):
        env.reset()
        entries = _read_log(crash_log_dir)
        assert len(entries) >= 1
        assert entries[0]["event"] == "reset"
        assert "simple_routing_board" in entries[0]["board"]

    def test_step_logs_action_and_state(self, env, crash_log_dir):
        env.reset()

        action = {
            "action_type": 0,  # net_select
            "x_mm": 10.0, "y_mm": 10.0, "layer": 0,
            "net_id": 1, "routing_mode": 2,
        }
        env.step(action)

        entries = _read_log(crash_log_dir)
        pre_steps = [e for e in entries if e["event"] == "pre_step"]
        post_steps = [e for e in entries if e["event"] == "post_step"]

        assert len(pre_steps) == 1
        assert len(post_steps) == 1
        assert pre_steps[0]["action"]["action_type"] == 0
        assert pre_steps[0]["action"]["net_id"] == 1
        assert "current_net" in pre_steps[0]["router_head"]
        assert "is_routing" in pre_steps[0]["router_head"]
        assert post_steps[0]["success"] is True

    def test_multi_step_episode(self, env, crash_log_dir):
        env.reset()

        actions = [
            {"action_type": 0, "x_mm": 0, "y_mm": 0, "layer": 0, "net_id": 1, "routing_mode": 2},
            {"action_type": 1, "x_mm": 10.0, "y_mm": 10.0, "layer": 1, "net_id": 1, "routing_mode": 2},
            {"action_type": 5, "x_mm": 40.0, "y_mm": 10.0, "layer": 1, "net_id": 1, "routing_mode": 2},
        ]
        for a in actions:
            env.step(a)

        entries = _read_log(crash_log_dir)
        pre_steps = [e for e in entries if e["event"] == "pre_step"]
        post_steps = [e for e in entries if e["event"] == "post_step"]

        assert len(pre_steps) == 3
        assert len(post_steps) == 3
        # Step numbers should be sequential
        assert [e["step"] for e in pre_steps] == [1, 2, 3]

    def test_reset_clears_previous_episode(self, env, crash_log_dir):
        env.reset()
        env.step({"action_type": 0, "x_mm": 0, "y_mm": 0, "layer": 0, "net_id": 1, "routing_mode": 2})

        # Second episode
        env.reset()
        entries = _read_log(crash_log_dir)

        # Should only have the latest reset, no steps from previous episode
        assert entries[0]["event"] == "reset"
        assert len([e for e in entries if e["event"] == "pre_step"]) == 0

    def test_routing_state_logged(self, env, crash_log_dir):
        """After start_route, pre_step should show is_routing=True and head_xy."""
        env.reset()
        env.step({"action_type": 0, "x_mm": 0, "y_mm": 0, "layer": 0, "net_id": 1, "routing_mode": 2})
        env.step({"action_type": 1, "x_mm": 10.0, "y_mm": 10.0, "layer": 1, "net_id": 1, "routing_mode": 2})

        # Clear and do another step while routing
        env.reset()
        env.step({"action_type": 0, "x_mm": 0, "y_mm": 0, "layer": 0, "net_id": 1, "routing_mode": 2})
        env.step({"action_type": 1, "x_mm": 10.0, "y_mm": 10.0, "layer": 1, "net_id": 1, "routing_mode": 2})

        # Third step should show routing state
        env.step({"action_type": 5, "x_mm": 40.0, "y_mm": 10.0, "layer": 1, "net_id": 1, "routing_mode": 2})

        entries = _read_log(crash_log_dir)
        finish_pre = [e for e in entries if e["event"] == "pre_step" and e["step"] == 3][0]
        assert finish_pre["router_head"]["is_routing"] is True
        assert finish_pre["router_head"]["head_xy"] == [10.0, 10.0]


class TestSafeSerialize:
    """Tests for _safe_serialize helper."""

    def test_numpy_array(self):
        import numpy as np
        assert _safe_serialize(np.array([1.0, 2.0])) == [1.0, 2.0]

    def test_numpy_scalar(self):
        import numpy as np
        assert _safe_serialize(np.float32(3.14)) == pytest.approx(3.14, rel=1e-5)
        assert _safe_serialize(np.int64(42)) == 42

    def test_nested_dict(self):
        import numpy as np
        data = {"a": np.array([1, 2]), "b": {"c": np.float64(3.0)}}
        result = _safe_serialize(data)
        assert result == {"a": [1, 2], "b": {"c": 3.0}}

    def test_plain_types_pass_through(self):
        assert _safe_serialize(42) == 42
        assert _safe_serialize("hello") == "hello"
        assert _safe_serialize(None) is None
