"""KDD few-shot generator (``prepare_plan_only_fewshot_llm.py``): errors are not quality drops.

An engine crash during replay is a failed sample; any other exception (an API error
included) fails the board. When any board errored, the run fails naming the error
instead of blaming the routability filter. Loaded the way ``python <script>`` loads it,
the script finds the v8 system prompt and the ``eval`` package (not ``scripts/eval.py``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pcb_world.engine.router_client import EngineServerCrashed

REPO = Path(__file__).resolve().parents[2]
LLM_EVAL = REPO / "experiments/kdd/llm_eval"
_RESPONSE = "<actions>\nnet_end\n</actions>\n"


@pytest.fixture
def prep(monkeypatch):
    # The script inserts into sys.path at import; restore it afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))
    from experiments.kdd.llm_eval import prepare_plan_only_fewshot_llm as mod
    return mod


def test_script_imports_resolve_as_when_run_directly(tmp_path):
    # A fresh interpreter with the script's directory first on sys.path, as
    # ``python prepare_plan_only_fewshot_llm.py`` has it: the v3 prompt is the v8
    # script's system prompt, and replay scoring imports eval.metrics.
    code = (
        "import sys; from pathlib import Path\n"
        f"sys.path.insert(0, {str(LLM_EVAL)!r})\n"
        "import prepare_plan_only_fewshot_llm as m\n"
        "from experiments.kdd.llm_eval.eval_plan_only_llm_v8_standalone import _SYSTEM_PROMPT\n"
        "assert m._system_prompt('v3') == _SYSTEM_PROMPT\n"
        "r = m._replay_and_score(Path('b.kicad_pcb'), 'no actions here', Path('s.kicad_pcb'))\n"
        "assert r['error'] == 'no_actions_parsed', r\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]


def _replay_raising(monkeypatch, exc: Exception) -> None:
    from experiments.kdd.llm_eval import eval_plan_only_llm_v8_standalone as v8

    def _raise(*_a, **_kw):
        raise exc

    monkeypatch.setattr(v8, "replay_actions_and_eval", _raise)


def test_engine_crash_in_replay_is_a_failed_sample(tmp_path, monkeypatch, prep):
    _replay_raising(monkeypatch, EngineServerCrashed("server died"))
    m = prep._replay_and_score(tmp_path / "b.kicad_pcb", _RESPONSE, tmp_path / "s.kicad_pcb")
    assert m["error"].startswith("replay_failed: EngineServerCrashed")


def test_other_replay_error_is_not_swallowed(tmp_path, monkeypatch, prep):
    _replay_raising(monkeypatch, ValueError("replay bug"))
    with pytest.raises(ValueError, match="replay bug"):
        prep._replay_and_score(tmp_path / "b.kicad_pcb", _RESPONSE, tmp_path / "s.kicad_pcb")


def test_every_board_erroring_fails_with_the_error(tmp_path, monkeypatch, capsys, prep):
    board = tmp_path / "b.kicad_pcb"
    board.write_text("")

    def _broken(*_a, **_kw):
        raise TypeError("PCBWorld got unknown kwarg(s)")

    monkeypatch.setattr(prep, "_board_static_for", _broken)
    monkeypatch.setattr(sys, "argv", ["prep", "--src-files", str(board), "-o", str(tmp_path / "out")])
    assert prep.main() == 1
    err = capsys.readouterr().err
    assert "1 of 1 boards errored; first: b.kicad_pcb: TypeError" in err
    assert "quality filter" not in err


def test_api_error_on_one_board_fails_the_run(tmp_path, monkeypatch, capsys, prep):
    for name in ("a", "b"):
        (tmp_path / f"{name}.kicad_pcb").write_text("")

    def _generate(**kw):
        if "board_b" in kw["user_prompt"]:
            raise RuntimeError("api down")
        return [_RESPONSE], {"total_tokens": 0}

    monkeypatch.setattr(prep, "_board_static_for", lambda path: f"board_{path.stem}")
    monkeypatch.setattr(prep, "_generate_action_sequences", _generate)
    monkeypatch.setattr(prep, "_replay_and_score", lambda *_a, **_kw: {
        "actions_text": "net_end", "n_actions": 1, "success": True, "clean_pass": True,
        "routability": 1.0, "drv_errors_and_promoted_count": 0, "final_potential": 0.0, "error": ""})
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prep", "--src-dir", str(tmp_path), "-o", str(out)])
    assert prep.main() == 1
    assert "1 of 2 boards errored; first: b.kicad_pcb: RuntimeError: api down" in capsys.readouterr().err
    assert (out / "a.json").exists()    # the healthy board's example is still written
