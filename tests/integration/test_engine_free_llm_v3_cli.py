"""KDD engine-free evaluator (``eval_engine_free_llm_v3_standalone.py``): exit status.

A run whose boards errored must not exit 0 like a finished one — live and on
``--reaggregate`` alike; a run whose boards all evaluated exits 0.
"""
from __future__ import annotations

import json
import sys

import pytest

_SAMPLE = {"success": True, "routability": 1.0, "drv_count": 0, "wirelength_mm": 1.0}


@pytest.fixture
def v3(monkeypatch):
    # The script inserts into sys.path at import; restore it afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))
    from experiments.kdd.llm_eval import eval_engine_free_llm_v3_standalone as mod
    return mod


def _run(monkeypatch, v3, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["v3", *map(str, argv)])
    return v3.main()


def test_every_board_failing_exits_nonzero_live_and_on_reaggregate(tmp_path, monkeypatch, capsys, v3):
    board = tmp_path / "b.kicad_pcb"
    board.write_text("")

    def _broken(*_a, **_kw):
        raise RuntimeError("unroute bug")

    monkeypatch.setattr(v3, "unroute_board", _broken)
    out = tmp_path / "out"
    assert _run(monkeypatch, v3, board, "-o", out, "--dry-run", "-k", 1) == 1
    assert "1 of 1 boards errored" in capsys.readouterr().err
    assert json.loads((out / "overall.json").read_text())["boards_failed"] == 1
    assert _run(monkeypatch, v3, "--reaggregate", "-o", out, "-k", 1) == 1


def test_reaggregate_of_a_clean_run_exits_zero_and_one_empty_board_fails_it(tmp_path, monkeypatch, v3):
    out = tmp_path / "out"
    good = out / "per_board/good"
    good.mkdir(parents=True)
    (good / "sample_00.json").write_text(json.dumps(_SAMPLE))
    assert _run(monkeypatch, v3, "--reaggregate", "-o", out, "-k", 1) == 0
    (out / "per_board/empty").mkdir()
    assert _run(monkeypatch, v3, "--reaggregate", "-o", out, "-k", 1) == 1
