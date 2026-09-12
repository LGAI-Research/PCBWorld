"""Unit tests for the central evaluator + sinks (eval/evaluator.py) and the
sink-agnostic summary type (eval/metrics.py EvalSummary).

No KiCad / torch / rollout dependency: a fake ``rollout_fn`` feeds synthetic
results so the data-type projection and the metric/logger split are exercised in
isolation.

NOTE: the ``eval.*`` imports are deliberately function-local rather than
module-level. Importing ``eval.evaluator``/``eval.metrics`` at *collection* time
(pure-Python though they are) reorders module initialisation enough to tip a
pre-existing, order-sensitive KiCad native double-init landmine elsewhere in the
full suite (see memory ``project_pycache_native_segfault``; full-suite-only
SIGSEGV around test_env/test_reward_modes). Deferring the imports to test
execution time — which runs after those env tests — keeps the suite green
without touching the unrelated native fragility.
"""
import json
import math

import pytest


class CapturingLogger:
    """A MetricLogger that records every add_scalar call."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def _eval_result():
    from eval.metrics import EvalResult

    return EvalResult(
        per_rollout=[{"board_index": 7, "board_id": "board_a", "final_potential": 1.25}],
        per_board=[
            {
                "board_id": "board_a",
                "board_index": 7,
                "final_potential": 1.25,
                "success": True,
                "wirelength_mm": math.nan,  # non-finite -> dropped
                "selected_rollout_idx": 3,  # not a metric field -> dropped
            }
        ],
        overall={
            "fp_mean_of_means": 1.25,
            "final_potential": 1.25,
            "selection_mode": "final_potential",  # non-numeric -> dropped
        },
    )


# ---------------------------------------------------------------------------
# EvalSummary projection
# ---------------------------------------------------------------------------


def test_to_logger_dict_projection_rules():
    from eval.metrics import EvalSummary

    m = EvalSummary.from_eval_result(_eval_result())
    out = m.to_logger_dict(prefix="val")

    assert out["val/fp_mean_of_means"] == 1.25
    # bare alias keys (== explicit *_mean twins) are dropped from logger
    # emission only (_LOGGER_DUPLICATE_KEYS); per_board keys are unaffected
    assert "val/final_potential" not in out
    assert out["val/per_board/board_a/final_potential"] == 1.25
    assert out["val/per_board/board_a/success"] == 1.0  # bool -> float
    # dropped: non-numeric overall, non-finite + non-metric per-board fields
    assert "val/selection_mode" not in out
    assert "val/per_board/board_a/wirelength_mm" not in out
    assert "val/per_board/board_a/selected_rollout_idx" not in out


def test_to_logger_dict_prefix_and_per_board_toggle():
    from eval.metrics import EvalSummary

    m = EvalSummary.from_eval_result(_eval_result())
    assert all(k.startswith("eval/") for k in m.to_logger_dict(prefix="eval"))
    no_pb = m.to_logger_dict(prefix="val", include_per_board=False)
    assert not any("/per_board/" in k for k in no_pb)


def test_roundtrip_eval_result_carries_per_rollout():
    from eval.metrics import EvalResult
    from eval.metrics import EvalSummary

    m = EvalSummary.from_eval_result(_eval_result())
    rt = m.as_eval_result()
    assert isinstance(rt, EvalResult)
    assert rt.per_rollout == m.per_rollout
    assert rt.overall == m.overall


def test_to_json_dict_sanitizes_non_finite(tmp_path):
    from eval.evaluator import export_json
    from eval.metrics import EvalSummary

    m = EvalSummary(
        overall={"final_potential": 1.0, "nan_metric": math.nan},
        per_board=[{"board_id": "b", "final_potential": math.inf}],
    )
    payload = m.to_json_dict()
    assert payload["overall"]["final_potential"] == 1.0
    assert payload["overall"]["nan_metric"] is None
    assert payload["per_board"][0]["final_potential"] is None

    out = tmp_path / "summary.json"
    export_json(m, out)
    loaded = json.loads(out.read_text())  # must be valid JSON (no NaN literals)
    assert loaded["overall"]["nan_metric"] is None


# ---------------------------------------------------------------------------
# Evaluator (rollout_fn injection)
# ---------------------------------------------------------------------------


def test_evaluator_accepts_eval_result_rollout_fn():
    from eval.evaluator import Evaluator
    from methods._shared.board_loader import BoardSpec
    from eval.metrics import EvalSummary

    captured = {}

    def rollout_fn(boards):
        captured["boards"] = boards
        return _eval_result()

    boards = [BoardSpec(7, "board_a", "/tmp/a.kicad_pcb")]
    m = Evaluator(rollout_fn, boards).run()

    assert captured["boards"] == boards
    assert isinstance(m, EvalSummary)
    assert m.overall["final_potential"] == 1.25


def test_evaluator_aggregates_raw_rollout_rows():
    from eval.evaluator import Evaluator
    from methods._shared.board_loader import BoardSpec

    # LLM-style rollout_fn returns raw per-rollout rows; evaluator aggregates.
    rows = [
        {"board_index": 0, "board_id": "b0", "final_potential": 2.0,
         "routability": 1.0, "success": True, "terminated": True},
        {"board_index": 0, "board_id": "b0", "final_potential": 4.0,
         "routability": 0.5, "success": False, "terminated": False},
    ]
    boards = [BoardSpec(0, "b0", "")]
    m = Evaluator(lambda b: rows, boards).run()

    assert m.per_rollout == rows
    # mean aggregation over the two rollouts
    assert m.overall["final_potential"] == pytest.approx(3.0)
    pb = {row["board_id"]: row for row in m.per_board}["b0"]
    assert pb["final_potential"] == pytest.approx(3.0)
    assert pb["success"] == pytest.approx(0.5)  # bool mean


# ---------------------------------------------------------------------------
# Metric / logger split
# ---------------------------------------------------------------------------


def test_log_metrics_emits_via_logger():
    from methods._shared.logger import MetricLogger, log_metrics
    from eval.metrics import EvalSummary

    logger = CapturingLogger()
    assert isinstance(logger, MetricLogger)  # runtime_checkable Protocol

    m = EvalSummary.from_eval_result(_eval_result())
    log_metrics(logger, m, step=42, prefix="val")

    emitted = {(tag, step): value for tag, value, step in logger.scalars}
    assert emitted[("val/fp_mean_of_means", 42)] == 1.25
    assert ("val/final_potential", 42) not in emitted  # alias — logger-dropped
    assert emitted[("val/per_board/board_a/success", 42)] == 1.0
    # nothing non-finite / non-numeric leaked through
    assert all(math.isfinite(v) for _, v, _ in logger.scalars)


# ---------------------------------------------------------------------------
# Mode B: score_boards (post-hoc scoring of saved boards, no rollout)
# ---------------------------------------------------------------------------


def _nested_score(*, success, routability, fp, wl=10.0, via=1, tracks=5):
    """A nested dict shaped like eval.metrics.evaluate_one's output."""
    return {
        "success": success, "clean_pass": success,
        "routability": routability, "track_count": tracks, "via_count": via,
        "wirelength_mm": wl, "drv_errors_only_count": 0,
        "drv_errors_and_promoted_count": 0,
        "track_angle_drv": {"count": 0}, "total_drv_count": 0,
        "final_potential": fp,
        "extras": {"initial_unrouted_edges": 4, "unrouted_edges_remaining": 0},
    }


def test_rows_from_scored_flattens_ok_and_error():
    from eval.evaluator import _rows_from_scored

    nested = [_nested_score(success=True, routability=1.0, fp=3.0),
              {"error": "boom"}]
    pairs = [("/tmp/a.kicad_pcb", None), ("/tmp/b.kicad_pcb", "/tmp/b.kicad_pro")]
    rows = _rows_from_scored(nested, pairs, reward_config="rc", check_angle=45)

    assert rows[0]["board_index"] == 0 and rows[0]["board_id"] == "a"
    assert rows[0]["status"] == "completed" and rows[0]["final_potential"] == 3.0
    assert rows[0]["routed_pcb"] == "/tmp/a.kicad_pcb"
    assert rows[1]["status"] == "crashed" and rows[1]["board_id"] == "b"


def test_score_boards_serial_aggregates(monkeypatch):
    from eval import metrics as metrics_mod
    from eval.evaluator import Evaluator

    scored = {
        "/tmp/a.kicad_pcb": _nested_score(success=True, routability=1.0, fp=4.0),
        "/tmp/b.kicad_pcb": _nested_score(success=False, routability=0.5, fp=2.0),
    }
    monkeypatch.setattr(
        metrics_mod, "evaluate_one",
        lambda pcb, pro, **kw: scored[pcb],
    )

    seen = []
    m = Evaluator.score_boards(
        [("/tmp/a.kicad_pcb", None), ("/tmp/b.kicad_pcb", None)],
        on_result=lambda i, r: seen.append(i),
    )
    assert seen == [0, 1]
    # two distinct boards -> two per_board rows; overall means across boards
    assert len(m.per_board) == 2
    assert m.overall["final_potential"] == pytest.approx(3.0)  # mean(4, 2)
    assert m.overall["success_rate"] == pytest.approx(0.5)
