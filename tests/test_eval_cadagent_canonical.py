"""The LLM eval (methods/llm_agent/rollout/cadagent.py) folds into the single shared metric vocabulary.

One episode ``result`` dict maps to a canonical ``PER_ROLLOUT_FIELDS`` row, and
aggregating those rows through the shared :class:`eval.metrics.EvalSummary`
yields the unified summary (RL/LLM share one schema), including the merged-in
LLM token fields.

Imports are function-local (see tests/test_evaluator.py) to avoid pulling the
LLM stack at collection time near the pre-existing native double-init landmine.
"""
import math


def _llm_result(*, won, reward, wl, via, tracks, rout, fp, sys_tok, resp_tok):
    """Shape of one entry produced by the cadagent rollout's _run_one_episode."""
    return {
        "board_id": "b0", "env": 0, "steps": 12, "reward": reward,
        "won": won, "drc_violations": 3, "wirelength": wl,
        "via_count": via, "track_count": tracks, "ratsnest_reduction": rout,
        "final_potential": fp,
        "system_tokens": sys_tok, "user_tokens": 0,
        "response_tokens": resp_tok, "call_count": 4,
        "early_stopped": False,
    }


def test_result_maps_to_canonical_row():
    from eval import eval_utils as u
    from methods.llm_agent.rollout.cadagent import _result_to_canonical_row

    r = _llm_result(won=True, reward=1.5, wl=42.0, via=2, tracks=9,
                    rout=1.0, fp=3.25, sys_tok=100, resp_tok=50)
    row = _result_to_canonical_row(r, board_index=7, board_path="/tmp/b.kicad_pcb",
                                   rollout_idx=2)

    # row conforms to the canonical schema (every key is a known field)
    assert set(row).issubset(set(u.PER_ROLLOUT_FIELDS))
    # name reconciliation: reward->episode_return, wirelength->wirelength_mm,
    # won->success/terminated
    assert row["episode_return"] == 1.5
    assert row["wirelength_mm"] == 42.0
    assert row["success"] is True and row["terminated"] is True
    assert row["status"] == "completed"
    assert row["board_index"] == 7 and row["rollout_idx"] == 2
    # tokens carried into the merged vocab
    assert row["system_tokens"] == 100 and row["response_tokens"] == 50


def test_crashed_result_maps_to_status_crashed():
    from methods.llm_agent.rollout.cadagent import _crashed_result, _result_to_canonical_row

    row = _result_to_canonical_row(_crashed_result("b0", 0), board_index=0,
                                   board_path="", rollout_idx=0)
    assert row["status"] == "crashed"
    assert row["crashed"] is True
    assert row["success"] is False


def test_merged_aggregation_includes_tokens():
    from methods.llm_agent.rollout.cadagent import _result_to_canonical_row
    from methods._shared.board_loader import BoardSpec
    from eval.metrics import EvalSummary

    results = [
        _llm_result(won=True, reward=2.0, wl=40.0, via=2, tracks=8,
                    rout=1.0, fp=4.0, sys_tok=100, resp_tok=40),
        _llm_result(won=False, reward=0.0, wl=20.0, via=1, tracks=4,
                    rout=0.5, fp=2.0, sys_tok=200, resp_tok=60),
    ]
    rows = [
        _result_to_canonical_row(r, board_index=0, board_path="", rollout_idx=i)
        for i, r in enumerate(results)
    ]
    metrics = EvalSummary.from_rollouts(rows, [BoardSpec(0, "b0", "")])

    # tokens aggregate through the shared handler (merged schema)
    log = metrics.to_logger_dict(prefix="val")
    assert log["val/per_board/b0/system_tokens"] == 150.0  # mean(100, 200)
    assert log["val/per_board/b0/response_tokens"] == 50.0  # mean(40, 60)
    assert log["val/per_board/b0/episode_return"] == 1.0    # mean(2.0, 0.0)
    assert log["val/per_board/b0/success"] == 0.5           # bool mean
    assert all(math.isfinite(v) for v in log.values())
