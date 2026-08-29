import math

from methods._shared.logger import emit_tensorboard
from eval import eval_utils as u
from eval.metrics import EvalResult


class CapturingWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def test_emit_tensorboard_logs_per_board_numeric_fields_with_same_names():
    writer = CapturingWriter()
    result = EvalResult(
        per_board=[
            {
                "board_id": "board_a",
                "board_index": 7,
                "board_path": "/tmp/board_a.kicad_pcb",
                "aggregation_mode": "mean",
                "fp_mean": 1.25,
                "final_potential": 1.25,
                "success": True,
                "selected_rollout_idx": 3,
                "wirelength_mm": math.nan,
            },
        ],
        overall={
            "fp_mean_of_means": 1.25,
            "final_potential": 1.25,
            "selection_mode": "final_potential",
        },
    )

    emit_tensorboard(result, writer, step=42)

    emitted = {(tag, step): value for tag, value, step in writer.scalars}
    assert emitted[("eval/fp_mean_of_means", 42)] == 1.25
    # bare alias of fp_mean_of_means — dropped from logger emission only
    # (_LOGGER_DUPLICATE_KEYS); per_board keys below are unaffected
    assert ("eval/final_potential", 42) not in emitted
    assert emitted[("eval/per_board/board_a/final_potential", 42)] == 1.25
    assert emitted[("eval/per_board/board_a/success", 42)] == 1.0
    assert u.PER_BOARD_METRIC_FIELDS is u.CADAGENT_VALUE_METRIC_FIELDS
    assert "final_potential" in u.CADAGENT_VALUE_METRIC_FIELDS
    assert ("eval/per_board/board_a/board_index", 42) not in emitted
    assert ("eval/per_board/board_a/selected_rollout_idx", 42) not in emitted
    assert ("eval/per_board/board_a/fp_mean", 42) not in emitted
    assert ("eval/per_board/board_a/wirelength_mm", 42) not in emitted
    assert ("eval/selection_mode", 42) not in emitted
