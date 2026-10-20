"""Incremental DRC == full DRC, on a real violation-rich board.

One test group split across files so pytest-xdist's loadfile scheduler can run
them on separate workers. Comparison convention + shared helpers:
tests/helpers/drc_keying.py.
"""
import pytest

from tests.helpers.drc_keying import BOARD
from tests.helpers.engine import open_engine


@pytest.fixture
def engine():
    e = open_engine(BOARD)
    e.build_connectivity()
    yield e
    e.close()
