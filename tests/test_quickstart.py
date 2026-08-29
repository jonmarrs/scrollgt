"""The documented quickstart must actually run.

The README claimed "cold clone to a scored fiber cube is about a minute" and
showed three commands. The first two worked. The third could not: it named
`labels.npy`, a file the reader supplies, and the text never said so, so a cold
reader's first command produced a numpy FileNotFoundError traceback.

Clone and install were fine. Nobody had ever run the third line from a clean
checkout, which is the one place a prospective user is guaranteed to be.

These tests execute the documented invocation through the CLI, so the front door
cannot rot again without a failure.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scrollgt.cli import main  # noqa: E402

TARGET = os.path.join(ROOT, "data", "fibers_s1_00497_01497_03997_256")


@pytest.mark.skipif(not os.path.isdir(TARGET), reason="fiber target not present")
def test_quickstart_command_runs_with_no_user_inputs(capsys):
    """Exactly the command the README quickstart shows."""
    rc = main(["score-fibers", "--floor", "connected_components", TARGET])
    assert rc == 0
    out = capsys.readouterr().out
    assert "your labelling" in out
    assert "floor: connected components" in out


@pytest.mark.skipif(not os.path.isdir(TARGET), reason="fiber target not present")
def test_missing_prediction_explains_itself_instead_of_tracing_back(capsys):
    """A cold reader who runs the file form without a file gets a sentence, not a
    numpy stack trace, and is told about the zero-input form."""
    with pytest.raises(FileNotFoundError) as e:
        main(["score-fibers", "labels.npy", TARGET])
    msg = str(e.value)
    assert "score-fibers scores YOUR tracer" in msg
    assert "--floor connected_components" in msg


@pytest.mark.skipif(not os.path.isdir(TARGET), reason="fiber target not present")
def test_prediction_and_floor_are_mutually_exclusive():
    """Both, or neither, is a user error worth naming rather than resolving silently."""
    with pytest.raises(SystemExit):
        main(["score-fibers", "labels.npy", "--floor", "connected_components", TARGET])
    with pytest.raises(SystemExit):
        main(["score-fibers", TARGET])
