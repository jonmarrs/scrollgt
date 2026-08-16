"""A fiber score is only readable against its own class ceiling.

ERL is expected run length in voxels (`_erl(runs) = sum r^2 / sum r`), so a 512-cube
admits longer fibers and scores higher for geometric reasons alone. Every scorecard
therefore states its size class and that class's oracle, so a number is never read
against the wrong ceiling.
"""

import json
import pathlib

import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
FIBER_TARGETS = sorted(DATA.glob("fibers_*"))


def test_there_are_fiber_targets_to_check():
    assert FIBER_TARGETS, "no fiber targets found; this test would vacuously pass"


@pytest.mark.parametrize("target", FIBER_TARGETS, ids=lambda p: p.name)
def test_every_fiber_target_declares_its_size_class(target):
    meta = json.loads((target / "meta.json").read_text())
    assert "size_class" in meta, f"{target.name} has no size_class"
    assert meta["size_class"] in (256, 512)
    # The class must agree with the cube's own declared shape, not just be present.
    assert meta["shape"][0] == meta["size_class"]


def test_scorecard_reports_the_class_and_its_oracle(tmp_path):
    """Exercise the scorer, not just the metadata.

    A prediction of all-background is a legitimate input (it labels nothing), so this
    reaches `score_fiber_prediction` without needing a real tracer output. What is being
    pinned is that the card carries the class and its ceiling -- assert on `meta` alone and
    the scorer could stop emitting either without a test noticing.
    """
    import numpy as np

    from scrollgt.fibers.target import load_fiber_target, score_fiber_prediction

    target = FIBER_TARGETS[0]
    _, mask, meta = load_fiber_target(str(target))

    pred = tmp_path / "empty.npy"
    np.save(pred, np.zeros(mask.shape, dtype=np.int32))

    card = score_fiber_prediction(str(pred), str(target))
    assert card["size_class"] == meta["size_class"]
    assert "class_oracle_erl" in card, "a score is unreadable without its class ceiling"
