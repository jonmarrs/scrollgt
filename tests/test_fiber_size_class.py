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


def _card_for(target, tmp_path):
    """Score an all-background labelling against `target`; return (card, meta).

    A prediction of all-background is a legitimate input (it labels nothing), so this
    reaches `score_fiber_prediction` without needing a real tracer output.
    """
    import numpy as np

    from scrollgt.fibers.target import load_fiber_target, score_fiber_prediction

    _, mask, meta = load_fiber_target(str(target))
    pred = tmp_path / "empty.npy"
    np.save(pred, np.zeros(mask.shape, dtype=np.int32))
    return score_fiber_prediction(str(pred), str(target)), meta


def test_scorecard_reports_the_class_and_its_oracle(tmp_path):
    """Exercise the scorer, not just the metadata.

    What is pinned is that the card carries the class and its ceiling -- assert on `meta`
    alone and the scorer could stop emitting either without a test noticing.

    `class_oracle_erl` is read off the card through a triple `.get` chain, so the earlier
    `assert "class_oracle_erl" in card` passed even if that chain yielded `None` for every
    shipped target. It is compared against the published oracle instead.
    """
    target = FIBER_TARGETS[0]
    card, meta = _card_for(target, tmp_path)
    assert card["size_class"] == meta["size_class"]
    assert card["class_oracle_erl"] == pytest.approx(meta["floors"]["oracle"]["erl"]), (
        "a score is unreadable without its class ceiling, and a None ceiling is no ceiling"
    )


def test_a_512_target_reports_a_512_ceiling(tmp_path):
    """The whole point of size classes, exercised on the class that motivated them.

    Every other scorer test here and in test_fiber_target.py runs against
    `FIBER_TARGETS[0]`, a 256 cube, so until now nothing scored a 512 target end to end in
    the suite whose stated purpose is that 512 cards report a 512 ceiling.

    Scoring a 512 cube costs ~45 s, so this deliberately does the card, the rendered
    report, and the never-average refusal in one scoring pass rather than three separate
    tests. (`aggregate_fiber_scores` also has cheap dedicated coverage in
    test_fiber_gaming.py; what is added here is a *real* 512 card on the mixed input.)
    """
    from scrollgt.fibers.report import fiber_markdown_report
    from scrollgt.fibers.target import aggregate_fiber_scores

    target = next((t for t in FIBER_TARGETS if t.name.endswith("_512")), None)
    assert target is not None, "no 512 target shipped; this test would vacuously pass"

    card, meta = _card_for(target, tmp_path)
    assert card["size_class"] == 512
    assert meta["shape"] == [512, 512, 512]
    assert card["class_oracle_erl"] == pytest.approx(meta["floors"]["oracle"]["erl"])
    # Guards a card that reports class 512 while carrying a 256 cube's ceiling: shipped
    # 512 oracles are 497-513, shipped 256 oracles 222-262.
    assert card["class_oracle_erl"] > 400.0

    text = fiber_markdown_report(card)
    assert "size class 512" in text
    assert f"{float(card['class_oracle_erl']):.2f}" in text

    small = {**card, "size_class": 256}
    with pytest.raises(ValueError, match="does not compare between cube sizes"):
        aggregate_fiber_scores([card, small])
    assert aggregate_fiber_scores([card])["size_class"] == 512
