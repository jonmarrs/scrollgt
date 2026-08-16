"""The scorecard contract: both ERL variants, tolerance, floors, and the flag."""

import json
import pathlib

import numpy as np
import pytest

from scrollgt.cli import main
from scrollgt.fibers.report import fiber_markdown_report

TARGET = "data/fibers_s1_00497_01497_03997_256"


@pytest.fixture
def cc_labels(tmp_path):
    """A real entry: the connected-components labelling of the shipped mask."""
    from scrollgt.fibers import floor_connected_components
    from scrollgt.fibers.target import load_fiber_target

    _, mask, _ = load_fiber_target(TARGET)
    p = tmp_path / "cc.npy"
    np.save(p, floor_connected_components(mask).astype(np.int32))
    return p


def test_report_prints_both_erl_variants_and_tolerance(cc_labels):
    from scrollgt.fibers.target import score_fiber_prediction

    text = fiber_markdown_report(score_fiber_prediction(cc_labels, TARGET))
    assert "ERL" in text
    assert "ERLpen" in text, "merge-penalized ERL must never be omitted"
    assert "tolerance" in text.lower()
    assert "splits" in text.lower() and "merges" in text.lower()


def _meta(target):
    return json.loads((pathlib.Path(target) / "meta.json").read_text())


def test_report_prints_the_size_class_and_its_oracle_ceiling(cc_labels):
    """The promise is that a score never appears without the ceiling for its own class.

    The card has carried `size_class` and `class_oracle_erl` since size class became
    first-class, but the markdown report rendered neither, so the guarantee held only for
    `--json-out` readers. The terminal reader is exactly the one who would otherwise
    compare a 512 cube's ERL against a 256 cube's ceiling by eye.
    """
    from scrollgt.fibers.target import score_fiber_prediction

    text = fiber_markdown_report(score_fiber_prediction(cc_labels, TARGET))
    assert "size class 256" in text
    oracle = _meta(TARGET)["floors"]["oracle"]["erl"]
    assert f"{oracle:.2f}" in text, "the class ceiling must be printed, not just carried"
    assert "does not compare across size classes" in text


def test_cli_json_carries_the_size_class_and_ceiling(tmp_path, cc_labels):
    """Neither field was asserted in the JSON either, in the CLI's own test file."""
    out = tmp_path / "card.json"
    main(["score-fibers", str(cc_labels), TARGET, "--json-out", str(out)])
    card = json.loads(out.read_text())
    meta = _meta(TARGET)
    assert card["size_class"] == meta["size_class"] == 256
    assert card["class_oracle_erl"] == pytest.approx(meta["floors"]["oracle"]["erl"])


def test_cli_stdout_prints_the_class_line(cc_labels, capsys):
    """The report reaches stdout, not just the return value.

    The 512 side of this -- that a 512 card renders a 512 ceiling -- is covered by
    test_fiber_size_class.py::test_a_512_target_reports_a_512_ceiling, which scores a real
    512 cube. It is not repeated here: scoring a 512 cube costs ~45 s and the branch under
    test is the same one.
    """
    main(["score-fibers", str(cc_labels), TARGET])
    out = capsys.readouterr().out
    assert "size class 256" in out
    assert f"{_meta(TARGET)['floors']['oracle']['erl']:.2f}" in out


def test_report_lists_all_four_floors(cc_labels):
    from scrollgt.fibers.target import score_fiber_prediction

    text = fiber_markdown_report(score_fiber_prediction(cc_labels, TARGET))
    for floor in ("one instance for everything", "connected components",
                  "one instance per voxel", "50 random"):
        assert floor in text, floor


def test_cli_writes_json(tmp_path, cc_labels, capsys):
    out = tmp_path / "card.json"
    main(["score-fibers", str(cc_labels), TARGET, "--json-out", str(out)])
    card = json.loads(out.read_text())
    assert card["tolerance"] == 2.0
    assert "erl" in card["metrics"] and "erl_merge_penalized" in card["metrics"]
    assert set(card["floors"]) == {
        "floor_single_instance", "floor_connected_components",
        "floor_voxel_instances", "floor_random_instances",
    }


def test_report_explains_why_coverage_and_precision_are_identical(cc_labels):
    """A reader seeing four identical coverage values must not read it as a bug."""
    from scrollgt.fibers.target import score_fiber_prediction

    text = fiber_markdown_report(score_fiber_prediction(cc_labels, TARGET))
    assert "That is the point, not a bug" in text
    assert "cannot rank a tracer" in text


def test_report_flags_an_entry_that_trails_the_baseline(tmp_path):
    """The BELOW-baseline warning is the card's sharpest honest signal.

    The other tests all score connected components against itself, so the flag is
    never set and this rendering branch would otherwise go unexercised.
    """
    from scrollgt.fibers import floor_voxel_instances
    from scrollgt.fibers.target import load_fiber_target, score_fiber_prediction

    _, mask, _ = load_fiber_target(TARGET)
    p = tmp_path / "voxels.npy"
    np.save(p, floor_voxel_instances(mask).astype(np.int32))

    card = score_fiber_prediction(p, TARGET)
    assert card["below_baseline"] is True

    text = fiber_markdown_report(card)
    assert "BELOW the naive baseline" in text
    cc = card["floors"]["floor_connected_components"]["erl"]
    assert f"{cc:.2f}" in text, "the card must name the baseline it trails"


def test_cli_reports_cross_scroll_split(tmp_path, capsys):
    from scrollgt.fibers import floor_connected_components
    from scrollgt.fibers.target import load_fiber_target

    target = "data/fibers_s5_03997_01497_03997_256"
    _, mask, _ = load_fiber_target(target)
    p = tmp_path / "cc5.npy"
    np.save(p, floor_connected_components(mask).astype(np.int32))
    main(["score-fibers", str(p), target])
    assert "cross-scroll" in capsys.readouterr().out.lower()
