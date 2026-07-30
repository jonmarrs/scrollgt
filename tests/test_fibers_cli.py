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


def test_cli_reports_cross_scroll_split(tmp_path, capsys):
    from scrollgt.fibers import floor_connected_components
    from scrollgt.fibers.target import load_fiber_target

    target = "data/fibers_s5_03997_01497_03997_256"
    _, mask, _ = load_fiber_target(target)
    p = tmp_path / "cc5.npy"
    np.save(p, floor_connected_components(mask).astype(np.int32))
    main(["score-fibers", str(p), target])
    assert "cross-scroll" in capsys.readouterr().out.lower()
