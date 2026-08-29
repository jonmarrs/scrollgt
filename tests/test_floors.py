"""Floors are properties, not observations, so they get enforced rather than noted.

This repository's recorded failure mode is not wrong metric code. It is a property
measured once, written into a README, and never re-checked, which is exactly how
"the all-positive floor sits at 0.518" survived: a genuinely all-positive
prediction is constant, and a constant predictor's ROC-AUC is 0.5 by definition,
so that number could never have been an all-positive floor.

These tests pin the floors that are true by construction. If one ever moves, the
scorer changed underneath them.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from scrollgt.columns import column_floors, load_column_target  # noqa: E402
from scrollgt.score import ink_floors, load_target  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INK_TARGET = os.path.join(DATA, "scroll1_20231210121321")
COL_TARGET = os.path.join(DATA, "pherc1667_merged_columns")


@pytest.mark.skipif(not os.path.isdir(INK_TARGET), reason="ink target not present")
def test_constant_ink_predictors_score_exactly_chance():
    """A constant predictor cannot separate anything, at any prevalence.

    Both all-positive and constant-0.5 are constant, so both must be exactly 0.5.
    This is the assertion the README's 0.518 violated.
    """
    gt, mask, _ = load_target(INK_TARGET)
    f = ink_floors(gt, mask)
    assert f["floor_all_positive"]["roc_auc"] == pytest.approx(0.5, abs=1e-9)
    assert f["floor_constant"]["roc_auc"] == pytest.approx(0.5, abs=1e-9)


@pytest.mark.skipif(not os.path.isdir(INK_TARGET), reason="ink target not present")
def test_ink_floor_ap_equals_prevalence_so_lift_is_one():
    """Average precision of a constant predictor is the positive rate, so the
    prevalence lift is 1.0. A lift above 1 is the thing a real model must show."""
    gt, mask, _ = load_target(INK_TARGET)
    f = ink_floors(gt, mask)
    for name in ("floor_all_positive", "floor_constant"):
        assert f[name]["ap_prevalence_lift"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not os.path.isdir(INK_TARGET), reason="ink target not present")
def test_ink_floors_report_a_nontrivial_f1():
    """The floor that actually matters for F1. An all-positive prediction earns
    2p/(1+p), which is far from zero and has been misread as skill in this
    project's history."""
    gt, mask, _ = load_target(INK_TARGET)
    f1 = ink_floors(gt, mask)["floor_all_positive"]["val_f1"]
    assert 0.0 < f1 < 1.0
    sel = np.asarray(mask).astype(bool)
    p = float((np.asarray(gt)[sel] > 0.5).mean())
    assert f1 == pytest.approx(2 * p / (1 + p), rel=0.02)


@pytest.mark.skipif(not os.path.isdir(COL_TARGET), reason="column target not present")
def test_papyrus_mask_scores_chance_because_gutters_are_papyrus_too():
    """The floor that makes the column family honest: predicting the valid mask
    looks like a real prediction and separates nothing, because a gutter is as
    much papyrus as a column."""
    meta, cols, valid = load_column_target(COL_TARGET)
    f = column_floors(meta, cols, valid)
    assert f["floor_papyrus_mask"]["col_gutter_auc"] == pytest.approx(0.5, abs=1e-9)
    assert f["floor_constant"]["col_gutter_auc"] == pytest.approx(0.5, abs=1e-9)


@pytest.mark.skipif(not os.path.isdir(COL_TARGET), reason="column target not present")
def test_column_floors_go_through_the_submission_code_path():
    """Floors must be scored by the same function a real submission uses, or they
    can drift from it silently. Asserted by shape of the result rather than by
    inspection: every floor carries the same metric keys a scored prediction does."""
    meta, cols, valid = load_column_target(COL_TARGET)
    f = column_floors(meta, cols, valid)
    for name, row in f.items():
        assert "col_gutter_auc" in row, name
        assert "col_gutter_pixel_auc" in row, name
        assert row["what"], name
