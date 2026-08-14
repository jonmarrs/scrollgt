import json
import os

import numpy as np
import pytest
from PIL import Image

from scrollgt.score import load_probability_map, score_prediction


def _make_target(tmp_path, size=64, pos_rows=24):
    tdir = tmp_path / "target"
    tdir.mkdir()
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[:pos_rows, :] = 255
    Image.fromarray(gt).save(tdir / "gt_ink.png")
    with open(tdir / "meta.json", "w") as f:
        json.dump({
            "target_id": "test_target",
            "region": {"level": 2, "y0": 0, "x0": 0, "size": size},
            "registration": {"median_residual": 7.9, "validation_basis": "test"},
        }, f)
    return tdir, gt


def test_score_png_prediction(tmp_path):
    tdir, gt = _make_target(tmp_path)
    pred = (gt.astype(np.float64) * 0.9 + 12).astype(np.uint8)  # near-perfect
    ppath = tmp_path / "pred.png"
    Image.fromarray(pred).save(ppath)
    result = score_prediction(str(ppath), str(tdir))
    assert result["target"] == "test_target"
    assert result["metrics"]["val_f1"] > 0.99
    assert result["metrics"]["roc_auc"] > 0.99


def test_score_npy_prediction(tmp_path):
    tdir, gt = _make_target(tmp_path)
    prob = gt.astype(np.float64) / 255.0 * 0.9 + 0.05
    ppath = tmp_path / "pred.npy"
    np.save(ppath, prob)
    result = score_prediction(str(ppath), str(tdir))
    assert result["metrics"]["val_f1"] > 0.99


def test_shape_mismatch_rejected(tmp_path):
    tdir, _ = _make_target(tmp_path, size=64)
    ppath = tmp_path / "pred.npy"
    np.save(ppath, np.zeros((32, 32)))
    with pytest.raises(ValueError, match="shape"):
        score_prediction(str(ppath), str(tdir))


def test_npy_out_of_range_rejected(tmp_path):
    ppath = tmp_path / "pred.npy"
    np.save(ppath, np.full((8, 8), 3.0))
    with pytest.raises(ValueError, match="outside"):
        load_probability_map(str(ppath))


def test_mask_respected(tmp_path):
    tdir, gt = _make_target(tmp_path)
    # valid mask covers only the left half; make the right half adversarial
    mask = np.zeros_like(gt)
    mask[:, :32] = 255
    Image.fromarray(mask).save(os.path.join(tdir, "mask.png"))
    prob = gt.astype(np.float64) / 255.0 * 0.9 + 0.05
    prob[:, 32:] = 1.0 - prob[:, 32:]  # inverted (wrong) outside the mask
    ppath = tmp_path / "pred.npy"
    np.save(ppath, prob)
    result = score_prediction(str(ppath), str(tdir))
    assert result["metrics"]["val_f1"] > 0.99  # adversarial half excluded by mask


# --- placement refusal (2026-08) -------------------------------------------------------
# A target whose ground truth is misplaced must not be silently scoreable: the number would
# measure misalignment rather than reading. See the upstream report
# reports/detector/registration_offset_2026-08-07.md.

def _target_with_placement(tmp_path, passed, size=64):
    import json as _json
    import numpy as _np
    from PIL import Image as _Image
    d = tmp_path / "tgt"
    d.mkdir()
    gt = _np.zeros((size, size), _np.uint8)
    gt[16:48, 16:48] = 255
    _Image.fromarray(gt).save(d / "gt_ink.png")
    _json.dump({
        "target_id": "unit_test_target",
        "registration": {"median_residual": 7.9, "placement": {
            "offset_level2_px": 53.3, "offset_mm": 0.51,
            "gate_threshold_level2_px": 48.0, "passed": passed, "note": "unit test"}},
    }, open(d / "meta.json", "w"))
    return d


def test_score_refuses_target_failing_placement(tmp_path):
    from scrollgt.score import score_prediction
    d = _target_with_placement(tmp_path, passed=False)
    pred = tmp_path / "pred.png"
    Image.fromarray(np.full((64, 64), 200, np.uint8)).save(pred)
    with pytest.raises(ValueError, match="FAILS its placement check"):
        score_prediction(str(pred), str(d))


def test_score_allows_failing_placement_when_opted_in(tmp_path):
    from scrollgt.score import score_prediction
    d = _target_with_placement(tmp_path, passed=False)
    pred = tmp_path / "pred.png"
    Image.fromarray(np.full((64, 64), 200, np.uint8)).save(pred)
    card = score_prediction(str(pred), str(d), allow_failing_placement=True)
    assert card["registration"]["placement_passed"] is False
    assert "metrics" in card


def test_score_surfaces_placement_on_passing_target(tmp_path):
    """Placement must appear in every scorecard, not just failing ones."""
    from scrollgt.score import score_prediction
    d = _target_with_placement(tmp_path, passed=True)
    pred = tmp_path / "pred.png"
    Image.fromarray(np.full((64, 64), 200, np.uint8)).save(pred)
    card = score_prediction(str(pred), str(d))
    assert card["registration"]["placement_passed"] is True
    assert card["registration"]["placement_mm"] == 0.51


# --- non-scoring targets (2026-08-14) ---------------------------------------------------
# Distinct from a placement failure: a target can pass its gates and still be a poor thing
# to evaluate against. Both 20230702185753 regions are marked non-scoring because local
# placement error there reaches ~1.9x the 512um prize analysis window, so a model can be
# scored against ground truth from a different part of the sheet.

def _target(tmp_path, scoring=None, placement_passed=True, size=64):
    import json as _json
    import numpy as _np
    from PIL import Image as _Image
    d = tmp_path / "tgt"
    d.mkdir()
    gt = _np.zeros((size, size), _np.uint8)
    gt[16:48, 16:48] = 255
    _Image.fromarray(gt).save(d / "gt_ink.png")
    meta = {"target_id": "unit_test_target",
            "registration": {"placement": {
                "offset_level2_px": 46.6, "offset_mm": 0.45,
                "gate_threshold_level2_px": 48.0, "passed": placement_passed,
                "note": "unit test"}}}
    if scoring is not None:
        meta["scoring"] = scoring
    _json.dump(meta, open(d / "meta.json", "w"))
    return d


def _pred(tmp_path):
    p = tmp_path / "pred.png"
    Image.fromarray(np.full((64, 64), 200, np.uint8)).save(p)
    return str(p)


def test_non_scoring_target_is_refused_even_though_placement_passes(tmp_path):
    d = _target(tmp_path, scoring={"enabled": False, "reason": "local error ~1.9 windows"})
    with pytest.raises(ValueError, match="NON-SCORING"):
        score_prediction(_pred(tmp_path), str(d))


def test_non_scoring_refusal_states_the_reason(tmp_path):
    d = _target(tmp_path, scoring={"enabled": False, "reason": "SENTINEL-REASON"})
    with pytest.raises(ValueError, match="SENTINEL-REASON"):
        score_prediction(_pred(tmp_path), str(d))


def test_non_scoring_can_be_overridden_to_reproduce_the_record(tmp_path):
    d = _target(tmp_path, scoring={"enabled": False, "reason": "r"})
    card = score_prediction(_pred(tmp_path), str(d), allow_non_scoring=True)
    assert card["scoring_enabled"] is False
    assert "metrics" in card


def test_the_two_refusals_are_independent(tmp_path):
    """A non-scoring target that ALSO fails placement needs both overrides."""
    d = _target(tmp_path, scoring={"enabled": False, "reason": "r"}, placement_passed=False)
    with pytest.raises(ValueError, match="NON-SCORING"):
        score_prediction(_pred(tmp_path), str(d))
    with pytest.raises(ValueError, match="FAILS its placement check"):
        score_prediction(_pred(tmp_path), str(d), allow_non_scoring=True)
    card = score_prediction(_pred(tmp_path), str(d),
                            allow_non_scoring=True, allow_failing_placement=True)
    assert card["scoring_enabled"] is False


def test_targets_without_a_scoring_block_still_score(tmp_path):
    """Absence of the flag must mean scoreable, not refused."""
    d = _target(tmp_path)
    card = score_prediction(_pred(tmp_path), str(d))
    assert card["scoring_enabled"] is True


def test_shipped_targets_declare_their_scoring_status_consistently():
    """Every shipped pixel target's flag must match what BASELINES says about it."""
    import json as _json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data"
    expected = {"scroll1_20230702185753": False,
                "scroll1_20230702185753_y7000_x4000": False,
                "scroll1_20231210121321": True}
    for name, should_score in expected.items():
        meta = _json.loads((root / name / "meta.json").read_text())
        got = (meta.get("scoring") or {}).get("enabled", True)
        assert got is should_score, f"{name}: scoring.enabled={got}, expected {should_score}"
