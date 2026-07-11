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
