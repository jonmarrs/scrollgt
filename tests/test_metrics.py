import numpy as np

from scrollgt.metrics import segmentation_metrics


def test_perfect_prediction():
    label = np.zeros((32, 32), dtype=np.uint8)
    label[:16, :] = 1
    prob = label.astype(np.float64) * 0.98 + 0.01
    m = segmentation_metrics(prob, label, np.ones_like(label, dtype=bool))
    assert m["val_f1"] > 0.99
    assert m["roc_auc"] > 0.99
    assert m["ap_prevalence_lift"] > 1.5


def test_constant_prediction_has_no_lift():
    # The honest-metrics signature: an all-positive predictor gets the trivial F1
    # (2p/(1+p)) but NO prevalence lift and chance ROC.
    rng = np.random.default_rng(0)
    label = (rng.random((64, 64)) < 0.2).astype(np.uint8)
    prob = np.full((64, 64), 0.9)
    m = segmentation_metrics(prob, label, np.ones_like(label, dtype=bool))
    p = m["positive_rate"]
    assert abs(m["val_f1"] - (2 * p / (1 + p))) < 1e-6
    assert m["ap_prevalence_lift"] < 1.05
    assert abs(m["roc_auc"] - 0.5) < 1e-6


def test_degenerate_all_negative_is_nan():
    label = np.zeros((16, 16), dtype=np.uint8)
    prob = np.full((16, 16), 0.3)
    m = segmentation_metrics(prob, label, np.ones_like(label, dtype=bool))
    assert np.isnan(m["val_f1"])
    assert "degenerate" in m["note"]


def test_mask_restriction():
    # Signal only inside the mask; garbage outside must not affect the score.
    label = np.zeros((32, 32), dtype=np.uint8)
    label[:8, :16] = 1
    prob = label.astype(np.float64) * 0.9 + 0.05
    prob[:, 16:] = 0.99  # garbage outside mask
    label[:, 16:] = 0
    mask = np.zeros((32, 32), dtype=bool)
    mask[:, :16] = True
    m = segmentation_metrics(prob, label, mask)
    assert m["val_f1"] > 0.99
    assert m["n_pixels"] == 32 * 16
