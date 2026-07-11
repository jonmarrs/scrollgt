"""Score a prediction against a ScrollGT registered-ground-truth target.

Contract: a prediction is a probability map over EXACTLY the target's region
(see the target's meta.json: SOTA surface volume, pyramid level, y0/x0/size).
Accepted formats: 8-bit PNG (interpreted as prob = pixel/255) or a .npy float
array in [0, 1]. The map must match the ground-truth tile's height/width.
"""

import json
import os

import numpy as np
from PIL import Image

from .metrics import segmentation_metrics

Image.MAX_IMAGE_PIXELS = None  # benchmark tiles are 4096x4096; trusted local files

REPORT_COLS = [
    "val_f1", "f1_at_0.5", "average_precision", "ap_prevalence_lift",
    "precision", "recall", "positive_rate", "roc_auc",
]


def load_probability_map(path):
    """Load a prediction as float64 probabilities in [0, 1]."""
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float64)
        if arr.ndim != 2:
            raise ValueError(f"prediction .npy must be 2-D, got shape {arr.shape}")
        if np.nanmin(arr) < -1e-6 or np.nanmax(arr) > 1 + 1e-6:
            raise ValueError(
                f"prediction values outside [0,1]: min={np.nanmin(arr)}, max={np.nanmax(arr)}"
            )
        return np.clip(arr, 0.0, 1.0)
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return arr / 255.0


def load_target(target_dir):
    """Load a benchmark target: (gt binary uint8, valid mask bool, meta dict)."""
    meta_path = os.path.join(target_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    gt = np.asarray(Image.open(os.path.join(target_dir, "gt_ink.png")).convert("L"))
    gt_bin = (gt > 127).astype(np.uint8)
    mask_path = os.path.join(target_dir, "mask.png")
    if os.path.exists(mask_path):
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    else:
        mask = np.ones_like(gt_bin, dtype=bool)
    return gt_bin, mask, meta


def score_prediction(pred_path, target_dir):
    """Score one prediction file against one target directory. Returns the scorecard."""
    gt, mask, meta = load_target(target_dir)
    prob = load_probability_map(pred_path)
    if prob.shape != gt.shape:
        raise ValueError(
            f"prediction shape {prob.shape} != ground-truth shape {gt.shape}; "
            f"predict exactly the region in meta.json: {meta.get('region', {})}"
        )
    card = segmentation_metrics(prob, gt, mask)
    card.pop("metrics_by_threshold", None)
    return {
        "target": meta.get("target_id", os.path.basename(os.path.normpath(target_dir))),
        "prediction": os.path.basename(pred_path),
        "registration": {
            "median_residual_voxels": meta.get("registration", {}).get("median_residual"),
            "validation": meta.get("registration", {}).get("validation_basis"),
        },
        "metrics": card,
    }


def format_row(name, card):
    cells = " | ".join(
        f"{card.get(c, float('nan')):.4f}" if isinstance(card.get(c), (int, float))
        else "n/a"
        for c in REPORT_COLS
    )
    return f"| {name} | {cells} |"


def markdown_report(results):
    lines = [
        "| model / target | " + " | ".join(REPORT_COLS) + " |",
        "|---|" + "|".join(["---"] * len(REPORT_COLS)) + "|",
    ]
    for r in results:
        lines.append(format_row(f"{r['prediction']} vs {r['target']}", r["metrics"]))
    return "\n".join(lines)
