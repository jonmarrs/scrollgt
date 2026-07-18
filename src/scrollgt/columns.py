"""Column-level scoring for targets with scholar-validated column ground truth
(v0.2, PHerc 1667 merged full-reading geometry).

Unlike the pixel targets, no pixel-level ground truth exists here: the GT is the
published papyrological reading at COLUMN granularity (which columns carry text, per
eight papyrologists' consensus), plus the column layout registered onto the canonical
merged geometry. The scores therefore measure whether a model's output is *consistent
with the reading* — signal concentrated in text columns rather than inter-column
gutters, with text-line periodicity inside columns — and never letter accuracy, which
column-level GT cannot support.

Contract: the prediction is a probability map at the target's GRID resolution (see
meta.json `geometry.grid_shape`), covering either the full grid or a rectangular
sub-extent whose top-left grid coordinate is passed as `origin`. Only columns (and
gutters) fully inside the extent are scored, and they are listed in the scorecard.
Gutters adjacent to a `cross_strip`-flagged column (bracket bbox carries strip-crop
slack) are excluded from the negative set — exclusions are counted in the output.
"""

import json
import os

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from .score import load_probability_map

Image.MAX_IMAGE_PIXELS = None  # the merged-grid valid mask is ~62 Mpx; trusted local file

REPORT_COLS = [
    "col_gutter_auc", "col_gutter_pixel_auc", "line_period_peak_mean",
    "n_text_cols", "n_gutters", "traces_mean_ratio",
]


def load_column_target(target_dir):
    """Load a column target: (meta, columns list, valid mask bool or None)."""
    with open(os.path.join(target_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(target_dir, "columns.json")) as f:
        columns = json.load(f)["columns"]
    mask_path = os.path.join(target_dir, "valid_mask.png")
    valid = None
    if os.path.exists(mask_path):
        valid = np.asarray(Image.open(mask_path).convert("L")) > 127
    return meta, columns, valid


def _extent_contains(origin, shape, y0, y1, x0, x1):
    oy, ox = origin
    return y0 >= oy and x0 >= ox and y1 <= oy + shape[0] and x1 <= ox + shape[1]


def _region_pixels(prob, valid, origin, y0, y1, x0, x1):
    """Valid-masked prediction pixels of a grid-coordinate box, or None if empty."""
    oy, ox = origin
    sub = prob[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    if valid is not None:
        v = valid[y0:y1, x0:x1]
        sub = sub[v]
    else:
        sub = sub.ravel()
    return sub if sub.size else None

def _line_period_peak(prob, valid, origin, y0, y1, x0, x1, pitch_lo, pitch_hi):
    """Peak normalized autocorrelation of the column's row profile over the plausible
    line-pitch lag range. ~0 for constant/aperiodic signal, →1 for strong line
    structure at some pitch in range."""
    oy, ox = origin
    sub = prob[y0 - oy:y1 - oy, x0 - ox:x1 - ox].astype(np.float64)
    if valid is not None:
        v = valid[y0:y1, x0:x1]
        with np.errstate(invalid="ignore"):
            profile = np.where(v.any(axis=1), (sub * v).sum(axis=1) / v.sum(axis=1), np.nan)
    else:
        profile = sub.mean(axis=1)
    good = np.isfinite(profile)
    if good.sum() < 3 * pitch_lo:
        return float("nan")
    p = np.where(good, profile, np.nanmean(profile))
    p = p - p.mean()
    denom = float((p * p).sum())
    if denom <= 0:
        return 0.0
    best = 0.0
    for lag in range(pitch_lo, min(pitch_hi + 1, len(p) - 1)):
        r = float((p[:-lag] * p[lag:]).sum()) / denom
        best = max(best, r)
    return best


def score_columns(pred_path, target_dir, origin=(0, 0)):
    """Score a prediction against a column-level target. Returns the scorecard."""
    meta, columns, valid = load_column_target(target_dir)
    prob = load_probability_map(pred_path)
    pitch_lo, pitch_hi = meta.get("line_pitch_range", [60, 220])

    scored = []          # columns fully inside the extent, with region stats
    for c in columns:
        band = c.get("text_band")
        if band is None:
            continue
        y0, y1 = band
        x0, x1 = c["gx0"], c["gx1"]
        if not _extent_contains(origin, prob.shape, y0, y1, x0, x1):
            continue
        px = _region_pixels(prob, valid, origin, y0, y1, x0, x1)
        if px is None:
            continue
        peak = (_line_period_peak(prob, valid, origin, y0, y1, x0, x1,
                                  pitch_lo, pitch_hi)
                if c["transcription"] == "text" else None)
        scored.append({"col": c["col"], "status": c["transcription"],
                       "cross_strip": bool(c.get("cross_strip")),
                       "mean": float(px.mean()), "pixels": px,
                       "line_period_peak": peak,
                       "y": (y0, y1), "x": (x0, x1)})

    # gutters between consecutive scored columns; skip those touching flagged columns
    gutters, excluded = [], 0
    for a, b in zip(scored, scored[1:]):
        if b["x"][0] - a["x"][1] < 2:
            continue
        if a["cross_strip"] or b["cross_strip"]:
            excluded += 1
            continue
        y0 = max(a["y"][0], b["y"][0])
        y1 = min(a["y"][1], b["y"][1])
        if y1 - y0 < 8:
            continue
        px = _region_pixels(prob, valid, origin, y0, y1, a["x"][1], b["x"][0])
        if px is None:
            continue
        gutters.append({"between": (a["col"], b["col"]),
                        "mean": float(px.mean()), "pixels": px})

    text_cols = [c for c in scored if c["status"] == "text"]
    traces_cols = [c for c in scored if c["status"] == "traces"]

    metrics = {
        "cols_scored": [c["col"] for c in scored],
        "n_text_cols": len(text_cols),
        "n_traces_cols": len(traces_cols),
        "n_gutters": len(gutters),
        "excluded_gutters": excluded,
    }
    if text_cols and gutters:
        y = [1] * len(text_cols) + [0] * len(gutters)
        s = [c["mean"] for c in text_cols] + [g["mean"] for g in gutters]
        metrics["col_gutter_auc"] = (0.5 if len(set(s)) == 1
                                     else float(roc_auc_score(y, s)))
        ypx = np.concatenate([np.ones(c["pixels"].size, np.uint8) for c in text_cols]
                             + [np.zeros(g["pixels"].size, np.uint8) for g in gutters])
        spx = np.concatenate([c["pixels"] for c in text_cols]
                             + [g["pixels"] for g in gutters])
        metrics["col_gutter_pixel_auc"] = (0.5 if np.all(spx == spx[0])
                                           else float(roc_auc_score(ypx, spx)))
    else:
        metrics["col_gutter_auc"] = float("nan")
        metrics["col_gutter_pixel_auc"] = float("nan")
        metrics["note"] = "extent contains too few columns/gutters to score"

    peaks = [c["line_period_peak"] for c in text_cols
             if c["line_period_peak"] is not None and np.isfinite(c["line_period_peak"])]
    metrics["line_period_peak_mean"] = float(np.mean(peaks)) if peaks else float("nan")
    if text_cols and traces_cols:
        tmean = float(np.mean([c["mean"] for c in text_cols]))
        metrics["traces_mean_ratio"] = (float(np.mean([c["mean"] for c in traces_cols]))
                                        / tmean if tmean > 0 else float("nan"))

    per_col = [{k: c[k] for k in ("col", "status", "mean", "line_period_peak")}
               for c in scored]
    return {
        "target": meta.get("target_id", os.path.basename(os.path.normpath(target_dir))),
        "prediction": os.path.basename(pred_path),
        "granularity": "column (no pixel GT exists for this target — see meta.json)",
        "metrics": metrics,
        "per_column": per_col,
        "gutters": [{"between": g["between"], "mean": g["mean"]} for g in gutters],
    }
