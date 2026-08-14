"""Tests for the column-level scoring contract (v0.2, PHerc 1667 merged geometry)."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scrollgt.columns import score_columns  # noqa: E402


def _make_target(tmp_path, grid_h=200, grid_w=1200):
    """Synthetic column target: 4 columns (first is 'traces'), pitch range 10-40."""
    d = tmp_path / "toy_columns"
    d.mkdir()
    cols = []
    status = ["traces", "text", "text", "text"]
    for i in range(4):
        gx0 = 60 + i * 300
        cols.append({
            "col": i + 1, "gx0": gx0, "gx1": gx0 + 200,
            "text_band": [40, 160], "transcription": status[i],
            "cross_strip": False,
        })
    (d / "columns.json").write_text(json.dumps({"columns": cols}))
    (d / "meta.json").write_text(json.dumps({
        "target_id": "toy_columns",
        "geometry": {"grid_shape": [grid_h, grid_w]},
        "line_pitch_range": [10, 40],
    }))
    valid = np.full((grid_h, grid_w), 255, np.uint8)
    Image.fromarray(valid).save(d / "valid_mask.png")
    return d, cols


def _save_pred(tmp_path, arr, name="pred.npy"):
    p = tmp_path / name
    np.save(p, arr.astype(np.float32))
    return str(p)


def test_column_pred_scores_perfect_discrimination(tmp_path):
    d, cols = _make_target(tmp_path)
    pred = np.zeros((200, 1200), np.float32)
    for c in cols:
        if c["transcription"] == "text":
            pred[40:160, c["gx0"]:c["gx1"]] = 0.9
    card = score_columns(_save_pred(tmp_path, pred), str(d))
    m = card["metrics"]
    assert m["col_gutter_auc"] == 1.0
    assert m["col_gutter_pixel_auc"] > 0.99
    assert m["n_text_cols"] == 3 and m["n_gutters"] >= 2


def test_constant_pred_is_chance_and_aperiodic(tmp_path):
    d, _ = _make_target(tmp_path)
    pred = np.full((200, 1200), 0.7, np.float32)
    card = score_columns(_save_pred(tmp_path, pred), str(d))
    m = card["metrics"]
    assert abs(m["col_gutter_auc"] - 0.5) < 1e-9
    assert abs(m["col_gutter_pixel_auc"] - 0.5) < 1e-6
    assert m["line_period_peak_mean"] < 0.2


def test_striped_pred_shows_line_periodicity(tmp_path):
    d, cols = _make_target(tmp_path)
    pred = np.zeros((200, 1200), np.float32)
    ys = np.arange(200)
    stripes = (np.sin(2 * np.pi * ys / 20.0) > 0).astype(np.float32)  # pitch 20
    for c in cols:
        if c["transcription"] == "text":
            pred[:, c["gx0"]:c["gx1"]] = stripes[:, None]
    card = score_columns(_save_pred(tmp_path, pred), str(d))
    assert card["metrics"]["line_period_peak_mean"] > 0.5


def test_partial_extent_with_origin_scores_contained_columns_only(tmp_path):
    d, cols = _make_target(tmp_path)
    # extent covering only columns 3 and 4 (gx 660..1160) plus margins
    y0, x0 = 0, 600
    pred = np.zeros((200, 600), np.float32)
    for c in cols[2:]:
        pred[40:160, c["gx0"] - x0:c["gx1"] - x0] = 1.0
    card = score_columns(_save_pred(tmp_path, pred), str(d), origin=(y0, x0))
    m = card["metrics"]
    assert m["cols_scored"] == [3, 4]
    assert m["n_gutters"] == 1  # only the gutter between cols 3 and 4
    assert m["col_gutter_auc"] == 1.0


def test_cross_strip_column_gutters_excluded(tmp_path):
    d, cols = _make_target(tmp_path)
    cj = json.loads((d / "columns.json").read_text())
    cj["columns"][2]["cross_strip"] = True  # col 3 flagged
    (d / "columns.json").write_text(json.dumps(cj))
    pred = np.random.default_rng(0).random((200, 1200)).astype(np.float32)
    card = score_columns(_save_pred(tmp_path, pred), str(d))
    m = card["metrics"]
    # gutters adjacent to col 3 (between 2-3 and 3-4) are excluded; col 3 still scored
    assert 3 in m["cols_scored"]
    assert m["n_gutters"] == 1  # only gutter between cols 1 and 2 remains
    assert m["excluded_gutters"] == 2


def test_cli_score_columns_wires(tmp_path, capsys):
    from scrollgt.cli import main
    d, cols = _make_target(tmp_path)
    pred = np.zeros((200, 1200), np.float32)
    for c in cols[1:]:
        pred[40:160, c["gx0"]:c["gx1"]] = 1.0
    p = _save_pred(tmp_path, pred)
    out = tmp_path / "card.json"
    rc = main(["score-columns", p, str(d), "--json-out", str(out)])
    assert rc == 0
    card = json.loads(out.read_text())
    assert card["metrics"]["col_gutter_auc"] == 1.0
    assert "col_gutter_auc" in capsys.readouterr().out


# --- anti-gaming floors the README publishes but nothing pinned (audit, 2026-08) --------
# BASELINES.md and the README state that a papyrus-mask copy scores EXACTLY 0.5, because
# the gutters between columns are papyrus too. That is the column target's most
# distinctive anti-gaming property, and it was a documented measurement rather than an
# enforced one: if the metric drifted so that "predict papyrus everywhere" earned credit,
# no test would have caught it.

def test_papyrus_mask_copy_scores_exactly_chance(tmp_path):
    """Predicting the papyrus mask must earn nothing: gutters are papyrus too."""
    d, _ = _make_target(tmp_path)
    valid = np.array(Image.open(d / "valid_mask.png")).astype(np.float32) / 255.0
    card = score_columns(_save_pred(tmp_path, valid), str(d))
    m = card["metrics"]
    assert abs(m["col_gutter_auc"] - 0.5) < 1e-9, (
        "a papyrus-mask copy scored above chance; the anti-gaming floor is broken")
    assert abs(m["col_gutter_pixel_auc"] - 0.5) < 1e-6


def test_partial_papyrus_mask_still_scores_chance(tmp_path):
    """Same, with a mask that excludes some of the grid, as a real valid_mask does."""
    d, _ = _make_target(tmp_path)
    valid = np.array(Image.open(d / "valid_mask.png")).astype(np.uint8)
    valid[:, :100] = 0          # a genuinely masked-off strip
    valid[:20, :] = 0
    Image.fromarray(valid).save(d / "valid_mask.png")
    card = score_columns(_save_pred(tmp_path, valid.astype(np.float32) / 255.0), str(d))
    assert abs(card["metrics"]["col_gutter_auc"] - 0.5) < 1e-9


def test_inverted_prediction_scores_below_chance_not_above(tmp_path):
    """Sanity on the metric's orientation: predicting gutters must not look like reading."""
    d, cols = _make_target(tmp_path)
    pred = np.zeros((200, 1200), np.float32)
    for c in cols:
        if c["transcription"] == "text":
            pred[40:160, c["gx0"]:c["gx1"]] = 0.9
    inverted = 0.9 - pred
    card = score_columns(_save_pred(tmp_path, inverted), str(d))
    assert card["metrics"]["col_gutter_auc"] < 0.5, (
        "an inverted prediction scored at or above chance; the metric is not orientated")
