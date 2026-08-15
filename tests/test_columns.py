"""Tests for the column-level scoring contract (v0.2, PHerc 1667 merged geometry)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
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


# --- the SHIPPED column geometry (audit, 2026-08-14) ------------------------------------
# Every test above builds a synthetic target in tmp_path. Nothing touched the shipped
# pherc1667_merged_columns geometry, which is the actual product: its registration claims
# ("exactly 22 bracket intervals", "tiling closure 3 px over 30097", cols 9 and 16 flagged)
# lived only as prose in meta.json, measured once in another repo. These re-check them.

COLTGT = Path(__file__).resolve().parents[1] / "data" / "pherc1667_merged_columns"
pytestmark_col = pytest.mark.skipif(not COLTGT.exists(), reason="column target not present")


def _col_target():
    cols = json.loads((COLTGT / "columns.json").read_text())["columns"]
    meta = json.loads((COLTGT / "meta.json").read_text())
    return cols, meta


@pytestmark_col
def test_shipped_columns_match_the_registration_claims():
    cols, meta = _col_target()
    assert len(cols) == 22, meta["registration"]["method"]
    assert sorted(c["col"] for c in cols if c.get("cross_strip")) == [9, 16], (
        "meta flags cols 9 and 16 as spanning strip-crop gaps; the data must agree")


@pytestmark_col
def test_valid_mask_matches_the_declared_grid_shape():
    """A regenerated mask at a different size would silently rescale every column box."""
    _, meta = _col_target()
    h, w = meta["geometry"]["grid_shape"]
    vm = np.array(Image.open(COLTGT / "valid_mask.png"))
    if vm.ndim == 3:
        vm = vm[..., 0]
    assert vm.shape == (h, w), f"valid_mask {vm.shape} != declared grid {(h, w)}"


@pytestmark_col
def test_shipped_columns_are_ordered_disjoint_and_in_bounds():
    cols, meta = _col_target()
    h, w = meta["geometry"]["grid_shape"]
    xs = [(c["col"], c["gx0"], c["gx1"]) for c in cols]
    for name, a, b in xs:
        assert 0 <= a < b <= w, f"col {name} bounds {a},{b} outside grid width {w}"
    for c in cols:
        t0, t1 = c["text_band"]
        assert 0 <= t0 < t1 <= h, f"col {c['col']} text_band {t0},{t1} outside grid height"
    for i in range(len(xs) - 1):
        assert xs[i][1] < xs[i + 1][1], "columns must be left-to-right ordered"
        assert xs[i][2] <= xs[i + 1][1], f"cols {xs[i][0]} and {xs[i+1][0]} overlap"


@pytestmark_col
def test_registered_columns_land_on_papyrus():
    """Column boxes should sit on valid surface, not off the edge of the sheet.

    WEAK CHECK, deliberately stated as such: the valid mask is ~75% of the grid, so the
    achievable enrichment is small (measured median 0.98 vs 0.755 overall, ~1.3x). This
    catches a grossly misplaced registration, not a subtle one. The column family has no
    cross-scan bridge and so no placement gate equivalent; do not read this as one.
    """
    cols, meta = _col_target()
    h, w = meta["geometry"]["grid_shape"]
    vm = np.array(Image.open(COLTGT / "valid_mask.png"))
    if vm.ndim == 3:
        vm = vm[..., 0]
    vm = vm > 127
    fracs = []
    for c in cols:
        t0, t1 = c["text_band"]
        fracs.append(vm[t0:t1, c["gx0"]:c["gx1"]].mean())
    med = float(np.median(fracs))
    assert med > vm.mean(), (
        f"median column-box valid fraction {med:.3f} is not above the overall mask density "
        f"{vm.mean():.3f}; columns may not be landing on papyrus")
    assert med > 0.90, f"median column-box valid fraction {med:.3f} is implausibly low"
