"""Tolerance growth must retain SciPy's exact labels while bounding EDT scratch space."""

import numpy as np
import pytest
from scipy import ndimage

from scrollgt.fibers import eval_trace as ev
from scrollgt.fibers.skeleton_io import Fiber, Skeleton


def full_growth(labels, radius):
    if radius <= 0:
        return labels
    background = labels == 0
    distance, indices = ndimage.distance_transform_edt(background, return_indices=True)
    out = labels.copy()
    grow = background & (distance <= radius)
    out[grow] = labels[tuple(axis[grow] for axis in indices)]
    return out


@pytest.mark.parametrize("radius", [0, 0.5, 1, 2, 2.5, 5, np.inf, np.nan])
def test_growth_matches_the_full_transform_including_ties(monkeypatch, radius):
    monkeypatch.setattr(ev, "_EDT_BLOCK_DEPTH", 4, raising=False)
    rng = np.random.default_rng(7)
    for shape in ((1, 7, 9), (11, 9, 13), (17, 12, 9)):
        labels = rng.integers(0, 6, size=shape, dtype=np.int64)
        labels[rng.random(shape) < 0.8] = 0
        original = labels.copy()
        np.testing.assert_array_equal(ev._dilate_labels(labels, radius), full_growth(labels, radius))
        np.testing.assert_array_equal(labels, original)


@pytest.mark.parametrize("radius", [1, 2, 2.5])
def test_empty_blocks_and_foreground_just_beyond_the_halo(monkeypatch, radius):
    monkeypatch.setattr(ev, "_EDT_BLOCK_DEPTH", 4, raising=False)
    labels = np.zeros((25, 9, 11), dtype=np.uint32)
    # Several slabs have no nearby foreground. The all-background EDT's artificial
    # (-1, 0, 0) index must not borrow a label from the opposite end of the volume.
    labels[-1, 0, 0] = 2**31 + 7
    labels[12, 4, 5] = 3
    np.testing.assert_array_equal(ev._dilate_labels(labels, radius), full_growth(labels, radius))
    empty = np.zeros_like(labels)
    np.testing.assert_array_equal(ev._dilate_labels(empty, radius), empty)


def _long_volume():
    coords = np.array([[65., 8., 2.], [65., 8., 16.]])
    fiber = Fiber(1, "f", np.arange(2), coords, np.array([[0, 1]]))
    labels = np.zeros((160, 17, 19), dtype=np.int32)
    labels[65, 8, 2:17] = 1
    return Skeleton([fiber]), labels


def test_growth_never_requests_a_whole_long_volume(monkeypatch):
    transform = ndimage.distance_transform_edt
    calls = []

    def bounded(background, *args, **kwargs):
        calls.append(background.shape)
        assert background.shape[0] <= 68  # 64 interior planes plus radius 2 on each side
        return transform(background, *args, **kwargs)

    monkeypatch.setattr(ndimage, "distance_transform_edt", bounded)
    _, labels = _long_volume()
    ev._dilate_labels(labels, 2)
    assert calls


def test_precision_transform_is_bounded_too(monkeypatch):
    transform = ndimage.distance_transform_edt
    calls = []

    def bounded(background, *args, **kwargs):
        calls.append((background.shape, kwargs.get("return_indices", False)))
        assert background.shape[0] <= 68
        return transform(background, *args, **kwargs)

    monkeypatch.setattr(ndimage, "distance_transform_edt", bounded)
    gt, labels = _long_volume()
    score = ev.score_tracing(gt, labels, tolerance=2)
    assert score.coverage == score.precision == 1
    assert {indices for _, indices in calls} == {False, True}


def test_all_raw_score_fields_equal_the_full_transform(monkeypatch):
    gt, labels = _long_volume()
    labels[65, 8, 8:10] = 0
    labels[64, 10, 4:12] = 2
    monkeypatch.setattr(ev, "_EDT_BLOCK_DEPTH", 160, raising=False)
    expected = ev.score_tracing(gt, labels, tolerance=2.5)
    monkeypatch.setattr(ev, "_EDT_BLOCK_DEPTH", 4)
    assert ev.score_tracing(gt, labels, tolerance=2.5) == expected
