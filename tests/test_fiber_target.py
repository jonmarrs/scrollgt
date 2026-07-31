"""The shipped targets must be scoreable with no GPU, no model, and no network."""

import json
import pathlib

import numpy as np
import pytest

from scrollgt.fibers import score_tracing
from scrollgt.fibers.target import load_fiber_target, score_fiber_prediction

TARGETS = sorted(pathlib.Path("data").glob("fibers_*"))


def test_six_targets_are_shipped():
    assert len(TARGETS) == 6, [t.name for t in TARGETS]


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_loads_skeleton_and_mask(target):
    skel, mask, meta = load_fiber_target(target)
    assert len(skel) == meta["ground_truth"]["n_fibers_scored"]
    assert mask.shape == tuple(meta["shape"])
    assert mask.dtype == bool
    assert 0.0 < mask.mean() < 0.5


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_convention_proof_is_recorded(target):
    _, _, meta = load_fiber_target(target)
    rate = meta["convention_check"]["measured_node_landing_rate_on_semantic_label"]
    assert rate >= 0.999, f"{target.name}: landing rate {rate}"


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_published_floors_reproduce_from_shipped_data(target):
    """The regression that keeps the zero-GPU path real.

    Recomputes the connected-components floor from the shipped mask alone and
    requires it to match the number published in meta.json. A data packaging
    mistake would otherwise be silent.
    """
    from scrollgt.fibers import floor_connected_components

    skel, mask, meta = load_fiber_target(target)
    published = meta["floors"]["floor_connected_components"]
    got = score_tracing(skel, floor_connected_components(mask),
                        tolerance=meta["tolerance"]).as_row()
    assert got["erl"] == pytest.approx(published["erl"], rel=1e-3)
    assert got["erl_merge_penalized"] == pytest.approx(
        published["erl_merge_penalized"], rel=1e-3)
    assert got["coverage"] == pytest.approx(published["coverage"], rel=1e-3)


def test_scoring_an_empty_labelling_is_zero_not_an_error(tmp_path):
    target = TARGETS[0]
    _, mask, _ = load_fiber_target(target)
    p = tmp_path / "empty.npy"
    np.save(p, np.zeros(mask.shape, dtype=np.int32))
    card = score_fiber_prediction(p, target)
    assert card["metrics"]["erl"] == 0.0
    assert card["metrics"]["coverage"] == 0.0


def test_shape_mismatch_names_the_expected_shape(tmp_path):
    target = TARGETS[0]
    p = tmp_path / "wrong.npy"
    np.save(p, np.zeros((64, 64, 64), dtype=np.int32))
    with pytest.raises(ValueError, match="256"):
        score_fiber_prediction(p, target)


def test_probability_input_is_rejected_with_a_useful_message(tmp_path):
    target = TARGETS[0]
    _, mask, _ = load_fiber_target(target)
    p = tmp_path / "probs.npy"
    np.save(p, mask.astype(np.float32))
    with pytest.raises(ValueError, match="instance ids"):
        score_fiber_prediction(p, target)


def test_floors_default_to_published_and_can_be_recomputed(tmp_path):
    """The default path must not pay the ~50 s recomputation cost."""
    target = TARGETS[0]
    _, mask, _ = load_fiber_target(target)
    p = tmp_path / "empty.npy"
    np.save(p, np.zeros(mask.shape, dtype=np.int32))

    published = score_fiber_prediction(p, target)
    assert published["floors_source"] == "published"
    assert set(published["floors"]) == {
        "floor_single_instance", "floor_connected_components",
        "floor_voxel_instances", "floor_random_instances",
    }

    recomputed = score_fiber_prediction(p, target, recompute_floors=True)
    assert recomputed["floors_source"] == "recomputed"
    for key, row in recomputed["floors"].items():
        assert row["erl"] == pytest.approx(published["floors"][key]["erl"], rel=1e-3), key


def test_below_baseline_flag_is_set_when_entry_trails_connected_components(tmp_path):
    target = TARGETS[0]
    _, mask, _ = load_fiber_target(target)
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[mask] = np.arange(1, int(mask.sum()) + 1)  # one instance per voxel
    p = tmp_path / "voxels.npy"
    np.save(p, labels)
    card = score_fiber_prediction(p, target)
    assert card["below_baseline"] is True
