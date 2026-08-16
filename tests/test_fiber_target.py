"""The shipped targets must be scoreable with no GPU, no model, and no network."""

import json
import pathlib

import numpy as np
import pytest

from scrollgt.fibers import score_tracing
from scrollgt.fibers.target import load_fiber_target, score_fiber_prediction

TARGETS = sorted(pathlib.Path("data").glob("fibers_*"))

# --- which targets pay the floor-recomputation cost by default -------------------------
# Recomputing the connected-components floor is the guarantee that the published numbers
# reproduce from the repo with no GPU. It is also the expensive test: a 512 cube measures
# ~70 s and ~8 GB peak RSS against a few seconds and well under 1 GB for a 256 cube.
# Running all three 512 cubes by default puts that load on a CI runner that has never
# once executed against this data, and OOMs a 7 GB runner outright.
#
# The call: every 256 target plus ONE 512 target runs by default, so the reproduction
# guarantee still covers both size classes and a 512-specific packaging mistake is still
# caught. The other two 512 targets are marked `heavy` and deselected by the
# `-m "not heavy"` in pyproject.toml; run them with `pytest -m heavy` (see CONTRIBUTING).
# Dropping 512 coverage entirely would have been cheaper and dishonest.
_512_TARGETS = [t for t in TARGETS if t.name.endswith("_512")]
DEFAULT_512 = _512_TARGETS[0] if _512_TARGETS else None
HEAVY_512 = _512_TARGETS[1:]
FLOOR_TARGETS = [
    pytest.param(t, marks=pytest.mark.heavy) if t in HEAVY_512 else t for t in TARGETS
]


def test_eleven_targets_are_shipped():
    assert len(TARGETS) == 11, [t.name for t in TARGETS]


def test_the_default_floor_run_still_covers_both_size_classes():
    """The opt-in split above is a compute budget, not a way to stop testing 512 cubes.

    If a future edit marked the last 512 target heavy, the default suite would reproduce
    floors for one size class only -- and would still pass, silently. That is precisely
    the pattern this repo keeps getting caught by: a property measured once and never
    re-checked.
    """
    assert DEFAULT_512 is not None, "no 512 target runs floors by default"
    assert len(HEAVY_512) == len(_512_TARGETS) - 1
    assert DEFAULT_512 not in HEAVY_512
    default_names = {t.name for t in TARGETS if t not in HEAVY_512}
    assert sum(n.endswith("_256") for n in default_names) == 8
    assert sum(n.endswith("_512") for n in default_names) == 1


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


@pytest.mark.parametrize("target", FLOOR_TARGETS, ids=lambda p: p.name)
def test_published_floors_reproduce_from_shipped_data(target):
    """The regression that keeps the zero-GPU path real.

    Recomputes the connected-components floor from the shipped mask alone and
    requires it to match the number published in meta.json. A data packaging
    mistake would otherwise be silent.

    Two of the three 512 cubes are `heavy` and deselected by default -- see the
    FLOOR_TARGETS comment above for why, and CONTRIBUTING.md for how to run them.
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
    """The default path must not pay the floor-recomputation cost.

    That cost is ~50 s for the 256 cube used here and minutes for a 512 cube, whose
    connected-components floor alone measures ~70 s.
    """
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


# --- skeleton/mask alignment, RECOMPUTED (audit, 2026-08-14) ----------------------------
# test_convention_proof_is_recorded reads measured_node_landing_rate_on_semantic_label out
# of meta.json and asserts it is >= 0.999. That is self-certifying: a misaligned skeleton
# would still carry 1.0 in its own metadata and pass. It is also measured against
# labelsTr, villa's semantic label, which is NOT the reference mask that scoring uses.
#
# This recomputes alignment from the shipped skeleton and mask, so a coordinate-convention
# error (axis order, origin, flip) fails here instead of silently becoming everyone's
# coverage and ERL numbers.

def _mask_and_nodes(target):
    sk = np.load(target / "skeleton.npz")
    mk = np.load(target / "mask.npz")
    shape = tuple(int(v) for v in mk["shape"])
    mask = np.unpackbits(mk["packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    idx = np.round(sk["coords"]).astype(int)
    inb = ((idx >= 0) & (idx < np.array(shape))).all(1)
    return mask, idx[inb]


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_skeleton_lands_on_the_scoring_mask_far_above_chance(target):
    """Hand-traced nodes must fall on the reference mask much more often than chance.

    Chance is the mask density (~5%). Measured across the eleven shipped targets is
    63.7-85.4%, an enrichment of 9.9-21.5x (lowest s5_06994_00994_04994_512, highest
    s5_07997_02997_05497_256). A frame error would collapse this toward density, which is
    exactly what the assertion catches; the 5x threshold is left where it is, well below
    the measured minimum and well above the ~1x an axis swap produces.
    """
    mask, nodes = _mask_and_nodes(target)
    assert len(nodes) > 100, f"{target.name}: too few in-bounds nodes to judge"
    rate = mask[nodes[:, 0], nodes[:, 1], nodes[:, 2]].mean()
    density = mask.mean()
    enrichment = rate / density
    assert enrichment > 5.0, (
        f"{target.name}: nodes land on the scoring mask at {rate:.3f} vs density "
        f"{density:.4f} (enrichment {enrichment:.1f}x). Below 5x suggests the skeleton and "
        "mask are not in the same coordinate frame.")


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_a_permuted_axis_convention_would_be_detected(target):
    """The alignment check must actually discriminate, not pass on anything.

    Swapping two axes is the most likely convention error (nml is x,y,z; volumes are
    z,y,x). If a swapped skeleton still cleared the bar, the test above would be decoration.
    """
    mask, nodes = _mask_and_nodes(target)
    density = mask.mean()
    swapped = nodes[:, [1, 0, 2]]
    keep = ((swapped >= 0) & (swapped < np.array(mask.shape))).all(1)
    if keep.sum() < 100:
        pytest.skip("too few nodes survive the swap to judge")
    s = swapped[keep]
    bogus = mask[s[:, 0], s[:, 1], s[:, 2]].mean() / density
    assert bogus < 5.0, (
        f"{target.name}: an axis-swapped skeleton also scores {bogus:.1f}x enrichment, so "
        "the alignment check cannot tell a correct convention from a wrong one")
