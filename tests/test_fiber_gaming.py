"""The finding this benchmark exists to publish, pinned as an executable claim.

Coverage and precision are properties of the fiber *mask*, not of the instance
*labelling*, so four wildly different labellings score identically on them. Only
ERL and the merge count separate a correct tracer from numpy.random.
"""

import pytest

from scrollgt.fibers import (
    floor_connected_components,
    floor_random_instances,
    floor_single_instance,
    floor_voxel_instances,
    score_tracing,
)
from scrollgt.fibers.target import load_fiber_target

TARGET = "data/fibers_s1_00497_01497_03997_256"


@pytest.fixture(scope="module")
def rows():
    skel, mask, meta = load_fiber_target(TARGET)
    tol = meta["tolerance"]
    return {
        "single": score_tracing(skel, floor_single_instance(mask), tolerance=tol).as_row(),
        "cc": score_tracing(skel, floor_connected_components(mask), tolerance=tol).as_row(),
        "voxel": score_tracing(skel, floor_voxel_instances(mask), tolerance=tol).as_row(),
        "random": score_tracing(skel, floor_random_instances(mask, n=50, seed=0),
                                tolerance=tol).as_row(),
    }


def test_coverage_and_precision_cannot_rank_a_labelling(rows):
    covs = {round(r["coverage"], 4) for r in rows.values()}
    precs = {round(r["precision"], 4) for r in rows.values()}
    assert len(covs) == 1, f"coverage should be identical across labellings, got {covs}"
    assert len(precs) == 1, f"precision should be identical across labellings, got {precs}"


def test_erl_does_separate_them(rows):
    erls = sorted(r["erl"] for r in rows.values())
    assert erls[-1] / max(erls[0], 1e-9) > 50, (
        f"ERL must separate these labellings by orders of magnitude, got {erls}")


def test_raw_erl_alone_is_gameable(rows):
    """Labelling everything once scores near the oracle on raw ERL."""
    _, _, meta = load_fiber_target(TARGET)
    oracle = meta["floors"]["oracle"]["erl"]
    assert rows["single"]["erl"] > 0.6 * oracle, (
        "the single-instance floor is supposed to look deceptively good on raw ERL")


def test_the_merge_penalty_is_what_catches_it(rows):
    assert rows["single"]["erl_merge_penalized"] == 0.0
    assert rows["random"]["erl_merge_penalized"] == 0.0


def test_merges_are_never_summed_into_splits(rows):
    for name, r in rows.items():
        assert "splits" in r and "merges" in r, name
        assert r["splits"] >= 0 and r["merges"] >= 0
