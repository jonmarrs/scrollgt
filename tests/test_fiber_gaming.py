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


def test_aggregating_across_size_classes_is_refused():
    """A mean over mixed cube sizes is meaningless, so it must raise, not compute.

    Without this, a tracer scoring 60 on a 512 cube outranks one scoring 45 on a 256 cube
    for reasons of geometry rather than quality -- the same confound class as the
    n_fibers / n_fibers_scored conflation this benchmark already had to fix.
    """
    from scrollgt.fibers.target import aggregate_fiber_scores

    mixed = [
        {"size_class": 256, "metrics": {"erl": 45.0, "erl_merge_penalized": 30.0}},
        {"size_class": 512, "metrics": {"erl": 60.0, "erl_merge_penalized": 40.0}},
    ]
    with pytest.raises(ValueError, match="size class"):
        aggregate_fiber_scores(mixed)


def test_aggregating_within_one_size_class_works():
    from scrollgt.fibers.target import aggregate_fiber_scores

    same = [
        {"size_class": 256, "metrics": {"erl": 40.0, "erl_merge_penalized": 30.0}},
        {"size_class": 256, "metrics": {"erl": 50.0, "erl_merge_penalized": 20.0}},
    ]
    out = aggregate_fiber_scores(same)
    assert out["size_class"] == 256
    assert out["n"] == 2
    assert out["erl_mean"] == pytest.approx(45.0)
    assert out["erl_merge_penalized_mean"] == pytest.approx(25.0)


def test_aggregating_nothing_is_refused():
    from scrollgt.fibers.target import aggregate_fiber_scores

    with pytest.raises(ValueError):
        aggregate_fiber_scores([])


def test_it_accepts_a_real_scorecard_not_just_a_hand_built_dict(tmp_path):
    """Guard the card shape itself.

    `score_fiber_prediction` returns ERL under `card["metrics"]`, while `size_class` sits at
    the top level. A version of this function reading `card["erl"]` passes every synthetic
    test above and raises KeyError on every real card. Score one and aggregate it.
    """
    import pathlib

    import numpy as np

    from scrollgt.fibers.target import (
        aggregate_fiber_scores,
        load_fiber_target,
        score_fiber_prediction,
    )

    data = pathlib.Path(__file__).resolve().parents[1] / "data"
    target = sorted(data.glob("fibers_*"))[0]
    _, mask, _ = load_fiber_target(str(target))

    pred = tmp_path / "empty.npy"
    np.save(pred, np.zeros(mask.shape, dtype=np.int32))
    card = score_fiber_prediction(str(pred), str(target))

    out = aggregate_fiber_scores([card])
    assert out["n"] == 1
    assert out["size_class"] == card["size_class"]
    assert out["erl_mean"] == pytest.approx(card["metrics"]["erl"])
