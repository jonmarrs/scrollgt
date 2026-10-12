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


def test_aggregating_held_out_with_in_sample_targets_is_refused():
    """The reference mask trained on every in-sample target and on no held-out one (0.6.0).

    On two held-out Scroll-5 cubes the mask's coverage is 0.53-0.64 against 0.82-0.91 in sample, so a mean over
    both splits would average away the one difference the held-out split exists to show.
    """
    from scrollgt.fibers.target import aggregate_fiber_scores

    mixed = [
        {"size_class": 256, "split": "cross_scroll", "metrics": {"erl": 45.0, "erl_merge_penalized": 30.0}},
        {"size_class": 256, "split": "held_out", "metrics": {"erl": 30.0, "erl_merge_penalized": 25.0}},
    ]
    with pytest.raises(ValueError, match="held-out with in-sample"):
        aggregate_fiber_scores(mixed)
    assert aggregate_fiber_scores([mixed[1], mixed[1]])["n"] == 2
    assert aggregate_fiber_scores([mixed[0], {**mixed[0], "split": "primary"}])["n"] == 2


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


def _fiber(fid, y, length=40):
    import numpy as np

    from scrollgt.fibers.skeleton_io import Fiber

    coords = np.array([[1, y, x] for x in range(1, length + 1)], dtype=float)
    return Fiber(fid, f"f{fid}", np.arange(length), coords,
                 np.array([[i, i + 1] for i in range(length - 1)], dtype=np.int64))  # fmt: skip


def test_dropping_fragments_cannot_buy_erl_by_shrinking_the_denominator():
    """Scoring version 2 divided ERL by the covered length, so deleting short fragments raised it.

    One fiber traced whole, the other in four short fragments. Deleting the fragments loses real coverage and, under
    version 2, raised ERL 28.11 -> 39.00. Version 3 divides by the ground-truth length, so the lost runs count as zero
    and ERL falls 20.54 -> 19.50.
    """
    import numpy as np

    from scrollgt.fibers.skeleton_io import Skeleton

    gt = Skeleton([_fiber(1, 2), _fiber(2, 20)])
    labels = np.zeros((3, 24, 48), dtype=np.int32)
    labels[1, 2, 1:41] = 1
    for k, x in enumerate((1, 11, 21, 31)):
        labels[1, 20, x : x + 5] = 2 + k
    pruned = np.where(labels == 1, labels, 0)
    before = score_tracing(gt, labels, tolerance=0)
    after = score_tracing(gt, pruned, tolerance=0)
    assert after.coverage < before.coverage
    assert (before.erl_covered, after.erl_covered) == pytest.approx((28.105, 39.0), abs=1e-3)  # the version-2 exploit
    assert (before.erl, after.erl) == pytest.approx((20.538, 19.5), abs=1e-3)
    assert after.erl == pytest.approx(sum(r * r for r in after.run_lengths) / after.gt_length, rel=1e-12)


def test_on_shipped_data_deleting_small_components_still_raises_only_the_old_erl(rows):
    """The same exploit on a real labelling. vesuvius-autoresearch finding 85 found it by deleting our tracer's
    instances under 400 voxels on this cube: version-2 ERL 35.39 -> 41.46 while coverage fell 0.62 -> 0.38. Here,
    deleting every connected component under 5,000 voxels raises erl_covered (200.47 -> 208.03) and lowers the
    version-3 headline (183.97 -> 181.92)."""
    import numpy as np

    skel, mask, meta = load_fiber_target(TARGET)
    cc = floor_connected_components(mask)
    ids, counts = np.unique(cc[cc > 0], return_counts=True)
    after = score_tracing(skel, np.where(np.isin(cc, ids[counts >= 5000]), cc, 0),
                          tolerance=meta["tolerance"]).as_row()  # fmt: skip
    before = rows["cc"]
    assert after["coverage"] < before["coverage"]
    assert after["erl_covered"] > before["erl_covered"], "the exploit this pins has vanished; re-check the scorer"
    assert after["erl"] < before["erl"]
