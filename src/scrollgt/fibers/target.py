"""Load and score the shipped fiber connectivity targets.

Everything here reads from files committed in this repository. No model, no
GPU, and no network access is used or required.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .eval_trace import (
    SCORING_VERSION,
    floor_connected_components,
    floor_random_instances,
    floor_single_instance,
    floor_voxel_instances,
    score_tracing,
)
from .skeleton_io import Fiber, Skeleton


def _unpack_skeleton(npz) -> Skeleton:
    coords = npz["coords"]
    edges = npz["edges"]
    f_off = npz["fiber_offsets"]
    e_off = npz["edge_offsets"]
    ids = npz["fiber_ids"]
    names = npz["fiber_names"]

    fibers = []
    for i in range(len(ids)):
        c0, c1 = int(f_off[i]), int(f_off[i + 1])
        e0, e1 = int(e_off[i]), int(e_off[i + 1])
        fibers.append(
            Fiber(
                id=int(ids[i]),
                name=str(names[i]),
                node_ids=np.arange(c0, c1, dtype=np.int64),
                coords=coords[c0:c1].astype(float),
                # edges were stored global; rebase to this fiber's local indices
                edges=(edges[e0:e1].astype(np.int64) - c0),
            )
        )
    return Skeleton(
        fibers=fibers,
        scale_um=tuple(float(v) for v in npz["scale_um"]),
        origin_zyx=tuple(int(v) for v in npz["origin_zyx"]),
    )


def load_fiber_target(target_dir):
    """Load one fiber target: (Skeleton, mask bool (Z,Y,X), meta dict).

    ``skeleton.npz`` on disk holds the *full* hand-traced skeleton, including
    fibers that fall mostly or entirely outside the cube (annotators traced
    somewhat beyond the cube boundary). The skeleton returned here has already
    been filtered down to the fibers that are actually scoreable — those with
    more than one in-bounds node, i.e.
    ``[f for f in skeleton.fibers if f.in_bounds_mask(shape).sum() > 1]`` — since
    a fiber needs at least two in-bounds nodes to contribute a measurable run.
    This is the same filter used to produce the published floor numbers in
    ``meta.json``, so scoring against what this function returns reproduces
    them; scoring against the raw unfiltered trace does not. The full trace
    remains on disk in ``skeleton.npz`` for anyone who wants it directly.
    """
    target_dir = str(target_dir)
    with open(os.path.join(target_dir, "meta.json")) as f:
        meta = json.load(f)

    with np.load(os.path.join(target_dir, "skeleton.npz"), allow_pickle=True) as npz:
        skeleton = _unpack_skeleton(npz)

    with np.load(os.path.join(target_dir, "mask.npz")) as npz:
        shape = tuple(int(v) for v in npz["shape"])
        n = int(np.prod(shape))
        mask = np.unpackbits(npz["packed"])[:n].astype(bool).reshape(shape)

    scored_fibers = [f for f in skeleton.fibers if f.in_bounds_mask(shape).sum() > 1]
    skeleton = Skeleton(
        fibers=scored_fibers,
        scale_um=skeleton.scale_um,
        origin_zyx=skeleton.origin_zyx,
    )

    expected = meta.get("ground_truth", {}).get("n_fibers_scored")
    if expected is not None and len(skeleton) != expected:
        raise ValueError(
            f"{target_dir}: filtered skeleton has {len(skeleton)} scoreable "
            f"fibers (>1 in-bounds node) but meta.json's "
            f"ground_truth.n_fibers_scored is {expected}; the shipped "
            f"skeleton.npz and meta.json are out of sync for this target"
        )

    return skeleton, mask, meta


def _floor_rows(skeleton, mask, tolerance) -> dict:
    return {
        "floor_single_instance": score_tracing(
            skeleton, floor_single_instance(mask), tolerance=tolerance).as_row(),
        "floor_connected_components": score_tracing(
            skeleton, floor_connected_components(mask), tolerance=tolerance).as_row(),
        "floor_voxel_instances": score_tracing(
            skeleton, floor_voxel_instances(mask), tolerance=tolerance).as_row(),
        "floor_random_instances": score_tracing(
            skeleton, floor_random_instances(mask, n=50, seed=0),
            tolerance=tolerance).as_row(),
    }


# The floors double as a zero-input demo. A cold reader following the quickstart
# had no `labels.npy` and hit a numpy traceback on their first command, which is
# the worst possible place for one: the front door of a tool whose whole purpose
# is that other people run it. `--floor` synthesises a prediction from the
# target's own mask so there is always something runnable with no inputs at all.
DEMO_FLOORS = {
    "connected_components": floor_connected_components,
    "single_instance": floor_single_instance,
    "voxel_instances": floor_voxel_instances,
    "random_instances": floor_random_instances,
}


def score_fiber_prediction(labels_path, target_dir, recompute_floors: bool = False,
                           floor: str | None = None) -> dict:
    """Score an instance labelling (.npy of ints, 0 = background) against a target.

    Floors come from the target's published meta.json by default. Recomputing
    them from the shipped mask is what `recompute_floors=True` is for, and costs
    ~45-50 s for a 256 cube and several times that for a 512 cube, whose
    connected-components floor alone measures ~70 s. The test suite already
    enforces that the published values reproduce, so users do not pay that cost
    on every run.
    """
    skeleton, mask, meta = load_fiber_target(target_dir)
    if floor is not None:
        if floor not in DEMO_FLOORS:
            raise ValueError(
                f"unknown floor {floor!r}; choose one of {sorted(DEMO_FLOORS)}")
        labels = DEMO_FLOORS[floor](mask)
    else:
        if not os.path.exists(str(labels_path)):
            raise FileNotFoundError(
                f"no prediction file at {labels_path!r}.\n"
                "score-fibers scores YOUR tracer's instance labelling: a .npy of "
                "integer instance ids (0 = background) shaped exactly like the cube "
                f"in meta.json (shape={meta.get('shape')}).\n"
                "To see the tool run with no inputs at all, score a built-in floor:\n"
                f"    scrollgt score-fibers --floor connected_components {target_dir}"
            )
        labels = np.load(str(labels_path))
    if labels.shape != mask.shape:
        raise ValueError(
            f"prediction shape {labels.shape} != cube shape {mask.shape}; "
            f"label exactly the cube described in meta.json (origin_zyx="
            f"{meta.get('origin_zyx')}, shape={meta.get('shape')})"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"prediction dtype {labels.dtype} is not integer; supply instance ids "
            f"(0 = background), not probabilities"
        )

    tolerance = float(meta["tolerance"])
    row = score_tracing(skeleton, labels, tolerance=tolerance).as_row()

    if recompute_floors:
        floors = _floor_rows(skeleton, mask, tolerance)
        floors_source = "recomputed"
    else:
        # Published floors are only comparable with a score from the same scorer. Version 1
        # (ScrollGT <= 0.3.2) read runs in stored edge-row order, so its floors moved by up
        # to 25% when the same ground truth's rows were reshuffled.
        published_version = meta.get("floors_scoring_version", 1)
        if published_version != SCORING_VERSION:
            raise ValueError(
                f"{target_dir}: the published floors in meta.json were computed with fiber "
                f"scoring version {published_version}, but this scorer is version "
                f"{SCORING_VERSION}, and their numbers are not comparable. Pass "
                f"--recompute-floors, or use target data shipped with this release."
            )
        floors = {k: v for k, v in meta.get("floors", {}).items()
                  if k.startswith("floor_")}
        floors_source = "published"

    cc = floors.get("floor_connected_components", {})
    below = bool(cc) and row["erl"] < cc["erl"]

    return {
        "target": meta.get("target_id", os.path.basename(os.path.normpath(target_dir))),
        # Name the floor when there is no file, so a demo run does not report
        # its prediction as "None".
        "prediction": (f"floor:{floor} (demo, no prediction file)" if floor is not None
                       else os.path.basename(str(labels_path))),
        "split": meta.get("split", "primary"),
        "tolerance": tolerance,
        # ERL is a length statistic, so a score means nothing without the ceiling for
        # its own cube size. Carry both on the card rather than leaving the reader to
        # look them up.
        "size_class": int(meta["size_class"]),
        "class_oracle_erl": meta.get("floors", {}).get("oracle", {}).get("erl"),
        "scoring_version": SCORING_VERSION,
        "metrics": row,
        "floors": floors,
        "floors_source": floors_source,
        "below_baseline": below,
    }


def aggregate_fiber_scores(cards) -> dict:
    """Mean ERL over cards of ONE size class. Raises on a mixed or empty set.

    Provided so that summarising several cubes has a correct implementation to reach for.
    Without one, a reader averages by hand across whatever cubes are in front of them, and
    ERL -- expected run length in voxels -- is not comparable between a 256 cube and a 512
    cube: the larger admits longer fibers and scores higher for geometric reasons.
    """
    cards = list(cards)
    if not cards:
        raise ValueError("aggregate_fiber_scores: no scorecards given")
    classes = {int(c["size_class"]) for c in cards}
    if len(classes) > 1:
        raise ValueError(
            f"refusing to aggregate across size class {sorted(classes)}: ERL is a length "
            "statistic and does not compare between cube sizes; aggregate each class "
            "separately"
        )
    # `size_class` is a top-level card key; the ERL figures live under `metrics`, which is
    # where `score_tracing(...).as_row()` is wrapped. Reading `card["erl"]` instead would
    # pass any hand-built test fixture and fail on every real card.
    n = len(cards)
    return {
        "size_class": classes.pop(),
        "n": n,
        "erl_mean": float(sum(float(c["metrics"]["erl"]) for c in cards) / n),
        "erl_merge_penalized_mean": float(
            sum(float(c["metrics"]["erl_merge_penalized"]) for c in cards) / n
        ),
    }
