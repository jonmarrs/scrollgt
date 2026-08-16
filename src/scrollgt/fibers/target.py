"""Load and score the shipped fiber connectivity targets.

Everything here reads from files committed in this repository. No model, no
GPU, and no network access is used or required.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .eval_trace import (
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


def score_fiber_prediction(labels_path, target_dir, recompute_floors: bool = False) -> dict:
    """Score an instance labelling (.npy of ints, 0 = background) against a target.

    Floors come from the target's published meta.json by default. Recomputing
    them from the shipped mask costs ~45-50 s per cube and is what
    `recompute_floors=True` is for; the test suite already enforces that the
    published values reproduce, so users do not pay that cost on every run.
    """
    skeleton, mask, meta = load_fiber_target(target_dir)
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
        floors = {k: v for k, v in meta.get("floors", {}).items()
                  if k.startswith("floor_")}
        floors_source = "published"

    cc = floors.get("floor_connected_components", {})
    below = bool(cc) and row["erl"] < cc["erl"]

    return {
        "target": meta.get("target_id", os.path.basename(os.path.normpath(target_dir))),
        "prediction": os.path.basename(str(labels_path)),
        "split": meta.get("split", "primary"),
        "tolerance": tolerance,
        # ERL is a length statistic, so a score means nothing without the ceiling for
        # its own cube size. Carry both on the card rather than leaving the reader to
        # look them up.
        "size_class": int(meta["size_class"]),
        "class_oracle_erl": meta.get("floors", {}).get("oracle", {}).get("erl"),
        "metrics": row,
        "floors": floors,
        "floors_source": floors_source,
        "below_baseline": below,
    }
