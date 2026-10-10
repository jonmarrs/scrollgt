"""Fiber scores carry the scorer version, and published floors are never compared across versions.

Version 1 (ScrollGT <= 0.3.2) read runs in stored edge-row order: reshuffling one shipped cube's rows moved a published
floor by up to 25% with identical geometry. Version 2 (scrollgt PR #1) walks edges in path order. A score from one
version read against floors from the other compares two different metrics, so the published floors record their
version and scoring refuses a mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scrollgt.fibers.eval_trace import SCORING_VERSION, score_tracing
from scrollgt.fibers.skeleton_io import Fiber, Skeleton
from scrollgt.fibers.target import score_fiber_prediction

DATA = Path(__file__).resolve().parents[1] / "data"
TARGETS = sorted(DATA.glob("fibers_*"))
COMPUTED = ("oracle", "floor_single_instance", "floor_connected_components", "floor_voxel_instances",
            "floor_random_instances")  # fmt: skip


def test_rows_carry_the_scoring_version():
    coords = np.array([[1, 1, x] for x in range(1, 6)], dtype=float)
    f = Fiber(1, "f", np.arange(5), coords, np.array([[i, i + 1] for i in range(4)], dtype=np.int64))
    row = score_tracing(Skeleton([f]), np.ones((3, 3, 8), dtype=np.int64), tolerance=0).as_row()
    assert row["scoring_version"] == SCORING_VERSION == 2


def test_a_zero_length_edge_keeps_the_fiber_one_run():
    """WEBKNOSSOS traces contain duplicate nodes joined by a zero-length edge (821 of 87,469 shipped edges)."""
    coords = np.array([[1, 1, 1], [1, 1, 3], [1, 1, 3], [1, 1, 5]], dtype=float)
    f = Fiber(1, "dup", np.arange(4), coords, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64))
    inst = np.ones((3, 3, 8), dtype=np.int64)
    for edges in (f.edges, f.edges[::-1], f.edges[:, ::-1]):
        g = Fiber(1, "dup", np.arange(4), coords, edges)
        s = score_tracing(Skeleton([g]), inst, tolerance=0)
        assert len(s.run_lengths) == 1
        assert s.splits == 0


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_shipped_floors_record_the_scorer_that_made_them(target):
    meta = json.loads((target / "meta.json").read_text())
    assert meta["floors_scoring_version"] == SCORING_VERSION
    for key in COMPUTED:
        assert meta["floors"][key]["scoring_version"] == SCORING_VERSION, key
    others = [k for k in meta["floors"] if k not in COMPUTED]
    stale, external = meta["floors_not_recomputed"], meta.get("floors_external", [])
    # Rows the repo cannot recompute (e.g. tracer rows, whose tracer needs a GPU model and the CT cube) are
    # listed exactly once: stale ones from an older scorer, or external ones re-measured with this scorer.
    assert sorted(others) == sorted(stale + external) and not set(stale) & set(external)
    for key in stale:
        assert meta["floors"][key]["scoring_version"] != SCORING_VERSION, key
    for key in external:
        assert meta["floors"][key]["scoring_version"] == SCORING_VERSION, key
        assert meta["floors_external_source"][key], key


def test_published_floors_from_another_version_are_refused(tmp_path):
    src = TARGETS[0]
    dst = tmp_path / src.name
    dst.mkdir()
    for name in ("skeleton.npz", "mask.npz"):
        (dst / name).write_bytes((src / name).read_bytes())
    meta = json.loads((src / "meta.json").read_text())
    meta["floors_scoring_version"] = 1
    (dst / "meta.json").write_text(json.dumps(meta))
    labels = tmp_path / "labels.npy"
    np.save(labels, np.zeros(tuple(meta["shape"]), dtype=np.int32))
    with pytest.raises(ValueError, match="scoring version 1"):
        score_fiber_prediction(labels, dst)


def test_cards_state_the_scoring_version(tmp_path):
    target = TARGETS[0]
    meta = json.loads((target / "meta.json").read_text())
    labels = tmp_path / "labels.npy"
    np.save(labels, np.zeros(tuple(meta["shape"]), dtype=np.int32))
    card = score_fiber_prediction(labels, target)
    assert card["scoring_version"] == SCORING_VERSION
    assert card["metrics"]["scoring_version"] == SCORING_VERSION
