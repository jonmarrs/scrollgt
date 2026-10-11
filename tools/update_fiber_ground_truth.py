"""Maintainer tool: record which villa NML each fiber target's ground truth comes from, and move a target to a newer
annotation when villa revises one.

villa's `fiber-skeletons` dataset is versioned (Dataset001 ... Dataset004), and an annotation can be revised between
releases. Targets did not record which file they were built from, so a stale one could not be detected.

Two modes, both dry runs unless --write is given:

* ``--verify-only``: re-parse the NML and require the packed result to equal the shipped ``skeleton.npz``, array for
  array. Then record the dataset, file name and SHA-256 under ``ground_truth.nml``. Nothing else changes.
* default (revision): replace the skeleton with the NML's. Before writing, the NML's nodes must land on villa's semantic
  label for the same release at >= 0.999, the exporter's convention gate. Then recompute the oracle and floor rows,
  re-score the external tracer labelling given with ``--tracer-labels``, and update the ground-truth counts and
  convention check. Every changed number is printed.

Usage:
  python tools/update_fiber_ground_truth.py --target data/fibers_<cube> --nml PATH --dataset NAME --verify-only [--write]
  python tools/update_fiber_ground_truth.py --target data/fibers_<cube> --nml PATH --dataset NAME \
      --semantic PATH --tracer-labels PATH --tracer-source TEXT [--write]

The semantic label is a TIFF; reading it needs ``tifffile``, a maintainer-only dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scrollgt.fibers.eval_trace import SCORING_VERSION, oracle_from_skeleton, score_tracing  # noqa: E402
from scrollgt.fibers.skeleton_io import Skeleton, origin_from_stem, parse_nml  # noqa: E402
from scrollgt.fibers.target import _floor_rows  # noqa: E402

COMPUTED = ("oracle", "floor_single_instance", "floor_connected_components", "floor_voxel_instances",
            "floor_random_instances")  # fmt: skip
SEMANTIC_GATE = 0.999


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_skeleton(skel: Skeleton, shape) -> dict:
    """The shipped skeleton.npz layout: concatenated coords/edges with per-fiber offsets (see target._unpack_skeleton)."""
    coords, edges, f_off, e_off, ids, names = [], [], [0], [0], [], []
    for fib in skel.fibers:
        coords.append(np.asarray(fib.coords, dtype=np.float32))
        edges.append(np.asarray(fib.edges, dtype=np.int32) + f_off[-1])
        f_off.append(f_off[-1] + len(fib.coords))
        e_off.append(e_off[-1] + len(fib.edges))
        ids.append(fib.id)
        names.append(fib.name)
    return {
        "coords": np.concatenate(coords) if coords else np.zeros((0, 3), np.float32),
        "edges": np.concatenate(edges) if edges else np.zeros((0, 2), np.int32),
        "fiber_offsets": np.asarray(f_off, dtype=np.int64),
        "edge_offsets": np.asarray(e_off, dtype=np.int64),
        "fiber_ids": np.asarray(ids, dtype=np.int64),
        "fiber_names": np.asarray(names, dtype=object),
        "scale_um": np.asarray(skel.scale_um, dtype=np.float64),
        "origin_zyx": np.asarray(skel.origin_zyx, dtype=np.int64),
        "shape": np.asarray(shape, dtype=np.int64),
    }


def packed_equal(a: dict, npz) -> list[str]:
    """Keys whose arrays differ between a packed skeleton and a loaded skeleton.npz."""
    bad = [k for k in a if k not in npz.files]
    bad += [k for k in npz.files if k not in a]
    for k in a:
        if k in npz.files and not (a[k].shape == npz[k].shape and np.array_equal(a[k], npz[k])):
            bad.append(k)
    return bad


def semantic_landing_rate(skel: Skeleton, semantic: np.ndarray) -> float:
    """Fraction of in-bounds NML nodes landing on a fiber-positive voxel (the exporter's convention gate)."""
    hits = total = 0
    for fib in skel.fibers:
        ok = fib.in_bounds_mask(semantic.shape)
        if not ok.any():
            continue
        idx = np.clip(np.rint(fib.coords[ok]).astype(int), 0, np.asarray(semantic.shape) - 1)
        hits += int((semantic[idx[:, 0], idx[:, 1], idx[:, 2]] > 0).sum())
        total += int(ok.sum())
    return hits / total if total else 0.0


def scoring_mask_stats(packed: dict, mask: np.ndarray) -> dict:
    """Bit-for-bit the measurement tests/test_fiber_target.py repeats from the shipped files."""
    idx = np.round(packed["coords"]).astype(int)
    inb = ((idx >= 0) & (idx < np.asarray(mask.shape))).all(axis=1)
    nodes = idx[inb]
    rate = float(mask[nodes[:, 0], nodes[:, 1], nodes[:, 2]].mean())
    density = float(mask.mean())
    return {"measured_node_landing_rate_on_scoring_mask": round(rate, 6), "scoring_mask_density": round(density, 6),
            "landing_enrichment_vs_chance": round(rate / density, 6),
            "in_bounds_node_fraction": round(float(inb.sum() / len(idx)), 6)}  # fmt: skip


def nml_record(nml: Path, dataset: str) -> dict:
    return {"dataset": dataset, "file": nml.name, "sha256": sha256(nml)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--nml", type=Path, required=True)
    ap.add_argument("--dataset", required=True, help="villa fiber-skeletons release, e.g. Dataset004_sk-fibers-binary-20250728")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--semantic", type=Path)
    ap.add_argument("--tracer-labels", type=Path)
    ap.add_argument("--tracer-source")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    t = args.target
    meta = json.loads((t / "meta.json").read_text())
    cube = meta["cube"]
    shape = tuple(meta["shape"])
    skel = parse_nml(args.nml, origin_zyx=origin_from_stem(cube))
    packed = pack_skeleton(skel, shape)
    with np.load(t / "skeleton.npz", allow_pickle=True) as shipped:
        diff = packed_equal(packed, shipped)

    if args.verify_only:
        if diff:
            print(f"{cube}: {args.nml.name} does NOT reproduce the shipped skeleton (differs in {diff})")
            return 1
        meta["ground_truth"]["nml"] = nml_record(args.nml, args.dataset)
        print(f"{cube}: {args.nml.name} reproduces the shipped skeleton exactly; sha256 {meta['ground_truth']['nml']['sha256']}")
        if args.write:
            (t / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return 0

    if not (args.semantic and args.tracer_labels and args.tracer_source):
        ap.error("a revision needs --semantic, --tracer-labels and --tracer-source")
    if not diff:
        print(f"{cube}: {args.nml.name} is the shipped skeleton already; use --verify-only")
        return 1
    import tifffile

    semantic = tifffile.imread(args.semantic)
    if semantic.shape != shape:
        raise SystemExit(f"{cube}: semantic label shape {semantic.shape} != {shape}")
    rate = semantic_landing_rate(skel, semantic)
    if rate < SEMANTIC_GATE:
        print(f"{cube}: nodes land on the semantic label at {rate:.6f} < {SEMANTIC_GATE}; refusing")
        return 1

    with np.load(t / "mask.npz") as mk:  # the mask is unchanged: villa's CT for these cubes did not change
        n = int(np.prod(shape))
        mask = np.unpackbits(mk["packed"])[:n].astype(bool).reshape(shape)
    scored = Skeleton(fibers=[f for f in skel.fibers if f.in_bounds_mask(shape).sum() > 1],
                      scale_um=skel.scale_um, origin_zyx=skel.origin_zyx)  # fmt: skip
    tol = float(meta["tolerance"])
    rows = {"oracle": score_tracing(scored, oracle_from_skeleton(scored, shape), tolerance=tol).as_row()}
    rows.update(_floor_rows(scored, mask, tol))
    rows["tracer_strict_relink"] = score_tracing(scored, np.load(args.tracer_labels), tolerance=tol).as_row()

    new = json.loads(json.dumps(meta))
    gt = new["ground_truth"]
    gt["n_fibers"] = len(skel.fibers)
    gt["n_fibers_scored"] = len(scored.fibers)
    gt["n_nodes_in_bounds"] = int(sum(int(f.in_bounds_mask(shape).sum()) for f in skel.fibers))
    gt["nml"] = nml_record(args.nml, args.dataset)
    gt["source_file"] = f"nml/{args.nml.name}"
    cc = new["convention_check"]
    cc["measured_node_landing_rate_on_semantic_label"] = round(rate, 6)
    cc["verified_against"] = f"labelsTr/{args.semantic.name} ({args.dataset}, the semantic label of the same release)"
    cc.update(scoring_mask_stats(packed, mask))
    for k, row in rows.items():
        new["floors"][k] = row
    new["floors_scoring_version"] = SCORING_VERSION
    new["floors_external_source"]["tracer_strict_relink"] = args.tracer_source

    for k in rows:
        a, b = meta["floors"][k], new["floors"][k]
        changed = {f: (a.get(f), b[f]) for f in b if a.get(f) != b[f]}
        print(f"{cube} {k}: {changed}")
    for k in ("n_fibers", "n_fibers_scored", "n_nodes_in_bounds"):
        print(f"{cube} ground_truth.{k}: {meta['ground_truth'][k]} -> {gt[k]}")
    print(f"{cube} convention_check: semantic {rate:.6f}; " + json.dumps(scoring_mask_stats(packed, mask)))
    if args.write:
        np.savez_compressed(t / "skeleton.npz", **packed)
        (t / "meta.json").write_text(json.dumps(new, indent=2) + "\n")
        print(f"{cube}: written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
