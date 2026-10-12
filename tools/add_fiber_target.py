"""Maintainer tool: add a fiber target from a villa NML, its semantic label, a fiber_hz_vt probability volume and an
external tracer labelling.

Writes ``data/fibers_<cube>/`` with ``skeleton.npz``, ``mask.npz`` and a ``meta.json`` in the current schema. The
boilerplate (convention claim, notes, mask provenance) is copied from an existing target, so it cannot drift.

Gates, each refusing to write:
* the NML's in-bounds nodes land on villa's semantic label at >= 0.999 (the exporter's convention gate);
* they land on the P >= 0.5 mask at more than 5x its density (the frame test tests/test_fiber_target.py repeats);
* with ``--expect-rows``, every recomputed row (oracle, four floors, tracer) equals the row a registered analysis
  published for this cube, field for field. A row published with scoring version 2 is compared through the version-2
  values a version-3 row carries (``erl_covered``, ``erl_merge_penalized_covered``).

Dry run unless --write is given.

Usage:
  python tools/add_fiber_target.py --cube <stem> --nml PATH --dataset NAME --semantic PATH --fiberprob PATH \
      --tracer-labels PATH --tracer-source TEXT --split held_out --split-note TEXT --scroll TEXT \
      [--expect-rows RESULT.json] [--write]

Reading the semantic TIFF needs ``tifffile``, a maintainer-only dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from update_fiber_ground_truth import (  # noqa: E402
    SEMANTIC_GATE,
    nml_record,
    pack_skeleton,
    scoring_mask_stats,
    semantic_landing_rate,
)

from scrollgt.fibers.eval_trace import SCORING_VERSION, oracle_from_skeleton, score_tracing  # noqa: E402
from scrollgt.fibers.skeleton_io import Skeleton, origin_from_stem, parse_nml, size_from_stem  # noqa: E402
from scrollgt.fibers.target import _floor_rows  # noqa: E402
from migrate_fiber_rows_v3 import mismatches  # noqa: E402

TEMPLATE = ROOT / "data" / "fibers_s5_14997_01497_01497_256"
ENRICHMENT_GATE = 5.0
THRESHOLD = 0.5
TOLERANCE = 2.0
VOXEL_UM = 7.91
# The registered analysis names our tracer "tracer"; ScrollGT ships it as an external floor.
EXPECTED_KEY = {"tracer_strict_relink": "tracer"}


def build(args) -> tuple[dict, dict, np.ndarray, list[str]]:
    """The meta, packed skeleton and mask for one cube, plus every gate or fidelity failure."""
    import tifffile

    cube = args.cube
    size = size_from_stem(cube)
    shape = (size,) * 3
    tmpl = json.loads((TEMPLATE / "meta.json").read_text())
    skel = parse_nml(args.nml, origin_zyx=origin_from_stem(cube))
    if skel.scale_um and any(abs(s - VOXEL_UM) > 1e-6 for s in skel.scale_um):
        return {}, {}, np.zeros(0), [f"{cube}: NML scale {skel.scale_um} is not {VOXEL_UM} um"]
    prob = np.load(args.fiberprob)
    semantic = tifffile.imread(args.semantic)
    if prob.shape != shape or semantic.shape != shape:
        return {}, {}, np.zeros(0), [f"{cube}: expected {shape}, got prob {prob.shape}, semantic {semantic.shape}"]
    mask = prob >= THRESHOLD
    packed = pack_skeleton(skel, shape)
    problems = []
    rate = semantic_landing_rate(skel, semantic)
    if rate < SEMANTIC_GATE:
        problems.append(f"{cube}: nodes land on the semantic label at {rate:.6f} < {SEMANTIC_GATE}")
    stats = scoring_mask_stats(packed, mask)
    if not stats["landing_enrichment_vs_chance"] > ENRICHMENT_GATE:
        problems.append(f"{cube}: mask enrichment {stats['landing_enrichment_vs_chance']} <= {ENRICHMENT_GATE}")

    scored = Skeleton(fibers=[f for f in skel.fibers if f.in_bounds_mask(shape).sum() > 1],
                      scale_um=skel.scale_um, origin_zyx=skel.origin_zyx)  # fmt: skip
    rows = {"oracle": score_tracing(scored, oracle_from_skeleton(scored, shape), tolerance=TOLERANCE).as_row()}
    rows.update(_floor_rows(scored, mask, TOLERANCE))
    rows["tracer_strict_relink"] = score_tracing(scored, np.load(args.tracer_labels), tolerance=TOLERANCE).as_row()
    if args.expect_rows:
        published = json.loads(args.expect_rows.read_text())["cubes"][cube]["rows"]
        for k, row in rows.items():
            want = published[EXPECTED_KEY.get(k, k)]
            if want.get("scoring_version") == 2:  # a row registered before 0.7.0: compare its version-2 fields
                diff = mismatches(want, row)
            else:
                diff = {f: (want.get(f), row[f]) for f in set(row) | set(want) if want.get(f) != row.get(f)}
            if diff:
                problems.append(f"{cube} {k}: differs from the published row {diff}")

    gt_t = tmpl["ground_truth"]
    meta = {
        "target_id": f"fibers_{cube}",
        "family": tmpl["family"],
        "cube": cube,
        "scroll": args.scroll,
        "split": args.split,
        "size_class": size,
        "split_note": args.split_note,
        "shape": list(shape),
        "origin_zyx": list(origin_from_stem(cube)),
        "voxel_size_um": VOXEL_UM,
        "tolerance": TOLERANCE,
        "ground_truth": {
            "origin": gt_t["origin"],
            "source_file": f"nml/{args.nml.name}",
            "n_fibers": len(skel.fibers),
            "n_fibers_scored": len(scored.fibers),
            "n_nodes_in_bounds": int(sum(int(f.in_bounds_mask(shape).sum()) for f in skel.fibers)),
            "note": gt_t["note"],
            "nml": nml_record(args.nml, args.dataset),
        },
        "convention_check": {
            "claim": tmpl["convention_check"]["claim"],
            "measured_node_landing_rate_on_semantic_label": round(rate, 6),
            "verified_against": f"labelsTr/{args.semantic.name} ({args.dataset}, the semantic label of the same release)",
            **stats,
            "note_on_the_two_rates": tmpl["convention_check"]["note_on_the_two_rates"],
        },
        "mask": {**tmpl["mask"], "density": round(float(mask.mean()), 6)},
        "floors": rows,
        "floors_scoring_version": SCORING_VERSION,
        "floors_not_recomputed": [],
        "floors_external": ["tracer_strict_relink"],
        "floors_external_source": {"tracer_strict_relink": args.tracer_source},
    }
    return meta, packed, mask, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--nml", type=Path, required=True)
    ap.add_argument("--dataset", required=True, help="villa fiber-skeletons release, e.g. Dataset004_sk-fibers-binary-20250728")
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--fiberprob", type=Path, required=True)
    ap.add_argument("--tracer-labels", type=Path, required=True)
    ap.add_argument("--tracer-source", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--split-note", required=True)
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--expect-rows", type=Path)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    meta, packed, mask, problems = build(args)
    if problems:
        print("\n".join(problems))
        print(f"{args.cube}: refusing")
        return 1
    f = meta["floors"]
    print(f"{args.cube}: {meta['ground_truth']['n_fibers']} fibers ({meta['ground_truth']['n_fibers_scored']} scored), "
          f"semantic {meta['convention_check']['measured_node_landing_rate_on_semantic_label']}, enrichment "
          f"{meta['convention_check']['landing_enrichment_vs_chance']}, cc ERL {f['floor_connected_components']['erl']}, "
          f"tracer ERLpen {f['tracer_strict_relink']['erl_merge_penalized']}"
          + (", rows equal the published ones" if args.expect_rows else ""))  # fmt: skip
    if args.write:
        out = args.out_root / f"fibers_{args.cube}"
        out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(out / "skeleton.npz", **packed)
        np.savez_compressed(out / "mask.npz", packed=np.packbits(mask), shape=np.asarray(mask.shape, dtype=np.int64))
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{args.cube}: written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
