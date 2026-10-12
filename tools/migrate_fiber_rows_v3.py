"""Maintainer tool: re-score every published fiber row with scoring version 3 (ScrollGT 0.7.0).

Version 3 changes only the ERL denominator, so a version-3 row carries the version-2 values under new names
(``erl_covered``, ``erl_merge_penalized_covered``) beside unchanged coverage, precision, splits and merges. A target is
rewritten only if, for every row, those carried values equal the published version-2 row exactly (as rounded by
``ConnectivityScores.as_row``). That is the stamp's evidence: the new rows come from the scorer that produced the old.

The oracle and floor rows are recomputed from the shipped mask. The tracer row cannot be (its tracer needs a GPU model
and the CT cube), so its saved instance labelling is re-scored, found as ``<cube>_instances.npy`` in a --tracer-dir.

Usage: python tools/migrate_fiber_rows_v3.py --tracer-dir DIR [--tracer-dir DIR ...] [--size 256|512] [cube ...]
Targets already at version 3 are skipped, so a run can be resumed. Exit status 1 if any row does not reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scrollgt.fibers.eval_trace import SCORING_VERSION, score_tracing  # noqa: E402
from scrollgt.fibers.target import load_fiber_target  # noqa: E402
from verify_fiber_floors import recompute  # noqa: E402

TRACER = "tracer_strict_relink"
RENAMED = {"erl_covered": "erl", "erl_merge_penalized_covered": "erl_merge_penalized"}
NEW = ("scoring_version", "erl", "erl_merge_penalized", *RENAMED)
NOTE = (" Re-scored from the same labelling with scoring version 3 in ScrollGT 0.7.0; its version-2 row reproduces "
        "exactly as erl_covered and erl_merge_penalized_covered.")  # fmt: skip


def mismatches(old: dict, new: dict) -> dict:
    """Fields where the new row, read as a version-2 row, differs from the published one."""
    as_v2 = {**{f: v for f, v in new.items() if f not in NEW}, **{v2: new[v3] for v3, v2 in RENAMED.items()}}
    pub = {f: v for f, v in old.items() if f != "scoring_version"}
    return {f: (pub.get(f), as_v2.get(f)) for f in set(pub) | set(as_v2) if pub.get(f) != as_v2.get(f)}


def migrate(target: Path, tracer_dirs: list[Path]) -> tuple[dict | None, list[str]]:
    meta, rows = recompute(target)
    cube = target.name.removeprefix("fibers_")
    found = [d / f"{cube}_instances.npy" for d in tracer_dirs if (d / f"{cube}_instances.npy").exists()]
    if len(found) != 1:
        return None, [f"{cube}: expected one {cube}_instances.npy across --tracer-dir, found {len(found)}"]
    skeleton, mask, _ = load_fiber_target(target)
    labels = np.load(found[0])
    if labels.shape != mask.shape:
        return None, [f"{cube}: tracer labelling shape {labels.shape} != cube {mask.shape}"]
    rows[TRACER] = score_tracing(skeleton, labels, tolerance=float(meta["tolerance"])).as_row()
    if set(rows) != set(meta["floors"]):
        return None, [f"{cube}: rows {sorted(rows)} != published {sorted(meta['floors'])}"]
    errors = []
    for key, old in meta["floors"].items():
        if old.get("scoring_version") != 2:
            errors.append(f"{cube} {key}: published row is version {old.get('scoring_version')}, not 2")
        elif diff := mismatches(old, rows[key]):
            errors.append(f"{cube} {key}: {diff}")
    if errors:
        return None, errors
    out = {**meta, "floors": {k: rows[k] for k in meta["floors"]}, "floors_scoring_version": SCORING_VERSION}
    out["floors_external_source"] = {k: v + NOTE for k, v in meta["floors_external_source"].items()}
    return out, []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer-dir", type=Path, action="append", required=True)
    ap.add_argument("--size", type=int, choices=(256, 512))
    ap.add_argument("cubes", nargs="*")
    args = ap.parse_args()
    targets = sorted(p for p in (ROOT / "data").glob("fibers_*")
                     if (args.size is None or p.name.endswith(f"_{args.size}"))
                     and (not args.cubes or p.name.removeprefix("fibers_") in args.cubes))  # fmt: skip
    bad = 0
    for t in targets:
        if json.loads((t / "meta.json").read_text())["floors_scoring_version"] == SCORING_VERSION:
            print(f"{t.name}: already version {SCORING_VERSION}, skipped", flush=True)
            continue
        t0 = time.time()
        out, errors = migrate(t, args.tracer_dir)
        for e in errors:
            print(f"MISMATCH {e}", flush=True)
        if out is None:
            bad += 1
            continue
        (t / "meta.json").write_text(json.dumps(out, indent=2) + "\n")
        tr = out["floors"][TRACER]
        print(f"{t.name}: migrated ({time.time() - t0:.0f}s); tracer ERL {tr['erl']} ERLpen {tr['erl_merge_penalized']}",
              flush=True)  # fmt: skip
    print(f"scoring version {SCORING_VERSION}: {len(targets)} targets, {bad} refused", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
