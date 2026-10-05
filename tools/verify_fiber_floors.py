"""Maintainer tool: verify that each fiber target's published oracle and floor rows reproduce with the current scorer,
and (with --stamp) record the scoring version they were verified under.

A version stamp is a claim that the published numbers come from this scorer, so it is written only after every
recomputed row matches the published one exactly (as rounded by ``ConnectivityScores.as_row``). Rows from other
producers (``tracer_*``) cannot be recomputed from the repo. They are stamped with the version they were last
computed under (1 if unstamped) and listed in ``floors_not_recomputed``.

Usage: python tools/verify_fiber_floors.py [--size 256|512] [--stamp]
Exit status 1 if any published row does not reproduce. Nothing is written in that case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scrollgt.fibers.eval_trace import SCORING_VERSION, oracle_from_skeleton, score_tracing  # noqa: E402
from scrollgt.fibers.target import _floor_rows, load_fiber_target  # noqa: E402

COMPUTED = ("oracle", "floor_single_instance", "floor_connected_components", "floor_voxel_instances",
            "floor_random_instances")  # fmt: skip


def recompute(target: Path) -> tuple[dict, dict]:
    skeleton, mask, meta = load_fiber_target(target)
    tol = float(meta["tolerance"])
    rows = {"oracle": score_tracing(skeleton, oracle_from_skeleton(skeleton, mask.shape), tolerance=tol).as_row()}
    rows.update(_floor_rows(skeleton, mask, tol))
    return meta, rows


def stamped(meta: dict, rows: dict) -> dict:
    floors = {k: ({**rows[k]} if k in COMPUTED else {**v, "scoring_version": v.get("scoring_version", 1)})
              for k, v in meta["floors"].items()}  # fmt: skip
    out = {}
    for k, v in meta.items():
        if k in ("floors_scoring_version", "floors_not_recomputed"):
            continue
        out[k] = floors if k == "floors" else v
        if k == "floors":
            out["floors_scoring_version"] = SCORING_VERSION
            out["floors_not_recomputed"] = sorted(x for x in floors if x not in COMPUTED)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, choices=(256, 512))
    ap.add_argument("--stamp", action="store_true")
    args = ap.parse_args()
    targets = sorted(p for p in (ROOT / "data").glob("fibers_*") if args.size is None or p.name.endswith(f"_{args.size}"))
    bad = 0
    verified = []
    for t in targets:
        t0 = time.time()
        meta, rows = recompute(t)
        verified.append((t, meta, rows))
        for k in COMPUTED:
            pub = {f: v for f, v in meta["floors"][k].items() if f != "scoring_version"}
            new = {f: v for f, v in rows[k].items() if f != "scoring_version"}
            if pub != new:
                bad += 1
                diff = {f: (pub.get(f), new.get(f)) for f in set(pub) | set(new) if pub.get(f) != new.get(f)}
                print(f"MISMATCH {t.name} {k}: {diff}", flush=True)
        print(f"{t.name}: {'ok' if not bad else 'MISMATCH'} ({time.time() - t0:.0f}s)", flush=True)
    print(f"scoring version {SCORING_VERSION}: {len(targets)} targets, {bad} mismatched rows", flush=True)
    if args.stamp and not bad:  # all-or-nothing: stamp only once every target has verified
        for t, meta, rows in verified:
            (t / "meta.json").write_text(json.dumps(stamped(meta, rows), indent=2) + "\n")
        print(f"stamped {len(verified)} targets", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
