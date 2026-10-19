"""Maintainer tool: write an external entrant's re-measured rows into each fiber target's meta.json.

External entrants (our tracer, villa's neural tracer) run elsewhere: they need a GPU model and the CT cubes, which
this repo does not ship, so their rows cannot be recomputed here. This tool scores the saved instance labellings with
this repo's own scorer, so each row is exactly what `scrollgt score-fibers` reports for that labelling. It stores the
row as `floors.<row>` and lists it in `floors_external` with its provenance, beside any entrant already there.

Usage: python tools/import_tracer_rows.py <dir of <cube>_instances.npy> --source "<provenance text>" [--row <name>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scrollgt.fibers.eval_trace import SCORING_VERSION  # noqa: E402
from scrollgt.fibers.target import score_fiber_prediction  # noqa: E402

ROW = "tracer_strict_relink"


def import_rows(data_dir: Path, instances_dir: Path, row: str, source: str) -> None:
    targets = sorted(data_dir.glob("fibers_*"))
    labellings = {t: instances_dir / f"{t.name.removeprefix('fibers_')}_instances.npy" for t in targets}
    missing = [str(p) for p in labellings.values() if not p.exists()]
    if missing:  # checked before any target is written
        raise SystemExit(f"missing {missing}; every target must be re-measured or none")
    for target, inst in labellings.items():
        scored = score_fiber_prediction(inst, target)["metrics"]
        assert scored["scoring_version"] == SCORING_VERSION
        meta = json.loads((target / "meta.json").read_text())
        meta["floors"][row] = scored
        meta["floors_not_recomputed"] = [k for k in meta["floors_not_recomputed"] if k != row]
        meta["floors_external"] = sorted({*meta.get("floors_external", []), row})
        meta["floors_external_source"] = {**meta.get("floors_external_source", {}), row: source}
        (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        cube = target.name.removeprefix("fibers_")
        print(f"{cube}: {row} ERL {scored['erl']} ERLpen {scored['erl_merge_penalized']} coverage {scored['coverage']}")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("instances_dir", type=Path)
    ap.add_argument("--source", required=True)
    ap.add_argument("--row", default=ROW, help=f"the row's name in floors (default {ROW})")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    import_rows(ROOT / "data", args.instances_dir, args.row, args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
