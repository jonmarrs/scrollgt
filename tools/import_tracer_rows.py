"""Maintainer tool: write our tracer's re-measured rows into each fiber target's meta.json.

The tracer runs in vesuvius-autoresearch (it needs a GPU model and the CT cubes, which this repo does not ship), so its
rows cannot be recomputed here. This tool scores the saved instance labellings with this repo's own scorer, so each
row is exactly what `scrollgt score-fibers` reports for that labelling. It stores the row as
`floors.tracer_strict_relink` and lists it in `floors_external` with its provenance.

Usage: python tools/import_tracer_rows.py <dir of <cube>_instances.npy> --source "<provenance text>"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instances_dir", type=Path)
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    for target in sorted((ROOT / "data").glob("fibers_*")):
        cube = target.name.removeprefix("fibers_")
        inst = args.instances_dir / f"{cube}_instances.npy"
        if not inst.exists():
            raise SystemExit(f"missing {inst}; every target must be re-measured or none")
        row = score_fiber_prediction(inst, target)["metrics"]
        assert row["scoring_version"] == SCORING_VERSION
        meta = json.loads((target / "meta.json").read_text())
        meta["floors"][ROW] = row
        meta["floors_not_recomputed"] = [k for k in meta["floors_not_recomputed"] if k != ROW]
        meta["floors_external"] = sorted({*meta.get("floors_external", []), ROW})
        meta["floors_external_source"] = {ROW: args.source}
        (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{cube}: tracer ERL {row['erl']} ERLpen {row['erl_merge_penalized']} coverage {row['coverage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
