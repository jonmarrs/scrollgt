"""0.8.2: villa's neural tracer's in-sample deficit is a tolerance-2 result on the 512³ cubes (vesuvius-autoresearch
finding 94, pre-registered). Scores here stay at tolerance 2; the note sits where the deficit is claimed.

Values from vesuvius-autoresearch `reports/tolerance4_rescore.json` (PR #162): merge-penalized ERL at tolerance 4,
villa's neural tracer against our tracer.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TOL4 = {
    "s5_06494_01994_03994_512": (65.29, 59.43),
    "s5_06994_00994_04994_512": (42.72, 37.92),
    "s5_07994_01994_05494_512": (49.50, 43.69),
}
MEDIAN_512 = (49.50, 43.69)
S3_RATIO = (2.22, 1.26)  # neural over our tracer on Scroll 3: tolerance 4, tolerance 2
MARK = "Tolerance (vesuvius-autoresearch finding 94"


def _flat(path):
    return " ".join(re.sub(r"^> ?", "", path.read_text(), flags=re.M).split())


def _source(target):
    meta = json.loads((target / "meta.json").read_text())
    return meta.get("floors_external_source", {}).get("villa_neural_tracer", "")


def test_the_three_cubes_note_their_tolerance_4_scores():
    for cube, (neural, ours) in TOL4.items():
        s = " ".join(_source(DATA / f"fibers_{cube}").split())
        assert MARK in s, cube
        assert f"at tolerance 4 it leads our tracer here ({neural:.2f} against {ours:.2f})" in s, cube
        assert "These rows stay at tolerance 2." in s, cube


def test_no_other_cube_carries_it():
    noted = {t.name.removeprefix("fibers_") for t in DATA.glob("fibers_*") if MARK in _source(t)}
    assert noted == set(TOL4)


def test_the_deficit_claims_carry_the_note():
    sentence = (
        f"Its in-sample deficit is a tolerance-2 result on the 512³ cubes (0.8.2, finding 94): at tolerance 4 it leads "
        f"our tracer on all three in-sample 512³ cubes (median {MEDIAN_512[0]:.2f} against {MEDIAN_512[1]:.2f}), "
        "and still trails on all eight in-sample 256³ cubes and both held-out Scroll-5 cubes. Its Scroll-3 lead holds "
        f"at tolerance 4 and widens (a median {S3_RATIO[0]:.2f} times our tracer, against {S3_RATIO[1]:.2f})."
    )
    # Twice in the README (the 0.8.2 note and the entrant summary), once in BASELINES.
    for doc, n in ((ROOT / "README.md", 2), (ROOT / "baselines/BASELINES.md", 1)):
        assert _flat(doc).count(sentence) == n, doc.name
    assert "villa's neural tracer's in-sample deficit is a tolerance-2 result on the 512³ cubes" in _flat(ROOT / "README.md")
