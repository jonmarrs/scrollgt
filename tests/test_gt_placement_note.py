"""0.8.2: on five in-sample fiber cubes the hand-traced centrelines sit about a voxel off the CT's fibers. The note
names each cube's measured values; nothing scored changes.

Values from vesuvius-autoresearch `reports/neural_tracer_gt_offset.json` (PR #157, merged 4f1deec2): the mean of the
two thin entrants' perpendicular-median shifts (z, y, x), the CT's offset along it (t* times its length), and villa's
neural tracer's share within 2 voxels before and after moving it by #1934's generator's shift.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

OFFSET = {
    "s1_00497_01497_03997_256": ((-1.09, 0.43, 0.90), 0.92, 0.450, 0.523),
    "s1_08997_02997_02497_256": ((0.55, 0.63, 0.58), 0.51, 0.572, 0.630),
    "s5_14997_01497_01497_256": ((0.65, 0.27, 0.89), 0.85, 0.588, 0.660),
    "s5_06494_01994_03994_512": ((0.71, 0.29, 0.98), 0.93, 0.551, 0.608),
    "s5_06994_00994_04994_512": ((1.01, 0.42, 1.07), 0.77, 0.497, 0.583),
}
# Each shift's length, from the unrounded components (the rounded ones above give 1.02 for s1_08997).
LENGTH = {
    "s1_00497_01497_03997_256": 1.47,
    "s1_08997_02997_02497_256": 1.01,
    "s5_14997_01497_01497_256": 1.14,
    "s5_06494_01994_03994_512": 1.25,
    "s5_06994_00994_04994_512": 1.53,
}


def _meta(target):
    return json.loads((DATA / f"fibers_{target}" / "meta.json").read_text())


def _flat(path):
    """Single-spaced, with blockquote markers dropped (the README's release notes wrap inside `> `)."""
    return " ".join(re.sub(r"^> ?", "", path.read_text(), flags=re.M).split())


def test_exactly_the_five_cubes_carry_the_note():
    noted = {t.name.removeprefix("fibers_") for t in DATA.glob("fibers_*") if "placement_note" in _meta(t.name.removeprefix("fibers_"))["ground_truth"]}
    assert noted == set(OFFSET)


def test_each_note_states_its_cubes_measurement():
    for cube, ((z, y, x), ct, before, after) in OFFSET.items():
        note = " ".join(_meta(cube)["ground_truth"]["placement_note"].split())
        assert f"a mean shift of (z, y, x) = ({z:.2f}, {y:.2f}, {x:.2f}) voxels" in note, cube
        assert f"brightest {ct:.2f} voxels off the hand-traced centreline" in note, cube
        assert f"from {before:.3f} to {after:.3f}" in note, cube
        assert "vesuvius-autoresearch PR #157" in note and "The ground truth and every row are unchanged." in note


def test_nothing_scored_changes():
    """The note sits beside the ground truth; the NML, its hash and every published row are as before."""
    for cube in OFFSET:
        gt = _meta(cube)["ground_truth"]
        assert set(gt) >= {"origin", "source_file", "nml", "placement_note"}
        assert len(gt["nml"]["sha256"]) == 64


def test_the_release_note_and_baselines_give_the_ranges():
    shifts = list(LENGTH.values())
    assert all(abs(LENGTH[c] - sum(v * v for v in s) ** 0.5) < 0.01 for c, (s, _, _, _) in OFFSET.items())
    cts = [c for _, c, _, _ in OFFSET.values()]
    gains = [a - b for _, _, b, a in OFFSET.values()]
    rng = f"{min(shifts):.2f}–{max(shifts):.2f}"
    for doc in (_flat(ROOT / "README.md"), _flat(ROOT / "baselines/BASELINES.md")):
        assert f"agree on a shift of {rng} voxels" in doc
        assert f"puts the fibers' centres {min(cts):.2f}–{max(cts):.2f} voxels off" in doc
        assert f"by {min(gains):.2f}–{max(gains):.2f} on these cubes" in doc
        assert "No row changes." in doc or "No row or ground truth changes." in doc
    assert "> **v0.8.2 —" in (ROOT / "README.md").read_text()
