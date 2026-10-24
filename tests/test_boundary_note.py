"""0.8.3: villa's neural tracer drops any seed whose CT window crosses the edge of the CT it is given, and every entrant
here is given the cube alone. The cost goes with cube size. The note names each 256³ cube's values; nothing scored
changes.

The share of seed dots it dropped (`no_heading`, from its run records) is from vesuvius-autoresearch
`reports/neural_tracer_boundary.json` (finding 96, PR #207, exploratory). Everything else is recomputed from the
shipped rows.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RADIUS = 38  # CT voxels: villa's CT_CONTEXT_RADIUS, a 77-voxel window
GRID_SCALE = 2  # its trace grid at --grid-scale 2
NO_HEADING_256 = {
    "s1_00497_01497_03997_256": 0.577,
    "s1_00497_02497_02997_256": 0.542,
    "s1_00997_02497_02997_256": 0.561,
    "s1_08997_02997_02497_256": 0.558,
    "s1_10997_02997_02997_256": 0.607,
    "s5_03997_01497_03997_256": 0.678,
    "s5_07997_02997_05497_256": 0.605,
    "s5_14997_01497_01497_256": 0.571,
    "s5_13497_01497_02497_256": 0.695,
    "s5_16997_03997_04497_256": 0.696,
}
NO_HEADING_512 = (0.256, 0.357)  # its range on the eleven 512³ cubes
MARK = "Seeding (vesuvius-autoresearch finding 96"
SENTENCE = (
    "On the 256³ cubes this benchmark's protocol keeps it from seeding near the faces (0.8.3, finding 96): it drops any "
    "seed whose 77-voxel CT window crosses the edge of the CT it is given, and it is given the cube alone."
)


def _flat(path):
    """Single-spaced, with blockquote markers dropped (the README's release notes wrap inside `> `)."""
    return " ".join(re.sub(r"^> ?", "", path.read_text(), flags=re.M).split())


def _metas():
    return {t.name.removeprefix("fibers_"): json.loads((t / "meta.json").read_text()) for t in DATA.glob("fibers_*")}


def _span(vals, fmt="{:.2f}"):
    return f"{fmt.format(min(vals))}–{fmt.format(max(vals))}"


def _output(meta):
    row = meta["floors"]["villa_neural_tracer"]
    return row["pred_length"] / row["gt_length"]


def test_the_ten_256_cubes_note_their_own_values():
    metas = _metas()
    small = {c for c, m in metas.items() if m["size_class"] == 256}
    assert small == set(NO_HEADING_256)
    for cube, nh in NO_HEADING_256.items():
        s = " ".join(metas[cube]["floors_external_source"]["villa_neural_tracer"].split())
        assert MARK in s, cube
        assert f"its closest traced seed lay {RADIUS} voxels from a face" in s, cube
        assert (
            f"Here it dropped {nh:.3f} of its seed dots as no_heading, and its output is {_output(metas[cube]):.3f} "
            "of the ground-truth length, against 0.26–0.36 and 0.83–1.43 on the 512³ cubes." in s
        ), cube
        assert "These rows are its score under the protocol every entrant gets" in s, cube
    assert not [c for c, m in metas.items() if c not in small and MARK in m["floors_external_source"].get("villa_neural_tracer", "")]


def test_the_figures_are_the_shipped_rows():
    metas = _metas()
    by = {s: {c: m for c, m in metas.items() if m["size_class"] == s} for s in (256, 512)}
    out = {s: [_output(m) for m in by[s].values()] for s in by}
    ours = {s: [m["floors"]["tracer_strict_relink"]["coverage"] for m in by[s].values()] for s in by}
    assert (len(by[256]), len(by[512])) == (10, 11)
    # the seedable share: a window fits RADIUS / GRID_SCALE grid voxels in from each face
    fits = {s: ((s // GRID_SCALE - 2 * RADIUS // GRID_SCALE) / (s // GRID_SCALE)) ** 3 for s in by}
    nh256 = _span(NO_HEADING_256.values())
    nh512 = _span(NO_HEADING_512)
    readme, base = _flat(ROOT / "README.md"), _flat(ROOT / "baselines/BASELINES.md")
    assert f"A seed's window fits in {fits[256]:.3f} of a 256³ cube and {fits[512]:.3f} of a 512³ one." in base
    assert f"{nh256} of its seed dots on the ten 256³ cubes, against {nh512} on the eleven 512³ ones." in base
    assert f"{_span(out[256])} of the ground-truth length on the 256³ cubes, against {_span(out[512])}." in base
    assert (
        f"On the ten 256³ cubes it drops {nh256} of its seed dots, and its output is {_span(out[256])} of the "
        f"ground-truth length. On the eleven 512³ cubes those are {nh512} and {_span(out[512])}." in readme
    )
    # both separate by size; our tracer's coverage does not
    assert min(NO_HEADING_256.values()) > NO_HEADING_512[1] and max(out[256]) < min(out[512])
    assert not (min(ours[256]) > max(ours[512]) or max(ours[256]) < min(ours[512]))
    assert (
        f"our tracer's coverage does not ({_span(ours[256], '{:.3f}')} on 256³ cubes, {_span(ours[512], '{:.3f}')} "
        "on 512³ ones)" in base
    )
    pen = "erl_merge_penalized"
    trails = [c for c, m in metas.items() if m["floors"]["villa_neural_tracer"][pen] < m["floors"]["tracer_strict_relink"][pen]]
    small = [c for c in trails if metas[c]["size_class"] == 256]
    assert (len(small), len(trails)) == (10, 13)
    for doc in (readme, base):
        assert "Ten of the 13 cubes on which it trails our tracer are 256³." in doc


def test_the_note_sits_where_the_rows_are_described():
    readme, base = _flat(ROOT / "README.md"), _flat(ROOT / "baselines/BASELINES.md")
    # once in the README's entrant summary and once in BASELINES, besides the 0.8.3 release note
    assert readme.count(SENTENCE) == 1 and base.count(SENTENCE) == 1
    assert "v0.8.3 — villa's neural tracer on the 256³ cubes: given the cube's CT alone, it cannot seed near the faces." in readme
    for doc in (readme, base):
        assert "(vesuvius-autoresearch finding 96, exploratory, `reports/neural_tracer_boundary.md`)" in doc
        assert "pre-registered run with real scroll CT around each 256³ cube is pending (vesuvius-autoresearch #209)" in doc
