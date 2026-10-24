"""0.9.0: villa's neural tracer with the scan around each cube, a new external row beside the cube-only one.

The rows come from vesuvius-autoresearch findings 97 (256³) and 98 (512³), re-scored here by tools/import_tracer_rows.py
from their labellings. Every claim about them in the README and BASELINES is computed from the shipped rows, except the
dropped seed shares and the tolerance-4 counts, which come from the findings' run records and re-scores (not shipped).
"""

import json
import pathlib
import statistics
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import context_ct  # noqa: E402
import fiber_tables  # noqa: E402

SCAN, ROW, TRACER, CC = fiber_tables.SCAN_CONTEXT, "villa_neural_tracer", "tracer_strict_relink", "floor_connected_components"
PEN, ERL, COV = "erl_merge_penalized", "erl", "coverage"
META = {t.name.removeprefix("fibers_"): json.loads((t / "meta.json").read_text()) for t in sorted((ROOT / "data").glob("fibers_*"))}
README = " ".join(" ".join(line.lstrip("> ") for line in (ROOT / "README.md").read_text().splitlines()).split())
BASELINES = " ".join((ROOT / "baselines" / "BASELINES.md").read_text().split())
NOTE = README.split("**v0.9.0")[1].split("**v0.8.3")[0]

# From the findings' run records (reports/neural_tracer_ct_context.json, reports/neural_tracer_ct_context_512.json):
# the share of seed dots dropped as no_heading, with the cube alone (finding 91's runs) and with the scan around it.
DROPPED = {
    "s1_00497_01497_03997_256": (0.577, 0.010), "s1_00497_02497_02997_256": (0.542, 0.007),
    "s1_00997_02497_02997_256": (0.561, 0.002), "s1_08997_02997_02497_256": (0.558, 0.002),
    "s1_10997_02997_02997_256": (0.607, 0.006), "s5_03997_01497_03997_256": (0.678, 0.010),
    "s5_07997_02997_05497_256": (0.605, 0.004), "s5_14997_01497_01497_256": (0.571, 0.005),
    "s5_13497_01497_02497_256": (0.695, 0.266), "s5_16997_03997_04497_256": (0.696, 0.353),
    "s5_06494_01994_03994_512": (0.283, 0.004), "s5_06994_00994_04994_512": (0.258, 0.012),
    "s5_07994_01994_05494_512": (0.263, 0.004), "s3_00994_00494_01994_512": (0.256, 0.074),
    "s3_00994_01994_00494_512": (0.263, 0.035), "s3_00994_02494_01494_512": (0.357, 0.088),
    "s3_01994_01494_00994_512": (0.345, 0.071), "s3_01994_01994_01494_512": (0.321, 0.062),
    "s3_02994_02494_01994_512": (0.266, 0.011), "s3_03494_01494_00494_512": (0.306, 0.053),
    "s3_04494_00994_01494_512": (0.273, 0.039),
}  # fmt: skip
# Cubes on which it leads our tracer on merge-penalized ERL at tolerance 4 (findings 94, 97, 98): cube alone, with scan.
TOL4_LEADS = (11, 19)

SCROLL3 = [c for c in META if c.startswith("s3_")]
HELD_S5 = [c for c in META if META[c]["split"] == "held_out" and c.startswith("s5_")]
IN_256 = [c for c in META if META[c]["split"] != "held_out" and c.endswith("_256")]
IN_512 = [c for c in META if META[c]["split"] != "held_out" and c.endswith("_512")]


def f(c, who):
    return META[c]["floors"][who]


def med(who, key, cubes):
    return statistics.median(f(c, who)[key] for c in cubes)


def span(vals, fmt="{:.2f}"):
    return f"{fmt.format(min(vals))}–{fmt.format(max(vals))}"


def first(who, c):
    return f(c, who)[PEN] > max(f(c, TRACER)[PEN], f(c, CC)[PEN])


@pytest.mark.parametrize("cube", sorted(META))
def test_every_target_carries_the_row_beside_the_cube_only_one(cube):
    meta = META[cube]
    assert meta["floors_external"] == [TRACER, ROW, SCAN]
    assert meta["floors"][SCAN]["scoring_version"] == meta["floors_scoring_version"] == 3
    assert set(meta["floors"][SCAN]) == set(meta["floors"][ROW])
    # the same ground truth, so only the labelling differs
    for k in ("n_gt_fibers", "gt_length", "tolerance"):
        assert meta["floors"][SCAN][k] == meta["floors"][ROW][k], k
    source = meta["floors_external_source"][SCAN]
    for fact in ("finding 97", "finding 98", "padded by 128 voxels", "20230205180739", "20231117143551", "20241024131838",
                 "tools/context_ct.py", "the box is the cube", "finding 91"):  # fmt: skip
        assert fact in source, fact
    assert "finding 91" in meta["floors_external_source"][ROW]


def test_the_tool_pads_as_the_rows_were_padded():
    assert context_ct.MARGIN == 128 and sorted(DROPPED) == sorted(META)


@pytest.mark.parametrize("size", [256, 512])
@pytest.mark.parametrize("held_out", [False, True])
def test_baselines_carries_the_generated_table(size, held_out):
    table = fiber_tables.entrant_table(size, held_out=held_out, row=SCAN)
    assert table.count("\n") >= 3, "an empty table would pass vacuously"
    assert " ".join(table.split()) in BASELINES


def test_coverage_rises_on_every_cube():
    factor = {c: f(c, SCAN)[COV] / f(c, ROW)[COV] for c in META}
    assert min(factor.values()) > 1
    small = [v for c, v in factor.items() if c.endswith("_256")]
    big = [v for c, v in factor.items() if c.endswith("_512")]
    claim = f"{span(small)} times the cube-only row on the 256³ cubes, and {span(big)} times on the 512³ ones"
    assert claim in NOTE and claim in BASELINES


def test_where_it_ranks_first():
    now = [c for c in META if first(SCAN, c)]
    before = [c for c in META if first(ROW, c)]
    assert sorted(now) == sorted(SCROLL3 + HELD_S5) and sorted(before) == sorted(SCROLL3)
    assert (len(now), len(before)) == (10, 8)
    assert "it ranks first, ahead of our tracer and connected components, on 10 of the 21 cubes" in NOTE
    assert "The cube-only row ranks first on 8." in NOTE
    assert "it is first, ahead of our tracer and connected components, on 10 of the 21 cubes" in BASELINES
    assert "The cube-only row is first on 8." in BASELINES
    assert "it ranks first on merge-penalized ERL on 10 of the 21 cubes, against the cube-only row's 8" in README
    assert f"all 8 Scroll-3 cubes (median {med(SCAN, PEN, SCROLL3):.2f}, against our tracer's {med(TRACER, PEN, SCROLL3):.2f})" in BASELINES


def test_on_scroll3_it_now_covers_more_than_our_tracer():
    assert all(f(c, SCAN)[COV] > f(c, TRACER)[COV] for c in SCROLL3)
    assert not all(f(c, ROW)[COV] > f(c, TRACER)[COV] for c in SCROLL3)
    assert (
        f"covers more than our tracer on all 8 cubes (median {med(SCAN, COV, SCROLL3):.3f}, against "
        f"{med(TRACER, COV, SCROLL3):.3f}; the cube-only row's is {med(ROW, COV, SCROLL3):.3f})" in BASELINES
    )


def test_in_sample_it_still_trails_our_tracer():
    ins = IN_256 + IN_512
    assert len(ins) == 11 and all(f(c, SCAN)[PEN] < f(c, TRACER)[PEN] for c in ins)
    assert (
        f"median {med(SCAN, PEN, IN_256):.2f} against {med(TRACER, PEN, IN_256):.2f} on the 256³ cubes, and "
        f"{med(SCAN, PEN, IN_512):.2f} against {med(TRACER, PEN, IN_512):.2f} on the 512³ ones" in BASELINES
    )
    for text in (NOTE, README):
        assert "still trails our tracer on all 11 in-sample cubes" in text


def test_connected_components_still_leads_on_raw_erl():
    lead = [f(c, CC)[ERL] / f(c, SCAN)[ERL] for c in META]
    assert min(lead) > 1
    assert f"connected components still leads it on every cube ({span(lead, '{:.1f}')}×)" in BASELINES
    assert "connected components still leads it on raw ERL on every cube" in NOTE
    # so CONTRIBUTING's "neither ... clears it" still holds
    assert not [c for c in META if f(c, SCAN)[ERL] > f(c, CC)[ERL] and f(c, SCAN)[PEN] > f(c, CC)[PEN]]


def test_the_recorded_figures():
    ins = IN_256 + IN_512
    assert f"it drops {span([DROPPED[c][1] for c in ins], '{:.3f}')} of its seed dots on the 11 in-sample cubes" in BASELINES
    assert f"{span([DROPPED[c][1] for c in SCROLL3], '{:.3f}')} on Scroll 3" in BASELINES
    h = sorted(DROPPED[c][1] for c in HELD_S5)
    assert f"and {h[0]:.3f} and {h[1]:.3f} on the two held-out Scroll-5 cubes" in BASELINES
    assert f"With the cube alone it dropped {span([v[0] for v in DROPPED.values()])}." in BASELINES
    assert f"it leads our tracer on {TOL4_LEADS[1]} of the 21 cubes, against {TOL4_LEADS[0]} with the cube alone" in BASELINES
