"""villa's neural fiber tracer (0.8.0): its rows are external, carry their provenance, and every claim about them in
the README and BASELINES is computed from the shipped rows.

The rows come from vesuvius-autoresearch finding 91's labellings, re-scored here by tools/import_tracer_rows.py. Like
our tracer's, they cannot be recomputed from this repository: the tracer needs a GPU model and the CT cubes.
"""

import json
import pathlib
import statistics
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fiber_tables  # noqa: E402

ROW, TRACER, CC = "villa_neural_tracer", "tracer_strict_relink", "floor_connected_components"
PEN, ERL, COV = "erl_merge_penalized", "erl", "coverage"
TARGETS = sorted((ROOT / "data").glob("fibers_*"))
META = {t.name.removeprefix("fibers_"): json.loads((t / "meta.json").read_text()) for t in TARGETS}
# Release notes are blockquotes: drop the "> " that starts each of their lines before joining them.
README = " ".join(" ".join(line.lstrip("> ") for line in (ROOT / "README.md").read_text().splitlines()).split())
BASELINES = " ".join((ROOT / "baselines" / "BASELINES.md").read_text().split())
CHECKPOINT_SHA256 = "3268a0ac1807debdeaded05fa7bc90b82d3ca8102125247e5cba688a10a4811f"


def rows(who: str, cubes) -> list[dict]:
    return [META[c]["floors"][who] for c in cubes]


def med(who: str, key: str, cubes) -> float:
    return statistics.median(r[key] for r in rows(who, cubes))


SCROLL3 = [c for c in META if c.startswith("s3_")]
HELD_S5 = [c for c in META if META[c]["split"] == "held_out" and c.startswith("s5_")]
IN_256 = [c for c in META if META[c]["split"] != "held_out" and c.endswith("_256")]
IN_512 = [c for c in META if META[c]["split"] != "held_out" and c.endswith("_512")]


def test_the_groups():
    assert (len(SCROLL3), len(HELD_S5), len(IN_256), len(IN_512)) == (8, 2, 8, 3)
    assert all(META[c]["split"] == "held_out" and c.endswith("_512") for c in SCROLL3)


@pytest.mark.parametrize("cube", sorted(META))
def test_every_target_carries_the_row_as_an_external_entrant(cube):
    meta = META[cube]
    assert meta["floors_external"][:2] == [TRACER, ROW]  # 0.9.0 adds villa_neural_tracer_scan_context after them
    assert meta["floors"][ROW]["scoring_version"] == meta["floors_scoring_version"] == 3
    source = meta["floors_external_source"][ROW]
    for fact in ("finding 91", "villa #2045", "ccb431818", "fiber_single_small", CHECKPOINT_SHA256,
                 "--ct-level 0 --ct-grid-scale 1 --grid-scale 2 --base-um 7.91", "no Scroll 3 or Scroll 5"):  # fmt: skip
        assert fact in source, fact
    # Our tracer's provenance survived the import beside it.
    assert "vesuvius-autoresearch finding 8" in meta["floors_external_source"][TRACER]


@pytest.mark.parametrize("size", [256, 512])
@pytest.mark.parametrize("held_out", [False, True])
def test_baselines_carries_the_generated_entrant_table(size, held_out):
    table = fiber_tables.entrant_table(size, held_out=held_out)
    assert table.count("\n") >= 3, "an empty table would pass vacuously"
    assert " ".join(table.split()) in BASELINES


def test_on_scroll3_it_ranks_first_on_merge_penalized_erl():
    n, t, c = rows(ROW, SCROLL3), rows(TRACER, SCROLL3), rows(CC, SCROLL3)
    assert all(a[PEN] > b[PEN] for a, b in zip(n, t)) and all(a[PEN] > b[PEN] for a, b in zip(n, c))
    ratio = [a[PEN] / b[PEN] for a, b in zip(n, t)]
    pen = f"median {med(ROW, PEN, SCROLL3):.2f} against {med(TRACER, PEN, SCROLL3):.2f}"
    cov = (f"median coverage {med(ROW, COV, SCROLL3):.3f}, against our tracer's {med(TRACER, COV, SCROLL3):.3f} and "
           f"connected components' {med(CC, COV, SCROLL3):.3f}")  # fmt: skip
    assert f"above our tracer on all 8 cubes ({min(ratio):.2f}–{max(ratio):.2f}×, {pen})" in BASELINES
    assert f"above connected components on all 8 (median {med(CC, PEN, SCROLL3):.2f})" in BASELINES
    assert cov in BASELINES
    assert (
        f"(median {med(ROW, PEN, SCROLL3):.2f}, against our tracer's {med(TRACER, PEN, SCROLL3):.2f} and connected "
        f"components' {med(CC, PEN, SCROLL3):.2f}), with less coverage (median {med(ROW, COV, SCROLL3):.3f}, against "
        f"{med(TRACER, COV, SCROLL3):.3f} and {med(CC, COV, SCROLL3):.3f})" in README
    )


def test_connected_components_still_leads_on_raw_erl_everywhere():
    """So it does not clear the bar of beating connected components on both metrics."""
    lead = [META[c]["floors"][CC][ERL] / META[c]["floors"][ROW][ERL] for c in META]
    assert min(lead) > 1
    s3 = [META[c]["floors"][CC][ERL] / META[c]["floors"][ROW][ERL] for c in SCROLL3]
    assert f"still leads on raw ERL on every cube ({min(s3):.1f}–{max(s3):.1f}×)" in BASELINES
    assert "neither our own tracer nor villa's neural tracer clears it" in " ".join(
        (ROOT / "CONTRIBUTING.md").read_text().split()
    )


def test_on_the_other_13_cubes_it_trails_our_tracer():
    others = IN_256 + IN_512 + HELD_S5
    assert len(others) == 13
    assert all(META[c]["floors"][ROW][PEN] < META[c]["floors"][TRACER][PEN] for c in others)
    assert sum(c.startswith("s1_") for c in others) == 5  # "including the five from Scroll 1"
    assert (
        f"median {med(ROW, PEN, IN_256):.2f} against {med(TRACER, PEN, IN_256):.2f} on the 8 in-sample 256³ cubes, "
        f"and {med(ROW, PEN, IN_512):.2f} against {med(TRACER, PEN, IN_512):.2f} on the 3 in-sample 512³ ones"
        in BASELINES
    )
    a, b = (META[c]["floors"] for c in sorted(HELD_S5))
    assert f"its coverage is {a[ROW][COV]:.3f} and {b[ROW][COV]:.3f}" in BASELINES
    leads = [f for f in (a, b) if f[ROW][PEN] > f[CC][PEN]]
    assert len(leads) == 1
    (f,) = leads
    assert f"one of them ({f[ROW][PEN]:.2f} against {f[CC][PEN]:.2f})" in BASELINES


def test_nothing_says_every_entrant_labels_the_shared_mask():
    """Up to 0.7.0 every entrant was a labelling of the shipped mask, and the docs said so of all of them. villa's
    neural tracer labels its own centrelines, so that sentence, in the README, CONTRIBUTING and each target's mask
    note, now has to name the entrants it covers."""
    contributing = " ".join((ROOT / "CONTRIBUTING.md").read_text().split())
    for text in (README, contributing, *(m["mask"]["note"] for m in META.values())):
        assert "every entrant" not in text.lower() or "every entrant built on" in text.lower()
        assert "every published row is scored against that same mask" not in text
    for m in META.values():
        assert "villa's neural tracer" in m["mask"]["note"] and "its own centrelines" in m["mask"]["note"]
    assert "villa's neural tracer labels its own centrelines" in contributing


# vesuvius-autoresearch finding 92 (reports/neural_tracer_pitch.json at fd45cea3, pre-registered at 5c052f7c): each cube
# resampled to 9.6 µm, the pitch of the tracer's Scroll 1 source, traced with finding 91's settings, mapped back and
# scored here in the cube's own frame. Merge-penalized ERL by cube. Its native rows and our tracer's in that report equal
# this repository's exactly; these 21 values are the only numbers it adds.
PITCH_9P6_PEN = {
    "s1_00497_01497_03997_256": 4.28, "s1_00497_02497_02997_256": 6.83, "s1_00997_02497_02997_256": 7.69,
    "s1_08997_02997_02497_256": 4.98, "s1_10997_02997_02997_256": 9.84, "s3_00994_00494_01994_512": 13.79,
    "s3_00994_01994_00494_512": 14.46, "s3_00994_02494_01494_512": 10.66, "s3_01994_01494_00994_512": 14.06,
    "s3_01994_01994_01494_512": 13.69, "s3_02994_02494_01994_512": 17.02, "s3_03494_01494_00494_512": 14.61,
    "s3_04494_00994_01494_512": 11.94, "s5_03997_01497_03997_256": 4.26, "s5_06494_01994_03994_512": 8.08,
    "s5_06994_00994_04994_512": 5.8, "s5_07994_01994_05494_512": 6.95, "s5_07997_02997_05497_256": 7.33,
    "s5_13497_01497_02497_256": 3.61, "s5_14997_01497_01497_256": 3.2, "s5_16997_03997_04497_256": 1.62,
}
IN_SAMPLE = IN_256 + IN_512


def test_the_scroll3_lead_is_annotated_with_its_pitch_dependence():
    """0.8.1: resampled to its 9.6 µm training pitch it scores lower on every cube and leads our tracer on no Scroll-3
    cube. The published rows stay (they are the native run, the higher on every cube), and the README, BASELINES and
    each target's provenance say the lead depends on the pitch."""
    assert sorted(PITCH_9P6_PEN) == sorted(META)
    ratio = [PITCH_9P6_PEN[c] / META[c]["floors"][ROW][PEN] for c in sorted(META)]
    assert len(ratio) == 21 and max(ratio) < 1
    assert len(SCROLL3) == 8 and not any(PITCH_9P6_PEN[c] > META[c]["floors"][TRACER][PEN] for c in SCROLL3)
    median = f"{statistics.median(ratio):.2f}"
    claim = (
        "Resampled to its 9.6 µm training pitch, it scores lower on all 21 cubes (merge-penalized ERL a median "
        f"{median} of the published rows) and leads our tracer on none of the 8 Scroll-3 cubes"
    )
    readme_body = README.split("## Roadmap")[0].split("**villa's neural fiber tracer (0.8.0)** is the third entrant")[1]
    note_081 = README.split("**v0.8.1")[1].split("**v0.8.0")[0]
    for name, text in (("BASELINES", BASELINES), ("README entrant summary", readme_body), ("README 0.8.1", note_081)):
        assert "Its Scroll-3 lead depends on running at the cubes' own 7.91 µm" in text, name
        assert claim in text, name
    assert f"Per cube the ratio is {min(ratio):.2f}–{max(ratio):.2f}" in note_081
    # Pitch is not the main cause of the in-sample deficit (the registered reading).
    assert all(PITCH_9P6_PEN[c] < META[c]["floors"][TRACER][PEN] for c in IN_SAMPLE) and len(IN_SAMPLE) == 11
    for text in (BASELINES, note_081):
        assert "at 9.6 µm it still trails our tracer on all 11 in-sample cubes" in text
    assert round(100 * (9.6 / 7.91 - 1)) == 21 and "each is 21% longer in physical units" in note_081
    for cube, meta in META.items():
        source = meta["floors_external_source"][ROW]
        for fact in ("finding 92", "reports/neural_tracer_pitch.md", "5c052f7c", "Scroll-3 lead depends on running at "
                     "the cubes' own 7.91 um", f"a median {median} of these rows", "none of the 8 Scroll-3 cubes"):  # fmt: skip
            assert fact in source, (cube, fact)
