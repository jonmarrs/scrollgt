"""Pin the shipped fiber metadata schema, key by key.

This file exists because the exporter has now twice deleted metadata that nothing
tested. The first time it dropped five `convention_check` fields -- the empirical proof
that skeleton and mask share a coordinate frame -- and they had to be restored. The
second time `mask.generated`, the provenance of the reference mask, was added to the
shipped targets by a one-off script that the exporter never learned about, so the next
re-export would have deleted it from all eleven public targets.

The common factor both times was that the *presence* of a key was never asserted
anywhere. A test that checks the fields it happens to read cannot catch a deletion; only
comparing against the complete expected key set can. So these assertions are equality
against a frozen set, not `issubset` -- an added key is a deliberate schema change and
should have to come here and say so.
"""

import json
import pathlib

import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
TARGETS = sorted(DATA.glob("fibers_*"))

CONVENTION_CHECK_KEYS = {
    "claim",
    "measured_node_landing_rate_on_semantic_label",
    "verified_against",
    "measured_node_landing_rate_on_scoring_mask",
    "scoring_mask_density",
    "landing_enrichment_vs_chance",
    "in_bounds_node_fraction",
    "note_on_the_two_rates",
}

MASK_KEYS = {"model", "threshold", "density", "note", "generated"}


def test_there_are_targets_to_check():
    assert len(TARGETS) == 11, [t.name for t in TARGETS]


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_convention_check_carries_every_expected_field(target):
    meta = json.loads((target / "meta.json").read_text())
    got = set(meta["convention_check"])
    assert got == CONVENTION_CHECK_KEYS, (
        f"{target.name}: convention_check key set changed. Missing "
        f"{sorted(CONVENTION_CHECK_KEYS - got)}, unexpected {sorted(got - CONVENTION_CHECK_KEYS)}. "
        "If the exporter dropped a field, restore it there; if the schema really changed, "
        "update CONVENTION_CHECK_KEYS deliberately."
    )


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_mask_block_carries_every_expected_field(target):
    meta = json.loads((target / "meta.json").read_text())
    got = set(meta["mask"])
    assert got == MASK_KEYS, (
        f"{target.name}: mask key set changed. Missing {sorted(MASK_KEYS - got)}, "
        f"unexpected {sorted(got - MASK_KEYS)}. `generated` in particular records that the "
        "probability volume was produced locally rather than downloaded, and has been "
        "silently deleted by a re-export before."
    )


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_mask_provenance_is_populated_not_merely_present(target):
    """A key present with an empty or placeholder value is the same defect as a missing one."""
    mask = json.loads((target / "meta.json").read_text())["mask"]
    assert isinstance(mask["generated"], str) and len(mask["generated"]) > 40
    assert "fiber_hz_vt" in mask["generated"]
    assert "not downloaded" in mask["generated"], (
        f"{target.name}: mask.generated no longer states that the probability volume was "
        "produced locally, which is the whole point of the field"
    )


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_the_two_rates_note_does_not_assert_a_rate_it_cannot_know(target):
    """One cube measures 0.999936, so boilerplate claiming "and is 1.0" was false there.

    The note is identical across targets by construction, so it must describe the bound
    the exporter actually enforces (>= 0.999) rather than a value that happens to hold
    for most cubes.
    """
    meta = json.loads((target / "meta.json").read_text())
    note = meta["convention_check"]["note_on_the_two_rates"]
    rate = meta["convention_check"]["measured_node_landing_rate_on_semantic_label"]
    assert "and is 1.0" not in note, f"{target.name}: measured rate is {rate}, not 1.0"
    assert "0.999" in note
