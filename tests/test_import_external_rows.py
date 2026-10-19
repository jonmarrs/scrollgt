"""tools/import_tracer_rows.py adds an external entrant's rows without disturbing the ones already published.

Until 0.8.0 it wrote one row name, `tracer_strict_relink`, and replaced `floors_external_source` outright. Adding a
second entrant that way would have erased the first one's provenance from every target.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_tracer_rows", ROOT / "tools" / "import_tracer_rows.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

CUBE = "s1_00497_01497_03997_256"


def _row(erl: float) -> dict:
    return {"scoring_version": tool.SCORING_VERSION, "erl": erl, "erl_merge_penalized": erl, "coverage": 0.5}


@pytest.fixture
def data(tmp_path, monkeypatch):
    """One shipped target's meta.json, and a stand-in scorer, so only the bookkeeping is under test."""
    d = tmp_path / "data" / f"fibers_{CUBE}"
    d.mkdir(parents=True)
    shutil.copy(ROOT / "data" / f"fibers_{CUBE}" / "meta.json", d / "meta.json")
    (tmp_path / "inst").mkdir()
    (tmp_path / "inst" / f"{CUBE}_instances.npy").write_bytes(b"")
    monkeypatch.setattr(tool, "score_fiber_prediction", lambda inst, target: {"metrics": _row(12.5)})
    return tmp_path


def _meta(data: Path) -> dict:
    return json.loads((data / "data" / f"fibers_{CUBE}" / "meta.json").read_text())


def test_another_entrant_keeps_every_listed_row_and_source(data):
    before = _meta(data)
    assert "tracer_strict_relink" in before["floors_external"]
    tool.import_rows(data / "data", data / "inst", "another_entrant", "its provenance")
    after = _meta(data)
    assert after["floors"]["another_entrant"] == _row(12.5)
    assert {k: v for k, v in after["floors"].items() if k != "another_entrant"} == before["floors"]
    assert after["floors_external"] == sorted([*before["floors_external"], "another_entrant"])
    assert after["floors_external_source"] == {**before["floors_external_source"], "another_entrant": "its provenance"}


def test_re_importing_a_row_replaces_only_that_row(data):
    before = _meta(data)
    tool.import_rows(data / "data", data / "inst", "another_entrant", "first")
    tool.import_rows(data / "data", data / "inst", "another_entrant", "second")
    after = _meta(data)
    assert after["floors_external"] == sorted([*before["floors_external"], "another_entrant"])
    assert after["floors_external_source"] == {**before["floors_external_source"], "another_entrant": "second"}


def test_a_missing_labelling_writes_nothing(data):
    """Every target is re-measured or none: a gap stops the import before any meta.json changes."""
    second = data / "data" / "fibers_s1_00497_02497_02997_256"
    second.mkdir()
    shutil.copy(ROOT / "data" / second.name / "meta.json", second / "meta.json")
    before = _meta(data)
    with pytest.raises(SystemExit, match="every target must be re-measured or none"):
        tool.import_rows(data / "data", data / "inst", "another_entrant", "x")
    assert _meta(data) == before


def test_the_default_row_is_still_our_tracer():
    assert tool.parse_args(["dir", "--source", "s"]).row == "tracer_strict_relink"
