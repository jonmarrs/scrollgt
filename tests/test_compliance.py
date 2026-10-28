import pytest

from scrollgt.cli import main
from scrollgt.compliance import (
    check_submission,
    regions_overlap,
    window_advisory,
    window_compliant,
)


def test_a_large_window_is_not_a_failure_but_needs_hallucination_evidence():
    """villa #1751 (2026-09-10) replaced the 0.5 x 0.5 mm guidance with "Large windows are fine, but you must
    demonstrate that the text is not hallucinated"."""
    assert check_submission(224, 8.0, [], {}) == (True, [])
    advice = window_advisory(224, 8.0)
    assert advice["large"] and advice["window_mm"] == pytest.approx(1.792)
    assert "hallucinated" in advice["note"] and "held-out" in advice["note"]


def test_the_old_guidance_size_needs_no_note():
    for window_px, scan_um in ((64, 8.0), (200, 2.4)):  # 0.512 mm (the old blessed pixel form), 0.48 mm
        advice = window_advisory(window_px, scan_um)
        assert not advice["large"] and advice["note"] is None


def test_window_compliant_is_the_superseded_guidance_and_says_so():
    with pytest.warns(DeprecationWarning, match="1751"):
        ok, mm = window_compliant(224, 8.0)
    assert not ok and mm > 1.7
    with pytest.warns(DeprecationWarning):
        assert window_compliant(64, 8.0)[0]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        window_advisory(0, 8.0)


def test_check_reports_a_large_window_as_a_note_and_passes(capsys):
    assert main(["check", "--window-px", "224", "--scan-um", "8.0"]) == 0
    out = capsys.readouterr().out
    assert "NOTE" in out and "hallucinated" in out and "0.5mm cap" not in out


def test_overlap_detection():
    a = {"y0": 0, "x0": 0, "h": 100, "w": 100}
    b = {"y0": 50, "x0": 50, "h": 100, "w": 100}
    c = {"y0": 100, "x0": 0, "h": 10, "w": 10}  # touches edge only
    assert regions_overlap(a, b)
    assert not regions_overlap(a, c)


def test_check_submission_flags_overlap_same_volume_only():
    train = [{"volume": "v1", "y0": 0, "x0": 0, "h": 100, "w": 100},
             {"volume": "v2", "y0": 0, "x0": 0, "h": 100, "w": 100}]
    pred = {"volume": "v1", "y0": 50, "x0": 50, "h": 100, "w": 100}
    ok, failures = check_submission(64, 8.0, train, pred)
    assert not ok
    assert len(failures) == 1  # only the same-volume region counts as overlap
    assert "region 0" in failures[0]
