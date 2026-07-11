import pytest

from scrollgt.compliance import check_submission, regions_overlap, window_compliant


def test_64px_at_8um_is_compliant():
    ok, mm = window_compliant(64, 8.0)
    assert ok and abs(mm - 0.512) < 1e-9  # the community-accepted 64px@8um window


def test_224px_at_8um_is_not_compliant():
    ok, mm = window_compliant(224, 8.0)
    assert not ok and mm > 1.7


def test_finer_scan_allows_more_pixels():
    ok, _ = window_compliant(200, 2.4)  # 0.48mm at 2.4um
    assert ok


def test_invalid_inputs():
    with pytest.raises(ValueError):
        window_compliant(0, 8.0)


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
