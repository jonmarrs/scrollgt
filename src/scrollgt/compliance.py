"""Prize-compliance checks for ink-detection submissions.

Adapted from vesuvius-autoresearch `scripts/validate_prize_artifact.py` (the
submission-package validator). This standalone version checks the two rules that
bound every Vesuvius letter-reading prize claim:

1. ML window size: <= 0.5 x 0.5 mm on papyrus (64 x 64 px at 8 um; scaled for
   other scan resolutions).
2. Train/predict separation: the declared training regions must not intersect
   the predicted region.

These checks are *necessary, not sufficient* — the official rules also require
programmatic generation, scale bars, 3D position metadata, and hallucination
mitigation documentation.
"""

MAX_WINDOW_MM = 0.5
MAX_WINDOW_PX = 64


def window_compliant(window_px, scan_um):
    """True iff the window satisfies the official cap.

    The rule text is "0.5 x 0.5 mm (64 x 64 pixels for 8 um scans)" — note that
    64 px x 8 um = 0.512 mm, so the pixel form is explicitly blessed even though
    it slightly exceeds the literal mm form. Mirroring the reference validator
    (vesuvius-autoresearch scripts/validate_prize_artifact.py), a window is
    compliant iff window_px <= 64 OR window_mm <= 0.5.
    """
    if window_px <= 0 or scan_um <= 0:
        raise ValueError("window_px and scan_um must be positive")
    window_mm = window_px * scan_um / 1000.0
    ok = window_px <= MAX_WINDOW_PX or window_mm <= MAX_WINDOW_MM + 1e-9
    return ok, window_mm


def regions_overlap(a, b):
    """Axis-aligned overlap test. Regions are dicts with y0, x0, h, w (pixels,
    same geometry/level). Returns True if the interiors intersect."""
    ay0, ax0, ah, aw = a["y0"], a["x0"], a["h"], a["w"]
    by0, bx0, bh, bw = b["y0"], b["x0"], b["h"], b["w"]
    return not (ay0 + ah <= by0 or by0 + bh <= ay0 or ax0 + aw <= bx0 or bx0 + bw <= ax0)


def check_submission(window_px, scan_um, train_regions, predict_region):
    """Run both checks. Returns (ok, list of failure strings)."""
    failures = []
    ok_window, window_mm = window_compliant(window_px, scan_um)
    if not ok_window:
        failures.append(
            f"ML window {window_px}px @ {scan_um}um = {window_mm:.3f}mm exceeds "
            f"the {MAX_WINDOW_MM}mm prize cap"
        )
    for i, tr in enumerate(train_regions):
        same_volume = tr.get("volume") == predict_region.get("volume")
        if same_volume and regions_overlap(tr, predict_region):
            failures.append(
                f"training region {i} ({tr}) overlaps the predicted region — "
                "zero train/predict overlap is required"
            )
    return (not failures), failures
