"""Prize-compliance checks for ink-detection submissions.

Adapted from vesuvius-autoresearch `scripts/validate_prize_artifact.py` (the
submission-package validator). This standalone version checks one rule that
bounds every Vesuvius letter-reading prize claim, and advises on a second:

1. Train/predict separation: the declared training regions must not intersect
   the predicted region. A failure.
2. ML window size: no longer capped. Villa #1751 (2026-09-10) replaced "we
   strongly discourage ... window sizes larger than 0.5x0.5 mm" with "Large
   windows are fine, but you must demonstrate that the text is not hallucinated
   -- for example by finding plausible text on held-out regions." A window over
   the old 0.5 mm guidance gets a note asking for that evidence, not a failure.

These checks are *necessary, not sufficient* -- the official rules also require
programmatic generation, scale bars, 3D position metadata, held-out validation
and hallucination mitigation documentation.
"""

import warnings

# The pre-2026-09-10 guidance: 0.5 x 0.5 mm, or 64 px at 8 um (64 x 8 um = 0.512 mm,
# explicitly blessed). Kept as the size above which the rules ask for evidence.
MAX_WINDOW_MM = 0.5
MAX_WINDOW_PX = 64


def _window_mm(window_px, scan_um):
    if window_px <= 0 or scan_um <= 0:
        raise ValueError("window_px and scan_um must be positive")
    return window_px * scan_um / 1000.0


def window_advisory(window_px, scan_um):
    """The window in mm, and whether it is larger than the old 0.5 mm guidance.

    A larger window is allowed; the rules ask for evidence that the text is not
    hallucinated. Returns {"window_mm", "large", "note"} (note is None when not large).
    """
    window_mm = _window_mm(window_px, scan_um)
    large = not (window_px <= MAX_WINDOW_PX or window_mm <= MAX_WINDOW_MM + 1e-9)
    note = None
    if large:
        note = (
            f"ML window {window_px}px @ {scan_um}um = {window_mm:.3f}mm is larger than the "
            f"old {MAX_WINDOW_MM}mm guidance. Large windows are allowed (villa #1751, "
            "2026-09-10), but you must demonstrate that the text is not hallucinated, "
            "e.g. by finding plausible text on held-out regions."
        )
    return {"window_mm": window_mm, "large": large, "note": note}


def window_compliant(window_px, scan_um):
    """Deprecated: the superseded pre-2026-09-10 window guidance, as (ok, window_mm).

    The prize rules no longer cap the window (villa #1751); use window_advisory.
    """
    warnings.warn(
        "window_compliant checks the window guidance villa #1751 replaced on 2026-09-10; "
        "large windows are allowed with evidence against hallucination (window_advisory)",
        DeprecationWarning,
        stacklevel=2,
    )
    advice = window_advisory(window_px, scan_um)
    return (not advice["large"]), advice["window_mm"]


def regions_overlap(a, b):
    """Axis-aligned overlap test. Regions are dicts with y0, x0, h, w (pixels,
    same geometry/level). Returns True if the interiors intersect."""
    ay0, ax0, ah, aw = a["y0"], a["x0"], a["h"], a["w"]
    by0, bx0, bh, bw = b["y0"], b["x0"], b["h"], b["w"]
    return not (ay0 + ah <= by0 or by0 + bh <= ay0 or ax0 + aw <= bx0 or bx0 + bw <= ax0)


def check_submission(window_px, scan_um, train_regions, predict_region):
    """Train/predict separation. Returns (ok, list of failure strings).

    The window is validated (positive sizes) but no longer fails a submission;
    see window_advisory for the note a large window gets.
    """
    _window_mm(window_px, scan_um)
    failures = []
    for i, tr in enumerate(train_regions):
        same_volume = tr.get("volume") == predict_region.get("volume")
        if same_volume and regions_overlap(tr, predict_region):
            failures.append(
                f"training region {i} ({tr}) overlaps the predicted region — "
                "zero train/predict overlap is required"
            )
    return (not failures), failures
