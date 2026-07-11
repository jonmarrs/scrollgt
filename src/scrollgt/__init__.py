"""ScrollGT: registered human ground-truth ink evaluation for the open Vesuvius
Challenge SOTA scroll data."""

from .compliance import check_submission, window_compliant
from .metrics import segmentation_metrics
from .score import load_probability_map, load_target, score_prediction

__version__ = "0.1.0"
__all__ = [
    "segmentation_metrics",
    "score_prediction",
    "load_target",
    "load_probability_map",
    "check_submission",
    "window_compliant",
]
