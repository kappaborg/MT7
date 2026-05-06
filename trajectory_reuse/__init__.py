"""
Portable trajectory prediction package.

Copy this folder into another project to reuse the predictor without pulling in
the rest of the ATMS codebase.
"""

import logging

from .adapters import bbox_to_center, bbox_xywh_to_center, bbox_xyxy_to_center, centers_from_bboxes
from .models import PredictionMode, TrajectoryPoint, TrajectoryPrediction
from .predictor import ReusableTrajectoryPredictor


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging; call once at the top of a script or application entry point."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


__all__ = [
    "PredictionMode",
    "TrajectoryPoint",
    "TrajectoryPrediction",
    "ReusableTrajectoryPredictor",
    "bbox_to_center",
    "bbox_xywh_to_center",
    "bbox_xyxy_to_center",
    "centers_from_bboxes",
    "setup_logging",
]
