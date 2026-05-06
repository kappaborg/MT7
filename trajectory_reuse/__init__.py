"""
Portable trajectory prediction package.

Copy this folder into another project to reuse the predictor without pulling in
the rest of the ATMS codebase.
"""

from .adapters import bbox_to_center, bbox_xywh_to_center, bbox_xyxy_to_center, centers_from_bboxes
from .models import PredictionMode, TrajectoryPoint, TrajectoryPrediction
from .predictor import ReusableTrajectoryPredictor

__all__ = [
    "PredictionMode",
    "TrajectoryPoint",
    "TrajectoryPrediction",
    "ReusableTrajectoryPredictor",
    "bbox_to_center",
    "bbox_xywh_to_center",
    "bbox_xyxy_to_center",
    "centers_from_bboxes",
]
