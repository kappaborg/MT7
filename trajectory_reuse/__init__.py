"""
Portable trajectory prediction package.

Copy this folder into another project to reuse the predictor without pulling in
the rest of the ATMS codebase.
"""

import logging

from .adapters import bbox_to_center, bbox_xywh_to_center, bbox_xyxy_to_center, centers_from_bboxes
from .models import PredictionMode, TrajectoryPoint, TrajectoryPrediction
from .predictor import ReusableTrajectoryPredictor
from .ml_predictor import MLTrajectoryPredictor, DroneTrajectoryTransformer
from .live_inference import LiveInferencePipeline
from .threat_scorer import ThreatScorer


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging; call once at the top of a script or application entry point."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


__all__ = [
    "PredictionMode",
    "TrajectoryPoint",
    "TrajectoryPrediction",
    "ReusableTrajectoryPredictor",
    "MLTrajectoryPredictor",
    "DroneTrajectoryTransformer",
    "LiveInferencePipeline",
    "ThreatScorer",
    "bbox_to_center",
    "bbox_xywh_to_center",
    "bbox_xyxy_to_center",
    "centers_from_bboxes",
    "setup_logging",
]
