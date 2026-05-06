from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class PredictionMode(Enum):
    """Prediction modes kept compatible with the existing ATMS code."""

    PHYSICS_ONLY = "physics_only"
    ML_ONLY = "ml_only"
    HYBRID = "hybrid"
    EMERGENCY = "emergency"


@dataclass
class TrajectoryPoint:
    """One predicted point along the future path."""

    timestamp: float
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    confidence: float
    acceleration: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrajectoryPrediction:
    """Prediction payload that can be serialized or reused across projects."""

    track_id: int
    object_type: str
    current_position: Tuple[float, float]
    current_velocity: Tuple[float, float]
    predicted_points: List[TrajectoryPoint]
    prediction_horizon: float
    confidence: float
    intention: str
    method_used: PredictionMode
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["method_used"] = self.method_used.value
        return data
