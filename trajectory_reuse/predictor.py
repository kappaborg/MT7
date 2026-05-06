from __future__ import annotations

import logging
import math
import time
from collections import deque
from statistics import median
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .models import PredictionMode, TrajectoryPoint, TrajectoryPrediction

logger = logging.getLogger(__name__)


class ReusableTrajectoryPredictor:
    """
    Lightweight trajectory predictor designed to be copied into other projects.

    The predictor improves on the existing implementation by:
    - rejecting large point outliers before forecasting
    - smoothing noisy trajectories with exponential filtering
    - estimating velocity from recent motion instead of trusting one frame
    - modeling gentle turning behaviour for non-linear motion
    - keeping dependencies to the Python standard library only
    """

    def __init__(
        self,
        prediction_horizon: float = 5.0,
        dt: float = 0.2,
        confidence_threshold: float = 0.7,
        min_history: int = 4,
        max_history: int = 30,
        smoothing_alpha: float = 0.35,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be greater than 0")
        if prediction_horizon <= 0.0:
            raise ValueError("prediction_horizon must be greater than 0")
        if min_history < 2:
            raise ValueError("min_history must be at least 2")
        if max_history < min_history:
            raise ValueError("max_history must be greater than or equal to min_history")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in the range (0, 1]")

        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.confidence_threshold = confidence_threshold
        self.min_history = min_history
        self.max_history = max_history
        self.smoothing_alpha = smoothing_alpha

        self.prediction_times_ms: Deque[float] = deque(maxlen=200)
        self.last_confidence: Optional[float] = None
        self.predictions_generated = 0
        self.track_cache: Dict[int, Dict[str, float]] = {}

    def predict(
        self,
        track_id: int,
        trajectory: Sequence[Tuple[float, float]],
        velocity: Tuple[float, float] = (0.0, 0.0),
        object_type: str = "object",
        context: Optional[Dict] = None,
    ) -> Optional[TrajectoryPrediction]:
        start_time = time.perf_counter()

        normalized = self._normalize_trajectory(trajectory)
        required_history = max(2, self.min_history)
        if len(normalized) < required_history:
            return None

        trimmed = normalized[-self.max_history :]
        filtered, removed_outliers = self._reject_outliers(trimmed)
        points_for_state = filtered if len(filtered) >= required_history else trimmed

        smoothed = self._smooth_points(points_for_state)
        if len(smoothed) < required_history:
            return None

        state = self._estimate_state(smoothed, velocity)
        state["x"], state["y"] = points_for_state[-1]
        predicted_points = self._rollout_predictions(
            state=state,
            object_type=object_type,
            context=context or {},
        )

        if not predicted_points:
            return None

        confidence = self._calculate_confidence(
            points=smoothed,
            predicted_points=predicted_points,
            removed_outliers=removed_outliers,
            state=state,
        )
        mode = self._select_mode(object_type=object_type, confidence=confidence)
        intention = self._infer_intention(state)

        prediction = TrajectoryPrediction(
            track_id=track_id,
            object_type=object_type,
            current_position=points_for_state[-1],
            current_velocity=(state["vx"], state["vy"]),
            predicted_points=predicted_points,
            prediction_horizon=self.prediction_horizon,
            confidence=confidence,
            intention=intention,
            method_used=mode,
            diagnostics={
                "points_used": float(len(points_for_state)),
                "outliers_removed": float(removed_outliers),
                "turn_rate": float(state["turn_rate"]),
                "speed": float(state["speed"]),
                "acceleration": float(state["acceleration_magnitude"]),
            },
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.prediction_times_ms.append(elapsed_ms)
        self.last_confidence = confidence
        self.predictions_generated += 1
        self.track_cache[track_id] = {
            "confidence": confidence,
            "updated_at": time.time(),
        }

        return prediction

    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        active = set(active_track_ids)
        stale_ids = [track_id for track_id in self.track_cache if track_id not in active]
        for track_id in stale_ids:
            del self.track_cache[track_id]

    def get_performance_metrics(self) -> Dict[str, float]:
        avg_time = (
            sum(self.prediction_times_ms) / len(self.prediction_times_ms)
            if self.prediction_times_ms
            else 0.0
        )
        return {
            "prediction_horizon": self.prediction_horizon,
            "step_dt": self.dt,
            "predictions_generated": float(self.predictions_generated),
            "active_track_buffers": float(len(self.track_cache)),
            "avg_prediction_time_ms": avg_time,
            "last_confidence": self.last_confidence or 0.0,
        }

    def _normalize_trajectory(
        self, trajectory: Sequence[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for point in trajectory:
            if point is None or len(point) < 2:
                continue
            x, y = float(point[0]), float(point[1])
            points.append((x, y))
        return points

    def _reject_outliers(
        self, points: Sequence[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], int]:
        if len(points) < 4:
            return list(points), 0

        step_lengths = [
            self._distance(points[index - 1], points[index])
            for index in range(1, len(points))
        ]
        typical_step = median(step_lengths)
        deviation = median(abs(step - typical_step) for step in step_lengths)

        if typical_step <= 0.0:
            return list(points), 0

        filtered = [points[0]]
        removed = 0

        for index in range(1, len(points)):
            step = self._distance(filtered[-1], points[index])
            if deviation <= 1e-6:
                is_outlier = step > typical_step * 3.0
            else:
                score = abs(step - typical_step) / (deviation + 1e-6)
                is_outlier = score > 4.5 and step > typical_step * 2.5
            if is_outlier:
                removed += 1
                continue
            filtered.append(points[index])

        return filtered, removed

    def _smooth_points(
        self, points: Sequence[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not points:
            return []

        smoothed = [points[0]]
        alpha = self.smoothing_alpha
        for point in points[1:]:
            last_x, last_y = smoothed[-1]
            x = alpha * point[0] + (1.0 - alpha) * last_x
            y = alpha * point[1] + (1.0 - alpha) * last_y
            smoothed.append((x, y))
        return smoothed

    def _estimate_state(
        self,
        points: Sequence[Tuple[float, float]],
        provided_velocity: Tuple[float, float],
    ) -> Dict[str, float]:
        recent_points = list(points[-6:])
        velocities: List[Tuple[float, float]] = []

        for index in range(1, len(recent_points)):
            px, py = recent_points[index - 1]
            cx, cy = recent_points[index]
            velocities.append(((cx - px) / self.dt, (cy - py) / self.dt))

        weights = list(range(1, len(velocities) + 1)) or [1]
        vx = self._weighted_average([v[0] for v in velocities], weights) if velocities else 0.0
        vy = self._weighted_average([v[1] for v in velocities], weights) if velocities else 0.0

        if provided_velocity != (0.0, 0.0):
            vx = 0.75 * vx + 0.25 * float(provided_velocity[0])
            vy = 0.75 * vy + 0.25 * float(provided_velocity[1])

        accelerations: List[Tuple[float, float]] = []
        for index in range(1, len(velocities)):
            pvx, pvy = velocities[index - 1]
            cvx, cvy = velocities[index]
            accelerations.append(((cvx - pvx) / self.dt, (cvy - pvy) / self.dt))

        acc_weights = list(range(1, len(accelerations) + 1)) or [1]
        ax = (
            self._weighted_average([a[0] for a in accelerations], acc_weights)
            if accelerations
            else 0.0
        )
        ay = (
            self._weighted_average([a[1] for a in accelerations], acc_weights)
            if accelerations
            else 0.0
        )

        turn_rate = self._estimate_turn_rate(velocities)
        speed = math.hypot(vx, vy)

        return {
            "x": points[-1][0],
            "y": points[-1][1],
            "vx": vx,
            "vy": vy,
            "ax": ax,
            "ay": ay,
            "speed": speed,
            "turn_rate": turn_rate,
            "acceleration_magnitude": math.hypot(ax, ay),
        }

    def _rollout_predictions(
        self,
        state: Dict[str, float],
        object_type: str,
        context: Dict,
    ) -> List[TrajectoryPoint]:
        if self.dt <= 0.0:
            return []

        max_speed = self._speed_cap(object_type, context)
        damping = 0.995 if object_type == "drone" else 0.99
        steps = max(1, int(self.prediction_horizon / self.dt))

        x = state["x"]
        y = state["y"]
        vx = state["vx"]
        vy = state["vy"]
        ax = state["ax"]
        ay = state["ay"]
        turn_rate = state["turn_rate"]
        timestamp = time.time()

        points: List[TrajectoryPoint] = []
        for step in range(1, steps + 1):
            vx += ax * self.dt
            vy += ay * self.dt

            speed = math.hypot(vx, vy)
            if speed > 1e-6 and abs(turn_rate) > 1e-6:
                heading = math.atan2(vy, vx) + turn_rate * self.dt
                speed = min(speed, max_speed)
                vx = speed * math.cos(heading)
                vy = speed * math.sin(heading)

            if speed > max_speed:
                ratio = max_speed / speed
                vx *= ratio
                vy *= ratio

            vx *= damping
            vy *= damping

            x += vx * self.dt
            y += vy * self.dt

            step_confidence = max(0.25, 0.92 - (step - 1) * 0.03)
            points.append(
                TrajectoryPoint(
                    timestamp=timestamp + step * self.dt,
                    position=(x, y),
                    velocity=(vx, vy),
                    confidence=step_confidence,
                    acceleration=(ax, ay),
                )
            )

        return points

    def _calculate_confidence(
        self,
        points: Sequence[Tuple[float, float]],
        predicted_points: Sequence[TrajectoryPoint],
        removed_outliers: int,
        state: Dict[str, float],
    ) -> float:
        history_factor = min(1.0, len(points) / max(self.min_history + 2, 8))

        step_lengths = [
            self._distance(points[index - 1], points[index])
            for index in range(1, len(points))
        ]
        avg_step = sum(step_lengths) / len(step_lengths) if step_lengths else 0.0
        step_variation = (
            sum(abs(step - avg_step) for step in step_lengths) / len(step_lengths)
            if step_lengths
            else 0.0
        )
        smoothness = 1.0 / (1.0 + (step_variation / (avg_step + 1e-6)))

        turn_penalty = min(abs(state["turn_rate"]) / 2.5, 0.25)
        outlier_penalty = min(removed_outliers * 0.08, 0.24)
        forecast_decay = 1.0 - min(len(predicted_points) * 0.01, 0.12)

        confidence = (0.35 + 0.3 * history_factor + 0.35 * smoothness) * forecast_decay
        confidence -= turn_penalty
        confidence -= outlier_penalty

        return max(0.05, min(0.99, confidence))

    def _select_mode(self, object_type: str, confidence: float) -> PredictionMode:
        if object_type == "emergency":
            return PredictionMode.EMERGENCY
        if confidence >= self.confidence_threshold:
            return PredictionMode.HYBRID
        return PredictionMode.PHYSICS_ONLY

    def _infer_intention(self, state: Dict[str, float]) -> str:
        speed = state["speed"]
        turn_rate = state["turn_rate"]
        acceleration = state["acceleration_magnitude"]

        if speed < 0.4:
            return "hover" if acceleration < 0.2 else "stationary"
        if turn_rate > 0.15:
            return "turn_left"
        if turn_rate < -0.15:
            return "turn_right"
        if acceleration > 0.8:
            return "accelerating"
        return "straight"

    def _estimate_turn_rate(self, velocities: Sequence[Tuple[float, float]]) -> float:
        headings: List[float] = []
        for vx, vy in velocities:
            if math.hypot(vx, vy) < 1e-6:
                continue
            headings.append(math.atan2(vy, vx))

        if len(headings) < 2:
            return 0.0

        angle_deltas: List[float] = []
        for index in range(1, len(headings)):
            delta = headings[index] - headings[index - 1]
            while delta > math.pi:
                delta -= 2.0 * math.pi
            while delta < -math.pi:
                delta += 2.0 * math.pi
            angle_deltas.append(delta / self.dt)

        if not angle_deltas:
            return 0.0

        weights = list(range(1, len(angle_deltas) + 1))
        turn_rate = self._weighted_average(angle_deltas, weights)
        return max(-1.5, min(1.5, turn_rate))

    def _speed_cap(self, object_type: str, context: Dict) -> float:
        configured = context.get("max_speed")
        if isinstance(configured, (int, float)) and configured > 0:
            return float(configured)

        caps = {
            "pedestrian": 4.0,
            "cyclist": 14.0,
            "vehicle": 45.0,
            "emergency": 55.0,
            "drone": 18.0,
        }
        return caps.get(object_type, 25.0)

    def _weighted_average(self, values: Sequence[float], weights: Sequence[int]) -> float:
        if not values:
            return 0.0
        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            return sum(values) / len(values)
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    def _distance(
        self, point_a: Tuple[float, float], point_b: Tuple[float, float]
    ) -> float:
        return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
