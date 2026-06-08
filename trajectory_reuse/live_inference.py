"""
Deliverable 6 — LiveInferencePipeline

Real-time adapter between a detector/tracker and the trajectory predictor.
Designed for the DJI Matrice 4T dual-sensor (EO + IR) stream.

Responsibilities:
  - Maintains a rolling history buffer (deque) of corrected centre positions
    per track_id across consecutive frames
  - Applies per-modality IR coordinate offsets from modality_config.json
    before storing positions (so the predictor never sees the raw sensor offset)
  - Converts bbox dimensions to area (px²) and passes as context
  - Converts laser rangefinder readings to a px/meter metric_scale and
    interpolates between 1 Hz laser pulses using GPS/IMU velocity
  - Calls the predictor for every active track and returns TrajectoryPrediction
  - Tracks "lost" tracks (no detection for up to max_track_age frames) and
    exposes their last predicted position for re-association by the tracker
  - Calls predictor.cleanup_old_tracks() when tracks are permanently dropped

Typical call pattern (one iteration per video frame):
    pipeline = LiveInferencePipeline(
        predictor=MLTrajectoryPredictor(checkpoint_path="output/checkpoints/best.pt"),
        modality_config_path="trajectory_reuse/modality_config.json",
    )

    for frame in video_stream:
        detections = tracker.update(detector.detect(frame))
        # detections: list of dicts with keys:
        #   track_id, center, bbox, object_type, confidence, modality
        predictions = pipeline.update(detections)
        for pred in predictions:
            visualize(pred)
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .models import TrajectoryPrediction
from .threat_scorer import ThreatScorer

# ── minimal threat level (full logic in threat_scorer.py — Deliverable 7) ────

def _basic_threat_level(object_type: str, velocity: tuple) -> str:
    """Baseline threat scoring injected into every prediction by the pipeline."""
    if object_type != "drone":
        return "benign"
    speed = math.hypot(velocity[0], velocity[1])
    if speed > 8.0:
        return "suspicious"
    return "benign"

logger = logging.getLogger(__name__)

# Maximum age (frames) before a track with no detections is permanently dropped
DEFAULT_MAX_TRACK_AGE = 30
DEFAULT_MAX_HISTORY   = 30
DEFAULT_DT            = 1.0   # annotation data; use 1/30 for live M4T video


class _TrackState:
    """Per-track mutable state held across frames."""

    __slots__ = (
        "history",      # deque[(x, y)] corrected centres
        "last_frame",   # int — frame number of last detection
        "last_pred",    # Optional[TrajectoryPrediction]
        "object_type",  # str
        "modality",     # str  "EO" | "IR"
        "area_history", # deque[float] bbox areas
    )

    def __init__(self, max_history: int):
        self.history:      Deque[Tuple[float, float]] = deque(maxlen=max_history)
        self.area_history: Deque[float]               = deque(maxlen=max_history)
        self.last_frame:   int                        = 0
        self.last_pred:    Optional[TrajectoryPrediction] = None
        self.object_type:  str = "drone"
        self.modality:     str = "EO"


class LiveInferencePipeline:
    """
    Per-frame orchestrator: tracker output → history buffers → predictor → predictions.

    Parameters
    ----------
    predictor
        Any object implementing predict() / cleanup_old_tracks() /
        get_performance_metrics() — MLTrajectoryPredictor or
        ReusableTrajectoryPredictor both work.
    max_history
        Maximum number of historical centres stored per track.
    max_track_age
        Frames of silence before a track is permanently removed.
        During this window the track's last predicted position is
        exposed for tracker re-association.
    dt
        Time step in seconds. Use 1.0 for offline annotation data
        and 1/30 ≈ 0.0333 for live M4T video at 30 fps.
    modality_config_path
        Path to modality_config.json.  IR centre offsets are applied
        automatically on every detection before it enters the buffer.
    laser_interpolation_alpha
        EMA weight for interpolating laser range between 1 Hz readings.
        Higher value trusts the most recent reading more.
    """

    def __init__(
        self,
        predictor,
        max_history: int = DEFAULT_MAX_HISTORY,
        max_track_age: int = DEFAULT_MAX_TRACK_AGE,
        dt: float = DEFAULT_DT,
        modality_config_path: Optional[str] = None,
        laser_interpolation_alpha: float = 0.3,
        threat_scorer: Optional["ThreatScorer"] = None,
    ):
        self._predictor       = predictor
        self._max_history     = max_history
        self._max_track_age   = max_track_age
        self._dt              = dt
        self._laser_alpha     = laser_interpolation_alpha

        # Per-track state
        self._tracks: Dict[int, _TrackState] = {}

        # Laser range state: interpolated px/meter scale
        self._metric_scale:    Optional[float] = None
        self._last_laser_time: float           = 0.0

        # Modality coordinate offsets loaded from config
        self._offsets: Dict[str, Tuple[float, float]] = {}
        if modality_config_path is not None:
            self._load_modality_offsets(modality_config_path)

        self._threat_scorer = threat_scorer

        # Pipeline-level performance tracking
        self._frame_count:      int   = 0
        self._total_pred_calls: int   = 0
        self._frame_times_ms:   Deque[float] = deque(maxlen=200)

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Dict],
        frame_number: int = 0,
        laser_range_m: Optional[float] = None,
        gimbal_pitch_deg: float = 0.0,
        camera_fov_deg: float = 15.0,
        frame_width_px: int = 1920,
        night_mode: bool = False,
        adsb_positions: Optional[Dict[int, Tuple[float, float]]] = None,
        physics_errors: Optional[Dict[int, float]] = None,
    ) -> List[TrajectoryPrediction]:
        """
        Process one frame of tracker output and return predictions for every
        active track.

        Parameters
        ----------
        detections
            List of detection dicts from the tracker.  Each dict must contain:
              track_id   : int
              center     : (x, y) in raw sensor pixel coordinates
              bbox       : [x1, y1, x2, y2]
              object_type: str   e.g. "drone"
              confidence : float detection confidence
              modality   : str   "EO" or "IR"
        frame_number
            Current frame index (used for track-age accounting).
        laser_range_m
            Range reading from the M4T laser rangefinder (meters).
            Only available at 1 Hz — pass None for the other 29 frames/s.
        gimbal_pitch_deg / camera_fov_deg / frame_width_px
            Used to compute px/meter metric_scale from laser_range_m.
        """
        t0 = time.perf_counter()
        self._frame_count += 1

        # Update laser scale estimate
        if laser_range_m is not None and laser_range_m > 1.0:
            self._update_metric_scale(
                laser_range_m, gimbal_pitch_deg, camera_fov_deg, frame_width_px
            )

        # Index active detections by track_id
        active_ids = {int(d["track_id"]) for d in detections}

        # Register new detections and update history
        for det in detections:
            tid        = int(det["track_id"])
            raw_center = (float(det["center"][0]), float(det["center"][1]))
            modality   = str(det.get("modality", "EO"))
            obj_type   = str(det.get("object_type", "drone"))
            bbox       = det.get("bbox", [])

            # Apply sensor offset correction
            corrected  = self._apply_offset(raw_center, modality)

            # Bbox area as proxy for depth change
            area = 0.0
            if len(bbox) == 4:
                area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

            if tid not in self._tracks:
                self._tracks[tid] = _TrackState(self._max_history)

            state = self._tracks[tid]
            state.history.append(corrected)
            state.area_history.append(area)
            state.last_frame  = frame_number
            state.object_type = obj_type
            state.modality    = modality

        # Run predictor on every track that has enough history
        predictions: List[TrajectoryPrediction] = []
        all_active_for_cleanup = list(self._tracks.keys())

        for tid, state in list(self._tracks.items()):
            age = frame_number - state.last_frame

            # Track is lost but within re-association window
            if age > 0 and age <= self._max_track_age:
                continue

            # Track is permanently lost
            if age > self._max_track_age:
                del self._tracks[tid]
                continue

            # Active track — run predictor
            context: Dict = {
                "modality":  state.modality,
                "bbox_area": list(state.area_history)[-1] if state.area_history else 0.0,
            }
            if self._metric_scale is not None:
                context["metric_scale"] = self._metric_scale

            pred = self._predictor.predict(
                track_id    = tid,
                trajectory  = list(state.history),
                object_type = state.object_type,
                context     = context,
            )
            if pred is not None:
                # Guarantee threat_level is always present regardless of predictor
                if "threat_level" not in pred.diagnostics:
                    pred.diagnostics["threat_level"] = _basic_threat_level(
                        pred.object_type, pred.current_velocity
                    )
                state.last_pred = pred
                predictions.append(pred)
                self._total_pred_calls += 1

        # Notify predictor of which tracks are still alive
        self._predictor.cleanup_old_tracks(all_active_for_cleanup)

        # Full threat scoring — upgrades basic threat_level in-place
        if self._threat_scorer is not None and predictions:
            self._threat_scorer.score_all(
                predictions,
                frame_number=frame_number,
                night_mode=night_mode,
                adsb_positions=adsb_positions,
                physics_errors=physics_errors,
            )
            self._threat_scorer.cleanup_old_tracks(list(self._tracks.keys()))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._frame_times_ms.append(elapsed_ms)

        return predictions

    def get_lost_track_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Last predicted position for each track that has gone silent but is
        still within the re-association window.  The tracker can use these
        to initialise the Kalman prior when a matching detection reappears.
        """
        lost: Dict[int, Tuple[float, float]] = {}
        for tid, state in self._tracks.items():
            if state.last_pred is not None and not state.history:
                lost[tid] = state.last_pred.current_position
        return lost

    def get_pipeline_metrics(self) -> Dict:
        pred_metrics = self._predictor.get_performance_metrics()
        avg_frame_ms = (
            sum(self._frame_times_ms) / len(self._frame_times_ms)
            if self._frame_times_ms else 0.0
        )
        return {
            **pred_metrics,
            "pipeline_frame_count":   self._frame_count,
            "pipeline_total_preds":   self._total_pred_calls,
            "pipeline_avg_frame_ms":  avg_frame_ms,
            "pipeline_active_tracks": len(self._tracks),
            "pipeline_metric_scale":  self._metric_scale or 0.0,
        }

    def reset(self) -> None:
        """Clear all track state. Call when switching to a new video source."""
        self._tracks.clear()
        self._metric_scale    = None
        self._last_laser_time = 0.0
        self._frame_count     = 0
        self._total_pred_calls = 0
        self._frame_times_ms.clear()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_modality_offsets(self, config_path: str) -> None:
        try:
            with open(config_path) as fh:
                cfg = json.load(fh)
            for modality, vals in cfg.items():
                if not isinstance(vals, dict):
                    continue
                dx = float(vals.get("center_offset_x", 0.0))
                dy = float(vals.get("center_offset_y", 0.0))
                if dx != 0.0 or dy != 0.0:
                    self._offsets[modality] = (dx, dy)
                    logger.info(
                        "Loaded %s offset: dx=%+.4f  dy=%+.4f", modality, dx, dy
                    )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not load modality config: %s", exc)

    def _apply_offset(
        self, center: Tuple[float, float], modality: str
    ) -> Tuple[float, float]:
        if modality not in self._offsets:
            return center
        dx, dy = self._offsets[modality]
        return (center[0] + dx, center[1] + dy)

    def _update_metric_scale(
        self,
        range_m: float,
        pitch_deg: float,
        fov_deg: float,
        frame_width_px: int,
    ) -> None:
        """
        Compute px/meter scale from laser reading and update the EMA estimate.

        Formula (pinhole camera, horizontal FOV):
            scale = frame_width_px / (2 * range_m * tan(fov/2))

        The range is the slant range; the horizontal ground extent at that
        distance is approximated assuming level ground and the given gimbal
        pitch.  For most surveillance scenarios this is sufficient.
        """
        fov_rad  = math.radians(fov_deg)
        ground_m = range_m * math.cos(math.radians(abs(pitch_deg)))
        if ground_m < 0.1:
            return
        new_scale = frame_width_px / (2.0 * ground_m * math.tan(fov_rad / 2.0))

        if self._metric_scale is None:
            self._metric_scale = new_scale
        else:
            alpha = self._laser_alpha
            self._metric_scale = alpha * new_scale + (1.0 - alpha) * self._metric_scale

        self._last_laser_time = time.time()
        logger.debug(
            "Laser: range=%.1fm  scale=%.2f px/m  (ema=%.2f)",
            range_m, new_scale, self._metric_scale,
        )
