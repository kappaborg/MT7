"""
Deliverable 7 — ThreatScorer

Upgrades every TrajectoryPrediction's diagnostics["threat_level"] from the
basic speed-only estimate to a full contextual score that accounts for:

  1. Speed & acceleration       — fast / rapidly-accelerating drones
  2. No-fly zone approach       — heading and predicted-trajectory checks
  3. Swarm detection            — N drones appearing simultaneously
  4. ADS-B / RF conflict        — visual position vs reported telemetry
  5. Evasion pattern            — physics↔ML prediction divergence spike
  6. Night / low-visibility     — adjusted thresholds on IR-only frames

Threat levels
─────────────
  "benign"           — no threat indicators present
  "suspicious"       — one or more soft indicators; requires attention
  "confirmed_threat" — multiple hard indicators with high confidence

Integration with LiveInferencePipeline
───────────────────────────────────────
    scorer = ThreatScorer.from_config("configs/zones.json")
    pipeline = LiveInferencePipeline(predictor=..., threat_scorer=scorer)

Standalone use (e.g. post-processing a recorded session):
    scorer = ThreatScorer()
    scorer.add_zone(center_px=(960, 540), radius_px=200, name="runway")
    threat_levels = scorer.score_all(predictions)
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .models import TrajectoryPrediction

logger = logging.getLogger(__name__)

# ── defaults (all in pixel space unless noted) ────────────────────────────────
SPEED_SUSPICIOUS  = 8.0    # px/step
SPEED_CONFIRMED   = 15.0   # px/step
APPROACH_ANGLE    = 60.0   # degrees — half-angle of "approaching" cone
SWARM_N           = 3      # simultaneous drones → swarm flag
SWARM_RADIUS      = 300.0  # px — max inter-drone distance for swarm grouping
CONF_CONFIRMED    = 0.70   # minimum predictor confidence for confirmed_threat
ADSB_CONFLICT_PX  = 150.0  # px — visual↔ADS-B divergence threshold
EVASION_SPIKE     = 3.5    # multiple of track's historical physics error → evasion
NIGHT_SPEED_SCALE = 1.4    # IR pixel scale correction factor for speed thresholds

# ── zone geometry helpers ─────────────────────────────────────────────────────

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _dot_normalized(
    v: Tuple[float, float], direction: Tuple[float, float]
) -> float:
    """Cosine of angle between v and direction. Returns 0 if either is zero."""
    mv = math.hypot(*v)
    md = math.hypot(*direction)
    if mv < 1e-9 or md < 1e-9:
        return 0.0
    return (v[0] * direction[0] + v[1] * direction[1]) / (mv * md)


def _heading_toward_zone(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zone_center: Tuple[float, float],
    half_angle_deg: float,
) -> bool:
    """True when the velocity vector points within half_angle_deg of the zone."""
    direction = (zone_center[0] - position[0], zone_center[1] - position[1])
    cos_angle = _dot_normalized(velocity, direction)
    return cos_angle >= math.cos(math.radians(half_angle_deg))


def _trajectory_enters_zone(
    predicted_points,   # List[TrajectoryPoint]
    zone_center: Tuple[float, float],
    zone_radius: float,
) -> bool:
    """True if any predicted position falls inside the zone circle."""
    for pt in predicted_points:
        if _dist(pt.position, zone_center) <= zone_radius:
            return True
    return False


# ── per-track history for evasion detection ───────────────────────────────────

class _TrackHistory:
    """Maintains a short window of physics-vs-actual errors for one track."""
    def __init__(self, window: int = 10):
        self._errors: Deque[float] = deque(maxlen=window)

    def push(self, physics_error_px: float) -> None:
        self._errors.append(physics_error_px)

    @property
    def mean_error(self) -> float:
        return sum(self._errors) / len(self._errors) if self._errors else 0.0

    @property
    def n(self) -> int:
        return len(self._errors)


# ── zone dataclass ────────────────────────────────────────────────────────────

class NoFlyZone:
    __slots__ = ("center", "radius_px", "name")

    def __init__(
        self,
        center: Tuple[float, float],
        radius_px: float,
        name: str = "",
    ):
        self.center    = center
        self.radius_px = radius_px
        self.name      = name

    def contains(self, point: Tuple[float, float]) -> bool:
        return _dist(point, self.center) <= self.radius_px

    def approach_buffer(self) -> float:
        """Outer ring used for 'heading toward zone' check (2× the radius)."""
        return self.radius_px * 2.0


# ── main scorer ───────────────────────────────────────────────────────────────

class ThreatScorer:
    """
    Stateful threat scorer — call score_all() once per frame with the
    complete list of active TrajectoryPredictions.

    Updates pred.diagnostics["threat_level"] and adds a
    pred.diagnostics["threat_reasons"] list of strings in-place.
    """

    def __init__(
        self,
        speed_suspicious:   float = SPEED_SUSPICIOUS,
        speed_confirmed:    float = SPEED_CONFIRMED,
        approach_angle_deg: float = APPROACH_ANGLE,
        swarm_n:            int   = SWARM_N,
        swarm_radius_px:    float = SWARM_RADIUS,
        min_confidence_confirmed: float = CONF_CONFIRMED,
        adsb_conflict_px:   float = ADSB_CONFLICT_PX,
        evasion_spike:      float = EVASION_SPIKE,
        night_speed_scale:  float = NIGHT_SPEED_SCALE,
    ):
        self._speed_susp   = speed_suspicious
        self._speed_conf   = speed_confirmed
        self._approach_deg = approach_angle_deg
        self._swarm_n      = swarm_n
        self._swarm_r      = swarm_radius_px
        self._conf_thresh  = min_confidence_confirmed
        self._adsb_thr     = adsb_conflict_px
        self._evasion_k    = evasion_spike
        self._night_scale  = night_speed_scale

        self._zones: List[NoFlyZone] = []
        self._track_history: Dict[int, _TrackHistory] = {}

    # ── zone management ───────────────────────────────────────────────────────

    def add_zone(
        self,
        center_px: Tuple[float, float],
        radius_px: float,
        name: str = "",
    ) -> None:
        self._zones.append(NoFlyZone(center_px, radius_px, name))
        logger.info("No-fly zone added: '%s' at %s r=%.0fpx", name, center_px, radius_px)

    def clear_zones(self) -> None:
        self._zones.clear()

    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> "ThreatScorer":
        """
        Load zones and optional scorer parameters from a JSON config file.

        Expected format:
          {
            "scorer": { "speed_suspicious": 8.0, ... },
            "zones": [
              {"center": [960, 540], "radius_px": 200, "name": "runway"},
              ...
            ]
          }
        """
        scorer = cls(**kwargs)
        try:
            with open(config_path) as fh:
                data = json.load(fh)
            scorer_cfg = data.get("scorer", {})
            for k, v in scorer_cfg.items():
                attr = f"_{k}"
                if hasattr(scorer, attr):
                    setattr(scorer, attr, v)
            for z in data.get("zones", []):
                cx, cy = z["center"]
                scorer.add_zone(
                    center_px=(float(cx), float(cy)),
                    radius_px=float(z["radius_px"]),
                    name=str(z.get("name", "")),
                )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load threat config '%s': %s", config_path, exc)
        return scorer

    # ── main scoring API ──────────────────────────────────────────────────────

    def score_all(
        self,
        predictions: List[TrajectoryPrediction],
        frame_number: int = 0,
        night_mode: bool = False,
        adsb_positions: Optional[Dict[int, Tuple[float, float]]] = None,
        physics_errors: Optional[Dict[int, float]] = None,
    ) -> List[str]:
        """
        Score every prediction in-place and return the list of threat levels.

        Parameters
        ----------
        predictions
            All active TrajectoryPredictions for this frame.
        frame_number
            Current frame index (for logging / audit trail).
        night_mode
            True when the primary camera is IR-only (luminance-based).
            Adjusts speed thresholds by night_speed_scale.
        adsb_positions
            Optional dict {track_id: (x_px, y_px)} of ADS-B or RF
            reported positions on the image plane (from transponder data).
        physics_errors
            Optional dict {track_id: px_error} — distance between the
            physics predictor's last forecast and the observed position.
            Used for evasion / GPS-spoof detection.
        """
        if not predictions:
            return []

        scale = self._night_scale if night_mode else 1.0
        speed_susp = self._speed_susp * scale
        speed_conf = self._speed_conf * scale

        adsb_pos    = adsb_positions  or {}
        phys_errors = physics_errors  or {}

        # Update evasion-detection history
        for tid, err in phys_errors.items():
            if tid not in self._track_history:
                self._track_history[tid] = _TrackHistory()
            self._track_history[tid].push(err)

        # ── 1. Per-prediction independent indicators ──────────────────────────
        scores: List[Dict] = []
        drone_preds: List[TrajectoryPrediction] = []

        for pred in predictions:
            s = self._score_single(pred, speed_susp, speed_conf, adsb_pos, phys_errors)
            scores.append(s)
            if pred.object_type == "drone":
                drone_preds.append(pred)

        # ── 2. Swarm detection ────────────────────────────────────────────────
        swarm_ids = self._detect_swarm(drone_preds)
        swarm_toward_zone = self._swarm_approaching_zone(drone_preds, swarm_ids)

        # ── 3. Apply swarm flags & finalise levels ────────────────────────────
        levels: List[str] = []
        for pred, score in zip(predictions, scores):
            tid = pred.track_id

            if tid in swarm_ids:
                score["reasons"].append(f"swarm({len(swarm_ids)} drones)")
                score["suspicious"] = True
                if swarm_toward_zone:
                    score["reasons"].append("swarm_approaching_zone")
                    score["hard_confirmed"] = True

            level = self._resolve_level(score, pred.confidence)
            pred.diagnostics["threat_level"]   = level
            pred.diagnostics["threat_reasons"] = score["reasons"]
            pred.diagnostics["threat_score"]   = int(score["suspicious"]) + 2 * int(
                score.get("hard_confirmed", False)
            )
            levels.append(level)

            if level != "benign":
                logger.info(
                    "[frame %d] track=%d  %s  reasons=%s",
                    frame_number, tid, level, score["reasons"],
                )

        return levels

    # ── internal helpers ──────────────────────────────────────────────────────

    def _score_single(
        self,
        pred: TrajectoryPrediction,
        speed_susp: float,
        speed_conf: float,
        adsb_pos: Dict[int, Tuple[float, float]],
        phys_errors: Dict[int, float],
    ) -> Dict:
        """Evaluate all single-track indicators. Returns a score dict."""
        score: Dict = {"suspicious": False, "hard_indicators": 0, "reasons": []}

        # Non-drones are always benign
        if pred.object_type != "drone":
            return score

        vel   = pred.current_velocity
        pos   = pred.current_position
        speed = math.hypot(vel[0], vel[1])

        # ── speed ─────────────────────────────────────────────────────────────
        if speed > speed_conf:
            score["suspicious"]    = True
            score["hard_indicators"] += 1
            score["reasons"].append(f"speed={speed:.1f}>{speed_conf:.1f}px/step")
        elif speed > speed_susp:
            score["suspicious"] = True
            score["reasons"].append(f"speed={speed:.1f}>{speed_susp:.1f}px/step")

        # ── no-fly zone checks ────────────────────────────────────────────────
        for zone in self._zones:
            # Already inside — unconditionally confirmed regardless of speed
            if zone.contains(pos):
                score["suspicious"]      = True
                score["hard_indicators"] += 2   # counts as two hard indicators alone
                score["hard_confirmed"]  = True
                score["reasons"].append(f"inside_zone:{zone.name or zone.center}")
                continue

            # Heading toward zone approach buffer
            if _dist(pos, zone.center) < zone.approach_buffer():
                if _heading_toward_zone(pos, vel, zone.center, self._approach_deg):
                    score["suspicious"]    = True
                    score["hard_indicators"] += 1
                    score["reasons"].append(f"approaching_zone:{zone.name or zone.center}")

            # Predicted trajectory enters zone
            if _trajectory_enters_zone(pred.predicted_points, zone.center, zone.radius_px):
                score["suspicious"]    = True
                score["hard_indicators"] += 1
                score["reasons"].append(f"trajectory_enters_zone:{zone.name or zone.center}")

        # ── ADS-B / RF conflict ───────────────────────────────────────────────
        if pred.track_id in adsb_pos:
            adsb = adsb_pos[pred.track_id]
            conflict = _dist(pos, adsb)
            if conflict > self._adsb_thr:
                score["suspicious"]    = True
                score["hard_indicators"] += 1
                score["reasons"].append(f"adsb_conflict={conflict:.0f}px")

        # ── evasion / GPS-spoof detection ─────────────────────────────────────
        hist = self._track_history.get(pred.track_id)
        if hist is not None and hist.n >= 5:
            current_err = phys_errors.get(pred.track_id, 0.0)
            if hist.mean_error > 0 and current_err > hist.mean_error * self._evasion_k:
                score["suspicious"] = True
                score["reasons"].append(
                    f"evasion_spike={current_err:.0f}px"
                    f"(baseline={hist.mean_error:.0f}px)"
                )

        # ── rapid acceleration ────────────────────────────────────────────────
        if pred.predicted_points:
            pred_speed_1 = math.hypot(*pred.predicted_points[0].velocity)
            accel = abs(pred_speed_1 - speed)
            if accel > speed_conf * 0.8:
                score["suspicious"] = True
                score["reasons"].append(f"rapid_accel={accel:.1f}px/step²")

        # Convenience flag for resolve step
        score["hard_confirmed"] = score["hard_indicators"] >= 2

        return score

    def _detect_swarm(
        self, drone_preds: List[TrajectoryPrediction]
    ) -> set:
        """
        Return the set of track_ids that are part of a swarm cluster.
        A swarm is defined as >= swarm_n drones within swarm_radius of each other.
        """
        if len(drone_preds) < self._swarm_n:
            return set()

        positions = [(p.track_id, p.current_position) for p in drone_preds]
        swarm_ids: set = set()

        for i, (tid_i, pos_i) in enumerate(positions):
            cluster = {tid_i}
            for tid_j, pos_j in positions:
                if tid_j != tid_i and _dist(pos_i, pos_j) <= self._swarm_r:
                    cluster.add(tid_j)
            if len(cluster) >= self._swarm_n:
                swarm_ids |= cluster

        return swarm_ids

    def _swarm_approaching_zone(
        self,
        drone_preds: List[TrajectoryPrediction],
        swarm_ids: set,
    ) -> bool:
        """
        True if the majority of swarm members have a trajectory entering
        the same no-fly zone.
        """
        if not swarm_ids or not self._zones:
            return False

        swarm_preds = [p for p in drone_preds if p.track_id in swarm_ids]
        for zone in self._zones:
            entering = sum(
                1 for p in swarm_preds
                if _trajectory_enters_zone(p.predicted_points, zone.center, zone.radius_px)
                or _heading_toward_zone(
                    p.current_position, p.current_velocity, zone.center, self._approach_deg
                )
            )
            if entering >= max(2, len(swarm_preds) // 2):
                return True
        return False

    @staticmethod
    def _resolve_level(score: Dict, confidence: float) -> str:
        if not score["suspicious"]:
            return "benign"
        if score.get("hard_confirmed") and confidence >= CONF_CONFIRMED:
            return "confirmed_threat"
        return "suspicious"

    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        """Remove history for tracks that have been dropped by the pipeline."""
        active = set(active_track_ids)
        stale = [tid for tid in self._track_history if tid not in active]
        for tid in stale:
            del self._track_history[tid]
