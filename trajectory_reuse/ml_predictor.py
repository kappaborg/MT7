"""
Deliverable 3 — MLTrajectoryPredictor

Drop-in replacement for ReusableTrajectoryPredictor using a trained
Temporal Transformer. Implements the identical predict() /
cleanup_old_tracks() / get_performance_metrics() interface so no changes
are needed in run_pipeline.py or serve.py.

Architecture: DroneTrajectoryTransformer
  - Causal Transformer encoder over obs_len history frames
  - K parallel MLP decoder heads → K trajectory hypotheses
  - Mixture weight head → softmax weights over K hypotheses
  - Physics predictor as fallback when ML confidence is low

Requires: torch (CPU or CUDA). Falls back to physics predictor if torch
is unavailable or no checkpoint has been loaded.

Example usage (inference):
    from trajectory_reuse.ml_predictor import MLTrajectoryPredictor

    predictor = MLTrajectoryPredictor(
        checkpoint_path="output/checkpoints/best.pt",
        obs_len=8,
        pred_len=4,
        dt=1.0,          # 1.0 for annotation data; use 1/30 for live M4T video
    )

    history = [(10.0,12.0),(10.7,12.3),(11.5,12.8),(12.4,13.6),
               (13.2,14.7),(13.9,15.9),(14.5,17.2),(15.1,18.8)]

    result = predictor.predict(
        track_id=101,
        trajectory=history,
        object_type="drone",
        context={"modality": "EO", "bbox_area": 144.0},
    )
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .models import PredictionMode, TrajectoryPoint, TrajectoryPrediction
from .predictor import ReusableTrajectoryPredictor

logger = logging.getLogger(__name__)

# ── constants (mirrored from predictor.py for consistency) ────────────────────
HOVER_SPEED_THRESHOLD = 0.4    # px/step
STEP_CONF_START = 0.92
STEP_CONF_DECAY = 0.03
STEP_CONF_MIN   = 0.25

OBJECT_TYPE_MAP: Dict[str, int] = {
    "drone":      0,
    "pedestrian": 1,
    "cyclist":    2,
    "vehicle":    3,
    "emergency":  4,
    "unknown":    5,
    "object":     5,
}
NUM_OBJECT_TYPES = 6

MODALITY_MAP: Dict[str, int] = {"EO": 0, "IR": 1}
NUM_MODALITIES = 2

# ── neural network ────────────────────────────────────────────────────────────

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


class DroneTrajectoryTransformer:
    """
    Lightweight Temporal Transformer for multi-class drone trajectory prediction.

    Designed to be importable even when PyTorch is not installed; the class
    body uses late imports so the rest of the module stays usable.

    Input features per timestep (feature_dim = 20 with default embedding dims):
      [x_norm, y_norm, vx, vy, ax, ay, area_norm, is_hover(float)]  → 8 continuous
      + type_embedding(8) + modality_embedding(4)                    → 12 from embeddings
      = 20 total (projected to d_model inside the network)

    Output:
      pred_xy : [B, K, T_pred, 2]   K candidate future trajectories
      weights : [B, K]              softmax mixture weights
    """

    # Class attribute so the constructor is importable without torch
    _torch_module_class = None

    def __new__(cls, *args, **kwargs):
        torch, nn = _try_import_torch()
        if torch is None:
            raise ImportError(
                "PyTorch is required for DroneTrajectoryTransformer. "
                "Install with: pip install torch"
            )

        # Build the actual nn.Module class on first use
        if cls._torch_module_class is None:
            cls._torch_module_class = cls._build_module_class(torch, nn)

        return cls._torch_module_class(*args, **kwargs)

    @staticmethod
    def _build_module_class(torch, nn):
        import math as _math

        class _Model(nn.Module):
            def __init__(
                self,
                obs_len: int = 8,
                pred_len: int = 4,
                d_model: int = 32,
                nhead: int = 2,
                num_encoder_layers: int = 2,
                K: int = 4,
                dropout: float = 0.2,
                type_embed_dim: int = 8,
                mod_embed_dim: int = 4,
            ):
                super().__init__()
                self.obs_len = obs_len
                self.pred_len = pred_len
                self.K = K
                self.d_model = d_model

                # Continuous input: x, y, vx, vy, ax, ay, area_norm, is_hover
                cont_dim = 8
                self.type_emb = nn.Embedding(NUM_OBJECT_TYPES, type_embed_dim)
                self.mod_emb  = nn.Embedding(NUM_MODALITIES, mod_embed_dim)
                input_dim = cont_dim + type_embed_dim + mod_embed_dim

                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                self.pos_enc = nn.Embedding(obs_len, d_model)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,    # Pre-LN: more stable with small datasets
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_encoder_layers,
                    enable_nested_tensor=False,  # norm_first=True is incompatible with nested tensors
                )

                self.decoder_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model, pred_len * 2),
                    )
                    for _ in range(K)
                ])
                self.weight_head = nn.Linear(d_model, K)

                self._init_weights()

            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Embedding):
                        nn.init.normal_(m.weight, std=0.02)

            def forward(
                self,
                obs_traj:   "torch.Tensor",   # [B, T, 2]
                obs_vel:    "torch.Tensor",    # [B, T, 2]
                obs_acc:    "torch.Tensor",    # [B, T, 2]
                obs_area:   "torch.Tensor",    # [B, T, 1]
                is_hover:   "torch.Tensor",    # [B, T] long
                obj_type:   "torch.Tensor",    # [B] long
                modality:   "torch.Tensor",    # [B] long
                pad_mask:   "Optional[torch.Tensor]" = None,  # [B, T] bool
            ) -> "Tuple[torch.Tensor, torch.Tensor]":
                B, T, _ = obs_traj.shape
                device = obs_traj.device

                cont = torch.cat(
                    [obs_traj, obs_vel, obs_acc, obs_area,
                     is_hover.unsqueeze(-1).float()],
                    dim=-1,
                )  # [B, T, 8]

                te = self.type_emb(obj_type).unsqueeze(1).expand(B, T, -1)
                me = self.mod_emb(modality).unsqueeze(1).expand(B, T, -1)
                x  = torch.cat([cont, te, me], dim=-1)   # [B, T, input_dim]
                x  = self.input_proj(x)                   # [B, T, d_model]
                x  = x + self.pos_enc(torch.arange(T, device=device))

                # Bidirectional attention over the fixed observation window —
                # no causal mask needed here. Causality is implicit: we never
                # pass future positions into the encoder.
                x = self.encoder(x, src_key_padding_mask=pad_mask)

                ctx = x[:, -1, :]   # [B, d_model]  — last (most recent) step

                hyps = []
                for head in self.decoder_heads:
                    out = head(ctx).view(B, self.pred_len, 2)
                    hyps.append(out)

                pred_xy = torch.stack(hyps, dim=1)                    # [B, K, T_pred, 2]
                weights = torch.softmax(self.weight_head(ctx), dim=-1) # [B, K]
                return pred_xy, weights

        return _Model


# ── helper computations (no torch required) ───────────────────────────────────

def _finite_diff_vel(
    pts: Sequence[Tuple[float, float]], dt: float
) -> List[Tuple[float, float]]:
    n = len(pts)
    vel = []
    for i in range(n):
        if i == 0:
            vx = (pts[1][0] - pts[0][0]) / dt if n > 1 else 0.0
            vy = (pts[1][1] - pts[0][1]) / dt if n > 1 else 0.0
        elif i == n - 1:
            vx = (pts[-1][0] - pts[-2][0]) / dt
            vy = (pts[-1][1] - pts[-2][1]) / dt
        else:
            vx = (pts[i+1][0] - pts[i-1][0]) / (2.0 * dt)
            vy = (pts[i+1][1] - pts[i-1][1]) / (2.0 * dt)
        vel.append((vx, vy))
    return vel


def _finite_diff_acc(
    vel: Sequence[Tuple[float, float]], dt: float
) -> List[Tuple[float, float]]:
    n = len(vel)
    acc = []
    for i in range(n):
        if i == 0:
            ax = (vel[1][0] - vel[0][0]) / dt if n > 1 else 0.0
            ay = (vel[1][1] - vel[0][1]) / dt if n > 1 else 0.0
        elif i == n - 1:
            ax = (vel[-1][0] - vel[-2][0]) / dt
            ay = (vel[-1][1] - vel[-2][1]) / dt
        else:
            ax = (vel[i+1][0] - vel[i-1][0]) / (2.0 * dt)
            ay = (vel[i+1][1] - vel[i-1][1]) / (2.0 * dt)
        acc.append((ax, ay))
    return acc


def _entropy_confidence(weights: List[float], K: int) -> float:
    """
    Convert mixture weight distribution to a scalar confidence in [0, 1].
    High confidence = most weight on one hypothesis (low entropy).
    """
    eps = 1e-9
    entropy = -sum(w * math.log(w + eps) for w in weights)
    max_entropy = math.log(K + eps)
    return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0


# ── predictor wrapper ─────────────────────────────────────────────────────────

class MLTrajectoryPredictor:
    """
    Drop-in replacement for ReusableTrajectoryPredictor.

    Wraps DroneTrajectoryTransformer for inference and falls back to the
    physics predictor when:
      - No checkpoint has been loaded yet.
      - Track history is shorter than min_history.
      - ML confidence falls below confidence_threshold.

    The additional context keys recognised by this predictor (beyond what
    ReusableTrajectoryPredictor accepts):
      context["modality"]   : "EO" or "IR"  (default "EO")
      context["bbox_area"]  : float          (raw pixel area of bounding box)
      context["metric_scale"]: float         (px/meter from laser rangefinder)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        obs_len: int = 8,
        pred_len: int = 4,
        dt: float = 1.0,
        confidence_threshold: float = 0.65,
        min_history: int = 4,
        max_history: int = 30,
        d_model: int = 32,
        nhead: int = 2,
        num_encoder_layers: int = 2,
        K: int = 4,
        dropout: float = 0.2,
        type_embed_dim: int = 8,
        mod_embed_dim: int = 4,
        device: str = "auto",
    ):
        self.prediction_horizon = pred_len * dt
        self.dt = dt
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.confidence_threshold = confidence_threshold
        self.min_history = min_history
        self.max_history = max_history
        self.K = K

        # Resolve device
        if device == "auto":
            torch, _ = _try_import_torch()
            if torch is not None:
                self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device_str = "cpu"
        else:
            self._device_str = device

        # Instantiate neural network
        self._model = None
        self._model_hparams = dict(
            obs_len=obs_len, pred_len=pred_len, d_model=d_model,
            nhead=nhead, num_encoder_layers=num_encoder_layers, K=K,
            dropout=dropout, type_embed_dim=type_embed_dim,
            mod_embed_dim=mod_embed_dim,
        )
        self._model_loaded = False

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # Physics fallback — always available, no deps
        self._physics = ReusableTrajectoryPredictor(
            prediction_horizon=self.prediction_horizon,
            dt=dt,
            confidence_threshold=confidence_threshold,
            min_history=min_history,
            max_history=max_history,
            smoothing_alpha=0.35,
        )

        # Tracking state
        self.track_cache: Dict[int, Dict] = {}
        self.prediction_times_ms: Deque[float] = deque(maxlen=200)
        self.last_confidence: Optional[float] = None
        self.predictions_generated: int = 0

    # ── public API (identical to ReusableTrajectoryPredictor) ─────────────────

    def predict(
        self,
        track_id: int,
        trajectory: Sequence[Tuple[float, float]],
        velocity: Tuple[float, float] = (0.0, 0.0),
        object_type: str = "object",
        context: Optional[Dict] = None,
    ) -> Optional[TrajectoryPrediction]:
        """
        Return a TrajectoryPrediction for the given history, or None if
        history is too short. Falls back to the physics predictor when the ML
        model is not loaded or confidence is too low.
        """
        start = time.perf_counter()
        ctx = context or {}

        # Clean and validate history
        pts = self._clean_trajectory(trajectory)
        if len(pts) < self.min_history:
            return self._physics_predict(track_id, pts, velocity, object_type, ctx)

        trimmed = pts[-self.max_history:]
        modality_str = ctx.get("modality", "EO")
        modality_id  = MODALITY_MAP.get(modality_str, 0)
        obj_type_id  = OBJECT_TYPE_MAP.get(object_type, OBJECT_TYPE_MAP["unknown"])

        # Update per-track area cache
        raw_area = ctx.get("bbox_area", None)
        if track_id not in self.track_cache:
            self.track_cache[track_id] = {
                "area_history": deque(maxlen=self.max_history),
                "confidence": 0.0,
                "updated_at": time.time(),
                "modality": modality_id,
            }
        cache = self.track_cache[track_id]
        cache["updated_at"] = time.time()
        cache["modality"] = modality_id
        if raw_area is not None:
            cache["area_history"].append(raw_area)

        # Compute normalization reference for bbox area
        area_history = list(cache["area_history"])
        area_ref = (
            sorted(area_history)[len(area_history) // 2]
            if area_history else 1.0
        )
        if area_ref <= 0.0:
            area_ref = 1.0

        # Attempt ML inference
        if self._model_loaded:
            try:
                result = self._ml_predict(
                    track_id=track_id,
                    pts=trimmed,
                    velocity=velocity,
                    obj_type_id=obj_type_id,
                    modality_id=modality_id,
                    area_ref=area_ref,
                    object_type=object_type,
                    context=ctx,
                )
                if result is not None:
                    elapsed = (time.perf_counter() - start) * 1000.0
                    self.prediction_times_ms.append(elapsed)
                    self.last_confidence = result.confidence
                    self.predictions_generated += 1
                    cache["confidence"] = result.confidence
                    return result
            except Exception as exc:
                logger.warning("ML inference failed for track %d: %s", track_id, exc)

        # Physics fallback
        return self._physics_predict(track_id, pts, velocity, object_type, ctx)

    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        active = set(active_track_ids)
        stale = [tid for tid in self.track_cache if tid not in active]
        for tid in stale:
            del self.track_cache[tid]
        self._physics.cleanup_old_tracks(active_track_ids)

    def get_performance_metrics(self) -> Dict[str, float]:
        avg = (
            sum(self.prediction_times_ms) / len(self.prediction_times_ms)
            if self.prediction_times_ms else 0.0
        )
        return {
            "prediction_horizon":      self.prediction_horizon,
            "step_dt":                 self.dt,
            "obs_len":                 float(self.obs_len),
            "pred_len":                float(self.pred_len),
            "K_hypotheses":            float(self.K),
            "model_loaded":            float(self._model_loaded),
            "predictions_generated":   float(self.predictions_generated),
            "active_track_buffers":    float(len(self.track_cache)),
            "avg_prediction_time_ms":  avg,
            "last_confidence":         self.last_confidence or 0.0,
        }

    # ── checkpoint management ─────────────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a checkpoint written by the training script."""
        torch, _ = _try_import_torch()
        if torch is None:
            logger.error("PyTorch not installed — cannot load ML checkpoint.")
            return

        checkpoint = torch.load(path, map_location=self._device_str, weights_only=True)

        # Support both raw state_dict and wrapped checkpoint dicts
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            saved_hparams = checkpoint.get("hparams", {})
        else:
            state_dict = checkpoint
            saved_hparams = {}

        hparams = {**self._model_hparams, **saved_hparams}
        self._model = DroneTrajectoryTransformer(**hparams)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device_str)
        self._model.eval()
        self._model_loaded = True
        logger.info("ML model loaded from %s (device=%s)", path, self._device_str)

    def save_torchscript(self, path: str) -> None:
        """Export the model to TorchScript for deployment without Python source."""
        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        import torch
        scripted = torch.jit.script(self._model)
        scripted.save(path)
        logger.info("TorchScript model saved to %s", path)

    # ── internal ML inference ─────────────────────────────────────────────────

    def _ml_predict(
        self,
        track_id: int,
        pts: List[Tuple[float, float]],
        velocity: Tuple[float, float],
        obj_type_id: int,
        modality_id: int,
        area_ref: float,
        object_type: str,
        context: Dict,
    ) -> Optional[TrajectoryPrediction]:
        import torch

        T = self.obs_len
        origin = pts[-1]
        local = [(x - origin[0], y - origin[1]) for x, y in pts]

        vel = _finite_diff_vel(local, self.dt)
        acc = _finite_diff_acc(vel, self.dt)

        # Pad or trim to obs_len
        pad_len = max(0, T - len(local))
        padded_xy  = [(0.0, 0.0)] * pad_len + local[-T:]
        padded_vel = [(0.0, 0.0)] * pad_len + vel[-T:]
        padded_acc = [(0.0, 0.0)] * pad_len + acc[-T:]

        # Build pad mask (True = position is a PAD token)
        pad_mask_list = [True] * pad_len + [False] * min(T, len(local))

        # Area and hover
        area_hist = list(self.track_cache.get(track_id, {}).get("area_history", []))
        if len(area_hist) < T:
            area_hist = [0.0] * (T - len(area_hist)) + area_hist
        area_hist = area_hist[-T:]
        area_norm = [a / area_ref for a in area_hist]

        is_hover = [
            1 if math.hypot(v[0], v[1]) < HOVER_SPEED_THRESHOLD else 0
            for v in padded_vel
        ]

        device = self._device_str

        def _f(lst2d):
            return torch.tensor([[list(p) for p in lst2d]], dtype=torch.float32, device=device)

        obs_traj_t  = _f(padded_xy)                                          # [1, T, 2]
        obs_vel_t   = _f(padded_vel)                                          # [1, T, 2]
        obs_acc_t   = _f(padded_acc)                                          # [1, T, 2]
        obs_area_t  = torch.tensor([[[a] for a in area_norm]],
                                   dtype=torch.float32, device=device)        # [1, T, 1]
        is_hover_t  = torch.tensor([is_hover], dtype=torch.long, device=device)  # [1, T]
        obj_type_t  = torch.tensor([obj_type_id], dtype=torch.long, device=device)
        modality_t  = torch.tensor([modality_id], dtype=torch.long, device=device)
        pad_mask_t  = torch.tensor([pad_mask_list], dtype=torch.bool, device=device)

        with torch.no_grad():
            pred_xy, weights = self._model(
                obs_traj_t, obs_vel_t, obs_acc_t, obs_area_t,
                is_hover_t, obj_type_t, modality_t, pad_mask_t,
            )

        # Select best hypothesis
        w_list = weights[0].cpu().tolist()
        best_k = max(range(self.K), key=lambda k: w_list[k])
        best_pred = pred_xy[0, best_k].cpu().tolist()  # [T_pred, 2]

        # Denormalize: add origin back
        predicted_abs = [(x + origin[0], y + origin[1]) for x, y in best_pred]

        # Compute velocity for each predicted point (for TrajectoryPoint.velocity)
        pred_vel = _finite_diff_vel(predicted_abs, self.dt)

        # Build TrajectoryPoint list
        now = time.time()
        predicted_points: List[TrajectoryPoint] = []
        for step, ((px, py), (pvx, pvy)) in enumerate(zip(predicted_abs, pred_vel), 1):
            step_conf = max(STEP_CONF_MIN, STEP_CONF_START - (step - 1) * STEP_CONF_DECAY)
            predicted_points.append(TrajectoryPoint(
                timestamp=now + step * self.dt,
                position=(px, py),
                velocity=(pvx, pvy),
                confidence=step_conf,
                acceleration=(0.0, 0.0),
            ))

        ml_confidence = _entropy_confidence(w_list, self.K)

        # Check whether physics predictor should supplement
        physics_result = self._physics.predict(
            track_id, list(pts), velocity, object_type, context
        )
        physics_confidence = (
            physics_result.confidence if physics_result is not None else 0.0
        )

        mode = self._select_mode(object_type, physics_confidence, ml_confidence)

        # HYBRID: blend best ML hypothesis with physics rollout position-wise
        if mode == PredictionMode.HYBRID and physics_result is not None:
            alpha = ml_confidence / (ml_confidence + physics_confidence + 1e-9)
            blended: List[TrajectoryPoint] = []
            for ml_pt, ph_pt in zip(predicted_points, physics_result.predicted_points):
                bx = alpha * ml_pt.position[0] + (1 - alpha) * ph_pt.position[0]
                by = alpha * ml_pt.position[1] + (1 - alpha) * ph_pt.position[1]
                blended.append(TrajectoryPoint(
                    timestamp=ml_pt.timestamp,
                    position=(bx, by),
                    velocity=ml_pt.velocity,
                    confidence=ml_pt.confidence,
                    acceleration=ml_pt.acceleration,
                ))
            predicted_points = blended

        combined_confidence = max(ml_confidence, physics_confidence * 0.8)
        intention = self._infer_intention(padded_vel[-1], padded_acc[-1])

        return TrajectoryPrediction(
            track_id=track_id,
            object_type=object_type,
            current_position=origin,
            current_velocity=padded_vel[-1] if padded_vel else (0.0, 0.0),
            predicted_points=predicted_points,
            prediction_horizon=self.prediction_horizon,
            confidence=combined_confidence,
            intention=intention,
            method_used=mode,
            diagnostics={
                "ml_confidence":    ml_confidence,
                "physics_confidence": physics_confidence,
                "best_hypothesis":  float(best_k),
                "best_weight":      float(w_list[best_k]),
                "weight_entropy":   float(
                    -sum(w * math.log(w + 1e-9) for w in w_list)
                ),
                "points_used":      float(len(pts)),
                "obs_len":          float(T),
                "pad_steps":        float(pad_len),
                "threat_level":     self._threat_level(object_type, padded_vel[-1]),
            },
        )

    def _physics_predict(
        self,
        track_id: int,
        pts: List[Tuple[float, float]],
        velocity: Tuple[float, float],
        object_type: str,
        context: Dict,
    ) -> Optional[TrajectoryPrediction]:
        result = self._physics.predict(track_id, pts, velocity, object_type, context)
        if result is not None:
            result.diagnostics["threat_level"] = self._threat_level(
                object_type, result.current_velocity
            )
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_trajectory(
        trajectory: Sequence[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        pts = []
        for p in trajectory:
            if p is not None and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        return pts

    @staticmethod
    def _select_mode(
        object_type: str,
        physics_confidence: float,
        ml_confidence: float,
    ) -> PredictionMode:
        if object_type == "emergency":
            return PredictionMode.EMERGENCY
        if ml_confidence > 0.65 and physics_confidence > 0.55:
            return PredictionMode.HYBRID
        if ml_confidence > 0.65:
            return PredictionMode.ML_ONLY
        return PredictionMode.PHYSICS_ONLY

    @staticmethod
    def _infer_intention(
        vel: Tuple[float, float], acc: Tuple[float, float]
    ) -> str:
        speed = math.hypot(vel[0], vel[1])
        accel = math.hypot(acc[0], acc[1])
        if speed < HOVER_SPEED_THRESHOLD:
            return "hover" if accel < 0.2 else "stationary"
        return "moving"

    @staticmethod
    def _threat_level(object_type: str, vel: Tuple[float, float]) -> str:
        if object_type not in ("drone",):
            return "benign"
        speed = math.hypot(vel[0], vel[1])
        if speed < 2.0:
            return "benign"
        if speed > 8.0:
            return "suspicious"
        return "benign"
