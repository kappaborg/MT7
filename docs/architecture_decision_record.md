# Architecture Decision Record
## Drone + Multi-Object Trajectory Prediction — ML Upgrade

**Project:** MT7 trajectory prediction pipeline  
**Date:** 2026-05-07  
**Hardware platform:** DJI Matrice 4T (observer/surveillance drone)  
**Status:** Proposed

---

## Hardware Context — DJI Matrice 4T

The M4T is the surveillance platform — it carries the cameras that observe targets.
This means the camera itself is always moving, which has deep implications for tracking.

| Sensor | Resolution | FOV | Frame rate | Notes |
|---|---|---|---|---|
| Wide EO (24mm) | 48MP / 1920×1080 live | 82° | 30fps | Primary wide-area search |
| Medium tele EO (70mm) | 48MP / 1920×1080 live | 35° | 30fps | Mid-range tracking |
| Telephoto EO (168mm) | 48MP / 1920×1080 live | 15° | 30fps | Long-range drone ID |
| Thermal IR | 640×512 native / 1280×1024 ultra | 45° | 30Hz | VOx, NETD ≤50mK |
| Laser rangefinder | — | point | **1 Hz** | ±(0.2 + 0.0015×D) m, max 1800m |

**O4 Enterprise video link:** 130ms inherent transmission latency to ground station.  
**Gimbal:** 3-axis stabilized, ±0.007° accuracy, SDK access to angles at up to 400Hz.  
**GPS/IMU:** Accessible via DJI Onboard SDK alongside video stream at up to 400Hz.

**Total system latency:** 130ms (O4) + ≤50ms (on-ground processing) = **≤180ms** end-to-end.
This is the figure to report in any operational spec — not just the 50ms processing budget.

---

## Context — Existing Codebase Gaps

The existing system (`trajectory_reuse/predictor.py`) is a **pure physics predictor** — it does
outlier rejection, EMA smoothing, weighted velocity estimation, and kinematic rollout. It handles
EO and IR modalities via `modality_config.json` with per-modality distance and gap thresholds.
There is **no live detection stage** — the pipeline consumes pre-labeled COCO annotations.

Three gaps to fill:

1. **Live detection + tracking** — replace offline COCO annotation with a real-time inference
   stage producing `(track_id, center, object_type)` per frame.
2. **ML prediction head** — train a model that plugs into `PredictionMode.ML_ONLY` / `HYBRID`.
3. **Threat classification** — add `threat_level` to `TrajectoryPrediction.diagnostics`.

---

## Decision 1 — Detection Backbone

### Evaluated Options

| Option | Small-object recall | IR support | ONNX export | RTX 3080 latency |
|---|---|---|---|---|
| YOLOv8-small + P2 head | ★★★★☆ | trivial | mature | ~8ms |
| RT-DETR-L | ★★★★★ | needs adapting | beta | ~18ms |
| YOLOv9-tiny (GELAN) | ★★★☆☆ | trivial | partial | ~7ms |
| Cascade R-CNN | ★★★★★ | harder | slow | ~60ms |

### Decision: **YOLOv8-small + P2 head + SAHI on EO only; IR runs natively**

#### EO cameras — SAHI 2×2 on 1920×1080 live stream

The O4 link delivers 1920×1080@30fps to the ground station — this is the working resolution,
not the 48MP native sensor. A 2×2 SAHI tile scheme divides each frame into 4 overlapping
960×540 crops, each padded to 640×640 for YOLOv8 inference. This costs ~4 YOLOv8 runs but
provides +15–25% mAP on small targets (10–200 px²) with no retraining.

**Which EO camera to use:** the telephoto (168mm, 15° FOV) is the primary detection camera
for distant drones because the smaller FOV means each pixel covers less physical area → drones
are larger in the frame. The wide camera (82°) handles simultaneous wide-area search at lower
detection range.

**P2 head** (stride-4 detection layer) is essential: without it, a 14×14 px drone disappears
below the stride-8 minimum detection threshold.

#### IR camera — native 640×512, no SAHI needed

The 640×512 IR frame fits natively into YOLOv8's 640×640 input (16px height padding only).
SAHI tiling is unnecessary and would hurt latency — the IR sensor already provides a pixel
density appropriate for the 45° FOV at typical drone engagement ranges.

The Ultra High-Resolution mode (1280×1024) is available but should only be used for
post-analysis, not real-time inference, as it would require SAHI and pushes latency over budget.

#### Sensor fusion (both streams live simultaneously)

1. Run YOLOv8-EO and YOLOv8-IR in parallel on separate CUDA streams.
2. Apply per-modality IR coordinate offset from `modality_config.json` to IR detections.
3. Merge using weighted NMS:
   - IR detections get `thermal_boost = 1.15` confidence multiplier when mean frame
     luminance (from EO) < 40/255 (night / low-visibility).
   - At daytime, both sensors weighted equally.
4. Pass merged detections to a single shared tracker.

---

## Decision 2 — Camera Motion Compensation

### The M4T problem

Because the M4T itself is a flying drone, the surveillance camera is always moving. Without
compensation, every frame shift causes all tracked objects to appear to move simultaneously —
corrupting velocity estimates in the physics predictor and Kalman filter states in the tracker.

### Evaluated CMC approaches

| Approach | Accuracy | Latency | Requires |
|---|---|---|---|
| Sparse optical flow (BoT-SORT default) | ★★★☆☆ | ~3ms | GPU |
| ECC (Enhanced Correlation Coefficient) | ★★★★☆ | ~5ms | CPU |
| DJI SDK telemetry (gimbal + GPS/IMU) | ★★★★★ | ~0.5ms | SDK subscription |

### Decision: **DJI SDK telemetry-based CMC**

The DJI Onboard SDK provides gimbal pitch/roll/yaw and GPS/IMU at up to 400Hz — well above
the 30fps frame rate. At each frame, compute the camera-plane homography from the gimbal angle
delta between consecutive frames and the platform translation from GPS/IMU. Apply this
homography to all tracked bounding boxes before association.

This replaces BoT-SORT's optical flow CMC with a cheaper, more accurate alternative. The
SDK subscription adds only ~0.5ms overhead vs ~3ms for sparse optical flow.

**Fallback:** if the SDK telemetry stream drops, fall back to ECC-based CMC automatically.

---

## Decision 3 — Object Tracker

### Decision: **BoT-SORT with DJI telemetry CMC, re-ID disabled**

#### Justification

**ByteTrack lineage:** BoT-SORT inherits low-confidence detection usage in a second-round
association pass. Drones at range produce weak, flickering detections — discarding
low-confidence detections causes constant track fragmentation.

**Re-ID disabled:** re-ID feature extraction adds ~8ms and pushes the latency over budget.
Lost-track recovery uses the ML predictor's forecasted position as the Kalman prior for
re-association instead (see Deliverable 6 — Adversarial Hardening).

**Multi-class track management:** BoT-SORT maintains per-class association cost matrices,
preventing drone↔bird ID swaps.

#### Tracker → Predictor Interface

```
BoT-SORT per-frame output:
  [track_id, x1, y1, x2, y2, class_id, detection_confidence]

Adapter layer:
  center      = ((x1 + x2) / 2, (y1 + y2) / 2)
  object_type = CLASS_MAP[class_id]       # "drone", "pedestrian", "vehicle", etc.
  area        = (x2 - x1) * (y2 - y1)   # proxy for depth / range change

history_buffer[track_id].append(center)   # rolling deque, maxlen=30

predictor.predict(
    track_id    = track_id,
    trajectory  = list(history_buffer[track_id]),
    object_type = object_type,
    context     = {"max_speed": SPEED_CAP[object_type], "bbox_area": area},
)
```

---

## Decision 4 — Laser Rangefinder Usage

### Constraint: 1 Hz measurement frequency

The laser rangefinder measures depth at only 1 Hz — one reading per 30 video frames. It
**cannot** be used as a per-frame depth input to the tracker or predictor.

### Decision: **Scale calibration only — px/meter mapping**

At each 1Hz laser reading:
1. Compute current px/meter scale: `scale = frame_width_px / (2 × D × tan(FOV/2))`
   where D is laser range and FOV is the active camera's horizontal FOV.
2. Store this scale in a `range_state` dict keyed by `(gimbal_pitch, gimbal_yaw)` bucket.
3. Between readings, interpolate scale linearly using the platform velocity from GPS/IMU.

This converts the physics predictor's units from `px/step` to approximate `m/step` for
drone tracks, enabling the speed cap in `_speed_cap()` to use real-world m/s values
(typical DJI consumer drone: 15–20 m/s; FPV racing: up to 50 m/s).

The `context` dict passed to `predict()` gains a `metric_scale` key when a valid laser
reading is available:
```python
context = {"max_speed": 18.0, "bbox_area": area, "metric_scale": px_per_meter}
```

---

## Decision 5 — ML Prediction Model

### Evaluated Options

| Option | Variable history | Multi-modal output | Multi-agent | Drone hover | Stability |
|---|---|---|---|---|---|
| Social LSTM | poor | ✗ | O(N²) pool | poor | good |
| Social GAN | poor | ✓ (samples) | O(N²) pool | poor | unstable |
| Temporal Transformer | ★★★★★ | ✓ (K heads) | ★★★★★ | ★★★★☆ | good |
| Mamba / SSM | ★★★★★ | partial | ★★★☆☆ | unknown | immature |

### Decision: **Lightweight Temporal Transformer with M4T-specific modifications**

#### Architecture Specification

```
Input per timestep (one agent):
  [x_norm, y_norm, vx, vy, ax, ay, det_confidence, bbox_area_norm,
   metric_scale_flag, type_embedding(8), modality_embedding(4)]
  → feature dim = 17

Encoder:
  Linear(17 → 64) + LayerNorm
  4× TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
  Causal mask: each timestep attends only to itself and earlier steps

Social attention (multi-agent):
  For each agent, cross-attend to all other agents' encoder outputs
  1× TransformerDecoderLayer(d_model=64, nhead=4)
  Disabled when N_agents == 1 to avoid overhead for single-drone scenes

Decoder:
  K=6 parallel MLP heads → (x, y) × T_pred each + softmax mixture weights
  → output: [K, T_pred, 2]  +  [K] weights

Inference: argmax(weights) hypothesis → primary output
           remaining K-1 → uncertainty → confidence score
```

#### Drone-Specific Modifications

**1. Hover-aware masking**
When `speed < HOVER_SPEED_THRESHOLD` (0.4 px/step from `predictor.py`), append a learned
`hover_token` embedding to the sequence. Biases the decoder toward near-zero displacement
without requiring the model to infer hover state purely from velocity.

**2. Short-history padding**
Prepend learned `[PAD]` tokens for tracks with fewer than `max_history=30` observations.
Allows valid inference from 3 observed points onward — matching the existing `min_history`
constraint in `ReusableTrajectoryPredictor`.

**3. Dual-modality conditioning**
Separate 4-dimensional embeddings for EO and IR modalities, concatenated per timestep.
EO and IR have different pixel scales for the same physical motion (IR at 640×512 and 45° FOV
vs EO at 1920×1080 and 15°/35°/82° FOV) — the model must learn these scale differences.

**4. Object-type conditioning**
Learned 8-dimensional embedding for each of the 5 types (`drone`, `pedestrian`, `cyclist`,
`vehicle`, `emergency`) concatenated to each timestep feature.

**5. Multi-modal WTA training**
K=6 hypothesis heads trained with Winner-Takes-All loss. Only the closest hypothesis to
ground truth contributes gradient each step. Prevents mode collapse for bimodal drone
motion (hover ↔ cruise).

**6. Confidence gate into existing `_select_mode()`**
```python
def _select_mode(self, object_type, physics_confidence, ml_confidence):
    if object_type == "emergency":
        return PredictionMode.EMERGENCY
    if ml_confidence > 0.65 and physics_confidence > 0.55:
        return PredictionMode.HYBRID
    if ml_confidence > 0.65:
        return PredictionMode.ML_ONLY
    return PredictionMode.PHYSICS_ONLY
```

---

## Decision 6 — Training Loss

### Decision: **minADE + NLL regularization on mixture weights**

**minADE** (minimum Average Displacement Error over K hypotheses) is the standard loss for
multi-modal trajectory prediction and maps directly to the K=6 WTA training setup.

**NLL regularization** on mixture weights prevents degenerate distributions where one head
dominates, causing the model to collapse to unimodal prediction.

**Not CVAE:** CVAE requires a recognition network at train time and sampling at inference
time. For ≤50ms real-time deployment, deterministic K-head inference is preferable.

---

## Decision 7 — Threat Level Scoring

Added to `TrajectoryPrediction.diagnostics` as a string field with three values:

| Level | Criteria |
|---|---|
| `benign` | Non-drone OR (speed < 2.0 px/step AND intention == "hover") |
| `suspicious` | Drone: speed > 8.0 px/step OR projected heading toward restricted zone centroid |
| `confirmed_threat` | Drone: high speed + heading toward restricted zone + confidence > 0.75 |

Stateless function taking `TrajectoryPrediction` as input. No model changes required.

---

## Latency Budget (RTX 3080, 1920×1080 EO + 640×512 IR)

| Stage | Component | Estimated time |
|---|---|---|
| Detection — EO | YOLOv8-small + P2 + SAHI 2×2 (4 tiles, parallel) | ~12ms |
| Detection — IR | YOLOv8-IR native 640×512 (parallel with EO) | ~4ms |
| Sensor fusion NMS | weighted NMS merge | ~0.5ms |
| CMC | DJI SDK telemetry homography | ~0.5ms |
| Tracking | BoT-SORT association | ~3ms |
| History buffer update | deque appends × N tracks | ~0.3ms |
| ML prediction | Transformer batched forward × N tracks | ~6ms |
| Physics prediction | `ReusableTrajectoryPredictor` × N tracks | ~1ms |
| Mode gate + merge | confidence selection | ~0.3ms |
| Threat scoring | stateless function | ~0.2ms |
| **Total** | | **~28ms** |

EO and IR detectors run in parallel on separate CUDA streams, so only the slower one (EO)
is on the critical path.

**Headroom:** ~22ms before the 50ms ground-processing budget.  
**End-to-end:** 130ms O4 link + 28ms processing = **~158ms** total system latency.

---

## Open Questions (non-blocking for Deliverable 2)

1. **Max simultaneous drone tracks** — determines whether social attention needs sparse
   approximation (threshold: N > 15 agents per frame).
2. **Existing labeled trajectory count** — determines ratio of public dataset vs own data
   needed in training mix.
3. **Which EO camera is primary** — telephoto (168mm) for long-range or wide (82°) for
   area coverage? Affects SAHI tile overlap ratio and training data normalization.
