#!/usr/bin/env python3
"""
Deliverable 2 — Training data preprocessor.

Converts drone_trajectories.json (output of build_trajectories.py) into
PyTorch tensors ready for training the MLTrajectoryPredictor.

Output layout (one .pt file per split):
  obs_traj      [N, T_obs, 2]   normalised (x, y) history in local frame
  pred_traj     [N, T_pred, 2]  normalised (x, y) future in local frame
  obs_vel       [N, T_obs, 2]   (vx, vy) per step in local frame
  obs_acc       [N, T_obs, 2]   (ax, ay) per step in local frame
  obs_area_norm [N, T_obs, 1]   bbox area normalised by sequence median
  obj_type      [N]             integer class id (see OBJECT_TYPE_MAP)
  modality      [N]             0=EO, 1=IR
  is_hover      [N, T_obs]      1 where speed < HOVER_SPEED_THRESHOLD
  seq_id        [N]             source sequence index (for debugging)
  track_id      [N]             source track id (for debugging)

Usage:
    python3 -m trajectory_reuse.prepare_training_data \\
        --trajectories output/drone_trajectories.json \\
        --output-dir   output/training_tensors \\
        --obs-len      20 \\
        --pred-len     30 \\
        --fps          30 \\
        --val-ratio    0.15 \\
        --test-ratio   0.10 \\
        --augment

Run with --dry-run to print dataset statistics without writing any files.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── constants matching predictor.py ──────────────────────────────────────────
HOVER_SPEED_THRESHOLD = 0.4   # px/step; from predictor.py

# Object type encoding — must stay in sync with models.py / predictor.py
OBJECT_TYPE_MAP: Dict[str, int] = {
    "drone":      0,
    "pedestrian": 1,
    "cyclist":    2,
    "vehicle":    3,
    "emergency":  4,
    "unknown":    5,
}

MODALITY_MAP: Dict[str, int] = {
    "EO": 0,
    "IR": 1,
}

# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class Window:
    """One training sample extracted from a track."""
    obs_xy: List[Tuple[float, float]]        # T_obs points in local frame
    pred_xy: List[Tuple[float, float]]       # T_pred points in local frame
    obs_vel: List[Tuple[float, float]]       # velocities for obs window
    obs_acc: List[Tuple[float, float]]       # accelerations for obs window
    obs_area_norm: List[float]               # normalised bbox areas
    is_hover: List[int]                      # 1 where speed < threshold
    obj_type: int
    modality: int
    seq_id: int
    track_id: int


@dataclass
class RawPoint:
    frame_number: int
    x: float
    y: float
    area: float


# ── pure computation helpers ──────────────────────────────────────────────────

def _finite_diff_velocity(
    points: List[Tuple[float, float]], dt: float
) -> List[Tuple[float, float]]:
    """Compute per-step velocity via central differences (forward/backward at ends)."""
    n = len(points)
    vel: List[Tuple[float, float]] = []
    for i in range(n):
        if i == 0:
            vx = (points[1][0] - points[0][0]) / dt if n > 1 else 0.0
            vy = (points[1][1] - points[0][1]) / dt if n > 1 else 0.0
        elif i == n - 1:
            vx = (points[-1][0] - points[-2][0]) / dt
            vy = (points[-1][1] - points[-2][1]) / dt
        else:
            vx = (points[i + 1][0] - points[i - 1][0]) / (2.0 * dt)
            vy = (points[i + 1][1] - points[i - 1][1]) / (2.0 * dt)
        vel.append((vx, vy))
    return vel


def _finite_diff_accel(
    vel: List[Tuple[float, float]], dt: float
) -> List[Tuple[float, float]]:
    """Compute per-step acceleration from velocity via central differences."""
    n = len(vel)
    acc: List[Tuple[float, float]] = []
    for i in range(n):
        if i == 0:
            ax = (vel[1][0] - vel[0][0]) / dt if n > 1 else 0.0
            ay = (vel[1][1] - vel[0][1]) / dt if n > 1 else 0.0
        elif i == n - 1:
            ax = (vel[-1][0] - vel[-2][0]) / dt
            ay = (vel[-1][1] - vel[-2][1]) / dt
        else:
            ax = (vel[i + 1][0] - vel[i - 1][0]) / (2.0 * dt)
            ay = (vel[i + 1][1] - vel[i - 1][1]) / (2.0 * dt)
        acc.append((ax, ay))
    return acc


def _to_local_frame(
    points: List[Tuple[float, float]], origin: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """Translate so that origin becomes (0, 0)."""
    ox, oy = origin
    return [(x - ox, y - oy) for x, y in points]


def _speed(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)


def _extract_raw_points(track_points: List[Dict]) -> List[RawPoint]:
    pts: List[RawPoint] = []
    for p in track_points:
        cx, cy = p["center"]
        pts.append(RawPoint(
            frame_number=p["frame_number"],
            x=float(cx),
            y=float(cy),
            area=float(p.get("area", 0.0)),
        ))
    return pts


def _median(values: List[float]) -> float:
    if not values:
        return 1.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# ── augmentation ─────────────────────────────────────────────────────────────

def _rotate_90(
    points: List[Tuple[float, float]], k: int
) -> List[Tuple[float, float]]:
    """Rotate by k×90° around origin."""
    k = k % 4
    out = points
    for _ in range(k):
        out = [(-y, x) for x, y in out]
    return out


def _flip_horizontal(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(-x, y) for x, y in points]


def _flip_vertical(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(x, -y) for x, y in points]


def _scale_speed(
    points: List[Tuple[float, float]], factor: float, origin: Tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float]]:
    """Scale displacement from origin by factor (simulates faster/slower motion)."""
    ox, oy = origin
    return [(ox + (x - ox) * factor, oy + (y - oy) * factor) for x, y in points]


def _insert_hover(
    points: List[Tuple[float, float]],
    hover_duration: int,
) -> List[Tuple[float, float]]:
    """
    Insert a stationary segment at a random position in the observation window.
    Simulates a drone stopping mid-trajectory — a common real-world behaviour
    that physics-only predictors systematically underfit.
    """
    if len(points) <= hover_duration + 2:
        return points
    insert_at = random.randint(1, len(points) - hover_duration - 1)
    hover_pos = points[insert_at]
    hovered = (
        points[:insert_at]
        + [hover_pos] * hover_duration
        + points[insert_at: len(points) - hover_duration]
    )
    return hovered


def _augment_window(window: Window, rng: random.Random) -> List[Window]:
    """Return a list of augmented copies (may be empty if augmentation disabled)."""
    results: List[Window] = []

    # Collect all xy lists for joint transformation
    all_obs = list(window.obs_xy)
    all_pred = list(window.pred_xy)

    def _make_copy(obs: List, pred: List, vel: List, acc: List) -> Window:
        hover = [1 if _speed(v[0], v[1]) < HOVER_SPEED_THRESHOLD else 0 for v in vel]
        return Window(
            obs_xy=obs, pred_xy=pred,
            obs_vel=vel, obs_acc=acc,
            obs_area_norm=window.obs_area_norm[:],
            is_hover=hover,
            obj_type=window.obj_type,
            modality=window.modality,
            seq_id=window.seq_id,
            track_id=window.track_id,
        )

    # 1. Random rotation (90° increments)
    k = rng.choice([1, 2, 3])
    aug_obs = _rotate_90(all_obs, k)
    aug_pred = _rotate_90(all_pred, k)
    aug_vel = _rotate_90(window.obs_vel, k)
    aug_acc = _rotate_90(window.obs_acc, k)
    results.append(_make_copy(aug_obs, aug_pred, aug_vel, aug_acc))

    # 2. Horizontal flip
    if rng.random() < 0.5:
        f_obs = _flip_horizontal(all_obs)
        f_pred = _flip_horizontal(all_pred)
        f_vel = _flip_horizontal(window.obs_vel)
        f_acc = _flip_horizontal(window.obs_acc)
        results.append(_make_copy(f_obs, f_pred, f_vel, f_acc))

    # 3. Speed scaling (0.5× – 2.0×, excluding 1.0×)
    factor = rng.uniform(0.5, 2.0)
    while 0.95 < factor < 1.05:
        factor = rng.uniform(0.5, 2.0)
    s_obs = _scale_speed(all_obs, factor)
    s_pred = _scale_speed(all_pred, factor)
    s_vel = [(vx * factor, vy * factor) for vx, vy in window.obs_vel]
    s_acc = [(ax * factor, ay * factor) for ax, ay in window.obs_acc]
    results.append(_make_copy(s_obs, s_pred, s_vel, s_acc))

    # 4. Hover insertion (drones only)
    if window.obj_type == OBJECT_TYPE_MAP["drone"] and rng.random() < 0.3:
        hover_dur = rng.randint(2, 5)
        h_obs = _insert_hover(list(all_obs), hover_dur)
        h_vel = _finite_diff_velocity(h_obs, dt=1.0)
        h_acc = _finite_diff_accel(h_vel, dt=1.0)
        results.append(_make_copy(h_obs[:len(all_obs)], all_pred, h_vel[:len(all_obs)], h_acc[:len(all_obs)]))

    return results


# ── window extraction ─────────────────────────────────────────────────────────

def _extract_windows(
    raw: List[RawPoint],
    median_area: float,
    obj_type: int,
    modality: int,
    seq_id: int,
    track_id: int,
    obs_len: int,
    pred_len: int,
    stride: int,
    fps: float,
) -> List[Window]:
    """Slide a window of length obs_len + pred_len over the track."""
    total_len = obs_len + pred_len
    if len(raw) < total_len:
        return []

    dt = 1.0 / fps
    xy = [(p.x, p.y) for p in raw]
    areas = [p.area for p in raw]
    vel = _finite_diff_velocity(xy, dt)
    acc = _finite_diff_accel(vel, dt)

    norm_area = median_area if median_area > 0.0 else 1.0

    windows: List[Window] = []
    for start in range(0, len(raw) - total_len + 1, stride):
        obs_end = start + obs_len
        pred_end = obs_end + pred_len

        obs_xy_abs = xy[start:obs_end]
        pred_xy_abs = xy[obs_end:pred_end]

        # Local frame: origin = last observed position
        origin = obs_xy_abs[-1]
        obs_local = _to_local_frame(obs_xy_abs, origin)
        pred_local = _to_local_frame(pred_xy_abs, origin)

        obs_vel_local = _to_local_frame(vel[start:obs_end], (0.0, 0.0))
        obs_acc_local = _to_local_frame(acc[start:obs_end], (0.0, 0.0))

        obs_area = [a / norm_area for a in areas[start:obs_end]]
        is_hover = [
            1 if _speed(v[0], v[1]) < HOVER_SPEED_THRESHOLD else 0
            for v in obs_vel_local
        ]

        windows.append(Window(
            obs_xy=obs_local,
            pred_xy=pred_local,
            obs_vel=obs_vel_local,
            obs_acc=obs_acc_local,
            obs_area_norm=obs_area,
            is_hover=is_hover,
            obj_type=obj_type,
            modality=modality,
            seq_id=seq_id,
            track_id=track_id,
        ))

    return windows


# ── main processing ───────────────────────────────────────────────────────────

def prepare_training_data(
    trajectories_path: Path,
    output_dir: Path,
    obs_len: int = 20,
    pred_len: int = 30,
    stride: int = 5,
    fps: float = 30.0,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    augment: bool = True,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Load trajectories JSON, extract windows, augment, split, and save tensors.

    Returns a summary dict with window counts per split.
    """
    rng = random.Random(seed)

    logger.info("Loading trajectories from %s", trajectories_path)
    with trajectories_path.open() as fh:
        data = json.load(fh)

    category_name = data.get("category_name", "drone")
    sequences = data.get("sequences", [])
    logger.info(
        "Loaded %d sequences, category=%s", len(sequences), category_name
    )

    all_windows: List[Window] = []
    skipped_short = 0

    for seq_id, seq in enumerate(sequences):
        modality_str = seq.get("modality", "EO")
        modality = MODALITY_MAP.get(modality_str, 0)
        tracks = seq.get("tracks", [])

        # Compute per-sequence median area for normalisation
        all_areas_in_seq = [
            p.get("area", 0.0)
            for track in tracks
            for p in track.get("points", [])
        ]
        median_area = _median([a for a in all_areas_in_seq if a > 0.0])

        for track in tracks:
            track_id = track.get("track_id", 0)
            raw = _extract_raw_points(track.get("points", []))
            # Per-track object_type overrides dataset-level category_name
            track_cat = track.get("object_type", category_name)
            obj_type = OBJECT_TYPE_MAP.get(track_cat, OBJECT_TYPE_MAP["unknown"])

            windows = _extract_windows(
                raw=raw,
                median_area=median_area,
                obj_type=obj_type,
                modality=modality,
                seq_id=seq_id,
                track_id=track_id,
                obs_len=obs_len,
                pred_len=pred_len,
                stride=stride,
                fps=fps,
            )

            if not windows:
                skipped_short += 1
                continue

            all_windows.extend(windows)
            if augment:
                for w in windows:
                    all_windows.extend(_augment_window(w, rng))

    logger.info(
        "Extracted %d windows (tracks skipped for length: %d)",
        len(all_windows), skipped_short,
    )

    if not all_windows:
        raise ValueError(
            "No training windows extracted. "
            "Check that trajectories have at least obs_len + pred_len points."
        )

    # Split by sequence id to prevent data leakage between splits
    seq_ids = sorted({w.seq_id for w in all_windows})
    rng.shuffle(seq_ids)
    n_seq = len(seq_ids)
    n_test = max(1, int(n_seq * test_ratio))
    n_val = max(1, int(n_seq * val_ratio))
    test_seqs = set(seq_ids[:n_test])
    val_seqs = set(seq_ids[n_test: n_test + n_val])

    splits: Dict[str, List[Window]] = {"train": [], "val": [], "test": []}
    for w in all_windows:
        if w.seq_id in test_seqs:
            splits["test"].append(w)
        elif w.seq_id in val_seqs:
            splits["val"].append(w)
        else:
            splits["train"].append(w)

    # Log class distribution
    for split_name, ws in splits.items():
        type_counts: Dict[int, int] = {}
        mod_counts: Dict[int, int] = {}
        for w in ws:
            type_counts[w.obj_type] = type_counts.get(w.obj_type, 0) + 1
            mod_counts[w.modality] = mod_counts.get(w.modality, 0) + 1
        inv_type = {v: k for k, v in OBJECT_TYPE_MAP.items()}
        inv_mod = {v: k for k, v in MODALITY_MAP.items()}
        type_str = ", ".join(f"{inv_type.get(k, k)}={v}" for k, v in sorted(type_counts.items()))
        mod_str = ", ".join(f"{inv_mod.get(k, k)}={v}" for k, v in sorted(mod_counts.items()))
        logger.info("  %s: %d windows  [%s]  [%s]", split_name, len(ws), type_str, mod_str)

    summary = {split: len(ws) for split, ws in splits.items()}

    if dry_run:
        logger.info("[dry-run] No files written.")
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to save training tensors. "
            "Install with: pip install torch"
        ) from None

    def _windows_to_tensors(ws: List[Window]) -> Dict[str, "torch.Tensor"]:
        import torch as t
        n = len(ws)
        obs_traj     = t.zeros(n, obs_len, 2)
        pred_traj    = t.zeros(n, pred_len, 2)
        obs_vel_t    = t.zeros(n, obs_len, 2)
        obs_acc_t    = t.zeros(n, obs_len, 2)
        obs_area_t   = t.zeros(n, obs_len, 1)
        is_hover_t   = t.zeros(n, obs_len, dtype=t.long)
        obj_type_t   = t.zeros(n, dtype=t.long)
        modality_t   = t.zeros(n, dtype=t.long)
        seq_id_t     = t.zeros(n, dtype=t.long)
        track_id_t   = t.zeros(n, dtype=t.long)

        for i, w in enumerate(ws):
            for j, (x, y) in enumerate(w.obs_xy):
                obs_traj[i, j, 0] = x
                obs_traj[i, j, 1] = y
            for j, (x, y) in enumerate(w.pred_xy):
                pred_traj[i, j, 0] = x
                pred_traj[i, j, 1] = y
            for j, (vx, vy) in enumerate(w.obs_vel):
                obs_vel_t[i, j, 0] = vx
                obs_vel_t[i, j, 1] = vy
            for j, (ax, ay) in enumerate(w.obs_acc):
                obs_acc_t[i, j, 0] = ax
                obs_acc_t[i, j, 1] = ay
            for j, a in enumerate(w.obs_area_norm):
                obs_area_t[i, j, 0] = a
            for j, h in enumerate(w.is_hover):
                is_hover_t[i, j] = h
            obj_type_t[i] = w.obj_type
            modality_t[i] = w.modality
            seq_id_t[i] = w.seq_id
            track_id_t[i] = w.track_id

        return {
            "obs_traj":      obs_traj,
            "pred_traj":     pred_traj,
            "obs_vel":       obs_vel_t,
            "obs_acc":       obs_acc_t,
            "obs_area_norm": obs_area_t,
            "is_hover":      is_hover_t,
            "obj_type":      obj_type_t,
            "modality":      modality_t,
            "seq_id":        seq_id_t,
            "track_id":      track_id_t,
        }

    for split_name, ws in splits.items():
        if not ws:
            logger.warning("Split '%s' is empty — skipping.", split_name)
            continue
        tensors = _windows_to_tensors(ws)
        out_path = output_dir / f"{split_name}.pt"
        import torch
        torch.save(tensors, out_path)
        logger.info("Saved %s → %s", split_name, out_path)

    # Save metadata alongside tensors
    meta = {
        "obs_len": obs_len,
        "pred_len": pred_len,
        "stride": stride,
        "fps": fps,
        "augmented": augment,
        "seed": seed,
        "object_type_map": OBJECT_TYPE_MAP,
        "modality_map": MODALITY_MAP,
        "source": str(trajectories_path),
        "split_counts": summary,
        "feature_dim": 17,
        "feature_layout": [
            "x_norm", "y_norm",           # 0-1
            "vx", "vy",                    # 2-3
            "ax", "ay",                    # 4-5
            "det_confidence",              # 6  (filled at inference time)
            "bbox_area_norm",              # 7
            "metric_scale_flag",           # 8  (filled at inference time)
            "type_embed_0..7",             # 9-16 (learned embedding, not raw)
            "modality_embed_0..3",         # appended in model
        ],
    }
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Metadata written to %s", meta_path)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert build_trajectories.py output to ML training tensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectories", required=True,
                        help="Path to drone_trajectories.json.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write train.pt / val.pt / test.pt.")
    parser.add_argument("--obs-len", type=int, default=20,
                        help="Observation window length in frames.")
    parser.add_argument("--pred-len", type=int, default=30,
                        help="Prediction horizon in frames.")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sliding window stride in frames.")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Camera frame rate (DJI M4T = 30fps).")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of sequences reserved for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="Fraction of sequences reserved for testing.")
    parser.add_argument("--augment", action="store_true",
                        help="Apply trajectory augmentation (rotation, flip, speed scale, hover).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print statistics without writing files.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        summary = prepare_training_data(
            trajectories_path=Path(args.trajectories),
            output_dir=Path(args.output_dir),
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            stride=args.stride,
            fps=args.fps,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            augment=args.augment,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except (ValueError, ImportError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None

    logger.info("Done.")
    for split, count in summary.items():
        logger.info("  %-6s %d windows", split + ":", count)


if __name__ == "__main__":
    main()
