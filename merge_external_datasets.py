#!/usr/bin/env python3
"""
merge_external_datasets.py — Convert and merge public tracking datasets into
our drone_trajectories.json format.

Supported input formats
-----------------------
  dut-anti-uav   DUT Anti-UAV Tracking (groundtruth.txt per sequence)
  visdrone-mot   VisDrone2019-MOT (MOT CSV annotations)
  mot            Any generic MOT-format dataset (frame,id,x,y,w,h,conf,class,vis)

Download instructions are printed by running with --help or at the end of this
docstring.

Quick usage
-----------
# 1. DUT Anti-UAV Tracking (MIT licence — directly tracks UAVs)
python3 merge_external_datasets.py \\
    --input  data/DUT-Anti-UAV/Tracking \\
    --format dut-anti-uav \\
    --base   output/drone_trajectories.json \\
    --output output/drone_trajectories_merged.json

# 2. VisDrone-MOT (motion diversity — pedestrians/vehicles; object_type set automatically)
python3 merge_external_datasets.py \\
    --input  data/VisDrone2019-MOT-train/annotations \\
    --format visdrone-mot \\
    --base   output/drone_trajectories_merged.json \\
    --output output/drone_trajectories_merged.json

# 3. Any generic MOT directory
python3 merge_external_datasets.py \\
    --input  data/MyDataset \\
    --format mot \\
    --object-type drone \\
    --modality EO \\
    --base   output/drone_trajectories.json \\
    --output output/drone_trajectories_merged.json

# 4. After merging, regenerate training tensors:
python3 -m trajectory_reuse.prepare_training_data \\
    --trajectories output/drone_trajectories_merged.json \\
    --output-dir   output/training_tensors_v2

# 5. Retrain:
python3 train_ml_predictor.py --config configs/train_config.yaml \\
    --data-dir output/training_tensors_v2

Dataset download commands
-------------------------
# DUT Anti-UAV (GitHub ZIP — ~200 MB for tracking subset)
git clone --depth 1 https://github.com/wangdongdut/DUT-Anti-UAV data/DUT-Anti-UAV
# Tracking GT is at: data/DUT-Anti-UAV/Tracking/train/ and /test/

# VisDrone-MOT 2019 (Google Drive — ~7 GB for training split)
# Manual download from:
#   https://github.com/VisDrone/VisDrone-Dataset  → Task 4: MOT
# Extract to: data/VisDrone2019-MOT-train/

# Anti-UAV 2022 Challenge (thermal IR + EO, single UAV tracking)
# https://anti-uav.github.io/  → request access form
# Convert with --format mot after extracting GT files

# M3OT (RGB+IR, MOT format, Nature 2025)
# https://figshare.com/s/01fa8d1163f4e9a5a13a
# Extract to: data/M3OT/ then use --format mot
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── object-type mapping (must stay in sync with prepare_training_data.py) ─────

VISDRONE_CLASS_MAP: Dict[int, str] = {
    1:  "pedestrian",
    2:  "pedestrian",   # person (sitting)
    3:  "cyclist",
    4:  "vehicle",      # car
    5:  "vehicle",      # van
    6:  "vehicle",      # truck
    7:  "vehicle",      # tricycle
    8:  "vehicle",      # awning-tricycle
    9:  "vehicle",      # bus
    10: "vehicle",      # motor
}

# ── shared helpers ─────────────────────────────────────────────────────────────

def _center_from_xywh(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    return (x + w / 2.0, y + h / 2.0)


def _area_from_wh(w: float, h: float) -> float:
    return abs(w * h)


def _make_sequence(
    seq_name: str,
    modality: str,
    tracks: List[Dict],
    source: str,
    min_track_len: int = 4,
) -> Optional[Dict]:
    """Wrap converted tracks into our sequence schema; returns None if empty."""
    valid_tracks = [t for t in tracks if len(t["points"]) >= min_track_len]
    if not valid_tracks:
        return None
    total_frames = max(
        (p["frame_number"] for t in valid_tracks for p in t["points"]), default=0
    ) + 1
    return {
        "sequence_key":          seq_name,
        "date_name":             source,
        "modality":              modality,
        "experiment_name":       seq_name,
        "frame_count":           total_frames,
        "annotated_frame_count": total_frames,
        "unannotated_frame_count": 0,
        "tracking_config":       {"source": source},
        "tracks":                valid_tracks,
    }


def _make_track(track_id: int, points: List[Dict], object_type: str) -> Dict:
    frames = sorted(p["frame_number"] for p in points)
    return {
        "track_id":    track_id,
        "object_type": object_type,
        "start_frame": frames[0] if frames else 0,
        "end_frame":   frames[-1] if frames else 0,
        "length":      len(points),
        "diagnostics": {"source": "external"},
        "points":      sorted(points, key=lambda p: p["frame_number"]),
    }


def _make_point(frame_number: int, cx: float, cy: float, bx: float, by: float,
                bw: float, bh: float) -> Dict:
    return {
        "frame_number": frame_number,
        "file_name":    "",
        "center":       [round(cx, 3), round(cy, 3)],
        "bbox":         [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
        "annotation_id": 0,
        "area":          round(_area_from_wh(bw, bh), 3),
    }


# ── format converters ──────────────────────────────────────────────────────────

def convert_dut_anti_uav(root: Path, modality: str = "EO") -> List[Dict]:
    """
    DUT Anti-UAV Tracking format.

    Expected layout under `root`:
        <root>/
          <split>/          (train / test — both processed)
            <sequence>/
              *.jpg         (frames — not required, skipped)
              groundtruth.txt   one bbox per line: x,y,w,h   (0-indexed, frame order)

    Single UAV per sequence → track_id always 1.
    """
    sequences: List[Dict] = []
    gt_files = list(root.rglob("groundtruth.txt"))
    if not gt_files:
        # Also accept gt.txt (some forks)
        gt_files = list(root.rglob("gt.txt"))
    if not gt_files:
        logger.warning("No groundtruth.txt files found under %s", root)
        return sequences

    for gt_path in sorted(gt_files):
        seq_name = gt_path.parent.name
        points: List[Dict] = []
        try:
            with open(gt_path) as fh:
                for frame_idx, line in enumerate(fh):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = re.split(r"[\s,\t]+", line)
                    if len(parts) < 4:
                        continue
                    try:
                        x, y, w, h = (float(p) for p in parts[:4])
                    except ValueError:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    cx, cy = _center_from_xywh(x, y, w, h)
                    points.append(_make_point(frame_idx, cx, cy, x, y, w, h))
        except OSError as exc:
            logger.warning("Skipping %s: %s", gt_path, exc)
            continue

        track  = _make_track(1, points, object_type="drone")
        seq    = _make_sequence(seq_name, modality, [track], "DUT-Anti-UAV")
        if seq:
            sequences.append(seq)
            logger.info("DUT-Anti-UAV: %s — %d points", seq_name, len(points))

    logger.info("DUT-Anti-UAV total: %d sequences", len(sequences))
    return sequences


def convert_visdrone_mot(annotations_dir: Path) -> List[Dict]:
    """
    VisDrone2019-MOT format.

    Each .txt annotation file is one sequence in MOT CSV format:
        frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
        score, object_category, truncation, occlusion

    Categories are mapped to our OBJECT_TYPE_MAP via VISDRONE_CLASS_MAP.
    Ignored (class=0) or occluded (occlusion=2) detections are skipped.
    Modality is always EO (all VisDrone data is optical).
    """
    sequences: List[Dict] = []
    txt_files = sorted(annotations_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", annotations_dir)
        return sequences

    for txt_path in txt_files:
        seq_name = txt_path.stem
        # Per-track accumulator: track_id → list[dict]
        track_points: Dict[int, List[Dict]] = {}
        track_class:  Dict[int, str]        = {}

        try:
            with open(txt_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        frame_idx  = int(parts[0]) - 1     # 1-indexed → 0-indexed
                        tid        = int(parts[1])
                        bx, by, bw, bh = (float(p) for p in parts[2:6])
                        score      = float(parts[6]) if len(parts) > 6 else 1.0
                        cat_id     = int(parts[7]) if len(parts) > 7 else 0
                        occlusion  = int(parts[9]) if len(parts) > 9 else 0
                    except (ValueError, IndexError):
                        continue

                    # Skip ignored, zero-size, or fully occluded
                    if cat_id == 0 or bw <= 0 or bh <= 0 or occlusion == 2:
                        continue

                    obj_type = VISDRONE_CLASS_MAP.get(cat_id, "unknown")
                    cx, cy   = _center_from_xywh(bx, by, bw, bh)

                    track_points.setdefault(tid, []).append(
                        _make_point(frame_idx, cx, cy, bx, by, bw, bh)
                    )
                    track_class.setdefault(tid, obj_type)
        except OSError as exc:
            logger.warning("Skipping %s: %s", txt_path, exc)
            continue

        tracks = [
            _make_track(tid, pts, track_class.get(tid, "unknown"))
            for tid, pts in track_points.items()
        ]
        seq = _make_sequence(seq_name, "EO", tracks, "VisDrone-MOT")
        if seq:
            sequences.append(seq)
            logger.info(
                "VisDrone-MOT: %s — %d tracks, %d points",
                seq_name, len(seq["tracks"]),
                sum(len(t["points"]) for t in seq["tracks"]),
            )

    logger.info("VisDrone-MOT total: %d sequences", len(sequences))
    return sequences


def convert_mot_generic(
    root: Path,
    object_type: str = "drone",
    modality: str    = "EO",
) -> List[Dict]:
    """
    Generic MOT format (compatible with MOTChallenge, M3OT, Anti-UAV GT).

    Searches recursively for gt.txt or *.txt files.  Each file is assumed to
    be one sequence.  Line format:
        frame, id, x, y, w, h, conf, class, visibility

    Fields after `h` are optional.  Entries with conf=0 are skipped.
    """
    sequences: List[Dict] = []
    gt_files  = list(root.rglob("gt.txt")) or list(root.rglob("*.txt"))

    for gt_path in sorted(gt_files):
        seq_name     = gt_path.parent.name or gt_path.stem
        track_points: Dict[int, List[Dict]] = {}

        try:
            with open(gt_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = re.split(r"[\s,\t]+", line)
                    if len(parts) < 6:
                        continue
                    try:
                        frame_idx = int(parts[0]) - 1   # 1-indexed → 0-indexed
                        tid       = int(parts[1])
                        bx, by, bw, bh = (float(p) for p in parts[2:6])
                        conf = float(parts[6]) if len(parts) > 6 else 1.0
                    except (ValueError, IndexError):
                        continue

                    if conf == 0 or bw <= 0 or bh <= 0:
                        continue

                    cx, cy = _center_from_xywh(bx, by, bw, bh)
                    track_points.setdefault(tid, []).append(
                        _make_point(frame_idx, cx, cy, bx, by, bw, bh)
                    )
        except OSError as exc:
            logger.warning("Skipping %s: %s", gt_path, exc)
            continue

        tracks = [_make_track(tid, pts, object_type) for tid, pts in track_points.items()]
        seq    = _make_sequence(seq_name, modality, tracks, str(root.name))
        if seq:
            sequences.append(seq)
            logger.info(
                "MOT generic: %s — %d tracks", seq_name, len(seq["tracks"])
            )

    logger.info("MOT generic total: %d sequences", len(sequences))
    return sequences


# ── merge ──────────────────────────────────────────────────────────────────────

def merge_into_base(base_path: Path, new_sequences: List[Dict], output_path: Path) -> None:
    """Load base JSON, append new sequences, write merged output."""
    if base_path.exists():
        with open(base_path) as fh:
            base = json.load(fh)
    else:
        base = {
            "source_annotations": str(base_path),
            "frames_root":        "",
            "category_name":      "drone",
            "max_gap":            8,
            "max_distance":       150.0,
            "min_track_length":   4,
            "sequences":          [],
        }

    existing_keys = {s.get("sequence_key", "") for s in base["sequences"]}
    added = 0
    for seq in new_sequences:
        key = seq.get("sequence_key", "")
        if key in existing_keys:
            logger.warning("Duplicate sequence key '%s' — skipping", key)
            continue
        base["sequences"].append(seq)
        existing_keys.add(key)
        added += 1

    # Recompute summary statistics
    all_tracks  = [t for s in base["sequences"] for t in s["tracks"]]
    all_points  = [p for t in all_tracks for p in t["points"]]
    base.update({
        "sequence_count":     len(base["sequences"]),
        "track_count":        len(all_tracks),
        "track_point_count":  len(all_points),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(base, fh, indent=2)

    logger.info(
        "Merged: added %d new sequences → %d total sequences, %d tracks, %d points",
        added, base["sequence_count"], base["track_count"], base["track_point_count"],
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert and merge external tracking datasets into drone_trajectories.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  required=True,
                        help="Root directory of the dataset to convert")
    parser.add_argument("--format", required=True,
                        choices=["dut-anti-uav", "visdrone-mot", "mot"],
                        help="Dataset format")
    parser.add_argument("--base",   default="output/drone_trajectories.json",
                        help="Existing trajectory file to merge into")
    parser.add_argument("--output", default="output/drone_trajectories_merged.json",
                        help="Output merged trajectory file")
    parser.add_argument("--modality", default="EO", choices=["EO", "IR"],
                        help="Sensor modality (used for dut-anti-uav and mot formats)")
    parser.add_argument("--object-type", default="drone",
                        choices=["drone", "pedestrian", "cyclist", "vehicle", "emergency", "unknown"],
                        help="Object type label (used for mot format only)")
    parser.add_argument("--min-track-len", type=int, default=4,
                        help="Minimum points per track; shorter tracks are discarded")
    parser.add_argument("--dry-run", action="store_true",
                        help="Convert and print stats but do not write output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        sys.exit(1)

    fmt = args.format
    if fmt == "dut-anti-uav":
        new_seqs = convert_dut_anti_uav(input_path, modality=args.modality)
    elif fmt == "visdrone-mot":
        new_seqs = convert_visdrone_mot(input_path)
    elif fmt == "mot":
        new_seqs = convert_mot_generic(input_path, args.object_type, args.modality)
    else:
        logger.error("Unknown format: %s", fmt)
        sys.exit(1)

    total_tracks  = sum(len(s["tracks"]) for s in new_seqs)
    total_points  = sum(len(t["points"]) for s in new_seqs for t in s["tracks"])
    print(f"\nConverted: {len(new_seqs)} sequences | {total_tracks} tracks | {total_points} points")

    if args.dry_run:
        print("Dry-run — no files written.")
        return

    merge_into_base(Path(args.base), new_seqs, Path(args.output))
    print(f"Output written to: {args.output}")
    print()
    print("Next steps:")
    print(f"  python3 -m trajectory_reuse.prepare_training_data \\")
    print(f"      --trajectories {args.output} \\")
    print(f"      --output-dir   output/training_tensors_v2")
    print(f"  python3 train_ml_predictor.py --config configs/train_config.yaml \\")
    print(f"      --data-dir output/training_tensors_v2")


if __name__ == "__main__":
    main()
