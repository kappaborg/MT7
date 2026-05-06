from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset_loader import DetectionRecord, FrameRecord, build_frame_records, group_frames_by_sequence


DEFAULT_MODALITY_CONFIG = {
    "EO": {"max_gap": 8, "max_distance": 150.0},
    "IR": {"max_gap": 6, "max_distance": 90.0},
}


@dataclass
class ActiveTrack:
    track_id: int
    last_center: Tuple[float, float]
    last_frame_number: int
    last_area: float
    velocity: Tuple[float, float]
    matched_steps: int
    total_link_distance: float
    largest_gap: int
    points: List[Dict]


def _distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def _predict_center(track: ActiveTrack, frame_number: int) -> Tuple[float, float]:
    frame_gap = max(0, frame_number - track.last_frame_number)
    return (
        track.last_center[0] + track.velocity[0] * frame_gap,
        track.last_center[1] + track.velocity[1] * frame_gap,
    )


def _area_ratio(area_a: float, area_b: float) -> float:
    if area_a <= 0.0 or area_b <= 0.0:
        return float("inf")
    return max(area_a, area_b) / min(area_a, area_b)


def _candidate_score(
    track: ActiveTrack,
    detection: DetectionRecord,
    frame_number: int,
    max_gap: int,
    max_distance: float,
) -> Optional[float]:
    frame_gap = frame_number - track.last_frame_number
    if frame_gap <= 0 or frame_gap > max_gap:
        return None

    predicted_center = _predict_center(track, frame_number)
    predicted_distance = _distance(predicted_center, detection.center)
    distance_limit = max_distance * max(1.0, math.sqrt(frame_gap))
    if predicted_distance > distance_limit:
        return None

    last_distance = _distance(track.last_center, detection.center)
    area_ratio = _area_ratio(track.last_area, detection.area)
    if area_ratio > 3.5:
        return None

    gap_penalty = frame_gap * 4.0
    area_penalty = max(0.0, area_ratio - 1.0) * 8.0
    return predicted_distance + 0.25 * last_distance + gap_penalty + area_penalty


def _match_detections(
    active_tracks: Sequence[ActiveTrack],
    detections: Sequence[DetectionRecord],
    frame_number: int,
    max_gap: int,
    max_distance: float,
) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for track_index, track in enumerate(active_tracks):
        for detection_index, detection in enumerate(detections):
            score = _candidate_score(
                track=track,
                detection=detection,
                frame_number=frame_number,
                max_gap=max_gap,
                max_distance=max_distance,
            )
            if score is not None:
                candidates.append((score, track_index, detection_index))

    candidates.sort(key=lambda item: item[0])
    matched_track_indices = set()
    matched_detection_indices = set()
    matches: List[Tuple[int, int, float]] = []

    for score, track_index, detection_index in candidates:
        if track_index in matched_track_indices or detection_index in matched_detection_indices:
            continue
        matched_track_indices.add(track_index)
        matched_detection_indices.add(detection_index)
        matches.append((track_index, detection_index, score))

    return matches


def _finalize_track(track: ActiveTrack) -> Dict:
    avg_match_score = (
        track.total_link_distance / track.matched_steps if track.matched_steps > 0 else 0.0
    )
    return {
        "track_id": track.track_id,
        "start_frame": track.points[0]["frame_number"],
        "end_frame": track.points[-1]["frame_number"],
        "length": len(track.points),
        "diagnostics": {
            "matched_steps": track.matched_steps,
            "avg_match_score": avg_match_score,
            "largest_gap": track.largest_gap,
        },
        "points": track.points,
    }


def _build_tracks_for_sequence(
    sequence_frames: Sequence[FrameRecord],
    max_gap: int,
    max_distance: float,
    min_track_length: int,
) -> List[Dict]:
    detection_counts = [len(frame.detections) for frame in sequence_frames]
    if detection_counts and max(detection_counts) <= 1:
        direct_points: List[Dict] = []
        direct_tracks: List[Dict] = []
        next_track_id = 1
        previous_frame_number: Optional[int] = None

        for frame in sequence_frames:
            if not frame.detections:
                continue
            detection = frame.detections[0]
            if (
                direct_points
                and previous_frame_number is not None
                and frame.frame_number - previous_frame_number > max_gap
            ):
                if len(direct_points) >= min_track_length:
                    direct_tracks.append(
                        {
                            "track_id": next_track_id,
                            "start_frame": direct_points[0]["frame_number"],
                            "end_frame": direct_points[-1]["frame_number"],
                            "length": len(direct_points),
                            "diagnostics": {
                                "matched_steps": max(0, len(direct_points) - 1),
                                "avg_match_score": 0.0,
                                "largest_gap": max(
                                    (
                                        direct_points[index]["frame_number"]
                                        - direct_points[index - 1]["frame_number"]
                                    )
                                    for index in range(1, len(direct_points))
                                )
                                if len(direct_points) > 1
                                else 0,
                                "source": "direct_single_detection",
                            },
                            "points": direct_points,
                        }
                    )
                    next_track_id += 1
                direct_points = []

            direct_points.append(
                {
                    "frame_number": frame.frame_number,
                    "file_name": frame.file_name,
                    "center": list(detection.center),
                    "bbox": list(detection.bbox),
                    "annotation_id": detection.annotation_id,
                    "area": detection.area,
                }
            )
            previous_frame_number = frame.frame_number

        if len(direct_points) >= min_track_length:
            direct_tracks.append(
                {
                    "track_id": next_track_id,
                    "start_frame": direct_points[0]["frame_number"],
                    "end_frame": direct_points[-1]["frame_number"],
                    "length": len(direct_points),
                    "diagnostics": {
                        "matched_steps": max(0, len(direct_points) - 1),
                        "avg_match_score": 0.0,
                        "largest_gap": max(
                            (
                                direct_points[index]["frame_number"]
                                - direct_points[index - 1]["frame_number"]
                            )
                            for index in range(1, len(direct_points))
                        )
                        if len(direct_points) > 1
                        else 0,
                        "source": "direct_single_detection",
                    },
                    "points": direct_points,
                }
            )

        direct_tracks.sort(key=lambda track: (track["start_frame"], track["track_id"]))
        return direct_tracks

    next_track_id = 1
    active_tracks: List[ActiveTrack] = []
    completed_tracks: List[Dict] = []

    for frame in sequence_frames:
        active_tracks = [
            track
            for track in active_tracks
            if frame.frame_number - track.last_frame_number <= max_gap
        ]

        matches = _match_detections(
            active_tracks=active_tracks,
            detections=frame.detections,
            frame_number=frame.frame_number,
            max_gap=max_gap,
            max_distance=max_distance,
        )
        matched_track_indices = {track_index for track_index, _, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index, _ in matches}

        for track_index, detection_index, score in matches:
            matched_track = active_tracks[track_index]
            detection = frame.detections[detection_index]
            frame_gap = frame.frame_number - matched_track.last_frame_number
            if frame_gap > 0:
                matched_track.velocity = (
                    (detection.center[0] - matched_track.last_center[0]) / frame_gap,
                    (detection.center[1] - matched_track.last_center[1]) / frame_gap,
                )
            matched_track.last_center = detection.center
            matched_track.last_frame_number = frame.frame_number
            matched_track.last_area = detection.area
            matched_track.matched_steps += 1
            matched_track.total_link_distance += score
            matched_track.largest_gap = max(matched_track.largest_gap, frame_gap)
            matched_track.points.append(
                {
                    "frame_number": frame.frame_number,
                    "file_name": frame.file_name,
                    "center": list(detection.center),
                    "bbox": list(detection.bbox),
                    "annotation_id": detection.annotation_id,
                    "area": detection.area,
                }
            )

        for detection_index, detection in enumerate(frame.detections):
            if detection_index in matched_detection_indices:
                continue
            active_tracks.append(
                ActiveTrack(
                    track_id=next_track_id,
                    last_center=detection.center,
                    last_frame_number=frame.frame_number,
                    last_area=detection.area,
                    velocity=(0.0, 0.0),
                    matched_steps=0,
                    total_link_distance=0.0,
                    largest_gap=0,
                    points=[
                        {
                            "frame_number": frame.frame_number,
                            "file_name": frame.file_name,
                            "center": list(detection.center),
                            "bbox": list(detection.bbox),
                            "annotation_id": detection.annotation_id,
                            "area": detection.area,
                        }
                    ],
                )
            )
            next_track_id += 1

        still_active: List[ActiveTrack] = []
        for track_index, track in enumerate(active_tracks):
            if track_index in matched_track_indices:
                still_active.append(track)
                continue
            if frame.frame_number - track.last_frame_number < max_gap:
                still_active.append(track)
                continue
            if len(track.points) >= min_track_length:
                completed_tracks.append(_finalize_track(track))
        active_tracks = still_active

    for track in active_tracks:
        if len(track.points) >= min_track_length:
            completed_tracks.append(_finalize_track(track))

    completed_tracks.sort(key=lambda track: (track["start_frame"], track["track_id"]))
    return completed_tracks


def build_trajectory_dataset(
    annotations_path: Path,
    frames_root: Path,
    output_path: Path,
    category_name: str = "drone",
    max_gap: int = 8,
    max_distance: float = 120.0,
    min_track_length: int = 2,
    eo_max_gap: Optional[int] = None,
    eo_max_distance: Optional[float] = None,
    ir_max_gap: Optional[int] = None,
    ir_max_distance: Optional[float] = None,
) -> Dict[str, int]:
    frames = build_frame_records(
        annotations_path=annotations_path,
        frames_root=frames_root,
        category_filter=[category_name],
    )
    grouped = group_frames_by_sequence(frames)

    sequences: List[Dict] = []
    total_tracks = 0
    total_points = 0
    sequence_counts_by_modality: Dict[str, int] = {}
    track_counts_by_modality: Dict[str, int] = {}
    point_counts_by_modality: Dict[str, int] = {}

    for sequence_key, sequence_frames in sorted(grouped.items()):
        modality = sequence_frames[0].modality
        modality_config = DEFAULT_MODALITY_CONFIG.get(
            modality,
            {"max_gap": max_gap, "max_distance": max_distance},
        )
        sequence_max_gap = int(modality_config["max_gap"])
        sequence_max_distance = float(modality_config["max_distance"])

        if modality == "EO":
            if eo_max_gap is not None:
                sequence_max_gap = eo_max_gap
            if eo_max_distance is not None:
                sequence_max_distance = eo_max_distance
        elif modality == "IR":
            if ir_max_gap is not None:
                sequence_max_gap = ir_max_gap
            if ir_max_distance is not None:
                sequence_max_distance = ir_max_distance

        tracks = _build_tracks_for_sequence(
            sequence_frames=sequence_frames,
            max_gap=sequence_max_gap,
            max_distance=sequence_max_distance,
            min_track_length=min_track_length,
        )
        total_tracks += len(tracks)
        total_points += sum(track["length"] for track in tracks)
        sequence_counts_by_modality[modality] = sequence_counts_by_modality.get(modality, 0) + 1
        track_counts_by_modality[modality] = track_counts_by_modality.get(modality, 0) + len(tracks)
        point_counts_by_modality[modality] = point_counts_by_modality.get(modality, 0) + sum(
            track["length"] for track in tracks
        )
        sequences.append(
            {
                "sequence_key": sequence_key,
                "date_name": sequence_frames[0].date_name,
                "modality": sequence_frames[0].modality,
                "experiment_name": sequence_frames[0].experiment_name,
                "frame_count": len(sequence_frames),
                "annotated_frame_count": sum(1 for frame in sequence_frames if frame.detections),
                "unannotated_frame_count": sum(1 for frame in sequence_frames if not frame.detections),
                "tracking_config": {
                    "max_gap": sequence_max_gap,
                    "max_distance": sequence_max_distance,
                },
                "tracks": tracks,
            }
        )

    payload = {
        "source_annotations": str(annotations_path),
        "frames_root": str(frames_root),
        "category_name": category_name,
        "max_gap": max_gap,
        "max_distance": max_distance,
        "min_track_length": min_track_length,
        "modality_tracking_config": {
            "EO": {
                "max_gap": eo_max_gap if eo_max_gap is not None else DEFAULT_MODALITY_CONFIG["EO"]["max_gap"],
                "max_distance": (
                    eo_max_distance
                    if eo_max_distance is not None
                    else DEFAULT_MODALITY_CONFIG["EO"]["max_distance"]
                ),
            },
            "IR": {
                "max_gap": ir_max_gap if ir_max_gap is not None else DEFAULT_MODALITY_CONFIG["IR"]["max_gap"],
                "max_distance": (
                    ir_max_distance
                    if ir_max_distance is not None
                    else DEFAULT_MODALITY_CONFIG["IR"]["max_distance"]
                ),
            },
        },
        "sequence_count": len(sequences),
        "track_count": total_tracks,
        "track_point_count": total_points,
        "sequence_count_by_modality": sequence_counts_by_modality,
        "track_count_by_modality": track_counts_by_modality,
        "track_point_count_by_modality": point_counts_by_modality,
        "sequences": sequences,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(payload, file, separators=(",", ":"))

    return {
        "frame_count": len(frames),
        "sequence_count": len(sequences),
        "track_count": total_tracks,
        "track_point_count": total_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build approximate per-sequence trajectories from reconciled COCO detections."
    )
    parser.add_argument(
        "--annotations",
        default="/Users/kappasutra/MT7/annotations/instances_reconciled.json",
        help="Path to the reconciled COCO annotations JSON file.",
    )
    parser.add_argument(
        "--frames-root",
        default="/Users/kappasutra/MT7/FRAMED-FINAL-INSALLAH",
        help="Root directory containing the actual image frames.",
    )
    parser.add_argument(
        "--output",
        default="/Users/kappasutra/MT7/annotations/drone_trajectories.json",
        help="Path to write the derived trajectory dataset.",
    )
    parser.add_argument(
        "--category",
        default="drone",
        help="Category name to convert into trajectories.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=8,
        help="Maximum frame gap allowed when linking detections.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=120.0,
        help="Maximum center distance allowed when linking detections.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=2,
        help="Minimum number of detections required to keep a track.",
    )
    parser.add_argument(
        "--eo-max-gap",
        type=int,
        default=None,
        help="Optional EO-only maximum frame gap override.",
    )
    parser.add_argument(
        "--eo-max-distance",
        type=float,
        default=None,
        help="Optional EO-only maximum center distance override.",
    )
    parser.add_argument(
        "--ir-max-gap",
        type=int,
        default=None,
        help="Optional IR-only maximum frame gap override.",
    )
    parser.add_argument(
        "--ir-max-distance",
        type=float,
        default=None,
        help="Optional IR-only maximum center distance override.",
    )
    args = parser.parse_args()

    summary = build_trajectory_dataset(
        annotations_path=Path(args.annotations),
        frames_root=Path(args.frames_root),
        output_path=Path(args.output),
        category_name=args.category,
        max_gap=args.max_gap,
        max_distance=args.max_distance,
        min_track_length=args.min_track_length,
        eo_max_gap=args.eo_max_gap,
        eo_max_distance=args.eo_max_distance,
        ir_max_gap=args.ir_max_gap,
        ir_max_distance=args.ir_max_distance,
    )

    print("Trajectory build summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
