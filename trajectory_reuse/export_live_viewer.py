from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .dataset_loader import parse_frame_name

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "viewer_template.html"


def _load_json(path: Path) -> Dict:
    with path.open() as file:
        return json.load(file)


def _image_src(frames_root: Path, file_name: str, frames_url_prefix: Optional[str] = None) -> str:
    if frames_url_prefix is not None:
        return f"{frames_url_prefix.rstrip('/')}/{file_name}"
    image_path = frames_root / file_name
    return image_path.resolve().as_uri()


def export_live_viewer(
    reconciled_annotations_path: Path,
    trajectories_path: Path,
    frames_root: Path,
    output_path: Path,
    evaluation_path: Optional[Path] = None,
    frames_url_prefix: Optional[str] = None,
) -> Dict[str, int]:
    reconciled = _load_json(reconciled_annotations_path)
    trajectories = _load_json(trajectories_path)
    evaluation = _load_json(evaluation_path) if evaluation_path and evaluation_path.exists() else {}

    sequences_by_key: Dict[str, Dict] = {}
    all_images = reconciled.get("images", [])
    total_images = len(all_images)
    for img_idx, image in enumerate(all_images, 1):
        file_name = str(image["file_name"])
        parsed = parse_frame_name(file_name)
        if parsed is None:
            continue
        sequence_key = f"{parsed['date_name']}/{parsed['modality']}/{parsed['experiment_name']}"
        sequence = sequences_by_key.setdefault(
            sequence_key,
            {
                "sequence_key": sequence_key,
                "date_name": str(parsed["date_name"]),
                "modality": str(parsed["modality"]),
                "experiment_name": str(parsed["experiment_name"]),
                "frames": [],
                "tracks": [],
            },
        )
        sequence["frames"].append(
            {
                "frame_number": int(parsed["frame_number"]),
                "file_name": file_name,
                "image_src": _image_src(frames_root, file_name, frames_url_prefix),
            }
        )
        if total_images > 0 and (
            img_idx == total_images or img_idx % max(1, total_images // 10) == 0
        ):
            logger.info(
                "Indexed %d/%d images (%.0f%%)",
                img_idx, total_images, img_idx / total_images * 100,
            )

    for sequence in sequences_by_key.values():
        sequence["frames"].sort(key=lambda item: (item["frame_number"], item["file_name"]))

    all_traj_sequences = trajectories.get("sequences", [])
    total_traj = len(all_traj_sequences)
    for traj_idx, sequence in enumerate(all_traj_sequences, 1):
        sequence_key = str(sequence["sequence_key"])
        if sequence_key not in sequences_by_key:
            continue
        tracks: List[Dict] = []
        for track in sequence.get("tracks", []):
            tracks.append(
                {
                    "track_id": int(track["track_id"]),
                    "start_frame": int(track["start_frame"]),
                    "end_frame": int(track["end_frame"]),
                    "length": int(track["length"]),
                    "diagnostics": dict(track.get("diagnostics", {})),
                    "points": [
                        {
                            "frame_number": int(point["frame_number"]),
                            "file_name": str(point["file_name"]),
                            "center": [float(point["center"][0]), float(point["center"][1])],
                            "bbox": [
                                float(point["bbox"][0]),
                                float(point["bbox"][1]),
                                float(point["bbox"][2]),
                                float(point["bbox"][3]),
                            ] if "bbox" in point else None,
                        }
                        for point in track.get("points", [])
                    ],
                }
            )
        sequences_by_key[sequence_key]["tracks"] = sorted(tracks, key=lambda item: item["track_id"])
        if "tracking_config" in sequence:
            sequences_by_key[sequence_key]["tracking_config"] = dict(sequence["tracking_config"])
        if total_traj > 0 and (
            traj_idx == total_traj or traj_idx % max(1, total_traj // 10) == 0
        ):
            logger.info(
                "Processed %d/%d trajectory sequences (%.0f%%)",
                traj_idx, total_traj, traj_idx / total_traj * 100,
            )

    evaluation_lookup: Dict[str, Dict] = {}
    for sample in evaluation.get("samples", []):
        key = f"{sample['sequence_key']}|{sample['track_id']}|{sample['history_end_frame']}"
        evaluation_lookup[key] = {
            "ade": float(sample["ade"]),
            "fde": float(sample["fde"]),
            "confidence": float(sample["confidence"]),
            "intention": str(sample["intention"]),
            "current_bbox": [float(v) for v in sample.get("current_bbox", [])],
            "target_future": [[float(p[0]), float(p[1])] for p in sample.get("target_future", [])],
            "target_future_bboxes": [
                [float(v) for v in bbox] for bbox in sample.get("target_future_bboxes", [])
            ],
            "predicted_future": [[float(p[0]), float(p[1])] for p in sample.get("predicted_future", [])],
        }

    sequences = sorted(sequences_by_key.values(), key=lambda item: item["sequence_key"])
    payload = {
        "summary": {
            "sequence_count": len(sequences),
            "frame_count": sum(len(sequence["frames"]) for sequence in sequences),
            "track_count": sum(len(sequence["tracks"]) for sequence in sequences),
            "evaluation_sample_count": len(evaluation_lookup),
        },
        "sequences": sequences,
        "evaluation_lookup": evaluation_lookup,
    }

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("%DATA_JSON%", json.dumps(payload))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return {
        "sequence_count": len(sequences),
        "frame_count": payload["summary"]["frame_count"],
        "track_count": payload["summary"]["track_count"],
        "evaluation_sample_count": payload["summary"]["evaluation_sample_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a live HTML trajectory viewer for sequence, track, and prediction inspection."
    )
    parser.add_argument(
        "--reconciled-annotations",
        required=True,
        help="Path to the reconciled COCO annotations file.",
    )
    parser.add_argument(
        "--trajectories",
        required=True,
        help="Path to the derived trajectory dataset.",
    )
    parser.add_argument(
        "--frames-root",
        required=True,
        help="Root directory containing real image frames.",
    )
    parser.add_argument(
        "--evaluation",
        default=None,
        help="Optional predictor evaluation JSON path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the live trajectory HTML viewer.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        summary = export_live_viewer(
            reconciled_annotations_path=Path(args.reconciled_annotations),
            trajectories_path=Path(args.trajectories),
            frames_root=Path(args.frames_root),
            output_path=Path(args.output),
            evaluation_path=Path(args.evaluation) if args.evaluation else None,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None

    logger.info("Live trajectory viewer summary")
    for key, value in summary.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
