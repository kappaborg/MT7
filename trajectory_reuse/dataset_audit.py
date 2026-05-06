from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict:
    with path.open() as file:
        return json.load(file)


def _sequence_name(file_name: str) -> str:
    parts = Path(file_name).parts
    if len(parts) >= 3:
        return "/".join(parts[:3])
    if parts:
        return parts[0]
    return "<unknown>"


def audit_dataset(annotations_path: Path, frames_root: Path) -> Dict:
    data = _load_json(annotations_path)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = {item.get("id"): item.get("name", str(item.get("id"))) for item in data.get("categories", [])}

    annotation_counts = Counter(annotation.get("image_id") for annotation in annotations)
    category_counts = Counter(categories.get(annotation.get("category_id"), str(annotation.get("category_id"))) for annotation in annotations)

    missing_images: List[str] = []
    existing_images = 0
    sequence_image_counts = Counter()
    sequence_annotation_counts = Counter()

    image_by_id = {}
    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name", "")
        image_by_id[image_id] = image
        sequence = _sequence_name(file_name)
        sequence_image_counts[sequence] += 1

        if (frames_root / file_name).exists():
            existing_images += 1
        else:
            missing_images.append(file_name)

    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            continue
        sequence = _sequence_name(image.get("file_name", ""))
        sequence_annotation_counts[sequence] += 1

    frame_annotation_distribution = Counter(annotation_counts.values())
    sample_annotation = annotations[0] if annotations else {}
    sample_keys = sorted(sample_annotation.keys())
    has_track_id = any(key in sample_keys for key in ("track_id", "trackId", "instance_id", "object_id"))

    return {
        "images_total": len(images),
        "images_existing": existing_images,
        "images_missing": len(missing_images),
        "annotations_total": len(annotations),
        "frames_with_annotations": len(annotation_counts),
        "max_annotations_in_frame": max(annotation_counts.values()) if annotation_counts else 0,
        "categories": dict(category_counts),
        "has_track_id": has_track_id,
        "annotation_keys": sample_keys,
        "sample_missing_images": missing_images[:10],
        "annotation_density": frame_annotation_distribution.most_common(10),
        "top_sequences_by_images": sequence_image_counts.most_common(10),
        "top_sequences_by_annotations": sequence_annotation_counts.most_common(10),
    }


def _format_pairs(pairs: Iterable[Tuple[object, object]]) -> str:
    return ", ".join(f"{left}: {right}" for left, right in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a COCO-style frame and annotation dataset.")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to the COCO annotations JSON file.",
    )
    parser.add_argument(
        "--frames-root",
        required=True,
        help="Root directory that contains the image files referenced by the annotations.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        summary = audit_dataset(
            annotations_path=Path(args.annotations),
            frames_root=Path(args.frames_root),
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None

    logger.info("Dataset audit summary")
    logger.info("images_total: %s", summary['images_total'])
    logger.info("images_existing: %s", summary['images_existing'])
    logger.info("images_missing: %s", summary['images_missing'])
    logger.info("annotations_total: %s", summary['annotations_total'])
    logger.info("frames_with_annotations: %s", summary['frames_with_annotations'])
    logger.info("max_annotations_in_frame: %s", summary['max_annotations_in_frame'])
    logger.info("categories: %s", summary['categories'])
    logger.info("has_track_id: %s", summary['has_track_id'])
    logger.info("annotation_keys: %s", summary['annotation_keys'])
    logger.info("annotation_density_top10: %s", _format_pairs(summary['annotation_density']))
    logger.info("top_sequences_by_images: %s", _format_pairs(summary['top_sequences_by_images']))
    logger.info("top_sequences_by_annotations: %s", _format_pairs(summary['top_sequences_by_annotations']))
    logger.info("sample_missing_images: %s", summary['sample_missing_images'])


if __name__ == "__main__":
    main()
