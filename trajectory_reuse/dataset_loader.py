from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .adapters import bbox_xywh_to_center


STANDARD_FRAME_NAME_RE = re.compile(
    r"^(?P<experiment>[^/]+)_frame_(?P<frame_number>\d+)\.[^.]+$"
)
TRAILING_NUMBER_RE = re.compile(r"(?P<frame_number>\d+)$")


@dataclass
class DetectionRecord:
    annotation_id: int
    category_id: int
    category_name: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    area: float
    attributes: Dict


@dataclass
class FrameRecord:
    image_id: int
    file_name: str
    full_path: str
    width: int
    height: int
    date_name: str
    modality: str
    experiment_name: str
    sequence_key: str
    frame_number: int
    detections: List[DetectionRecord] = field(default_factory=list)


def parse_frame_name(file_name: str) -> Optional[Dict[str, object]]:
    parts = Path(file_name).parts
    if len(parts) != 4:
        return None
    date_name, modality, experiment_name, basename = parts
    if modality not in {"EO", "IR"}:
        return None

    frame_number: Optional[int] = None
    standard_match = STANDARD_FRAME_NAME_RE.match(basename)
    if standard_match:
        frame_number = int(standard_match.group("frame_number"))
    else:
        trailing_match = TRAILING_NUMBER_RE.search(Path(basename).stem)
        if trailing_match:
            frame_number = int(trailing_match.group("frame_number"))

    if frame_number is None:
        return None
    return {
        "date_name": date_name,
        "modality": modality,
        "experiment_name": experiment_name,
        "frame_number": frame_number,
    }


def load_coco_dataset(annotations_path: Path) -> Dict:
    with annotations_path.open() as file:
        return json.load(file)


def build_frame_records(
    annotations_path: Path,
    frames_root: Path,
    category_filter: Optional[Sequence[str]] = None,
) -> List[FrameRecord]:
    data = load_coco_dataset(annotations_path)
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in data.get("categories", [])
    }
    allowed_categories = set(category_filter or [])

    annotations_by_image_id: Dict[int, List[Dict]] = {}
    for annotation in data.get("annotations", []):
        category_name = categories.get(int(annotation["category_id"]), str(annotation["category_id"]))
        if allowed_categories and category_name not in allowed_categories:
            continue
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    frames: List[FrameRecord] = []
    for image in data.get("images", []):
        parsed = parse_frame_name(str(image["file_name"]))
        if parsed is None:
            continue

        image_id = int(image["id"])
        detections: List[DetectionRecord] = []
        for annotation in annotations_by_image_id.get(image_id, []):
            bbox = tuple(float(value) for value in annotation["bbox"][:4])
            detections.append(
                DetectionRecord(
                    annotation_id=int(annotation["id"]),
                    category_id=int(annotation["category_id"]),
                    category_name=categories.get(int(annotation["category_id"]), str(annotation["category_id"])),
                    bbox=bbox,  # type: ignore[arg-type]
                    center=bbox_xywh_to_center(bbox),
                    area=float(annotation.get("area", bbox[2] * bbox[3])),
                    attributes=dict(annotation.get("attributes", {})),
                )
            )

        sequence_key = (
            f"{parsed['date_name']}/{parsed['modality']}/{parsed['experiment_name']}"
        )
        frames.append(
            FrameRecord(
                image_id=image_id,
                file_name=str(image["file_name"]),
                full_path=str((frames_root / str(image["file_name"])).resolve()),
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                date_name=str(parsed["date_name"]),
                modality=str(parsed["modality"]),
                experiment_name=str(parsed["experiment_name"]),
                sequence_key=sequence_key,
                frame_number=int(parsed["frame_number"]),
                detections=detections,
            )
        )

    frames.sort(key=lambda frame: (frame.sequence_key, frame.frame_number, frame.file_name))
    return frames


def group_frames_by_sequence(frames: Iterable[FrameRecord]) -> Dict[str, List[FrameRecord]]:
    grouped: Dict[str, List[FrameRecord]] = {}
    for frame in frames:
        grouped.setdefault(frame.sequence_key, []).append(frame)
    for sequence_frames in grouped.values():
        sequence_frames.sort(key=lambda frame: frame.frame_number)
    return grouped
