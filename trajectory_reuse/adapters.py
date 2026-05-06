from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def bbox_xyxy_to_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Convert an `(x1, y1, x2, y2)` box into its center point."""

    if len(bbox) < 4:
        raise ValueError("bbox must contain at least four values")

    x1, y1, x2, y2 = bbox[:4]
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def bbox_xywh_to_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Convert a COCO-style `(x, y, width, height)` box into its center point."""

    if len(bbox) < 4:
        raise ValueError("bbox must contain at least four values")

    x, y, width, height = bbox[:4]
    return (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0)


def bbox_to_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Convert a COCO-style `(x, y, width, height)` box into its center point."""

    return bbox_xywh_to_center(bbox)


def centers_from_bboxes(bboxes: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    """Convert a COCO bbox stream into trajectory points."""

    return [bbox_to_center(bbox) for bbox in bboxes]
