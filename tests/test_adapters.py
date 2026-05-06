from trajectory_reuse.adapters import (
    bbox_to_center,
    bbox_xywh_to_center,
    bbox_xyxy_to_center,
    centers_from_bboxes,
)


def test_bbox_xywh_known_value():
    cx, cy = bbox_xywh_to_center((2218.08, 1490.48, 86.29, 41.8))
    assert cx == 2261.225
    assert cy == 1511.38


def test_bbox_xywh_unit_box():
    cx, cy = bbox_xywh_to_center((0.0, 0.0, 2.0, 2.0))
    assert cx == 1.0
    assert cy == 1.0


def test_bbox_xyxy_basic():
    cx, cy = bbox_xyxy_to_center((0.0, 0.0, 4.0, 6.0))
    assert cx == 2.0
    assert cy == 3.0


def test_bbox_to_center_coco_format():
    # bbox_to_center is an alias for xywh
    cx, cy = bbox_to_center((10.0, 20.0, 8.0, 4.0))
    assert cx == 14.0
    assert cy == 22.0


def test_bbox_to_center_zero_size_does_not_crash():
    # Zero-size bbox should return the corner point, not crash
    cx, cy = bbox_to_center((5.0, 7.0, 0.0, 0.0))
    assert cx == 5.0
    assert cy == 7.0


def test_centers_from_bboxes_batch():
    bboxes = [(0.0, 0.0, 4.0, 2.0), (10.0, 10.0, 6.0, 4.0)]
    centers = centers_from_bboxes(bboxes)
    assert centers[0] == (2.0, 1.0)
    assert centers[1] == (13.0, 12.0)


def test_centers_from_bboxes_empty():
    assert centers_from_bboxes([]) == []
