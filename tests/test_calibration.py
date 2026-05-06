"""
Tests for IR coordinate calibration.

Covers:
  - estimate_bias() computes the correct mean residual from synthetic data
  - zero offset leaves DetectionRecord centres unchanged
  - known offset is correctly applied to DetectionRecord centres
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# calibrate_ir lives at the project root, not inside a package
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibrate_ir import estimate_bias

from trajectory_reuse.adapters import bbox_xywh_to_center
from trajectory_reuse.dataset_loader import DetectionRecord


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_sample(predicted: list, target: list) -> dict:
    return {"predicted_future": predicted, "target_future": target, "sequence_key": "d/IR/exp"}


def _apply_offset(bbox, offset_x: float, offset_y: float) -> tuple:
    """Replicate the logic in build_frame_records: raw_center + (dx, dy)."""
    raw_cx, raw_cy = bbox_xywh_to_center(bbox)
    return (raw_cx + offset_x, raw_cy + offset_y)


# ──────────────────────────────────────────────────────────────────────────────
# estimate_bias tests
# ──────────────────────────────────────────────────────────────────────────────

def test_estimate_bias_known_offset():
    """Mean residual should equal the known systematic shift."""
    samples = [
        _make_sample(
            predicted=[[110.0, 205.0], [115.0, 210.0]],
            target=   [[100.0, 200.0], [105.0, 205.0]],
        ),
        _make_sample(
            predicted=[[210.0, 305.0]],
            target=   [[200.0, 300.0]],
        ),
    ]
    mean_x, mean_y, std_x, std_y, n = estimate_bias(samples)
    assert n == 3
    assert abs(mean_x - 10.0) < 1e-9
    assert abs(mean_y - 5.0) < 1e-9


def test_estimate_bias_zero_residuals():
    """Perfect predictions → zero bias."""
    samples = [
        _make_sample([[10.0, 20.0], [11.0, 21.0]], [[10.0, 20.0], [11.0, 21.0]]),
    ]
    mean_x, mean_y, std_x, std_y, n = estimate_bias(samples)
    assert n == 2
    assert mean_x == 0.0
    assert mean_y == 0.0
    assert std_x == 0.0
    assert std_y == 0.0


def test_estimate_bias_empty_samples():
    """No samples → all zeros, n == 0."""
    mean_x, mean_y, std_x, std_y, n = estimate_bias([])
    assert n == 0
    assert mean_x == 0.0
    assert mean_y == 0.0


def test_estimate_bias_mixed_signs():
    """Residuals that cancel should produce near-zero mean."""
    samples = [
        _make_sample([[105.0, 100.0]], [[100.0, 100.0]]),   # +5
        _make_sample([[95.0, 100.0]],  [[100.0, 100.0]]),   # -5
    ]
    mean_x, mean_y, std_x, std_y, n = estimate_bias(samples)
    assert n == 2
    assert abs(mean_x) < 1e-9


def test_estimate_bias_std_nonzero():
    """Standard deviation should be non-zero when residuals vary."""
    samples = [
        _make_sample([[110.0, 100.0]], [[100.0, 100.0]]),   # +10
        _make_sample([[120.0, 100.0]], [[100.0, 100.0]]),   # +20
    ]
    _, _, std_x, _, _ = estimate_bias(samples)
    assert std_x > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Centre offset application tests
# ──────────────────────────────────────────────────────────────────────────────

def test_zero_offset_leaves_center_unchanged():
    bbox = (100.0, 200.0, 40.0, 20.0)
    raw_cx, raw_cy = bbox_xywh_to_center(bbox)
    corrected = _apply_offset(bbox, 0.0, 0.0)
    assert corrected == (raw_cx, raw_cy)


def test_known_offset_shifts_center():
    """Center should shift by exactly (dx, dy)."""
    bbox = (100.0, 200.0, 40.0, 20.0)
    dx, dy = -12.5, 3.0
    raw_cx, raw_cy = bbox_xywh_to_center(bbox)
    corrected = _apply_offset(bbox, dx, dy)
    assert abs(corrected[0] - (raw_cx + dx)) < 1e-9
    assert abs(corrected[1] - (raw_cy + dy)) < 1e-9


def test_offset_does_not_modify_bbox():
    """The correction must only touch the centre, not the bbox dimensions."""
    bbox = (50.0, 60.0, 30.0, 15.0)
    # bbox_xywh_to_center is pure: it takes bbox and returns a new tuple
    raw_cx, raw_cy = bbox_xywh_to_center(bbox)
    # Applying an offset does not mutate the bbox in any way
    corrected = _apply_offset(bbox, 99.0, -99.0)
    assert bbox == (50.0, 60.0, 30.0, 15.0)   # unchanged
    assert corrected != (raw_cx, raw_cy)        # but centre did change


def test_offset_applied_via_build_frame_records(tmp_path):
    """
    End-to-end: offsets passed to build_frame_records() must shift IR centres
    but leave EO centres unchanged.
    """
    import json
    from trajectory_reuse.dataset_loader import build_frame_records

    bbox = [10.0, 20.0, 8.0, 4.0]          # center = (14.0, 22.0)
    dx, dy = -5.0, 3.0

    coco = {
        "images": [
            {"id": 1, "file_name": "2024/IR/exp/exp_frame_001.jpg", "width": 640, "height": 512},
            {"id": 2, "file_name": "2024/EO/exp/exp_frame_001.jpg", "width": 1920, "height": 1080},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": bbox, "area": 32, "iscrowd": 0, "segmentation": [], "attributes": {}},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": bbox, "area": 32, "iscrowd": 0, "segmentation": [], "attributes": {}},
        ],
        "categories": [{"id": 1, "name": "drone"}],
    }
    ann_path = tmp_path / "coco.json"
    ann_path.write_text(json.dumps(coco))

    frames = build_frame_records(
        annotations_path=ann_path,
        frames_root=tmp_path,
        modality_offsets={"IR": (dx, dy)},
    )

    ir_frames = [f for f in frames if f.modality == "IR"]
    eo_frames = [f for f in frames if f.modality == "EO"]

    assert len(ir_frames) == 1 and len(eo_frames) == 1

    raw_cx, raw_cy = bbox_xywh_to_center(tuple(bbox))

    ir_center = ir_frames[0].detections[0].center
    assert abs(ir_center[0] - (raw_cx + dx)) < 1e-9
    assert abs(ir_center[1] - (raw_cy + dy)) < 1e-9

    eo_center = eo_frames[0].detections[0].center
    assert abs(eo_center[0] - raw_cx) < 1e-9
    assert abs(eo_center[1] - raw_cy) < 1e-9
