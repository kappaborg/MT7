from trajectory_reuse.build_trajectories import _build_tracks_for_sequence
from trajectory_reuse.dataset_loader import DetectionRecord, FrameRecord


def _make_frame(frame_number: int, center, bbox=(0.0, 0.0, 10.0, 10.0)) -> FrameRecord:
    detection = DetectionRecord(
        annotation_id=frame_number,
        category_id=1,
        category_name="drone",
        bbox=bbox,
        center=center,
        area=bbox[2] * bbox[3],
        attributes={},
    )
    return FrameRecord(
        image_id=frame_number,
        file_name=f"d/EO/exp/frame_{frame_number}.jpg",
        full_path=f"/fake/d/EO/exp/frame_{frame_number}.jpg",
        width=1920,
        height=1080,
        date_name="d",
        modality="EO",
        experiment_name="exp",
        sequence_key="d/EO/exp",
        frame_number=frame_number,
        detections=[detection],
    )


def test_nearby_detections_linked():
    frames = [
        _make_frame(1, (100.0, 100.0)),
        _make_frame(2, (105.0, 102.0)),
        _make_frame(3, (110.0, 104.0)),
    ]
    tracks = _build_tracks_for_sequence(frames, max_gap=5, max_distance=100.0, min_track_length=2)
    assert len(tracks) == 1
    assert tracks[0]["length"] == 3


def _make_frame_multi(frame_number: int, centers) -> FrameRecord:
    """Create a frame with multiple detections (forces the greedy matching path)."""
    detections = [
        DetectionRecord(
            annotation_id=frame_number * 100 + i,
            category_id=1,
            category_name="drone",
            bbox=(0.0, 0.0, 10.0, 10.0),
            center=center,
            area=100.0,
            attributes={},
        )
        for i, center in enumerate(centers)
    ]
    frame = _make_frame(frame_number, centers[0])
    frame.detections[:] = detections
    return frame


def test_far_apart_detections_not_linked():
    # In the greedy matcher (triggered when a frame has >1 detection), two detections that
    # exceed max_distance must not be linked into the same track.
    # Frame 1 has 2 detections to force the greedy path; frame 2 has one detection far from both.
    frames = [
        _make_frame_multi(1, [(100.0, 100.0), (110.0, 100.0)]),
        _make_frame(2, (900.0, 900.0)),
    ]
    tracks = _build_tracks_for_sequence(frames, max_gap=5, max_distance=50.0, min_track_length=1)
    # The detection at (900, 900) must be a separate track — not linked to either from frame 1
    assert all(
        not any(p["center"] == [900.0, 900.0] and p["frame_number"] == 2
                for p in track["points"]
                if any(pp["frame_number"] == 1 for pp in track["points"]))
        for track in tracks
    ), "Distant detection should not be linked to near track"
    track_lengths = [t["length"] for t in tracks]
    # The far detection starts its own 1-point track; the two near ones each have length 1 too
    assert sorted(track_lengths) == [1, 1, 1]


def test_gap_too_large_splits_track():
    frames = [
        _make_frame(1, (100.0, 100.0)),
        _make_frame(2, (101.0, 100.0)),
        # frame 3-10 missing — gap of 9 > max_gap=5
        _make_frame(11, (102.0, 100.0)),
        _make_frame(12, (103.0, 100.0)),
    ]
    tracks = _build_tracks_for_sequence(frames, max_gap=5, max_distance=200.0, min_track_length=2)
    assert len(tracks) == 2


def test_min_track_length_filters_short_tracks():
    frames = [_make_frame(1, (100.0, 100.0))]
    tracks = _build_tracks_for_sequence(frames, max_gap=5, max_distance=100.0, min_track_length=2)
    assert tracks == []


def test_empty_sequence_returns_no_tracks():
    tracks = _build_tracks_for_sequence([], max_gap=5, max_distance=100.0, min_track_length=2)
    assert tracks == []
