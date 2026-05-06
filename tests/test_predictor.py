import pytest
from trajectory_reuse import ReusableTrajectoryPredictor


@pytest.fixture()
def predictor():
    return ReusableTrajectoryPredictor(prediction_horizon=2.0, dt=0.2)


def test_straight_trajectory_predicts(predictor):
    history = [
        (10.0, 12.0),
        (10.7, 12.3),
        (11.5, 12.8),
        (12.4, 13.6),
        (13.2, 14.7),
        (13.9, 15.9),
    ]
    result = predictor.predict(track_id=1, trajectory=history, object_type="drone")
    assert result is not None
    assert len(result.predicted_points) > 0


def test_straight_trajectory_current_position(predictor):
    history = [
        (10.0, 12.0),
        (10.7, 12.3),
        (11.5, 12.8),
        (12.4, 13.6),
        (13.2, 14.7),
        (13.9, 15.9),
    ]
    result = predictor.predict(track_id=1, trajectory=history, object_type="drone")
    assert result is not None
    assert result.current_position == history[-1]


def test_hover_trajectory_predicts(predictor):
    history = [(5.0, 5.0)] * 6
    result = predictor.predict(track_id=2, trajectory=history, object_type="drone")
    assert result is not None
    assert result.current_position == (5.0, 5.0)
    assert result.intention == "hover"


def test_history_shorter_than_min_returns_none(predictor):
    # predictor has min_history=4 by default in __init__; fixture uses default=4
    result = predictor.predict(track_id=3, trajectory=[(0.0, 0.0), (1.0, 1.0)], object_type="drone")
    assert result is None


def test_outlier_history_still_predicts(predictor):
    history = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (50.0, 50.0),
        (3.0, 0.0),
        (4.0, 0.0),
        (5.0, 0.0),
    ]
    result = predictor.predict(track_id=4, trajectory=history, object_type="drone")
    assert result is not None
    assert result.diagnostics["outliers_removed"] >= 1.0


def test_performance_metrics_track_predictions(predictor):
    history = [(float(i), float(i)) for i in range(8)]
    predictor.predict(track_id=5, trajectory=history)
    metrics = predictor.get_performance_metrics()
    assert metrics["predictions_generated"] >= 1.0


def test_cleanup_old_tracks(predictor):
    history = [(float(i), 0.0) for i in range(8)]
    predictor.predict(track_id=10, trajectory=history)
    predictor.predict(track_id=11, trajectory=history)
    predictor.cleanup_old_tracks([10])
    assert 11 not in predictor.track_cache
    assert 10 in predictor.track_cache


def test_invalid_dt_raises():
    with pytest.raises(ValueError):
        ReusableTrajectoryPredictor(dt=0.0)


def test_invalid_min_history_raises():
    with pytest.raises(ValueError):
        ReusableTrajectoryPredictor(min_history=1)
