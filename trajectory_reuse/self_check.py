from __future__ import annotations

from trajectory_reuse import bbox_xywh_to_center
from trajectory_reuse import ReusableTrajectoryPredictor


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_check() -> None:
    _assert(
        bbox_xywh_to_center((2218.08, 1490.48, 86.29, 41.8)) == (2261.225, 1511.38),
        "COCO bbox centers should use x/y plus half width/height",
    )

    predictor = ReusableTrajectoryPredictor(prediction_horizon=2.0, dt=0.2)

    straight_history = [
        (10.0, 12.0),
        (10.7, 12.3),
        (11.5, 12.8),
        (12.4, 13.6),
        (13.2, 14.7),
        (13.9, 15.9),
    ]
    straight_prediction = predictor.predict(
        track_id=101,
        trajectory=straight_history,
        object_type="drone",
        context={"max_speed": 22.0},
    )
    _assert(straight_prediction is not None, "straight trajectory should predict")
    _assert(
        len(straight_prediction.predicted_points) > 0,
        "straight trajectory should produce future points",
    )
    _assert(
        straight_prediction.current_position == straight_history[-1],
        "current position should match the latest observed point",
    )

    hover_history = [(5.0, 5.0)] * 6
    hover_prediction = predictor.predict(
        track_id=102,
        trajectory=hover_history,
        object_type="drone",
    )
    _assert(hover_prediction is not None, "hover trajectory should predict")
    _assert(
        hover_prediction.current_position == (5.0, 5.0),
        "hover trajectory should keep the latest position",
    )
    _assert(
        hover_prediction.intention == "hover",
        "hover trajectory should be classified as hover",
    )

    short_history_prediction = predictor.predict(
        track_id=103,
        trajectory=[(0.0, 0.0), (1.0, 1.0)],
        object_type="drone",
    )
    _assert(
        short_history_prediction is None,
        "trajectory shorter than min_history should not predict",
    )

    outlier_history = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (50.0, 50.0),
        (3.0, 0.0),
        (4.0, 0.0),
        (5.0, 0.0),
    ]
    outlier_prediction = predictor.predict(
        track_id=104,
        trajectory=outlier_history,
        object_type="drone",
    )
    _assert(outlier_prediction is not None, "outlier history should still predict")
    _assert(
        outlier_prediction.diagnostics["outliers_removed"] >= 1.0,
        "outlier history should report at least one removed outlier",
    )

    metrics = predictor.get_performance_metrics()
    _assert(
        metrics["predictions_generated"] >= 3.0,
        "performance metrics should track successful predictions",
    )


def main() -> None:
    run_self_check()
    print("Self-check passed.")


if __name__ == "__main__":
    main()
