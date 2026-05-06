from trajectory_reuse import ReusableTrajectoryPredictor


def main() -> None:
    predictor = ReusableTrajectoryPredictor(prediction_horizon=3.0, dt=0.2)

    drone_history = [
        (10.0, 12.0),
        (10.7, 12.3),
        (11.5, 12.8),
        (12.4, 13.6),
        (13.2, 14.7),
        (13.9, 15.9),
    ]

    prediction = predictor.predict(
        track_id=101,
        trajectory=drone_history,
        object_type="drone",
        context={"max_speed": 22.0},
    )

    if prediction is None:
        print("Prediction could not be generated. Check history length and dt.")
        return

    print(f"Track: {prediction.track_id}")
    print(f"Confidence: {prediction.confidence:.2f}")
    print(f"Intention: {prediction.intention}")
    print(f"Current position: {prediction.current_position}")
    print(f"Current velocity: {prediction.current_velocity}")
    print("Diagnostics:", prediction.diagnostics)
    for point in prediction.predicted_points[:5]:
        print(point.to_dict())


if __name__ == "__main__":
    main()
