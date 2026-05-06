from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

from .predictor import ReusableTrajectoryPredictor

logger = logging.getLogger(__name__)


def _distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return math.hypot(float(point_b[0]) - float(point_a[0]), float(point_b[1]) - float(point_a[1]))


def _window_max_gap(points: Sequence[Dict]) -> int:
    if len(points) < 2:
        return 0
    return max(
        int(points[index]["frame_number"]) - int(points[index - 1]["frame_number"])
        for index in range(1, len(points))
    )


def _evaluate_sample(
    predictor: ReusableTrajectoryPredictor,
    sequence_key: str,
    track: Dict,
    start_index: int,
    history_size: int,
    forecast_steps: int,
    max_window_gap: int,
) -> Optional[Dict]:
    points = track["points"]
    history_points = points[max(0, start_index - history_size):start_index]
    future_points = points[start_index:start_index + forecast_steps]
    if len(history_points) < predictor.min_history or len(future_points) != forecast_steps:
        return None

    combined_points = history_points + future_points
    if _window_max_gap(combined_points) > max_window_gap:
        return None

    history_centers = [tuple(point["center"]) for point in history_points]
    future_centers = [tuple(point["center"]) for point in future_points]

    prediction = predictor.predict(
        track_id=int(track["track_id"]),
        trajectory=history_centers,
        object_type="drone",
    )
    if prediction is None or len(prediction.predicted_points) < forecast_steps:
        return None

    predicted_centers = [
        tuple(prediction.predicted_points[index].position) for index in range(forecast_steps)
    ]
    errors = [_distance(predicted, actual) for predicted, actual in zip(predicted_centers, future_centers)]
    if not errors:
        return None

    return {
        "sequence_key": sequence_key,
        "track_id": int(track["track_id"]),
        "history_start_frame": int(history_points[0]["frame_number"]),
        "history_end_frame": int(history_points[-1]["frame_number"]),
        "future_end_frame": int(future_points[-1]["frame_number"]),
        "history_length": len(history_points),
        "forecast_steps": forecast_steps,
        "max_window_gap": _window_max_gap(combined_points),
        "current_file_name": str(history_points[-1]["file_name"]),
        "current_bbox": [float(value) for value in history_points[-1]["bbox"]],
        "future_file_names": [str(point["file_name"]) for point in future_points],
        "ade": sum(errors) / len(errors),
        "fde": errors[-1],
        "confidence": float(prediction.confidence),
        "intention": prediction.intention,
        "history": [list(point) for point in history_centers],
        "target_future": [list(point) for point in future_centers],
        "target_future_bboxes": [[float(value) for value in point["bbox"]] for point in future_points],
        "predicted_future": [list(point) for point in predicted_centers],
        "diagnostics": prediction.diagnostics,
    }


def evaluate_trajectory_dataset(
    trajectories_path: Path,
    output_path: Path,
    dt: float = 1.0,
    prediction_horizon: float = 3.0,
    min_history: int = 4,
    max_history: int = 12,
    history_size: int = 12,
    max_window_gap: int = 3,
) -> Dict[str, float]:
    with trajectories_path.open() as file:
        data = json.load(file)

    predictor = ReusableTrajectoryPredictor(
        prediction_horizon=prediction_horizon,
        dt=dt,
        min_history=min_history,
        max_history=max_history,
    )
    forecast_steps = max(1, int(prediction_horizon / dt))

    samples: List[Dict] = []
    all_sequences = data.get("sequences", [])
    total_sequences = len(all_sequences)
    for seq_idx, sequence in enumerate(all_sequences, 1):
        sequence_key = str(sequence["sequence_key"])
        for track in sequence.get("tracks", []):
            point_count = len(track.get("points", []))
            for start_index in range(min_history, point_count - forecast_steps + 1):
                sample = _evaluate_sample(
                    predictor=predictor,
                    sequence_key=sequence_key,
                    track=track,
                    start_index=start_index,
                    history_size=history_size,
                    forecast_steps=forecast_steps,
                    max_window_gap=max_window_gap,
                )
                if sample is not None:
                    samples.append(sample)
        if total_sequences > 0 and (
            seq_idx == total_sequences or seq_idx % max(1, total_sequences // 10) == 0
        ):
            logger.info(
                "Evaluated %d/%d sequences (%.0f%%)",
                seq_idx, total_sequences, seq_idx / total_sequences * 100,
            )

    ade_values = [sample["ade"] for sample in samples]
    fde_values = [sample["fde"] for sample in samples]
    confidence_values = [sample["confidence"] for sample in samples]
    summary = {
        "trajectory_dataset": str(trajectories_path),
        "frames_root": str(data.get("frames_root", "")),
        "sample_count": len(samples),
        "dt": dt,
        "prediction_horizon": prediction_horizon,
        "forecast_steps": forecast_steps,
        "min_history": min_history,
        "max_history": max_history,
        "history_size": history_size,
        "max_window_gap": max_window_gap,
        "mean_ade": mean(ade_values) if ade_values else 0.0,
        "median_ade": median(ade_values) if ade_values else 0.0,
        "mean_fde": mean(fde_values) if fde_values else 0.0,
        "median_fde": median(fde_values) if fde_values else 0.0,
        "mean_confidence": mean(confidence_values) if confidence_values else 0.0,
    }

    payload = {
        "summary": summary,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(payload, file, separators=(",", ":"))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trajectory predictor on derived drone tracks and export samples."
    )
    parser.add_argument(
        "--trajectories",
        required=True,
        help="Path to the derived trajectory dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the predictor evaluation JSON file.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step used for the predictor. Defaults to one detection step.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=float,
        default=3.0,
        help="Prediction horizon used during evaluation.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=4,
        help="Minimum history length required by the predictor.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=12,
        help="Maximum history length passed to the predictor.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=12,
        help="Maximum history window used for each evaluation sample.",
    )
    parser.add_argument(
        "--max-window-gap",
        type=int,
        default=3,
        help="Maximum allowed frame-number gap inside an evaluation window.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        summary = evaluate_trajectory_dataset(
            trajectories_path=Path(args.trajectories),
            output_path=Path(args.output),
            dt=args.dt,
            prediction_horizon=args.prediction_horizon,
            min_history=args.min_history,
            max_history=args.max_history,
            history_size=args.history_size,
            max_window_gap=args.max_window_gap,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None

    logger.info("Predictor evaluation summary")
    for key, value in summary.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
