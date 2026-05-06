#!/usr/bin/env python3
"""
Estimate the systematic IR coordinate bias from predictor evaluation results
and write the correction offset into trajectory_reuse/modality_config.json.

How it works
------------
The predictor is evaluated on both EO and IR sequences.  For IR sequences the
predictions are consistently shifted because the raw pixel centres inherit a
sensor registration offset (different FOV / mounting vs. the EO camera).

This script measures that offset by computing the mean residual:
    residual_x = predicted_future[step][0] - target_future[step][0]
    residual_y = predicted_future[step][1] - target_future[step][1]

across every forecasted step of every IR evaluation sample.  The correction
to apply to raw centres before trajectory building is:
    center_offset_x = -mean_bias_x
    center_offset_y = -mean_bias_y

Usage
-----
    # Estimate and write correction
    python3 calibrate_ir.py \\
        --evaluation       annotations/predictor_evaluation.json \\
        --modality-config  trajectory_reuse/modality_config.json

    # Inspect without writing
    python3 calibrate_ir.py \\
        --evaluation annotations/predictor_evaluation.json \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pure estimation logic (importable for tests)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_bias(
    samples: List[Dict],
) -> Tuple[float, float, float, float, int]:
    """
    Compute the mean and standard deviation of (predicted − target) per axis
    across every forecasted step of every sample in *samples*.

    Returns
    -------
    (mean_x, mean_y, std_x, std_y, n)
        n is the total number of per-step residual observations.
        If n == 0, all values are 0.0.
    """
    residuals_x: List[float] = []
    residuals_y: List[float] = []

    for sample in samples:
        predicted = sample.get("predicted_future", [])
        target = sample.get("target_future", [])
        for pred, tgt in zip(predicted, target):
            residuals_x.append(float(pred[0]) - float(tgt[0]))
            residuals_y.append(float(pred[1]) - float(tgt[1]))

    n = len(residuals_x)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    mean_x = sum(residuals_x) / n
    mean_y = sum(residuals_y) / n

    var_x = sum((r - mean_x) ** 2 for r in residuals_x) / n
    var_y = sum((r - mean_y) ** 2 for r in residuals_y) / n

    return mean_x, mean_y, math.sqrt(var_x), math.sqrt(var_y), n


def calibrate(
    evaluation_path: Path,
    modality_config_path: Path,
    dry_run: bool = False,
) -> Dict[str, float]:
    """
    Estimate IR bias from *evaluation_path* and optionally persist the
    correction to *modality_config_path*.

    Returns a summary dict with bias, std, standard-error, and the
    center_offset values that will be (or were) written.

    Raises ValueError on missing/invalid input files.
    """
    try:
        with evaluation_path.open() as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ValueError(f"Evaluation file not found: {evaluation_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {evaluation_path}: {exc}") from None

    all_samples = data.get("samples", [])
    ir_samples = [
        s for s in all_samples
        if "/IR/" in str(s.get("sequence_key", ""))
    ]

    if not ir_samples:
        raise ValueError(
            "No IR samples found in the evaluation file. "
            "Make sure the dataset contains IR sequences and was evaluated first."
        )

    mean_x, mean_y, std_x, std_y, n = estimate_bias(ir_samples)
    se_x = std_x / math.sqrt(n) if n > 0 else 0.0
    se_y = std_y / math.sqrt(n) if n > 0 else 0.0

    # The correction sign: subtract the bias from raw centres
    offset_x = -mean_x
    offset_y = -mean_y

    result = {
        "ir_samples": len(ir_samples),
        "residual_points": n,
        "mean_bias_x": round(mean_x, 4),
        "mean_bias_y": round(mean_y, 4),
        "std_x": round(std_x, 4),
        "std_y": round(std_y, 4),
        "se_x": round(se_x, 4),
        "se_y": round(se_y, 4),
        "center_offset_x": round(offset_x, 4),
        "center_offset_y": round(offset_y, 4),
    }

    logger.info(
        "IR calibration — %d samples, %d residual observations",
        len(ir_samples), n,
    )
    logger.info("  Mean bias   X: %+.3f px  (±%.3f se)", mean_x, se_x)
    logger.info("  Mean bias   Y: %+.3f px  (±%.3f se)", mean_y, se_y)
    logger.info("  Std         X: %.3f px   Y: %.3f px", std_x, std_y)
    logger.info(
        "  Correction  center_offset_x=%+.4f  center_offset_y=%+.4f",
        offset_x, offset_y,
    )

    if dry_run:
        logger.info("[dry-run] Would write to %s — skipped.", modality_config_path)
        return result

    # Load existing config (preserves all other keys)
    if modality_config_path.exists():
        with modality_config_path.open() as fh:
            config = json.load(fh)
    else:
        config = {}

    config.setdefault("IR", {})
    config["IR"]["center_offset_x"] = round(offset_x, 4)
    config["IR"]["center_offset_y"] = round(offset_y, 4)

    with modality_config_path.open("w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    logger.info("Correction written to %s", modality_config_path)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate IR sensor coordinate bias and write correction to modality_config.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluation",
        required=True,
        help="Path to predictor_evaluation.json.",
    )
    parser.add_argument(
        "--modality-config",
        default="trajectory_reuse/modality_config.json",
        help="Path to modality_config.json to update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to modality_config.json.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        result = calibrate(
            evaluation_path=Path(args.evaluation),
            modality_config_path=Path(args.modality_config),
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Result summary:")
    for key, value in result.items():
        logger.info("  %-22s %s", key + ":", value)


if __name__ == "__main__":
    main()
