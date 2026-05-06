#!/usr/bin/env python3
"""
Full pipeline runner: reconcile → build_trajectories → evaluate_predictor → export_live_viewer.

Example usage:
    python3 run_pipeline.py \\
        --annotations   annotations/instances_default.json \\
        --frames-root   Frames \\
        --work-dir      output

Use --dry-run to preview the steps without executing them.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_step(name: str, fn, dry_run: bool, **kwargs):
    if dry_run:
        arg_summary = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info("[dry-run] %s(%s)", name, arg_summary)
        return {}
    logger.info("=== %s ===", name)
    return fn(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full drone trajectory pipeline in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to the raw COCO annotations JSON file (instances_default.json).",
    )
    parser.add_argument(
        "--frames-root",
        required=True,
        help="Root directory containing the actual image frames.",
    )
    parser.add_argument(
        "--work-dir",
        default="output",
        help="Directory where all intermediate and final output files are written.",
    )
    parser.add_argument(
        "--category",
        default="drone",
        help="Category name to track.",
    )
    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Skip the reconcile step and use an existing reconciled annotations file.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip the evaluation step.",
    )
    parser.add_argument(
        "--skip-viewer",
        action="store_true",
        help="Skip the live-viewer export step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing any step.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    work_dir = Path(args.work_dir)
    annotations_path = Path(args.annotations)
    frames_root = Path(args.frames_root)

    reconciled_path = work_dir / "instances_reconciled.json"
    trajectories_path = work_dir / "drone_trajectories.json"
    evaluation_path = work_dir / "predictor_evaluation.json"
    viewer_path = work_dir / "live_trajectory_view.html"

    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_reconcile:
            from trajectory_reuse.reconcile_dataset import reconcile_dataset
            summary = _run_step(
                "reconcile_dataset",
                reconcile_dataset,
                args.dry_run,
                annotations_path=annotations_path,
                frames_root=frames_root,
                output_path=reconciled_path,
            )
            if summary:
                for k, v in summary.items():
                    logger.info("  %s: %s", k, v)
        else:
            if not reconciled_path.exists():
                logger.error(
                    "--skip-reconcile specified but %s does not exist", reconciled_path
                )
                sys.exit(1)
            logger.info("Skipping reconcile; using %s", reconciled_path)

        from trajectory_reuse.build_trajectories import build_trajectory_dataset
        summary = _run_step(
            "build_trajectory_dataset",
            build_trajectory_dataset,
            args.dry_run,
            annotations_path=reconciled_path,
            frames_root=frames_root,
            output_path=trajectories_path,
            category_name=args.category,
        )
        if summary:
            for k, v in summary.items():
                logger.info("  %s: %s", k, v)

        if not args.skip_evaluate:
            from trajectory_reuse.evaluate_predictor import evaluate_trajectory_dataset
            summary = _run_step(
                "evaluate_trajectory_dataset",
                evaluate_trajectory_dataset,
                args.dry_run,
                trajectories_path=trajectories_path,
                output_path=evaluation_path,
            )
            if summary:
                for k, v in summary.items():
                    logger.info("  %s: %s", k, v)

        if not args.skip_viewer:
            from trajectory_reuse.export_live_viewer import export_live_viewer
            eval_arg = evaluation_path if (evaluation_path.exists() or args.dry_run) else None
            summary = _run_step(
                "export_live_viewer",
                export_live_viewer,
                args.dry_run,
                reconciled_annotations_path=reconciled_path,
                trajectories_path=trajectories_path,
                frames_root=frames_root,
                output_path=viewer_path,
                evaluation_path=eval_arg,
            )
            if summary:
                for k, v in summary.items():
                    logger.info("  %s: %s", k, v)

    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if not args.dry_run:
        logger.info("Pipeline complete. Output: %s", work_dir.resolve())
    else:
        logger.info("[dry-run] Pipeline steps printed above. No files were written.")


if __name__ == "__main__":
    main()
