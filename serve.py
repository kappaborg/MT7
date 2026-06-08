#!/usr/bin/env python3
"""
Web server for drone trajectory visualization and live prediction.

Usage:
    python3 serve.py \\
        --reconciled-annotations annotations/instances_reconciled.json \\
        --trajectories            annotations/drone_trajectories.json \\
        --frames-root             Frames \\
        --evaluation              annotations/predictor_evaluation.json \\
        --port                    8080

    # With ML predictor loaded at startup:
    python3 serve.py ... --checkpoint output/checkpoints/best.pt

Then open  http://localhost:8080  in your browser.

REST API
--------
POST /api/predict
  Stateless (creates a fresh predictor per call).
  Body  : {"trajectory": [[x,y], ...], "object_type": "drone",
           "prediction_horizon": 3.0, "dt": 1.0}
  Reply : {"predicted_points": [[x,y], ...],
           "confidence": float, "intention": str,
           "threat_level": str, "diagnostics": {...}}

POST /api/predict_live
  Stateful (reuses track history across calls — for live video streams).
  Body  : {"track_id": int, "center": [x, y], "bbox": [x1,y1,x2,y2],
           "object_type": "drone", "confidence": float, "modality": "EO"}
  Reply : same schema as /api/predict

GET /api/status
  Reply : {"status": "ok", "sequences": N, "frames": N, "tracks": N,
           "ml_loaded": bool, "predictor": "ML" | "Physics"}

GET /api/metrics
  Reply : pipeline performance metrics dict
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SAFE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
}


# ──────────────────────────────────────────────────────────────────────────────
# Request handler
# ──────────────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    # Set at startup by _make_handler_class()
    viewer_html:    bytes = b""
    frames_root:    Path  = Path(".")
    status_payload: bytes = b""
    predictor       = None   # MLTrajectoryPredictor or ReusableTrajectoryPredictor
    pipeline        = None   # LiveInferencePipeline (stateful, for /api/predict_live)
    ml_loaded:      bool  = False

    def log_message(self, fmt, *args):  # redirect to Python logging
        logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

    # ── routing ──────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urllib.parse.unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", self.viewer_html)
        elif path == "/api/status":
            self._send(200, "application/json", self.status_payload)
        elif path == "/api/metrics":
            self._handle_metrics()
        elif path.startswith("/Frames/"):
            self._serve_frame(path[len("/Frames/"):])
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        path = urllib.parse.unquote(self.path.split("?")[0])
        if path == "/api/predict":
            self._handle_predict()
        elif path == "/api/predict_live":
            self._handle_predict_live()
        else:
            self.send_error(404, "Not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ── handlers ─────────────────────────────────────────────────────────────

    def _serve_frame(self, rel_path: str):
        frame_path = (self.frames_root / rel_path).resolve()
        # Guard against path-traversal attacks
        try:
            frame_path.relative_to(self.frames_root.resolve())
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not frame_path.exists() or not frame_path.is_file():
            self.send_error(404, "Frame not found")
            return
        suffix = frame_path.suffix.lower()
        if suffix not in _SAFE_IMAGE_SUFFIXES:
            self.send_error(403, "Forbidden")
            return
        data = frame_path.read_bytes()
        self._send(200, _MIME.get(suffix, "application/octet-stream"), data)

    def _handle_metrics(self):
        if self.pipeline is not None:
            metrics = self.pipeline.get_pipeline_metrics()
        elif self.predictor is not None:
            metrics = self.predictor.get_performance_metrics()
        else:
            metrics = {}
        self._json(metrics)

    def _handle_predict(self):
        """Stateless endpoint — creates a fresh predictor per call (backward-compatible)."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            req    = json.loads(body)
        except Exception as exc:
            self._json_error(400, f"Bad request: {exc}")
            return

        trajectory = req.get("trajectory", [])
        if not trajectory:
            self._json_error(400, "trajectory is required")
            return

        object_type = req.get("object_type", "drone")
        horizon     = float(req.get("prediction_horizon", 3.0))
        dt          = float(req.get("dt", 1.0))

        # Use the server-level predictor if loaded (ML or physics fallback),
        # otherwise spin up a one-shot physics predictor for backward compat.
        if self.predictor is not None:
            predictor = self.predictor
        else:
            from trajectory_reuse import ReusableTrajectoryPredictor
            predictor = ReusableTrajectoryPredictor(prediction_horizon=horizon, dt=dt)

        try:
            points = [tuple(float(v) for v in p) for p in trajectory]
        except Exception as exc:
            self._json_error(400, f"Invalid trajectory format: {exc}")
            return

        result = predictor.predict(
            track_id=0,
            trajectory=points,
            object_type=object_type,
        )
        if result is None:
            self._json({"error": "not enough history (need at least 4 points)"})
            return

        self._json(self._prediction_to_dict(result))

    def _handle_predict_live(self):
        """
        Stateful endpoint for real-time streams.
        Accepts one detection per call; the pipeline maintains history.
        """
        if self.pipeline is None:
            self._json_error(503, "Live pipeline not initialised (start server with --checkpoint)")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            req    = json.loads(body)
        except Exception as exc:
            self._json_error(400, f"Bad request: {exc}")
            return

        required = ("track_id", "center")
        for field in required:
            if field not in req:
                self._json_error(400, f"Missing required field: {field}")
                return

        detection = {
            "track_id":    int(req["track_id"]),
            "center":      tuple(float(v) for v in req["center"][:2]),
            "bbox":        req.get("bbox", [0, 0, 0, 0]),
            "object_type": req.get("object_type", "drone"),
            "confidence":  float(req.get("confidence", 1.0)),
            "modality":    req.get("modality", "EO"),
        }

        predictions = self.pipeline.update(
            detections=[detection],
            frame_number=int(req.get("frame_number", 0)),
            laser_range_m=req.get("laser_range_m"),
        )

        if not predictions:
            self._json({"error": "not enough history yet"})
            return

        self._json(self._prediction_to_dict(predictions[0]))

    @staticmethod
    def _prediction_to_dict(result) -> dict:
        return {
            "predicted_points":  [list(p.position) for p in result.predicted_points],
            "confidence":        round(result.confidence, 4),
            "intention":         result.intention,
            "current_position":  list(result.current_position),
            "current_velocity":  list(result.current_velocity),
            "method_used":       result.method_used.value,
            "threat_level":      result.diagnostics.get("threat_level", "benign"),
            "diagnostics":       result.diagnostics,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, code: int, content_type: str, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        self._send(200, "application/json", json.dumps(obj).encode())

    def _json_error(self, code: int, msg: str):
        self._send(code, "application/json", json.dumps({"error": msg}).encode())


def _make_handler_class(
    viewer_html:    bytes,
    frames_root:    Path,
    status_payload: bytes,
    predictor=None,
    pipeline=None,
    ml_loaded: bool = False,
):
    class Handler(_Handler):
        pass
    Handler.viewer_html    = viewer_html
    Handler.frames_root    = frames_root
    Handler.status_payload = status_payload
    Handler.predictor      = predictor
    Handler.pipeline       = pipeline
    Handler.ml_loaded      = ml_loaded
    return Handler


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the drone trajectory viewer and live prediction API over HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reconciled-annotations", required=True,
                        help="Path to instances_reconciled.json")
    parser.add_argument("--trajectories", required=True,
                        help="Path to drone_trajectories.json")
    parser.add_argument("--frames-root", required=True,
                        help="Root directory containing image frames")
    parser.add_argument("--evaluation", default=None,
                        help="Optional path to predictor_evaluation.json")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to ML model checkpoint (enables ML predictor + /api/predict_live)")
    parser.add_argument("--config", default="configs/train_config.yaml",
                        help="Path to train_config.yaml (model hparams)")
    parser.add_argument("--port", type=int, default=8080,
                        help="TCP port to listen on")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host/IP to bind (0.0.0.0 = all interfaces)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    frames_root = Path(args.frames_root).resolve()
    if not frames_root.exists():
        logger.error("frames-root not found: %s", frames_root)
        sys.exit(1)

    # Generate the viewer HTML with web-relative image URLs
    logger.info("Building viewer HTML…")
    import tempfile
    from trajectory_reuse.export_live_viewer import export_live_viewer

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        summary = export_live_viewer(
            reconciled_annotations_path=Path(args.reconciled_annotations),
            trajectories_path=Path(args.trajectories),
            frames_root=frames_root,
            output_path=tmp_path,
            evaluation_path=Path(args.evaluation) if args.evaluation else None,
            frames_url_prefix="/Frames",
        )
        viewer_html = tmp_path.read_bytes()
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    finally:
        tmp_path.unlink(missing_ok=True)

    status_payload = json.dumps({
        "status": "ok",
        "sequences": summary["sequence_count"],
        "frames": summary["frame_count"],
        "tracks": summary["track_count"],
        "evaluation_samples": summary["evaluation_sample_count"],
    }).encode()

    logger.info(
        "Viewer ready: %d sequences, %d frames, %d tracks, %d evaluation samples",
        summary["sequence_count"], summary["frame_count"],
        summary["track_count"], summary["evaluation_sample_count"],
    )

    # ── ML predictor + live pipeline (optional) ───────────────────────────────
    predictor  = None
    pipeline   = None
    ml_loaded  = False

    if args.checkpoint is not None:
        try:
            import yaml
            from trajectory_reuse.ml_predictor import MLTrajectoryPredictor
            from trajectory_reuse.live_inference import LiveInferencePipeline

            cfg_model = {
                "obs_len": 8, "pred_len": 4, "d_model": 32, "nhead": 2,
                "num_encoder_layers": 2, "K": 4, "dropout": 0.2,
                "type_embed_dim": 8, "mod_embed_dim": 4,
            }
            cfg_dt = 1.0
            if Path(args.config).exists():
                with open(args.config) as fh:
                    _cfg = yaml.safe_load(fh)
                cfg_model.update(_cfg.get("model", {}))
                cfg_dt = _cfg.get("training", {}).get("dt", 1.0)

            predictor = MLTrajectoryPredictor(
                checkpoint_path=args.checkpoint,
                **cfg_model,
                dt=cfg_dt,
            )
            pipeline = LiveInferencePipeline(
                predictor=predictor,
                dt=cfg_dt,
                modality_config_path="trajectory_reuse/modality_config.json",
            )
            ml_loaded = True
            logger.info("ML predictor loaded from %s", args.checkpoint)
            logger.info("/api/predict_live endpoint is active")
        except Exception as exc:
            logger.warning("Could not load ML predictor: %s — falling back to physics.", exc)

    # Update status payload with ML info
    import json as _json
    _status = _json.loads(status_payload.decode())
    _status["ml_loaded"]  = ml_loaded
    _status["predictor"]  = "ML" if ml_loaded else "Physics"
    status_payload = _json.dumps(_status).encode()

    handler = _make_handler_class(
        viewer_html, frames_root, status_payload,
        predictor=predictor, pipeline=pipeline, ml_loaded=ml_loaded,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)

    url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
    logger.info("Serving at %s  (Ctrl-C to stop)", url)
    logger.info("API endpoint: POST %s/api/predict", url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
