#!/usr/bin/env python3
"""
Deliverable 5 — ML predictor evaluation script.

Runs three predictors on the same test.pt windows and prints a side-by-side
comparison table, then saves a full JSON report.

Predictors compared:
  Naive (CV)  — constant-velocity extrapolation from last observed step
  Physics     — ReusableTrajectoryPredictor (existing physics-only model)
  ML best     — DroneTrajectoryTransformer, argmax(weights) hypothesis
  ML oracle   — DroneTrajectoryTransformer, min-ADE hypothesis (upper bound)

Metrics reported:
  ADE         — Average Displacement Error (mean L2 over pred steps)
  FDE         — Final Displacement Error (L2 at last pred step)
  MR          — Miss Rate (fraction with FDE > miss_threshold)
  Intent Acc  — hover/moving classification accuracy on last obs step

All metrics reported overall, per object class, and per modality (EO / IR).

Usage:
    python3 evaluate_ml_predictor.py \\
        --checkpoint output/checkpoints/best.pt \\
        --test-data  output/training_tensors/test.pt \\
        --config     configs/train_config.yaml \\
        --output     output/evaluation_report.json \\
        --miss-threshold 120.0

    # Compare against a second checkpoint (e.g. mid-training)
    python3 evaluate_ml_predictor.py \\
        --checkpoint output/checkpoints/epoch_0050_val_ade_90.pt \\
        --test-data  output/training_tensors/test.pt \\
        --config     configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OBJECT_TYPE_NAMES = {0: "drone", 1: "pedestrian", 2: "cyclist", 3: "vehicle", 4: "emergency"}
MODALITY_NAMES    = {0: "EO", 1: "IR"}
HOVER_SPEED_THRESHOLD = 0.4   # matches predictor.py


# ── per-sample metric computation ─────────────────────────────────────────────

def _ade(pred: "torch.Tensor", gt: "torch.Tensor") -> float:
    """Mean L2 over T prediction steps. pred/gt: [T, 2]."""
    return (pred - gt).norm(dim=-1).mean().item()


def _fde(pred: "torch.Tensor", gt: "torch.Tensor") -> float:
    """L2 at final prediction step. pred/gt: [T, 2]."""
    return (pred[-1] - gt[-1]).norm().item()


def _intent(vel_last: "torch.Tensor") -> str:
    """Classify last-observed velocity as 'hover' or 'moving'."""
    return "hover" if vel_last.norm().item() < HOVER_SPEED_THRESHOLD else "moving"


# ── naive constant-velocity predictor ─────────────────────────────────────────

def _naive_cv_predict(
    obs_traj: "torch.Tensor",   # [T_obs, 2]
    obs_vel:  "torch.Tensor",   # [T_obs, 2]
    pred_len: int,
) -> "torch.Tensor":
    """
    Constant-velocity baseline: last observed position + last velocity × step.
    Returns [T_pred, 2] in the same local-frame coordinates as obs_traj.
    """
    import torch
    last_pos = obs_traj[-1]     # [2]
    last_vel = obs_vel[-1]      # [2]
    steps = torch.arange(1, pred_len + 1, dtype=torch.float32, device=obs_traj.device)
    return last_pos.unsqueeze(0) + last_vel.unsqueeze(0) * steps.unsqueeze(1)


# ── physics predictor (local-frame wrapper) ───────────────────────────────────

def _physics_predict_local(
    obs_traj:   "torch.Tensor",   # [T_obs, 2]  — local frame
    obj_type_id: int,
    pred_len:   int,
    dt:         float,
    physics,
) -> Optional["torch.Tensor"]:
    """
    Run ReusableTrajectoryPredictor on local-frame trajectory.
    The predictor is agnostic to absolute position, so local-frame input is valid.
    Returns [T_pred, 2] or None.
    """
    import torch
    traj = [tuple(p.tolist()) for p in obs_traj]
    obj_type_str = OBJECT_TYPE_NAMES.get(obj_type_id, "object")
    result = physics.predict(
        track_id=0,
        trajectory=traj,
        object_type=obj_type_str,
        context={"max_speed": 500.0},  # uncapped — local frame has no real-world scale
    )
    if result is None or len(result.predicted_points) < pred_len:
        return None
    pts = [result.predicted_points[i].position for i in range(pred_len)]
    return torch.tensor(pts, dtype=torch.float32)


# ── aggregate stats ───────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[idx]


def _miss_rate(fde_values: List[float], threshold: float) -> float:
    if not fde_values:
        return float("nan")
    return sum(1 for v in fde_values if v > threshold) / len(fde_values)


def _intent_accuracy(gt_intents: List[str], pred_intents: List[str]) -> float:
    if not gt_intents:
        return float("nan")
    return sum(g == p for g, p in zip(gt_intents, pred_intents)) / len(gt_intents)


# ── per-group stats aggregation ───────────────────────────────────────────────

class _Accumulator:
    """Collects per-sample results for one predictor method."""

    def __init__(self):
        self.ade:          List[float] = []
        self.fde:          List[float] = []
        self.gt_intent:    List[str]   = []
        self.pred_intent:  List[str]   = []
        self.missed:       int = 0
        self.total:        int = 0
        # keyed by class/modality id
        self.ade_by_class:    Dict[int, List[float]] = {}
        self.fde_by_class:    Dict[int, List[float]] = {}
        self.ade_by_modality: Dict[int, List[float]] = {}
        self.fde_by_modality: Dict[int, List[float]] = {}

    def add(
        self,
        ade: float, fde: float,
        gt_intent: str, pred_intent: str,
        obj_type: int, modality: int,
        miss_threshold: float,
    ) -> None:
        self.ade.append(ade)
        self.fde.append(fde)
        self.gt_intent.append(gt_intent)
        self.pred_intent.append(pred_intent)
        self.total += 1
        if fde > miss_threshold:
            self.missed += 1
        self.ade_by_class.setdefault(obj_type, []).append(ade)
        self.fde_by_class.setdefault(obj_type, []).append(fde)
        self.ade_by_modality.setdefault(modality, []).append(ade)
        self.fde_by_modality.setdefault(modality, []).append(fde)

    def summary(self, miss_threshold: float) -> Dict:
        return {
            "n": self.total,
            "ade_mean":   _mean(self.ade),
            "ade_median": _median(self.ade),
            "ade_p90":    _pct(self.ade, 90),
            "fde_mean":   _mean(self.fde),
            "fde_median": _median(self.fde),
            "miss_rate":  _miss_rate(self.fde, miss_threshold),
            "intent_acc": _intent_accuracy(self.gt_intent, self.pred_intent),
            "per_class": {
                OBJECT_TYPE_NAMES.get(k, str(k)): {
                    "n":          len(v),
                    "ade_mean":   _mean(v),
                    "fde_mean":   _mean(self.fde_by_class[k]),
                    "miss_rate":  _miss_rate(self.fde_by_class[k], miss_threshold),
                }
                for k, v in self.ade_by_class.items()
            },
            "per_modality": {
                MODALITY_NAMES.get(k, str(k)): {
                    "n":          len(v),
                    "ade_mean":   _mean(v),
                    "fde_mean":   _mean(self.fde_by_modality[k]),
                    "miss_rate":  _miss_rate(self.fde_by_modality[k], miss_threshold),
                }
                for k, v in self.ade_by_modality.items()
            },
        }


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: Optional[str],
    test_data_path: str,
    config_path: Optional[str],
    output_path: Optional[str],
    miss_threshold: float = 120.0,
    device_str: str = "auto",
) -> Dict:
    import torch
    from trajectory_reuse.ml_predictor import DroneTrajectoryTransformer
    from trajectory_reuse.predictor import ReusableTrajectoryPredictor

    # ── device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Evaluation device: %s", device)

    # ── config ────────────────────────────────────────────────────────────────
    cfg_model = {
        "obs_len": 8, "pred_len": 4, "d_model": 32, "nhead": 2,
        "num_encoder_layers": 2, "K": 4, "dropout": 0.2,
        "type_embed_dim": 8, "mod_embed_dim": 4,
    }
    cfg_train = {"dt": 1.0}

    if config_path is not None:
        import yaml
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg_model.update(cfg.get("model", {}))
        cfg_train.update(cfg.get("training", {}))

    obs_len  = cfg_model["obs_len"]
    pred_len = cfg_model["pred_len"]
    dt       = cfg_train.get("dt", 1.0)
    K        = cfg_model["K"]

    # ── load test tensors ─────────────────────────────────────────────────────
    logger.info("Loading test data from %s …", test_data_path)
    data = torch.load(test_data_path, map_location="cpu", weights_only=True)
    n_samples = data["obs_traj"].shape[0]
    logger.info("Test samples: %d", n_samples)

    # ── load ML model ─────────────────────────────────────────────────────────
    ml_model: Optional[DroneTrajectoryTransformer] = None
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        saved_hparams = ckpt.get("hparams", {})
        # Only pass keys the model constructor accepts — filter out training hparams
        _model_keys = {"obs_len", "pred_len", "d_model", "nhead", "num_encoder_layers",
                       "K", "dropout", "type_embed_dim", "mod_embed_dim"}
        merged_hparams = {k: saved_hparams.get(k, v)
                          for k, v in cfg_model.items() if k in _model_keys}
        ml_model = DroneTrajectoryTransformer(**merged_hparams).to(device)
        state = ckpt.get("model_state_dict", ckpt)
        ml_model.load_state_dict(state)
        ml_model.eval()
        logger.info("ML model loaded from %s", checkpoint_path)
        ckpt_epoch = ckpt.get("epoch", "?")
        ckpt_val_ade = ckpt.get("metrics", {}).get("val_ade", "?")
        logger.info("  checkpoint: epoch=%s  val_ade=%s", ckpt_epoch, ckpt_val_ade)
    else:
        logger.warning("No checkpoint provided — ML columns will show N/A.")

    # ── physics predictor ─────────────────────────────────────────────────────
    physics = ReusableTrajectoryPredictor(
        prediction_horizon=pred_len * dt,
        dt=dt,
        min_history=4,
        max_history=obs_len,
    )

    # ── accumulators ──────────────────────────────────────────────────────────
    acc_naive   = _Accumulator()
    acc_physics = _Accumulator()
    acc_ml_best = _Accumulator()
    acc_ml_ora  = _Accumulator()   # oracle: best hypothesis by ADE, not weights

    skipped_physics = 0
    skipped_ml      = 0

    # ── per-sample loop ───────────────────────────────────────────────────────
    with torch.no_grad():
        for idx in range(n_samples):
            obs_traj   = data["obs_traj"][idx]        # [T_obs, 2]
            pred_traj  = data["pred_traj"][idx]        # [T_pred, 2]
            obs_vel    = data["obs_vel"][idx]           # [T_obs, 2]
            obs_acc    = data["obs_acc"][idx]           # [T_obs, 2]
            obs_area   = data["obs_area_norm"][idx]    # [T_obs, 1]
            is_hover   = data["is_hover"][idx]          # [T_obs]
            obj_type   = int(data["obj_type"][idx].item())
            modality   = int(data["modality"][idx].item())

            gt_intent = _intent(obs_vel[-1])

            # ── naive CV ──────────────────────────────────────────────────────
            naive_pred = _naive_cv_predict(obs_traj, obs_vel, pred_len)
            naive_intent = _intent(naive_pred[0] - obs_traj[-1])  # approx velocity at step 1
            acc_naive.add(
                ade=_ade(naive_pred, pred_traj),
                fde=_fde(naive_pred, pred_traj),
                gt_intent=gt_intent,
                pred_intent=naive_intent,
                obj_type=obj_type,
                modality=modality,
                miss_threshold=miss_threshold,
            )

            # ── physics ───────────────────────────────────────────────────────
            phys_pred = _physics_predict_local(obs_traj, obj_type, pred_len, dt, physics)
            if phys_pred is not None:
                phys_intent = _intent(phys_pred[0] - obs_traj[-1])
                acc_physics.add(
                    ade=_ade(phys_pred, pred_traj),
                    fde=_fde(phys_pred, pred_traj),
                    gt_intent=gt_intent,
                    pred_intent=phys_intent,
                    obj_type=obj_type,
                    modality=modality,
                    miss_threshold=miss_threshold,
                )
            else:
                skipped_physics += 1

            # ── ML ────────────────────────────────────────────────────────────
            if ml_model is None:
                continue

            obs_t    = obs_traj.unsqueeze(0).to(device)
            vel_t    = obs_vel.unsqueeze(0).to(device)
            acc_t    = obs_acc.unsqueeze(0).to(device)
            area_t   = obs_area.unsqueeze(0).to(device)
            hover_t  = is_hover.unsqueeze(0).to(device)
            type_t   = torch.tensor([obj_type], device=device)
            mod_t    = torch.tensor([modality], device=device)

            pred_xy, weights = ml_model(obs_t, vel_t, acc_t, area_t, hover_t, type_t, mod_t)
            # pred_xy: [1, K, T_pred, 2]   weights: [1, K]
            pred_xy  = pred_xy[0]    # [K, T_pred, 2]
            weights  = weights[0]    # [K]
            gt_t     = pred_traj.to(device)

            # Best hypothesis by argmax(weights)
            best_k = weights.argmax().item()
            best_pred = pred_xy[best_k].cpu()
            best_intent = _intent(best_pred[0] - obs_traj[-1])
            acc_ml_best.add(
                ade=_ade(best_pred, pred_traj),
                fde=_fde(best_pred, pred_traj),
                gt_intent=gt_intent,
                pred_intent=best_intent,
                obj_type=obj_type,
                modality=modality,
                miss_threshold=miss_threshold,
            )

            # Oracle: best hypothesis by ADE (upper bound)
            ade_per_k = [(pred_xy[k].cpu() - pred_traj).norm(dim=-1).mean().item() for k in range(K)]
            oracle_k  = min(range(K), key=lambda k: ade_per_k[k])
            oracle_pred = pred_xy[oracle_k].cpu()
            acc_ml_ora.add(
                ade=_ade(oracle_pred, pred_traj),
                fde=_fde(oracle_pred, pred_traj),
                gt_intent=gt_intent,
                pred_intent=_intent(oracle_pred[0] - obs_traj[-1]),
                obj_type=obj_type,
                modality=modality,
                miss_threshold=miss_threshold,
            )

    logger.info("Physics skipped (short history): %d/%d", skipped_physics, n_samples)
    if ml_model is not None:
        logger.info("ML skipped: %d/%d", skipped_ml, n_samples)

    # ── build report ──────────────────────────────────────────────────────────
    report = {
        "checkpoint":      checkpoint_path,
        "test_data":       test_data_path,
        "n_samples":       n_samples,
        "miss_threshold":  miss_threshold,
        "obs_len":         obs_len,
        "pred_len":        pred_len,
        "dt":              dt,
        "methods": {
            "naive_cv":  acc_naive.summary(miss_threshold),
            "physics":   acc_physics.summary(miss_threshold),
            "ml_best":   acc_ml_best.summary(miss_threshold) if ml_model else None,
            "ml_oracle": acc_ml_ora.summary(miss_threshold)  if ml_model else None,
        },
    }

    # ── print table ───────────────────────────────────────────────────────────
    _print_report(report, miss_threshold)

    # ── save JSON ─────────────────────────────────────────────────────────────
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Full report saved to %s", output_path)

    return report


# ── pretty printer ────────────────────────────────────────────────────────────

def _fmt(v, fmt=".2f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  N/A  "
    return format(v, fmt)


def _print_report(report: Dict, miss_threshold: float) -> None:
    w = 72
    sep  = "═" * w
    line = "─" * w

    def _row(label, m, indent=""):
        if m is None:
            return f"  {indent}{label:<18}│{'  N/A  ':>9}│{'  N/A  ':>9}│{'  N/A  ':>9}│{'  N/A  ':>9}"
        ia = m.get("intent_acc")
        return (
            f"  {indent}{label:<18}"
            f"│{_fmt(m['ade_mean']):>9}"
            f"│{_fmt(m['fde_mean']):>9}"
            f"│{_fmt(m.get('miss_rate', float('nan')), '.1%'):>9}"
            f"│{_fmt(ia, '.1%') if ia is not None else '  N/A  ':>9}"
        )

    header = (
        f"  {'Method':<18}"
        f"│{'ADE ↓':>9}"
        f"│{'FDE ↓':>9}"
        f"│{f'MR@{miss_threshold:.0f}px ↓':>9}"
        f"│{'IntAcc ↑':>9}"
    )

    methods = report["methods"]
    print()
    print(sep)
    print(f"  TRAJECTORY PREDICTION EVALUATION")
    print(f"  test samples={report['n_samples']}  obs={report['obs_len']}  pred={report['pred_len']}  dt={report['dt']}")
    print(sep)
    print(header)
    print(line)
    print(_row("Naive (CV)",  methods["naive_cv"]))
    print(_row("Physics",     methods["physics"]))
    print(_row("ML best-k",   methods["ml_best"]))
    print(_row("ML oracle",   methods["ml_oracle"]))
    print(sep)

    # Per-class breakdown (ML best-k)
    ml = methods.get("ml_best") or methods.get("physics")
    if ml and ml.get("per_class"):
        print("  Per-class ADE — ML best-k (or Physics if ML unavailable):")
        for cls, stats in ml["per_class"].items():
            if not math.isnan(stats["ade_mean"]):
                print(f"    {cls:<12}  ADE={_fmt(stats['ade_mean'])}  FDE={_fmt(stats['fde_mean'])}  MR={_fmt(stats['miss_rate'],'.1%')}  n={stats['n']}")
        print(line)

    # Per-modality breakdown (ML best-k)
    if ml and ml.get("per_modality"):
        print("  Per-modality ADE — ML best-k (or Physics):")
        for mod, stats in ml["per_modality"].items():
            if not math.isnan(stats["ade_mean"]):
                print(f"    {mod:<6}  ADE={_fmt(stats['ade_mean'])}  FDE={_fmt(stats['fde_mean'])}  MR={_fmt(stats['miss_rate'],'.1%')}  n={stats['n']}")
        print(sep)

    # Improvement vs physics
    phys_m = methods.get("physics")
    ml_m   = methods.get("ml_best")
    if phys_m and ml_m and not math.isnan(phys_m["ade_mean"]) and not math.isnan(ml_m["ade_mean"]):
        delta_ade = phys_m["ade_mean"] - ml_m["ade_mean"]
        pct_ade   = 100 * delta_ade / phys_m["ade_mean"]
        delta_fde = phys_m["fde_mean"] - ml_m["fde_mean"]
        pct_fde   = 100 * delta_fde / phys_m["fde_mean"]
        print(f"  ML vs Physics:  ΔADE={delta_ade:+.2f} ({pct_ade:+.1f}%)   ΔFDE={delta_fde:+.2f} ({pct_fde:+.1f}%)")
        print(sep)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ML, Physics, and Naive predictors on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",      default=None,
                        help="Path to trained model checkpoint (best.pt).")
    parser.add_argument("--test-data",       default="output/training_tensors/test.pt",
                        help="Path to test.pt produced by prepare_training_data.py.")
    parser.add_argument("--config",          default="configs/train_config.yaml",
                        help="Path to train_config.yaml (used for model hparams).")
    parser.add_argument("--output",          default=None,
                        help="Path to save the full JSON report.")
    parser.add_argument("--miss-threshold",  type=float, default=120.0,
                        help="FDE threshold in pixels for Miss Rate calculation.")
    parser.add_argument("--device",          default="auto",
                        help="Device: auto / cuda / mps / cpu.")
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    evaluate(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        config_path=args.config,
        output_path=args.output,
        miss_threshold=args.miss_threshold,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
