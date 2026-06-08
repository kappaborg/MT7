#!/usr/bin/env python3
"""
Deliverable 4 — Training script for MLTrajectoryPredictor.

Loss:  minADE (Winner-Takes-All over K hypotheses)
     + lambda_nll  × NLL on mixture weights  (prevents hypothesis collapse)
     + lambda_fde  × minFDE                  (penalises final-step error)

Metrics per epoch: ADE, FDE, per-class ADE (drone / pedestrian / vehicle / …)

Logging: TensorBoard (default) or Weights & Biases
Checkpointing: top-k by val_ade; compatible with MLTrajectoryPredictor.load_checkpoint()

Usage:
    python3 train_ml_predictor.py --config configs/train_config.yaml

    # resume from a checkpoint
    python3 train_ml_predictor.py --config configs/train_config.yaml \\
        --resume output/checkpoints/best.pt

    # quick smoke-test (2 epochs, tiny batch)
    python3 train_ml_predictor.py --config configs/train_config.yaml \\
        --epochs 2 --batch-size 8 --dry-run
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ── reproducibility ───────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── dataset ───────────────────────────────────────────────────────────────────

class TrajectoryDataset:
    """
    Thin wrapper around the .pt files produced by prepare_training_data.py.
    Returns per-sample dicts that the DataLoader collates into batches.
    """

    def __init__(self, path: str):
        import torch
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.obs_traj      = data["obs_traj"]       # [N, T_obs, 2]
        self.pred_traj     = data["pred_traj"]       # [N, T_pred, 2]
        self.obs_vel       = data["obs_vel"]         # [N, T_obs, 2]
        self.obs_acc       = data["obs_acc"]         # [N, T_obs, 2]
        self.obs_area_norm = data["obs_area_norm"]   # [N, T_obs, 1]
        self.is_hover      = data["is_hover"]        # [N, T_obs]  long
        self.obj_type      = data["obj_type"]        # [N]         long
        self.modality      = data["modality"]        # [N]         long

    def __len__(self) -> int:
        return self.obs_traj.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        return {
            "obs_traj":      self.obs_traj[idx],
            "pred_traj":     self.pred_traj[idx],
            "obs_vel":       self.obs_vel[idx],
            "obs_acc":       self.obs_acc[idx],
            "obs_area_norm": self.obs_area_norm[idx],
            "is_hover":      self.is_hover[idx],
            "obj_type":      self.obj_type[idx],
            "modality":      self.modality[idx],
        }


# ── loss functions ────────────────────────────────────────────────────────────

def min_ade_loss(
    pred_xy: "torch.Tensor",   # [B, K, T_pred, 2]
    gt_xy:   "torch.Tensor",   # [B, T_pred, 2]
    weights: "torch.Tensor",   # [B, K]
    lambda_nll: float = 0.15,
    lambda_fde: float = 0.5,
) -> "Tuple[torch.Tensor, Dict[str, float]]":
    """
    Winner-Takes-All minADE loss with NLL regularization and FDE penalty.

    Returns (scalar_loss, metrics_dict).
    """
    import torch

    B, K, T, _ = pred_xy.shape
    gt = gt_xy.unsqueeze(1).expand_as(pred_xy)   # [B, K, T, 2]

    # Per-hypothesis ADE: mean displacement over time steps
    displacement = torch.norm(pred_xy - gt, dim=-1)  # [B, K, T]
    ade_per_hyp  = displacement.mean(dim=-1)          # [B, K]
    fde_per_hyp  = displacement[:, :, -1]             # [B, K]  final step

    # Winner-Takes-All: pick the hypothesis with minimum ADE
    best_idx     = ade_per_hyp.argmin(dim=1)          # [B]
    batch_idx    = torch.arange(B, device=pred_xy.device)

    wta_ade = ade_per_hyp[batch_idx, best_idx].mean()
    wta_fde = fde_per_hyp[batch_idx, best_idx].mean()

    # NLL: penalise low weight on the winning hypothesis
    best_weight = weights[batch_idx, best_idx].clamp(min=1e-9)
    nll_loss    = -torch.log(best_weight).mean()

    loss = wta_ade + lambda_fde * wta_fde + lambda_nll * nll_loss

    metrics = {
        "ade":     wta_ade.item(),
        "fde":     wta_fde.item(),
        "nll":     nll_loss.item(),
        "loss":    loss.item(),
    }
    return loss, metrics


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_xy: "torch.Tensor",   # [B, K, T, 2]
    gt_xy:   "torch.Tensor",   # [B, T, 2]
    weights: "torch.Tensor",   # [B, K]
    obj_type: "torch.Tensor",  # [B]  long
    type_names: Dict[int, str],
) -> Dict[str, float]:
    """
    Compute ADE, FDE, and per-class ADE using the best hypothesis (argmax weights).
    Used for validation — no gradient flow.
    """
    import torch

    B, K, T, _ = pred_xy.shape
    best_k  = weights.argmax(dim=1)                                # [B]
    b_idx   = torch.arange(B, device=pred_xy.device)
    best_xy = pred_xy[b_idx, best_k]                              # [B, T, 2]

    disp    = torch.norm(best_xy - gt_xy, dim=-1)                 # [B, T]
    ade     = disp.mean(dim=-1)                                    # [B]
    fde     = disp[:, -1]                                          # [B]

    out: Dict[str, float] = {
        "val_ade": ade.mean().item(),
        "val_fde": fde.mean().item(),
    }

    # Per-class ADE
    for type_id, type_name in type_names.items():
        mask = obj_type == type_id
        if mask.any():
            out[f"val_ade_{type_name}"] = ade[mask].mean().item()

    return out


# ── LR schedule ──────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
):
    import torch

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lr  = optimizer.param_groups[0]["initial_lr"]
        ratio    = min_lr / base_lr if base_lr > 0 else 0.0
        return ratio + (1.0 - ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── logger backends ───────────────────────────────────────────────────────────

class _Logger:
    def log(self, metrics: Dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class TensorBoardLogger(_Logger):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard log dir: %s", log_dir)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, global_step=step)

    def close(self) -> None:
        self._writer.close()


class WandbLogger(_Logger):
    def __init__(self, project: str, entity: str, config: Dict):
        import wandb
        self._wandb = wandb
        wandb.init(project=project, entity=entity or None, config=config)
        logger.info("W&B run: %s", wandb.run.url)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._wandb.finish()


class NullLogger(_Logger):
    def log(self, metrics: Dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


def _build_logger(cfg: Dict, hparams: Dict) -> _Logger:
    backend = cfg.get("backend", "none")
    try:
        if backend == "tensorboard":
            return TensorBoardLogger(cfg.get("log_dir", "output/runs"))
        if backend == "wandb":
            return WandbLogger(
                project=cfg.get("wandb_project", "mt7"),
                entity=cfg.get("wandb_entity", ""),
                config=hparams,
            )
    except ImportError as exc:
        logger.warning("Logger backend '%s' unavailable (%s) — using null logger.", backend, exc)
    return NullLogger()


# ── checkpoint management ─────────────────────────────────────────────────────

class CheckpointManager:
    """Keeps the top-k checkpoints by a monitored metric (lower = better)."""

    def __init__(self, output_dir: Path, save_top_k: int = 3, monitor: str = "val_ade"):
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._top_k = save_top_k
        self._monitor = monitor
        # min-heap of (metric_value, path)
        self._heap: List[Tuple[float, str]] = []
        self.best_path: Optional[Path] = None
        self.best_score: float = float("inf")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, float],
        hparams: Dict,
    ) -> bool:
        import torch

        score = metrics.get(self._monitor, float("inf"))
        path  = str(self._dir / f"epoch_{epoch:04d}_{self._monitor}_{score:.4f}.pt")

        checkpoint = {
            "epoch":             epoch,
            "model_state_dict":  model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics":           metrics,
            "hparams":           hparams,
        }
        torch.save(checkpoint, path)

        # Maintain top-k heap
        if len(self._heap) < self._top_k:
            heapq.heappush(self._heap, (-score, path))
        else:
            worst_neg, worst_path = heapq.heappop(self._heap)
            if -score < -worst_neg:   # new score is better (lower)
                try:
                    Path(worst_path).unlink(missing_ok=True)
                except OSError:
                    pass
                heapq.heappush(self._heap, (-score, path))
            else:
                heapq.heappush(self._heap, (worst_neg, worst_path))
                Path(path).unlink(missing_ok=True)
                return False

        is_best = score < self.best_score
        if is_best:
            self.best_score = score
            best_path = self._dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_path = best_path

        last_path = self._dir / "last.pt"
        torch.save(checkpoint, last_path)
        return is_best


# ── early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self._patience   = patience
        self._min_delta  = min_delta
        self._best       = float("inf")
        self._wait       = 0
        self.should_stop = False

    def step(self, metric: float) -> None:
        if metric < self._best - self._min_delta:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                self.should_stop = True


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: Dict, resume_path: Optional[str] = None) -> None:
    import torch
    from torch.utils.data import DataLoader

    _set_seed(cfg["training"]["seed"])

    # ── device ────────────────────────────────────────────────────────────────
    device_str = cfg["training"]["device"]
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Training device: %s", device)

    # ── data ──────────────────────────────────────────────────────────────────
    dcfg = cfg["data"]
    train_ds = TrajectoryDataset(dcfg["train_path"])
    val_ds   = TrajectoryDataset(dcfg["val_path"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=dcfg.get("pin_memory", False),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=dcfg.get("pin_memory", False),
    )

    logger.info(
        "Dataset: train=%d  val=%d  batch=%d  steps/epoch=%d",
        len(train_ds), len(val_ds),
        cfg["training"]["batch_size"],
        len(train_loader),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    from trajectory_reuse.ml_predictor import DroneTrajectoryTransformer
    mcfg = cfg["model"]
    model = DroneTrajectoryTransformer(**mcfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s  params=%d (%.1fk)", type(model).__name__, total_params, total_params / 1000)

    # ── optimizer ─────────────────────────────────────────────────────────────
    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )
    # Store initial_lr for the LR lambda
    for pg in optimizer.param_groups:
        pg["initial_lr"] = tcfg["lr"]

    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=tcfg["warmup_epochs"],
        total_epochs=tcfg["epochs"],
        min_lr=tcfg["min_lr"],
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from %s (epoch %d)", resume_path, ckpt["epoch"])

    # ── logger + checkpointing + early stopping ───────────────────────────────
    type_names = {0: "drone", 1: "pedestrian", 2: "cyclist", 3: "vehicle", 4: "emergency"}
    hparams = {**mcfg, **{f"train_{k}": v for k, v in tcfg.items()}}

    exp_logger = _build_logger(cfg.get("logging", {}), hparams)
    ckpt_mgr   = CheckpointManager(
        output_dir=Path(cfg["checkpointing"]["output_dir"]),
        save_top_k=cfg["checkpointing"].get("save_top_k", 3),
        monitor=cfg["checkpointing"].get("monitor", "val_ade"),
    )
    stopper = EarlyStopping(
        patience=tcfg["early_stopping_patience"],
        min_delta=tcfg.get("early_stopping_min_delta", 0.0),
    )

    global_step = start_epoch * len(train_loader)
    log_every   = cfg.get("logging", {}).get("log_every_n_steps", 10)

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tcfg["epochs"]):
        epoch_start = time.perf_counter()

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_metrics: Dict[str, float] = {"ade": 0, "fde": 0, "nll": 0, "loss": 0}
        n_batches = 0

        for batch in train_loader:
            obs_traj  = batch["obs_traj"].to(device)
            pred_traj = batch["pred_traj"].to(device)
            obs_vel   = batch["obs_vel"].to(device)
            obs_acc   = batch["obs_acc"].to(device)
            obs_area  = batch["obs_area_norm"].to(device)
            is_hover  = batch["is_hover"].to(device)
            obj_type  = batch["obj_type"].to(device)
            modality  = batch["modality"].to(device)

            pred_xy, weights = model(
                obs_traj, obs_vel, obs_acc, obs_area,
                is_hover, obj_type, modality,
            )

            loss, step_metrics = min_ade_loss(
                pred_xy, pred_traj, weights,
                lambda_nll=tcfg["lambda_nll"],
                lambda_fde=tcfg["lambda_fde"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip_norm"])
            optimizer.step()

            for k, v in step_metrics.items():
                train_metrics[k] = train_metrics.get(k, 0) + v
            n_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                exp_logger.log(
                    {f"train/{k}": v for k, v in step_metrics.items()},
                    step=global_step,
                )

        train_metrics = {k: v / n_batches for k, v in train_metrics.items()}

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        all_val: Dict[str, List[float]] = {}

        with torch.no_grad():
            for batch in val_loader:
                obs_traj  = batch["obs_traj"].to(device)
                pred_traj = batch["pred_traj"].to(device)
                obs_vel   = batch["obs_vel"].to(device)
                obs_acc   = batch["obs_acc"].to(device)
                obs_area  = batch["obs_area_norm"].to(device)
                is_hover  = batch["is_hover"].to(device)
                obj_type  = batch["obj_type"].to(device)
                modality  = batch["modality"].to(device)

                pred_xy, weights = model(
                    obs_traj, obs_vel, obs_acc, obs_area,
                    is_hover, obj_type, modality,
                )

                bmetrics = compute_metrics(
                    pred_xy, pred_traj, weights, obj_type, type_names
                )
                for k, v in bmetrics.items():
                    all_val.setdefault(k, []).append(v)

        val_metrics = {k: sum(vs) / len(vs) for k, vs in all_val.items()}

        # ── LR step ───────────────────────────────────────────────────────────
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── log epoch ─────────────────────────────────────────────────────────
        epoch_secs = time.perf_counter() - epoch_start
        log_payload = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "lr": current_lr,
        }
        exp_logger.log(log_payload, step=global_step)

        val_ade = val_metrics.get("val_ade", float("inf"))
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_ade=%.2f  val_ade=%.2f  val_fde=%.2f  "
            "lr=%.2e  [%.1fs]%s",
            epoch + 1, tcfg["epochs"],
            train_metrics["loss"], train_metrics["ade"],
            val_ade, val_metrics.get("val_fde", 0.0),
            current_lr, epoch_secs,
            "  ← BEST" if val_ade < ckpt_mgr.best_score else "",
        )

        # Per-class ADE summary
        class_ades = {k: v for k, v in val_metrics.items() if k.startswith("val_ade_")}
        if class_ades:
            class_str = "  ".join(f"{k.replace('val_ade_','')}={v:.2f}" for k, v in class_ades.items())
            logger.info("  class ADE → %s", class_str)

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt_mgr.save(model, optimizer, scheduler, epoch, val_metrics, hparams)

        # ── early stopping ────────────────────────────────────────────────────
        stopper.step(val_ade)
        if stopper.should_stop:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).",
                epoch + 1, tcfg["early_stopping_patience"],
            )
            break

    exp_logger.close()
    logger.info("Training complete. Best val_ade=%.4f  checkpoint=%s",
                ckpt_mgr.best_score, ckpt_mgr.best_path)

    # ── final test evaluation ─────────────────────────────────────────────────
    test_path = dcfg.get("test_path")
    if test_path and Path(test_path).exists() and ckpt_mgr.best_path is not None:
        logger.info("Evaluating best checkpoint on test set …")
        ckpt = torch.load(ckpt_mgr.best_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_ds = TrajectoryDataset(test_path)
        test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"] * 2, shuffle=False)
        all_test: Dict[str, List[float]] = {}

        with torch.no_grad():
            for batch in test_loader:
                obs_traj  = batch["obs_traj"].to(device)
                pred_traj = batch["pred_traj"].to(device)
                obs_vel   = batch["obs_vel"].to(device)
                obs_acc   = batch["obs_acc"].to(device)
                obs_area  = batch["obs_area_norm"].to(device)
                is_hover  = batch["is_hover"].to(device)
                obj_type  = batch["obj_type"].to(device)
                modality  = batch["modality"].to(device)

                pred_xy, weights = model(
                    obs_traj, obs_vel, obs_acc, obs_area,
                    is_hover, obj_type, modality,
                )
                bm = compute_metrics(pred_xy, pred_traj, weights, obj_type, type_names)
                for k, v in bm.items():
                    all_test.setdefault(k, []).append(v)

        test_metrics = {k: sum(vs) / len(vs) for k, vs in all_test.items()}
        logger.info("Test set results:")
        for k, v in test_metrics.items():
            logger.info("  %-22s %.4f", k + ":", v)

        results_path = Path(cfg["checkpointing"]["output_dir"]) / "test_results.json"
        with results_path.open("w") as fh:
            json.dump(test_metrics, fh, indent=2)
        logger.info("Test results saved to %s", results_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MLTrajectoryPredictor (Temporal Transformer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default="configs/train_config.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--resume",      default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--epochs",      type=int,   default=None,
                        help="Override config epochs.")
    parser.add_argument("--batch-size",  type=int,   default=None,
                        help="Override config batch_size.")
    parser.add_argument("--lr",          type=float, default=None,
                        help="Override config lr.")
    parser.add_argument("--device",      default=None,
                        help="Override config device (cuda/cpu/mps/auto).")
    parser.add_argument("--data-dir",    default=None,
                        help="Override data directory (sets train/val/test paths inside it).")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Run 2 epochs of 2 batches each to verify the pipeline.")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # CLI overrides
    if args.epochs     is not None: cfg["training"]["epochs"]     = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size
    if args.lr         is not None: cfg["training"]["lr"]         = args.lr
    if args.device     is not None: cfg["training"]["device"]     = args.device

    if args.data_dir is not None:
        d = args.data_dir.rstrip("/")
        cfg["data"]["train_path"] = f"{d}/train.pt"
        cfg["data"]["val_path"]   = f"{d}/val.pt"
        cfg["data"]["test_path"]  = f"{d}/test.pt"

    if args.dry_run:
        logger.info("[dry-run] Overriding epochs=2, batch_size=8")
        cfg["training"]["epochs"]                  = 2
        cfg["training"]["batch_size"]              = 8
        cfg["training"]["early_stopping_patience"] = 999
        cfg["logging"]["backend"]                  = "none"

    try:
        train(cfg, resume_path=args.resume)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")


if __name__ == "__main__":
    main()
