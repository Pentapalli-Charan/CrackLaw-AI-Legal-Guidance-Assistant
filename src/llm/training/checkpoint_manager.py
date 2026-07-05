"""
CrackLawLM Checkpoint Manager
===============================
Saves/loads complete training state for resume-able training.
"""

import os
import json
import glob
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from src.llm.training.config import TrainingConfig

logger = logging.getLogger("CrackLaw.LLM.Training.Checkpoint")


class CheckpointManager:
    """Production-grade checkpoint manager for long-running training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, global_step,
             train_loss, val_loss, best_val_loss,
             amp_state=None, extra_metadata=None) -> str:
        """Saves a complete training checkpoint."""
        path = os.path.join(self.checkpoint_dir,
                            f"checkpoint_epoch_{epoch:04d}_step_{global_step:08d}.pt")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch, "global_step": global_step,
            "train_loss": train_loss, "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "training_config": {
                "optimizer_type": self.config.optimizer_type,
                "learning_rate": self.config.learning_rate,
                "scheduler_type": self.config.scheduler_type,
                "warmup_steps": self.config.warmup_steps,
                "gradient_clip_max_norm": self.config.gradient_clip_max_norm,
            },
        }
        if amp_state:
            ckpt["amp_scaler_state_dict"] = amp_state
        if extra_metadata:
            ckpt["extra_metadata"] = extra_metadata
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved: {path}")
        # Sidecar JSON
        meta_path = path.replace(".pt", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch, "global_step": global_step,
                        "train_loss": train_loss, "val_loss": val_loss}, f, indent=2)
        self._cleanup_old_checkpoints()
        return path

    def save_best(self, model, optimizer, scheduler, epoch, global_step,
                  train_loss, val_loss, amp_state=None) -> str:
        """Saves the 'best' model checkpoint."""
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch, "global_step": global_step,
            "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": val_loss,
        }
        if amp_state:
            ckpt["amp_scaler_state_dict"] = amp_state
        torch.save(ckpt, path)
        logger.info(f"Best model saved: {path} (val_loss={val_loss:.6f})")
        return path

    def load(self, checkpoint_path, model, optimizer=None, scheduler=None,
             device=torch.device("cpu")) -> Dict[str, Any]:
        """Loads checkpoint and restores model/optimizer/scheduler state."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        info = {
            "epoch": ckpt.get("epoch", 0),
            "global_step": ckpt.get("global_step", 0),
            "train_loss": ckpt.get("train_loss", float("inf")),
            "val_loss": ckpt.get("val_loss", float("inf")),
            "best_val_loss": ckpt.get("best_val_loss", float("inf")),
            "amp_scaler_state_dict": ckpt.get("amp_scaler_state_dict", {}),
            "extra_metadata": ckpt.get("extra_metadata", {}),
        }
        logger.info(f"Resumed from epoch {info['epoch']}, step {info['global_step']}")
        return info

    def get_latest_checkpoint(self) -> Optional[str]:
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        ckpts = sorted(glob.glob(pattern))
        return ckpts[-1] if ckpts else None

    def get_best_checkpoint(self) -> Optional[str]:
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        return path if os.path.exists(path) else None

    def _cleanup_old_checkpoints(self):
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        ckpts = sorted(glob.glob(pattern))
        while len(ckpts) > self.config.keep_last_n_checkpoints:
            oldest = ckpts.pop(0)
            os.remove(oldest)
            meta = oldest.replace(".pt", "_meta.json")
            if os.path.exists(meta):
                os.remove(meta)
