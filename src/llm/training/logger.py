"""
CrackLawLM Training Logger
============================
Records and persists all training metrics to disk and console.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

import torch

from src.llm.training.config import TrainingConfig

logger = logging.getLogger("CrackLaw.LLM.Training.Logger")


class TrainingLogger:
    """
    Comprehensive training metrics logger.

    Records per-step and per-epoch:
      - Training loss, validation loss
      - Learning rate
      - Gradient norm
      - Tokens per second
      - Epoch duration
      - GPU memory usage (if CUDA available)
      - CPU fallback stats

    Persists all metrics to a JSON log file for post-training analysis.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.log_dir = config.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, "training_metrics.jsonl")
        self.summary_file = os.path.join(self.log_dir, "training_summary.json")

        # Accumulated metrics
        self.step_metrics: List[Dict[str, Any]] = []
        self.epoch_metrics: List[Dict[str, Any]] = []

        # Timing
        self._epoch_start_time: Optional[float] = None
        self._step_start_time: Optional[float] = None

        self.cuda_available = torch.cuda.is_available()

    def start_epoch(self, epoch: int):
        """Called at the start of each epoch."""
        self._epoch_start_time = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch} started")
        logger.info(f"{'='*60}")

    def start_step(self):
        """Called at the start of each training step."""
        self._step_start_time = time.time()

    def log_step(self, step: int, epoch: int, train_loss: float,
                 learning_rate: float, gradient_norm: float,
                 num_tokens: int, global_step: int):
        """Logs metrics for a single training step."""
        elapsed = time.time() - self._step_start_time if self._step_start_time else 0
        tokens_per_sec = num_tokens / max(elapsed, 1e-6)

        record = {
            "type": "step", "epoch": epoch, "step": step,
            "global_step": global_step, "train_loss": train_loss,
            "learning_rate": learning_rate, "gradient_norm": gradient_norm,
            "tokens_per_sec": tokens_per_sec, "step_time_sec": elapsed,
        }

        # GPU memory
        if self.cuda_available:
            record["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            record["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

        self.step_metrics.append(record)
        self._write_jsonl(record)

        if global_step % self.config.log_every_n_steps == 0:
            mem_str = ""
            if self.cuda_available:
                mem_str = (f" | GPU: {record['gpu_memory_allocated_mb']:.0f}MB")
            logger.info(
                f"  Step {step} (global={global_step}) | "
                f"loss={train_loss:.4f} | lr={learning_rate:.2e} | "
                f"grad_norm={gradient_norm:.4f} | "
                f"tok/s={tokens_per_sec:.0f}{mem_str}"
            )

    def log_validation(self, epoch: int, val_loss: float, val_perplexity: float):
        """Logs validation results."""
        record = {
            "type": "validation", "epoch": epoch,
            "val_loss": val_loss, "val_perplexity": val_perplexity,
        }
        self._write_jsonl(record)
        logger.info(
            f"  Validation | loss={val_loss:.4f} | perplexity={val_perplexity:.2f}"
        )

    def end_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  learning_rate: float):
        """Logs epoch-level summary."""
        duration = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        record = {
            "type": "epoch", "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "learning_rate": learning_rate, "epoch_duration_sec": duration,
        }
        if self.cuda_available:
            record["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            record["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

        self.epoch_metrics.append(record)
        self._write_jsonl(record)

        logger.info(
            f"Epoch {epoch} complete | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"lr={learning_rate:.2e} | duration={duration:.1f}s"
        )

    def save_summary(self, final_train_loss: float, final_val_loss: float,
                     best_val_loss: float, total_steps: int,
                     total_epochs: int, total_time: float):
        """Saves a final training summary JSON."""
        summary = {
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "total_steps": total_steps,
            "total_epochs": total_epochs,
            "total_time_sec": total_time,
            "total_time_human": self._format_time(total_time),
            "device": "cuda" if self.cuda_available else "cpu",
        }
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Training summary saved to {self.summary_file}")

    def _write_jsonl(self, record: Dict[str, Any]):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}h {m:02d}m {s:02d}s"
