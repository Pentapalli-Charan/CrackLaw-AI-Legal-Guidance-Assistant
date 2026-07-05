"""
CrackLawLM Training Configuration
==================================
Central dataclass governing every tunable hyperparameter for the training engine.
All modules read from this single source of truth.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from src.config import PROJECT_ROOT


@dataclass
class TrainingConfig:
    """Complete configuration for the CrackLawLM Training Engine."""

    # ──────────────────────────── Training Loop ────────────────────────────
    num_epochs: int = 50
    log_every_n_steps: int = 10
    validate_every_n_steps: int = 100
    seed: int = 42

    # ──────────────────────────── Optimizer ─────────────────────────────────
    optimizer_type: str = "adamw"          # Options: adam, adamw, sgd
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    epsilon: float = 1e-9
    momentum: float = 0.9                 # Used only for SGD

    # ──────────────────────────── Scheduler ─────────────────────────────────
    scheduler_type: str = "cosine"        # Options: linear, cosine, warmup, step
    warmup_steps: int = 4000
    total_training_steps: Optional[int] = None  # Auto-computed if None
    step_decay_factor: float = 0.5
    step_decay_every: int = 10000

    # ──────────────────────────── Gradient Clipping ─────────────────────────
    gradient_clip_enabled: bool = True
    gradient_clip_max_norm: float = 1.0
    gradient_clip_norm_type: float = 2.0  # L2 norm

    # ──────────────────────────── Mixed Precision ───────────────────────────
    mixed_precision_enabled: bool = True  # Auto-disabled when CUDA unavailable

    # ──────────────────────────── Early Stopping ───────────────────────────
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # ──────────────────────────── Checkpointing ────────────────────────────
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3

    # ──────────────────────────── Logging ───────────────────────────────────
    log_dir: str = os.path.join(PROJECT_ROOT, "logs", "training")

    # ──────────────────────────── Loss ──────────────────────────────────────
    label_smoothing: float = 0.0
    ignore_index: int = -100             # Standard PyTorch ignore index for CE loss

    def __post_init__(self):
        valid_optimizers = ["adam", "adamw", "sgd"]
        assert self.optimizer_type in valid_optimizers, \
            f"optimizer_type must be one of {valid_optimizers}, got '{self.optimizer_type}'"

        valid_schedulers = ["linear", "cosine", "warmup", "step"]
        assert self.scheduler_type in valid_schedulers, \
            f"scheduler_type must be one of {valid_schedulers}, got '{self.scheduler_type}'"

        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.gradient_clip_max_norm > 0, "gradient_clip_max_norm must be positive"
        assert self.early_stopping_patience > 0, "early_stopping_patience must be positive"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
