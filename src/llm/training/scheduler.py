"""
CrackLawLM Learning Rate Schedulers
=====================================
Implements linear decay, cosine decay, warmup, and step decay schedules.
All schedulers use PyTorch's LambdaLR for maximum compatibility.
"""

import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from src.llm.training.config import TrainingConfig


class SchedulerFactory:
    """
    Creates learning rate schedulers from TrainingConfig.

    Supported schedules:
      - linear:  Linearly decays LR from initial to zero over total steps.
      - cosine:  Cosine annealing from initial to zero with optional warmup.
      - warmup:  Linear warmup for `warmup_steps`, then constant.
      - step:    Multiplies LR by `step_decay_factor` every `step_decay_every` steps.

    All schedulers include linear warmup when config.warmup_steps > 0.
    """

    @staticmethod
    def create(
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        total_steps: int,
    ) -> LambdaLR:
        """
        Factory method: creates the scheduler specified in config.

        Args:
            optimizer:    The optimizer to schedule.
            config:       TrainingConfig with scheduler_type, warmup_steps, etc.
            total_steps:  Total number of training steps across all epochs.

        Returns:
            A PyTorch LambdaLR scheduler.
        """
        warmup_steps = config.warmup_steps

        if config.scheduler_type == "linear":
            return SchedulerFactory._create_linear(optimizer, warmup_steps, total_steps)

        elif config.scheduler_type == "cosine":
            return SchedulerFactory._create_cosine(optimizer, warmup_steps, total_steps)

        elif config.scheduler_type == "warmup":
            return SchedulerFactory._create_warmup_only(optimizer, warmup_steps)

        elif config.scheduler_type == "step":
            return SchedulerFactory._create_step_decay(
                optimizer, warmup_steps, config.step_decay_factor, config.step_decay_every
            )

        else:
            raise ValueError(
                f"Unknown scheduler_type '{config.scheduler_type}'. "
                f"Supported: linear, cosine, warmup, step"
            )

    # ────────────────────────── Schedule Implementations ──────────────────

    @staticmethod
    def _create_linear(
        optimizer: optim.Optimizer, warmup_steps: int, total_steps: int
    ) -> LambdaLR:
        """Linear warmup then linear decay to zero."""
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1.0, float(warmup_steps))
            progress = float(current_step - warmup_steps) / max(
                1.0, float(total_steps - warmup_steps)
            )
            return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _create_cosine(
        optimizer: optim.Optimizer, warmup_steps: int, total_steps: int
    ) -> LambdaLR:
        """Linear warmup then cosine annealing to zero."""
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1.0, float(warmup_steps))
            progress = float(current_step - warmup_steps) / max(
                1.0, float(total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _create_warmup_only(
        optimizer: optim.Optimizer, warmup_steps: int
    ) -> LambdaLR:
        """Linear warmup then constant LR."""
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1.0, float(warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _create_step_decay(
        optimizer: optim.Optimizer,
        warmup_steps: int,
        decay_factor: float,
        decay_every: int,
    ) -> LambdaLR:
        """Linear warmup then step decay: multiply by factor every N steps."""
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1.0, float(warmup_steps))
            steps_after_warmup = current_step - warmup_steps
            num_decays = steps_after_warmup // decay_every
            return decay_factor ** num_decays

        return LambdaLR(optimizer, lr_lambda)
