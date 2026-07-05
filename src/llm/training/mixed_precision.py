"""
CrackLawLM Mixed Precision Training
=====================================
Optional Automatic Mixed Precision (AMP) support using PyTorch's native
torch.amp module. Automatically disables when CUDA is unavailable.
"""

import logging
import torch

from src.llm.training.config import TrainingConfig

logger = logging.getLogger("CrackLaw.LLM.Training.MixedPrecision")


class MixedPrecisionManager:
    """
    Manages PyTorch Automatic Mixed Precision (AMP) for training.

    AMP uses float16 for forward/backward passes on CUDA while keeping
    master weights in float32, reducing memory usage and improving throughput
    on GPUs with Tensor Cores.

    Behavior:
      - Automatically disables AMP when CUDA is not available.
      - Wraps the forward pass with autocast.
      - Scales the loss to prevent float16 underflow.
      - Unscales gradients before clipping.

    Usage:
        amp = MixedPrecisionManager(config)
        with amp.autocast_context():
            logits = model(...)
        amp.scale_loss(loss).backward()
        amp.unscale(optimizer)
        # ... clip gradients ...
        amp.optimizer_step(optimizer)
        amp.update_scaler()
    """

    def __init__(self, config: TrainingConfig):
        self.requested = config.mixed_precision_enabled
        self.enabled = self.requested and torch.cuda.is_available()

        if self.requested and not self.enabled:
            logger.warning(
                "Mixed precision was requested but CUDA is unavailable. "
                "Falling back to full-precision (float32) training."
            )

        if self.enabled:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision training ENABLED (CUDA AMP + GradScaler).")
        else:
            self.scaler = None
            logger.info("Mixed precision training DISABLED (full-precision float32).")

    def autocast_context(self):
        """
        Returns an autocast context manager for the forward pass.

        When AMP is enabled, tensors are automatically cast to float16
        inside this context. When disabled, returns a no-op context.
        """
        if self.enabled:
            return torch.amp.autocast("cuda")
        return _NoOpContext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scales the loss before .backward() to prevent float16 gradient underflow.

        Args:
            loss: The computed loss tensor.

        Returns:
            Scaled loss (if AMP enabled) or original loss.
        """
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscales gradients before gradient clipping.

        Must be called AFTER backward() and BEFORE gradient clipping.
        """
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Performs the optimizer step, skipping if gradients contain inf/NaN.
        """
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update_scaler(self) -> None:
        """Updates the gradient scaler after each optimizer step."""
        if self.enabled:
            self.scaler.update()

    def state_dict(self) -> dict:
        """Returns scaler state for checkpointing."""
        if self.enabled:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores scaler state from checkpoint."""
        if self.enabled and state_dict:
            self.scaler.load_state_dict(state_dict)


class _NoOpContext:
    """No-op context manager for when AMP is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
