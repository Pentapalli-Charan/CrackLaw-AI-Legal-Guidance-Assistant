"""
CrackLawLM Training Callbacks
===============================
Extensible callback system for injecting custom logic at training lifecycle points.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("CrackLaw.LLM.Training.Callbacks")


class TrainingCallback:
    """
    Base class for training callbacks.
    Override any method to inject custom behavior at that training phase.
    """

    def on_training_start(self, **kwargs):
        """Called once before the training loop begins."""
        pass

    def on_training_end(self, **kwargs):
        """Called once after the training loop completes."""
        pass

    def on_epoch_start(self, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, train_loss: float,
                     val_loss: float, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_step_start(self, step: int, global_step: int, **kwargs):
        """Called before each training step."""
        pass

    def on_step_end(self, step: int, global_step: int,
                    loss: float, **kwargs):
        """Called after each training step."""
        pass

    def on_validation_start(self, **kwargs):
        """Called before validation begins."""
        pass

    def on_validation_end(self, val_loss: float, **kwargs):
        """Called after validation completes."""
        pass

    def on_checkpoint_saved(self, path: str, **kwargs):
        """Called after a checkpoint is saved."""
        pass

    def on_early_stop(self, epoch: int, best_loss: float, **kwargs):
        """Called when early stopping is triggered."""
        pass


class CallbackManager:
    """Manages and dispatches events to registered callbacks."""

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks: List[TrainingCallback] = callbacks or []

    def add(self, callback: TrainingCallback):
        """Register a new callback."""
        self.callbacks.append(callback)

    def fire(self, event: str, **kwargs):
        """
        Fires an event, calling the corresponding method on all callbacks.

        Args:
            event: Method name to call (e.g., 'on_epoch_start').
            **kwargs: Arguments to pass to the callback method.
        """
        for cb in self.callbacks:
            method = getattr(cb, event, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    logger.warning(
                        f"Callback {cb.__class__.__name__}.{event} "
                        f"raised an exception: {e}"
                    )


# ─────────────────────── Built-in Callbacks ───────────────────────

class ProgressCallback(TrainingCallback):
    """Logs high-level progress messages."""

    def on_training_start(self, **kwargs):
        logger.info("Training started.")

    def on_training_end(self, **kwargs):
        logger.info("Training completed.")

    def on_early_stop(self, epoch: int, best_loss: float, **kwargs):
        logger.info(
            f"Early stopping triggered at epoch {epoch}. "
            f"Best val_loss: {best_loss:.6f}"
        )


class GradientMonitorCallback(TrainingCallback):
    """Warns about abnormal gradient norms."""

    def __init__(self, warn_threshold: float = 10.0, zero_threshold: float = 1e-8):
        self.warn_threshold = warn_threshold
        self.zero_threshold = zero_threshold

    def on_step_end(self, step: int, global_step: int,
                    loss: float, **kwargs):
        grad_norm = kwargs.get("gradient_norm", 0.0)
        if grad_norm > self.warn_threshold:
            logger.warning(
                f"Step {global_step}: gradient norm={grad_norm:.4f} "
                f"exceeds threshold ({self.warn_threshold}). "
                f"Possible exploding gradients."
            )
        elif grad_norm < self.zero_threshold:
            logger.warning(
                f"Step {global_step}: gradient norm={grad_norm:.10f} "
                f"is near zero. Possible vanishing gradients."
            )
