"""
CrackLawLM Early Stopping
===========================
Monitors validation loss and stops training when no improvement is observed.
"""

import logging
from typing import Optional

from src.llm.training.config import TrainingConfig

logger = logging.getLogger("CrackLaw.LLM.Training.EarlyStopping")


class EarlyStopping:
    """
    Tracks validation loss and signals when training should stop.

    Attributes:
        patience:   Number of epochs to wait for improvement.
        min_delta:  Minimum decrease in loss to qualify as improvement.
        counter:    Current wait counter.
        best_loss:  Best observed validation loss.
        should_stop: Boolean flag checked by the training loop.
    """

    def __init__(self, config: TrainingConfig):
        self.enabled = config.early_stopping_enabled
        self.patience = config.early_stopping_patience
        self.min_delta = config.early_stopping_min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Call after each validation. Returns True if training should stop.

        Args:
            val_loss: Current epoch validation loss.

        Returns:
            True if patience exhausted and training should stop.
        """
        if not self.enabled:
            return False

        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        improvement = self.best_loss - val_loss

        if improvement > self.min_delta:
            # Meaningful improvement: reset counter
            self.best_loss = val_loss
            self.counter = 0
            logger.info(
                f"EarlyStopping: improvement detected "
                f"(delta={improvement:.6f}). Counter reset."
            )
        else:
            # No meaningful improvement
            self.counter += 1
            logger.info(
                f"EarlyStopping: no improvement. "
                f"Counter={self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.warning(
                    f"EarlyStopping TRIGGERED after {self.patience} epochs "
                    f"without improvement. Best val_loss={self.best_loss:.6f}"
                )
                return True

        return False

    def reset(self):
        """Resets the early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
