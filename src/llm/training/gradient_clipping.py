"""
CrackLawLM Gradient Clipping & Statistics
==========================================
Implements gradient clipping by max norm and provides gradient statistics
for monitoring training health (exploding/vanishing gradients).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from src.llm.training.config import TrainingConfig


class GradientClipper:
    """
    Handles gradient clipping and gradient health monitoring.

    Gradient clipping prevents exploding gradients by scaling the gradient
    vector so its total norm does not exceed `max_norm`.

    Also provides gradient statistics (total norm, max, min, mean) for
    debugging and logging.
    """

    def __init__(self, config: TrainingConfig):
        self.enabled = config.gradient_clip_enabled
        self.max_norm = config.gradient_clip_max_norm
        self.norm_type = config.gradient_clip_norm_type

    def clip(self, model: nn.Module) -> float:
        """
        Clips gradients by global norm and returns the pre-clip gradient norm.

        Args:
            model: The model whose gradients will be clipped.

        Returns:
            The total gradient norm BEFORE clipping (for logging).
        """
        if not self.enabled:
            return self.compute_total_norm(model)

        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )

        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

    @staticmethod
    def compute_total_norm(model: nn.Module, norm_type: float = 2.0) -> float:
        """
        Computes the total gradient norm across all parameters.

        Args:
            model:     The model to inspect.
            norm_type: Type of norm (default: L2).

        Returns:
            The total gradient norm as a float.
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    @staticmethod
    def compute_gradient_statistics(model: nn.Module) -> Dict[str, Optional[float]]:
        """
        Computes detailed gradient statistics for training health monitoring.

        Returns:
            Dictionary with: total_norm, max_grad, min_grad, mean_grad, num_params_with_grad.
            Returns None values if no gradients exist.
        """
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach())

        if not grads:
            return {
                "total_norm": None,
                "max_grad": None,
                "min_grad": None,
                "mean_grad": None,
                "num_params_with_grad": 0,
            }

        all_grads = torch.cat([g.flatten() for g in grads])
        total_norm = torch.norm(torch.stack([torch.norm(g, 2.0) for g in grads]), 2.0).item()

        return {
            "total_norm": total_norm,
            "max_grad": all_grads.max().item(),
            "min_grad": all_grads.min().item(),
            "mean_grad": all_grads.mean().item(),
            "num_params_with_grad": len(grads),
        }
