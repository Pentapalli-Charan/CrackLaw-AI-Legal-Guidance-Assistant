"""
CrackLawLM Optimizer Management
================================
Factory for creating and configuring PyTorch optimizers from TrainingConfig.
Supports Adam, AdamW, and SGD with full parameter control.
"""

import torch.nn as nn
import torch.optim as optim

from src.llm.training.config import TrainingConfig


class OptimizerFactory:
    """
    Creates and configures optimizers for the CrackLaw Transformer.

    Supported optimizers:
      - Adam:  Adaptive learning rates with momentum (Kingma & Ba, 2014)
      - AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019)
      - SGD:   Stochastic Gradient Descent with optional momentum

    Usage:
        optimizer = OptimizerFactory.create(model, config)
    """

    @staticmethod
    def create(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """
        Factory method: creates the optimizer specified in config.

        Args:
            model:  The nn.Module whose parameters will be optimized.
            config: TrainingConfig with optimizer_type, learning_rate, etc.

        Returns:
            A configured PyTorch Optimizer instance.
        """
        params = model.parameters()

        if config.optimizer_type == "adam":
            return optim.Adam(
                params,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            )

        elif config.optimizer_type == "adamw":
            return optim.AdamW(
                params,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            )

        elif config.optimizer_type == "sgd":
            return optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )

        else:
            raise ValueError(
                f"Unknown optimizer_type '{config.optimizer_type}'. "
                f"Supported: adam, adamw, sgd"
            )

    @staticmethod
    def get_param_count(model: nn.Module) -> dict:
        """Returns total and trainable parameter counts for logging."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
        }
