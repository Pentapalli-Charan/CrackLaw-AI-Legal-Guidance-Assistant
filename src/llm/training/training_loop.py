"""
CrackLawLM Training Loop
==========================
The core training and validation loops. Pure algorithmic logic — no orchestration.
Called by the Trainer which handles setup, checkpointing, and lifecycle.
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.llm.training.config import TrainingConfig
from src.llm.training.loss import LanguageModelingLoss
from src.llm.training.gradient_clipping import GradientClipper
from src.llm.training.mixed_precision import MixedPrecisionManager
from src.llm.training.logger import TrainingLogger
from src.llm.training.callbacks import CallbackManager

logger = logging.getLogger("CrackLaw.LLM.Training.Loop")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: LanguageModelingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    gradient_clipper: GradientClipper,
    amp_manager: MixedPrecisionManager,
    training_logger: TrainingLogger,
    callback_manager: CallbackManager,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """
    Executes one complete training epoch.

    Steps per mini-batch:
      1. Move data to device
      2. Forward pass (with optional AMP autocast)
      3. Loss computation
      4. Backward propagation (with optional AMP loss scaling)
      5. Gradient unscaling (for AMP)
      6. Gradient clipping
      7. Optimizer step
      8. Scheduler step
      9. Gradient zeroing
      10. Metric collection

    Returns:
        Dictionary with: avg_loss, total_tokens, global_step (updated).
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        training_logger.start_step()
        callback_manager.fire("on_step_start", step=step, global_step=global_step)

        # 1. Move data to device
        input_ids = batch["input_ids"].to(device)           # (B, seq_len)
        labels = batch["labels"].to(device)                 # (B, seq_len)
        attention_mask = batch["attention_mask"].to(device)  # (B, seq_len)

        # Reshape attention mask for the transformer: (B, seq_len) -> (B, 1, 1, seq_len)
        src_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 2. Forward pass with optional mixed precision
        with amp_manager.autocast_context():
            logits = model(
                src_input_ids=input_ids,
                tgt_input_ids=input_ids,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=src_padding_mask,
            )
            # 3. Loss computation
            loss_out = loss_fn(logits, labels)
            loss = loss_out["loss"]

        # 4. Backward propagation
        optimizer.zero_grad()
        scaled_loss = amp_manager.scale_loss(loss)
        scaled_loss.backward()

        # 5. Unscale gradients (required before clipping with AMP)
        amp_manager.unscale(optimizer)

        # 6. Gradient clipping
        grad_norm = gradient_clipper.clip(model)

        # 7. Optimizer step (AMP-aware)
        amp_manager.optimizer_step(optimizer)

        # 8. Scheduler step
        if scheduler:
            scheduler.step()

        # 9. Update AMP scaler
        amp_manager.update_scaler()

        # 10. Metric collection
        batch_loss = loss.item()
        batch_tokens = loss_out["num_tokens"].item()
        total_loss += batch_loss
        total_tokens += batch_tokens
        num_batches += 1
        global_step += 1

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log step metrics
        training_logger.log_step(
            step=step, epoch=epoch, train_loss=batch_loss,
            learning_rate=current_lr, gradient_norm=grad_norm,
            num_tokens=batch_tokens, global_step=global_step,
        )

        callback_manager.fire(
            "on_step_end", step=step, global_step=global_step,
            loss=batch_loss, gradient_norm=grad_norm,
        )

    avg_loss = total_loss / max(num_batches, 1)
    return {
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "global_step": global_step,
        "num_batches": num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: LanguageModelingLoss,
    amp_manager: MixedPrecisionManager,
    device: torch.device,
) -> Dict[str, float]:
    """
    Runs a validation pass over the entire validation set.

    Returns:
        Dictionary with: avg_loss, avg_perplexity, total_tokens.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        src_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        with amp_manager.autocast_context():
            logits = model(
                src_input_ids=input_ids,
                tgt_input_ids=input_ids,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=src_padding_mask,
            )
            loss_out = loss_fn(logits, labels)

        total_loss += loss_out["loss"].item()
        total_tokens += loss_out["num_tokens"].item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_perplexity = torch.exp(torch.tensor(min(avg_loss, 100.0))).item()

    return {
        "avg_loss": avg_loss,
        "avg_perplexity": avg_perplexity,
        "total_tokens": total_tokens,
    }
