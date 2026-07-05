"""
CrackLawLM Loss Module
=======================
Cross-entropy loss for autoregressive language modeling with PAD token ignoring
and optional label smoothing.
"""

import torch
import torch.nn as nn
from typing import Dict

from src.llm.training.config import TrainingConfig


class LanguageModelingLoss(nn.Module):
    """
    Cross-Entropy loss tailored for autoregressive language modeling.

    Key behaviors:
      - Ignores PAD tokens via ignore_index (default: -100).
      - Supports label shifting: model predicts token at position t+1 given input up to t.
      - Optional label smoothing for regularization.
      - Reports per-token loss and perplexity for monitoring.
    """

    def __init__(self, config: TrainingConfig):
        super(LanguageModelingLoss, self).__init__()
        self.config = config

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the autoregressive language modeling loss.

        The logits and labels must already be aligned so that
        logits[:, t, :] predicts labels[:, t]. The Dataset/SequenceBuilder
        already handles the shift (input = [BOS, t1, ...], label = [t1, ..., EOS]).

        Args:
            logits: Model output logits.  Shape: (batch_size, seq_len, vocab_size)
            labels: Ground-truth token IDs. Shape: (batch_size, seq_len)
                    Positions to ignore should be set to `config.ignore_index` (-100).

        Returns:
            Dictionary containing:
              - loss: scalar tensor (mean cross-entropy over non-ignored tokens)
              - num_tokens: number of tokens that contributed to the loss
              - perplexity: exp(loss), clamped for numerical safety
        """
        batch_size, seq_len, vocab_size = logits.size()

        # Reshape for CrossEntropyLoss: (N, C) and (N,)
        logits_flat = logits.reshape(-1, vocab_size)       # (batch * seq, vocab)
        labels_flat = labels.reshape(-1)                    # (batch * seq,)

        loss = self.criterion(logits_flat, labels_flat)

        # Count tokens that actually contributed (not ignored)
        num_tokens = (labels_flat != self.config.ignore_index).sum()

        # Perplexity = exp(loss), clamped to avoid inf on early noisy batches
        perplexity = torch.exp(torch.clamp(loss, max=100.0))

        return {
            "loss": loss,
            "num_tokens": num_tokens,
            "perplexity": perplexity,
        }
