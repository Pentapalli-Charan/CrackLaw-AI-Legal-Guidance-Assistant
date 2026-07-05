"""
CrackLawLM Perplexity Computation
====================================
Computes perplexity from model logits and labels.

Mathematical Definition:
  PPL = exp( -(1/N) Σ_{t} log P(y_t | y_{<t}, x) )

  where:
    N     = number of non-ignored tokens
    y_t   = ground-truth token at position t
    P(·)  = model's predicted probability distribution

Interpretation:
  - PPL=1 means the model perfectly predicts every token.
  - PPL=V (vocab size) means the model is randomly guessing.
  - Lower is always better.
  - For a vocab_size=5000 model, PPL < 100 indicates meaningful learning.
"""

import torch
import torch.nn.functional as F
from typing import Dict


class PerplexityCalculator:
    """Computes token-level and sequence-level perplexity."""

    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes perplexity from logits and labels.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)

        Returns:
            Dictionary with:
              - perplexity: exp(mean negative log-likelihood)
              - avg_loss: mean cross-entropy loss
              - total_tokens: count of non-ignored tokens
        """
        batch_size, seq_len, vocab_size = logits.size()

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Compute per-token log-probabilities
        log_probs = F.log_softmax(logits_flat, dim=-1)

        # Mask for non-ignored tokens
        valid_mask = labels_flat != self.ignore_index
        num_tokens = valid_mask.sum().item()

        if num_tokens == 0:
            return {"perplexity": 0.0, "avg_loss": 0.0, "total_tokens": 0}

        # Gather log-probs for the correct tokens
        valid_labels = labels_flat[valid_mask]
        valid_log_probs = log_probs[valid_mask]
        token_log_probs = valid_log_probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

        # Negative log-likelihood
        nll = -token_log_probs.mean().item()

        # Perplexity = exp(NLL), clamped for safety
        perplexity = torch.exp(torch.tensor(min(nll, 100.0))).item()

        return {
            "perplexity": perplexity,
            "avg_loss": nll,
            "total_tokens": num_tokens,
        }

    def compute_per_sequence(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes per-sequence perplexity (averaged across sequences, not tokens).

        Returns:
            Dictionary with avg_sequence_loss and per_sequence_perplexity.
        """
        batch_size, seq_len, vocab_size = logits.size()
        log_probs = F.log_softmax(logits, dim=-1)

        seq_losses = []
        for b in range(batch_size):
            valid_mask = labels[b] != self.ignore_index
            if valid_mask.sum() == 0:
                continue
            valid_labels = labels[b][valid_mask]
            valid_lp = log_probs[b][valid_mask]
            token_lp = valid_lp.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
            seq_loss = -token_lp.mean().item()
            seq_losses.append(seq_loss)

        if not seq_losses:
            return {"avg_sequence_loss": 0.0, "per_sequence_perplexity": 0.0}

        avg_seq_loss = sum(seq_losses) / len(seq_losses)
        ppl = torch.exp(torch.tensor(min(avg_seq_loss, 100.0))).item()

        return {
            "avg_sequence_loss": avg_seq_loss,
            "per_sequence_perplexity": ppl,
        }
