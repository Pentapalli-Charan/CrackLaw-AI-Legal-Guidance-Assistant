"""
CrackLawLM Accuracy Metrics
==============================
Computes Top-1, Top-K accuracy and prediction confidence.

Mathematical Definitions:
─────────────────────────
  Top-1 Accuracy:
    Acc@1 = (1/N) Σ_t 𝟙[argmax P(·|y_{<t}) == y_t]
    The fraction of tokens where the model's highest-probability
    prediction matches the ground truth.

  Top-K Accuracy:
    Acc@K = (1/N) Σ_t 𝟙[y_t ∈ topK(P(·|y_{<t}))]
    The fraction of tokens where the ground truth is among the
    model's K most likely predictions. Acc@5 is standard in NLP.

  Prediction Confidence:
    Conf = (1/N) Σ_t max_j P(j | y_{<t})
    The average probability the model assigns to its top prediction.
    High confidence with low accuracy suggests overconfidence.
    Low confidence with decent accuracy suggests calibration issues.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List


class AccuracyCalculator:
    """Computes Top-1, Top-K accuracy and prediction confidence."""

    def __init__(self, ignore_index: int = -100, top_k_values: List[int] = None):
        self.ignore_index = ignore_index
        self.top_k_values = top_k_values or [1, 5]

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes all accuracy metrics from logits and labels.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)

        Returns:
            Dictionary with top_1_accuracy, top_5_accuracy, avg_confidence,
            and total_tokens.
        """
        batch_size, seq_len, vocab_size = logits.size()

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Valid (non-ignored) token mask
        valid_mask = labels_flat != self.ignore_index
        num_tokens = valid_mask.sum().item()

        if num_tokens == 0:
            result = {"total_tokens": 0, "avg_confidence": 0.0}
            for k in self.top_k_values:
                result[f"top_{k}_accuracy"] = 0.0
            return result

        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]

        # Probabilities for confidence
        probs = F.softmax(valid_logits, dim=-1)

        # Top-K accuracy for each k
        result = {"total_tokens": num_tokens}
        for k in self.top_k_values:
            actual_k = min(k, vocab_size)
            _, top_k_preds = valid_logits.topk(actual_k, dim=-1)
            matches = (top_k_preds == valid_labels.unsqueeze(1)).any(dim=1)
            result[f"top_{k}_accuracy"] = matches.float().mean().item()

        # Prediction confidence: average max probability
        max_probs, _ = probs.max(dim=-1)
        result["avg_confidence"] = max_probs.mean().item()

        return result
