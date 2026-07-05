"""
CrackLawLM Core Metrics
=========================
Aggregated metric container and utilities used across the evaluation engine.

Mathematical Definitions:
─────────────────────────
  Loss (Cross-Entropy):
    L = -(1/N) Σ log P(y_t | y_{<t}, x)
    where N = number of non-ignored tokens

  Perplexity:
    PPL = exp(L)
    Measures how "surprised" the model is. Lower = better.

  Token Accuracy (Top-1):
    Acc@1 = (# correct predictions) / (# non-ignored tokens)
    Measures exact next-token prediction rate.

  Top-K Accuracy:
    Acc@K = (# tokens where true label ∈ top-K predictions) / N
    Relaxed accuracy — the correct token is in the top K guesses.

  Prediction Confidence:
    Conf = (1/N) Σ max_j P(j | y_{<t}, x)
    Average probability assigned to the most-likely token.

  Average Sequence Loss:
    SeqLoss = (1/B) Σ_b [ (1/T_b) Σ_t L_{b,t} ]
    Per-sequence average rather than per-token average.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics from a single evaluation run."""

    # Loss & Perplexity
    avg_loss: float = 0.0
    perplexity: float = 0.0

    # Accuracy
    top_1_accuracy: float = 0.0
    top_5_accuracy: float = 0.0

    # Confidence
    avg_confidence: float = 0.0

    # Sequence-level
    avg_sequence_loss: float = 0.0

    # Token counts
    total_tokens: int = 0
    total_sequences: int = 0

    # Metadata
    epoch: int = 0
    global_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes all metrics to a dictionary."""
        return {
            "avg_loss": round(self.avg_loss, 6),
            "perplexity": round(self.perplexity, 4),
            "top_1_accuracy": round(self.top_1_accuracy, 6),
            "top_5_accuracy": round(self.top_5_accuracy, 6),
            "avg_confidence": round(self.avg_confidence, 6),
            "avg_sequence_loss": round(self.avg_sequence_loss, 6),
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

    def summary_str(self) -> str:
        """Returns a formatted one-line summary string."""
        return (
            f"loss={self.avg_loss:.4f} | ppl={self.perplexity:.2f} | "
            f"acc@1={self.top_1_accuracy:.4f} | acc@5={self.top_5_accuracy:.4f} | "
            f"conf={self.avg_confidence:.4f} | seq_loss={self.avg_sequence_loss:.4f}"
        )
