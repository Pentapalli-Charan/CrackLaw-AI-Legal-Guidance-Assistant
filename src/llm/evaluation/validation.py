"""
CrackLawLM Validation Loop
=============================
Extended validation loop that computes all evaluation metrics over a DataLoader.
Reuses the existing model forward pass without modification.
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.llm.evaluation.config import EvaluationConfig
from src.llm.evaluation.metrics import EvaluationMetrics
from src.llm.evaluation.perplexity import PerplexityCalculator
from src.llm.evaluation.accuracy import AccuracyCalculator

logger = logging.getLogger("CrackLaw.LLM.Evaluation.Validation")


class ValidationRunner:
    """
    Runs a full validation pass computing all evaluation metrics.

    Collects per-batch metrics and aggregates them into a single
    EvaluationMetrics result.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.perplexity_calc = PerplexityCalculator(config.ignore_index)
        self.accuracy_calc = AccuracyCalculator(
            config.ignore_index, config.top_k_values
        )

    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int = 0,
        global_step: int = 0,
    ) -> EvaluationMetrics:
        """
        Runs full evaluation on the dataloader.

        Args:
            model:       The CrackLawTransformer model.
            dataloader:  Validation DataLoader.
            device:      Torch device.
            epoch:       Current training epoch (for metadata).
            global_step: Current global step (for metadata).

        Returns:
            EvaluationMetrics with all computed values.
        """
        model.eval()

        # Accumulators
        total_loss = 0.0
        total_ppl_tokens = 0
        total_acc_tokens = 0
        total_top1_correct = 0.0
        total_top5_correct = 0.0
        total_confidence = 0.0
        seq_losses = []
        num_batches = 0
        total_sequences = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            src_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            logits = model(
                src_input_ids=input_ids,
                tgt_input_ids=input_ids,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=src_padding_mask,
            )

            # Perplexity
            ppl_result = self.perplexity_calc.compute(logits, labels)
            batch_tokens = ppl_result["total_tokens"]
            if batch_tokens > 0:
                total_loss += ppl_result["avg_loss"] * batch_tokens
                total_ppl_tokens += batch_tokens

            # Per-sequence loss
            seq_result = self.perplexity_calc.compute_per_sequence(logits, labels)
            if seq_result["avg_sequence_loss"] > 0:
                seq_losses.append(seq_result["avg_sequence_loss"])

            # Accuracy
            acc_result = self.accuracy_calc.compute(logits, labels)
            acc_tokens = acc_result["total_tokens"]
            if acc_tokens > 0:
                total_top1_correct += acc_result["top_1_accuracy"] * acc_tokens
                total_top5_correct += acc_result["top_5_accuracy"] * acc_tokens
                total_confidence += acc_result["avg_confidence"] * acc_tokens
                total_acc_tokens += acc_tokens

            total_sequences += input_ids.size(0)
            num_batches += 1

        # Aggregate
        avg_loss = total_loss / max(total_ppl_tokens, 1)
        perplexity = torch.exp(torch.tensor(min(avg_loss, 100.0))).item()
        top1_acc = total_top1_correct / max(total_acc_tokens, 1)
        top5_acc = total_top5_correct / max(total_acc_tokens, 1)
        avg_conf = total_confidence / max(total_acc_tokens, 1)
        avg_seq_loss = sum(seq_losses) / max(len(seq_losses), 1) if seq_losses else 0.0

        metrics = EvaluationMetrics(
            avg_loss=avg_loss,
            perplexity=perplexity,
            top_1_accuracy=top1_acc,
            top_5_accuracy=top5_acc,
            avg_confidence=avg_conf,
            avg_sequence_loss=avg_seq_loss,
            total_tokens=total_ppl_tokens,
            total_sequences=total_sequences,
            epoch=epoch,
            global_step=global_step,
        )

        logger.info(f"Evaluation [epoch={epoch}]: {metrics.summary_str()}")
        return metrics
