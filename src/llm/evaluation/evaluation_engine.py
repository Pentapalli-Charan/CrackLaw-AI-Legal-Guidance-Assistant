"""
CrackLawLM Evaluation Engine
===============================
Top-level orchestrator that assembles all evaluation components and integrates
with the Training Engine via the callback system.

Usage as standalone:
    engine = EvaluationEngine(config, tokenizer)
    metrics = engine.evaluate(model, val_loader, device, epoch=5)

Usage as training callback:
    callback = engine.as_callback()
    trainer = CrackLawTrainer(..., callbacks=[callback])
"""

import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.llm.evaluation.config import EvaluationConfig
from src.llm.evaluation.metrics import EvaluationMetrics
from src.llm.evaluation.validation import ValidationRunner
from src.llm.evaluation.benchmark import InferenceBenchmark
from src.llm.evaluation.text_generation import TextGenerationEvaluator
from src.llm.evaluation.report_generator import ReportGenerator
from src.llm.training.callbacks import TrainingCallback

logger = logging.getLogger("CrackLaw.LLM.Evaluation.Engine")


class EvaluationEngine:
    """
    Production-grade evaluation engine for CrackLawLM.

    Orchestrates:
      - Full metric evaluation (loss, perplexity, accuracy, confidence)
      - Text generation from legal prompts
      - Inference benchmarking
      - Report generation

    Can be used standalone or plugged into the Trainer as a callback.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        tokenizer=None,
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Sub-components
        self.validation_runner = ValidationRunner(config)
        self.benchmark = InferenceBenchmark(config)
        self.text_generator = TextGenerationEvaluator(config)
        self.report_generator = ReportGenerator(config)

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int = 0,
        global_step: int = 0,
    ) -> EvaluationMetrics:
        """
        Runs a full evaluation cycle: metrics + generation + benchmark.

        Args:
            model:       The CrackLawTransformer.
            dataloader:  Validation DataLoader.
            device:      Torch device.
            epoch:       Current epoch.
            global_step: Current global step.

        Returns:
            EvaluationMetrics with all computed values.
        """
        logger.info(f"{'='*50}")
        logger.info(f"  Evaluation Engine — Epoch {epoch}")
        logger.info(f"{'='*50}")

        # 1. Compute all metrics
        metrics = self.validation_runner.run(
            model, dataloader, device, epoch, global_step
        )
        self.report_generator.record_metrics(metrics)

        # 2. Generate sample text
        if self.config.generation_enabled and self.tokenizer:
            logger.info("  Generating sample text...")
            self.text_generator.generate_samples(
                model, self.tokenizer, device, epoch, global_step
            )

        # 3. Run inference benchmark
        if self.config.benchmark_enabled:
            logger.info("  Running inference benchmark...")
            bench_result = self.benchmark.run(model, dataloader, device)
            self.report_generator.record_benchmark(bench_result, epoch)

        return metrics

    def generate_final_report(self) -> str:
        """
        Generates the final comprehensive evaluation report.
        Should be called at the end of training.

        Returns:
            Path to the generated report file.
        """
        comparisons = self.text_generator.compare_checkpoints()
        report_path = self.report_generator.generate_report(comparisons)
        return report_path

    def as_callback(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ) -> "EvaluationCallback":
        """
        Creates a TrainingCallback that runs evaluation automatically
        after each epoch's validation phase.

        Args:
            model:      The model being trained.
            val_loader: Validation DataLoader.
            device:     Torch device.

        Returns:
            A TrainingCallback instance.
        """
        return EvaluationCallback(
            engine=self,
            model=model,
            val_loader=val_loader,
            device=device,
        )


class EvaluationCallback(TrainingCallback):
    """
    Training callback that runs the EvaluationEngine after each epoch.
    Plugs directly into the CrackLawTrainer's callback system.
    """

    def __init__(
        self,
        engine: EvaluationEngine,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.engine = engine
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self._current_global_step = 0

    def on_epoch_end(self, epoch: int, train_loss: float,
                     val_loss: float, **kwargs):
        """Runs full evaluation after each epoch."""
        self.engine.evaluate(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            epoch=epoch,
            global_step=self._current_global_step,
        )

    def on_step_end(self, step: int, global_step: int,
                    loss: float, **kwargs):
        """Tracks global step for metadata."""
        self._current_global_step = global_step

    def on_training_end(self, **kwargs):
        """Generates the final report when training completes."""
        report_path = self.engine.generate_final_report()
        logger.info(f"Final evaluation report: {report_path}")
