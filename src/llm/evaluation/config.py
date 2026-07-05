"""
CrackLawLM Evaluation Configuration
=====================================
Central dataclass for all evaluation hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import List

from src.config import PROJECT_ROOT


@dataclass
class EvaluationConfig:
    """Complete configuration for the CrackLawLM Evaluation Engine."""

    # ──────────────────────── Output ────────────────────────────────────
    output_dir: str = os.path.join(PROJECT_ROOT, "logs", "evaluation")
    reports_dir: str = os.path.join(PROJECT_ROOT, "logs", "evaluation", "reports")

    # ──────────────────────── Metrics ───────────────────────────────────
    ignore_index: int = -100  # Token ID to ignore in metric computation

    # ──────────────────────── Accuracy ──────────────────────────────────
    top_k_values: List[int] = field(default_factory=lambda: [1, 5])

    # ──────────────────────── Text Generation ──────────────────────────
    generation_enabled: bool = True
    max_generation_tokens: int = 128
    num_generation_samples: int = 3
    generation_prompts: List[str] = field(default_factory=lambda: [
        "The Indian Penal Code Section",
        "Under the Constitution of India",
        "The court hereby orders that",
    ])

    # ──────────────────────── Benchmarking ──────────────────────────────
    benchmark_enabled: bool = True
    benchmark_num_batches: int = 10
    benchmark_warmup_batches: int = 2

    # ──────────────────────── Report ────────────────────────────────────
    generate_report: bool = True

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
