# CrackLawLM Training Engine
# ===========================
# Production-grade training framework for the CrackLaw Transformer.
# Built with native PyTorch — no external training libraries.

from src.llm.training.config import TrainingConfig
from src.llm.training.trainer import CrackLawTrainer

__all__ = [
    "TrainingConfig",
    "CrackLawTrainer",
]
