# CrackLawLM Evaluation Engine
# =============================
# Comprehensive model evaluation framework for the CrackLaw Transformer.
# Plugs into the Training Engine via the callback system.

from src.llm.evaluation.config import EvaluationConfig
from src.llm.evaluation.evaluation_engine import EvaluationEngine

__all__ = [
    "EvaluationConfig",
    "EvaluationEngine",
]
