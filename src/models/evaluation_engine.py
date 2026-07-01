import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from src.models.metrics import MetricCalculator
from src.models.exceptions import ValidationError

logger = logging.getLogger("CrackLaw.Models.EvaluationEngine")

class EvaluationEngine:
    """Compiles model test metrics, creates training loss history curves, and generates reports."""

    @staticmethod
    def generate_evaluation_report(
        model_name: str,
        framework: str,
        metrics: Dict[str, Any],
        history: Optional[Dict[str, List[float]]] = None,
        export_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Constructs a comprehensive model evaluation report with metric sets and training curves."""
        logger.info("Generating evaluation report for '%s' (%s)", model_name, framework)

        report = {
            "model_name": model_name,
            "framework": framework,
            "metrics": metrics,
            "loss_curves": {
                "epochs": list(range(1, len(history["loss"]) + 1)) if history and "loss" in history else [],
                "train_loss": history["loss"] if history and "loss" in history else [],
                "val_loss": history["val_loss"] if history and "val_loss" in history else []
            }
        }

        # Format Confusion Matrix if it exists in metrics
        if "confusion_matrix" in metrics:
            cm = np.asarray(metrics["confusion_matrix"])
            report["confusion_matrix_summary"] = {
                "dimensions": cm.shape,
                "values": cm.tolist()
            }

        # If export path is requested, write to disk JSON file
        if export_path:
            try:
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info("Saved evaluation report to: %s", export_path)
            except Exception as e:
                logger.warning("Failed to save evaluation report file: %s", str(e))

        return report

    @staticmethod
    def print_markdown_summary(report: Dict[str, Any]) -> str:
        """Formats the evaluation report details into a readable markdown string."""
        metrics = report["metrics"]
        
        md = []
        md.append(f"# Model Evaluation Summary: `{report['model_name']}`")
        md.append(f"- **Framework**: {report['framework']}")
        md.append("")
        md.append("## Core Performance Metrics")
        for k, v in metrics.items():
            if k == "confusion_matrix":
                continue
            # Format float metrics beautifully
            if isinstance(v, float):
                md.append(f"- **{k.title()}**: {v:.4f}")
            else:
                md.append(f"- **{k.title()}**: {v}")

        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            md.append("")
            md.append("## Confusion Matrix")
            md.append("```")
            for row in cm:
                md.append("  " + "  ".join(f"{val:4d}" for val in row))
            md.append("```")

        loss_curves = report.get("loss_curves", {})
        if loss_curves.get("train_loss"):
            md.append("")
            md.append("## Loss Trajectory per Epoch")
            md.append("| Epoch | Training Loss | Validation Loss |")
            md.append("| :--- | :--- | :--- |")
            epochs = loss_curves.get("epochs", [])
            train = loss_curves.get("train_loss", [])
            val = loss_curves.get("val_loss", [])
            for i in range(len(epochs)):
                val_str = f"{val[i]:.4f}" if i < len(val) else "N/A"
                md.append(f"| {epochs[i]} | {train[i]:.4f} | {val_str} |")

        return "\n".join(md)
