"""
CrackLawLM Report Generator
==============================
Generates comprehensive evaluation reports with training curves,
loss/perplexity/accuracy history, and checkpoint comparisons.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.llm.evaluation.config import EvaluationConfig
from src.llm.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger("CrackLaw.LLM.Evaluation.Report")


class ReportGenerator:
    """
    Generates structured evaluation reports in JSON and Markdown.

    Reports include:
      - Metric history across epochs (loss, perplexity, accuracy)
      - Training curves data (plottable JSON arrays)
      - Checkpoint comparison tables
      - Generation sample evolution
      - Benchmark results
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics_history: List[Dict[str, Any]] = []
        self.benchmark_history: List[Dict[str, Any]] = []
        self.history_file = os.path.join(config.output_dir, "metrics_history.jsonl")

    def record_metrics(self, metrics: EvaluationMetrics):
        """Records an evaluation snapshot for history tracking."""
        entry = metrics.to_dict()
        entry["timestamp"] = datetime.now().isoformat()
        self.metrics_history.append(entry)

        # Append to JSONL file
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def record_benchmark(self, benchmark: Dict[str, Any], epoch: int):
        """Records benchmark results."""
        benchmark["epoch"] = epoch
        self.benchmark_history.append(benchmark)

    def generate_report(
        self,
        generation_comparisons: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generates a comprehensive Markdown evaluation report.

        Returns:
            Path to the generated report file.
        """
        report_path = os.path.join(
            self.config.reports_dir,
            f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        lines = []
        lines.append("# CrackLawLM Evaluation Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # ─── Metrics Summary ───
        lines.append("## Metrics Summary\n")
        if self.metrics_history:
            latest = self.metrics_history[-1]
            lines.append("### Latest Evaluation\n")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Loss | {latest['avg_loss']:.6f} |")
            lines.append(f"| Perplexity | {latest['perplexity']:.4f} |")
            lines.append(f"| Top-1 Accuracy | {latest['top_1_accuracy']:.4f} |")
            lines.append(f"| Top-5 Accuracy | {latest['top_5_accuracy']:.4f} |")
            lines.append(f"| Avg Confidence | {latest['avg_confidence']:.4f} |")
            lines.append(f"| Avg Sequence Loss | {latest['avg_sequence_loss']:.4f} |")
            lines.append(f"| Total Tokens | {latest['total_tokens']} |")
            lines.append(f"| Epoch | {latest['epoch']} |")
            lines.append("")

        # ─── Training Curves Data ───
        lines.append("## Training Curves\n")
        lines.append("### Loss History\n")
        lines.append("| Epoch | Loss | Perplexity |")
        lines.append("|-------|------|------------|")
        for entry in self.metrics_history:
            lines.append(
                f"| {entry['epoch']} | {entry['avg_loss']:.6f} | "
                f"{entry['perplexity']:.4f} |"
            )
        lines.append("")

        lines.append("### Accuracy History\n")
        lines.append("| Epoch | Top-1 | Top-5 | Confidence |")
        lines.append("|-------|-------|-------|------------|")
        for entry in self.metrics_history:
            lines.append(
                f"| {entry['epoch']} | {entry['top_1_accuracy']:.4f} | "
                f"{entry['top_5_accuracy']:.4f} | "
                f"{entry['avg_confidence']:.4f} |"
            )
        lines.append("")

        # ─── Checkpoint Comparison ───
        if len(self.metrics_history) >= 2:
            lines.append("## Checkpoint Comparison\n")
            first = self.metrics_history[0]
            last = self.metrics_history[-1]
            lines.append("| Metric | First | Latest | Delta |")
            lines.append("|--------|-------|--------|-------|")
            for key in ["avg_loss", "perplexity", "top_1_accuracy", "top_5_accuracy"]:
                delta = last[key] - first[key]
                arrow = "↓" if delta < 0 and key in ["avg_loss", "perplexity"] else "↑"
                if key in ["top_1_accuracy", "top_5_accuracy"]:
                    arrow = "↑" if delta > 0 else "↓"
                lines.append(
                    f"| {key} | {first[key]:.4f} | {last[key]:.4f} | "
                    f"{arrow} {abs(delta):.4f} |"
                )
            lines.append("")

        # ─── Generation Samples ───
        if generation_comparisons:
            lines.append("## Generation Evolution\n")
            for comp in generation_comparisons:
                lines.append(f"### Prompt: \"{comp['prompt'][:60]}...\"\n")
                for gen in comp.get("generations_over_time", []):
                    lines.append(f"**Epoch {gen['epoch']}** ({gen['num_tokens']} tokens):")
                    lines.append(f"> {gen['text'][:200]}\n")
                lines.append("")

        # ─── Benchmark Results ───
        if self.benchmark_history:
            lines.append("## Benchmark Results\n")
            latest_bench = self.benchmark_history[-1]
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            for k, v in latest_bench.items():
                if k != "epoch":
                    lines.append(f"| {k} | {v} |")
            lines.append("")

        # Write report
        report_content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # Also save raw JSON for programmatic access
        json_path = report_path.replace(".md", ".json")
        json_data = {
            "metrics_history": self.metrics_history,
            "benchmark_history": self.benchmark_history,
            "generation_comparisons": generation_comparisons or [],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, default=str)

        logger.info(f"Evaluation report generated: {report_path}")
        return report_path
