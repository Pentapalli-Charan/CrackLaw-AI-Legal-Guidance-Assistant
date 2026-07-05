"""
CrackLawLM Benchmark Module
==============================
Measures inference performance: latency, tokens/sec, memory, GPU utilization.
"""

import time
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.llm.evaluation.config import EvaluationConfig

logger = logging.getLogger("CrackLaw.LLM.Evaluation.Benchmark")


class InferenceBenchmark:
    """
    Benchmarks model inference performance.

    Measures:
      - Inference latency (ms/batch, ms/token)
      - Throughput (tokens/second)
      - Memory usage (GPU allocated/reserved, or CPU process memory)
      - GPU utilization percentage (if available)
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.cuda_available = torch.cuda.is_available()

    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Runs inference benchmark over a limited number of batches.

        Args:
            model:      The model to benchmark.
            dataloader: DataLoader providing input batches.
            device:     Torch device.

        Returns:
            Dictionary with latency, throughput, and memory metrics.
        """
        model.eval()

        num_batches = self.config.benchmark_num_batches
        warmup_batches = self.config.benchmark_warmup_batches
        total_batches = num_batches + warmup_batches

        batch_times = []
        total_tokens = 0
        batches_processed = 0

        # Clear CUDA cache before benchmarking
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        for i, batch in enumerate(dataloader):
            if i >= total_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            src_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Synchronize before timing (CUDA)
            if self.cuda_available:
                torch.cuda.synchronize(device)

            start = time.perf_counter()

            _ = model(
                src_input_ids=input_ids,
                tgt_input_ids=input_ids,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=src_padding_mask,
            )

            if self.cuda_available:
                torch.cuda.synchronize(device)

            elapsed = time.perf_counter() - start

            # Skip warmup batches
            if i >= warmup_batches:
                batch_times.append(elapsed)
                tokens_in_batch = (attention_mask.sum()).item()
                total_tokens += tokens_in_batch
                batches_processed += 1

        # Compute statistics
        if not batch_times:
            return self._empty_result()

        total_time = sum(batch_times)
        avg_batch_time = total_time / len(batch_times)
        tokens_per_sec = total_tokens / max(total_time, 1e-9)
        avg_token_time = (total_time / max(total_tokens, 1)) * 1000  # ms

        result = {
            "avg_batch_latency_ms": avg_batch_time * 1000,
            "avg_token_latency_ms": avg_token_time,
            "tokens_per_second": tokens_per_sec,
            "total_tokens": total_tokens,
            "batches_processed": batches_processed,
            "total_inference_time_sec": total_time,
            "device": str(device),
        }

        # Memory metrics
        if self.cuda_available and device.type == "cuda":
            result["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
            result["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2)
            result["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
            # GPU utilization (not available without nvidia-smi, report what we can)
            result["gpu_name"] = torch.cuda.get_device_name(device)
        else:
            import os
            try:
                import psutil
                process = psutil.Process(os.getpid())
                result["cpu_memory_rss_mb"] = process.memory_info().rss / (1024**2)
            except ImportError:
                result["cpu_memory_rss_mb"] = None

        logger.info(
            f"Benchmark: {tokens_per_sec:.0f} tok/s | "
            f"batch={avg_batch_time*1000:.1f}ms | "
            f"token={avg_token_time:.3f}ms | "
            f"device={device}"
        )

        return result

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "avg_batch_latency_ms": 0.0,
            "avg_token_latency_ms": 0.0,
            "tokens_per_second": 0.0,
            "total_tokens": 0,
            "batches_processed": 0,
            "total_inference_time_sec": 0.0,
            "device": "unknown",
        }
