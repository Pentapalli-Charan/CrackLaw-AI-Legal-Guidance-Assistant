"""
CrackLawLM Evaluation Engine — Comprehensive Unit Tests
=========================================================
Verifies: metric correctness, validation loop, perplexity calculation,
accuracy calculations, benchmark execution, report generation,
text generation, and callback integration.
"""

import os
import sys
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Ensure project root is on sys.path ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.evaluation.config import EvaluationConfig
from src.llm.evaluation.metrics import EvaluationMetrics
from src.llm.evaluation.perplexity import PerplexityCalculator
from src.llm.evaluation.accuracy import AccuracyCalculator
from src.llm.evaluation.validation import ValidationRunner
from src.llm.evaluation.benchmark import InferenceBenchmark
from src.llm.evaluation.sampler import PromptSampler
from src.llm.evaluation.text_generation import TextGenerationEvaluator
from src.llm.evaluation.report_generator import ReportGenerator
from src.llm.evaluation.evaluation_engine import EvaluationEngine, EvaluationCallback
from src.llm.training.callbacks import CallbackManager


# ─── Test Helpers ───

def _create_tiny_model():
    """Creates a minimal transformer for fast testing."""
    config = TransformerConfig(
        vocab_size=100, max_seq_len=32, d_model=64,
        num_heads=4, d_ff=128, num_encoder_layers=1,
        num_decoder_layers=1, dropout_rate=0.0,
    )
    return CrackLawTransformer(config), config


def _create_dummy_dataloader(vocab_size=100, seq_len=16, batch_size=4, num_batches=3):
    """Creates a DataLoader with random token data mimicking collator output."""
    all_input_ids = []
    all_labels = []
    all_masks = []
    for _ in range(num_batches * batch_size):
        ids = torch.randint(4, vocab_size, (seq_len,))
        labels = torch.randint(4, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)
        all_input_ids.append(ids)
        all_labels.append(labels)
        all_masks.append(mask)

    dataset = TensorDataset(
        torch.stack(all_input_ids),
        torch.stack(all_labels),
        torch.stack(all_masks),
    )

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        masks = torch.stack([b[2] for b in batch])
        return {"input_ids": ids, "labels": labels, "attention_mask": masks}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


class _MockTokenizer:
    """Minimal tokenizer mock for text generation tests."""

    class _SpecialTokens:
        def get_id(self, token):
            mapping = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
            return mapping.get(token, 1)

    class _Config:
        bos_token = "<BOS>"
        eos_token = "<EOS>"
        pad_token = "<PAD>"
        unk_token = "<UNK>"

    def __init__(self):
        self.special_tokens = self._SpecialTokens()
        self.config = self._Config()

    def encode(self, text):
        return [10, 20, 30, 40, 50]

    def decode(self, token_ids):
        return f"[decoded {len(token_ids)} tokens]"


# ─── Tests ───

class TestEvaluationConfig(unittest.TestCase):
    """Tests for EvaluationConfig."""

    def test_default_values(self):
        config = EvaluationConfig()
        self.assertTrue(config.generation_enabled)
        self.assertTrue(config.benchmark_enabled)
        self.assertTrue(config.generate_report)
        self.assertEqual(config.ignore_index, -100)
        self.assertEqual(config.top_k_values, [1, 5])

    def test_directories_created(self):
        config = EvaluationConfig()
        self.assertTrue(os.path.isdir(config.output_dir))
        self.assertTrue(os.path.isdir(config.reports_dir))

    def test_custom_prompts(self):
        config = EvaluationConfig(
            generation_prompts=["Test prompt 1", "Test prompt 2"]
        )
        self.assertEqual(len(config.generation_prompts), 2)


class TestEvaluationMetrics(unittest.TestCase):
    """Tests for EvaluationMetrics container."""

    def test_to_dict(self):
        m = EvaluationMetrics(avg_loss=1.5, perplexity=4.48, top_1_accuracy=0.25)
        d = m.to_dict()
        self.assertIn("avg_loss", d)
        self.assertIn("perplexity", d)
        self.assertIn("top_1_accuracy", d)
        self.assertEqual(d["avg_loss"], 1.5)

    def test_summary_str(self):
        m = EvaluationMetrics(avg_loss=1.5, perplexity=4.48)
        s = m.summary_str()
        self.assertIn("loss=1.5000", s)
        self.assertIn("ppl=4.48", s)

    def test_default_values(self):
        m = EvaluationMetrics()
        self.assertEqual(m.avg_loss, 0.0)
        self.assertEqual(m.total_tokens, 0)


class TestPerplexityCalculator(unittest.TestCase):
    """Tests for perplexity calculation."""

    def setUp(self):
        self.calc = PerplexityCalculator(ignore_index=-100)

    def test_basic_perplexity(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        result = self.calc.compute(logits, labels)
        self.assertIn("perplexity", result)
        self.assertIn("avg_loss", result)
        self.assertGreater(result["perplexity"], 0)
        self.assertGreater(result["total_tokens"], 0)

    def test_perplexity_equals_exp_loss(self):
        """PPL should be exp(loss)."""
        logits = torch.randn(1, 5, 50)
        labels = torch.randint(0, 50, (1, 5))
        result = self.calc.compute(logits, labels)
        expected_ppl = torch.exp(torch.tensor(result["avg_loss"])).item()
        self.assertAlmostEqual(result["perplexity"], expected_ppl, places=2)

    def test_ignored_tokens_excluded(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.full((2, 10), -100, dtype=torch.long)
        labels[0, 0] = 5  # Only 1 valid token
        result = self.calc.compute(logits, labels)
        self.assertEqual(result["total_tokens"], 1)

    def test_all_ignored(self):
        logits = torch.randn(1, 5, 100)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        result = self.calc.compute(logits, labels)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["perplexity"], 0.0)

    def test_per_sequence_perplexity(self):
        logits = torch.randn(3, 8, 50)
        labels = torch.randint(0, 50, (3, 8))
        result = self.calc.compute_per_sequence(logits, labels)
        self.assertIn("avg_sequence_loss", result)
        self.assertIn("per_sequence_perplexity", result)
        self.assertGreater(result["avg_sequence_loss"], 0)

    def test_perfect_prediction_low_perplexity(self):
        """When logits perfectly predict labels, perplexity should be near 1."""
        vocab_size = 20
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        logits = torch.full((1, 5, vocab_size), -10.0)
        for t in range(5):
            logits[0, t, labels[0, t]] = 10.0  # High logit for correct token
        result = self.calc.compute(logits, labels)
        self.assertLess(result["perplexity"], 2.0)


class TestAccuracyCalculator(unittest.TestCase):
    """Tests for accuracy metrics."""

    def setUp(self):
        self.calc = AccuracyCalculator(ignore_index=-100, top_k_values=[1, 5])

    def test_basic_accuracy(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        result = self.calc.compute(logits, labels)
        self.assertIn("top_1_accuracy", result)
        self.assertIn("top_5_accuracy", result)
        self.assertIn("avg_confidence", result)
        self.assertGreaterEqual(result["top_1_accuracy"], 0.0)
        self.assertLessEqual(result["top_1_accuracy"], 1.0)

    def test_top5_geq_top1(self):
        """Top-5 accuracy must always be >= Top-1 accuracy."""
        logits = torch.randn(4, 16, 50)
        labels = torch.randint(0, 50, (4, 16))
        result = self.calc.compute(logits, labels)
        self.assertGreaterEqual(result["top_5_accuracy"], result["top_1_accuracy"])

    def test_perfect_accuracy(self):
        """When model perfectly predicts, Top-1 should be 1.0."""
        vocab_size = 20
        labels = torch.tensor([[0, 1, 2, 3]])
        logits = torch.full((1, 4, vocab_size), -10.0)
        for t in range(4):
            logits[0, t, labels[0, t]] = 10.0
        result = self.calc.compute(logits, labels)
        self.assertAlmostEqual(result["top_1_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(result["top_5_accuracy"], 1.0, places=4)

    def test_confidence_range(self):
        """Confidence must be between 0 and 1."""
        logits = torch.randn(2, 8, 50)
        labels = torch.randint(0, 50, (2, 8))
        result = self.calc.compute(logits, labels)
        self.assertGreater(result["avg_confidence"], 0.0)
        self.assertLessEqual(result["avg_confidence"], 1.0)

    def test_ignored_tokens(self):
        logits = torch.randn(1, 5, 50)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        result = self.calc.compute(logits, labels)
        self.assertEqual(result["total_tokens"], 0)


class TestValidationRunner(unittest.TestCase):
    """Tests for the full validation loop."""

    def test_validation_produces_metrics(self):
        model, _ = _create_tiny_model()
        config = EvaluationConfig()
        runner = ValidationRunner(config)
        loader = _create_dummy_dataloader()

        metrics = runner.run(model, loader, torch.device("cpu"), epoch=0)
        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertGreater(metrics.avg_loss, 0)
        self.assertGreater(metrics.perplexity, 0)
        self.assertGreater(metrics.total_tokens, 0)
        self.assertGreater(metrics.total_sequences, 0)

    def test_accuracy_populated(self):
        model, _ = _create_tiny_model()
        config = EvaluationConfig()
        runner = ValidationRunner(config)
        loader = _create_dummy_dataloader()

        metrics = runner.run(model, loader, torch.device("cpu"))
        self.assertGreaterEqual(metrics.top_1_accuracy, 0.0)
        self.assertGreaterEqual(metrics.top_5_accuracy, 0.0)
        self.assertGreater(metrics.avg_confidence, 0.0)


class TestBenchmark(unittest.TestCase):
    """Tests for inference benchmarking."""

    def test_benchmark_runs(self):
        model, _ = _create_tiny_model()
        config = EvaluationConfig(benchmark_num_batches=2, benchmark_warmup_batches=1)
        bench = InferenceBenchmark(config)
        loader = _create_dummy_dataloader(num_batches=5)

        result = bench.run(model, loader, torch.device("cpu"))
        self.assertIn("avg_batch_latency_ms", result)
        self.assertIn("tokens_per_second", result)
        self.assertIn("total_tokens", result)
        self.assertGreater(result["tokens_per_second"], 0)
        self.assertGreater(result["avg_batch_latency_ms"], 0)

    def test_benchmark_device_reported(self):
        model, _ = _create_tiny_model()
        config = EvaluationConfig(benchmark_num_batches=1, benchmark_warmup_batches=0)
        bench = InferenceBenchmark(config)
        loader = _create_dummy_dataloader(num_batches=2)

        result = bench.run(model, loader, torch.device("cpu"))
        self.assertEqual(result["device"], "cpu")


class TestPromptSampler(unittest.TestCase):
    """Tests for prompt sampler."""

    def test_prepare_prompt(self):
        sampler = PromptSampler(["Test legal prompt"])
        tokenizer = _MockTokenizer()
        result = sampler.prepare_prompt("Test", tokenizer, torch.device("cpu"))
        self.assertIn("src_input_ids", result)
        self.assertIn("src_padding_mask", result)
        self.assertEqual(result["src_input_ids"].dim(), 2)
        self.assertEqual(result["src_input_ids"].size(0), 1)

    def test_get_all_prompts(self):
        prompts = ["P1", "P2", "P3"]
        sampler = PromptSampler(prompts)
        self.assertEqual(sampler.get_all_prompts(), prompts)


class TestTextGeneration(unittest.TestCase):
    """Tests for text generation evaluator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = EvaluationConfig(
            output_dir=self.tmpdir,
            generation_enabled=True,
            num_generation_samples=1,
            max_generation_tokens=10,
            generation_prompts=["The court orders"],
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generation_produces_output(self):
        model, _ = _create_tiny_model()
        tokenizer = _MockTokenizer()
        gen = TextGenerationEvaluator(self.config)
        results = gen.generate_samples(model, tokenizer, torch.device("cpu"), epoch=0)
        self.assertEqual(len(results), 1)
        self.assertIn("generated_text", results[0])
        self.assertIn("prompt", results[0])

    def test_generation_disabled(self):
        self.config.generation_enabled = False
        model, _ = _create_tiny_model()
        gen = TextGenerationEvaluator(self.config)
        results = gen.generate_samples(model, None, torch.device("cpu"))
        self.assertEqual(len(results), 0)

    def test_checkpoint_comparison(self):
        model, _ = _create_tiny_model()
        tokenizer = _MockTokenizer()
        gen = TextGenerationEvaluator(self.config)
        gen.generate_samples(model, tokenizer, torch.device("cpu"), epoch=0)
        gen.generate_samples(model, tokenizer, torch.device("cpu"), epoch=1)
        comps = gen.compare_checkpoints()
        self.assertGreater(len(comps), 0)
        self.assertIn("generations_over_time", comps[0])
        self.assertEqual(len(comps[0]["generations_over_time"]), 2)


class TestReportGenerator(unittest.TestCase):
    """Tests for report generation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = EvaluationConfig(
            output_dir=self.tmpdir,
            reports_dir=os.path.join(self.tmpdir, "reports"),
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_and_generate(self):
        rg = ReportGenerator(self.config)
        m1 = EvaluationMetrics(avg_loss=5.0, perplexity=148.4, top_1_accuracy=0.01, epoch=0)
        m2 = EvaluationMetrics(avg_loss=3.0, perplexity=20.1, top_1_accuracy=0.15, epoch=1)
        rg.record_metrics(m1)
        rg.record_metrics(m2)
        report_path = rg.generate_report()
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith(".md"))

        # Check JSON sidecar
        json_path = report_path.replace(".md", ".json")
        self.assertTrue(os.path.exists(json_path))

    def test_report_contains_tables(self):
        rg = ReportGenerator(self.config)
        rg.record_metrics(EvaluationMetrics(avg_loss=2.0, perplexity=7.4, epoch=0))
        report_path = rg.generate_report()
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Loss History", content)
        self.assertIn("Accuracy History", content)
        self.assertIn("Metrics Summary", content)

    def test_benchmark_recorded(self):
        rg = ReportGenerator(self.config)
        rg.record_metrics(EvaluationMetrics(epoch=0))
        rg.record_benchmark({"tokens_per_second": 1500, "avg_batch_latency_ms": 12.5}, epoch=0)
        report_path = rg.generate_report()
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Benchmark Results", content)


class TestEvaluationEngine(unittest.TestCase):
    """Integration tests for the full EvaluationEngine."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = EvaluationConfig(
            output_dir=self.tmpdir,
            reports_dir=os.path.join(self.tmpdir, "reports"),
            generation_enabled=True,
            benchmark_enabled=True,
            benchmark_num_batches=1,
            benchmark_warmup_batches=0,
            num_generation_samples=1,
            max_generation_tokens=5,
            generation_prompts=["Section 302"],
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_evaluation(self):
        model, _ = _create_tiny_model()
        tokenizer = _MockTokenizer()
        engine = EvaluationEngine(self.config, tokenizer)
        loader = _create_dummy_dataloader(num_batches=2)

        metrics = engine.evaluate(model, loader, torch.device("cpu"), epoch=0)
        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertGreater(metrics.avg_loss, 0)
        self.assertGreater(metrics.perplexity, 0)

    def test_final_report(self):
        model, _ = _create_tiny_model()
        tokenizer = _MockTokenizer()
        engine = EvaluationEngine(self.config, tokenizer)
        loader = _create_dummy_dataloader(num_batches=2)

        engine.evaluate(model, loader, torch.device("cpu"), epoch=0)
        engine.evaluate(model, loader, torch.device("cpu"), epoch=1)
        path = engine.generate_final_report()
        self.assertTrue(os.path.exists(path))

    def test_as_callback(self):
        model, _ = _create_tiny_model()
        tokenizer = _MockTokenizer()
        engine = EvaluationEngine(self.config, tokenizer)
        loader = _create_dummy_dataloader(num_batches=2)

        callback = engine.as_callback(model, loader, torch.device("cpu"))
        self.assertIsInstance(callback, EvaluationCallback)

        # Simulate training lifecycle
        mgr = CallbackManager([callback])
        mgr.fire("on_step_end", step=0, global_step=10, loss=5.0)
        mgr.fire("on_epoch_end", epoch=0, train_loss=5.0, val_loss=4.5)
        mgr.fire("on_training_end")

    def test_evaluation_without_tokenizer(self):
        """Engine should still work for metrics even without tokenizer (skips generation)."""
        model, _ = _create_tiny_model()
        engine = EvaluationEngine(self.config, tokenizer=None)
        loader = _create_dummy_dataloader(num_batches=2)
        metrics = engine.evaluate(model, loader, torch.device("cpu"), epoch=0)
        self.assertGreater(metrics.avg_loss, 0)


if __name__ == "__main__":
    unittest.main()
