"""
CrackLawLM Training Engine — Comprehensive Unit Tests
=======================================================
Verifies: forward pass, backward pass, checkpoint save/load,
resume training, gradient clipping, optimizer updates,
scheduler updates, loss computation, early stopping, AMP, callbacks.
"""

import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Ensure project root is on sys.path ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.training.config import TrainingConfig
from src.llm.training.loss import LanguageModelingLoss
from src.llm.training.optimizer import OptimizerFactory
from src.llm.training.scheduler import SchedulerFactory
from src.llm.training.gradient_clipping import GradientClipper
from src.llm.training.mixed_precision import MixedPrecisionManager
from src.llm.training.checkpoint_manager import CheckpointManager
from src.llm.training.early_stopping import EarlyStopping
from src.llm.training.logger import TrainingLogger
from src.llm.training.callbacks import (
    CallbackManager, TrainingCallback, ProgressCallback, GradientMonitorCallback
)
from src.llm.training.training_loop import train_one_epoch, validate
from src.llm.training.trainer import CrackLawTrainer


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


class TestTrainingConfig(unittest.TestCase):
    """Tests for TrainingConfig validation and defaults."""

    def test_default_values(self):
        config = TrainingConfig()
        self.assertEqual(config.optimizer_type, "adamw")
        self.assertEqual(config.scheduler_type, "cosine")
        self.assertTrue(config.gradient_clip_enabled)
        self.assertTrue(config.early_stopping_enabled)

    def test_invalid_optimizer_raises(self):
        with self.assertRaises(AssertionError):
            TrainingConfig(optimizer_type="invalid")

    def test_invalid_scheduler_raises(self):
        with self.assertRaises(AssertionError):
            TrainingConfig(scheduler_type="invalid")

    def test_directories_created(self):
        config = TrainingConfig()
        self.assertTrue(os.path.isdir(config.checkpoint_dir))
        self.assertTrue(os.path.isdir(config.log_dir))


class TestLoss(unittest.TestCase):
    """Tests for LanguageModelingLoss."""

    def setUp(self):
        self.config = TrainingConfig()
        self.loss_fn = LanguageModelingLoss(self.config)

    def test_forward_produces_loss(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        result = self.loss_fn(logits, labels)
        self.assertIn("loss", result)
        self.assertIn("perplexity", result)
        self.assertIn("num_tokens", result)
        self.assertFalse(torch.isnan(result["loss"]))

    def test_ignore_index_tokens(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.full((2, 10), -100, dtype=torch.long)
        labels[0, 0] = 5
        result = self.loss_fn(logits, labels)
        self.assertEqual(result["num_tokens"].item(), 1)

    def test_perplexity_is_exp_loss(self):
        logits = torch.randn(1, 5, 100)
        labels = torch.randint(0, 100, (1, 5))
        result = self.loss_fn(logits, labels)
        expected_ppl = torch.exp(result["loss"])
        self.assertAlmostEqual(
            result["perplexity"].item(), expected_ppl.item(), places=3
        )


class TestOptimizer(unittest.TestCase):
    """Tests for OptimizerFactory."""

    def setUp(self):
        self.model, _ = _create_tiny_model()

    def test_adam(self):
        config = TrainingConfig(optimizer_type="adam")
        opt = OptimizerFactory.create(self.model, config)
        self.assertIsInstance(opt, torch.optim.Adam)

    def test_adamw(self):
        config = TrainingConfig(optimizer_type="adamw")
        opt = OptimizerFactory.create(self.model, config)
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_sgd(self):
        config = TrainingConfig(optimizer_type="sgd")
        opt = OptimizerFactory.create(self.model, config)
        self.assertIsInstance(opt, torch.optim.SGD)

    def test_param_count(self):
        info = OptimizerFactory.get_param_count(self.model)
        self.assertGreater(info["total_parameters"], 0)
        self.assertEqual(info["total_parameters"], info["trainable_parameters"])


class TestScheduler(unittest.TestCase):
    """Tests for SchedulerFactory — all four schedule types."""

    def _make_optimizer(self):
        model, _ = _create_tiny_model()
        config = TrainingConfig()
        return OptimizerFactory.create(model, config)

    def test_linear_decay(self):
        opt = self._make_optimizer()
        config = TrainingConfig(scheduler_type="linear", warmup_steps=10)
        sched = SchedulerFactory.create(opt, config, total_steps=100)
        # Step through warmup
        for _ in range(10):
            sched.step()
        lr_after_warmup = opt.param_groups[0]["lr"]
        for _ in range(50):
            sched.step()
        lr_later = opt.param_groups[0]["lr"]
        self.assertLess(lr_later, lr_after_warmup)

    def test_cosine_decay(self):
        opt = self._make_optimizer()
        config = TrainingConfig(scheduler_type="cosine", warmup_steps=5)
        sched = SchedulerFactory.create(opt, config, total_steps=50)
        lrs = []
        for _ in range(50):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # LR should decrease after warmup
        self.assertGreater(lrs[10], lrs[-1])

    def test_warmup_only(self):
        opt = self._make_optimizer()
        config = TrainingConfig(scheduler_type="warmup", warmup_steps=10)
        sched = SchedulerFactory.create(opt, config, total_steps=50)
        for _ in range(20):
            sched.step()
        # After warmup, LR should be at base LR
        lr = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(lr, config.learning_rate, places=6)

    def test_step_decay(self):
        opt = self._make_optimizer()
        config = TrainingConfig(
            scheduler_type="step", warmup_steps=0,
            step_decay_factor=0.5, step_decay_every=10
        )
        sched = SchedulerFactory.create(opt, config, total_steps=100)
        for _ in range(10):
            sched.step()
        lr1 = opt.param_groups[0]["lr"]
        for _ in range(10):
            sched.step()
        lr2 = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(lr2, lr1 * 0.5, places=6)


class TestGradientClipper(unittest.TestCase):
    """Tests for gradient clipping and statistics."""

    def test_clip_reduces_norm(self):
        model, _ = _create_tiny_model()
        config = TrainingConfig(gradient_clip_max_norm=0.01)
        clipper = GradientClipper(config)

        # Create artificial large gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 100.0

        clipper.clip(model)

        total_norm = GradientClipper.compute_total_norm(model)
        self.assertLessEqual(total_norm, 0.02)  # Slight tolerance

    def test_statistics(self):
        model, _ = _create_tiny_model()
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        stats = GradientClipper.compute_gradient_statistics(model)
        self.assertIsNotNone(stats["total_norm"])
        self.assertIsNotNone(stats["max_grad"])
        self.assertGreater(stats["num_params_with_grad"], 0)

    def test_no_grad_statistics(self):
        model, _ = _create_tiny_model()
        stats = GradientClipper.compute_gradient_statistics(model)
        self.assertIsNone(stats["total_norm"])
        self.assertEqual(stats["num_params_with_grad"], 0)


class TestMixedPrecision(unittest.TestCase):
    """Tests for MixedPrecisionManager."""

    def test_disabled_on_cpu(self):
        config = TrainingConfig(mixed_precision_enabled=True)
        amp = MixedPrecisionManager(config)
        if not torch.cuda.is_available():
            self.assertFalse(amp.enabled)
            self.assertIsNone(amp.scaler)

    def test_autocast_context_cpu(self):
        config = TrainingConfig(mixed_precision_enabled=False)
        amp = MixedPrecisionManager(config)
        with amp.autocast_context():
            x = torch.randn(2, 2)
            self.assertEqual(x.dtype, torch.float32)

    def test_state_dict_empty_on_cpu(self):
        config = TrainingConfig(mixed_precision_enabled=False)
        amp = MixedPrecisionManager(config)
        self.assertEqual(amp.state_dict(), {})


class TestEarlyStopping(unittest.TestCase):
    """Tests for EarlyStopping."""

    def test_improvement_resets_counter(self):
        config = TrainingConfig(early_stopping_patience=3, early_stopping_min_delta=0.01)
        es = EarlyStopping(config)
        es.step(1.0)
        es.step(0.95)  # Improvement
        self.assertEqual(es.counter, 0)

    def test_no_improvement_increments(self):
        config = TrainingConfig(early_stopping_patience=3, early_stopping_min_delta=0.01)
        es = EarlyStopping(config)
        es.step(1.0)
        es.step(1.0)  # No improvement
        self.assertEqual(es.counter, 1)

    def test_triggers_after_patience(self):
        config = TrainingConfig(early_stopping_patience=2, early_stopping_min_delta=0.01)
        es = EarlyStopping(config)
        es.step(1.0)
        es.step(1.0)
        result = es.step(1.0)
        self.assertTrue(result)
        self.assertTrue(es.should_stop)

    def test_disabled(self):
        config = TrainingConfig(early_stopping_enabled=False)
        es = EarlyStopping(config)
        for _ in range(100):
            result = es.step(999.0)
            self.assertFalse(result)

    def test_reset(self):
        config = TrainingConfig(early_stopping_patience=2)
        es = EarlyStopping(config)
        es.step(1.0)
        es.step(1.0)
        es.reset()
        self.assertEqual(es.counter, 0)
        self.assertIsNone(es.best_loss)


class TestCheckpointManager(unittest.TestCase):
    """Tests for checkpoint save/load and resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            checkpoint_dir=self.tmpdir, keep_last_n_checkpoints=2
        )
        self.ckpt_mgr = CheckpointManager(self.config)
        self.model, _ = _create_tiny_model()
        self.optimizer = OptimizerFactory.create(
            self.model, TrainingConfig(optimizer_type="adamw")
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        path = self.ckpt_mgr.save(
            self.model, self.optimizer, None,
            epoch=3, global_step=100,
            train_loss=0.5, val_loss=0.6, best_val_loss=0.6,
        )
        self.assertTrue(os.path.exists(path))

        # Load into fresh model
        new_model, _ = _create_tiny_model()
        new_opt = OptimizerFactory.create(new_model, TrainingConfig())
        info = self.ckpt_mgr.load(path, new_model, new_opt)
        self.assertEqual(info["epoch"], 3)
        self.assertEqual(info["global_step"], 100)

        # Verify weights match
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_save_best(self):
        path = self.ckpt_mgr.save_best(
            self.model, self.optimizer, None,
            epoch=1, global_step=50, train_loss=0.3, val_loss=0.4,
        )
        self.assertTrue(os.path.exists(path))
        self.assertIn("best_model", path)

    def test_cleanup_keeps_n(self):
        for i in range(5):
            self.ckpt_mgr.save(
                self.model, self.optimizer, None,
                epoch=i, global_step=i * 10,
                train_loss=0.5, val_loss=0.5, best_val_loss=0.5,
            )
        import glob
        pattern = os.path.join(self.tmpdir, "checkpoint_epoch_*.pt")
        remaining = glob.glob(pattern)
        self.assertLessEqual(len(remaining), 2)

    def test_get_latest(self):
        for i in range(3):
            self.ckpt_mgr.save(
                self.model, self.optimizer, None,
                epoch=i, global_step=i * 10,
                train_loss=0.5, val_loss=0.5, best_val_loss=0.5,
            )
        latest = self.ckpt_mgr.get_latest_checkpoint()
        self.assertIsNotNone(latest)
        self.assertIn("epoch_0002", latest)


class TestForwardBackwardPass(unittest.TestCase):
    """Tests that the full forward+backward pass works through the model."""

    def test_forward_pass_produces_logits(self):
        model, config = _create_tiny_model()
        input_ids = torch.randint(4, 100, (2, 16))
        mask = torch.ones(2, 1, 1, 16, dtype=torch.bool)
        logits = model(input_ids, input_ids, mask, mask)
        self.assertEqual(logits.shape, (2, 16, config.vocab_size))

    def test_backward_pass_computes_gradients(self):
        model, _ = _create_tiny_model()
        loss_fn = LanguageModelingLoss(TrainingConfig())
        input_ids = torch.randint(4, 100, (2, 16))
        labels = torch.randint(4, 100, (2, 16))
        mask = torch.ones(2, 1, 1, 16, dtype=torch.bool)

        logits = model(input_ids, input_ids, mask, mask)
        result = loss_fn(logits, labels)
        result["loss"].backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)

    def test_optimizer_updates_weights(self):
        model, _ = _create_tiny_model()
        config = TrainingConfig(optimizer_type="adamw")
        optimizer = OptimizerFactory.create(model, config)
        loss_fn = LanguageModelingLoss(config)

        # Snapshot original weights
        original = {n: p.clone() for n, p in model.named_parameters()}

        input_ids = torch.randint(4, 100, (2, 16))
        labels = torch.randint(4, 100, (2, 16))
        mask = torch.ones(2, 1, 1, 16, dtype=torch.bool)

        logits = model(input_ids, input_ids, mask, mask)
        result = loss_fn(logits, labels)
        result["loss"].backward()
        optimizer.step()

        # At least some weights should have changed
        changed = False
        for n, p in model.named_parameters():
            if not torch.equal(p, original[n]):
                changed = True
                break
        self.assertTrue(changed)


class TestTrainingLoop(unittest.TestCase):
    """Tests for train_one_epoch and validate functions."""

    def test_train_one_epoch(self):
        model, _ = _create_tiny_model()
        config = TrainingConfig(
            log_every_n_steps=1, mixed_precision_enabled=False,
            gradient_clip_enabled=True,
        )
        loader = _create_dummy_dataloader()
        loss_fn = LanguageModelingLoss(config)
        optimizer = OptimizerFactory.create(model, config)
        scheduler = SchedulerFactory.create(optimizer, config, total_steps=50)
        clipper = GradientClipper(config)
        amp = MixedPrecisionManager(config)
        tlogger = TrainingLogger(config)
        cb = CallbackManager()

        result = train_one_epoch(
            model, loader, loss_fn, optimizer, scheduler, clipper,
            amp, tlogger, cb, torch.device("cpu"), epoch=0,
            global_step=0, config=config,
        )
        self.assertIn("avg_loss", result)
        self.assertGreater(result["global_step"], 0)

    def test_validate(self):
        model, _ = _create_tiny_model()
        config = TrainingConfig(mixed_precision_enabled=False)
        loader = _create_dummy_dataloader()
        loss_fn = LanguageModelingLoss(config)
        amp = MixedPrecisionManager(config)

        result = validate(model, loader, loss_fn, amp, torch.device("cpu"))
        self.assertIn("avg_loss", result)
        self.assertIn("avg_perplexity", result)
        self.assertGreater(result["avg_perplexity"], 0)


class TestCallbacks(unittest.TestCase):
    """Tests for callback system."""

    def test_callback_fires(self):
        fired = {"count": 0}

        class CountCallback(TrainingCallback):
            def on_epoch_start(self, epoch, **kwargs):
                fired["count"] += 1

        mgr = CallbackManager([CountCallback()])
        mgr.fire("on_epoch_start", epoch=0)
        mgr.fire("on_epoch_start", epoch=1)
        self.assertEqual(fired["count"], 2)

    def test_callback_exception_handled(self):
        class BadCallback(TrainingCallback):
            def on_epoch_start(self, epoch, **kwargs):
                raise ValueError("test error")

        mgr = CallbackManager([BadCallback()])
        # Should not raise
        mgr.fire("on_epoch_start", epoch=0)


class TestFullTrainer(unittest.TestCase):
    """Integration test for the CrackLawTrainer."""

    def test_trainer_runs_one_epoch(self):
        model, _ = _create_tiny_model()
        config = TrainingConfig(
            num_epochs=1, mixed_precision_enabled=False,
            early_stopping_enabled=False,
            log_every_n_steps=1, warmup_steps=1,
            save_every_n_epochs=1,
        )
        train_loader = _create_dummy_dataloader()
        val_loader = _create_dummy_dataloader(num_batches=1)

        trainer = CrackLawTrainer(
            model=model, train_loader=train_loader,
            val_loader=val_loader, training_config=config,
        )
        trainer.train()

        self.assertGreater(trainer.global_step, 0)
        self.assertLess(trainer.best_val_loss, float("inf"))


if __name__ == "__main__":
    unittest.main()
