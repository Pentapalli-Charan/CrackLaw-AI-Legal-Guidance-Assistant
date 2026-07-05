"""
CrackLawLM Trainer
====================
Top-level orchestrator that assembles every training component and drives the
full training lifecycle: setup → train → validate → checkpoint → early stop.
"""

import time
import logging
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

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
    CallbackManager, TrainingCallback,
    ProgressCallback, GradientMonitorCallback,
)
from src.llm.training.training_loop import train_one_epoch, validate

logger = logging.getLogger("CrackLaw.LLM.Training.Trainer")


class CrackLawTrainer:
    """
    Production-grade trainer for the CrackLaw Transformer.

    Orchestrates:
      - Model creation / device placement
      - Optimizer, scheduler, loss, gradient clipper, AMP
      - Checkpoint save/load/resume
      - Early stopping
      - Callbacks
      - Full epoch loop: train → validate → checkpoint → log

    Usage:
        trainer = CrackLawTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=training_config,
        )
        trainer.train()
    """

    def __init__(
        self,
        model: CrackLawTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: TrainingConfig,
        callbacks: Optional[List[TrainingCallback]] = None,
        resume_from: Optional[str] = None,
    ):
        self.config = training_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.resume_from = resume_from

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training device: {self.device}")

        # Model
        self.model = model.to(self.device)
        param_info = OptimizerFactory.get_param_count(self.model)
        logger.info(
            f"Model parameters: {param_info['total_parameters']:,} total, "
            f"{param_info['trainable_parameters']:,} trainable"
        )

        # Loss
        self.loss_fn = LanguageModelingLoss(self.config)

        # Optimizer
        self.optimizer = OptimizerFactory.create(self.model, self.config)

        # Scheduler
        total_steps = self._compute_total_steps()
        self.scheduler = SchedulerFactory.create(
            self.optimizer, self.config, total_steps
        )

        # Gradient clipping
        self.gradient_clipper = GradientClipper(self.config)

        # Mixed precision
        self.amp_manager = MixedPrecisionManager(self.config)

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(self.config)

        # Early stopping
        self.early_stopping = EarlyStopping(self.config)

        # Logger
        self.training_logger = TrainingLogger(self.config)

        # Callbacks
        self.callback_manager = CallbackManager(callbacks or [])
        self.callback_manager.add(ProgressCallback())
        self.callback_manager.add(GradientMonitorCallback())

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Resume if checkpoint provided
        if self.resume_from:
            self._resume_training(self.resume_from)

    def train(self):
        """
        Executes the full training loop across all epochs.

        For each epoch:
          1. Train on all mini-batches
          2. Validate on the full validation set
          3. Check early stopping
          4. Save checkpoint
          5. Track best model
        """
        logger.info("=" * 60)
        logger.info("  CrackLawLM Training Engine — Starting")
        logger.info("=" * 60)
        self._log_config()

        training_start = time.time()
        self.callback_manager.fire("on_training_start")

        final_train_loss = float("inf")
        final_val_loss = float("inf")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.training_logger.start_epoch(epoch)
            self.callback_manager.fire("on_epoch_start", epoch=epoch)

            # ─── Train ───
            train_result = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                gradient_clipper=self.gradient_clipper,
                amp_manager=self.amp_manager,
                training_logger=self.training_logger,
                callback_manager=self.callback_manager,
                device=self.device,
                epoch=epoch,
                global_step=self.global_step,
                config=self.config,
            )
            self.global_step = train_result["global_step"]
            final_train_loss = train_result["avg_loss"]

            # ─── Validate ───
            self.callback_manager.fire("on_validation_start")
            val_result = validate(
                model=self.model,
                dataloader=self.val_loader,
                loss_fn=self.loss_fn,
                amp_manager=self.amp_manager,
                device=self.device,
            )
            final_val_loss = val_result["avg_loss"]

            self.training_logger.log_validation(
                epoch, val_result["avg_loss"], val_result["avg_perplexity"]
            )
            self.callback_manager.fire(
                "on_validation_end", val_loss=val_result["avg_loss"]
            )

            # ─── Epoch summary ───
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.training_logger.end_epoch(
                epoch, final_train_loss, final_val_loss, current_lr
            )
            self.callback_manager.fire(
                "on_epoch_end", epoch=epoch,
                train_loss=final_train_loss, val_loss=final_val_loss,
            )

            # ─── Best model tracking ───
            if final_val_loss < self.best_val_loss:
                self.best_val_loss = final_val_loss
                best_path = self.checkpoint_manager.save_best(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, final_train_loss, final_val_loss,
                    amp_state=self.amp_manager.state_dict(),
                )
                self.callback_manager.fire(
                    "on_checkpoint_saved", path=best_path
                )

            # ─── Periodic checkpoint ───
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                ckpt_path = self.checkpoint_manager.save(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, final_train_loss, final_val_loss,
                    self.best_val_loss,
                    amp_state=self.amp_manager.state_dict(),
                )
                self.callback_manager.fire(
                    "on_checkpoint_saved", path=ckpt_path
                )

            # ─── Early stopping ───
            if self.early_stopping.step(final_val_loss):
                self.callback_manager.fire(
                    "on_early_stop", epoch=epoch,
                    best_loss=self.early_stopping.best_loss,
                )
                logger.warning(
                    f"Early stopping at epoch {epoch}. "
                    f"Restoring best model (val_loss={self.best_val_loss:.6f})"
                )
                # Restore best model
                best_ckpt = self.checkpoint_manager.get_best_checkpoint()
                if best_ckpt:
                    self.checkpoint_manager.load(
                        best_ckpt, self.model, device=self.device
                    )
                break

        # ─── Training complete ───
        total_time = time.time() - training_start
        self.training_logger.save_summary(
            final_train_loss, final_val_loss, self.best_val_loss,
            self.global_step, self.config.num_epochs, total_time,
        )
        self.callback_manager.fire("on_training_end")

        logger.info("=" * 60)
        logger.info("  CrackLawLM Training Complete")
        logger.info(f"  Best val_loss: {self.best_val_loss:.6f}")
        logger.info(f"  Total time: {self.training_logger._format_time(total_time)}")
        logger.info("=" * 60)

    def _compute_total_steps(self) -> int:
        """Computes total training steps across all epochs."""
        if self.config.total_training_steps:
            return self.config.total_training_steps
        steps_per_epoch = len(self.train_loader)
        return steps_per_epoch * self.config.num_epochs

    def _resume_training(self, checkpoint_path: str):
        """Resumes training from a checkpoint."""
        info = self.checkpoint_manager.load(
            checkpoint_path, self.model, self.optimizer,
            self.scheduler, self.device,
        )
        self.start_epoch = info["epoch"] + 1
        self.global_step = info["global_step"]
        self.best_val_loss = info["best_val_loss"]
        self.amp_manager.load_state_dict(info.get("amp_scaler_state_dict", {}))
        logger.info(
            f"Resumed training from epoch {self.start_epoch}, "
            f"step {self.global_step}"
        )

    def _log_config(self):
        """Logs the training configuration."""
        logger.info(f"  Epochs:        {self.config.num_epochs}")
        logger.info(f"  Optimizer:     {self.config.optimizer_type}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Scheduler:     {self.config.scheduler_type}")
        logger.info(f"  Warmup steps:  {self.config.warmup_steps}")
        logger.info(f"  Grad clip:     {self.config.gradient_clip_max_norm}")
        logger.info(f"  Mixed prec:    {self.amp_manager.enabled}")
        logger.info(f"  Early stop:    {self.config.early_stopping_enabled}")
        logger.info(f"  Device:        {self.device}")
