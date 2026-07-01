import time
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.config import DEFAULT_HYPERPARAMS
from src.models.exceptions import TrainingError
from src.models.checkpoint_manager import CheckpointManager
from src.models.metrics import MetricCalculator

logger = logging.getLogger("CrackLaw.Models.TrainingEngine")

class EarlyStopping:
    """Monitors validation loss and stops training if improvements plateau."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ModelTrainingEngine:
    """Orchestrates model training pipelines across PyTorch, TensorFlow/Emulator, and Scikit-learn."""

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        # Resolve training hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device resolved: %s", self.device)

    def train_pytorch(
        self,
        model_name: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        hyperparams: Optional[Dict[str, Any]] = None,
        resume: bool = True
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Handles PyTorch training loop with mixed precision, learning rate decay, and checkpoints."""
        hparams = DEFAULT_HYPERPARAMS.copy()
        if hyperparams:
            hparams.update(hyperparams)

        epochs = hparams["epochs"]
        lr = hparams["learning_rate"]
        patience = hparams["early_stopping_patience"]
        min_delta = hparams["early_stopping_min_delta"]

        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=hparams["weight_decay"])
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=hparams["learning_rate_decay_patience"],
            factor=hparams["learning_rate_decay_factor"]
        )
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        start_epoch = 0
        train_loss_history = []
        val_loss_history = []
        start_time = time.time()

        # Automatic Resume check
        if resume:
            latest_ckpt = self.checkpoint_manager.find_latest_checkpoint(model_name)
            if latest_ckpt:
                try:
                    ckpt = self.checkpoint_manager.load_checkpoint(latest_ckpt)
                    model.load_state_dict(ckpt["model_state_dict"])
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    start_epoch = ckpt["epoch"] + 1
                    train_loss_history = ckpt.get("metrics", {}).get("train_loss_history", [])
                    val_loss_history = ckpt.get("metrics", {}).get("val_loss_history", [])
                    logger.info("Resumed PyTorch model '%s' from epoch %d", model_name, start_epoch)
                except Exception as e:
                    logger.warning("Failed to auto-resume checkpoint: %s. Starting from scratch.", str(e))

        # Mixed precision setup
        use_amp = hparams["mixed_precision"] and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

        # Standard CrossEntropyLoss for Classification tasks
        criterion = nn.CrossEntropyLoss()

        logger.info("Starting PyTorch training loop for '%s'...", model_name)
        for epoch in range(start_epoch, epochs):
            model.train()
            total_train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()

                with autocast_ctx:
                    outputs = model(batch_x)
                    # Handle shapes (e.g. sequence labeling)
                    if outputs.ndim == 3:
                        # [batch, seq, tags] -> [batch * seq, tags]
                        loss = criterion(outputs.view(-1, outputs.shape[-1]), batch_y.view(-1))
                    else:
                        loss = criterion(outputs, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item() * len(batch_x)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            train_loss_history.append(avg_train_loss)

            # Validation step
            model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_trues = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    
                    if outputs.ndim == 3:
                        loss = criterion(outputs.view(-1, outputs.shape[-1]), batch_y.view(-1))
                        preds = torch.argmax(outputs, dim=-1).cpu().numpy().flatten()
                        trues = batch_y.cpu().numpy().flatten()
                    else:
                        loss = criterion(outputs, batch_y)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        trues = batch_y.cpu().numpy()
                    
                    total_val_loss += loss.item() * len(batch_x)
                    all_preds.extend(preds)
                    all_trues.extend(trues)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_loss_history.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            logger.info("Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f", epoch + 1, epochs, avg_train_loss, avg_val_loss)

            # Checkpoint step
            metrics_dict = {
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history
            }
            self.checkpoint_manager.save_checkpoint(
                model_name=model_name,
                framework="pytorch",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics_dict,
                hyperparams=hparams
            )

            if early_stopping(avg_val_loss):
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        # Calculate final training metrics
        training_time = time.time() - start_time
        eval_metrics = MetricCalculator.calculate_classification_metrics(all_trues, all_preds)
        eval_metrics["training_time_sec"] = training_time
        
        return model, eval_metrics

    def train_sklearn(
        self,
        model_name: str,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Trains a standard Scikit-learn estimator model."""
        logger.info("Training Scikit-learn model '%s'...", model_name)
        start_time = time.time()
        
        model.fit(x_train, y_train)
        
        training_time = time.time() - start_time
        preds = model.predict(x_val)
        
        eval_metrics = MetricCalculator.calculate_classification_metrics(y_val, preds)
        eval_metrics["training_time_sec"] = training_time
        
        # Save baseline checkpoint
        self.checkpoint_manager.save_checkpoint(
            model_name=model_name,
            framework="sklearn",
            model=model,
            optimizer=None,
            epoch=0,
            metrics=eval_metrics,
            hyperparams={}
        )
        
        return model, eval_metrics

    def train_keras_emulator(
        self,
        model_name: str,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Trains TensorFlow Keras Emulator model with custom parameters."""
        logger.info("Training TensorFlow Keras emulator for '%s'...", model_name)
        hparams = DEFAULT_HYPERPARAMS.copy()
        if hyperparams:
            hparams.update(hyperparams)

        start_time = time.time()
        
        # Compile and fit
        is_regression = model_name == "legal_risk_predictor"
        model.compile(lr=hparams["learning_rate"])
        history = model.fit(
            x_train, y_train,
            epochs=hparams["epochs"],
            batch_size=hparams["batch_size"],
            validation_data=(x_val, y_val)
        )
        
        training_time = time.time() - start_time
        preds = model.predict(x_val)

        if is_regression:
            eval_metrics = MetricCalculator.calculate_regression_metrics(y_val, preds.flatten())
        else:
            preds_cls = np.argmax(preds, axis=1)
            y_val_cls = y_val if y_val.ndim == 1 else np.argmax(y_val, axis=1)
            eval_metrics = MetricCalculator.calculate_classification_metrics(y_val_cls, preds_cls, y_probs=preds)
            
        eval_metrics["training_time_sec"] = training_time

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model_name=model_name,
            framework="tensorflow",
            model=model,
            optimizer=None,
            epoch=hparams["epochs"] - 1,
            metrics=eval_metrics,
            hyperparams=hparams
        )

        return model, eval_metrics
