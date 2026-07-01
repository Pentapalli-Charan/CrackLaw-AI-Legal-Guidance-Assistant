import os
import glob
import logging
import re
from typing import Dict, Any, Optional, Tuple
import torch
from src.models.config import CHECKPOINTS_DIR
from src.models.exceptions import CheckpointError

logger = logging.getLogger("CrackLaw.Models.CheckpointManager")

class CheckpointManager:
    """Manages training checkpoints: saving progress states and supporting automatic resumption."""

    def __init__(self, checkpoints_dir: str = CHECKPOINTS_DIR):
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _get_checkpoint_filename(self, model_name: str, epoch: int) -> str:
        """Helper to resolve checkpoint path."""
        return os.path.join(self.checkpoints_dir, f"{model_name}_epoch_{epoch}.pt")

    def save_checkpoint(
        self,
        model_name: str,
        framework: str,
        model: Any,
        optimizer: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        hyperparams: Dict[str, Any]
    ) -> str:
        """Saves current state dicts and metrics to a checkpoint file."""
        filepath = self._get_checkpoint_filename(model_name, epoch)
        logger.info("Saving checkpoint for '%s' epoch %d to %s", model_name, epoch, filepath)

        try:
            state = {
                "model_name": model_name,
                "framework": framework,
                "epoch": epoch,
                "metrics": metrics,
                "hyperparameters": hyperparams
            }

            if framework == "pytorch":
                state["model_state_dict"] = model.state_dict()
                if optimizer:
                    state["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(state, filepath)

            elif framework == "tensorflow":
                # For TF or emulator, we can check if it has state_dict (emulator)
                if hasattr(model, "state_dict"):
                    state["model_state_dict"] = model.state_dict()
                    if optimizer:
                        state["optimizer_state_dict"] = optimizer.state_dict()
                else:
                    # Pure Keras model - save weights to temp folder and serialize path
                    keras_weights_path = filepath + ".h5"
                    model.save_weights(keras_weights_path)
                    state["keras_weights_path"] = keras_weights_path
                
                torch.save(state, filepath)

            elif framework == "sklearn":
                # For sklearn, we serialize the entire estimator
                state["model_binary"] = pickle.dumps(model)
                torch.save(state, filepath)

            logger.info("Checkpoint saved successfully.")
            return filepath

        except Exception as e:
            logger.error("Failed to save checkpoint: %s", str(e))
            raise CheckpointError(f"Checkpoint save error: {e}") from e

    def find_latest_checkpoint(self, model_name: str) -> Optional[str]:
        """Scans folder to find the checkpoint file representing the latest epoch."""
        pattern = os.path.join(self.checkpoints_dir, f"{model_name}_epoch_*.pt")
        files = glob.glob(pattern)
        if not files:
            return None

        # Extract epoch numbers using regex
        latest_file = None
        max_epoch = -1
        
        for f in files:
            match = re.search(r"_epoch_(\d+)\.pt$", f)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_file = f

        return latest_file

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Loads and returns state details from a specific checkpoint file."""
        if not os.path.exists(filepath):
            raise CheckpointError(f"Checkpoint file not found: {filepath}")

        try:
            logger.info("Loading checkpoint from: %s", filepath)
            # Load with map_location to ensure CPU resiliency
            checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
            return checkpoint
        except Exception as e:
            logger.error("Failed to load checkpoint file: %s", str(e))
            raise CheckpointError(f"Failed to parse checkpoint payload: {e}") from e

    def clean_checkpoints(self, model_name: str) -> None:
        """Deletes all checkpoint records matching the model name."""
        pattern = os.path.join(self.checkpoints_dir, f"{model_name}_epoch_*")
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                logger.warning("Failed to delete checkpoint file %s: %s", f, str(e))
