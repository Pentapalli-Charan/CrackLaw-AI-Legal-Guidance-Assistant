import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.models.config import EXPERIMENTS_DIR

logger = logging.getLogger("CrackLaw.Models.ExperimentManager")

class ExperimentManager:
    """Tracks training hyperparameter configurations, metrics, and best performing models."""

    def __init__(self, experiments_dir: str = EXPERIMENTS_DIR):
        self.experiments_dir = experiments_dir
        self.runs_file = os.path.join(self.experiments_dir, "runs.json")
        self.lock = threading.Lock()
        self.runs: List[Dict[str, Any]] = self._load_runs()

    def _load_runs(self) -> List[Dict[str, Any]]:
        """Loads logged training runs from disk JSON file."""
        if not os.path.exists(self.runs_file):
            return []
        try:
            with open(self.runs_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load experiment runs log: %s", str(e))
            return []

    def _save_runs(self) -> None:
        """Saves current training run logs to disk JSON file."""
        try:
            os.makedirs(self.experiments_dir, exist_ok=True)
            with open(self.runs_file, "w", encoding="utf-8") as f:
                json.dump(self.runs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save experiment run: %s", str(e))

    def log_run(
        self,
        model_name: str,
        version: str,
        hyperparams: Dict[str, Any],
        training_metrics: List[Dict[str, float]],
        validation_metrics: List[Dict[str, float]],
        best_metrics: Dict[str, float],
        training_time_sec: float,
        best_model_path: str
    ) -> Dict[str, Any]:
        """Creates a training run record and appends it to the experiments history log."""
        with self.lock:
            run_record = {
                "run_id": f"{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "model_name": model_name,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": hyperparams,
                "epoch_history": {
                    "train": training_metrics,
                    "val": validation_metrics
                },
                "best_metrics": best_metrics,
                "training_time_seconds": round(training_time_sec, 2),
                "best_model_path": best_model_path
            }

            self.runs.append(run_record)
            self._save_runs()
            logger.info("Logged experiment run: '%s'", run_record["run_id"])
            return run_record

    def get_model_runs(self, model_name: str) -> List[Dict[str, Any]]:
        """Retrieves history of training runs for a specific model name."""
        with self.lock:
            return [run for run in self.runs if run["model_name"] == model_name]

    def get_best_run(self, model_name: str, target_metric: str = "f1") -> Optional[Dict[str, Any]]:
        """Finds the run with the highest value of a target metric for a given model."""
        runs = self.get_model_runs(model_name)
        if not runs:
            return None

        best_run = None
        best_value = -1.0
        
        for run in runs:
            val = run["best_metrics"].get(target_metric, 0.0)
            if val > best_value:
                best_value = val
                best_run = run

        return best_run

    def clear_runs(self) -> None:
        """Purges experiment runs log (intended for test tearDowns)."""
        with self.lock:
            self.runs.clear()
            if os.path.exists(self.runs_file):
                try:
                    os.remove(self.runs_file)
                except Exception as e:
                    logger.warning("Failed to remove runs file: %s", str(e))
