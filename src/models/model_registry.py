import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.models.config import REGISTRY_FILE
from src.models.exceptions import RegistryError

logger = logging.getLogger("CrackLaw.Models.ModelRegistry")

class ModelRegistry:
    """Manages tracking metadata of all trained models, checkpoints, versions, and deployment statuses."""

    def __init__(self, registry_file: str = REGISTRY_FILE):
        self.registry_file = registry_file
        self.lock = threading.Lock()
        self.registry: Dict[str, List[Dict[str, Any]]] = self._load_registry()

    def _load_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        """Loads model registry from disk JSON file."""
        if not os.path.exists(self.registry_file):
            return {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load model registry: %s", str(e))
            raise RegistryError(f"Corrupt model registry file: {e}") from e

    def _save_registry(self) -> None:
        """Saves current memory registry state to disk JSON file."""
        try:
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save model registry: %s", str(e))
            raise RegistryError(f"Failed to write model registry: {e}") from e

    def register_model(
        self,
        model_name: str,
        version: str,
        framework: str,
        dataset_used: str,
        metrics: Dict[str, Any],
        checkpoint_path: str,
        status: str = "trained",
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registers a newly trained model version with metadata and evaluation metrics."""
        with self.lock:
            # Construct registry record
            record = {
                "model_name": model_name,
                "version": version,
                "framework": framework,
                "dataset_used": dataset_used,
                "training_date": datetime.now().isoformat(),
                "metrics": metrics,
                "checkpoint_path": checkpoint_path,
                "status": status,
                "hyperparameters": hyperparams or {}
            }

            if model_name not in self.registry:
                self.registry[model_name] = []

            # Deactivate previous active models if this is marked active
            if status == "active":
                for r in self.registry[model_name]:
                    if r["status"] == "active":
                        r["status"] = "inactive"

            # Check if version already exists to overwrite, otherwise append
            version_exists = False
            for idx, r in enumerate(self.registry[model_name]):
                if r["version"] == version:
                    self.registry[model_name][idx] = record
                    version_exists = True
                    break

            if not version_exists:
                self.registry[model_name].append(record)

            self._save_registry()
            logger.info("Successfully registered model '%s' version '%s'", model_name, version)
            return record

    def get_active_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata of the active version of a given model name."""
        with self.lock:
            records = self.registry.get(model_name, [])
            for r in records:
                if r["status"] == "active":
                    return r
            # Fallback to the latest trained version if no model is marked active
            if records:
                sorted_records = sorted(records, key=lambda x: x["training_date"], reverse=True)
                return sorted_records[0]
            return None

    def get_model_metadata(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata of a specific model name and version."""
        with self.lock:
            records = self.registry.get(model_name, [])
            for r in records:
                if r["version"] == version:
                    return r
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """Flattens and lists all registered models across all versions."""
        with self.lock:
            all_records = []
            for records in self.registry.values():
                all_records.extend(records)
            return all_records

    def update_model_status(self, model_name: str, version: str, status: str) -> bool:
        """Updates deployment/active status of a model version."""
        with self.lock:
            records = self.registry.get(model_name, [])
            updated = False
            
            # Deactivate others if setting status to active
            if status == "active":
                for r in records:
                    if r["status"] == "active":
                        r["status"] = "inactive"

            for r in records:
                if r["version"] == version:
                    r["status"] = status
                    updated = True
                    break

            if updated:
                self._save_registry()
                logger.info("Updated status of '%s' (v%s) to '%s'", model_name, version, status)
                return True
            logger.warning("Failed to find model '%s' version '%s' to update status", model_name, version)
            return False

    def clear_registry(self) -> None:
        """Purges registry history (intended for test tearDowns)."""
        with self.lock:
            self.registry.clear()
            if os.path.exists(self.registry_file):
                try:
                    os.remove(self.registry_file)
                except Exception as e:
                    logger.warning("Failed to remove registry file: %s", str(e))
