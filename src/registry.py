import os
import json
import time
import logging
from typing import Optional, Dict, Any, List
from src.config import Config

logger = logging.getLogger("CrackLaw.Registry")

class KnowledgeRegistry:
    """Manages central dataset registry tracking external resources and synchronization statuses."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.registry_path = os.path.normpath(
            os.path.join(self.config.metadata_dir, "knowledge_registry.json")
        )
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Loads knowledge registry JSON index."""
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load knowledge registry: %s. Re-initializing.", str(e))
            return {}

    def save_registry(self) -> None:
        """Saves the knowledge registry JSON index."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def register_dataset(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Registers a dataset configuration in the central registry."""
        name = dataset_config.get("name")
        if not name:
            raise ValueError("Dataset config must have a 'name'.")

        if name not in self.registry:
            self.registry[name] = {
                "name": name,
                "description": dataset_config.get("description", ""),
                "source_type": dataset_config.get("source_type", "unknown"),
                "version": dataset_config.get("version", "1.0.0"),
                "license": dataset_config.get("license", "Unknown"),
                "document_types": dataset_config.get("document_types", []),
                "supported_languages": dataset_config.get("supported_languages", ["en"]),
                "download_status": "pending",  # pending, completed, failed
                "last_updated": None,
                "processing_status": "pending"  # pending, raw, processed, cleaned, chunked
            }
            self.save_registry()
            logger.info("Registered new dataset in knowledge registry: %s", name)
        
        return self.registry[name]

    def update_status(
        self,
        name: str,
        download_status: Optional[str] = None,
        processing_status: Optional[str] = None,
        version: Optional[str] = None
    ) -> None:
        """Updates synchronization and processing statuses for a dataset."""
        if name not in self.registry:
            logger.warning("Attempted to update status of unregistered dataset: %s", name)
            return

        if download_status:
            self.registry[name]["download_status"] = download_status
        if processing_status:
            self.registry[name]["processing_status"] = processing_status
        if version:
            self.registry[name]["version"] = version

        self.registry[name]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_registry()
        logger.info("Updated registry status for %s", name)

    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Returns registered metadata for a dataset."""
        return self.registry.get(name)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Returns all registered datasets as a list."""
        return list(self.registry.values())
