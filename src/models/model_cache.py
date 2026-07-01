import time
import logging
import threading
from typing import Dict, Any, Optional
from src.models.exceptions import ModelNotFoundError
from src.models.model_loader import ModelLoader

logger = logging.getLogger("CrackLaw.Models.ModelCache")

class ModelCache:
    """In-memory cache for loaded models with lazy loading and idle model eviction capabilities."""

    def __init__(self, model_loader: Optional[ModelLoader] = None):
        self.model_loader = model_loader or ModelLoader()
        # Mapping: { model_name -> { "model": model_instance, "framework": str, "filepath": str, "last_accessed": float } }
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def get_model(self, model_name: str, framework: str, filepath: str, **kwargs) -> Any:
        """Retrieves model from cache, loading it lazily if not present, and updates access times."""
        with self.lock:
            current_time = time.time()
            
            if model_name in self.cache:
                entry = self.cache[model_name]
                # If cached under same filepath, update timestamp and return
                if entry["filepath"] == filepath:
                    entry["last_accessed"] = current_time
                    logger.debug("Cache hit for model: '%s'", model_name)
                    return entry["model"]
                else:
                    # Filepath changed, force reload
                    logger.info("Model path changed for '%s', reloading...", model_name)

            # Lazy load model
            model_instance = self.model_loader.load_model(framework, model_name, filepath, **kwargs)
            
            self.cache[model_name] = {
                "model": model_instance,
                "framework": framework,
                "filepath": filepath,
                "last_accessed": current_time
            }
            logger.info("Cached new model instance: '%s'", model_name)
            return model_instance

    def unload_model(self, model_name: str) -> bool:
        """Manually ejects a model instance from memory cache."""
        with self.lock:
            if model_name in self.cache:
                del self.cache[model_name]
                logger.info("Evicted model '%s' from memory cache.", model_name)
                return True
            return False

    def unload_idle_models(self, max_idle_seconds: float) -> int:
        """Scans cache records and unloads models that have not been accessed within the window."""
        with self.lock:
            current_time = time.time()
            to_evict = []
            
            for m_name, entry in self.cache.items():
                idle_duration = current_time - entry["last_accessed"]
                if idle_duration > max_idle_seconds:
                    to_evict.append(m_name)

            for m_name in to_evict:
                del self.cache[m_name]
                logger.info("Auto-evicted idle model '%s' (idle for %.1f seconds)", m_name, current_time - entry["last_accessed"])

            return len(to_evict)

    def clear(self) -> None:
        """Clears all cached models from memory."""
        with self.lock:
            self.cache.clear()
            logger.info("Cleared all items from model memory cache.")
