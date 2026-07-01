import os
import json
import logging
from typing import Any, Dict, List

# Determine project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")

class Config:
    """Manages system configuration, path resolution, and logging setup."""
    
    _instance = None

    def __new__(cls, config_path: str = DEFAULT_CONFIG_PATH):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str) -> None:
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = json.load(f)
            
        self._resolve_paths()
        self._setup_logging()

    def _resolve_paths(self) -> None:
        """Resolves configuration paths relative to project root and creates them."""
        self._resolved_paths: Dict[str, str] = {}
        paths_config = self._data.get("paths", {})
        
        for key, val in paths_config.items():
            # Resolve path relative to project root if it is not absolute
            abs_path = val if os.path.isabs(val) else os.path.join(PROJECT_ROOT, val)
            abs_path = os.path.normpath(abs_path)
            self._resolved_paths[key] = abs_path
            
            # Auto-create directory (except for files)
            if not key.endswith("_file"):
                os.makedirs(abs_path, exist_ok=True)

    def _setup_logging(self) -> None:
        """Sets up python standard logging based on configuration."""
        log_config = self._data.get("logging", {})
        log_level_str = log_config.get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        log_file_rel = log_config.get("log_file", "logs/cracklaw.log")
        log_file = log_file_rel if os.path.isabs(log_file_rel) else os.path.join(PROJECT_ROOT, log_file_rel)
        log_file = os.path.normpath(log_file)
        
        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("CrackLaw")
        self.logger.info("Logging initialized. Configuration loaded from %s", self.config_path)

    @property
    def raw_dir(self) -> str:
        return self._resolved_paths["raw_dir"]

    @property
    def processed_dir(self) -> str:
        return self._resolved_paths["processed_dir"]

    @property
    def cleaned_dir(self) -> str:
        return self._resolved_paths["cleaned_dir"]

    @property
    def chunks_dir(self) -> str:
        return self._resolved_paths["chunks_dir"]

    @property
    def embeddings_dir(self) -> str:
        return self._resolved_paths["embeddings_dir"]

    @property
    def metadata_dir(self) -> str:
        return self._resolved_paths["metadata_dir"]

    @property
    def cache_dir(self) -> str:
        return self._resolved_paths["cache_dir"]

    @property
    def downloads_dir(self) -> str:
        return self._resolved_paths["downloads_dir"]

    @property
    def logs_dir(self) -> str:
        return self._resolved_paths["logs_dir"]

    @property
    def downloader_settings(self) -> Dict[str, Any]:
        return self._data.get("downloader", {})

    @property
    def formats(self) -> List[str]:
        return self._data.get("formats", [])

    @property
    def cleaning_settings(self) -> Dict[str, Any]:
        return self._data.get("cleaning", {})

    @property
    def chunking_settings(self) -> Dict[str, Any]:
        return self._data.get("chunking", {})

    @property
    def embeddings_settings(self) -> Dict[str, Any]:
        return self._data.get("embeddings", {})

    @property
    def retrieval_settings(self) -> Dict[str, Any]:
        return self._data.get("retrieval", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
