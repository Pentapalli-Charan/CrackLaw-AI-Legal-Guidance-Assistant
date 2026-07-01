import time
import logging
import threading
from typing import Dict, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

from src.config import Config
from src.metadata import MetadataManager
from src.models.model_hub import ModelHub

logger = logging.getLogger("CrackLaw.Services.HealthService")

class HealthService:
    """Telemetry collector monitoring endpoint requests, processing durations, and resource usage."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern for tracking global app metrics."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HealthService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[Config] = None, model_hub: Optional[ModelHub] = None):
        if self._initialized:
            return
        self.config = config or Config()
        self.model_hub = model_hub or ModelHub()
        self.metadata_mgr = MetadataManager(self.config)
        
        # Telemetry metrics
        self.metrics_lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.latency_sum = 0.0
        self.route_counts: Dict[str, int] = {}
        
        self._initialized = True
        logger.info("HealthService singleton instantiated.")

    def log_request(self, route: str, latency_ms: float, is_error: bool = False) -> None:
        """Records telemetry details from a processed request."""
        with self.metrics_lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            self.latency_sum += latency_ms
            self.route_counts[route] = self.route_counts.get(route, 0) + 1

    def get_health(self) -> Dict[str, str]:
        """Simple health check representation."""
        return {"status": "healthy"}

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieves collected telemetry statistics."""
        with self.metrics_lock:
            avg_latency = (self.latency_sum / self.request_count) if self.request_count > 0 else 0.0
            return {
                "total_requests": self.request_count,
                "error_requests": self.error_count,
                "average_latency_ms": round(avg_latency, 2),
                "route_requests": self.route_counts.copy()
            }

    def get_status(self) -> Dict[str, Any]:
        """Gathers system status diagnostics (DB indices count, cache size, CPU/RAM usage)."""
        try:
            # 1. Read document registry length
            doc_count = len(self.metadata_mgr.registry)
            
            # 2. Get active memory cache details
            loaded_models = list(self.model_hub.cache.cache.keys())
            
            # 3. System resources
            if psutil:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                ram_percent = memory.percent
                ram_available = round(memory.available / (1024 * 1024), 2)
            else:
                cpu_percent = 0.0
                ram_percent = 0.0
                ram_available = 0.0
            
            return {
                "status": "ready",
                "database": {
                    "registered_documents": doc_count
                },
                "models": {
                    "loaded_in_memory_cache": loaded_models,
                    "cache_size": len(loaded_models)
                },
                "resources": {
                    "cpu_usage_percent": cpu_percent,
                    "ram_usage_percent": ram_percent,
                    "ram_available_mb": ram_available
                }
            }
        except Exception as e:
            logger.error("Failed to gather system status diagnostics: %s", str(e))
            return {
                "status": "degraded",
                "error": str(e)
            }
