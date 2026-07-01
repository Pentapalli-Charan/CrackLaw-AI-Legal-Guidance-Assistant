import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.models.model_registry import ModelRegistry
from src.models.model_cache import ModelCache
from src.models.preprocessing import VocabularyIndexer, NumericalScaler
from src.models.feature_engineering import LegalFeatureExtractor
from src.models.dataset_loader import ModelDatasetLoader
from src.models.checkpoint_manager import CheckpointManager
from src.models.experiment_manager import ExperimentManager
from src.models.training_engine import ModelTrainingEngine
from src.models.inference_engine import ModelInferenceEngine
from src.models.model_loader import ModelLoader

logger = logging.getLogger("CrackLaw.Models.ModelHub")

class ModelHub:
    """Central orchestrator facade for the CrackLaw AI Model Hub."""

    def __init__(self, registry_file: Optional[str] = None):
        self.registry = ModelRegistry(registry_file) if registry_file else ModelRegistry()
        self.loader = ModelLoader()
        self.cache = ModelCache(self.loader)
        self.vocab = VocabularyIndexer()
        self.scaler = NumericalScaler()
        self.extractor = LegalFeatureExtractor()
        self.dataset_loader = ModelDatasetLoader()
        self.checkpoint_manager = CheckpointManager()
        self.experiment_manager = ExperimentManager()
        
        self.training_engine = ModelTrainingEngine(self.checkpoint_manager)
        self.inference_engine = ModelInferenceEngine(
            model_cache=self.cache,
            model_registry=self.registry,
            vocab_indexer=self.vocab,
            numerical_scaler=self.scaler,
            feature_extractor=self.extractor
        )
        logger.info("Central ModelHub instantiated successfully.")

    # --- Unified Inference API Routes ---

    def predict(self, model_name: str, input_data: Any) -> Any:
        """Runs predictions through the active version of a model."""
        return self.inference_engine.predict(model_name, input_data)

    def classify(self, text: str, task: str = "intent") -> str:
        """Classifies text queries using either the Intent Classifier or the Document Classifier."""
        return self.inference_engine.classify(text, task)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Identifies legal entities (Acts, Sections, Courts) using the PyTorch NER model."""
        return self.inference_engine.extract_entities(text)

    def recommend(self, user_id: str, user_history: List[str]) -> List[str]:
        """Provides content-based case recommendations for active session users."""
        return self.inference_engine.recommend(user_id, user_history)

    def calculate_risk(self, contract_text: str) -> Dict[str, Any]:
        """Analyzes a contract string and returns a risk score index between 0.0 and 1.0."""
        return self.inference_engine.calculate_risk(contract_text)

    # --- Training orchestrations ---

    def train_pytorch_model(
        self,
        model_name: str,
        version: str,
        dataset_name: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        hyperparams: Optional[Dict[str, Any]] = None,
        resume: bool = True
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Trains and registers a PyTorch neural network model."""
        trained_model, metrics = self.training_engine.train_pytorch(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hyperparams=hyperparams,
            resume=resume
        )

        # Export path
        export_path = self.checkpoint_manager._get_checkpoint_filename(model_name, epoch=999) # final version
        torch.save(trained_model.state_dict(), export_path)

        # Register
        self.registry.register_model(
            model_name=model_name,
            version=version,
            framework="pytorch",
            dataset_used=dataset_name,
            metrics=metrics,
            checkpoint_path=export_path,
            status="active",
            hyperparams=hyperparams
        )

        # Log Experiment
        self.experiment_manager.log_run(
            model_name=model_name,
            version=version,
            hyperparams=hyperparams or {},
            training_metrics=metrics.get("train_loss_history", []),
            validation_metrics=metrics.get("val_loss_history", []),
            best_metrics=metrics,
            training_time_sec=metrics.get("training_time_sec", 0.0),
            best_model_path=export_path
        )

        return trained_model, metrics

    def train_sklearn_model(
        self,
        model_name: str,
        version: str,
        dataset_name: str,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Trains and registers a Scikit-learn estimator model."""
        trained_model, metrics = self.training_engine.train_sklearn(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val
        )

        # Export path
        export_path = self.checkpoint_manager._get_checkpoint_filename(model_name, epoch=0)
        import pickle
        with open(export_path, "wb") as f:
            pickle.dump(trained_model, f)

        # Register
        self.registry.register_model(
            model_name=model_name,
            version=version,
            framework="sklearn",
            dataset_used=dataset_name,
            metrics=metrics,
            checkpoint_path=export_path,
            status="active"
        )

        return trained_model, metrics

    def train_keras_model(
        self,
        model_name: str,
        version: str,
        dataset_name: str,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Trains and registers a TensorFlow Keras model (or Keras PyTorch emulator)."""
        trained_model, metrics = self.training_engine.train_keras_emulator(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            hyperparams=hyperparams
        )

        # Export path
        export_path = self.checkpoint_manager._get_checkpoint_filename(model_name, epoch=0)
        trained_model.save(export_path)

        # Register
        self.registry.register_model(
            model_name=model_name,
            version=version,
            framework="tensorflow",
            dataset_used=dataset_name,
            metrics=metrics,
            checkpoint_path=export_path,
            status="active",
            hyperparams=hyperparams
        )

        return trained_model, metrics

    # --- Cache Lifecycles ---

    def unload_model(self, model_name: str) -> bool:
        """Evicts a loaded model instance from memory cache."""
        return self.cache.unload_model(model_name)

    def unload_idle_models(self, max_idle_seconds: float) -> int:
        """Evicts idle model instances exceeding the access timeout limit."""
        return self.cache.unload_idle_models(max_idle_seconds)
