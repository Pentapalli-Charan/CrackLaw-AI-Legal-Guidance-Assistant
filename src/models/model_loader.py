import pickle
import logging
from typing import Any
import torch
from src.models.exceptions import ModelNotFoundError
from src.models.model_factory import ModelFactory, HAS_TF

logger = logging.getLogger("CrackLaw.Models.ModelLoader")

class ModelLoader:
    """Loads trained model binaries from disk across PyTorch, TensorFlow, and Scikit-learn frameworks."""

    @staticmethod
    def load_model(framework: str, model_name: str, filepath: str, **kwargs) -> Any:
        """Loads weights and returns an executable model instance based on framework types."""
        framework = framework.lower().strip()
        logger.info("Loading model '%s' (%s) from: %s", model_name, framework, filepath)

        try:
            if framework == "sklearn":
                with open(filepath, "rb") as f:
                    model = pickle.load(f)
                logger.info("Scikit-learn model loaded successfully.")
                return model

            elif framework == "pytorch":
                # 1. Resolve architecture from factory
                if model_name == "ner_extractor":
                    model = ModelFactory.create_ner_extractor(**kwargs)
                elif model_name == "doc_classifier":
                    model = ModelFactory.create_doc_classifier(**kwargs)
                else:
                    raise ModelNotFoundError(f"Unknown PyTorch model architecture name: '{model_name}'")

                # 2. Load state dict
                # Use map_location to ensure CPU compatibility
                state_dict = torch.load(filepath, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.eval()
                logger.info("PyTorch model loaded and set to eval mode.")
                return model

            elif framework == "tensorflow":
                if HAS_TF:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(filepath)
                    logger.info("TensorFlow Keras model loaded successfully.")
                    return model
                else:
                    # Load weights into the PyTorch-based Keras emulator
                    if model_name == "intent_classifier":
                        model = ModelFactory.create_intent_classifier(**kwargs)
                    elif model_name == "legal_risk_predictor":
                        model = ModelFactory.create_legal_risk_predictor(**kwargs)
                    else:
                        raise ModelNotFoundError(f"Unknown TensorFlow model architecture name: '{model_name}'")
                    
                    model.load_weights(filepath)
                    logger.info("TensorFlow Keras emulator loaded successfully.")
                    return model

            else:
                raise ModelNotFoundError(f"Unsupported framework type: '{framework}'")

        except FileNotFoundError as e:
            logger.error("Failed to load model file: %s", str(e))
            raise ModelNotFoundError(f"Model file not found at: '{filepath}'") from e
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise ModelNotFoundError(f"Corrupt or incompatible model file: {e}") from e
