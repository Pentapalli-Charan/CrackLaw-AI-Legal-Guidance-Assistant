import logging
import os
import re
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from src.models.exceptions import InferenceError, ModelNotFoundError
from src.models.model_cache import ModelCache
from src.models.model_registry import ModelRegistry
from src.models.preprocessing import clean_text, VocabularyIndexer, NumericalScaler
from src.models.feature_engineering import LegalFeatureExtractor

logger = logging.getLogger("CrackLaw.Models.InferenceEngine")

class ModelInferenceEngine:
    """Unified API for model inferences: classification, entity extraction, recommendation, and risk predictions."""

    def __init__(
        self,
        model_cache: Optional[ModelCache] = None,
        model_registry: Optional[ModelRegistry] = None,
        vocab_indexer: Optional[VocabularyIndexer] = None,
        numerical_scaler: Optional[NumericalScaler] = None,
        feature_extractor: Optional[LegalFeatureExtractor] = None
    ):
        self.cache = model_cache or ModelCache()
        self.registry = model_registry or ModelRegistry()
        self.vocab = vocab_indexer or VocabularyIndexer()
        self.scaler = numerical_scaler or NumericalScaler()
        self.extractor = feature_extractor or LegalFeatureExtractor()

    def _get_active_model_instance(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Resolves model metadata and fetches/loads model instance from cache."""
        meta = self.registry.get_active_model(model_name)
        if not meta:
            raise ModelNotFoundError(f"No active model found registered under name: '{model_name}'")

        kwargs = {}
        if model_name in ["intent_classifier", "doc_classifier", "ner_extractor"]:
            if self.vocab.is_fitted:
                kwargs["vocab_size"] = len(self.vocab.vocab)
        elif model_name == "legal_risk_predictor":
            kwargs["num_features"] = len(self.extractor.get_feature_names())

        model_instance = self.cache.get_model(
            model_name=model_name,
            framework=meta["framework"],
            filepath=meta["checkpoint_path"],
            **kwargs
        )
        return model_instance, meta

    def predict(self, model_name: str, input_data: Any) -> Any:
        """Central predictor route executing forwarding predictions on the active version of a model."""
        try:
            model_instance, meta = self._get_active_model_instance(model_name)
            framework = meta["framework"].lower()

            if framework == "sklearn":
                return model_instance.predict(input_data)
            elif framework == "pytorch":
                # Ensure torch model is on CPU/GPU and in eval mode
                model_instance.eval()
                with torch.no_grad():
                    inputs = torch.tensor(input_data, dtype=torch.long)
                    outputs = model_instance(inputs)
                    return outputs.cpu().numpy()
            elif framework == "tensorflow":
                # Call predict on keras model or emulator
                return model_instance.predict(input_data)
            else:
                raise InferenceError(f"Unsupported model framework type: '{framework}'")
        except Exception as e:
            if isinstance(e, (ModelNotFoundError, InferenceError)):
                raise e
            raise InferenceError(f"Inference prediction failed for '{model_name}': {e}") from e

    def classify(self, text: str, task: str = "intent") -> str:
        """Classifies text queries using either the Intent Classifier or the Document Classifier."""
        cleaned = clean_text(text)
        
        if task == "intent":
            # Map classes to string categories
            intent_labels = ["Contract Review", "Case Search", "Legal Query", "General Conversation", "Statute Lookup", "Procedure Guideline", "Risk Assessment", "Drafting Help"]
            
            # If vocab indexer has not been fitted, we fallback to a simple default heuristic
            if not self.vocab.is_fitted:
                logger.warning("VocabularyIndexer not fitted. Falling back to default intent mapping.")
                return "Legal Query"
            
            try:
                # Transform text to sequence
                seq = self.vocab.transform([cleaned], max_len=100)
                probs = self.predict("intent_classifier", seq)[0]
                idx = int(np.argmax(probs))
                return intent_labels[idx] if idx < len(intent_labels) else "Legal Query"
            except Exception as e:
                logger.warning("Failed intent classification prediction: %s. Falling back.", str(e))
                return "Legal Query"
        
        elif task == "document":
            doc_labels = ["Criminal Law", "Civil Law", "Tax Law", "Corporate Law", "Environmental Law"]
            if not self.vocab.is_fitted:
                logger.warning("VocabularyIndexer not fitted. Falling back to default document category.")
                return "Civil Law"

            try:
                seq = self.vocab.transform([cleaned], max_len=100)
                probs = self.predict("doc_classifier", seq)[0]
                idx = int(np.argmax(probs))
                return doc_labels[idx] if idx < len(doc_labels) else "Civil Law"
            except Exception as e:
                logger.warning("Failed document classification prediction: %s. Falling back.", str(e))
                return "Civil Law"
        
        else:
            raise InferenceError(f"Unknown classification task type requested: '{task}'")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Identifies legal entities (Acts, Sections, Courts) using the PyTorch NER model."""
        if not text:
            return []

        words = text.split()
        ner_tags = ["O", "B-ACT", "I-ACT", "B-SECTION", "I-SECTION", "B-COURT", "I-COURT", "B-PARTY", "I-PARTY"]

        if not self.vocab.is_fitted:
            # Fallback regex extractor if model vocab is not ready
            logger.warning("VocabularyIndexer not fitted. Falling back to regex entity parser.")
            entities = []
            # Match "Section 73", "Section 12", etc.
            for match in re.finditer(r"\b(section|sec\.?)\s+(\d+)\b", text, re.IGNORECASE):
                entities.append({
                    "text": match.group(0),
                    "label": "SECTION",
                    "start": match.start(),
                    "end": match.end()
                })
            return entities

        try:
            # Feed sequences to PyTorch NER Model
            seq = self.vocab.transform([text], max_len=100)[0]
            # Predict tag probabilities
            model_instance, _ = self._get_active_model_instance("ner_extractor")
            model_instance.eval()
            with torch.no_grad():
                inputs = torch.tensor([seq], dtype=torch.long)
                outputs = model_instance(inputs)[0].cpu().numpy()  # [seq_len, num_tags]
            
            predicted_tag_indices = np.argmax(outputs, axis=1)

            entities = []
            current_entity = None
            
            # Map tags back to text offsets (matching sequence tokens)
            for idx, word in enumerate(words[:100]):  # Cap at max_len
                tag_idx = predicted_tag_indices[idx]
                tag = ner_tags[tag_idx]
                
                if tag == "O":
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                else:
                    tag_type = tag.split("-")[1]
                    if tag.startswith("B-"):
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {"text": word, "label": tag_type, "tokens": [word]}
                    elif tag.startswith("I-") and current_entity and current_entity["label"] == tag_type:
                        current_entity["tokens"].append(word)
                        current_entity["text"] = " ".join(current_entity["tokens"])
            
            if current_entity:
                entities.append(current_entity)

            # Cleanup tokens from final entities dict
            for ent in entities:
                ent.pop("tokens", None)
            
            return entities

        except Exception as e:
            logger.warning("Failed NER entity inference: %s. Returning empty.", str(e))
            return []

    def recommend(self, user_id: str, user_history: List[str]) -> List[str]:
        """Provides content-based case recommendations for active session users."""
        try:
            model_instance, _ = self._get_active_model_instance("recommendation_engine")
            # Call recommendations on recommendation engine instance
            return model_instance.recommend(user_history)
        except Exception as e:
            logger.warning("Failed recommendation inference: %s. Returning empty list.", str(e))
            return []

    def calculate_risk(self, contract_text: str) -> Dict[str, Any]:
        """Analyzes a contract string and returns a risk score index between 0.0 and 1.0."""
        try:
            # 1. Feature Engineering
            feats = self.extractor.transform_to_array([contract_text])
            
            # 2. Preprocessing Scaling
            if self.scaler.is_fitted:
                feats_scaled = self.scaler.transform(feats)
            else:
                feats_scaled = feats

            # 3. Model forward prediction
            model_instance, _ = self._get_active_model_instance("legal_risk_predictor")
            score = float(model_instance.predict(feats_scaled)[0][0])

            # 4. Resolve level label
            if score < 0.35:
                level = "LOW"
            elif score < 0.70:
                level = "MEDIUM"
            else:
                level = "HIGH"

            return {
                "risk_score": round(score, 4),
                "risk_level": level,
                "engineered_features": self.extractor.extract_features_from_text(contract_text)
            }
        except Exception as e:
            logger.warning("Failed risk calculation inference: %s.", str(e))
            return {
                "risk_score": 0.5,
                "risk_level": "MEDIUM",
                "error": str(e)
            }
