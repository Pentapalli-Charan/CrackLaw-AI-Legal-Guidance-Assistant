import logging
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("CrackLaw.Embeddings.Models")

class EmbeddingModel(ABC):
    """Abstract interface defining required behaviors for retrieval embedding engines."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generates a single dense vector embedding for a search query string."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates a list of dense vector embeddings for a collection of document texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the output vector dimensionality of the model."""
        pass


class SentenceTransformersEmbeddingModel(EmbeddingModel):
    """Wrapper leveraging the Hugging Face SentenceTransformers library for vectorization."""

    def __init__(self, model_name: str, use_gpu: bool = False):
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info("Initializing embedding model '%s' on device '%s'", model_name, self.device)
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_name, str(e))
            # Fallback load on CPU
            logger.info("Retrying model load on CPU device fallback...")
            self.model = SentenceTransformer(model_name, device="cpu")
            self.device = "cpu"

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return list(map(float, embedding))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        return [list(map(float, emb)) for emb in embeddings]

    @property
    def dimension(self) -> int:
        # Inspect model's internal sentence embedding dimension
        return int(self.model.get_sentence_embedding_dimension())


class EmbeddingModelFactory:
    """Instantiates embedding model drivers based on name tags or model identifiers."""

    @staticmethod
    def get_model(model_name: str, use_gpu: bool = False) -> EmbeddingModel:
        """Loads and returns an EmbeddingModel matching the configuration identifier."""
        name_lower = model_name.lower().strip()

        # 1. Map simple string constants to corresponding model paths
        if name_lower == "sentence-transformers":
            path = "sentence-transformers/all-MiniLM-L6-v2"
        elif name_lower == "bge":
            path = "BAAI/bge-small-en-v1.5"
        elif name_lower == "e5":
            path = "intfloat/e5-small-v2"
        elif name_lower == "mpnet":
            path = "sentence-transformers/all-mpnet-base-v2"
        elif name_lower == "legalbert":
            # SentenceTransformers will load this model and apply auto mean-pooling
            path = "nlpaueb/legal-bert-base-uncased"
        else:
            # Allow direct model identifier strings (e.g., custom fine-tuned weights path)
            logger.info("Direct model path supplied to factory: %s", model_name)
            path = model_name

        return SentenceTransformersEmbeddingModel(model_name=path, use_gpu=use_gpu)
