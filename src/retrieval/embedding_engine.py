import time
import logging
from typing import List, Optional
from src.config import Config
from src.retrieval.exceptions import EmbeddingGenerationError
from src.retrieval.embedding_models import EmbeddingModelFactory, EmbeddingModel
from src.retrieval.cache import EmbeddingCache

logger = logging.getLogger("CrackLaw.Retrieval.Embeddings")

class EmbeddingEngine:
    """Manages legal text embedding generation, caching, batching, and performance metrics."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Load configurations
        emb_settings = self.config.embeddings_settings
        ret_settings = self.config.retrieval_settings
        
        # Use retrieval-specific model name override if available, else fallback to embeddings
        self.model_name = ret_settings.get("embedding_model") or emb_settings.get("model_name", "bge")
        self.batch_size = ret_settings.get("batch_size", 32)
        self.use_gpu = ret_settings.get("use_gpu", False)
        
        cache_settings = ret_settings.get("cache", {})
        cache_size = cache_settings.get("embedding_cache_size", 5000)
        
        logger.info("Initializing EmbeddingEngine with model: %s", self.model_name)
        
        # Initialize dependencies
        try:
            self.model: EmbeddingModel = EmbeddingModelFactory.get_model(
                model_name=self.model_name,
                use_gpu=self.use_gpu
            )
        except Exception as e:
            logger.error("Failed to initialize embedding model: %s", str(e))
            raise EmbeddingGenerationError(f"Embedding model initialization failed: {e}") from e
            
        self.cache = EmbeddingCache(max_size=cache_size)

    def embed_query(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a query string, utilizing caching if available."""
        if not text:
            raise EmbeddingGenerationError("Cannot generate embedding for empty query text.")

        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            logger.debug("Embedding cache hit for query: '%s...'", text[:30])
            return cached

        # Generate embedding
        try:
            t0 = time.time()
            vector = self.model.embed_query(text)
            latency = (time.time() - t0) * 1000
            logger.info("Query embedding generated in %.2f ms (dim=%d)", latency, len(vector))
            
            # Cache the result
            self.cache.set(text, vector)
            return vector
        except Exception as e:
            logger.error("Query embedding generation failed: %s", str(e), exc_info=True)
            raise EmbeddingGenerationError(f"Failed to generate query embedding: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of document strings, utilizing caching and batching."""
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # 1. Inspect cache
        for idx, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        # 2. Compute for uncached texts in batches
        if uncached_texts:
            logger.info("Batch embedding generation started for %d uncached items", len(uncached_texts))
            try:
                t0 = time.time()
                generated_embeddings = []
                
                # Split into batches
                for i in range(0, len(uncached_texts), self.batch_size):
                    batch = uncached_texts[i : i + self.batch_size]
                    batch_embs = self.model.embed_documents(batch)
                    generated_embeddings.extend(batch_embs)
                
                latency = (time.time() - t0) * 1000
                logger.info("Document batch embeddings generated in %.2f ms", latency)

                # Store back to results and update cache
                for text_idx, emb in zip(uncached_indices, generated_embeddings):
                    results[text_idx] = emb
                    self.cache.set(texts[text_idx], emb)

            except Exception as e:
                logger.error("Batch document embedding generation failed: %s", str(e), exc_info=True)
                raise EmbeddingGenerationError(f"Failed to generate document batch embeddings: {e}") from e

        # Ensure all elements have been populated
        final_results: List[List[float]] = []
        for r in results:
            if r is None:
                raise EmbeddingGenerationError("Encountered unpopulated embedding reference in batch result.")
            final_results.append(r)
            
        return final_results

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of vectors produced by the selected model."""
        return self.model.dimension
