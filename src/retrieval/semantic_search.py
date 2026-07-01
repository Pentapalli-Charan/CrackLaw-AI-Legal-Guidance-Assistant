import time
import logging
from typing import List, Dict, Any, Optional
from src.config import Config
from src.retrieval.exceptions import SearchError
from src.retrieval.embedding_engine import EmbeddingEngine
from src.retrieval.vector_store import VectorStore
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.Retrieval.Semantic")

class SemanticSearch:
    """Manages vector similarity searching, metadata filtering, and threshold parameters."""

    def __init__(
        self,
        config: Optional[Config] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.config = config or Config()
        self.emb_engine = embedding_engine or EmbeddingEngine(self.config)
        self.vector_store = vector_store or VectorStore(self.config, dimension=self.emb_engine.dimension)
        
        ret_settings = self.config.retrieval_settings
        self.default_top_k = ret_settings.get("top_k", 5)
        self.default_threshold = ret_settings.get("similarity_threshold", 0.3)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResultItem]:
        """Calculates dense vector similarity with threshold auditing, logging search latency."""
        if not query:
            return []

        top_k = top_k or self.default_top_k
        min_similarity = min_similarity if min_similarity is not None else self.default_threshold

        try:
            # 1. Generate query embedding vector
            t_emb_start = time.time()
            query_vector = self.emb_engine.embed_query(query)
            t_emb = (time.time() - t_emb_start) * 1000

            # 2. Query nearest neighbors from vector database
            t_search_start = time.time()
            results = self.vector_store.search_vector(
                query_embedding=query_vector,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters
            )
            t_search = (time.time() - t_search_start) * 1000
            
            total_latency = t_emb + t_search
            logger.info(
                "Semantic search completed in %.2f ms [Vectorization: %.1fms, Database Query: %.1fms] (Retrieved %d candidates)",
                total_latency,
                t_emb,
                t_search,
                len(results)
            )
            return results
            
        except Exception as e:
            logger.error("Semantic search operation failed: %s", str(e), exc_info=True)
            raise SearchError(f"Semantic search failed: {e}") from e
