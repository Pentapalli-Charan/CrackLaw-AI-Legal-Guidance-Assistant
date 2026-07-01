import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from src.config import Config
from src.retrieval.exceptions import RetrievalError
from src.retrieval.search_types import SearchMode
from src.retrieval.search_result import RetrievalResponse, SearchResultItem
from src.retrieval.embedding_engine import EmbeddingEngine
from src.retrieval.vector_store import VectorStore
from src.retrieval.metadata_filter import MetadataFilter
from src.retrieval.semantic_search import SemanticSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.cache import QueryCache

logger = logging.getLogger("CrackLaw.Retrieval.Service")

class RetrievalService:
    """Unified access interface orchestrating indexing, caching, hybrid search, and context formatting."""

    def __init__(
        self,
        config: Optional[Config] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[VectorStore] = None,
        semantic_search: Optional[SemanticSearch] = None,
        keyword_search: Optional[KeywordSearch] = None,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[Reranker] = None,
        context_builder: Optional[ContextBuilder] = None
    ):
        self.config = config or Config()
        self.emb_engine = embedding_engine or EmbeddingEngine(self.config)
        self.vector_store = vector_store or VectorStore(self.config, dimension=self.emb_engine.dimension)
        
        # Instantiate retrieval models
        self.semantic_search = semantic_search or SemanticSearch(
            self.config, self.emb_engine, self.vector_store
        )
        self.keyword_search = keyword_search or KeywordSearch(
            self.config, self.vector_store
        )
        self.hybrid_search = hybrid_search or HybridSearch(
            self.config, self.semantic_search, self.keyword_search
        )
        self.reranker = reranker or Reranker(self.config)
        self.context_builder = context_builder or ContextBuilder(self.config)
        
        # Initialize Query Cache
        ret_settings = self.config.retrieval_settings
        cache_settings = ret_settings.get("cache", {})
        query_cache_size = cache_settings.get("query_cache_size", 1000)
        self.query_cache = QueryCache(max_size=query_cache_size)

    def index_document(self, document_id: str) -> int:
        """Loads chunk JSON segments for a document, generates embeddings, and indexes them in the VectorStore."""
        logger.info("Indexing requested for document: '%s'", document_id)
        
        # 1. Locate matching chunk files in datasets/chunks/
        # Filename is usually constructed from document hash or document_id
        chunk_file = os.path.join(self.config.chunks_dir, f"{document_id}_chunks.json")
        if not os.path.exists(chunk_file):
            # Try searching the directory for files containing document_id
            found = False
            for f in os.listdir(self.config.chunks_dir):
                if document_id in f and f.endswith(".json"):
                    chunk_file = os.path.join(self.config.chunks_dir, f)
                    found = True
                    break
            if not found:
                raise RetrievalError(f"No chunk JSON files found for document ID: {document_id}")

        # 2. Read chunks
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
        except Exception as e:
            raise RetrievalError(f"Failed to read chunk file {chunk_file}: {e}") from e

        # Standard ingestion output is a list of chunk dicts
        chunks_list = chunk_data.get("chunks", []) if isinstance(chunk_data, dict) else chunk_data
        if not chunks_list:
            logger.warning("Chunk JSON file is empty or does not contain a list of chunks: %s", chunk_file)
            return 0

        # Extract text elements to generate embeddings in batch
        texts_to_embed = [chunk["text"] for chunk in chunks_list]
        logger.info("Generating embeddings in batch for %d text chunks...", len(texts_to_embed))
        
        embeddings = self.emb_engine.embed_documents(texts_to_embed)

        # 3. Construct schemas and upsert
        chunks_to_insert = []
        for idx, chunk in enumerate(chunks_list):
            meta = chunk.get("metadata", {})
            
            # Map structural components
            chunks_to_insert.append({
                "chunk_id": chunk.get("chunk_id", f"{document_id}_{idx}"),
                "document_id": document_id,
                "text": chunk["text"],
                "embedding": embeddings[idx],
                "metadata": meta,
                "act": meta.get("act") or chunk.get("act"),
                "chapter": meta.get("chapter") or chunk.get("chapter"),
                "section": meta.get("section") or chunk.get("section"),
                "subsection": meta.get("subsection") or chunk.get("subsection"),
                "language": meta.get("language", "en"),
                "source": meta.get("source"),
                "version": meta.get("version"),
                "checksum": chunk.get("checksum")
            })

        self.vector_store.insert_chunks_batch(chunks_to_insert)
        logger.info("Successfully indexed %d chunks for document '%s'.", len(chunks_to_insert), document_id)
        return len(chunks_to_insert)

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        mode: Optional[SearchMode] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        token_budget: int = 2048
    ) -> RetrievalResponse:
        """Executes full search and context building, caching results."""
        if not query:
            return RetrievalResponse(query=query)

        t_start = time.time()
        
        # 1. Clean filters
        cleaned_filters = MetadataFilter.clean_filters(filters)
        
        # Load configs
        ret_settings = self.config.retrieval_settings
        top_k = top_k or ret_settings.get("top_k", 5)
        min_similarity = min_similarity if min_similarity is not None else ret_settings.get("similarity_threshold", 0.3)
        search_mode = mode or SearchMode(ret_settings.get("hybrid_mode", "hybrid"))

        # 2. Check query cache
        cached_response = self.query_cache.get(query, cleaned_filters, search_mode.value, top_k)
        if cached_response is not None:
            logger.info("Retrieval query cache hit for: '%s'", query)
            # Re-calculate overall latency for accuracy
            cached_response.latency_ms = (time.time() - t_start) * 1000
            return cached_response

        try:
            # 3. Execute hybrid search
            results = self.hybrid_search.search(
                query=query,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=cleaned_filters,
                mode=search_mode
            )

            # 4. Rerank matches
            results = self.reranker.rerank(query, results)

            # 5. Format prompt context
            context = self.context_builder.build_context(results, token_budget=token_budget)

            latency_ms = (time.time() - t_start) * 1000
            logger.info("Retrieval completed in %.2f ms (Mode: %s)", latency_ms, search_mode.value)

            response = RetrievalResponse(
                query=query,
                results=results,
                context=context,
                latency_ms=latency_ms
            )

            # 6. Save in cache
            self.query_cache.set(query, cleaned_filters, search_mode.value, top_k, response)
            return response

        except Exception as e:
            logger.error("Unified retrieval service run failed: %s", str(e), exc_info=True)
            raise RetrievalError(f"Retrieval failed: {e}") from e

    def clear_caches(self) -> None:
        """Evicts cache indexes."""
        self.emb_engine.cache.clear()
        self.query_cache.clear()
        logger.info("Cleared embedding and query caches.")
