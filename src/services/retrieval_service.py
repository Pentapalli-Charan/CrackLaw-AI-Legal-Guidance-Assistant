import logging
from typing import Dict, Any, Optional, List
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.search_types import SearchMode
from src.services.exceptions import ServiceError

logger = logging.getLogger("CrackLaw.Services.RetrievalService")

class RetrievalServiceWrapper:
    """Orchestrates database queries across semantic, keyword, and hybrid vector spaces."""

    def __init__(self, retrieval_service: Optional[RetrievalService] = None):
        self.retrieval_service = retrieval_service or RetrievalService()

    def _format_response(self, response) -> Dict[str, Any]:
        """Serializes SearchResultItems and metadata blocks into dict format."""
        results_list = []
        for r in response.results:
            results_list.append({
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "score": float(r.score) if hasattr(r, "score") else 0.0,
                "metadata": r.metadata,
                "act": r.act,
                "chapter": r.chapter,
                "section": r.section,
                "subsection": r.subsection,
                "source": r.source
            })
        return {
            "query": response.query,
            "results": results_list,
            "context": response.context,
            "latency_ms": response.latency_ms
        }

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",
        top_k: int = 5,
        min_similarity: float = 0.25
    ) -> Dict[str, Any]:
        """Performs queries matching the requested mode (hybrid, semantic, or keyword)."""
        try:
            logger.info("Executing retrieval search: query='%s', mode='%s'", query, mode)
            search_mode = SearchMode(mode.lower())
            response = self.retrieval_service.retrieve(
                query=query,
                filters=filters,
                mode=search_mode,
                top_k=top_k,
                min_similarity=min_similarity
            )
            return self._format_response(response)
        except Exception as e:
            logger.error("Error in retrieval service search: %s", str(e))
            raise ServiceError(f"Search retrieval failed: {e}") from e

    def search_semantic(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_similarity: float = 0.25
    ) -> Dict[str, Any]:
        """Performs semantic vector-based database lookups."""
        return self.search(query, filters, mode="semantic", top_k=top_k, min_similarity=min_similarity)

    def search_hybrid(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_similarity: float = 0.25
    ) -> Dict[str, Any]:
        """Performs hybrid (semantic + keyword lexical) database searches."""
        return self.search(query, filters, mode="hybrid", top_k=top_k, min_similarity=min_similarity)
