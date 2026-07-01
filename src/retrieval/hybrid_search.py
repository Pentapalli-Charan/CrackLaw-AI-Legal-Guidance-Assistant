import logging
from typing import List, Dict, Any, Optional
from src.config import Config
from src.retrieval.exceptions import SearchError
from src.retrieval.search_types import SearchMode
from src.retrieval.search_result import SearchResultItem
from src.retrieval.semantic_search import SemanticSearch
from src.retrieval.keyword_search import KeywordSearch

logger = logging.getLogger("CrackLaw.Retrieval.Hybrid")

class HybridSearch:
    """Combines semantic vector proximity search and keyword BM25 retrieval."""

    def __init__(
        self,
        config: Optional[Config] = None,
        semantic_search: Optional[SemanticSearch] = None,
        keyword_search: Optional[KeywordSearch] = None
    ):
        self.config = config or Config()
        self.semantic_search = semantic_search or SemanticSearch(self.config)
        self.keyword_search = keyword_search or KeywordSearch(self.config, self.semantic_search.vector_store)
        
        ret_settings = self.config.retrieval_settings
        self.default_mode = ret_settings.get("hybrid_mode", "hybrid")
        self.weights = ret_settings.get("hybrid_weights", {"semantic": 0.7, "keyword": 0.3})
        self.fusion_method = ret_settings.get("fusion_method", "rrf")  # rrf or linear

    def _reciprocal_rank_fusion(
        self,
        semantic_items: List[SearchResultItem],
        keyword_items: List[SearchResultItem],
        k: int = 60
    ) -> List[SearchResultItem]:
        """Calculates ranks using RRF to combine scoring vectors."""
        rrf_scores: Dict[str, float] = {}
        items_map: Dict[str, SearchResultItem] = {}

        # 1. Process semantic ranks
        for rank, item in enumerate(semantic_items):
            items_map[item.chunk_id] = item
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (k + rank + 1))

        # 2. Process keyword ranks
        for rank, item in enumerate(keyword_items):
            if item.chunk_id not in items_map:
                items_map[item.chunk_id] = item
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (k + rank + 1))

        # 3. Create merged result lists
        merged_results: List[SearchResultItem] = []
        for chunk_id, rrf_score in rrf_scores.items():
            item = items_map[chunk_id]
            
            # Find scores from individual runs
            sem_score = next((x.score for x in semantic_items if x.chunk_id == chunk_id), None)
            key_score = next((x.score for x in keyword_items if x.chunk_id == chunk_id), None)

            merged_results.append(SearchResultItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                metadata=item.metadata,
                score=rrf_score,
                semantic_score=sem_score,
                keyword_score=key_score,
                citation=item.citation
            ))

        # Sort descending by RRF score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results

    def _linear_fusion(
        self,
        semantic_items: List[SearchResultItem],
        keyword_items: List[SearchResultItem]
    ) -> List[SearchResultItem]:
        """Merges results linearly by scaling scores using weights."""
        items_map: Dict[str, SearchResultItem] = {}
        sem_scores: Dict[str, float] = {}
        key_scores: Dict[str, float] = {}

        # Collect all items
        for item in semantic_items:
            items_map[item.chunk_id] = item
            sem_scores[item.chunk_id] = item.score

        for item in keyword_items:
            items_map[item.chunk_id] = item
            key_scores[item.chunk_id] = item.score

        # Normalize semantic scores to range [0, 1] if not empty
        if sem_scores:
            sem_min = min(sem_scores.values())
            sem_max = max(sem_scores.values())
            sem_diff = sem_max - sem_min
            for cid in sem_scores:
                sem_scores[cid] = (sem_scores[cid] - sem_min) / sem_diff if sem_diff > 0 else 1.0

        # Normalize keyword scores to range [0, 1] if not empty
        if key_scores:
            key_min = min(key_scores.values())
            key_max = max(key_scores.values())
            key_diff = key_max - key_min
            for cid in key_scores:
                key_scores[cid] = (key_scores[cid] - key_min) / key_diff if key_diff > 0 else 1.0

        # Combine
        w_sem = self.weights.get("semantic", 0.7)
        w_key = self.weights.get("keyword", 0.3)
        
        merged_results: List[SearchResultItem] = []
        for cid, item in items_map.items():
            norm_sem = sem_scores.get(cid, 0.0)
            norm_key = key_scores.get(cid, 0.0)
            
            combined_score = (w_sem * norm_sem) + (w_key * norm_key)
            
            sem_raw = next((x.score for x in semantic_items if x.chunk_id == cid), None)
            key_raw = next((x.score for x in keyword_items if x.chunk_id == cid), None)

            merged_results.append(SearchResultItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                metadata=item.metadata,
                score=combined_score,
                semantic_score=sem_raw,
                keyword_score=key_raw,
                citation=item.citation
            ))

        # Sort descending by combined score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        mode: Optional[SearchMode] = None
    ) -> List[SearchResultItem]:
        """Runs search routines in target mode, returning unified result lists."""
        mode = mode or SearchMode(self.default_mode)
        logger.info("Executing hybrid search in mode '%s' for query: '%s'", mode.value, query)

        # 1. Semantic Only Mode
        if mode == SearchMode.SEMANTIC:
            return self.semantic_search.search(
                query=query,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters
            )

        # 2. Keyword Only Mode
        if mode == SearchMode.KEYWORD:
            return self.keyword_search.search(
                query=query,
                top_k=top_k,
                filters=filters
            )

        # 3. Hybrid Mode
        try:
            # Retrieve semantic matches (requesting up to 2x top_k for fusion depth)
            semantic_candidates = self.semantic_search.search(
                query=query,
                top_k=top_k * 2,
                min_similarity=min_similarity,
                filters=filters
            )
            
            # Retrieve keyword matches
            keyword_candidates = self.keyword_search.search(
                query=query,
                top_k=top_k * 2,
                filters=filters
            )
            
            # Fuse results
            if self.fusion_method == "rrf":
                logger.info("Fusing results using Reciprocal Rank Fusion (RRF).")
                fused = self._reciprocal_rank_fusion(semantic_candidates, keyword_candidates)
            else:
                logger.info("Fusing results using Linear Weighted combination.")
                fused = self._linear_fusion(semantic_candidates, keyword_candidates)

            # Cap at top_k
            return fused[:top_k]

        except Exception as e:
            logger.error("Hybrid search operation failed: %s", str(e), exc_info=True)
            raise SearchError(f"Hybrid search failed: {e}") from e
