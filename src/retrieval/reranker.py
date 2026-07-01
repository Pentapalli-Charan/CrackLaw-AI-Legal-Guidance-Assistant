import time
import logging
import torch
from typing import List, Optional
from sentence_transformers import CrossEncoder
from src.config import Config
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.Retrieval.Reranker")

class Reranker:
    """Reranks retrieved candidate chunks using deep Cross-Encoder relevance classifiers."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        ret_settings = self.config.retrieval_settings
        rerank_settings = ret_settings.get("reranking", {})
        
        self.enabled = rerank_settings.get("enabled", False)
        self.model_name = rerank_settings.get("model_name", "BAAI/bge-reranker-base")
        self.rerank_depth = rerank_settings.get("rerank_depth", 10)
        self.use_gpu = ret_settings.get("use_gpu", False)
        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        
        self.model: Optional[CrossEncoder] = None
        
        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Helper to initialize the CrossEncoder network."""
        logger.info("Initializing CrossEncoder reranker model: '%s' on %s", self.model_name, self.device)
        try:
            # sentence-transformers CrossEncoder class handles prediction of sentence pairs
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            logger.warning(
                "Failed to load CrossEncoder model '%s': %s. Reranking will be bypassed.",
                self.model_name,
                str(e)
            )
            self.enabled = False

    def rerank(self, query: str, items: List[SearchResultItem]) -> List[SearchResultItem]:
        """Re-scores the retrieved items using query-document cross-attention pools."""
        if not self.enabled or not self.model or not items:
            return items

        # Rerank only up to rerank_depth
        candidates_to_rerank = items[:self.rerank_depth]
        remaining_candidates = items[self.rerank_depth:]

        logger.info("Reranking top %d candidate chunks for query: '%s'", len(candidates_to_rerank), query)
        
        try:
            t0 = time.time()
            # 1. Prepare query-document text pairs
            pairs = [(query, item.text) for item in candidates_to_rerank]
            
            # 2. Run model predictions (produces relevance logits/scores)
            scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
            
            # 3. Update scores and re-sort
            for idx, score in enumerate(scores):
                candidates_to_rerank[idx].score = float(score)

            # Sort reranked items descending
            candidates_to_rerank.sort(key=lambda x: x.score, reverse=True)
            
            latency = (time.time() - t0) * 1000
            logger.info("Cross-Encoder reranking completed in %.2f ms", latency)
            
            # 4. Concatenate reranked set with untouched overflow candidates
            return candidates_to_rerank + remaining_candidates

        except Exception as e:
            logger.error("Reranking prediction process failed: %s. Returning raw query rankings.", str(e))
            return items
class RerankerFactory:
    """Factory supporting loading of future reranking models."""
    @staticmethod
    def get_reranker(config: Optional[Config] = None) -> Reranker:
        return Reranker(config)
