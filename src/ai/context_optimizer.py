import logging
from typing import List, Optional
from src.retrieval.search_result import SearchResultItem
from src.ai.token_manager import TokenManager
from src.ai.context_injector import ContextInjector

logger = logging.getLogger("CrackLaw.AI.ContextOptimizer")

class ContextOptimizer:
    """Optimizes and filters retrieved search results to fit within a specified token budget."""

    def __init__(self, token_manager: Optional[TokenManager] = None, context_injector: Optional[ContextInjector] = None):
        self.token_manager = token_manager or TokenManager()
        self.context_injector = context_injector or ContextInjector()

    def optimize_context(
        self,
        results: List[SearchResultItem],
        token_budget: int,
        min_similarity: float = 0.0
    ) -> List[SearchResultItem]:
        """Filters, deduplicates, and trims search results to fit the token budget."""
        if not results:
            return []

        # 1. Deduplicate by chunk_id
        seen_ids = set()
        deduplicated = []
        for item in results:
            if item.chunk_id not in seen_ids:
                seen_ids.add(item.chunk_id)
                deduplicated.append(item)

        # 2. Filter by similarity score (if score exists)
        filtered = [
            item for item in deduplicated
            if item.score >= min_similarity
        ]

        # 3. Sort by score descending (highest priority first)
        filtered.sort(key=lambda x: x.score, reverse=True)

        # 4. Token-aware selection
        optimized_results = []
        for item in filtered:
            # Create a test candidate list
            test_list = optimized_results + [item]
            formatted_context = self.context_injector.inject_context(test_list)
            tokens = self.token_manager.count_tokens(formatted_context)

            if tokens <= token_budget:
                optimized_results.append(item)
            else:
                # If adding it exceeds, see if we can do partial truncation of text for the last chunk
                # to maximize budget usage.
                remaining_tokens = token_budget - self.token_manager.count_tokens(
                    self.context_injector.inject_context(optimized_results)
                )
                if remaining_tokens > 50:  # Only truncate if we have a reasonable budget left
                    # Copy the item to avoid mutating original
                    from copy import deepcopy
                    item_copy = deepcopy(item)
                    
                    # Truncate text using token manager
                    item_copy.text = self.token_manager.truncate_text(item_copy.text, remaining_tokens)
                    if len(item_copy.text.strip()) > 20:
                        optimized_results.append(item_copy)
                break

        logger.info(
            "Context optimized: reduced from %d to %d chunks, fitting within %d tokens.",
            len(results), len(optimized_results), token_budget
        )
        return optimized_results
