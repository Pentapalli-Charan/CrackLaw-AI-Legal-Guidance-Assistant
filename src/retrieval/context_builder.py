import logging
from typing import List, Optional
from src.config import Config
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.Retrieval.Context")

class ContextBuilder:
    """Assembles disjoint search results into a unified, formatted prompt context for LLMs."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def estimate_tokens(self, text: str) -> int:
        """Estimates token counts using standard 4-characters per token average."""
        return max(1, len(text) // 4)

    def build_context(
        self,
        items: List[SearchResultItem],
        token_budget: int = 2048,
        include_citations: bool = True
    ) -> str:
        """Combines chunks sequentially into a unified string within the target token budget."""
        if not items:
            return ""

        context_blocks = []
        accumulated_tokens = 0
        seen_chunks = set()

        for idx, item in enumerate(items):
            # 1. Prevent duplicate chunks
            if item.chunk_id in seen_chunks:
                continue
            seen_chunks.add(item.chunk_id)

            # 2. Format the chunk text with citations
            block = ""
            if include_citations:
                cite_label = item.citation if item.citation else f"Document ID: {item.document_id}"
                block = f"[{idx + 1}] Source: {cite_label}\nContent: {item.text.strip()}\n\n"
            else:
                block = f"{item.text.strip()}\n\n"

            # 3. Budget checks
            block_tokens = self.estimate_tokens(block)
            if accumulated_tokens + block_tokens > token_budget:
                logger.info(
                    "Context assembly truncated: filled %d/%d tokens with %d chunks.",
                    accumulated_tokens,
                    token_budget,
                    len(context_blocks)
                )
                break

            context_blocks.append(block)
            accumulated_tokens += block_tokens

        assembled = "".join(context_blocks).strip()
        logger.info("Assembled retrieval context containing %d chunks.", len(context_blocks))
        return assembled
