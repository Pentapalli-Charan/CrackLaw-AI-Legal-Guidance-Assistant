import logging
from typing import List
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.AI.ContextInjector")

class ContextInjector:
    """Formats retrieved chunks from SearchResultItems into structural blocks for prompt injection."""

    def __init__(self):
        pass

    def inject_context(self, results: List[SearchResultItem]) -> str:
        """Formats search items into XML/Markdown sections with metadata preserved."""
        if not results:
            return ""

        context_blocks = []
        for idx, item in enumerate(results, start=1):
            meta = item.metadata or {}
            
            # Resolve fields from metadata or direct properties
            act = meta.get("act") or getattr(item, "act", None) or "N/A"
            section = meta.get("section") or getattr(item, "section", None) or "N/A"
            chapter = meta.get("chapter") or getattr(item, "chapter", None) or "N/A"
            source = meta.get("source") or getattr(item, "source", None) or item.document_id
            citation = item.citation or f"{act}, Section {section}" if act != "N/A" and section != "N/A" else "N/A"

            block = (
                f"<document index=\"{idx}\" id=\"{item.chunk_id}\">\n"
                f"Source Document: {source}\n"
                f"Act: {act}\n"
                f"Chapter: {chapter}\n"
                f"Section: {section}\n"
                f"Citation Reference: {citation}\n"
                f"Content:\n{item.text.strip()}\n"
                f"</document>"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        logger.debug("Injected context with %d segments.", len(results))
        return formatted_context
